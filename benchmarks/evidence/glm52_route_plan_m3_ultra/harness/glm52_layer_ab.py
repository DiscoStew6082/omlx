#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark stock and block-plan paths on one real GLM-5.2 MoE layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.activations import swiglu

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.patches.glm_moe_dsa.switch_layers import (
    _GLM_AFFINE_BLOCK_BM,
    _GLM_AFFINE_BLOCK_VARIANT,
    _gather_counting_sort,
    _gather_sort,
)

NUM_EXPERTS = 256
ROUTES_PER_TOKEN = 8
HIDDEN_SIZE = 6144
MOE_INTERMEDIATE_SIZE = 2048
GROUP_SIZE = 64
BITS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.layer < 0 or args.warmup < 0:
        parser.error("layer and warmup must not be negative")
    for name in ("tokens", "iterations", "samples"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not (args.model_dir / "model.safetensors.index.json").is_file():
        parser.error("model_dir does not contain model.safetensors.index.json")
    if not (args.source_root / ".git").exists():
        parser.error("source-root is not a Git worktree")
    if args.output.exists():
        parser.error("output already exists")
    return args


def _run_git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_revision(model_dir: Path) -> str | None:
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if not metadata.is_file():
        return None
    revision = metadata.read_text(encoding="utf-8").splitlines()[0]
    return revision if len(revision) == 40 else None


def _load_projection(model_dir: Path, layer: int, name: str) -> tuple:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{name}"
    keys = tuple(f"{prefix}.{suffix}" for suffix in ("weight", "scales", "biases"))
    shard_names = {index["weight_map"][key] for key in keys}
    if len(shard_names) != 1:
        raise RuntimeError(f"{prefix} spans multiple shards: {sorted(shard_names)}")
    tensors = mx.load(str(model_dir / shard_names.pop()))
    projection = tuple(tensors[key] for key in keys)
    mx.eval(*projection)
    return projection


def _make_indices(tokens: int):
    token = mx.arange(tokens, dtype=mx.int32)[:, None]
    route = mx.arange(ROUTES_PER_TOKEN, dtype=mx.int32)[None]
    return ((token * 13 + route * 29) % NUM_EXPERTS)[None]


def _stock_qmm(x, projection, indices):
    weight, scales, biases = projection
    return mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        rhs_indices=indices,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="affine",
        sorted_indices=True,
    )


def _block_qmm(x, projection, block_plan):
    weight, scales, biases = projection
    block_meta, block_count = block_plan
    return glm_fast.deepseek_affine_gather_qmm_blocks(
        x,
        weight,
        scales,
        biases,
        block_meta,
        block_count,
        GROUP_SIZE,
        BITS,
        _GLM_AFFINE_BLOCK_VARIANT,
    )


def _eval_result(result) -> None:
    mx.eval(*result)


def _time_ms(function: Callable, iterations: int) -> float:
    mx.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        _eval_result(function())
    mx.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1_000_000


def _measure(
    baseline_fn: Callable,
    candidate_fn: Callable,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, list[float]]:
    for function in (baseline_fn, candidate_fn):
        for _ in range(warmup):
            _eval_result(function())
        mx.synchronize()
    results = {"baseline": [], "candidate": []}
    functions = {"baseline": baseline_fn, "candidate": candidate_fn}
    for sample in range(samples):
        order = (
            ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        )
        for name in order:
            results[name].append(_time_ms(functions[name], iterations))
    return results


def main() -> int:
    args = parse_args()
    head = _run_git(args.source_root, "rev-parse", "HEAD")
    if head != args.candidate:
        raise RuntimeError(f"candidate mismatch: expected {args.candidate}, got {head}")
    tracked_status = _run_git(
        args.source_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError(f"tracked worktree changes present:\n{tracked_status}")
    if not mx.metal.is_available():
        raise RuntimeError("Metal is required")
    if not glm_fast.is_native_available():
        raise RuntimeError("GLM native extension is unavailable")
    if not glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks"):
        raise RuntimeError("native affine block-list kernel is unavailable")

    mx.random.seed(23)
    indices = _make_indices(args.tokens)
    x = mx.random.normal((1, args.tokens, 1, 1, HIDDEN_SIZE), dtype=mx.bfloat16)
    scores = mx.softmax(mx.random.normal(indices.shape, dtype=mx.float32), axis=-1)
    gate_up = _load_projection(args.model_dir, args.layer, "gate_up_proj")
    down = _load_projection(args.model_dir, args.layer, "down_proj")
    mx.eval(x, indices, scores)

    def baseline_fn():
        sorted_x, sorted_indices, inverse = _gather_sort(
            x, indices, inverse_scatter=True
        )
        gate_up_out = _stock_qmm(sorted_x, gate_up, sorted_indices)
        gate, up = mx.split(gate_up_out, 2, axis=-1)
        hidden = swiglu(gate, up)
        output = _stock_qmm(hidden, down, sorted_indices)
        output = glm_fast.glm_moe_weighted_sum(output, inverse, scores)
        return output, sorted_indices

    def candidate_fn():
        sorted_x, sorted_indices, inverse, block_plan = _gather_counting_sort(
            x, indices, NUM_EXPERTS, _GLM_AFFINE_BLOCK_BM
        )
        gate_up_out = _block_qmm(sorted_x, gate_up, block_plan)
        gate, up = mx.split(gate_up_out, 2, axis=-1)
        hidden = swiglu(gate, up)
        output = _block_qmm(hidden, down, block_plan)
        output = glm_fast.glm_moe_weighted_sum(output, inverse, scores)
        block_meta, block_count = block_plan
        return output, sorted_indices, block_meta, block_count

    baseline = baseline_fn()
    candidate = candidate_fn()
    _eval_result(baseline)
    _eval_result(candidate)
    baseline_output = baseline[0].astype(mx.float32)
    candidate_output = candidate[0].astype(mx.float32)
    abs_error = mx.abs(baseline_output - candidate_output)
    max_abs_error = float(mx.max(abs_error).item())
    mean_abs_error = float(mx.mean(abs_error).item())
    max_reference = float(mx.max(mx.abs(baseline_output)).item())
    relative_max_error = max_abs_error / max(max_reference, 1e-12)
    numerically_close = bool(
        mx.allclose(baseline_output, candidate_output, rtol=0.02, atol=0.05).item()
    )
    sorted_experts_match = bool(mx.array_equal(baseline[1], candidate[1]).item())
    if not numerically_close or not sorted_experts_match:
        raise RuntimeError(
            "correctness check failed: "
            f"max_abs_error={max_abs_error}, "
            f"mean_abs_error={mean_abs_error}, "
            f"relative_max_error={relative_max_error}, "
            f"sorted_experts_match={sorted_experts_match}"
        )
    del baseline, candidate

    samples = _measure(
        baseline_fn,
        candidate_fn,
        args.warmup,
        args.iterations,
        args.samples,
    )
    baseline_median = statistics.median(samples["baseline"])
    candidate_median = statistics.median(samples["candidate"])
    paired_speedups = [
        baseline_time / candidate_time
        for baseline_time, candidate_time in zip(
            samples["baseline"], samples["candidate"], strict=True
        )
    ]
    median_paired_speedup = statistics.median(paired_speedups)
    device = mx.device_info()
    model_dir = args.model_dir.resolve()
    result = {
        "benchmark": "glm52_real_moe_layer_ab",
        "command": sys.argv,
        "source": {
            "candidate": head,
            "upstream_base": args.upstream_base,
            "tracked_worktree_clean": True,
        },
        "model": {
            "revision": _model_revision(model_dir),
            "config_sha256": _sha256_file(model_dir / "config.json"),
            "weight_index_sha256": _sha256_file(
                model_dir / "model.safetensors.index.json"
            ),
            "layer": args.layer,
        },
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "nanobind": importlib.metadata.version("nanobind"),
            "device": device,
        },
        "parameters": {
            "random_seed": 23,
            "tokens": args.tokens,
            "routes": indices.size,
            "routes_per_token": ROUTES_PER_TOKEN,
            "experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "dtype": "bf16",
            "quantization": "4-bit affine, group_size=64",
            "block_bm": _GLM_AFFINE_BLOCK_BM,
            "block_variant": _GLM_AFFINE_BLOCK_VARIANT,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples": args.samples,
        },
        "correctness": {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "relative_max_error": relative_max_error,
            "allclose_rtol": 0.02,
            "allclose_atol": 0.05,
            "numerically_close": numerically_close,
            "sorted_experts_match": sorted_experts_match,
            "weighted_sum_restoration": True,
        },
        "baseline": {
            "median_ms": baseline_median,
            "samples_ms": samples["baseline"],
        },
        "candidate": {
            "median_ms": candidate_median,
            "samples_ms": samples["candidate"],
        },
        "paired_speedups": paired_speedups,
        "median_paired_speedup": median_paired_speedup,
        "time_reduction_percent": (1 - 1 / median_paired_speedup) * 100,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
