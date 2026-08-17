#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Commit-pinned GLM-5.2 engine A/B benchmark with dispatch/output evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _int_list(value: str) -> list[int]:
    values = [_positive_int(item.strip()) for item in value.split(",")]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gen", type=_positive_int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    loaded = subparsers.add_parser("loaded")
    _add_common(loaded)
    loaded.add_argument("--lengths", type=_int_list, required=True)
    loaded.add_argument("--batch-length", type=_positive_int, required=True)
    loaded.add_argument("--batch-size", type=_positive_int, default=2)
    loaded.add_argument("--warmup", type=int, default=1)
    loaded.add_argument("--samples", type=_positive_int, default=3)

    single = subparsers.add_parser("single")
    _add_common(single)
    single.add_argument("--length", type=_positive_int, required=True)
    single.add_argument("--mode", choices=("off", "on"), required=True)

    args = parser.parse_args()
    if getattr(args, "warmup", 0) < 0:
        parser.error("warmup must be non-negative")
    if not (args.model_dir / "config.json").is_file():
        parser.error("model_dir does not contain config.json")
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


def _prompt_hash(prompt: list[int]) -> str:
    encoded = b"".join(
        int(token).to_bytes(4, "little", signed=False) for token in prompt
    )
    return hashlib.sha256(encoded).hexdigest()


def _output_payload(output: Any) -> dict:
    text = getattr(output, "text", None)
    if text is None:
        text = getattr(output, "output_text", "")
    tokens = getattr(output, "tokens", None)
    if tokens is None:
        tokens = getattr(output, "output_token_ids", [])
    return {
        "finish_reason": getattr(output, "finish_reason", None),
        "text": text or "",
        "token_ids": [int(token) for token in tokens],
    }


def _output_record(output: Any) -> dict:
    payload = _output_payload(output)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "finish_reason": payload["finish_reason"],
        "token_ids": payload["token_ids"],
        "text_sha256": hashlib.sha256(payload["text"].encode()).hexdigest(),
        "cached_tokens": int(getattr(output, "cached_tokens", 0)),
        "prompt_tokens": int(getattr(output, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(output, "completion_tokens", 0)),
    }


def _native_identity(source_root: Path) -> dict:
    package = source_root / "omlx/custom_kernels/glm_moe_dsa"
    artifacts = []
    for pattern in ("_ext*.so", "*.dylib", "*.metallib"):
        for path in sorted(package.glob(pattern)):
            artifacts.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {"artifacts": artifacts}


def _source_identity(args: argparse.Namespace) -> dict:
    head = _run_git(args.source_root, "rev-parse", "HEAD")
    if head != args.candidate:
        raise RuntimeError(f"candidate mismatch: expected {args.candidate}, got {head}")
    tracked_status = _run_git(
        args.source_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(f"tracked worktree changes present:\n{tracked_status}")
    return {
        "candidate": head,
        "upstream_base": args.upstream_base,
        "tracked_worktree_clean": True,
    }


def _summary(results: dict[str, list[dict]], metric: str, latency: str) -> dict:
    baseline = [sample[metric] for sample in results["off"]]
    candidate = [sample[metric] for sample in results["on"]]
    paired = [on / off for off, on in zip(baseline, candidate, strict=True)]
    hashes = [
        output_hash
        for samples in results.values()
        for sample in samples
        for output_hash in sample["output_sha256"]
    ]
    cached = [
        cached_tokens
        for samples in results.values()
        for sample in samples
        for cached_tokens in sample["cached_tokens"]
    ]
    return {
        "baseline": {
            metric: baseline,
            f"median_{metric}": statistics.median(baseline),
            latency: [sample[latency] for sample in results["off"]],
            f"median_{latency}": statistics.median(
                sample[latency] for sample in results["off"]
            ),
            "dispatch_calls": [sample["dispatch_calls"] for sample in results["off"]],
        },
        "candidate": {
            metric: candidate,
            f"median_{metric}": statistics.median(candidate),
            latency: [sample[latency] for sample in results["on"]],
            f"median_{latency}": statistics.median(
                sample[latency] for sample in results["on"]
            ),
            "dispatch_calls": [sample["dispatch_calls"] for sample in results["on"]],
        },
        "paired_speedups": paired,
        "median_paired_speedup": statistics.median(paired),
        "outputs_match": len(set(hashes)) == 1,
        "all_cached_tokens_zero": all(value == 0 for value in cached),
        "samples": results,
    }


async def _run(args: argparse.Namespace) -> dict:
    import mlx.core as mx

    import omlx.admin.benchmark as benchmark
    from omlx.admin.benchmark import _run_batch_test, _run_single_test
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
    from omlx.engine.batched import BatchedEngine
    from omlx.patches.glm_moe_dsa import switch_layers
    from omlx.patches.mlx_lm_mtp import is_mtp_active

    source = _source_identity(args)
    if not glm_fast.is_native_available():
        raise RuntimeError("GLM native extension is unavailable")
    if not glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks"):
        raise RuntimeError("required affine block symbol is unavailable")

    model_dir = args.model_dir.resolve()
    engine = BatchedEngine(str(model_dir))
    started = time.perf_counter()
    await engine.start()
    load_seconds = time.perf_counter() - started
    if engine.model_type != "glm_moe_dsa":
        await engine.stop()
        raise RuntimeError(f"expected glm_moe_dsa, got {engine.model_type}")

    lengths = (
        list(dict.fromkeys([*args.lengths, args.batch_length]))
        if args.command == "loaded"
        else [args.length]
    )
    original_uuid4 = benchmark.uuid.uuid4
    benchmark.uuid.uuid4 = lambda: uuid.UUID(int=0)
    try:
        prompts = {
            length: benchmark._generate_prompt(engine.tokenizer, length)
            for length in lengths
        }
    finally:
        benchmark.uuid.uuid4 = original_uuid4

    original_route_plan = switch_layers._gather_counting_sort
    original_mode = switch_layers._GLM_AFFINE_BLOCK_MODE
    original_stream_generate = engine.stream_generate
    active_mode = {"value": ""}
    dispatch_totals = {"off": 0, "on": 0}

    def route_plan_spy(*call_args, **call_kwargs):
        mode = active_mode["value"]
        if mode in dispatch_totals:
            dispatch_totals[mode] += 1
        return original_route_plan(*call_args, **call_kwargs)

    switch_layers._gather_counting_sort = route_plan_spy

    async def single(mode: str, length: int) -> dict:
        active_mode["value"] = mode
        switch_layers._GLM_AFFINE_BLOCK_MODE = mode
        captured = {}

        async def capture(**kwargs):
            async for output in original_stream_generate(**kwargs):
                captured["last"] = output
                yield output

        engine.stream_generate = capture
        before = dispatch_totals[mode]
        try:
            metrics = await _run_single_test(
                engine,
                prompts[length],
                args.gen,
                length,
            )
        finally:
            engine.stream_generate = original_stream_generate
        output = captured.get("last")
        if output is None:
            raise RuntimeError("single request produced no output")
        output_record = _output_record(output)
        result = {
            "processing_tps": metrics["processing_tps"],
            "ttft_ms": metrics["ttft_ms"],
            "e2e_latency_s": metrics["e2e_latency_s"],
            "dispatch_calls": dispatch_totals[mode] - before,
            "output_sha256": [output_record["sha256"]],
            "cached_tokens": [output_record["cached_tokens"]],
            "outputs": [output_record],
        }
        print(
            json.dumps(
                {"kind": "single", "length": length, "mode": mode, **result},
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    async def batch(mode: str) -> dict:
        active_mode["value"] = mode
        switch_layers._GLM_AFFINE_BLOCK_MODE = mode
        engine_core = benchmark._get_batch_benchmark_core(engine)
        if engine_core is None:
            raise RuntimeError("batch engine core is unavailable")
        original_stream_outputs = engine_core.stream_outputs
        captured = {}

        async def capture_stream_outputs(request_id):
            async for output in original_stream_outputs(request_id):
                if output.finished:
                    captured[request_id] = output
                yield output

        engine_core.stream_outputs = capture_stream_outputs
        before = dispatch_totals[mode]
        try:
            metrics = await _run_batch_test(
                engine,
                [prompts[args.batch_length]] * args.batch_size,
                args.batch_length,
                args.gen,
                args.batch_size,
            )
        finally:
            engine_core.stream_outputs = original_stream_outputs
        if len(captured) != args.batch_size:
            raise RuntimeError(
                f"captured {len(captured)} batch outputs, expected {args.batch_size}"
            )
        outputs = [_output_record(captured[key]) for key in sorted(captured)]
        result = {
            "pp_tps": metrics["pp_tps"],
            "avg_ttft_ms": metrics["avg_ttft_ms"],
            "e2e_latency_s": metrics["e2e_latency_s"],
            "total_gen_tokens": metrics["total_gen_tokens"],
            "dispatch_calls": dispatch_totals[mode] - before,
            "output_sha256": [output["sha256"] for output in outputs],
            "cached_tokens": [output["cached_tokens"] for output in outputs],
            "outputs": outputs,
        }
        print(
            json.dumps({"kind": "batch", "mode": mode, **result}, sort_keys=True),
            flush=True,
        )
        return result

    try:
        if args.command == "single":
            result = await single(args.mode, args.length)
            benchmark_results = {
                "single": {str(args.length): {args.mode: result}},
                "batch": {},
            }
        else:
            single_results = {}
            for length in args.lengths:
                for _ in range(args.warmup):
                    await single("off", length)
                    await single("on", length)
                dispatch_totals = {"off": 0, "on": 0}
                results = {"off": [], "on": []}
                for sample in range(args.samples):
                    order = ("off", "on") if sample % 2 == 0 else ("on", "off")
                    for mode in order:
                        results[mode].append(await single(mode, length))
                summary = _summary(results, "processing_tps", "ttft_ms")
                expected_dispatch = (length - 1) * 8 >= 8192
                baseline_calls = summary["baseline"]["dispatch_calls"]
                candidate_calls = summary["candidate"]["dispatch_calls"]
                if any(baseline_calls):
                    raise RuntimeError(f"baseline dispatched at pp{length}")
                if expected_dispatch != all(call > 0 for call in candidate_calls):
                    raise RuntimeError(f"unexpected candidate dispatch at pp{length}")
                if not summary["outputs_match"]:
                    raise RuntimeError(f"single outputs differ at pp{length}")
                if not summary["all_cached_tokens_zero"]:
                    raise RuntimeError(f"single cache hit at pp{length}")
                single_results[str(length)] = summary

            for _ in range(args.warmup):
                await batch("off")
                await batch("on")
            dispatch_totals = {"off": 0, "on": 0}
            results = {"off": [], "on": []}
            for sample in range(args.samples):
                order = ("off", "on") if sample % 2 == 0 else ("on", "off")
                for mode in order:
                    results[mode].append(await batch(mode))
            batch_summary = _summary(results, "pp_tps", "avg_ttft_ms")
            if any(batch_summary["baseline"]["dispatch_calls"]):
                raise RuntimeError("batch baseline dispatched")
            if not all(
                call > 0 for call in batch_summary["candidate"]["dispatch_calls"]
            ):
                raise RuntimeError("batch candidate did not dispatch")
            if not batch_summary["outputs_match"]:
                raise RuntimeError("batch outputs differ")
            if not batch_summary["all_cached_tokens_zero"]:
                raise RuntimeError("batch cache hit")
            benchmark_results = {
                "single": single_results,
                "batch": {str(args.batch_size): batch_summary},
            }

        scheduler_config = engine._engine.engine.scheduler.config
        runtime = {
            "mtp_enabled": is_mtp_active(),
            "prefix_cache_enabled": engine.prefix_cache_enabled,
            "paged_ssd_cache_dir": (
                str(scheduler_config.paged_ssd_cache_dir)
                if scheduler_config.paged_ssd_cache_dir is not None
                else None
            ),
        }
        if runtime != {
            "mtp_enabled": False,
            "prefix_cache_enabled": False,
            "paged_ssd_cache_dir": None,
        }:
            raise RuntimeError(f"unexpected runtime features: {runtime}")
        result = {
            "benchmark": f"glm52_engine_{args.command}",
            "source": source,
            "model": {
                "revision": _model_revision(model_dir),
                "config_sha256": _sha256_file(model_dir / "config.json"),
                "weight_index_sha256": _sha256_file(
                    model_dir / "model.safetensors.index.json"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "mlx": importlib.metadata.version("mlx"),
                "nanobind": importlib.metadata.version("nanobind"),
                "device": mx.device_info(),
                "native": _native_identity(args.source_root),
            },
            "command": sys.argv,
            "load_seconds": load_seconds,
            "parameters": {
                "generation_tokens": args.gen,
                "temperature": 0.0,
                "top_p": 1.0,
                "skip_cache_store": True,
                "fixed_prompt_uuid": "00000000-0000-0000-0000-000000000000",
                "context_profile": "code_python",
                "lengths": lengths,
                "batch_size": getattr(args, "batch_size", None),
                "warmup_per_mode": getattr(args, "warmup", None),
                "samples_per_mode": getattr(args, "samples", None),
                "mode": getattr(args, "mode", None),
            },
            "runtime": runtime,
            "prompt_sha256": {
                str(length): _prompt_hash(prompt) for length, prompt in prompts.items()
            },
            "results": benchmark_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"complete": str(args.output)}, sort_keys=True), flush=True)
        return result
    finally:
        engine.stream_generate = original_stream_generate
        switch_layers._gather_counting_sort = original_route_plan
        switch_layers._GLM_AFFINE_BLOCK_MODE = original_mode
        await engine.stop()


def main() -> int:
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
