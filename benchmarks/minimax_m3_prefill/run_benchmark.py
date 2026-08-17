#!/usr/bin/env python3
"""Run fixed MiniMax M3 prefill measurements through the real oMLX VLM path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

from omlx.engine.vlm import VLMBatchedEngine
from omlx.scheduler import SchedulerConfig
from omlx.utils.proc_memory import get_phys_footprint

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(
    os.environ.get("OMLX_MINIMAX_MODEL", "models/MiniMax-M3-4bit")
)
DEFAULT_WORKLOAD_ROOT = (
    REPO_ROOT / "benchmarks/evidence/minimax_m3_prefill_m3_ultra/workload"
)
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0
SEED = 1729


def _set_adaptive_ab_variant(
    scheduler: Any,
    label: str,
    _original: Any,
    model_path: Path,
) -> None:
    """Toggle only the existing MiniMax adaptive-prefill config for loaded A/B."""
    if label.startswith("upstream"):
        scheduler._minimax_m3_adaptive_prefill = None
        return
    if not label.startswith("candidate"):
        raise ValueError(
            "--adaptive-ab labels must start with 'upstream' or 'candidate'"
        )
    from omlx.patches.minimax_m3.generate_patch import (
        _minimax_m3_adaptive_prefill_config,
    )

    candidate = _minimax_m3_adaptive_prefill_config(
        scheduler.model,
        scheduler.config.prefill_step_size,
        model_path,
    )
    if candidate is None:
        raise RuntimeError("Explicit-path MiniMax adaptive-prefill detection failed")
    scheduler._minimax_m3_adaptive_prefill = candidate


def _msa_k1_impl_for_label(label: str) -> str:
    if label.startswith(
        (
            "candidate-k1-reference",
            "upstream-k1-reference",
            "candidate-k1-auto",
            "upstream-k1-auto",
        )
    ):
        return "steel_mma"
    if label.startswith(
        (
            "candidate-k1-bq64",
            "upstream-k1-bq64",
            "candidate-k1-steel-mma-bq64",
            "upstream-k1-steel-mma-bq64",
        )
    ):
        return "steel_mma_bq64"
    raise ValueError(
        "--msa-k1-ab labels must select a candidate/upstream K1 "
        "'reference' or 'bq64' variant"
    )


def _install_msa_k1_ab() -> Callable[[], None]:
    from mlx_vlm.models.minimax_m3_vl import language

    original = language.msa_sparse_attention_b1

    def wrapped(*args, **kwargs):
        label = os.environ.get("OMLX_MINIMAX_M3_PREFILL_BENCH_VARIANT", "")
        kwargs["k1_impl"] = _msa_k1_impl_for_label(label)
        return original(*args, **kwargs)

    language.msa_sparse_attention_b1 = wrapped
    return lambda: setattr(language, "msa_sparse_attention_b1", original)


def _moe_sort_for_label(label: str) -> bool:
    if label.startswith("candidate-moe-sorted"):
        return True
    if label.startswith("candidate-moe-unsorted"):
        return False
    raise ValueError(
        "--moe-sort-ab labels must start with 'candidate-moe-sorted' or "
        "'candidate-moe-unsorted'"
    )


def _install_moe_sort_ab() -> Callable[[], None]:
    from mlx_vlm.models.minimax_m3_vl import language

    cls = language.MiniMaxPackedSwitchGLU
    original = cls.__call__

    def wrapped(instance, x, indices):
        label = os.environ.get("OMLX_MINIMAX_M3_PREFILL_BENCH_VARIANT", "")
        if _moe_sort_for_label(label):
            return original(instance, x, indices)

        x = mx.expand_dims(x, (-2, -3))
        idx = mx.stop_gradient(indices) if instance.training else indices
        gate_up = instance.gate_up_proj(x, idx, sorted_indices=False)
        gate, up = mx.split(gate_up, 2, axis=-1)
        x = instance.down_proj(
            instance.activation(up, gate),
            idx,
            sorted_indices=False,
        )
        return x.squeeze(-2)

    cls.__call__ = wrapped
    return lambda: setattr(cls, "__call__", original)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _memory_snapshot() -> dict[str, int]:
    return {
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "process_phys_footprint_bytes": int(get_phys_footprint()),
    }


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _cache_summary(cache: Any) -> dict[str, Any]:
    if not isinstance(cache, (list, tuple)):
        return {"layer_cache_count": 0, "layer_cache_types": {}}
    types_count = Counter(type(item).__name__ for item in cache)
    known_bytes = 0
    known_byte_layers = 0
    for item in cache:
        try:
            value = (
                item.nbytes()
                if callable(getattr(item, "nbytes", None))
                else item.nbytes
            )
            known_bytes += int(value)
            known_byte_layers += 1
        except (AttributeError, TypeError, ValueError):
            continue
    return {
        "layer_cache_count": len(cache),
        "layer_cache_types": dict(sorted(types_count.items())),
        "known_cache_bytes": known_bytes,
        "known_cache_byte_layers": known_byte_layers,
    }


def _quality_result(text: str, manifest: dict[str, Any]) -> dict[str, Any]:
    stripped = text.strip()
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as nested:
                parse_error = str(nested)
        else:
            parse_error = str(exc)

    expected = manifest["quality_answer"]
    checks = {
        "json_object": isinstance(parsed, dict),
        "sentinel": isinstance(parsed, dict)
        and parsed.get("sentinel") == manifest["sentinel"],
    }
    for key, value in expected.items():
        actual = parsed.get(key) if isinstance(parsed, dict) else None
        checks[key] = actual in value if isinstance(value, list) else actual == value

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parse_error": parse_error,
        "parsed": parsed,
    }


class RuntimeProbe:
    """Low-overhead prefill timing plus optional Python dispatch counters."""

    def __init__(
        self,
        scheduler: Any,
        profile: bool,
        component_profile: str | None = None,
    ):
        self.scheduler = scheduler
        self.profile = profile
        self.component_profile = component_profile
        self._restores: list[Callable[[], None]] = []
        self.reset()

    def reset(self) -> None:
        self.prefill_started_at: float | None = None
        self.prefill_ended_at: float | None = None
        self.prefill_memory_before: dict[str, int] | None = None
        self.prefill_memory_after: dict[str, int] | None = None
        self.cache_summary: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.dispatch_counts: Counter[str] = Counter()
        self.dispatch_shapes: dict[str, Counter[str]] = {}
        self.component_calls: Counter[str] = Counter()
        self.component_seconds: Counter[str] = Counter()
        self.component_prefill_calls: Counter[str] | None = None
        self.component_prefill_seconds: Counter[str] | None = None

    def _count(self, name: str, values: tuple[Any, ...]) -> None:
        self.dispatch_counts[name] += 1
        shapes = [_shape(value) for value in values]
        shapes = [shape for shape in shapes if shape is not None]
        if shapes:
            counter = self.dispatch_shapes.setdefault(name, Counter())
            counter[json.dumps(shapes, separators=(",", ":"))] += 1

    def _record_component(self, name: str, duration_s: float) -> None:
        self.component_calls[name] += 1
        self.component_seconds[name] += duration_s

    @staticmethod
    def _eval_direct_arrays(*values: Any) -> None:
        arrays: list[Any] = []

        def visit(value: Any) -> None:
            if type(value).__module__.startswith("mlx.") and hasattr(value, "shape"):
                arrays.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        for value in values:
            visit(value)
        if arrays:
            mx.eval(*arrays)

    def install(self) -> None:
        original_prefill = self.scheduler._do_external_prefill

        def timed_prefill(request, tokens, existing_cache, vlm_embeds=None):
            self.prefill_started_at = time.perf_counter()
            self.prefill_memory_before = _memory_snapshot()
            try:
                result = original_prefill(
                    request, tokens, existing_cache, vlm_embeds=vlm_embeds
                )
                self.cache_summary = _cache_summary(result[0])
                return result
            finally:
                if self.component_profile:
                    self.component_prefill_calls = self.component_calls.copy()
                    self.component_prefill_seconds = self.component_seconds.copy()
                self.prefill_ended_at = time.perf_counter()
                self.prefill_memory_after = _memory_snapshot()

        self.scheduler._do_external_prefill = timed_prefill
        self._restores.append(
            lambda: setattr(self.scheduler, "_do_external_prefill", original_prefill)
        )

        original_record = self.scheduler._record_chunk_transient

        def record_chunk(n_tokens, before, after, **kwargs):
            if self.component_profile == "sync":
                started_at = time.perf_counter()
                mx.synchronize()
                self._record_component(
                    "post_eval_idle_synchronize",
                    time.perf_counter() - started_at,
                )
            self.chunks.append(
                {
                    "tokens": int(n_tokens),
                    "before_phys_bytes": int(before),
                    "after_phys_bytes": int(after),
                    "delta_phys_bytes": int(after) - int(before),
                    "kv_len": int(kwargs.get("kv_len", 0)),
                    "requested_step": int(kwargs.get("requested_step", 0)),
                    "loop_label": str(kwargs.get("loop_label", "")),
                }
            )
            return original_record(n_tokens, before, after, **kwargs)

        self.scheduler._record_chunk_transient = record_chunk
        self._restores.append(
            lambda: setattr(self.scheduler, "_record_chunk_transient", original_record)
        )

        if self.profile:
            self._install_profile_counters()
        if self.component_profile:
            self._install_component_timers()

    def _patch_module_function(self, module: Any, name: str) -> None:
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            self._count(name, args)
            return original(*args, **kwargs)

        setattr(module, name, wrapped)
        self._restores.append(lambda: setattr(module, name, original))

    def _patch_method(self, cls: type, name: str, label: str) -> None:
        original = getattr(cls, name)

        def wrapped(instance, *args, **kwargs):
            self._count(label, args)
            return original(instance, *args, **kwargs)

        setattr(cls, name, wrapped)
        self._restores.append(lambda: setattr(cls, name, original))

    def _time_module_function(self, module: Any, name: str, label: str) -> None:
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            self._eval_direct_arrays(args, kwargs)
            started_at = time.perf_counter()
            result = original(*args, **kwargs)
            self._eval_direct_arrays(result)
            mx.synchronize()
            self._record_component(label, time.perf_counter() - started_at)
            return result

        setattr(module, name, wrapped)
        self._restores.append(lambda: setattr(module, name, original))

    def _time_method(self, cls: type, name: str, label: str) -> None:
        original = getattr(cls, name)

        def wrapped(instance, *args, **kwargs):
            self._eval_direct_arrays(args, kwargs)
            started_at = time.perf_counter()
            result = original(instance, *args, **kwargs)
            self._eval_direct_arrays(result)
            mx.synchronize()
            self._record_component(label, time.perf_counter() - started_at)
            return result

        setattr(cls, name, wrapped)
        self._restores.append(lambda: setattr(cls, name, original))

    def _install_profile_counters(self) -> None:
        from mlx_vlm.models.minimax_m3_vl import language

        for name in (
            "build_grouped_msa_topk",
            "build_grouped_msa_topk_blockwise",
            "msa_sparse_attention_b1",
        ):
            self._patch_module_function(language, name)

        for cls, label in (
            (language.MiniMaxAttention, "MiniMaxAttention.__call__"),
            (language.MiniMaxAttention, "MiniMaxAttention._msa_prefill_attention"),
            (language.MiniMaxSparseMoeBlock, "MiniMaxSparseMoeBlock.__call__"),
            (language.MiniMaxPackedSwitchGLU, "MiniMaxPackedSwitchGLU.__call__"),
        ):
            method = label.rsplit(".", 1)[-1]
            self._patch_method(cls, method, label)

        import mlx.nn as nn

        quantized_linear = getattr(nn, "QuantizedLinear", None)
        if quantized_linear is not None:
            self._patch_method(
                quantized_linear,
                "__call__",
                "QuantizedLinear.__call__",
            )

        try:
            from mlx_lm.models import switch_layers
        except ImportError:
            switch_layers = None
        if switch_layers is not None:
            for cls_name in ("QuantizedSwitchLinear", "SwitchLinear"):
                cls = getattr(switch_layers, cls_name, None)
                if cls is not None:
                    self._patch_method(cls, "__call__", f"{cls_name}.__call__")

    def _install_component_timers(self) -> None:
        from mlx_vlm.models.minimax_m3_vl import language

        if self.component_profile == "layer":
            self._time_method(
                language.MiniMaxAttention,
                "__call__",
                "MiniMaxAttention.__call__",
            )
            self._time_method(
                language.MiniMaxSparseMoeBlock,
                "__call__",
                "MiniMaxSparseMoeBlock.__call__",
            )
        elif self.component_profile == "sparse":
            for name in (
                "build_grouped_msa_topk",
                "build_grouped_msa_topk_blockwise",
                "msa_sparse_attention_b1",
            ):
                self._time_module_function(language, name, name)
        elif self.component_profile == "moe":
            self._time_module_function(
                language,
                "_minimax_moe_select",
                "_minimax_moe_select",
            )
            self._time_method(
                language.MiniMaxPackedSwitchGLU,
                "__call__",
                "MiniMaxPackedSwitchGLU.__call__",
            )
        elif self.component_profile == "projections":
            import mlx.nn as nn
            from mlx_lm.models import switch_layers

            self._time_method(
                nn.QuantizedLinear,
                "__call__",
                "QuantizedLinear.__call__",
            )
            self._time_method(
                switch_layers.QuantizedSwitchLinear,
                "__call__",
                "QuantizedSwitchLinear.__call__",
            )
        elif self.component_profile == "sync":
            from omlx import scheduler as scheduler_module

            original_sync_clear = scheduler_module._sync_and_clear_cache

            def timed_sync_clear(stream):
                started_at = time.perf_counter()
                result = original_sync_clear(stream)
                self._record_component(
                    "_sync_and_clear_cache", time.perf_counter() - started_at
                )
                return result

            scheduler_module._sync_and_clear_cache = timed_sync_clear
            self._restores.append(
                lambda: setattr(
                    scheduler_module,
                    "_sync_and_clear_cache",
                    original_sync_clear,
                )
            )
        else:
            raise ValueError(
                f"Unsupported component profile {self.component_profile!r}"
            )

    def snapshot(self) -> dict[str, Any]:
        duration = None
        if self.prefill_started_at is not None and self.prefill_ended_at is not None:
            duration = self.prefill_ended_at - self.prefill_started_at
        component_profile = None
        if self.component_profile:
            def timings(
                calls: Counter[str], seconds: Counter[str]
            ) -> dict[str, dict[str, float | int]]:
                return {
                    name: {
                        "calls": int(calls[name]),
                        "total_s": float(seconds[name]),
                    }
                    for name in sorted(calls)
                }

            component_profile = {
                "mode": self.component_profile,
                "external_prefill_timings": (
                    None
                    if self.component_prefill_calls is None
                    or self.component_prefill_seconds is None
                    else timings(
                        self.component_prefill_calls,
                        self.component_prefill_seconds,
                    )
                ),
                "request_timings": timings(
                    self.component_calls,
                    self.component_seconds,
                ),
            }
        return {
            "prefill_duration_s": duration,
            "prefill_memory_before": self.prefill_memory_before,
            "prefill_memory_after": self.prefill_memory_after,
            "cache": self.cache_summary,
            "chunks": list(self.chunks),
            "dispatch_counts": dict(sorted(self.dispatch_counts.items())),
            "dispatch_shapes": {
                name: dict(sorted(counter.items()))
                for name, counter in sorted(self.dispatch_shapes.items())
            },
            "component_profile": component_profile,
        }

    def restore(self) -> None:
        for restore in reversed(self._restores):
            restore()
        self._restores.clear()


def _static_model_activity(engine: VLMBatchedEngine) -> dict[str, Any]:
    language_model = getattr(engine._vlm_model, "language_model", None)
    model = getattr(language_model, "model", None)
    layers = list(getattr(model, "layers", []) or [])
    sparse_layers = 0
    moe_layers = 0
    module_types: Counter[str] = Counter()
    projection_types: Counter[str] = Counter()
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if bool(getattr(attention, "has_sparse_index", False)):
            sparse_layers += 1
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None:
            moe_layers += 1
            module_types[type(moe).__name__] += 1
            switch = getattr(moe, "switch_mlp", None)
            if switch is not None:
                module_types[type(switch).__name__] += 1
                for name in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
                    projection = getattr(switch, name, None)
                    if projection is not None:
                        projection_types[type(projection).__name__] += 1
        if attention is not None:
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "index_q_proj",
                "index_k_proj",
            ):
                projection = getattr(attention, name, None)
                if projection is not None:
                    projection_types[type(projection).__name__] += 1
    return {
        "layer_count": len(layers),
        "sparse_attention_layers": sparse_layers,
        "moe_layers": moe_layers,
        "module_types": dict(sorted(module_types.items())),
        "projection_types": dict(sorted(projection_types.items())),
    }


def _scheduler_state(engine: VLMBatchedEngine) -> dict[str, Any]:
    scheduler = engine._engine.engine.scheduler
    block_cache = getattr(scheduler, "block_aware_cache", None)
    adaptive = getattr(scheduler, "_minimax_m3_adaptive_prefill", None)
    adapter_model_type = getattr(scheduler.model, "model_type", "")
    adapter_minimax_flag = getattr(
        scheduler.model, "_uses_minimax_m3_positions", False
    )
    adaptive_recomputed: dict[str, int] | str | None
    try:
        from omlx.patches.minimax_m3.generate_patch import (
            _declares_minimax_m3,
            _minimax_m3_adaptive_prefill_config,
        )

        declares_minimax = _declares_minimax_m3(scheduler.model)
        recomputed = _minimax_m3_adaptive_prefill_config(
            scheduler.model,
            scheduler.config.prefill_step_size,
            getattr(scheduler.config, "model_name", None),
        )
        adaptive_recomputed = (
            None
            if recomputed is None
            else {
                "step_size": int(recomputed.step_size),
                "after": int(recomputed.after),
                "min_remaining": int(recomputed.min_remaining),
            }
        )
    except Exception as exc:
        declares_minimax = f"{type(exc).__name__}: {exc}"
        adaptive_recomputed = f"{type(exc).__name__}: {exc}"
    return {
        "scheduler_model_name": str(getattr(scheduler.config, "model_name", "")),
        "scheduler_prefill_step_size": int(scheduler.config.prefill_step_size),
        "adaptive_prefill_env": os.environ.get(
            "MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP"
        ),
        "minimax_adaptive_prefill": (
            None
            if adaptive is None
            else {
                "step_size": int(adaptive.step_size),
                "after": int(adaptive.after),
                "min_remaining": int(adaptive.min_remaining),
            }
        ),
        "minimax_adaptive_prefill_recomputed": adaptive_recomputed,
        "adapter_minimax_flag": bool(adapter_minimax_flag),
        "adapter_minimax_flag_type": type(adapter_minimax_flag).__name__,
        "adapter_minimax_flag_repr": repr(adapter_minimax_flag),
        "adapter_model_type": str(adapter_model_type),
        "adapter_model_type_value_type": type(adapter_model_type).__name__,
        "adapter_model_type_repr": repr(adapter_model_type),
        "adapter_python_type": (
            f"{type(scheduler.model).__module__}.{type(scheduler.model).__name__}"
        ),
        "adapter_declares_minimax": declares_minimax,
        "prefix_cache_enabled": engine.prefix_cache_enabled,
        "block_aware_cache_present": block_cache is not None,
        "draft_prefix_cache_present": getattr(scheduler, "_draft_prefix_cache", None)
        is not None,
        "specprefill_draft_model_present": getattr(
            scheduler, "_specprefill_draft_model", None
        )
        is not None,
        "vlm_mtp_drafter_present": getattr(engine, "_vlm_mtp_drafter", None)
        is not None,
        "waiting_requests": len(getattr(scheduler, "waiting", [])),
        "prefilling_requests": len(getattr(scheduler, "prefilling", [])),
        "tracked_requests": len(getattr(scheduler, "requests", {})),
    }


async def _run_request(
    engine: VLMBatchedEngine,
    probe: RuntimeProbe,
    token_ids: list[int],
    workload_kind: str,
    manifest: dict[str, Any],
    label: str,
    max_tokens: int,
    capture_path: Path | None,
) -> dict[str, Any]:
    os.environ["OMLX_MINIMAX_M3_PREFILL_BENCH_VARIANT"] = label
    probe.reset()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    memory_before = _memory_snapshot()

    capture_started = False
    if capture_path is not None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(capture_path))
        capture_started = True

    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    previous_completion = 0
    last_output = None
    try:
        async for output in engine.stream_generate(
            prompt=token_ids,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            seed=SEED,
            skip_cache_store=True,
        ):
            if output.completion_tokens > previous_completion:
                produced = output.generated_at or time.perf_counter()
                if first_token_at is None:
                    first_token_at = float(produced)
                last_token_at = float(output.generated_until or produced)
            previous_completion = output.completion_tokens
            last_output = output
    finally:
        if capture_started:
            mx.metal.stop_capture()

    finished_at = time.perf_counter()
    mx.synchronize()
    memory_after = _memory_snapshot()
    if last_output is None:
        raise RuntimeError("Generation produced no output")
    if last_output.prompt_tokens != len(token_ids):
        raise RuntimeError(
            f"Prompt token mismatch: engine={last_output.prompt_tokens}, "
            f"workload={len(token_ids)}"
        )
    if last_output.cached_tokens != 0:
        raise RuntimeError(
            f"Expected cold prefill, got {last_output.cached_tokens} cached tokens"
        )

    probe_data = probe.snapshot()
    prefill_duration = probe_data["prefill_duration_s"]
    if prefill_duration is None or prefill_duration <= 0:
        raise RuntimeError("Exact external-prefill timing was not observed")

    ttft = None if first_token_at is None else first_token_at - started_at
    decode_duration = None
    if first_token_at is not None and last_token_at is not None:
        decode_duration = max(0.0, last_token_at - first_token_at)
    e2e = finished_at - started_at
    output_text = last_output.text
    external_prefill_tokens = len(token_ids) - 1

    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "recorded_at": datetime.now(UTC).isoformat(),
        "label": label,
        "workload_kind": workload_kind,
        "git_commit": _git_commit(),
        "model": {
            "path": str(Path(engine.model_name).resolve()),
            "config_sha256": _sha256(Path(engine.model_name) / "config.json"),
            "index_sha256": _sha256(
                Path(engine.model_name) / "model.safetensors.index.json"
            ),
        },
        "workload": {
            "name": manifest["name"],
            "prompt_tokens": len(token_ids),
            "prompt_sha256": (
                manifest["prompt_sha256"]
                if workload_kind == "long"
                else manifest["short_control"]["prompt_sha256"]
            ),
            "token_ids_sha256": (
                manifest["token_ids_sha256"]
                if workload_kind == "long"
                else manifest["short_control"]["token_ids_sha256"]
            ),
        },
        "settings": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "thinking_mode": "disabled",
            "max_tokens": max_tokens,
            "prefix_cache": False,
            "speculative_decoding": False,
            "prefill_step_size": 2048,
            "chunked_prefill": False,
            "prefill_speed_priority": True,
        },
        "metrics": {
            "prefill_duration_s": prefill_duration,
            "prefill_tokens_per_s": external_prefill_tokens / prefill_duration,
            "external_prefill_tokens": external_prefill_tokens,
            "ttft_s": ttft,
            "end_to_end_s": e2e,
            "decode_duration_s": decode_duration,
            "decode_tokens_per_s": (
                None
                if not decode_duration or last_output.completion_tokens <= 1
                else (last_output.completion_tokens - 1) / decode_duration
            ),
            "prompt_tokens": last_output.prompt_tokens,
            "completion_tokens": last_output.completion_tokens,
            "cached_tokens": last_output.cached_tokens,
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "peak_mlx_bytes": int(mx.get_peak_memory()),
        },
        "runtime_probe": probe_data,
        "scheduler_before_cleanup": _scheduler_state(engine),
        "output": {
            "text": output_text,
            "sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "quality": _quality_result(output_text, manifest),
            "finish_reason": last_output.finish_reason,
        },
        "capture_path": str(capture_path) if capture_path is not None else None,
    }


async def run(args: argparse.Namespace) -> None:
    workload_root = args.workload_root.resolve()
    manifest = json.loads((workload_root / "manifest.json").read_text())
    token_file = (
        workload_root / "prompt_token_ids.json"
        if args.workload == "long"
        else workload_root / "short_prompt_token_ids.json"
    )
    token_ids = [int(token) for token in json.loads(token_file.read_text())]

    config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        prefill_step_size=2048,
        chunked_prefill=False,
        prefill_speed_priority=True,
        paged_ssd_cache_dir=None,
        hot_cache_only=False,
        hot_cache_max_size=0,
    )
    engine = VLMBatchedEngine(
        str(args.model.resolve()),
        scheduler_config=config,
        enable_thinking=False,
        model_settings=None,
    )
    load_started = time.perf_counter()
    await engine.start()
    load_duration = time.perf_counter() - load_started
    scheduler = engine._engine.engine.scheduler
    original_adaptive_prefill = scheduler._minimax_m3_adaptive_prefill
    state = _scheduler_state(engine)
    forbidden = [
        name
        for name, active in (
            ("prefix cache", state["prefix_cache_enabled"]),
            ("block-aware cache", state["block_aware_cache_present"]),
            ("draft prefix cache", state["draft_prefix_cache_present"]),
            ("SpecPrefill draft", state["specprefill_draft_model_present"]),
            ("VLM MTP drafter", state["vlm_mtp_drafter_present"]),
        )
        if active
    ]
    if forbidden:
        await engine.stop()
        raise RuntimeError(f"Benchmark acceleration was not disabled: {forbidden}")

    probe = RuntimeProbe(
        scheduler,
        profile=args.profile,
        component_profile=args.component_profile,
    )
    probe.install()
    restore_msa_k1 = _install_msa_k1_ab() if args.msa_k1_ab else lambda: None
    restore_moe_sort = (
        _install_moe_sort_ab() if args.moe_sort_ab else lambda: None
    )
    static_activity = _static_model_activity(engine)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = (
        [item.strip() for item in args.sequence.split(",") if item.strip()]
        if args.sequence
        else [args.label] * args.runs
    )

    try:
        for warmup_index in range(args.warmup):
            if args.msa_k1_ab:
                k1_prefix = (
                    "upstream"
                    if labels and labels[0].startswith("upstream")
                    else "candidate"
                )
                warmup_label = (
                    f"{k1_prefix}-k1-reference-warmup"
                    if warmup_index % 2 == 0
                    else f"{k1_prefix}-k1-bq64-warmup"
                )
            elif args.moe_sort_ab:
                warmup_label = (
                    "candidate-moe-sorted-warmup"
                    if warmup_index % 2 == 0
                    else "candidate-moe-unsorted-warmup"
                )
            else:
                warmup_label = (
                    "upstream-warmup"
                    if not args.adaptive_ab or warmup_index % 2 == 0
                    else "candidate-warmup"
                )
            if args.adaptive_ab:
                _set_adaptive_ab_variant(
                    scheduler,
                    warmup_label,
                    original_adaptive_prefill,
                    args.model.resolve(),
                )
            await _run_request(
                engine,
                probe,
                token_ids,
                args.workload,
                manifest,
                warmup_label,
                min(args.max_tokens, 16),
                None,
            )

        for index, label in enumerate(labels, start=1):
            if args.adaptive_ab:
                _set_adaptive_ab_variant(
                    scheduler,
                    label,
                    original_adaptive_prefill,
                    args.model.resolve(),
                )
            capture_path = None
            if args.metal_capture:
                stem = output_path.stem
                capture_path = (
                    output_path.parent / f"{stem}-{index:02d}-{label}.gputrace"
                )
            record = await _run_request(
                engine,
                probe,
                token_ids,
                args.workload,
                manifest,
                label,
                args.max_tokens,
                capture_path,
            )
            record["model_load_duration_s"] = load_duration
            record["static_model_activity"] = static_activity
            record["software"] = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "mlx_vlm": importlib.metadata.version("mlx-vlm"),
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            summary = {
                "run": index,
                "label": label,
                "prefill_s": round(record["metrics"]["prefill_duration_s"], 4),
                "prefill_tok_s": round(record["metrics"]["prefill_tokens_per_s"], 2),
                "ttft_s": None
                if record["metrics"]["ttft_s"] is None
                else round(record["metrics"]["ttft_s"], 4),
                "e2e_s": round(record["metrics"]["end_to_end_s"], 4),
                "peak_gib": round(record["memory"]["peak_mlx_bytes"] / 2**30, 3),
                "cached_tokens": record["metrics"]["cached_tokens"],
                "chunks": len(record["runtime_probe"]["chunks"]),
                "quality": record["output"]["quality"]["passed"],
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        restore_moe_sort()
        restore_msa_k1()
        probe.restore()
        await engine.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--workload", choices=("long", "short"), default="long")
    parser.add_argument("--label", default="upstream")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--component-profile",
        choices=("layer", "sparse", "moe", "projections", "sync"),
    )
    parser.add_argument("--adaptive-ab", action="store_true")
    parser.add_argument("--msa-k1-ab", action="store_true")
    parser.add_argument("--moe-sort-ab", action="store_true")
    parser.add_argument("--metal-capture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.max_tokens <= 1:
        parser.error("--max-tokens must be greater than 1")
    if args.profile and args.component_profile:
        parser.error("--profile and --component-profile are mutually exclusive")
    if args.msa_k1_ab and args.component_profile:
        parser.error("--msa-k1-ab and --component-profile are mutually exclusive")
    if args.moe_sort_ab and args.component_profile:
        parser.error("--moe-sort-ab and --component-profile are mutually exclusive")
    if args.msa_k1_ab and args.moe_sort_ab:
        parser.error("--msa-k1-ab and --moe-sort-ab are mutually exclusive")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
