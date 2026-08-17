#!/usr/bin/env python3
"""Compare MiniMax M3's 128- and 64-thread sparse-attention K1 tiles."""

from __future__ import annotations

import json
import time

import mlx.core as mx

from omlx.patches.mlx_vlm_minimax_m3_compat import (
    apply_mlx_vlm_minimax_m3_compat_patch,
)

apply_mlx_vlm_minimax_m3_compat_patch()

from mlx_vlm.models.minimax_m3_vl.msa import msa_sparse_attention_b1  # noqa: E402


def run_case(query_length: int, kv_length: int) -> dict[str, object]:
    mx.random.seed(1729 + query_length + kv_length)
    q = mx.random.normal((query_length, 64, 128)).astype(mx.float16)
    k = mx.random.normal((kv_length, 4, 128)).astype(mx.float16)
    v = mx.random.normal((kv_length, 4, 128)).astype(mx.float16)
    q_start = kv_length - query_length
    newest_block = mx.arange(q_start, kv_length, dtype=mx.int32) // 128
    offsets = mx.arange(16, dtype=mx.int32)
    q2k = mx.maximum(newest_block[:, None] - offsets[None, :], 0)
    q2k = mx.broadcast_to(q2k[None, ...], (4, query_length, 16))
    mx.eval(q, k, v, q2k)

    outputs = {}
    timings = {}
    for implementation in ("steel_mma", "steel_mma_bq64"):
        started_at = time.perf_counter()
        output = msa_sparse_attention_b1(
            q,
            k,
            v,
            q2k,
            q_start=q_start,
            scale=128**-0.5,
            block_size=128,
            k1_impl=implementation,
            full_splits=True,
        )
        mx.eval(output)
        mx.synchronize()
        timings[implementation] = time.perf_counter() - started_at
        outputs[implementation] = output

    difference = mx.abs(outputs["steel_mma"] - outputs["steel_mma_bq64"])
    maximum = mx.max(difference)
    mean = mx.mean(difference)
    mismatches = mx.sum(difference != 0)
    mx.eval(maximum, mean, mismatches)
    return {
        "query_length": query_length,
        "kv_length": kv_length,
        "shape": list(outputs["steel_mma"].shape),
        "steel_mma_s": timings["steel_mma"],
        "steel_mma_bq64_s": timings["steel_mma_bq64"],
        "max_abs_difference": float(maximum.item()),
        "mean_abs_difference": float(mean.item()),
        "mismatches": int(mismatches.item()),
        "exact": int(mismatches.item()) == 0,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "cases": [
                    run_case(4096, 8192),
                    run_case(4096, 12288),
                    run_case(4096, 16384),
                    run_case(124, 16508),
                ]
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
