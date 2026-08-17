#!/usr/bin/env python3
"""Summarize MiniMax M3 benchmark JSONL records with robust statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "variance": None,
            "stddev": None,
            "cv_percent": None,
            "min": None,
            "max": None,
        }
    mean = statistics.fmean(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    stddev = variance**0.5
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "variance": variance,
        "stddev": stddev,
        "cv_percent": 100.0 * stddev / mean if mean else None,
        "min": min(values),
        "max": max(values),
    }


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source"] = f"{path}:{line_number}"
                records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["label"], record["workload_kind"])].append(record)

    groups = []
    metric_paths = {
        "prefill_duration_s": ("metrics", "prefill_duration_s"),
        "prefill_tokens_per_s": ("metrics", "prefill_tokens_per_s"),
        "ttft_s": ("metrics", "ttft_s"),
        "end_to_end_s": ("metrics", "end_to_end_s"),
        "decode_tokens_per_s": ("metrics", "decode_tokens_per_s"),
        "peak_mlx_bytes": ("memory", "peak_mlx_bytes"),
        "process_phys_after_prefill_bytes": (
            "runtime_probe",
            "prefill_memory_after",
            "process_phys_footprint_bytes",
        ),
    }
    for (label, workload), items in sorted(grouped.items()):
        metrics = {}
        for name, path in metric_paths.items():
            values: list[float] = []
            for item in items:
                value: Any = item
                for part in path:
                    value = value.get(part) if isinstance(value, dict) else None
                if value is not None:
                    values.append(float(value))
            metrics[name] = _stats(values)

        groups.append(
            {
                "label": label,
                "workload_kind": workload,
                "runs": len(items),
                "quality_passes": sum(
                    bool(item["output"]["quality"]["passed"]) for item in items
                ),
                "cached_tokens": sorted(
                    {int(item["metrics"]["cached_tokens"]) for item in items}
                ),
                "cache_layer_counts": sorted(
                    {
                        int(item["runtime_probe"]["cache"]["layer_cache_count"])
                        for item in items
                    }
                ),
                "chunk_counts": sorted(
                    {len(item["runtime_probe"]["chunks"]) for item in items}
                ),
                "sources": [item["_source"] for item in items],
                "metrics": metrics,
            }
        )
    return {"schema_version": 1, "record_count": len(records), "groups": groups}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(_load(args.inputs))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
