#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run an alternating fresh-process GLM-5.2 engine A/B bracket."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

SEQUENCE = ("off", "on", "on", "off", "off", "on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--engine-harness", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--gen", type=int, default=1)
    args = parser.parse_args()
    if args.length <= 0 or args.gen <= 0:
        parser.error("length and gen must be positive")
    if not args.python.is_file():
        parser.error("python executable does not exist")
    if not args.engine_harness.is_file():
        parser.error("engine harness does not exist")
    if args.output_dir.exists():
        parser.error("output-dir already exists")
    return args


def _sample(result: dict, length: int, mode: str) -> dict:
    return result["results"]["single"][str(length)][mode]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True)
    records = []
    for index, mode in enumerate(SEQUENCE, start=1):
        result_path = args.output_dir / f"run-{index:02d}-{mode}.json"
        command = [
            str(args.python),
            str(args.engine_harness),
            "single",
            str(args.model_dir),
            "--source-root",
            str(args.source_root),
            "--candidate",
            args.candidate,
            "--upstream-base",
            args.upstream_base,
            "--output",
            str(result_path),
            "--length",
            str(args.length),
            "--gen",
            str(args.gen),
            "--mode",
            mode,
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=args.source_root,
            text=True,
            capture_output=True,
            timeout=900,
        )
        wall_seconds = time.perf_counter() - started
        process_record = {
            "index": index,
            "mode": mode,
            "command": command,
            "returncode": completed.returncode,
            "wall_seconds": wall_seconds,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (args.output_dir / f"process-{index:02d}-{mode}.json").write_text(
            json.dumps(process_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"fresh-process run {index} ({mode}) failed")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sample = _sample(result, args.length, mode)
        records.append(
            {
                "index": index,
                "mode": mode,
                "wall_seconds": wall_seconds,
                "source": result["source"],
                "model": result["model"],
                "environment": result["environment"],
                "runtime": result["runtime"],
                "parameters": result["parameters"],
                "prompt_sha256": result["prompt_sha256"][str(args.length)],
                "sample": sample,
            }
        )
        print(
            f"run={index:02d} mode={mode} "
            f"tps={sample['processing_tps']:.3f} "
            f"ttft_ms={sample['ttft_ms']:.3f} "
            f"dispatch={sample['dispatch_calls']} "
            f"wall_s={wall_seconds:.1f}",
            flush=True,
        )

    grouped = {
        mode: [record for record in records if record["mode"] == mode]
        for mode in ("off", "on")
    }

    def values(mode: str, name: str) -> list[float]:
        return [record["sample"][name] for record in grouped[mode]]

    baseline_tps = values("off", "processing_tps")
    candidate_tps = values("on", "processing_tps")
    baseline_ttft = values("off", "ttft_ms")
    candidate_ttft = values("on", "ttft_ms")
    baseline_tps_median = statistics.median(baseline_tps)
    candidate_tps_median = statistics.median(candidate_tps)
    baseline_ttft_median = statistics.median(baseline_ttft)
    candidate_ttft_median = statistics.median(candidate_ttft)
    output_hashes = [
        output_hash
        for record in records
        for output_hash in record["sample"]["output_sha256"]
    ]
    prompt_hashes = {record["prompt_sha256"] for record in records}
    revisions = {record["model"]["revision"] for record in records}
    source_pairs = {
        (
            record["source"]["candidate"],
            record["source"]["upstream_base"],
        )
        for record in records
    }
    runtimes = [record["runtime"] for record in records]
    baseline_dispatch = [
        record["sample"]["dispatch_calls"] for record in grouped["off"]
    ]
    candidate_dispatch = [
        record["sample"]["dispatch_calls"] for record in grouped["on"]
    ]
    invariants = {
        "outputs_match": len(set(output_hashes)) == 1,
        "prompt_hashes_match": len(prompt_hashes) == 1,
        "model_revision_stable": len(revisions) == 1,
        "source_identity_stable": len(source_pairs) == 1,
        "all_cached_tokens_zero": all(
            cached == 0
            for record in records
            for cached in record["sample"]["cached_tokens"]
        ),
        "all_mtp_disabled": all(not runtime["mtp_enabled"] for runtime in runtimes),
        "all_prefix_cache_disabled": all(
            not runtime["prefix_cache_enabled"] for runtime in runtimes
        ),
        "all_paged_ssd_cache_disabled": all(
            runtime["paged_ssd_cache_dir"] is None for runtime in runtimes
        ),
        "baseline_never_dispatched": not any(baseline_dispatch),
        "candidate_always_dispatched": all(call > 0 for call in candidate_dispatch),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"cold bracket invariant failures: {failed}")

    summary = {
        "benchmark": "glm52_fresh_process_bracket",
        "orchestrator_command": sys.argv,
        "sequence": list(SEQUENCE),
        "parameters": {
            "prompt_tokens": args.length,
            "generation_tokens": args.gen,
            "fresh_process_per_run": True,
            "os_file_cache_flushed": False,
        },
        "source": {
            "candidate": args.candidate,
            "upstream_base": args.upstream_base,
        },
        "model_revision": next(iter(revisions)),
        "baseline": {
            "processing_tps": baseline_tps,
            "median_processing_tps": baseline_tps_median,
            "ttft_ms": baseline_ttft,
            "median_ttft_ms": baseline_ttft_median,
            "dispatch_calls": baseline_dispatch,
        },
        "candidate": {
            "processing_tps": candidate_tps,
            "median_processing_tps": candidate_tps_median,
            "ttft_ms": candidate_ttft,
            "median_ttft_ms": candidate_ttft_median,
            "dispatch_calls": candidate_dispatch,
        },
        "median_processing_speedup": candidate_tps_median / baseline_tps_median,
        "median_ttft_reduction": 1 - candidate_ttft_median / baseline_ttft_median,
        "output_sha256": sorted(set(output_hashes)),
        "prompt_sha256": sorted(prompt_hashes),
        "invariants": invariants,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
