import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_jsonl(relative_path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / relative_path).read_text().splitlines()
        if line
    ]


def test_fixed_workload_hashes_and_token_ranges_verify():
    builder = _load_module(
        "benchmarks/minimax_m3_prefill/build_workload.py", "minimax_workload"
    )

    result = builder.verify()

    assert result["ok"] is True
    assert 16_000 <= result["manifest"]["prompt_tokens"] <= 17_000
    assert 2_000 <= result["manifest"]["short_control"]["prompt_tokens"] <= 2_048


def test_q3_evidence_has_accuracy_parity_and_repeated_performance_pairs():
    evidence_root = "benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3"
    matrices = {
        "adapter_long": _load_jsonl(
            f"{evidence_root}/q3-loaded-ab-current-worktree.jsonl"
        ),
        "adapter_short": _load_jsonl(
            f"{evidence_root}/q3-loaded-ab-short-current-worktree.jsonl"
        ),
        "k1_long": _load_jsonl(
            f"{evidence_root}/q3-candidate-k1-loaded-ab-long.jsonl"
        ),
        "k1_short": _load_jsonl(
            f"{evidence_root}/q3-candidate-k1-loaded-ab-short.jsonl"
        ),
    }

    for rows in matrices.values():
        assert len(rows) == 6
        assert all(row["metrics"]["cached_tokens"] == 0 for row in rows)
        assert all(
            row["runtime_probe"]["cache"]["layer_cache_count"] == 60
            for row in rows
        )
        assert all(row["settings"]["temperature"] == 1.0 for row in rows)
        assert all(row["settings"]["top_p"] == 0.95 for row in rows)
        assert all(row["settings"]["prefix_cache"] is False for row in rows)
        assert all(
            row["settings"]["speculative_decoding"] is False for row in rows
        )

    expected_q3_long_checks = {
        "attention_executor": False,
        "json_object": True,
        "num_experts_per_tok": True,
        "sentinel": True,
        "sparse_block_size": True,
        "sparse_layer_count": True,
        "sparse_topk_blocks": True,
        "topk_builder": True,
    }
    assert all(
        row["output"]["quality"]["checks"] == expected_q3_long_checks
        for key in ("adapter_long", "k1_long")
        for row in matrices[key]
    )
    assert len({row["output"]["sha256"] for row in matrices["k1_long"]}) == 1

    for key in ("adapter_short", "k1_short"):
        assert all(row["output"]["quality"]["passed"] for row in matrices[key])
        assert len({row["output"]["sha256"] for row in matrices[key]}) == 1

    paired_medians = {}
    for key, rows in matrices.items():
        paired_deltas = [
            100
            * (
                rows[index + 1]["metrics"]["prefill_tokens_per_s"]
                / rows[index]["metrics"]["prefill_tokens_per_s"]
                - 1
            )
            for index in range(0, len(rows), 2)
        ]
        paired_medians[key] = statistics.median(paired_deltas)

    assert paired_medians["adapter_long"] > 5.0
    assert paired_medians["k1_long"] > 0.5
    assert abs(paired_medians["adapter_short"]) < 0.1
    assert abs(paired_medians["k1_short"]) < 0.1


def test_quality_gate_requires_all_frozen_facts():
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_prefill_bench"
    )
    manifest = json.loads(
        (
            ROOT
            / "benchmarks/evidence/minimax_m3_prefill_m3_ultra/workload/manifest.json"
        ).read_text()
    )
    expected = manifest["quality_answer"]
    answer = {
        "sentinel": manifest["sentinel"],
        **expected,
        "topk_builder": expected["topk_builder"][0],
        "bottleneck_candidate": "Sparse top-k dispatch may dominate.",
        "correctness_risk": "Block selection must remain causal.",
    }

    passed = bench._quality_result(json.dumps(answer), manifest)
    answer["sparse_block_size"] = 64
    failed = bench._quality_result(json.dumps(answer), manifest)

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["sparse_block_size"] is False


def test_cache_summary_counts_layer_types_and_bytes():
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_prefill_cache"
    )

    class CacheA:
        def nbytes(self):
            return 10

    class CacheB:
        nbytes = 7

    summary = bench._cache_summary([CacheA(), CacheA(), CacheB()])

    assert summary == {
        "layer_cache_count": 3,
        "layer_cache_types": {"CacheA": 2, "CacheB": 1},
        "known_cache_bytes": 27,
        "known_cache_byte_layers": 3,
    }


def test_component_timing_snapshot_reports_counts_and_totals():
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_component_timing"
    )

    probe = bench.RuntimeProbe(
        scheduler=object(), profile=False, component_profile="layer"
    )
    probe._record_component("attention", 0.25)
    probe._record_component("attention", 0.75)
    probe._record_component("moe", 1.5)

    assert probe.snapshot()["component_profile"] == {
        "mode": "layer",
        "external_prefill_timings": None,
        "request_timings": {
            "attention": {"calls": 2, "total_s": 1.0},
            "moe": {"calls": 1, "total_s": 1.5},
        },
    }


def test_msa_k1_ab_label_selects_only_the_requested_kernel():
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_msa_k1_ab"
    )

    assert bench._msa_k1_impl_for_label("candidate-k1-reference") == "steel_mma"
    assert bench._msa_k1_impl_for_label("upstream-k1-reference") == "steel_mma"
    assert bench._msa_k1_impl_for_label("candidate-k1-auto") == "steel_mma"
    assert (
        bench._msa_k1_impl_for_label("candidate-k1-steel-mma-bq64")
        == "steel_mma_bq64"
    )
    assert (
        bench._msa_k1_impl_for_label("upstream-k1-steel-mma-bq64")
        == "steel_mma_bq64"
    )
    assert bench._msa_k1_impl_for_label("candidate-k1-bq64") == "steel_mma_bq64"

    try:
        bench._msa_k1_impl_for_label("candidate-unknown")
    except ValueError as exc:
        assert "reference" in str(exc)
    else:
        raise AssertionError("unknown K1 label must fail closed")


def test_moe_sort_ab_label_fails_closed():
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_moe_sort_ab"
    )

    assert bench._moe_sort_for_label("candidate-moe-sorted") is True
    assert bench._moe_sort_for_label("candidate-moe-unsorted") is False

    try:
        bench._moe_sort_for_label("candidate-unknown")
    except ValueError as exc:
        assert "candidate-moe-sorted" in str(exc)
    else:
        raise AssertionError("unknown MoE-sort label must fail closed")


def test_adaptive_ab_variant_restores_upstream_and_enables_candidate(tmp_path):
    bench = _load_module(
        "benchmarks/minimax_m3_prefill/run_benchmark.py", "minimax_prefill_ab"
    )
    (tmp_path / "config.json").write_text('{"model_type":"minimax_m3_vl"}')

    class Config:
        prefill_step_size = 2048

    class Scheduler:
        model = object()
        config = Config()
        _minimax_m3_adaptive_prefill = None

    scheduler = Scheduler()
    committed_candidate = object()
    bench._set_adaptive_ab_variant(
        scheduler, "candidate-1", committed_candidate, tmp_path
    )
    assert scheduler._minimax_m3_adaptive_prefill.step_size == 4096
    bench._set_adaptive_ab_variant(
        scheduler, "upstream-1", committed_candidate, tmp_path
    )
    assert scheduler._minimax_m3_adaptive_prefill is None


def test_summary_reports_sample_variance_and_quality_counts():
    summary_module = _load_module(
        "benchmarks/minimax_m3_prefill/summarize_results.py",
        "minimax_m3_prefill_summary",
    )
    records = []
    for duration, quality in ((2.0, True), (4.0, False)):
        records.append(
            {
                "_source": "sample.jsonl:1",
                "label": "candidate",
                "workload_kind": "long",
                "metrics": {
                    "prefill_duration_s": duration,
                    "prefill_tokens_per_s": 100.0 / duration,
                    "ttft_s": duration + 1,
                    "end_to_end_s": duration + 2,
                    "decode_tokens_per_s": 10.0,
                    "cached_tokens": 0,
                },
                "memory": {"peak_mlx_bytes": 1000},
                "runtime_probe": {
                    "prefill_memory_after": {
                        "process_phys_footprint_bytes": 900
                    },
                    "cache": {"layer_cache_count": 60},
                    "chunks": [{}, {}],
                },
                "output": {"quality": {"passed": quality}},
            }
        )

    result = summary_module.summarize(records)
    group = result["groups"][0]
    assert group["quality_passes"] == 1
    assert group["cache_layer_counts"] == [60]
    assert group["chunk_counts"] == [2]
    assert group["metrics"]["prefill_duration_s"]["median"] == 3.0
    assert group["metrics"]["prefill_duration_s"]["variance"] == 2.0
