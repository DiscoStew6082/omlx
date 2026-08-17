# Reproduction commands

Run from the isolated worktree on the candidate branch. The original checkout's
virtual environment is reused read-only.

```bash
export REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
export OMLX_BENCH_VENV="${OMLX_BENCH_VENV:-${REPO_ROOT}/.venv}"
export OMLX_MINIMAX_MODEL="${OMLX_MINIMAX_MODEL:-models/MiniMax-M3-4bit}"
export OMLX_MINIMAX_OQ3_MODEL="${OMLX_MINIMAX_OQ3_MODEL:-models/MiniMax-M3-oQ3}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONPATH="$PWD"

"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/build_workload.py --verify

# Untouched-upstream process-cold baseline (run while HEAD contains no runtime change).
for run_id in 01 02 03; do
  "$OMLX_BENCH_VENV/bin/python" -B \
    benchmarks/minimax_m3_prefill/run_benchmark.py \
    --workload long --label upstream-cold --runs 1 --max-tokens 256 \
    --output "benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/upstream-cold-${run_id}.jsonl"
done

# Candidate process-cold behavior while the runtime is still untouched.
for run_id in 01 02 03; do
  "$OMLX_BENCH_VENV/bin/python" -B \
    benchmarks/minimax_m3_prefill/run_benchmark.py \
    --workload long --label candidate-cold --runs 1 --max-tokens 256 \
    --adaptive-ab \
    --output "benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/candidate-cold-${run_id}.jsonl"
done

# Alternating same-loaded-model A/B, with one full-prompt warmup per variant.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --warmup 2 --max-tokens 256 --adaptive-ab \
  --sequence upstream-loaded,candidate-loaded,upstream-loaded,candidate-loaded,upstream-loaded,candidate-loaded,upstream-loaded,candidate-loaded,upstream-loaded,candidate-loaded \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/loaded-ab-long.jsonl

# Short regression control.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload short --warmup 2 --max-tokens 256 --adaptive-ab \
  --sequence upstream-short,candidate-short,upstream-short,candidate-short,upstream-short,candidate-short,upstream-short,candidate-short,upstream-short,candidate-short \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/loaded-ab-short.jsonl

# Committed candidate's default path: no benchmark variant override.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --label candidate-default-committed --runs 1 \
  --max-tokens 256 \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/candidate-default-committed.jsonl

# Summary and verification.
"$OMLX_BENCH_VENV/bin/python" \
  benchmarks/minimax_m3_prefill/summarize_results.py \
  benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/loaded-ab-long.jsonl

"$OMLX_BENCH_VENV/bin/python" -m pytest -q \
  tests/test_scheduler_chunked_prefill.py \
  tests/test_mlx_vlm_minimax_m3_compat.py \
  tests/test_minimax_m3_prefill_benchmark.py
```

The process-cold commands append JSONL. Use new output paths for a fresh run.
`--adaptive-ab` now forces every `upstream-*` label to the untouched `None`
configuration even when the checked-out runtime contains the committed
candidate. This prevents a post-commit A/B from silently comparing the
candidate with itself.

## Jundot oQ3 controls

The Q3 tokenizer reproduces the committed prompt token IDs exactly, so the same
workload and quality gate can be used. Keep `PYTHONPATH="$PWD"` explicit so the
records exercise the current isolated worktree rather than another installed
oMLX checkout.

```bash
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --model ${OMLX_MINIMAX_OQ3_MODEL} \
  --workload long --warmup 2 --max-tokens 256 --adaptive-ab \
  --sequence upstream-q3-current,candidate-q3-current,upstream-q3-current,candidate-q3-current,upstream-q3-current,candidate-q3-current \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3/q3-loaded-ab-current-worktree.jsonl

"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --model ${OMLX_MINIMAX_OQ3_MODEL} \
  --workload short --warmup 2 --max-tokens 256 --adaptive-ab \
  --sequence upstream-q3-short,candidate-q3-short,upstream-q3-short,candidate-q3-short,upstream-q3-short,candidate-q3-short \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3/q3-loaded-ab-short-current-worktree.jsonl
```

The earlier Q3 smoke/safety files are retained as diagnostics but excluded from
the conclusion: they did not record explicit current-worktree import
provenance. The two `current-worktree` files above are the authoritative Q3
A/B records.

## Existing native MiniMax extension

The initial in-sandbox `metallib` failure was a sandbox restriction. The
existing upstream extension was built outside the command sandbox with:

```bash
export OMLX_NATIVE_BUILD=${TMPDIR}/omlx-minimax-native-rgH8EY

/opt/homebrew/bin/cmake \
  -S omlx/custom_kernels/minimax_m3/csrc \
  -B "$OMLX_NATIVE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OMLX_NATIVE_BUILD/out" \
  -DPython_EXECUTABLE="$OMLX_BENCH_VENV/bin/python"
/opt/homebrew/bin/cmake --build "$OMLX_NATIVE_BUILD" --parallel

cp "$OMLX_NATIVE_BUILD/out/_ext.cpython-311-darwin.so" \
  "$OMLX_NATIVE_BUILD/out/libomlx_minimax_m3_kernel_ops.dylib" \
  "$OMLX_NATIVE_BUILD/out/omlx_minimax_m3_kernels.metallib" \
  omlx/custom_kernels/minimax_m3/

"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --warmup 2 --max-tokens 256 --adaptive-ab \
  --sequence upstream-native,candidate-native,upstream-native,candidate-native,upstream-native,candidate-native \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/loaded-ab-long-native.jsonl
```

The built files are deliberately untracked and removed from the source package
after the run. `native_topk_build.json` records tool paths, binary hashes, ABI
status, and the exact native-versus-fallback numerical checks. The numerical
probe source used for this run remains at
`${TMPDIR}/verify_minimax_native_topk.py`.

The separately authorized Jundot download was verified with the already
installed Hugging Face CLI 1.10.2 (no global CLI update was needed):

```bash
/opt/homebrew/bin/hf cache verify Jundot/MiniMax-M3-oQ3 \
  --local-dir ${OMLX_MINIMAX_OQ3_MODEL} \
  --fail-on-missing-files
```

The stricter `--fail-on-extra-files` form is not appropriate for a direct local
directory because HF CLI counts its own `.cache/huggingface` metadata as local
extras. The command above verified all remote files and checksums.

## Barrier-isolated component profiles

These runs are diagnostic. Each component boundary forces inputs and evaluates
outputs, so do not compare their total duration with the official A/B gate.
`--max-tokens 2` intentionally avoids spending decode time on a timing-only
profile; quality evidence comes from the full 256-token runs above.

```bash
for mode in layer moe projections sparse sync; do
  "$OMLX_BENCH_VENV/bin/python" -B \
    benchmarks/minimax_m3_prefill/run_benchmark.py \
    --workload long --label "candidate-component-$mode" --runs 2 \
    --max-tokens 2 --component-profile "$mode" \
    --output "benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/profile/component-recheck-$mode.jsonl"
done
```

The recorded layer result uses
`raw/profile/candidate-component-layer-prefill.jsonl`; the earlier
`candidate-component-layer.jsonl` was produced before the profiler separated
external-prefill-only timing and is excluded from attribution.

The fail-closed benchmark-only MoE sort diagnostic is:

```bash
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --warmup 2 --max-tokens 2 --moe-sort-ab \
  --sequence candidate-moe-sorted,candidate-moe-unsorted \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/profile/candidate-moe-sort-loaded-ab-long.jsonl
```

Only one complete pair was run because the unsorted side was already 65.8%
slower in throughput. That stop is intentional, not missing evidence.

## Sparse K1 tile A/B

The harness uses explicit `steel_mma` and `steel_mma_bq64` selectors, so the
reference remains reproducible after `auto` changes to the 64-thread tile.

```bash
# Candidate adaptive-prefill path, Q4 long.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --warmup 2 --max-tokens 256 --msa-k1-ab \
  --sequence candidate-k1-reference,candidate-k1-bq64,candidate-k1-reference,candidate-k1-bq64,candidate-k1-reference,candidate-k1-bq64 \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/k1-recheck-candidate-long.jsonl

# Untouched upstream 2,048-token behavior in the same loaded model.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload long --warmup 2 --max-tokens 256 --adaptive-ab --msa-k1-ab \
  --sequence upstream-k1-reference,upstream-k1-bq64,upstream-k1-reference,upstream-k1-bq64,upstream-k1-reference,upstream-k1-bq64 \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/k1-recheck-upstream-long.jsonl

# Q4 short-prompt regression control.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --workload short --warmup 2 --max-tokens 256 --msa-k1-ab \
  --sequence candidate-k1-reference-short,candidate-k1-bq64-short,candidate-k1-reference-short,candidate-k1-bq64-short,candidate-k1-reference-short,candidate-k1-bq64-short \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/official/k1-recheck-candidate-short.jsonl

# Q3 long accuracy/performance control.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --model ${OMLX_MINIMAX_OQ3_MODEL} \
  --workload long --warmup 2 --max-tokens 256 --msa-k1-ab \
  --sequence candidate-k1-reference-q3,candidate-k1-bq64-q3,candidate-k1-reference-q3,candidate-k1-bq64-q3,candidate-k1-reference-q3,candidate-k1-bq64-q3 \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3/k1-recheck-q3-long.jsonl

# Q3 short-prompt accuracy/regression control.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/run_benchmark.py \
  --model ${OMLX_MINIMAX_OQ3_MODEL} \
  --workload short --warmup 2 --max-tokens 256 --msa-k1-ab \
  --sequence candidate-k1-reference-q3-short,candidate-k1-bq64-q3-short,candidate-k1-reference-q3-short,candidate-k1-bq64-q3-short,candidate-k1-reference-q3-short,candidate-k1-bq64-q3-short \
  --output benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3/k1-recheck-q3-short.jsonl

# Direct native numerical comparison at every production chunk shape.
"$OMLX_BENCH_VENV/bin/python" -B \
  benchmarks/minimax_m3_prefill/verify_k1_bq64.py
```

The historical raw A/B labels use `*-k1-auto` for the reference because they
were captured before the source default changed; at that commit, `auto` was
the explicit 128-thread `steel_mma` behavior.
