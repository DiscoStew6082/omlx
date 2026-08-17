# GLM-5.2 long-prefill MoE route-plan evidence

This directory is a publication-safe evidence bundle for the GLM-only route-plan
optimization. It is intentionally separate from the minimal code change. The code
candidate is commit `3474c3a3b3f4d0ca5e864c07196707464013436e`, based on upstream
commit `2c10f0fb69af2a95e2c5764313f128774f1dda9f`.

The full benchmark campaign used candidate
`477da886f48d566c7012cb3e11accc489472e4b7` on base
`cfec50213da8ee46e18e03a2e27cc3e402c072f2`. The final confirmation reran the
important loaded cases and complete relevant tests after the candidate was rebased
onto the current review base. The diffs for both commit pairs are byte-identical;
their SHA-256 is `544d56d432e803f4d8e1bef73b3bf18e9735365dd6b196d80adeabc7823656eb`.

## Test identity and controls

- Hardware: Apple M3 Ultra, 512 GiB unified memory
- Runtime: Python 3.11.15, MLX 0.32.0, nanobind 2.13.0
- Model: `Jundot/GLM-5.2-oQ4e-mtp`
- Model revision: `dad7d33661ccaba82753141f13fd454c657d6f91`
- Model configuration SHA-256: `08ea1b1a7661eae041c7ecaff2976fd6ef7071c73fab331c9823d1947a9c6023`
- Weight-index SHA-256: `125b97ea2652a3feb8a96feb2754f2a69eed890ced82d78333940e9454afd256`
- Temperature 0, top-p 1, no cached prompt tokens
- MTP disabled, prefix cache disabled, paged SSD cache disabled
- The same generated prompt and model revision were used in each paired comparison
- Candidate dispatch counts were recorded; baseline dispatch counts remained zero

## Reproduced results

The final loaded confirmation on the review candidate produced:

| Case | Baseline | Candidate | Paired speedup | Baseline TTFT | Candidate TTFT | Dispatches off/on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 x 2,048-token prefill | 206.2 tok/s | 218.1 tok/s | 5.77% | 9.931 s | 9.389 s | 0 / 75 |
| 2 x 2,048-token prefill | 204.25 tok/s | 215.75 tok/s | 5.63% | 20.052 s | 18.983 s | 0 / 150 |

The larger loaded campaign produced:

| Case | Baseline | Candidate | Paired speedup | Dispatches off/on |
| --- | ---: | ---: | ---: | ---: |
| 1 x 1,024 tokens | 194.4 tok/s | 194.4 tok/s | 0.00% | 0 / 0 |
| 1 x 1,025 tokens | 194.7 tok/s | 205.4 tok/s | 5.50% | 0 / 75 |
| 1 x 2,048 tokens | 205.9 tok/s | 217.8 tok/s | 5.78% | 0 / 75 |
| 1 x 4,096 tokens | 215.7 tok/s | 227.4 tok/s | 5.42% | 0 / 75 |
| 2 x 2,048 tokens | 204.2 tok/s | 215.8 tok/s | 5.68% | 0 / 150 |

The 1,024-token case stayed on the stock path. The threshold crossing at 1,025
tokens dispatched exactly as intended.

The alternating fresh-process sequence was `off, on, on, off, off, on`. The
baseline samples were 122.0, 114.6, and 114.9 tok/s; candidate samples were 119.3,
118.4, and 114.3 tok/s. Their medians were 114.9 and 118.4 tok/s, a 3.05% increase.
Median TTFT changed from 17.818 s to 17.291 s, a 2.96% reduction. Each sample used
a fresh process, but the operating-system file cache was not flushed.

On one real GLM MoE layer, the candidate reduced measured time by 9.88% at 1,024
tokens and 10.08% at 2,048 tokens. Both comparisons reported zero absolute and
relative output error, matching sorted experts, and exact weighted-sum restoration.

All same-prompt engine outputs matched. The final native checks ran 2 tests with no
skips, failures, or errors; the final relevant GLM suite ran 56 tests with no skips,
failures, or errors. The earlier focused route-plan suite ran 5 tests with the same
result. JUnit reports are under `validation/`.

## Reproduction

Run the harnesses from a clean checkout of the code candidate, not from this
evidence-only branch. A fresh output path is required for every command.

```sh
export OMLX_SOURCE_ROOT=/path/to/omlx-candidate-worktree
export MODEL_DIR=/path/to/Jundot/GLM-5.2-oQ4e-mtp
export PYTHON=/path/to/python
export HARNESS_DIR=/path/to/this/directory/harness
export EVIDENCE_DIR=/path/to/new/output-directory
```

Final loaded confirmation:

```sh
"$PYTHON" "$HARNESS_DIR/glm52_engine_ab.py" loaded "$MODEL_DIR" \
  --source-root "$OMLX_SOURCE_ROOT" \
  --candidate 3474c3a3b3f4d0ca5e864c07196707464013436e \
  --upstream-base 2c10f0fb69af2a95e2c5764313f128774f1dda9f \
  --output "$EVIDENCE_DIR/final-loaded-confirmation.json" \
  --lengths 2048 --batch-length 2048 --batch-size 2 \
  --gen 8 --warmup 1 --samples 2
```

Loaded threshold and scale matrix:

```sh
"$PYTHON" "$HARNESS_DIR/glm52_engine_ab.py" loaded "$MODEL_DIR" \
  --source-root "$OMLX_SOURCE_ROOT" \
  --candidate 3474c3a3b3f4d0ca5e864c07196707464013436e \
  --upstream-base 2c10f0fb69af2a95e2c5764313f128774f1dda9f \
  --output "$EVIDENCE_DIR/loaded-matrix.json" \
  --lengths 1024,1025,2048,4096 --batch-length 2048 --batch-size 2 \
  --gen 8 --warmup 1 --samples 3
```

Alternating fresh-process bracket:

```sh
"$PYTHON" "$HARNESS_DIR/glm52_cold_bracket.py" "$MODEL_DIR" \
  --source-root "$OMLX_SOURCE_ROOT" \
  --engine-harness "$HARNESS_DIR/glm52_engine_ab.py" \
  --python "$PYTHON" \
  --candidate 3474c3a3b3f4d0ca5e864c07196707464013436e \
  --upstream-base 2c10f0fb69af2a95e2c5764313f128774f1dda9f \
  --output-dir "$EVIDENCE_DIR/fresh-process" --length 2048 --gen 1
```

Real-layer timing (repeat once with `--tokens 1024` and once with
`--tokens 2048`):

```sh
"$PYTHON" "$HARNESS_DIR/glm52_layer_ab.py" "$MODEL_DIR" \
  --source-root "$OMLX_SOURCE_ROOT" \
  --candidate 3474c3a3b3f4d0ca5e864c07196707464013436e \
  --upstream-base 2c10f0fb69af2a95e2c5764313f128774f1dda9f \
  --output "$EVIDENCE_DIR/layer-3-pp2048.json" \
  --layer 3 --tokens 2048 --warmup 2 --iterations 1 --samples 5
```

The harnesses reject a candidate SHA mismatch or tracked source changes. Rebuild
the native extension for the selected checkout before measuring, and verify that
the recorded native artifact hashes are stable across paired runs.

## Bundle layout and projection policy

- `harness/` contains the three benchmark programs.
- `results/` contains structured engine and real-layer results.
- `validation/` contains JUnit reports.
- `provenance.json` records commit pairs, the canonical diff hash, projected
  artifact hashes, and every transformation.
- `SHA256SUMS` covers every other public artifact in this directory.

Absolute paths and the test host name were replaced with documented placeholders.
The original and projected hashes are both retained in `provenance.json`. Duplicate
process wrappers, local drafts, archives, native binaries, and model weights are
not included.

This evidence covers one GLM-5.2 4-bit quant on one M3 Ultra. It does not establish
cross-chip behavior, another GLM quant, or a cross-model-family optimization.
