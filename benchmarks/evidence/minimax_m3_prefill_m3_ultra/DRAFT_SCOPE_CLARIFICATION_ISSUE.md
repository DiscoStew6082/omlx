# Filed scope-clarification issue

Filed as [jundot/omlx#2590](https://github.com/jundot/omlx/issues/2590) with
explicit user approval.

Title: `Scope clarification: MiniMax M3 long-prefill fixes across Q4 and oQ3`

## Scope clarification requested

Before opening implementation PRs, could you confirm the intended ownership and
quantization scope for two independently validated MiniMax M3 long-prefill
changes?

The original target was `mlx-community/MiniMax-M3-4bit`. Follow-up testing on
`Jundot/MiniMax-M3-oQ3` shows that both candidates retain Q3 correctness parity
and their expected performance behavior, but it raises two scope questions:
whether Q3 should be a required compatibility gate, and whether the vendored
`mlx-vlm` sparse-attention launch change belongs in oMLX.

## Public benchmark evidence

The complete benchmark bundle is published on the fork's evidence branch at
commit
[`172beecaadecf045b99a229e547305dc4ae8ebc6`](https://github.com/DiscoStew6082/omlx/commit/172beecaadecf045b99a229e547305dc4ae8ebc6).
The links below are commit-pinned so the evidence reviewed here cannot move:

- [Complete results and limitations](https://github.com/DiscoStew6082/omlx/blob/172beecaadecf045b99a229e547305dc4ae8ebc6/benchmarks/evidence/minimax_m3_prefill_m3_ultra/RESULTS.md)
- [Consolidated Q3 accuracy/performance matrix](https://github.com/DiscoStew6082/omlx/blob/172beecaadecf045b99a229e547305dc4ae8ebc6/benchmarks/evidence/minimax_m3_prefill_m3_ultra/summaries/q3-accuracy-performance-matrix.json)
- [Raw Q3 K1 short-prompt A/B records](https://github.com/DiscoStew6082/omlx/blob/172beecaadecf045b99a229e547305dc4ae8ebc6/benchmarks/evidence/minimax_m3_prefill_m3_ultra/raw/q3/q3-candidate-k1-loaded-ab-short.jsonl)
- [Exact reproduction commands](https://github.com/DiscoStew6082/omlx/blob/172beecaadecf045b99a229e547305dc4ae8ebc6/benchmarks/evidence/minimax_m3_prefill_m3_ultra/REPRODUCE.md)
- [PR-boundary analysis](https://github.com/DiscoStew6082/omlx/blob/172beecaadecf045b99a229e547305dc4ae8ebc6/benchmarks/evidence/minimax_m3_prefill_m3_ultra/PR_BOUNDARIES.md)

The isolated source commits in the evidence-branch history are:

- [Attribute-first adapter detection](https://github.com/DiscoStew6082/omlx/commit/9428648dc4446bcd4c5bfe51359106fa936d2b55)
- [64-thread sparse-attention K1 tile](https://github.com/DiscoStew6082/omlx/commit/82b884a80008edc85b8ed614c6f26d491eb1fbf7)

## Proposed PR split

1. **Adapter-detection bug fix**
   - Prefer attributes before mapping keys when detecting MiniMax models.
   - This fixes production `VLMModelAdapter`, whose MLX base is also a `dict`
     subclass, and activates the existing 4,096-token adaptive-prefill path.
2. **Sparse-attention K1 performance change**
   - Make the eligible Steel-MMA `auto` path use the existing 64-thread K1 tile
     instead of the 128-thread tile.
   - Preserve all current eligibility checks and fallback paths.

These changes have different causes, source owners, memory behavior, and
rollback surfaces. Each improves untouched upstream independently, so the
current recommendation is two PRs rather than a combined performance patch.

## Q3 evidence

All measurements use a fixed hashed 16,509-token code prompt (plus a 2,047-token
control), temperature 1.0, top-p 0.95, seed 1729, full sampled generation,
prefix caching off, and speculative paths off. Each table entry is the median
of three alternating loaded-model pairs.

| Change | Q3 workload | Reference | Candidate | Result |
| --- | --- | ---: | ---: | ---: |
| Adapter detection | 16.5K | 319.670 tok/s | 342.964 tok/s | +7.29% |
| Adapter detection | 2K | 379.765 tok/s | 379.910 tok/s | +0.033% paired median |
| 64-thread K1 | 16.5K | 347.161 tok/s | 350.388 tok/s | +0.93% |
| 64-thread K1 | 2K | 379.799 tok/s | 379.753 tok/s | -0.016% paired median |

- The long adapter comparison reduced median prefill time from 51.641 to
  48.133 seconds and TTFT from 51.792 to 48.260 seconds. Peak MLX increased by
  0.797 GiB, matching the larger-chunk tradeoff seen on Q4.
- The long K1 comparison reduced prefill time from 47.551 to 47.114 seconds and
  TTFT from 47.676 to 47.237 seconds with identical peak MLX memory.
- Both 2K controls are noise-flat and use identical peak memory.
- Prefix/speculative caches were absent, all runs created 60 cache layers, and
  every run cached zero prompt tokens.

## Q3 correctness interpretation

- **Existing upstream Q3 accuracy on this frozen fact gate was 7/8 (87.5%) on
  the 16.5K workload and 8/8 (100%) on the 2K control, repeated across all
  three baseline runs.** These percentages describe this workload-specific
  factual gate, not general model accuracy.
- On the 2K control, all 12 outputs across both changes passed all eight frozen
  fact checks. Each A/B comparison produced the exact same output hash on both
  sides.
- On the 16.5K prompt, upstream Q3 already answers seven of eight Q4-derived
  strict facts. It reports `_msa_prefill_attention` where the Q4 expectation is
  `msa_sparse_attention_b1`.
- The adapter candidate misses exactly that same fact on every run and passes
  the same other seven facts. Chunking changes floating-point order, so its
  sampled output hash differs from upstream, but the fact matrix is identical.
- The K1 reference and candidate outputs are byte-identical on all six long
  runs and retain that same seven-of-eight result.
- A direct native kernel comparison across every production Q/KV chunk shape
  found zero differing output elements between the two K1 tiles.

This supports a **no-regression relative to the Q3 baseline** claim. It does not
support claiming that Q3 satisfies the Q4 checkpoint's absolute
`attention_executor` expectation.

## Workload decision

The existing v1 workload should remain frozen. Changing its prompt or expected
answer would break comparison with the recorded Q4/Q3 series, and relabeling
the Q3 near-miss as correct would overstate the evidence.

If broader Q3 accuracy confidence is required, it should be added as a separate
v2 workload rather than replacing v1. That workload should explicitly ask for
both:

- `prefill_orchestrator`: `_msa_prefill_attention`
- `attention_executor`: `msa_sparse_attention_b1`

It should also add several independent long-code comprehension prompts and,
if the goal is a general sampled-quality claim, multiple fixed seeds. That
broader suite is useful follow-up evidence but is not required to establish
regression safety for these two PRs: the adapter change preserves Q3's fact
matrix, while the K1 change is bit-exact and preserves sampled output hashes.

## Questions

1. Should the adapter-detection fix and K1 launch change be reviewed as two
   independent PRs, as proposed above?
2. Is changing the vendored MiniMax M3 `mlx-vlm` sparse-attention launch choice
   in scope for oMLX, or should that optimization first be proposed upstream to
   `mlx-vlm`?
3. Should oQ3 be a required regression/performance gate for both PRs even
   though the original performance target is Q4?
4. For Q3, should acceptance use parity with its own upstream seven-of-eight
   long-prompt result, or should the Q3-specific expected
   `attention_executor` value be recorded separately from Q4?

## Proposed resolution if there is no objection

- Submit the adapter-detection bug fix first as a narrow scheduler/runtime PR.
- Submit the K1 launch change separately in the repository selected by the
  ownership decision above.
- Keep Q4 as the primary performance target and require Q3 baseline parity,
  exact-hash parity where math is unchanged, plus a short-prompt no-regression
  control.
- Exclude MoE routing, synchronization removal, top-k rewrites, and custom
  quantized projection kernels from both PRs; profiling does not justify those
  changes in this scope.
