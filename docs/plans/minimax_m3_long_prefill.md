# MiniMax M3 Long-Prompt Prefill Plan Ledger

Status: Complete.

## Scope

Investigate MiniMax M3 long-prompt prefill performance from a clean checkout of
`upstream/main` using the local
`${OMLX_MINIMAX_MODEL}` checkpoint. Preserve raw
evidence and local commits. Do not change the original checkout, unload the LM
Studio embedding model, publish a branch, or mutate any pull request or issue.

The separately authorized `Jundot/MiniMax-M3-oQ3` download is a follow-up
accuracy and portability control. The 4-bit model remains the frozen
performance target for this investigation.

## Completed Outcomes

- Established untouched-upstream cold and alternating loaded-model baselines
  for a fixed 16,509-token production-sampled code workload.
- Validated attribute-first adapter detection as an independent scheduler fix.
- Used barrier-isolated component timing to justify and validate the existing
  64-thread sparse-attention K1 tile as a second independent candidate.
- Completed Q4 and oQ3 long/short performance and correctness controls for both
  candidates. Rejected routing, synchronization, and unsorted-MoE hypotheses.

## Current Decisions And Frozen Seams

- Runtime baseline commit: `ded2bbe4bb30b37dc291a0f76dd34932dbf90ee9`.
- Benchmark model: `${OMLX_MINIMAX_MODEL}`.
- Sampling: temperature `1.0`, top-p `0.95`, top-k disabled, seed `1729`.
- Thinking mode is disabled to keep output length and quality checks bounded.
- Prefix caching, paged SSD caching, SpecPrefill, DFlash, and VLM MTP are off.
- The workload prompt text and token IDs are committed and SHA-256 hashed.
- Long-prompt gains must exceed measured noise. A candidate must retain the
  factual output-quality gate and must not regress the short-prompt control by
  more than 3% at the median without an explicit, evidence-backed tradeoff.
- Evidence and commits remain local. Stop before every remote write.

## Work Waves

| Wave | Slice | Owned Files | Notes |
| --- | --- | --- | --- |
| 1 | Workload and harness | `benchmarks/minimax_m3_prefill/`, `benchmarks/evidence/minimax_m3_prefill_m3_ultra/` | No runtime changes |
| 2 | Upstream baseline | Evidence tree only | Repeated process-cold and loaded-model runs |
| 3 | Profiling | Evidence tree only | Counters plus Metal capture where usable |
| 4 | Isolated candidates | MiniMax runtime files and focused tests | One hypothesis at a time |
| 5 | Final A/B gate | Evidence, tests, and this ledger | Alternating loaded-model A/B plus cold confirmation |

## Definition Of Done

- Fixed workload is between 16,000 and 17,000 tokens and both prompt text and
  token IDs have verified hashes.
- Raw runs include exact prefill time/rate, TTFT, end-to-end latency, memory,
  cache counts, output, quality result, and dispatch/chunk activity.
- Upstream and candidate each have repeated cold evidence; the final comparison
  includes alternating loaded-model A/B runs, medians, and variance.
- Every tried optimization has a kept/rejected decision with evidence.
- Focused correctness tests and the relevant upstream test slice pass after the
  final runtime change.
- Candidate and evidence are committed locally; nothing is pushed or published.

## Verification

```bash
python benchmarks/minimax_m3_prefill/build_workload.py --verify
python benchmarks/minimax_m3_prefill/run_benchmark.py --help
pytest -q tests/test_minimax_m3_prefill_benchmark.py
git diff --check
```

## Implementation Ledger

- 2026-08-10: Created the isolated branch from live upstream and froze the
  experiment contract. No runtime changes yet.
- 2026-08-10: Profiled the live 16,509-token path. Upstream ran nine external
  prefill forwards and the 57 sparse/MoE layers produced 399 grouped-MSA
  top-k/sparse-attention calls. The intended 4,096-token path reduced this to
  five forwards and 228 calls.
- 2026-08-10: Located the production activation defect: `mlx.nn.Module`
  subclasses `dict`, while `_get_attr_or_key` read mapping keys before real
  attributes. The live adapter's Python `bool` MiniMax marker and `str`
  `model_type` were therefore ignored. The isolated candidate reverses that
  lookup order and adds a dict-subclass regression test.
- 2026-08-10: Corrected an A/B-harness post-commit hazard so `upstream-*`
  labels always disable the candidate, then repeated current-worktree Q3 and
  native-kernel controls.
- 2026-08-10: Barrier-isolated timing attributed 58.8% of measured prefill to
  MoE compute, 37.1% to attention, 11.5% specifically to sparse attention,
  0.39% to MoE route selection, and about 0.07% to cache-clear
  synchronization. This rejected routing/sync changes and justified testing
  only the existing K1 launch alternative.
- 2026-08-10: Validated the 64-thread K1 tile independently on untouched
  upstream (+1.67%) and with the adapter fix (+0.95%). Q4 and oQ3 long outputs
  retained correctness parity; Q4 and oQ3 short controls were noise-flat.
- 2026-08-10: Completed the missing oQ3 K1 short control: all six outputs passed
  all facts with identical hashes and memory; paired-median throughput was
  -0.016%. Drafted a local, unpublished scope-clarification issue.

## Review Findings

- Keep: attribute-first MiniMax model detection, which activates the existing
  4,096-token adaptive prefill configuration. Five loaded-model A/B pairs show
  a stable long-prefill improvement with all quality gates passing.
- Reject: 8,192-token chunks. The diagnostic added only about 3.6% beyond the
  4,096 candidate while raising peak MLX memory to 231.64 GiB, an unattractive
  tradeoff on the 256-GB-class systems capable of loading this checkpoint.
- Reject: 6,144-token chunks and coalescing the 124-token final tail. Both were
  slower than the 4,096 candidate and raised peak MLX memory to 230.47 GiB.
- Keep separately: the existing 64-thread Steel-MMA K1 tile. Sparse attention
  was a measured 11.5% of prefill, and the tile improved untouched upstream by
  1.67% with zero peak-memory change and bit-exact native outputs.
- Reject: MoE route-selection and synchronization changes. They accounted for
  only 0.39% and about 0.07% of measured prefill respectively. Disabling expert
  sorting regressed throughput by 65.8%.
- Defer: custom quantized-projection/MoE kernels. Projection compute is
  material, but no narrow, numerically safe oMLX implementation hypothesis was
  found.

## Final Completion Evidence

- Candidate code: `9428648dc4446bcd4c5bfe51359106fa936d2b55`.
- Five alternating loaded-model pairs improved median long-prefill throughput
  from 322.376 to 346.373 tok/s (+7.44%) and TTFT from 51.333 to 47.784
  seconds (-6.91%). Paired throughput deltas ranged from +7.420% to +7.466%.
- Three process-cold runs per variant improved median throughput from 322.584
  to 335.298 tok/s (+3.94%) and TTFT from 52.940 to 49.408 seconds (-6.67%).
- Five alternating 2K controls measured -0.002% throughput, within noise; both
  variants used one chunk. Peak MLX memory increased by 0.793 GiB at 16.5K.
- With oMLX's existing native MiniMax top-k extension active for both sides,
  three alternating pairs improved median throughput from 325.902 to 350.899
  tok/s (+7.67%) and TTFT from 50.780 to 47.172 seconds (-7.11%). The extension
  built successfully outside the command sandbox and passed ABI plus exact
  numerical equivalence checks against fallback.
- All 26 official A/B/cold outputs and the committed default-path confirmation
  passed the frozen factual quality gate. The relevant combined test slice
  passed 81 tests; Ruff, workload hash verification, and `git diff --check`
  passed.
- The separately authorized Jundot oQ3 download completed and independently
  verified all 48 repository files/checksums at revision
  `27ff107bc3711e4610cd74b2a7f8bb38eebeb225`. Three current-worktree long A/B
  pairs showed +7.29% prefill throughput; both upstream and candidate returned
  the same seven of eight strict facts, so no candidate-specific Q3 accuracy
  regression was observed. All six short Q3 controls passed with a noise-sized
  +0.038% throughput difference.
- Three oQ3 K1 long pairs improved prefill throughput by +0.93%, with identical
  output hashes, memory, and the same seven-of-eight baseline fact matrix. All
  six oQ3 K1 short outputs passed all facts with identical hashes and memory;
  throughput was noise-flat at -0.016% paired median.
- No command targeted LM Studio. Its logs show that the pre-existing embedding
  and Qwen workers were automatically unloaded at 11:17 by LM Studio's TTL
  policy, rather than by this investigation.
- Raw evidence, summaries, environment identity, reproduction commands,
  rejected approaches, limitations, and proposed PR text are preserved under
  `benchmarks/evidence/minimax_m3_prefill_m3_ultra/`.
- The original checkout remains unchanged. The evidence branch was pushed to
  the user's fork and the approved scope-clarification issue was filed as
  [jundot/omlx#2590](https://github.com/jundot/omlx/issues/2590). No PR was
  created and no candidate branch was pushed to upstream.

## Future Scope Decision

The local engineering investigation is complete. Maintainer scope/ownership
guidance is requested in [jundot/omlx#2590](https://github.com/jundot/omlx/issues/2590):
whether to accept two independent PRs, whether the vendored `mlx-vlm` K1 launch
choice belongs in oMLX, and whether oQ3 parity should be a required gate. The
exact filed body is preserved in
`benchmarks/evidence/minimax_m3_prefill_m3_ultra/DRAFT_SCOPE_CLARIFICATION_ISSUE.md`.
No additional remote mutation should occur without explicit user approval.
