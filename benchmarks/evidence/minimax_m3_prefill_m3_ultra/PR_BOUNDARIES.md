# Recommended PR boundaries

## Decision

Open two independent product PRs.

| PR | Product change | Independent evidence | Recommendation |
| --- | --- | --- | --- |
| 1 | Attribute-first detection for dict-backed VLM adapters | +7.67% native long-prefill rate; short control flat; Q4 pass; Q3 parity | Open independently |
| 2 | Use the existing 64-thread Steel-MMA sparse K1 tile in `auto` | +1.67% on untouched upstream chunking; +0.95% on PR 1; short control flat; bit-exact Q4/Q3 | Open independently |
| None | Remove MoE expert sorting | -65.8% throughput in a complete pair | Reject |
| None | Rewrite MoE routing | Routing is only 0.39% of measured prefill | Reject as noise-sized |
| None | Remove scheduler synchronization | Measured total is about 0.07% | Reject as unsafe/noise-sized |
| None | Rewrite native grouped top-k | Existing upstream native path is exact and already adds 1.29%; measured share is 1.48% | No new PR |
| Future research | Custom quantized MoE/projection kernels | Material compute share, but no narrow safe oMLX hypothesis | Do not mix into either PR |

## Why PR 1 and PR 2 should not be combined

- They fix different causes: scheduler model discovery versus a Metal launch
  geometry choice.
- PR 2 improves untouched upstream independently: three alternating pairs were
  +1.613%, +1.658%, and +1.682% in prefill tok/s.
- Their risk and review surfaces differ. PR 1 changes Python object discovery;
  PR 2 changes only the eligible sparse-attention K1 launch tile.
- Either can be reverted without removing the other benefit.
- Separate PRs make the memory tradeoff explicit: PR 1 adds about 1.207 GiB in
  the native comparison by using larger chunks; PR 2 adds zero peak MLX memory.

The benchmark harness, component profiler, raw JSONL, and reproduction scripts
are support evidence. They do not constitute a third product PR. They may be
included as review artifacts in the relevant PR or retained locally if the
maintainers prefer a smaller product diff.

## Local branch proof

- PR 1 was cherry-picked cleanly onto untouched `upstream/main` in the local
  branch `fix/minimax-m3-adapter-detection` at
  `eab69f6fbd9fc0aacd3fccb0a800da8d5a15d70f`; its source/test diff is only two
  files, and all 60 scheduler chunked-prefill tests pass on that branch. The
  original investigation commit is
  `9428648dc4446bcd4c5bfe51359106fa936d2b55`.
- PR 2 was cherry-picked cleanly onto untouched `upstream/main` in the separate
  local branch `perf/minimax-m3-k1-bq64` at
  `d531358e3a26e60d20b5db06ca65defa4733a313`; its source/test diff is only two
  files, and all 12 MiniMax compatibility tests pass on that branch.
- No branch was pushed and no remote object was created or updated.
