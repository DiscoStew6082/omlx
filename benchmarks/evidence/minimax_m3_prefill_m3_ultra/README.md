# MiniMax M3 long-prefill evidence

Status: complete, two validated local candidates; unpublished.

This directory preserves the fixed workload, raw benchmark/profile records,
summaries, rejected hypotheses, and exact reproduction commands for MiniMax M3
long-prompt prefill work on the local M3 Ultra host.

Frozen baseline:

- oMLX upstream commit: `ded2bbe4bb30b37dc291a0f76dd34932dbf90ee9`
- model: `${OMLX_MINIMAX_MODEL}`
- prefix and paged SSD caching: disabled
- speculative decoding, VLM MTP, and SpecPrefill: disabled
- sampling: temperature 1.0, top-p 0.95, top-k disabled, seed 1729
- fixed workload: see `workload/manifest.json`

Raw result directories are append-only during the experiment. Results are not
public claims until the final ledger records the candidate gate and limitations.
The exact prefill timer covers the scheduler's external prefill of
`prompt_tokens - 1`; the final prompt token is intentionally included in TTFT
through `BatchGenerator.insert()`.

Start with `RESULTS.md` for the conclusion, `PR_BOUNDARIES.md` for the separate
PR decision, and `REPRODUCE.md` for exact commands. `PROPOSED_PR.md` contains
the adapter-detection PR text; `PROPOSED_PR_K1_BQ64.md` contains the independent
sparse-K1 PR text. `jundot_oq3_download.json` records the separately authorized
download verification and follow-up Q3 accuracy/portability control.
`native_topk_build.json` records the successful out-of-sandbox build and
numerical verification of oMLX's existing native MiniMax top-k extension; no
compiled binary is committed. `DRAFT_SCOPE_CLARIFICATION_ISSUE.md` is a local
issue draft only; it has not been published.
