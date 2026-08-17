# Proposed PR 1

Title: `fix(minimax): detect dict-backed VLM adapters for adaptive prefill`

This PR should remain independent from the separately validated sparse-K1 tile
change in `PROPOSED_PR_K1_BQ64.md`.

## Summary

- Prefer object attributes before mapping keys in MiniMax model discovery.
- Detect the production `VLMModelAdapter`, whose `mlx.nn.Module` base is also a
  `dict` subclass.
- Activate the existing 4,096-token adaptive prefill path for long MiniMax M3
  requests; short requests retain the 2,048-token path.
- Add a regression test using the same dict-subclass shape.

## Root cause

`_get_attr_or_key()` checked `isinstance(obj, dict)` first. MLX modules inherit
from `dict`, so the helper called `.get("model_type")` and never read the
adapter's `model_type` property or `_uses_minimax_m3_positions` attribute. The
scheduler therefore left `_minimax_m3_adaptive_prefill` disabled in production
even though unit fixtures using ordinary Python objects passed.

## Benchmark

Apple M3 Ultra, 512 GB, MiniMax-M3-4bit, fixed 16,509-token code prompt,
temperature 1.0/top-p 0.95/seed 1729, all caches and speculative paths off.

- Five alternating loaded-model pairs: 322.38 -> 346.37 prefill tok/s (+7.44%);
  TTFT 51.33 -> 47.78 seconds (-6.91%).
- Three alternating pairs with oMLX's existing native MiniMax top-k extension:
  325.90 -> 350.90 prefill tok/s (+7.67%); TTFT 50.78 -> 47.17 seconds
  (-7.11%).
- Three process-cold runs each: 322.58 -> 335.30 median prefill tok/s (+3.94%);
  TTFT 52.94 -> 49.41 seconds (-6.67%).
- Five alternating 2K pairs: -0.002% median prefill throughput; no meaningful
  short-prompt regression.
- Peak MLX memory: +0.793 GiB; post-prefill physical footprint unchanged.
- All 26 original official sampled outputs and all six native-path A/B outputs
  passed the frozen factual quality gate.
- On Jundot oQ3, both upstream and candidate answered the same seven of eight
  strict long-prompt facts (the checkpoint's upstream output already missed
  the exact attention-executor name); the candidate retained its exact output
  hash with native top-k. All six 2K Q3 controls passed, with +0.038% median
  prefill throughput. No candidate-specific Q3 accuracy regression was found.

## Tests

- `81 passed`: scheduler chunked-prefill, MiniMax VLM compatibility, and
  benchmark harness/evidence tests.
- All 60 scheduler chunked-prefill tests pass on the minimal branch based
  directly on `upstream/main`.
- Ruff passes on changed runtime/harness files and on the scheduler test with
  its pre-existing `SIM117` findings ignored.
- Workload text and token-ID hashes verify; `git diff --check` passes.

## Risk

The change is scoped to MiniMax discovery. Plain dict configs retain the same
key fallback, while dict-subclass MLX modules now correctly prefer their real
attributes. Existing environment opt-out behavior remains unchanged.
