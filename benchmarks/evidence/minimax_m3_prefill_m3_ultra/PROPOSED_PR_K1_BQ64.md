# Proposed PR 2

Title: `perf(minimax): use 64-thread sparse-attention K1 tile`

## Summary

- Make the eligible MiniMax M3 Steel-MMA sparse-attention K1 `auto` path use
  four query tokens per group (64 threads) instead of eight (128 threads).
- Preserve every existing eligibility check and fallback path.
- Add a focused regression test for the automatic tile selection.

## Profiling justification

Barrier-isolated profiling of the real 16.5K external-prefill path attributed
about 5.51 seconds (11.5%) to `msa_sparse_attention_b1`, versus 0.71 seconds
to grouped top-k. This made the already-present alternative K1 tile a narrow,
evidence-backed hypothesis; it does not change attention math.

## Benchmark

Apple M3 Ultra, 512 GB, fixed hashed 16,509-token code prompt, MiniMax-M3-4bit,
temperature 1.0/top-p 0.95/seed 1729, full 256-token generation, prefix caching
and speculative paths disabled.

- Independently on untouched upstream 2,048-token chunking, three alternating
  loaded pairs improved median prefill from 325.995 to 331.447 tok/s (+1.67%);
  TTFT fell 1.64%. Paired gains: +1.613%, +1.658%, +1.682%.
- With the separate adapter-detection candidate's 4,096-token chunking, three
  alternating pairs improved 350.825 to 354.155 tok/s (+0.95%); TTFT fell
  0.94%. Paired gains: +0.913%, +0.930%, +0.962%.
- Peak MLX memory was identical on both sides of both comparisons.
- Three 2K pairs were noise-flat at -0.026% paired-median throughput, with
  identical memory and output hashes.
- Jundot oQ3 improved 347.161 to 350.388 tok/s (+0.93%). Both variants had the
  exact same output hash and same pre-existing seven-of-eight strict-fact
  result.
- Three 2K oQ3 pairs were noise-flat at -0.016% paired-median throughput. All
  six outputs passed all facts with identical output hashes and peak memory.

## Correctness

- Every Q4 A/B output passed the frozen fact gate; each comparison produced
  byte-identical sampled output on both sides.
- Four direct native cases cover every actual long-prompt chunk shape:
  `(Q, KV) = (4096, 8192), (4096, 12288), (4096, 16384), (124, 16508)`.
  All had zero differing output elements (`max_abs_difference=0`).
- An unhooked default-runtime request after the source change passed and ran at
  354.482 prefill tok/s, confirming the benchmark selector is not required.

## Tests

- All 12 MiniMax compatibility tests pass independently on a branch based
  directly on `upstream/main`.
- `81 passed`: scheduler chunked-prefill, MiniMax VLM compatibility, and
  benchmark harness/evidence tests. Ruff passes with pre-existing vendored-file
  `UP045`/`N806` findings ignored.

## Risk

The change affects only the Steel-MMA K1 path after its existing Metal, shape,
dimension, block-size, and GQA eligibility checks pass. Scalar, SIMD, packed
SIMD, and unsupported-shape fallbacks are unchanged. The two launch shapes are
bit-exact on the four production shapes tested, and the candidate adds no peak
memory.
