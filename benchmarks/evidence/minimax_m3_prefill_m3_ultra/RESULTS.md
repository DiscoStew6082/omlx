# MiniMax M3 long-prefill results

## Outcome

Two independently validated candidates remain:

1. Make MiniMax model discovery prefer attributes before mapping keys.
   `mlx.nn.Module` and therefore `VLMModelAdapter` subclass `dict`; untouched
   upstream treated the adapter as a plain mapping and missed both its
   `model_type` and `_uses_minimax_m3_positions` attributes. This activates
   oMLX's existing 4,096-token adaptive prefill path. Independent branch/commit:
   `fix/minimax-m3-adapter-detection` at
   `eab69f6fbd9fc0aacd3fccb0a800da8d5a15d70f` (the original investigation
   commit is `9428648dc4446bcd4c5bfe51359106fa936d2b55`).
2. Make the eligible Steel-MMA sparse-attention K1 `auto` path use its existing
   64-thread tile instead of the 128-thread tile. This is independently useful
   with untouched upstream's 2,048-token chunking and is bit-exact on every
   production chunk shape tested. Independent branch/commit:
   `perf/minimax-m3-k1-bq64` at
   `d531358e3a26e60d20b5db06ca65defa4733a313`.

These should be separate PRs. They address different root causes and source
owners, have independent performance wins, and can be reviewed/reverted
without coupling scheduler model detection to a Metal-kernel launch choice.

## Frozen workload and settings

- Model: `${OMLX_MINIMAX_MODEL}`
- Upstream runtime: `ded2bbe4bb30b37dc291a0f76dd34932dbf90ee9`
- Long prompt: 16,509 tokens; external prefill: 16,508 tokens
- Prompt SHA-256: `e31aa541996901bff2e597a0b152839ee7415042607886284393f2ee26f903d5`
- Token-ID SHA-256: `653beb6397c7d06ca3d14827b60ad18fbf12dbdae6111a2a16b1dde01d9605ab`
- Short prompt: 2,047 tokens; external prefill: 2,046 tokens
- Temperature `1.0`, top-p `0.95`, top-k disabled, seed `1729`, thinking disabled
- Maximum generation: 256 tokens
- Prefix/paged SSD cache, SpecPrefill, speculative decoding, and VLM MTP: off
- One sequence; scheduler setting remains `prefill_step_size=2048`; the
  candidate merely restores the intended MiniMax adaptive `4096` path.

The exact prompt, token IDs, source hashes, and expected quality facts are in
`workload/`.

## Performance

Values are medians. Variance is the sample variance across independent runs.

### Process-cold, three processes per variant

| Metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 51.1743 s (var 0.856132) | 49.2338 s (var 0.855108) | -3.79% |
| Prefill rate | 322.584 tok/s (var 32.0062) | 335.298 tok/s (var 42.3672) | +3.94% |
| TTFT | 52.9404 s (var 0.003090) | 49.4084 s (var 0.000047) | -6.67% |
| End-to-end | 60.7275 s (var 0.003980) | 56.5560 s (var 0.000004) | -6.87% |
| Model load | 8.8060 s (var 0.003842) | 8.8225 s (var 0.000849) | +0.19% |
| Peak MLX | 228.850 GiB | 229.643 GiB | +0.793 GiB |

### Alternating loaded-model A/B, five pairs after one warmup per variant

| Metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 51.2073 s (var 0.000164) | 47.6597 s (var 0.000025) | -6.93% |
| Prefill rate | 322.376 tok/s (var 0.006481) | 346.373 tok/s (var 0.001314) | +7.44% |
| TTFT | 51.3326 s (var 0.000161) | 47.7840 s (var 0.000029) | -6.91% |
| End-to-end | 58.9095 s (var 0.000307) | 54.6925 s (var 0.000696) | -7.16% |
| Peak MLX | 228.850 GiB | 229.643 GiB | +0.793 GiB |
| Physical footprint after prefill | 230.356 GiB | 230.337 GiB | -0.018 GiB |

The five paired prefill-rate improvements were 7.420%, 7.427%, 7.466%,
7.465%, and 7.459% (median 7.459%; paired-delta variance 0.000494).

### Native MiniMax top-k path, three alternating loaded-model pairs

The first tables used the Python fallback because the isolated worktree had no
built extension. Xcode's `metallib` was present but blocked by the command
sandbox. Building oMLX's existing upstream MiniMax extension outside that
sandbox succeeded, passed its ABI probe, and enabled the native path for both
variants in this A/B:

| Metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 50.6532 s (var 0.000996) | 47.0449 s (var 0.000296) | -7.12% |
| Prefill rate | 325.902 tok/s (var 0.041214) | 350.899 tok/s (var 0.016431) | +7.67% |
| TTFT | 50.7798 s (var 0.000987) | 47.1716 s (var 0.000335) | -7.11% |
| End-to-end | 57.7069 s (var 0.000703) | 54.0944 s (var 0.002250) | -6.26% |
| Peak MLX | 228.437 GiB | 229.643 GiB | +1.207 GiB |

The paired throughput improvements were 7.624%, 7.752%, and 7.605% (median
7.624%; variance 0.006352). A separate warmed candidate-only comparison showed
that native top-k contributed another 1.29% over fallback (346.440 to 350.907
tok/s) with no memory increase. The candidate's improvement therefore remains
substantial when the release-style native kernel is active.

### Sparse-attention K1 64-thread tile

Barrier-isolated profiling attributed about 5.51 seconds, or 11.5% of
candidate external prefill, to `msa_sparse_attention_b1`. oMLX already carried
two Steel-MMA K1 launch shapes: `auto` used eight query tokens per group (128
threads), while `steel_mma_bq64` used four (64 threads). Three alternating
native pairs after one warmup per variant produced:

| 16.5K Q4 metric | Auto / 128 threads | 64 threads | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 47.0549 s (var 0.0000969) | 46.6124 s (var 0.0000539) | -0.94% |
| Prefill rate | 350.825 tok/s (var 0.005391) | 354.155 tok/s (var 0.003113) | +0.95% |
| TTFT | 47.1785 s | 46.7369 s | -0.94% |
| End-to-end | 54.0711 s | 53.6441 s | -0.79% |
| Peak MLX | 229.643 GiB | 229.643 GiB | 0 |

The paired prefill-rate gains were 0.913%, 0.930%, and 0.962% (median
0.930%; variance 0.000625). All six outputs passed and had the exact same hash.

The same isolated change, while forcing the adapter fix off and restoring
untouched upstream's 2,048-token chunks, improved median prefill from 325.995
to 331.447 tok/s (+1.67%). The three paired gains were 1.613%, 1.658%, and
1.682%; peak MLX was identical and all six output hashes were identical. This
proves the kernel change is independently useful and does not need to be
stacked on the adapter-detection PR.

The 2K Q4 control was noise-flat at -0.026% paired-median throughput with
identical memory and output hashes. On Jundot oQ3, the 64-thread tile improved
347.161 to 350.388 tok/s (+0.93%); all six outputs had the same hash and both
sides retained the same pre-existing seven-of-eight strict-fact result. A
new three-pair 2K Q3 control measured 379.799 versus 379.753 tok/s (-0.016%
paired-median), with identical peak memory and output hashes; all six outputs
passed all eight facts. A
direct Metal numerical probe compared the two launch shapes at `(Q, KV)` of
`(4096, 8192)`, `(4096, 12288)`, `(4096, 16384)`, and `(124, 16508)`: every
output element was bit-identical (`max_abs_difference=0`, zero mismatches).

Finally, an unhooked default-runtime request after the source change achieved
354.482 tok/s, used the intended five chunks, and passed the quality gate. The
A/B hook is therefore not required to activate the candidate.

### 2K short-prompt control, five alternating pairs

| Metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 5.268441 s (var 0.000004) | 5.268543 s (var 0.000003) | +0.002% |
| Prefill rate | 388.350 tok/s (var 0.022338) | 388.343 tok/s (var 0.017163) | -0.002% |
| TTFT | 5.345388 s | 5.344470 s | -0.017% |
| End-to-end | 11.109762 s | 11.121493 s | +0.106% |
| Peak MLX | 227.525 GiB | 227.525 GiB | 0 |

The short path remained one 2,046-token chunk for both variants.

### Jundot oQ3 accuracy and portability control

The downloaded `Jundot/MiniMax-M3-oQ3` tokenizer produced the exact same long
and short token-ID hashes as the frozen 4-bit workload. Three alternating
loaded-model pairs on the current worktree produced:

| Long metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Prefill time | 51.6408 s (var 0.001127) | 48.1334 s (var 0.000174) | -6.79% |
| Prefill rate | 319.670 tok/s (var 0.043226) | 342.964 tok/s (var 0.008847) | +7.29% |
| TTFT | 51.7918 s (var 0.537737) | 48.2599 s (var 0.000166) | -6.82% |
| End-to-end | 57.5041 s (var 0.603650) | 53.9291 s (var 0.000706) | -6.22% |
| Peak MLX | 182.305 GiB | 183.102 GiB | +0.797 GiB |

Both variants answered seven of eight exact long-prompt facts on every run and
missed the same fact in the same way: both returned
`_msa_prefill_attention`, while the frozen 4-bit expectation was
`msa_sparse_attention_b1`. Upstream was therefore already 0/3 on the strict
cross-quant gate, and the candidate was neither better nor worse (also 0/3).
The stable output hashes were `a8a6f754...1bb9` upstream and
`af0fa072...606d` candidate. Enabling native top-k for the candidate reproduced
its output hash exactly.

The 2K Q3 control passed all facts on all six runs. Median prefill throughput
was 379.765 tok/s upstream and 379.910 tok/s candidate (+0.038%); both paths
used one chunk. This establishes no observed candidate accuracy regression on
Q3, while also recording that Q3 itself does not fully satisfy the 4-bit long
prompt's exact-answer gate.

## Profile and dispatch evidence

The profile used the real external-prefill scheduler path. Python dispatch
counters perturb timing, so its timing is diagnostic rather than part of the
official performance gate.

| Activity | Upstream 2,048 | Candidate 4,096 |
| --- | ---: | ---: |
| External-prefill forwards | 9 | 5 |
| Sparse/MoE-layer forward groups | 513 | 285 |
| Grouped MSA top-k calls | 399 | 228 |
| B=1 sparse-attention calls | 399 | 228 |
| Quantized SwitchLinear prefill calls | 1,026 | 570 |
| Profiled prefill time | 54.2346 s | 48.4673 s |

Upstream used sparse MSA only after density fell below 0.5, yielding seven
sparse chunks across 57 layers. The candidate yielded four sparse chunks.
Both used `build_grouped_msa_topk` at this 16.5K KV length; the blockwise
builder threshold is 32,768.

The first dispatch profile justified a second, lower-overhead pass. That pass
forced each target's inputs, evaluated its output, and synchronized only at the
measured component boundary. Two repeated candidate runs gave:

| Component | Calls/run | Time/run | Share of measured prefill | Decision |
| --- | ---: | ---: | ---: | --- |
| `MiniMaxSparseMoeBlock` | 285 | 28.733 s | 58.8% | Inspect compute vs routing |
| `MiniMaxAttention` | 300 | 18.129 s | 37.1% | Inspect sparse executor |
| `MiniMaxPackedSwitchGLU` | 285 | 28.290 s | 58.1% | Compute is material |
| `_minimax_moe_select` | 285 | 0.188 s | 0.39% | Reject routing rewrite |
| `msa_sparse_attention_b1` | 228 | 5.508 s | 11.5% | Test K1 launch shape |
| grouped MSA top-k | 228 | 0.712 s | 1.48% | Existing native path sufficient |
| `_sync_and_clear_cache` | 7 | 0.034 s | 0.07% | Reject sync removal |
| post-eval idle synchronize | 5 | 0.00016 s | <0.001% | Reject sync removal |

Layer-level attention plus MoE accounted for 95.8% of measured prefill.
Projection-level attribution inserted 2,675 barriers and slowed the profile by
10–15%, so only its direction is used: `QuantizedSwitchLinear` accounted for
about 28.27 seconds versus 13.92 seconds for dense `QuantizedLinear`. This
supports future projection-kernel work but not a narrow oMLX source change.

As a direct MoE routing hypothesis check, disabling the existing expert sort
reduced throughput from 351.04 to 119.93 tok/s and increased prefill from
47.03 to 137.65 seconds. The remaining predeclared pairs were stopped after
that complete pair because the proposed direction was already a 65.8%
throughput regression. Sorting is essential; there is no routing PR.

## Correctness and cache evidence

- All 26 official measured outputs passed every frozen fact check. The final
  committed default-path long run and all six native-path A/B outputs also
  passed.
- Each long variant produced one stable output hash across cold and loaded
  runs: upstream `933a39b...4189d`; candidate `5277aa31...1a5a`. Different
  sampled wording is expected when chunk shape changes floating-point order;
  both JSON answers contain every required fact.
- The committed default-path candidate exactly reproduced the candidate A/B
  output hash and showed the live detector/config enabled without harness
  override.
- Every Q4 K1 A/B output passed: six candidate-long, six upstream-long, and six
  short-control outputs. Each comparison produced one identical output hash on
  both sides. The unhooked K1 default-path smoke also passed.
- All six long Q3 K1 outputs were byte-identical and retained the same
  pre-existing seven-of-eight strict-fact result on both sides. All six short
  Q3 K1 outputs passed all facts and were byte-identical. The four direct
  native kernel cases had zero element mismatches.
- Both variants produced 60 caches: 3 `KVCache`, 57 `MiniMaxM3KVCache`,
  2,287,534,080 known cache bytes, and zero cached prompt tokens.
- Prefix cache, block-aware cache, draft cache, SpecPrefill draft, and VLM MTP
  drafter were absent in every official record.
- Relevant test slice: 81 passed (scheduler chunked prefill, MiniMax VLM
  compatibility, and benchmark harness/evidence tests). The frozen workload
  hash verifier also passed every check.

## Rejected or deferred approaches

- **8,192-token chunks — rejected.** Diagnostic prefill improved only another
  ~3.6% over 4,096, while peak MLX rose to 231.642 GiB (+2.79 GiB versus
  upstream). That is not a safe default for 256-GB-class deployments.
- **6,144-token chunks — rejected.** An unprofiled run reached only 316.61
  tok/s and 52.139 s, slower than the 4,096 candidate, while peak MLX rose to
  230.474 GiB.
- **Coalesce the 124-token tail — rejected.** Merging it into a 4,220-token
  final chunk reduced the forward count from five to four but reached only
  338.18 tok/s and raised peak MLX to 230.474 GiB. Avoiding a tiny final
  dispatch did not repay the larger attention allocation.
- **Disable adaptive prefill for sub-4-bit defaults — rejected.** An initial Q3
  comparison lacked explicit current-worktree import provenance and appeared
  to favor a safety guard. After correcting provenance and the post-commit A/B
  toggle, current upstream and candidate missed the same one strict fact while
  the candidate retained +7.29% prefill throughput. The guard was unnecessary
  and was fully reverted; its raw diagnostics remain for auditability.
- **Remove scheduler synchronization — rejected by measurement.** Cache-clear
  sync averaged about 34 ms across the entire 47-second prefill (0.07%); the
  post-eval idle synchronization totaled about 0.16 ms. These are
  memory-pressure and stream-safety seams, and their entire measured cost is
  noise-sized.
- **Sparse-attention rewrite — narrowed to the K1 tile PR.** Sparse attention
  was material at about 5.51 seconds, while grouped top-k was only 0.71 seconds.
  The existing native top-k already added 1.29%. The safe existing 64-thread
  K1 launch produced a further validated gain, so no math or top-k rewrite is
  proposed.
- **Remove MoE expert sorting — rejected after one complete A/B pair.** The
  unsorted path was 65.8% slower in throughput (119.93 versus 351.04 tok/s).
- **MoE route selection rewrite — rejected.** Route selection was only about
  0.19 seconds, or 0.39% of prefill; MoE compute was about 28.29 seconds.
- **Quantized projection rewrite — deferred, not proposed.** Barrier profiling
  confirms projection compute is material, especially
  `QuantizedSwitchLinear`, but the primitive is supplied by MLX and no narrow,
  numerically safe oMLX implementation hypothesis emerged. A new custom
  quantized MoE kernel would be a separate research project, not part of these
  PRs.
- **Blockwise top-k threshold/chunk tuning — not applicable.** The 16.5K path
  stays below the 32,768-KV threshold.

## Limitations

- One M3 Ultra 512-GB host and two local checkpoints were measured: the 4-bit
  performance target and Jundot oQ3 as an accuracy/portability control.
- Process-cold means a new Python/model process, not a purged macOS filesystem
  cache. Alternating loaded-model pairs are the primary low-variance result.
- Metal capture was not used for duration attribution. A broad `xctrace` Metal
  System Trace hit its 90-second limit before the first measured request and
  reported shader timeline disabled. A scoped MLX GPU capture grew to 225 GiB
  without completing and was stopped; its incomplete trace was deleted. Both
  methods severely distorted this 240+ GiB workload. Attribution therefore
  uses explicit MLX evaluation barriers at the Python/model boundary, while
  official timing remains at the scheduler seam without those barriers.
- Production concurrent-request latency and quantizations beyond these two
  checkpoints were not benchmarked. The short single-request controls guard
  the observed local regression surface.
- Q3 baseline and candidate both miss one strict long-prompt fact, so the Q3
  evidence proves parity rather than absolute acceptance against the 4-bit
  expected answer.
- The pre-existing LM Studio embedding and Qwen workers remained untouched by
  benchmark commands. LM Studio later unloaded both automatically at 11:17 due
  to its TTL policy; its system-resource worker remained running at the final
  process check.

## Separately authorized Jundot quantization check

`Jundot/MiniMax-M3-oQ3` was downloaded in the background to
`${OMLX_MINIMAX_OQ3_MODEL}` at Hub revision
`27ff107bc3711e4610cd74b2a7f8bb38eebeb225`. Hugging Face CLI 1.10.2 verified
all 48 repository files and checksums with no missing files. The directory has
31 referenced weight shards (none missing), an indexed tensor size of
191,500,496,134 bytes, and 178 GiB allocated on disk.

The config uses affine group-size-64 quantization with a 3-bit default and 365
per-module overrides: 162 at 4-bit, 110 at 5-bit, 11 at 6-bit, and 82 at 8-bit.
It declares `minimax_m3_vl` and
`MiniMaxM3SparseForConditionalGeneration`. It was initially verified while the
4-bit investigation proceeded, then used in the explicitly requested Q3
accuracy and short-prompt controls above. It was not substituted into the
official 4-bit performance result.
