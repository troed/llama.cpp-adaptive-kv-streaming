# llama.cpp adaptive KV streaming - fork notes

This repository is a fork of
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming).

Upstream adds an experimental, block-granular KV cache streaming path to the CUDA
`llama-server`. With `--kv-stream-arena-mib N` the authoritative KV tensors live
in pinned host memory while one bounded CUDA arena is shared between
phase-specific compute buffers, resident KV pages, and the transfer ring. The
upstream README (linked below) covers that design, its build, and its benchmarks.

This fork keeps that implementation and adds speculative-decoding support and
memory management for the MTP draft context. Everything below is experimental.

## Sync status

- ggml-org/llama.cpp: master `1af554f8f` (2026-09-19)
- RaymondHuang210129/llama.cpp-adaptive-kv-streaming: master `f280b2698` (2026-08-24)

## TLDR

If you have a 16GB CUDA GPU - just do the following:

* Clone this repo, build with these parameters

```sh
cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA_FA_QUANTS=q8_0-q4_0 -DGGML_CUDA=ON
cmake --build build --config Release -j
```

-DGGML_CUDA_FA_QUANTS selects which K/V cache type combinations get Flash Attention kernels compiled: `type_K-type_V` pairs separated by `;` (legal types `f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0`; f16-f16 is always compiled). Include the combinations your `cache-type-k`/`cache-type-v` and the draft's `-ctkd`/`-ctvd` use, or use `all` to compile every combination (much slower build). Combinations not in the list still work: Flash Attention falls back to the f16-f16 kernel with a one-time warning, and the KV streaming direct path uses a slower F16 conversion path. `GGML_CUDA_FA_ALL_QUANTS` is a deprecated alias for `=all`.

* Download the model and its DFlash2 draft

[the ASCII condensed IQ4_XS target and the condensed DFlash2 draft](https://huggingface.co/troed/Qwen3.8-27B-ASCII-Condensed)

* Use the following parameters (models-preset.ini format) when launching llama-server

```
m = Qwen3.8-27B-ASCII-Condensed-IQ4_XS-3.84bpw.gguf
md = Qwen3.8-27B-ASCII-Condensed-DFlash2-Q2_K_S-MIX.gguf
device-draft = CUDA0
n-gpu-layers-draft = all
ctx-size = 160000
n-gpu-layers = 99
batch-size = 256
ubatch-size = 256
# lower this value if you don't have the full 16GB available for the model
kv-stream-arena-mib = 4352
cache-type-k = q8_0
cache-type-v = q4_0
spec-type = draft-dflash
spec-draft-n-max = 5
kv-stream-spec-dynamic = on
kv-stream-spec-keep-pages = 334
kv-stream-spec-reenable-pages = 8
kv-stream-spec-stable-decodes = 4
fit = off
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
reasoning = on
reasoning-preserve = on
# (slow) CPU only multimodal is better than none
no-mmproj-offload = on
mmproj = Qwen3.8-mmproj-BF16.gguf
load-mode = none
flash-attn = on
```

## Results

Measured on an RTX 5060 Ti 16 GB with Qwen3.8-27B (Q8_0 K cache, Q4_0 V
cache), `-ngl 99`, `--flash-attn on`, and `--parallel 1`. The model used is
[bsaleh03's ASCII condensed version of Unsloth UD-IQ4_XS](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed).

### Upstream vs Speculative decoding (this fork)

This build with speculative decoding disabled (`--spec-type none`) against MTP
and DFlash2, each at the largest `--kv-stream-arena-mib` that decodes on the
16 GB card (3072 MiB upstream, 3136 MiB MTP, 3200 MiB DFlash2). All run
`--ctx-size 160000`; the prompt is a source tree followed by a review
instruction, with 256 tokens generated at temperature 0. The draft KV is
quantized (`-ctkd q8_0 -ctvd q4_0`), which is why the MTP arena is smaller here
than in the F16 configuration below.

![MTP and DFlash2 decode throughput vs context](media/draft-thresholds-decode.png)

Decode t/s:

| ctx | upstream | MTP eject | MTP keep | DFlash2 eject | DFlash2 keep |
|---:|---:|---:|---:|---:|---:|
| 8K | 27.6 | 73.3 | 72.5 | 54.6 | 54.3 |
| 16K | 26.3 | 64.5 | 63.1 | 58.8 | 58.8 |
| 24K | 25.2 | 50.3 | 49.9 | 39.9 | 39.6 |
| 32K | 23.9 | 58.7 | 58.3 | 43.5 | 45.0 |
| 40K | 23.1 | 55.7 | 55.6 | 42.3 | 41.9 |
| 49K | 22.2 | 22.2 | 47.2 | 40.4 | 40.2 |
| 57K | 21.4 | 21.4 | 22.9 | 21.5 | 39.1 |
| 65K | 20.7 | 20.6 | 22.7 | 20.7 | 21.2 |
| 73K | 19.9 | 19.9 | 21.4 | 20.0 | 17.3 |
| 81K | 19.3 | 19.4 | 22.7 | 19.5 | 21.2 |
| 90K | 18.7 | 18.8 | 16.6 | 18.7 | 14.3 |
| 98K | 18.1 | 18.2 | 10.0 | 18.1 | 10.3 |
| 106K | 17.5 | 17.6 | 15.4 | 17.5 | 13.0 |
| 114K | 17.0 | 17.2 | 14.3 | 17.1 | 13.2 |
| 122K | 15.9 | 15.8 | | 16.0 | 11.3 |
| 131K | 15.1 | 15.0 | 7.6 | 15.2 | 7.7 |
| 139K | 14.5 | 14.3 | 9.9 | 14.6 | 9.3 |
| 147K | 13.2 | 13.5 | 8.8 | 13.7 | 8.0 |
| 155K | 12.1 | 12.5 | 11.2 | 13.2 | 10.8 |
| 160K | 10.8 | 11.8 | 6.4 | 13.1 | 5.9 |

![MTP and DFlash2 prefill throughput vs context](media/draft-thresholds-prefill.png)

Prefill t/s:

| ctx | upstream | MTP eject | MTP keep | DFlash2 eject | DFlash2 keep |
|---:|---:|---:|---:|---:|---:|
| 8K | 997 | 835 | 826 | 899 | 892 |
| 16K | 965 | 810 | 804 | 882 | 875 |
| 24K | 925 | 778 | 774 | 852 | 848 |
| 32K | 889 | 745 | 742 | 819 | 818 |
| 40K | 857 | 713 | 713 | 795 | 789 |
| 49K | 827 | 699 | 686 | 768 | 763 |
| 57K | 798 | 694 | 662 | 745 | 740 |
| 65K | 772 | 684 | 637 | 726 | 718 |
| 73K | 747 | 674 | 617 | 709 | 691 |
| 81K | 723 | 661 | 593 | 691 | 667 |
| 90K | 702 | 649 | 575 | 673 | 646 |
| 98K | 682 | 636 | 552 | 657 | 624 |
| 106K | 663 | 623 | 533 | 640 | 601 |
| 114K | 643 | 607 | 515 | 623 | 581 |
| 122K | 624 | 592 | | 608 | 566 |
| 131K | 604 | 576 | 483 | 590 | 547 |
| 139K | 585 | 560 | 472 | 572 | 531 |
| 147K | 566 | 544 | 458 | 556 | 514 |
| 155K | 549 | 529 | 445 | 542 | 499 |
| 160K | 540 | 521 | 438 | 534 | 490 |

Keep divided by eject (decode): above 1.0 the draft should stay.

| ctx | MTP | DFlash2 |
|---:|---:|---:|
| 8K-40K | 0.98-1.00x | 0.99-1.04x |
| 49K | 2.12x | 0.99x |
| 57K | 1.07x | 1.82x |
| 65K-81K | 1.07-1.17x | 0.87-1.09x |
| 90K | 0.88x | 0.76x |
| 98K-160K | 0.51-0.90x | 0.45-0.82x |

Findings:

- The draft is never ejected while the working set fits, so eject and keep are
  identical up to the streaming onset: about 49K tokens for MTP, 57K for DFlash2.
- The default controller ejects at that onset, which is earlier than the data
  supports. At the onset, keeping the draft is 1.8 to 2.1x faster (MTP 49K: 47.2
  vs 22.2 t/s; DFlash2 57K: 39.1 vs 21.5 t/s).
- Keeping wins through about 81K and loses from about 90K on. The keep/eject
  crossover is about 85K, or about 330 pages/layer, for both drafts.
- After ejection the decode rate matches upstream (49K: 22.2 vs 22.2; 98K: 18.2
  vs 18.1), so the draft is cleanly disabled.
- DFlash2 trails MTP at short context (its draft is five layers, not one) but its
  eject curve stays ahead of MTP at long context (160K: 13.1 vs 11.8 t/s).
- The keep arm is noisy (DFlash2 dips to 17.3 t/s at 73K), so a robust eject
  threshold is about 300 pages/layer rather than the exact crossover.

Both drafts would gain from moving the default eject point from streaming onset
(about 180 to 224 pages/layer) to about 300 pages/layer: that recovers the 1.8
to 2.1x decode advantage across the 49K to 80K band and still ejects before keep
turns negative. The MTP keep point at 122880 is missing because a pre-existing
streaming-kernel launch timeout aborts that configuration; it reproduces on a
build without any of this work, so it is unrelated to the draft generalization.

Reproduce with `benchmarks/benchmark_mtp_streaming.py` (MTP and DFlash2),
`benchmarks/benchmark_upstream_vs_mtp.py` (upstream), and
`benchmarks/plot_draft_thresholds.py` (combined table and figure).

### MTP KV quantization

The MTP draft KV defaults to F16 and does not inherit the target `-ctk`/`-ctv`.
Pass `-ctkd`/`-ctvd` to quantize it. The automatic pin now sizes the pinned MTP
KV from the draft types, so the saved bytes become decode window:

| MTP KV type | MTP KV pin | decode window |
|---|---|---|
| F16 | 164 pages / 41984 tokens | 164 pages |
| q8_0 K / q4_0 V | 178 pages / 45568 tokens | 178 pages |
| q4_0 K / q4_0 V | 182 pages / 46592 tokens | 181 pages |

Measured at arena 3072, ctx 160000, ub 256, auto pin. The window grows by 14 to
18 pages, which moves the MTP crossover from about 39K to about 42K tokens.
Prefill loses 1.5 to 4 percent versus F16. Quantizing shrinks the pin, so the
shared arena compute region grows and the init peak rises: at arena 3264 the MTP
context can fail to allocate its compute buffer. Details and the throughput
table are in
[docs/next-steps/01-mtp-kv-quantization.md](docs/next-steps/01-mtp-kv-quantization.md).

## Differences from upstream

### Phase arena: speculative verify batches
- Upstream rejected generation batches whose token count was not exactly
  `n_seq_max` ("phase arena currently supports TG1 without speculative batches").
  Speculative decoding verifies `1 + n_draft` tokens on a single sequence.
- New context parameter `llama_context_params.n_max_spec_draft` ("max speculative
  draft tokens, 0 = none"). `common_context_params_to_llama()` sets it from
  `common_speculative_n_max(&params.speculative)`, so it follows the configured
  spec type (ngram, MTP, DFlash) instead of a hardcoded value.
- `llama_context::sched_reserve()` measures the token-generation graph at
  `max(n_seq_max, 1 + n_max_spec_draft)` tokens instead of `n_seq_max`.
- `llama_context::kv_stream_switch_phase()` re-reserves the decode layout at the
  same width, so the arena compute slab fits a verify batch.
- `llama_context::process_ubatch()` admits generation batches up to
  `max(n_seq_max, 1 + n_max_spec_draft)` and otherwise fails with
  "phase arena decode batch too wide".
- `n_max_spec_draft = 0` reproduces the original behaviour.

### MTP draft context
- The MTP draft context's `n_ubatch` is capped to
  `max(8, draft.n_max + 2) * n_seq_max` so its compute graph stays small.
  `n_batch` is left unchanged, because the Qwen3.5 MTP path runs a prefill
  catch-up decode into its own KV.
- The MTP context's KV cache can be allocated from a pinned region inside the
  target's phase arena instead of a separate full-length F16 `cudaMalloc`.
- The MTP block weights can be allocated from the same pinned region, so an
  evicted MTP context returns both its KV and its weights to the arena pool.
- A separate MTP-only GGUF can be supplied with `-md` while using
  `--spec-type draft-mtp`. The target then skips its embedded MTP tensors
  (`load_mtp = false`), and the draft borrows the target's LM head.
- The pin, window cap, and dynamic eject are generalized to any pinned draft
  (MTP or DFlash2). A pinned draft reserves its weights and the widened
  recurrent-state cache in the target arena; MTP additionally pins its nextn
  KV. DFlash2's five sliding-window KV layers stay in ordinary VRAM.

### New arena and context API
- ggml-cuda: `ggml_backend_cuda_phase_arena_set_pinned()`,
  `_reset_pinned()`, and `_pinned_buffer_type()`. The pinned region is
  bump-allocated from the top of the arena; the compute region must stay below
  it. The pinned buffer type shares the arena name so it is recognised as a CUDA
  buffer.
- `llama_kv_stream_pinned_buft()` returns the target context's pinned buffer
  type.
- New experimental context parameters: `spec_mtp`, `spec_draft`,
  `draft_weights_bytes`, `n_max_spec_draft`, `kv_stream_mtp_kv_pages`,
  `kv_stream_mtp_dynamic`. `spec_draft` marks any pinned draft (MTP or DFlash2);
  `draft_weights_bytes` is the draft file size reserved in the pin. The status
  struct gains `mtp_kv_pages` (the pinned MTP KV size in pages) and
  `draft_reserved_bytes` (total pinned bytes for the active draft).
- `llama_kv_stream_draft_set()` enables or disables any pinned draft (formerly
  `llama_kv_stream_mtp_set()`).
- `llama_model_borrow_output()` lets a draft model share the target's LM head
  (`output` / `output_s`) instead of carrying a duplicate copy.

## Configuration

### Phase arena (upstream)
- `--kv-stream-arena-mib N` (alias `--kv-stream-stage-mib`): size of the shared CUDA arena in MiB; `0` disables it. The phase arena requires `--parallel 1`, `--flash-attn on`, KV offload, and a Qwen3.5-family target.

### Speculative decoding
- `--spec-type draft-mtp`: enable MTP speculative decoding.
- `-md <file>`: optional separate MTP-only GGUF; the target then skips its embedded MTP tensors (`load_mtp = false`) and the draft borrows the target LM head.
- `--spec-draft-n-max N`: number of draft tokens. It also widens the target recurrent-state cache and the decode compute slab.
- `--spec-draft-type-k T` / `--spec-draft-type-v T`: draft KV cache types (default F16). The main `--cache-type-k`/`--cache-type-v` do not affect the draft.

### Creating a separate MTP model

The MTP block can be split out of a merged GGUF with
`gguf-py/gguf/scripts/gguf_extract_mtp.py`:

```sh
python3 gguf-py/gguf/scripts/gguf_extract_mtp.py \
    Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf \
    Qwen3.8-27B-ASCII-Condensed-MTP.gguf
```

The output keeps the target vocab metadata (`token_embd`, `output_norm`) and the
`blk.<mtp>.` block. It deliberately drops `output.weight` so the draft borrows
the target LM head; pass `--with-lm-head` to keep it. Use the result with
`-md <file>`; the target then skips its embedded MTP tensors, saving their VRAM.

### Dynamic MTP eject (this fork, opt-in)
- `--kv-stream-spec-dynamic`: eject MTP when the decode working set exceeds the MTP-active decode capacity, and re-enable it when it fits again (default: disabled).
- `--kv-stream-spec-keep-pages N`: the single eject threshold: keep MTP active until the target's decode working set exceeds `N` 256-token pages, then eject. `0` (default) ejects at streaming onset; use a large value to keep MTP active throughout. The draft KV slides to follow the target. Requires `--kv-stream-spec-dynamic`.
- `--kv-stream-spec-reenable-pages N`: re-enable once the active pages fit at least `N` pages below the eject threshold (default: 8).
- `--kv-stream-spec-stable-decodes N`: consecutive decode batches required before a transition (default: 4).
- `--kv-stream-spec-kv-pages N`: size of the pinned MTP KV reservation, in 256-token pages. `0` (default) sizes the pin to the MTP-active decode window automatically; a positive `N` pins exactly `N` pages and caps the window there. Requires `--kv-stream-spec-dynamic`.

Both `--kv-stream-spec-*` and the older `--kv-stream-mtp-*` spellings are accepted; the options were renamed to cover any speculative draft.

The `LLAMA_ARG_KV_STREAM_SPEC_*` environment variables mirror these options. Ejecting returns the draft weights, the draft KV cache, and the widened recurrent-state cache to the arena pool; re-enabling restores them. (The DFlash2 draft has no pinned KV, so ejection returns its weights and the recurrent-state widening.) The default configuration ejects at streaming onset; `--kv-stream-spec-keep-pages` sets the eject threshold and `--kv-stream-spec-reenable-pages` the hysteresis band.

**One pinned draft type only.** Dynamic eject changes only the configured draft (MTP or DFlash2). If `--spec-type` mixes MTP with DFlash2 or a non-pinned speculator (for example `--spec-type draft-mtp,ngram-mod`), the server disables dynamic eject with a warning and keeps the draft pinned for the whole run.

### Tuning the dynamic MTP window

MTP is kept while the decode working set fits the MTP-active decode pool, which
is what remains of the arena after the pinned reservation (MTP weights,
recurrent-state cache, MTP KV) and the phase compute slab. A larger
`--ctx-size` reserves more and shrinks the window.

The pinned MTP KV is the largest term, and a full-context pin reserves about
4 MiB per 1000 context tokens. MTP only runs while the working set fits the
decode pool, so the MTP KV never needs the full context. The default sizes the
pin to that decode window instead. The two are coupled: a smaller pin leaves
more arena for KV and grows the window, which in turn needs a larger pin. The
default solves that fixed point directly, so the pin matches the decode
capacity with no wasted reservation. Measured at `--kv-stream-arena-mib 3264`
with this model and draft:

| `--ctx-size` | prefill resident pages/layer | decode resident pages/layer | MTP-active window   |
|-------------:|-----------------------------:|----------------------------:|---------------------|
|        32768 |                          267 |                         276 | full context (~32k) |
|        65536 |                          228 |                         239 | full context (~61k) |
|       160000 |                          172 |                         190 | ~49k tokens         |

A page is 256 tokens, so the window is `decode resident pages/layer * 256`
tokens, bounded by `--ctx-size`. When the whole context fits the decode pool,
as at `--ctx-size 32768`, no cap is applied and MTP stays active throughout.

- `--kv-stream-spec-kv-pages N` overrides the automatic pin and reserves exactly
  `N` pages of MTP KV. The window is then capped at `N` pages minus a small
  catch-up margin (the MTP context decodes every target batch, so it must absorb
  a few batches past the nominal window before the eject lands). Requires
  `--kv-stream-spec-dynamic`. Use it to trade MTP reach against pool size, or to
  bound a known working set. `--kv-stream-spec-kv-pages 0` restores the
  automatic sizing.
- `--kv-stream-spec-keep-pages N` sets the eject threshold directly: MTP is kept
  until the active pages exceed `N`. `N = 0` keeps MTP as long as possible
  (eject at streaming onset). It cannot extend the window past the pool
  capacity.
- `--kv-stream-spec-reenable-pages N` is the hysteresis: MTP is re-enabled once
  the active pages fall `N` pages below the threshold; a larger value makes
  re-enable later and less
  prone to flapping. Default 8.
- `--kv-stream-spec-stable-decodes N` debounces a transition until `N` consecutive
  decode batches agree. Default 4. Raise it if a mixed workload flaps.

Set `--ctx-size` to the longest prompt you need; beyond that MTP ejects and
generation returns to the baseline rate. Keep `--spec-draft-n-max` small (the
recurrent cache is `149.6 MiB * (1 + n_max)`) and give the arena as much room as
the model leaves.

## Scope and status

- Validated on an RTX 5060 Ti 16 GB with Qwen3.8-27B, a Q8_0 K cache, a Q4_0 V
  cache, one server slot (`-np 1`), and Flash Attention enabled.
- Single GPU and `llama-server` only. The phase arena requires
  `n_seq_max == 1`, Flash Attention, KV offload, and a Qwen3.5-family target.
- DFlash2 requires a draft whose vocabulary matches the target: the draft has no
  token embedding and embeds through the target's `token_embd`. A
  condensed-vocabulary target (129006 tokens) does not work with the
  full-vocabulary DFlash2 draft (248320 tokens). The results above use a
  condensed-vocabulary DFlash2 draft built by
  `gguf-py/gguf/scripts/gguf_condense_dflash.py`. A pre-built pair ships in
  [the model repo](https://huggingface.co/troed/Qwen3.8-27B-ASCII-Condensed).
- The DFlash2 draft is pinned by its weights plus the widened recurrent-state
  cache. Its five KV layers are all sliding-window (window 2048), so the draft
  KV is only about 40 MB; it stays in ordinary VRAM rather than the pinned
  region.
- ngram-map and ngram-simple currently produce zero drafts in this configuration.
- Quantizing the MTP draft KV (`-ctkd`/`-ctvd`) shrinks the pin, which grows the
  arena compute side and raises the init peak. At arena 3264 with `-ub 256` this
  can fail to allocate the MTP context's compute buffer. Use arena 3072 or
  `-ub 128` at 3264 for now.
- Research code, no upstream guarantees.

## Upstream

[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
