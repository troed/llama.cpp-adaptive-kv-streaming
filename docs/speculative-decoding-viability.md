# Speculative decoding viability investigation

These are the experiments that motivated the current draft setup. The
recommended configuration lives in the
[main README](../README.md#tldr---recommended-setup); this page keeps the
draft-vs-upstream comparisons and the MTP tuning notes that informed it.

## Speculative decoding

Both MTP and DFlash2 drafts work with the phase arena, on the same pinned-draft
machinery: the draft weights (and, for MTP, the draft KV) are reserved in the
target arena, and the dynamic eject trades the draft for decode capacity as the
working set grows. Only one pinned draft family can be active at a time.

A pinned draft can be mixed with an ngram speculator (for example
`draft-dflash,ngram-simple`). The pinned draft is still ejected when the working
set grows, but the ngram speculator keeps drafting through the eject: it needs
no draft model and no pinned arena bytes, so decode keeps a small draft win at
long context instead of losing speculation entirely. The ngram speculator is
first in the priority chain and the pinned draft fills the remaining draft
rounds.

| draft | `--spec-type` | draft source | notes |
|---|---|---|---|
| MTP | `draft-mtp` | the target's own embedded MTP block, with no `-md` (the TLDR model; see [MTP uses the target's embedded block](../README.md#mtp-uses-the-targets-embedded-block)) | the MTP block weights and the nextn KV are pinned in the target arena |
| DFlash2 | `draft-dflash` | [the condensed DFlash2 draft](https://huggingface.co/troed/Qwen3.8-27B-ASCII-Condensed) | the draft has no token embedding and embeds through the target's `token_embd`, so its vocabulary must match the target's; its five KV layers are all sliding-window (window 2048) and stay in ordinary VRAM, so the pin is only the weights plus the widened recurrent-state cache |

The ngram speculators draft from the model's own token history, so they pay off
when the output repeats text (quoting, copying, structured edits) and stay close
to idle on fresh prose. `ngram-mod` needs a 24-token match and drafts least;
`ngram-simple` and `ngram-map-k` share a 12-token key.

## Results

Measured on an RTX 5060 Ti 16 GB with Q8_0 K cache, Q4_0 V cache, `-ngl 99`,
`--flash-attn on`, `--parallel 1`, `--ctx-size 160000`, a source-tree prompt
followed by a review instruction, and 256 tokens generated at temperature 0.
The model is [bsaleh03's ASCII condensed version of Unsloth UD-IQ4_XS](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed),
not the TLDR one: the runs compare `--spec-type none` against MTP and DFlash2,
each at the largest `--kv-stream-arena-mib` that decodes on the 16 GB card
(3072 MiB upstream, 3136 MiB MTP, 3200 MiB DFlash2). The draft KV is
quantized (`-ctkd q8_0 -ctvd q4_0`).

![MTP and DFlash2 decode throughput vs context](../media/draft-thresholds-decode.png)

![MTP and DFlash2 prefill throughput vs context](../media/draft-thresholds-prefill.png)

Findings (eject = the default controller, keep = a high `--kv-stream-spec-keep-pages`):

- The draft is never ejected while the working set fits, so eject and keep are
  identical up to the streaming onset: about 49K tokens for MTP, 57K for DFlash2.
- At the onset the default controller ejects, which is earlier than the data
  supports: keeping is 1.8 to 2.1x faster there, and keeping wins through about
  81K. The keep/eject crossover is about 85K tokens for both drafts; the keep
  arm is noisy, so a robust eject threshold is about 300 pages/layer.
- After ejection the decode rate matches upstream (49K: 22.2 vs 22.2; 98K: 18.2
  vs 18.1 t/s), so the draft is cleanly disabled.
- DFlash2 trails MTP at short context (its draft is five layers, not one) but
  its eject curve stays ahead at long context (160K: 13.1 vs 11.8 t/s).

Both drafts would gain from moving the default eject point from streaming onset
(about 180 to 224 pages/layer) to about 300 pages/layer: that recovers the 1.8
to 2.1x decode advantage across the 49K to 80K band and still ejects before
keep turns negative.

Full numbers: [benchmarks/results/draft-thresholds.csv](../benchmarks/results/draft-thresholds.csv).
Reproduce with `benchmarks/benchmark_mtp_streaming.py` (MTP and DFlash2),
`benchmarks/benchmark_upstream_vs_mtp.py` (upstream), and
`benchmarks/plot_draft_thresholds.py` (combined table and figure).

### Drafting after the eject

With `spec-type = draft-dflash,ngram-simple` the pinned draft is ejected at the
same working-set threshold as above, and the ngram speculator keeps drafting.
Measured at the same prompt and 256 generated tokens, the pinned-only arm stops
drafting after the eject while the mix arm keeps a nonzero draft count, and
decode runs at 20.4 t/s against 19.4 t/s with `spec-type = none`.

### MTP KV quantization

The MTP draft KV defaults to F16 and does not inherit the target `-ctk`/`-ctv`;
pass `-ctkd`/`-ctvd` to quantize it. The automatic pin sizes the pinned MTP KV
from the draft types, so quantizing shrinks the pin and grows the decode window
(arena 3072, ctx 160000, auto pin):

| MTP KV type | MTP KV pin | decode window |
|---|---|---|
| F16 | 164 pages / 41984 tokens | 164 pages |
| q8_0 K / q4_0 V | 178 pages / 45568 tokens | 178 pages |
| q4_0 K / q4_0 V | 182 pages / 46592 tokens | 181 pages |

The window grows by 14 to 18 pages, which moves the MTP crossover from about
39K to about 42K tokens; prefill loses 1.5 to 4 percent versus F16. Details in
[next-steps/01-mtp-kv-quantization.md](next-steps/01-mtp-kv-quantization.md).
