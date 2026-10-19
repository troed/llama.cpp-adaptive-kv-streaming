# DFlash2 with an attached ngram drafter

Status: implemented and validated on the host, 2026-10-19. Ready for review.

## 1. Problem

`--spec-type ngram-simple,draft-dflash` (or any `ngram-*` alongside `draft-dflash`) was
rejected at argument parse by `common_streamed_dflash_config_error`, whose admission allowed
only `draft-dflash` in addition to `none`. Attached MTP already shares a target with the
ngram self-speculators, so the goal was the same cascade for DFlash2: the ngram drafter
proposes when it has a match, the DFlash2 sidecar otherwise, on one streamed target.

Admission alone is not enough. With the gate relaxed the config parses and the target logs
`streamed verify width 49` (ngram-simple's default `size_m` 48), but the process aborts at
load in `ggml_gallocr_reserve_n_size`, from the target's `sched_reserve()` invoked by the
DFlash2 draft constructor. The log shows
`borrowed buffer 1 requires 4295168 bytes in 2 chunks; range ends at 21266688` immediately
before the abort, and the baseline DFlash2 target (`streamed verify width 5`) loads on the
same binary.

## 2. Root cause

DFlash2's draft context borrows the target's serial arena: it reserves a bounded draft KV
cache and recurrent rollback staging from the target's fixed arena, then the target is
re-reserved (`keep_parent=1`) with those regions fixed. The widened ngram verify inflates the
target's reserved verify graph (49 rows vs the sidecar's 5), and the retained borrowed
workspace cannot host it. The failure is a layout/range failure, not device memory: shrinking
the arena by 64 MiB does not move the reported range.

Two simpler fixes were measured and rejected:

- Reserve the wide graph but let the serial draft borrow elsewhere: the draft's own params are
  identical with and without ngram, so the delta is purely the target's wider verify phase;
  moving the graph is the shared arena contract change (out of scope here).
- Cap only the target's verify reservation: loads, but the first real wide verify then fails
  at the decode phase transition (`failed to reserve graph after text phase transition`)
  because the retained workspace really is too small.

Measured load boundary on the qualified preset (arena 3728 MiB): verify widths 8, 16, 32 and
40 load and run; width 48 aborts.

## 3. Design

Two changes:

1. `common/common.cpp` - `common_streamed_dflash_config_error` admits the five `ngram-*`
   types alongside `draft-dflash`, mirroring the attached-MTP switch. A second neural drafter
   (`draft-mtp`, `draft-eagle3`, `draft-dspark`), auxiliary MTP layers, and `mmproj` stay
   rejected. Unchanged: serial slot, arena budget, `--fit off`, 1-7 DFlash2 draft tokens,
   text only, `-fa on`, microbatch covers the sidecar block.

2. `common/speculative.cpp` - when a `draft-dflash` implementation is present, cap the
   effective speculative depth at `COMMON_SPECULATIVE_DFLASH_NGRAM_MAX = 7` (one less than
   the 8-row streamed span width):
   - `common_speculative_n_max(const common_params_speculative *)` caps the value used to
     derive `cparams.kv_stream_verify_width` and the server output limits;
   - `common_speculative_n_max(const common_speculative *)` caps the impl-side aggregate;
   - `common_speculative_draft()` truncates any draft (ngram or sidecar) to the capped depth.

   The resulting streamed verify width is 8, so the target reserves only the span-path graph
   and the wide serial reserve that broke the DFlash2 arena borrow never materialises. The
   DFlash2 sidecar's own block (`n_max + 1`, default 5) is below the cap and is unchanged.

## 4. Why a cap rather than full ngram depth

The streamed span path serves at most 8 query rows. A wider ngram verify either reserves a
wider graph (which the DFlash2 serial arena cannot host, per section 2) or falls back to the
gather path (the memory penalty this fork exists to remove). Capping the attached ngram
drafter keeps the combination on the fast span path: no gather, no extra KV-pool cost, and a
verify width that is arena-independent. The trade-off is that ngram matches longer than 7
tokens are not used with DFlash2. MTP+ngram is unaffected and keeps its configured depth.

## 5. Verification

- CPU suite (host, `llama-test`): 9/9 pass, including the new
  `test-arg-parser` assertions for ngram admission and the depth cap (RED first: the cap
  assertion failed with the cap disabled).
- Host, RTX 5060 Ti, the live Qwen3.8-27B DFlash2 preset, `ngram-simple` at its default
  `size_m` 48:
  - loads, `streamed verify width 8`, both `ngram-simple` and `draft-dflash` initialise;
  - a repetitive-prompt decode accepted 73/73 drafted tokens over 96 generated tokens;
  - `memory_phase: phase=decode` reports `attention=7422592` (the span-tile figure) and
    `kv_pool=3831468672`, i.e. no gather reservation and no KV-pool loss vs DFlash2-only;
  - width sweep: 8/16/32/40 load, 48 aborts (the boundary this cap avoids).

## 6. Frozen sweep

One `logsweep.py` run, frozen corpus, 4096 decode tokens, temperature 0. Baseline is the
README DFlash2 column (mean of three runs).

| context | DFlash2 TG | DFlash2+ngram TG | DFlash2 PP | DFlash2+ngram PP |
| --- | --- | --- | --- | --- |
| 40k | 46.83 | 49.16 | 787.6 | 786.3 |
| 160k | 34.30 | 34.90 | 463.6 | 490.1 |

Accepted 2681/3278 at 40k (mean len 3.55) and 2596/3352 at 160k (mean len 3.29). No
regression; the decode gain is small because the cap keeps the verify on the span path and
the frozen corpus's ngram matches are short. Raw: `.superpowers/dflash2-ngram-sweep/r1/`.

## 7. Non-goals

- Changing MTP+ngram, DSpark, the vision arena, or the shared serial-arena contract.
- Graph-level splitting of a wide verify (the two-round protocol).
- Full-depth ngram matches on DFlash2.

## 8. Open items

- Re-run the sweep three times if the small delta needs tighter error bars.
- Revisit the cap if the DFlash2 serial arena learns to host a wider reserved verify without
  the gather penalty.

