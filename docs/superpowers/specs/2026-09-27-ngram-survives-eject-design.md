# Keeping an ngram drafter alive across a pinned-draft eject

## Status

Approved design. Implementation to follow. Branch: `feature/ngram-drafting`,
based on `feature/kv-stream-phase-arena-spec` at `95ed6b4b2`.

## Background

The fork pins the draft model (MTP, DFlash2, DSpark) into the target phase
arena. With `--kv-stream-spec-dynamic` the server trades the draft for decode
capacity: `update_draft_dynamic()` (`tools/server/server-context.cpp:2892`)
ejects the pinned draft once the decode working set outgrows the MTP-active
capacity, and `draft_enable()` (:3018) brings it back when the working set fits
again.

`draft_eject()` (:2991) is a full spec teardown:

1. `clear_draft_target_features()` (:2980) releases the DFlash target-layer taps;
2. `spec_destroy()` (:1001) destroys the draft model, `ctx_dft` and the whole
   spec engine;
3. `spec_rewire_slots(false)` (:3001) sets `slot.spec = nullptr`, so
   `can_speculate()` is false and no slot drafts any more;
4. `llama_set_embeddings_nextn(ctx_tgt, false, false)` for MTP;
5. `llama_kv_stream_draft_set(ctx_tgt, false)` unpins the draft arena.

Because step 2 and 3 would throw away any co-configured speculator, `load_model`
disables dynamic eject as soon as a pinned draft is mixed with anything else
(`has_other_spec`, :1121-1135). The server then keeps the draft pinned for the
whole run, which forfeits the eject for the whole run. That is the only reason
the README says a mix like `draft-dflash,ngram-simple` is something you likely
do not want.

Measurements (2026-09-27, Qwen3.8-27B through the conduct router, prompt of
about 2250 tokens of fork sources plus a review instruction, 256 generated
tokens, temperature 0; drafted/accepted tokens, tokens per second):

| `--spec-type`               | source workload             | control (repeat)          |
|-----------------------------|-----------------------------|---------------------------|
| `draft-dflash`              | 504/153 (30.4%), 46.3 t/s   | -                         |
| `ngram-simple`              | 144/19 (13.2%), 32.1 t/s    | 267/237 (88.8%), 231 t/s  |
| `ngram-map-k`               | 144/19 (13.2%), 32.0 t/s    | 281/237 (84.3%), 220 t/s  |
| `ngram-mod`                 | 0/0, 31.3 t/s               | 308/223 (72.4%), 142 t/s  |
| `draft-dflash,ngram-simple` | 537/169 (31.5%), 49.6 t/s   | 287/244 (85.0%), 258 t/s  |

Two facts follow. First, an ngram drafter costs no VRAM and no pinned arena
bytes, and it does produce drafts even on non-repetitive code review (144
tokens over 3 rounds), and it is strong where the model repeats text (72 to 89
percent acceptance). Second, ngram-simple and ngram-map-k share a 12-token key,
so they behave alike; ngram-mod needs a 24-token match and can never beat them.
The upstream side of those ngram behaviours (including the `ngram-mod` key size
and the `common_ngram_map` cross-request state bug) is tracked separately and
is out of scope here.

## Problem

Ejecting the pinned draft should leave the cheap drafters running. Two things
block that today:

1. the engine is destroyed and `slot.spec` is nulled, so nothing drafts after an
   eject, and the guard in `load_model` therefore disabled the eject for mixes;
2. more subtly, the unpin removes the target's rollback capability. The toggle
   sets `cparams.n_rs_seq = 0` (`src/llama-context.cpp:1577`), and the recurrent
   memory refuses any bounded rollback when `n_rs_seq == 0`
   (`src/llama-memory-recurrent.cpp:250`). Verifying a draft needs exactly that
   rollback.

## Design

### Contract

Explicit fallback, controller keeps toggling. ngram drafts after an eject only
if the user listed an ngram type in `--spec-type`. Before the eject the chain
behaves as today (ngram first, pinned draft fills the rest). After the eject
only the surviving implementations draft. The controller keeps re-enabling the
pinned draft whenever the working set fits and ejecting it again when it grows.

### Component 1: type filter

Add one server-side predicate for the implementations that need no draft
context, in `tools/server/server-context.cpp`:

```cpp
static bool spec_type_survives_eject(common_speculative_type type);
```

It must be a whitelist of the ngram family: `NGRAM_SIMPLE`, `NGRAM_MAP_K`,
`NGRAM_MAP_K4V`, `NGRAM_MOD`, `NGRAM_CACHE`. A blacklist of the pinned family is
not enough: `DRAFT_SIMPLE` also throws without a draft context
(`common/speculative.cpp:193`), and the remaining `DRAFT_*` types are gated on
`params.draft.ctx_dft` in `common_speculative_init` (:2691-2694) but are not
guaranteed to stay that way.

### Component 2: spec_create gains a no-draft mode

```cpp
bool spec_create(llama_progress_callback progress_cb = nullptr,
                 void * progress_ud = nullptr,
                 bool with_draft_model = true);
```

- `with_draft_model = true` is the existing path, unchanged (draft model load,
  `ctx_dft`, pinned buft override, `common_speculative_init_from_params`).
- `with_draft_model = false`: skip `common_speculative_init_from_params`
  entirely, set `model_dft = ctx_dft = nullptr` and
  `params_base.speculative.draft.ctx_dft = nullptr`, then build the engine from
  a copy of `params_base.speculative` whose `types` are filtered by
  `spec_type_survives_eject`. `common_speculative_config` stores its params by
  value (`common/speculative.cpp:59`), so a local copy is safe. The existing
  `ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO` gate stays.
- In no-draft mode the return value means "no error", not "an engine exists":
  an empty filtered list makes `common_speculative_init` return null, which is
  the pinned-draft-only case and must leave the server in today's post-eject
  state. Only `with_draft_model` keeps the `ctx_dft != nullptr` requirement.
- `params_base.speculative.types` is never mutated; the filter is applied to a
  local copy only, so `--spec-type` keeps its meaning for every later rebuild.

### Component 3: eject keeps the survivors

`draft_eject()` keeps its order and gains a rebuild step after the unpin:

1. `clear_draft_target_features()`, `spec_destroy()`, existing;
2. `spec_rewire_slots(false)` before the toggle, existing (slots must not
   reference the freed engine across the toggle);
3. MTP nextn off, existing;
4. `llama_kv_stream_draft_set(ctx_tgt, false)`, existing, and return false if
   the toggle fails;
5. new: `spec_create(nullptr, nullptr, /*with_draft_model=*/false)`;
6. new: if `llama_n_rs_seq(ctx_tgt) == 0`, set
   `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL` (see Component 4);
7. new: if the rebuild produced an engine, `spec_rewire_slots(true)` and then
   `common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens())`
   for every slot in `SLOT_STATE_GENERATING`, so `ngram-mod`'s hash pool and
   `ngram-map`'s index are seeded from the existing history instead of warming
   up from scratch;
8. `mtp_ejected = true` and the existing log line.

`draft_enable()` is unchanged: it re-pins, calls `spec_create()` in draft mode
and `spec_rewire_slots(true)`. The rebuilt engine replaces the survivor engine,
which is the intended "pinned draft takes over again" behaviour.

### Component 4: rollback after the eject

The verify path removes the rejected draft suffix from the target. Partial
acceptance either restores a checkpoint
(`tools/server/server-context.cpp:4215-4234`, followed by a suffix trim at
:4229) or, without a checkpoint, relies on the target's bounded rollback
through the trim at :4276. The latter needs `n_rs_seq > 0`, which the unpin
takes away, so after an eject the checkpoint path is the one that must carry
the feature.

The server already derives this from two values:
`use_ckpt_tgt = type == FULL || (type == RS && n_rollback > llama_n_rs_seq(ctx))`.
With the stale `RS` type and `n_rs_seq == 0` every rollback already takes the
checkpoint path, so the feature works by accident; Component 3 step 6 makes it
explicit. Checkpoints use `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, which skips the
attention cache (`src/llama-memory-hybrid.cpp:200`) and saves the recurrent
state only; the rejected attention tokens are trimmed by the existing
`mem.seq_rm(pos_max + 1, -1)`, a no-op for the recurrent part after a restore.

The cost of one recurrent-state save per draft round is the open question of
this design; it is measured in Testing, and the fallback (keep a small `rs`
window pinned for the survivor) is deliberately out of scope until the numbers
say otherwise.

### Component 5: allow mixing again

In `load_model` (:1121-1135) drop the `has_other_spec` clause and the lambda
that computes it, keep the clause that rejects two pinned families
(`spec_mtp && (spec_dflash || spec_dspark)`). `--spec-type
draft-dflash,ngram-simple` then keeps dynamic eject enabled, and the warning
about disabled dynamic ejection disappears for mixes that only add survivors.

### Component 6: documentation

README: rewrite the eject paragraph so it says the mix keeps the eject and that
ngram keeps drafting after it, update the results tables with the measured
post-eject numbers, and drop the sentence claiming ngram-map and ngram-simple
produce zero drafts. The upstream side of that sentence is tracked separately.

## Ordering summary

Eject: teardown + unpin (existing) -> rebuild from the surviving types ->
removal type -> rewire slots to the new engine -> re-seed the survivors from the
current history -> log.

Enable: re-pin -> full rebuild -> rewire slots (existing).

## Failure handling

- Rebuild throws or returns null: log, leave `spec` null, keep the eject
  successful. `can_speculate()` is then false, exactly today's post-eject
  state, and the controller can still re-enable later.
- Unpin fails: return false before any rebuild, as today.
- `--spec-type` with no survivor: no engine is built; behaviour is unchanged.
- Rebuild allocation happens inside the decode loop. The ngram engines are
  host-only: `ngram-mod` allocates its pool, `ngram-cache` may load cache files,
  `begin()` re-seeds from the history. This happens once per state change (the
  controller has hysteresis), so the design accepts it; the eject latency is
  measured in Testing.

## Testing

All runs use the existing probe harness (`ngram-probe/probe.py` plus
`ngram-probe/set-preset.py`) driven through the host actions
(`llama-build`, `llama-restart`, `llama-log`). The `llama-build` step needs the
CUDA FA kernel recompile once, which exceeds the 300 s action cap, so that first
build is run by hand.

1. Force an early eject (low `--kv-stream-spec-keep-pages`) with
   `draft-dflash,ngram-simple` and read the journal: after `draft ejected` the
   ngram statistics keep accumulating and `draft_n > 0` for requests that start
   after the eject.
2. Same configuration, mid-request eject: generation continues, the journal
   shows no abort, and the output stays coherent.
3. Re-enable: after the working set fits again, `draft re-enabled` appears and
   the pinned draft takes over the draft rounds again without breaking
   accounting.
4. Cost: at long context compare the eject arm with `--spec-type none` and with
   `draft-dflash` pinned; report decode and prefill tokens per second, plus the
   checkpoint share from the journal.
5. Regression: `--spec-type draft-dflash` alone and ngram-only legs must match
   the numbers in Background.

## Milestones

1. Filter predicate and `spec_create` no-draft mode; exercised by forcing an
   eject with a low keep-pages value.
2. Eject rebuild, removal type, slot re-arm.
3. Guard removal and README.
4. Measurements and, if the checkpoint cost dominates, a follow-up decision on
   keeping a small `rs` pin.

## Risks

- Checkpoint cost after the eject: one recurrent-state save per draft round.
  This is the make-or-break number for the feature.
- Re-seeding the survivors mid-generation: `common_speculative_begin` resets
  ngram state; the rewire clears `spec_draft`, `spec_i_batch` and `spec_ckpt`,
  so no index accounting must survive the rebuild. Verify with the mid-request
  eject test.
- Eject latency: host-side allocation and history ingest inside the decode loop.
- The prompt cache is cleared by `spec_rewire_slots`, as it already is on every
  eject today.
- `ngram-cache` loads files at rebuild time; if that proves slow, it can be
  excluded from the survivor set.

## Out of scope

- The upstream ngram behaviours: `ngram-mod`'s 24-token key, the
  `common_ngram_map` cross-request state bug, and the README claim about them.
- Keeping a small `rs` window pinned after the eject (the fallback if
  checkpoints are too expensive).
- Chain priority changes while the pinned draft is active.
- Non-ngram speculators that need a draft context, and side-by-side pinned
  families (MTP plus DFlash).
