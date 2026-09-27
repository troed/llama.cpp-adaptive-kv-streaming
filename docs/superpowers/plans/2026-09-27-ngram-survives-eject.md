# Keeping ngram drafting alive across a pinned-draft eject: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the pinned draft (DFlash2 / MTP) is ejected, keep drafting through the configured ngram speculator, and let a pinned + ngram mix keep dynamic eject enabled.

**Architecture:** The server stops treating a pinned + ngram mix as eject-hostile. At eject time it rebuilds the speculative engine from a filtered type list (only the implementations that need no draft context) instead of leaving it null, re-arms the generating slots with `common_speculative_begin()`, and makes the target removal type explicit so the checkpoint-based verify path carries the survivor. `common/` and `src/` are not modified.

**Tech Stack:** C++17 (llama.cpp server), CUDA build, Python probe harness, systemd user unit `llama` driven through umwelt host actions.

**Spec:** `docs/superpowers/specs/2026-09-27-ngram-survives-eject-design.md`

## Global Constraints

- Build only, never reconfigure: the CUDA FA kernels are already compiled with `GGML_CUDA_FA_QUANTS=q8_0-q4_0`. Build with host action `llama-build` (300 s cap).
- All work happens in this checkout; the running service execs this tree's `build/bin/llama-server` via the launcher's `router.sh`.
- The test surface is the host service plus `ngram-probe/probe.py` and `ngram-probe/set-preset.py`. Do not add files under `tests/`.
- The eject fires on the controller in `update_draft_dynamic()`: with `kv-stream-spec-keep-pages > 0` the limit is `max(mtp_capacity_pages, keep_pages)` pages per layer. A low `--keep-pages` therefore does not force an early eject by itself; the prompt working set must cross the resident capacity. The long-prompt legs below spend roughly 80 to 130 s in prefill each, so a probe looks idle for a while before it prints.
- Journal reading, used by every verification step: call host action `llama-log`; when its output is truncated in-line, the full text is in the harness spill directory (`jq -r '.stdout' <spill-file>`). Save it to `$LOG`, then keep only the newest server run, because the raw window is just the last 400 lines and still holds earlier legs. Every line of a run carries that run's child tag, for example `[56819]`, and the tag changes on each restart, so:
  ```bash
  mkdir -p /tmp/tools
  LOG=/tmp/tools/visit.log
  TAG=$(grep -oE '\[[0-9]+\] [0-9]+\.[0-9]+\.[0-9]+' "$LOG" | tail -1 | grep -oE '\[[0-9]+\]')
  grep -F "$TAG" "$LOG" > /tmp/tools/visit-tail.log
  ```
  Grep `/tmp/tools/visit-tail.log` in the steps below. This matters because the eject guard warning is logged *before* the `loading model` line, so cutting at that line would hide it.
- After every test round: `python3 ngram-probe/set-preset.py --restore`, restart, and confirm the preset matches `ngram-probe/models-preset.ini.orig` with `cmp`.
- ASCII only, in code, comments and docs: no em dash, no unicode arrows.
- Do not commit or push on the contributor's behalf without an explicit approval for that action; the commit steps below are marked accordingly and use `Assisted-by: <assistant name>` when approved.
- The spec's Background numbers were measured before the FA quant fix landed, so reference timings shift. Re-measure them, do not compare against the spec table.
- `common/` and `src/` stay untouched. All code edits are in `tools/server/server-context.cpp` plus the probe tooling.

## Review Focus

- A pinned-only config (`--spec-type draft-dflash`, no ngram) must still eject and then simply have no drafting; nothing may warn, crash, or keep a half-built engine.
- The survivor list can be empty, so the rebuild may leave `spec` null; every later use must go through `can_speculate()` and must not assume an engine exists.
- An eject during prompt processing (not generation) must still produce a usable survivor: the controller ejects on `streaming || over_limit` regardless of phase, and the slot is only re-armed later at the prompt to generating transition.
- Partial acceptance immediately after an eject runs without `n_rs_seq` rollback, so it must take the checkpoint path; a failed `seq_rm` aborts the server.
- Repeated eject and re-enable cycles rebuild the survivor each time; the old engine must be freed and slots must never keep a stale `slot.spec`.
- A context that reports `COMMON_CONTEXT_SEQ_RM_TYPE_NO` cannot be exercised on this model; that path is covered by code review only.

---

### Task 1: Mixing keeps dynamic eject

**Files:**
- Modify: `tools/server/server-context.cpp:1104-1136` (`load_model`)
- Modify: `ngram-probe/set-preset.py`
- Test: host service journal plus `ngram-probe/probe.py` (no `tests/` additions)

**Interfaces:**
- Consumes: `params_base.speculative.types`, `params_base.speculative.kv_stream_spec_dynamic`
- Produces: dynamic eject stays enabled for a pinned + ngram mix; the preset tool learns `--keep-pages N` and `--arena-mib N`; the calibrated shell variable `BUDGET` used by every long-prompt leg.

- [ ] **Step 1: Teach the preset tool the two knobs**

In `ngram-probe/set-preset.py`, change the transform signature:

```python
def transform(text, spec_type, n_max, draft, keep_pages, arena_mib):
```

Add the two branches next to the existing `spec-type` branch:

```python
        if key == "kv-stream-spec-keep-pages" and keep_pages is not None:
            if not commented:
                out.append(f"kv-stream-spec-keep-pages = {keep_pages}")
            continue
        if key == "kv-stream-arena-mib" and arena_mib is not None:
            if not commented:
                out.append(f"kv-stream-arena-mib = {arena_mib}")
            continue
```

Add the arguments and pass them through in `main()`:

```python
    ap.add_argument("--keep-pages", type=int, default=None)
    ap.add_argument("--arena-mib", type=int, default=None)
```

```python
        text = transform(orig, args.spec_type, args.n_max, args.draft == "on",
                         args.keep_pages, args.arena_mib)
        action = (f"spec-type={args.spec_type} n-max={args.n_max} draft={args.draft} "
                  f"keep-pages={args.keep_pages} arena-mib={args.arena_mib}")
```

- [ ] **Step 2: Calibrate the eject onset for this arena**

The preset keeps `kv-stream-arena-mib = 4352` and `kv-stream-spec-keep-pages = 360` (about 92K tokens). Hold the limit down to the resident capacity with `--keep-pages 1`, then find the prompt size that ejects:

```bash
cd <repo>
python3 ngram-probe/set-preset.py draft-dflash,ngram-simple --n-max 5 --draft on --keep-pages 1
```

Call host action `llama-restart`, wait for the router, then probe at the first candidate budget:

```bash
for i in $(seq 1 60); do curl -s --max-time 2 --proxy "$PROBE_PROXY" "http://$PROBE_HOST/v1/models" >/dev/null && break; sleep 2; done
BUDGET=180000
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 64 --timeout 900 --dump ngram-probe/cal-180k.json
```

Call `llama-log`, prepare `/tmp/tools/visit-tail.log` as in Global Constraints, and check `grep -c "draft ejected" /tmp/tools/visit-tail.log`. If it is 0, restart and repeat with `BUDGET=240000`, then `BUDGET=300000`. If none of the three ejects, re-set the preset with `--arena-mib 3200` (the README's DFlash2 floor), restart, and repeat from 240000.

Expected: an eject at one of the three budgets, between 180000 and 300000 characters (about 51K to 84K prompt tokens). Note the chosen `BUDGET`; every long-prompt leg below sets it to that value.

- [ ] **Step 3: Observe the current behaviour (this is the failing test)**

Preset unchanged (`draft-dflash,ngram-simple --keep-pages 1`); restart via `llama-restart`, wait for the router, then:

```bash
BUDGET=<the value calibrated in Step 2>
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task1-before-1.json
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task1-before-2.json
```

Call `llama-log`, prepare the tail, and check:

```bash
grep -c "disabling"        /tmp/tools/visit-tail.log   # expect at least 1
grep -c "draft ejected"    /tmp/tools/visit-tail.log   # expect 0
```

Expected: no eject at all, because the guard disabled dynamic eject for the whole run; both probes report a nonzero `draft_n` (DFlash2 never left).

- [ ] **Step 4: Remove the mixed-config clause**

In `tools/server/server-context.cpp`, delete the `has_other_spec` lambda and the clause that uses it, and adjust the warning text, so the block reads:

```cpp
        // only one pinned draft family at a time; mixing MTP with DFlash would
        // reserve the wrong weights
        if (params_base.speculative.kv_stream_spec_dynamic && pin_draft &&
                (spec_mtp && (spec_dflash || spec_dspark))) {
            SRV_WRN("%s", "dynamic draft ejection requires a single pinned draft family, disabling\n");
            spec_draft_enabled_dynamic = false;
        }
```

- [ ] **Step 5: Build**

Call host action `llama-build`. Expected: `exit_code 0`, incremental only (no CUDA kernel recompile).

- [ ] **Step 6: Verify the eject now fires for the mix**

Restart via `llama-restart`, wait for the router, repeat the two probes into `ngram-probe/task1-after-1.json` and `ngram-probe/task1-after-2.json`, then call `llama-log`, prepare the tail, and check:

```bash
grep -c "disabling"        /tmp/tools/visit-tail.log   # expect 0
grep -c "draft ejected"    /tmp/tools/visit-tail.log   # expect at least 1
```

Expected: the eject happens, during the first request's prefill. The second probe is expected to report `draft_n=0` and to have no `spec common_specu: statistics ngram-simple:` line, because the engine is still destroyed on eject. That gap is what Task 2 closes; record the probe outputs and journal lines as Task 1 evidence.

- [ ] **Step 7: Commit (needs human approval)**

```bash
git add tools/server/server-context.cpp ngram-probe/set-preset.py
git commit -m "spec : allow dynamic draft eject for pinned + ngram mixes"
```

---

### Task 2: Survivor engine across the eject

**Files:**
- Modify: `tools/server/server-context.cpp` (`spec_create` at :947, `draft_eject` at :2991)
- Test: host service journal plus `ngram-probe/probe.py`

**Interfaces:**
- Consumes: `spec_create(progress_cb, progress_ud, with_draft_model)` call sites in `load_model` (:1236) and `draft_enable` (:3038); `spec_rewire_slots(bool)`; `common_speculative_init`, `common_speculative_begin`, `llama_n_rs_seq`; `BUDGET` from Task 1
- Produces: `static bool spec_type_survives_eject(common_speculative_type type)`; `bool spec_create(llama_progress_callback progress_cb = nullptr, void * progress_ud = nullptr, bool with_draft_model = true)`; `draft_eject()` that leaves a drafting engine behind when a survivor type is configured.

- [ ] **Step 1: Observe the gap this task closes**

State left by Task 1. Preset at `draft-dflash,ngram-simple --keep-pages 1`; restart via `llama-restart`, wait for the router, then:

```bash
BUDGET=<the value calibrated in Step 2 of Task 1>
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task2-before-1.json
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task2-before-2.json
```

Call `llama-log`, prepare the tail, and confirm:

```bash
grep -c "draft ejected"    /tmp/tools/visit-tail.log   # expect at least 1
grep -c "ngram-simple:"    /tmp/tools/visit-tail.log   # expect 0
```

Expected: the second request reports `draft_n=0` and no `statistics ngram-simple:` line exists. Keep `task2-before-2.json`; its first 200 response characters are the losslessness reference for Step 6.

- [ ] **Step 2: Add the survivor predicate**

Insert directly above `spec_create()` in `tools/server/server-context.cpp`:

```cpp
    // implementations that keep drafting after the pinned draft is ejected:
    // they need no draft context, only host memory
    static bool spec_type_survives_eject(common_speculative_type type) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                return true;
            default:
                return false;
        }
    }
```

- [ ] **Step 3: Give spec_create a no-draft mode**

Change the signature and wrap the draft-model block, keeping the rest of the body otherwise intact:

```cpp
    bool spec_create(llama_progress_callback progress_cb = nullptr, void * progress_ud = nullptr, bool with_draft_model = true) {
        const bool has_draft = params_base.speculative.has_dft();

        if (with_draft_model) {
            common_params params_dft = common_base_params_to_speculative(params_base);

            params_dft.load_progress_callback           = progress_cb;
            params_dft.load_progress_callback_user_data = progress_ud;

            spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
            model_dft = spec_init->model();
            ctx_dft   = spec_init->context();

            if (has_draft && model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft;
        } else {
            // eject path: no draft model, no ctx_dft; keep only the survivors
            model_dft = nullptr;
            ctx_dft   = nullptr;
            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = nullptr;
        }
```

Then replace the engine creation block with a branch that filters a local copy:

```cpp
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                if (with_draft_model) {
                    spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
                } else {
                    // the filtered list is local: --spec-type must stay intact
                    // for the re-enable path
                    common_params_speculative params_spec = params_base.speculative;
                    params_spec.types.clear();
                    for (const auto type : params_base.speculative.types) {
                        if (spec_type_survives_eject(type)) {
                            params_spec.types.push_back(type);
                        }
                    }
                    SRV_INF("keeping %s for the ejected draft\n", common_speculative_type_name_str(params_spec.types).c_str());
                    spec.reset(common_speculative_init(params_spec, params_base.n_parallel));
                }
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }
```

The trailing `if (spec == nullptr)` block stays as it is: with an empty survivor list it leaves `spec`, `spec_init`, `ctx_dft` and `model_dft` empty, which is today's post-eject state.

- [ ] **Step 4: Rebuild the survivors at eject**

In `draft_eject()`, after the arena toggle and before `mtp_ejected = true;`, insert:

```cpp
        // keep the implementations that need no draft context, so decoding
        // keeps drafting while the draft is out
        try {
            spec_create(nullptr, nullptr, /*with_draft_model=*/false);
        } catch (const std::exception & e) {
            SRV_ERR("failed to rebuild the speculative context after the eject: %s\n", e.what());
            spec_destroy();
        }
        if (llama_n_rs_seq(ctx_tgt) == 0) {
            // the unpin removed the bounded rollback; the draft verify path
            // falls back to target checkpoints
            ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        }
        if (spec) {
            spec_rewire_slots(true);
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_GENERATING) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            }
        }
```

An eject during prompt processing skips the loop above, because the slot is not generating yet; the normal prompt to generating transition calls `common_speculative_begin` at :4118 and seeds the survivors there. Both cases are exercised below.

- [ ] **Step 5: Build**

Call host action `llama-build`. Expected: `exit_code 0`.

- [ ] **Step 6: Verify drafting continues after the eject**

Preset unchanged from Step 1; restart via `llama-restart`, wait for the router, then:

```bash
BUDGET=<the value calibrated in Step 2 of Task 1>
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task2-after-1.json
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task2-after-2.json
```

Call `llama-log`, prepare the tail, and check all of:

```bash
grep -c "keeping ngram"        /tmp/tools/visit-tail.log   # expect at least 1
grep -c "draft ejected"        /tmp/tools/visit-tail.log   # expect at least 1
grep     "ngram-simple:"       /tmp/tools/visit-tail.log   # expect a line with #gen drafts > 0
grep -c "failed to remove"     /tmp/tools/visit-tail.log   # expect 0
```

Expected probe facts: `task2-after-2.json` has `draft_n > 0` and `draft_n_accepted > 0`. The survivor is not capped by `spec-draft-n-max = 5`: the journal's `#gen tokens / #gen drafts` for ngram-simple must be greater than 5.

Losslessness: the survivor run must produce the same greedy text as the pre-change run.

```bash
python3 - <<'PY'
import json
a = json.load(open("ngram-probe/task2-before-2.json"))["choices"][0]["message"]["content"]
b = json.load(open("ngram-probe/task2-after-2.json"))["choices"][0]["message"]["content"]
print("identical first 200 chars:", a[:200] == b[:200])
PY
```

Expected: `True`. A mismatch means the survivor path is not lossless and must be investigated before continuing.

- [ ] **Step 7: Verify the pinned-only config is unchanged**

```bash
python3 ngram-probe/set-preset.py draft-dflash --n-max 5 --draft on --keep-pages 1
```

Restart via `llama-restart`, wait for the router, then:

```bash
BUDGET=<the value calibrated in Step 2 of Task 1>
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task2-pinned-only.json
```

Call `llama-log`, prepare the tail, and check:

```bash
grep -c "keeping ngram"        /tmp/tools/visit-tail.log   # expect 0: nothing survives this config
grep -c "draft ejected"        /tmp/tools/visit-tail.log   # expect at least 1
grep -c "failed to"            /tmp/tools/visit-tail.log   # expect 0
```

Expected: the eject happens, no survivor is built, and a post-eject request behaves exactly like the pre-change server.

- [ ] **Step 8: Commit (needs human approval)**

```bash
git add tools/server/server-context.cpp
git commit -m "spec : keep the ngram speculator drafting after a draft eject"
```

---

### Task 3: Documentation, re-enable, and measurements

**Files:**
- Modify: `README.md:78-96` (Speculative decoding section)
- Test: host service journal plus `ngram-probe/probe.py`

**Interfaces:**
- Consumes: the behaviour from Tasks 1 and 2; `BUDGET` from Task 1
- Produces: an up-to-date README section plus the measured post-eject numbers

- [ ] **Step 1: Verify re-enable still works with a survivor**

Re-enable fires when the working set fits again, so follow a long request with a short one. Preset at `draft-dflash,ngram-simple --keep-pages 1`; restart via `llama-restart`, wait for the router, then:

```bash
BUDGET=<the value calibrated in Step 2 of Task 1>
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task3-cycle-long.json
python3 ngram-probe/probe.py --kind source --budget 8000 --max-tokens 16 --dump ngram-probe/task3-cycle-short.json
```

Call `llama-log`, prepare the tail, and check:

```bash
grep -c "draft ejected"        /tmp/tools/visit-tail.log   # expect at least 1
grep -c "draft re-enabled"     /tmp/tools/visit-tail.log   # expect at least 1
grep     "draft-dflash:"       /tmp/tools/visit-tail.log   # expect statistics after the re-enable
```

Expected: an eject, then a re-enable on the short request, then DFlash2 drafting again, with no abort. Run the long probe once more afterwards to prove a second eject cycle works.

- [ ] **Step 2: Measure the post-eject cost at the user's own thresholds**

Use the preset as the user runs it (keep-pages 360) with the same long prompt. Run three arms, restarting between them, and record the timings from each dump:

```bash
BUDGET=<the value calibrated in Step 2 of Task 1>
python3 ngram-probe/set-preset.py draft-dflash,ngram-simple --n-max 5 --draft on
# llama-restart, wait for the router, then:
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task3-mix.json
```

```bash
python3 ngram-probe/set-preset.py draft-dflash --n-max 5 --draft on
# llama-restart, wait for the router, then:
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task3-pinned.json
```

```bash
python3 ngram-probe/set-preset.py none --n-max 5 --draft off
# llama-restart, wait for the router, then:
python3 ngram-probe/probe.py --kind source --budget "$BUDGET" --max-tokens 256 --timeout 900 --dump ngram-probe/task3-none.json
```

Extract the numbers in one go:

```bash
python3 - <<'PY'
import json
for arm in ("mix", "pinned", "none"):
    t = json.load(open(f"ngram-probe/task3-{arm}.json"))["timings"]
    print(f"{arm:6s} draft_n={t.get('draft_n', 0):5d} accepted={t.get('draft_n_accepted', 0):5d} "
          f"tps={t['predicted_per_second']:.1f}")
PY
```

Read the checkpoint share for the mix arm from the journal after that run:

```bash
grep -c "restoring speculative checkpoint" /tmp/tools/visit-tail.log
```

Expected: the mix arm keeps a nonzero draft count where the pinned arm has stopped. Record whether the mix arm beats `none` in decode tokens per second. If the checkpoint path dominates and it does not beat `none`, say so plainly in the report; the fallback (keeping a small `rs` pin) is out of scope for this plan.

- [ ] **Step 3: Update the README section**

Replace the paragraph that ends with "You likely don't want that." with:

```markdown
Both MTP and DFlash2 drafts work with the phase arena, on the same pinned-draft
machinery: the draft weights (and, for MTP, the draft KV) are reserved in the
target arena, and the dynamic eject below trades the draft for decode capacity
as the working set grows. Only one pinned draft family can be active at a time.

A pinned draft can be mixed with an ngram speculator (for example
`draft-dflash,ngram-simple`). The pinned draft is still ejected when the working
set grows, but the ngram speculator keeps drafting through the eject: it needs
no draft model and no pinned arena bytes, so decode keeps a small draft win at
long context instead of losing speculation entirely. The ngram speculator is
first in the priority chain and the pinned draft fills the remaining draft
rounds.
```

Replace the line `ngram-map and ngram-simple currently produce zero drafts in this configuration.` with:

```markdown
The ngram speculators draft from the model's own token history, so they pay off
when the output repeats text (quoting, copying, structured edits) and stay close
to idle on fresh prose. `ngram-mod` needs a 24-token match and drafts least;
`ngram-simple` and `ngram-map-k` share a 12-token key.
```

- [ ] **Step 4: Add the measured numbers**

Under `## Results`, after the existing paragraphs, add this subsection, replacing `{MIX}` and `{NONE}` with the `tps` values Step 2 printed for `task3-mix.json` and `task3-none.json`. The subsection is not done while a brace remains.

```markdown
### Drafting after the eject

With `spec-type = draft-dflash,ngram-simple` the pinned draft is ejected at the
same working-set threshold as above, and the ngram speculator keeps drafting.
Measured at the same prompt and 256 generated tokens, the pinned-only arm stops
drafting after the eject while the mix arm keeps a nonzero draft count, and
decode runs at {MIX} t/s against {NONE} t/s with `spec-type = none`.
```

- [ ] **Step 5: Restore the user's configuration**

```bash
python3 ngram-probe/set-preset.py --restore
cmp "$PROBE_MODELS_DIR/models-preset.ini" ngram-probe/models-preset.ini.orig && echo identical
```

Call `llama-restart`, wait for the router, and confirm the model list answers.

- [ ] **Step 6: Commit (needs human approval)**

```bash
git add README.md ngram-probe
git commit -m "docs : describe drafting after a pinned draft eject"
```
