# Branch `fuss`: port our v2 delta onto Raymond's rewritten branch

- Date: 2026-10-11
- Status: approved in principle; awaiting written-spec review
- Repo: `/home/troed/llama.cpp-adaptive-kv-streaming`
- Supersedes: none. Related prior design: none written (`sync/adaptive-kv-stream-v2-2026-10-*` were ad hoc).

## 1. Goal

Move our fork's trunk onto Raymond's rewritten `feature/adaptive-kv-stream-v2` (tip
`1240f34c7`, ggml-org base `10a60cf30`) and re-apply only our genuinely-local features,
so that future syncs with his branch are cheap and we stop being 182 ggml commits behind.

His branch was **force-pushed** (`05c0a5e45 -> 1240f34c7`; the old tip is not an
ancestor of the new one). He re-ported the whole v2 architecture onto a much newer
ggml-org base. Our trunk is a port of his *old* v2 plus our own features, so the two
sides are two parallel rewrites of the same lineage.

## 2. Success criteria

1. CUDA Release build succeeds and the fork's tests pass, with a failure set compared
   against untouched `1240f34c7` so pre-existing failures are not attributed to the port.
2. A live server run on the host (`llama-restart` + a request): TG/PP sane and the
   `memory_phase:` lines behave.
3. Every ported feature passes the **improvement gate** (section 7): it measurably
   improves on Raymond's untouched tip, or it is dropped.

## 3. Scope

In scope: the trunk's local delta only — 36 commits / 93 files / +987 −572 lines
(excluding ggml-org and Raymond's old branch), of which 20 commits are substantive.

Out of scope (throwaway, not ported): `feature/kv-stream-phase-arena`,
`feature/ngram-drafting`, `safety/v2-sync-f1dc3b994`, `ngram-fast-work`,
`perf/pp-optimizations`, `diag/*`, `wip/*`.

## 4. Base, branch, workspace

- New branch **`fuss`**, created from `upstream/feature/adaptive-kv-stream-v2`
  (= `1240f34c7`). Not from ggml-org master, not from our old trunk.
- **Safety refs first** (he rewrites history): push his tip to our own remote as
  `origin/v2-raymond-1240f34c7`, and tag the old trunk tip `ebe3f1e0c` as
  `trunk-pre-fuss`, so neither is lost if he force-pushes again.
- **Work in the main workspace, not a worktree.** The host actions are hard-wired to
  `/home/troed/llama.cpp-adaptive-kv-streaming/{build,build-tests,build-tests-cuda}`;
  only the main checkout is built, tested and benchmarked.
- **Step 0 — baseline before porting:** on plain `fuss`, run `llama-build`,
  `llama-build-tests`, `llama-build-tests-cuda`, then `llama-test` and `llama-test-cuda`.
  Record the failure set; it is the reference for every later comparison.

## 5. Port inventory

Shared merge-base with his new tip is ggml-org `436f6f89e`. A straight 3-way merge of
his tip into our trunk produces **754 conflicted files**, so this is a port, not a merge.

### 5.1 Drop (16 of 36)

- `de0aad10a` (follow upstream 2026-10-03 API changes) and `f4bd82c47` (graph_optimize
  signature): his base is 164 ggml commits newer.
- `8cc443265` (repo intro cleanup), the docs commits (`3b46afd1f`, `c7a644afe`,
  `92617f306`, `4eac545f9`, `8da40ded4`, `713eef2a6`, `3fc03e043`, `c7a0d716f`,
  `a0671e66e`) and the README/gitignore commits (`041788b1f`, `41763fb65`, `34f19d472`):
  rewrite against his tree where still wanted.
- `tests/test-cuda-tile-{spans,access}.cu` are not in our 36; they came from his old work
  (`adb64391a`, `b9e1b35de`) and he dropped them in the new port. Follow him: no action.

### 5.2 Port (20 commits, original chronological order)

`†` = re-author against his restructured CUDA (see section 6).

| # | commit | subject | class |
|---|---|---|---|
| 1 | `c10812538` | spec : let ngram drafters share the streamed verify round with MTP | replay |
| 2 | `afa4ab41f` | spec : widen attached MTP KV streaming to draft depth 5 | replay |
| 3 | `f5718e53a` | ggml : add optional `alloc_size` to execution ops | replay (clean) |
| 4 | `42c5b37f5` | recurrent-spill : complete the execution ops literal | replay (clean) |
| 5 | `d239e6498` | ggml-cuda : size attention via the execution owner, drop width constant | † re-author |
| 6 | `e62f13fb0` | ggml-cuda : never size attention below the stock figure | † re-author |
| 7 | `b05184446` | ggml-cuda : size a span-covered verify from its own workspace | † re-author |
| 8 | `065acd3c2` | kv-stream : answer alloc_size from the stream model owner | replay |
| 9 | `16d1137e9` | kv-stream : decline alloc_size above the span ceiling | replay |
| 10 | `4d5343ba4` | kv-stream : answer alloc_size from the mtp draft proxy owner | replay (clean) |
| 11 | `72d9752fe` | tests : pin attention-only delegation of alloc_size | replay |
| 12 | `99ed6a5e3` | ggml-cuda : pin the sizing floor in the contract test | replay |
| 13 | `87bd27485` | kv-stream : widen the streamed verify width from 4 to 8 | † re-author |
| 14 | `5e0454450` | kv-stream : reserve the rebuilt MTP draft graph at the streamed width | replay |
| 15 | `3af00bffa` | tests : admit the wide verify arm in the context check | replay (clean) |
| 16 | `cc126b07f` | tests : correct the stated failure mode of the CPY guard test | replay |
| 17 | `5b6398a9a` | server : admit up to `LLAMA_KV_STREAM_MTP_DRAFT_MAX` drafts with the vision arena | replay |
| 18 | `43fcee5d6` | kv-stream : run ngram drafters alongside MTP through the span path | replay |
| 19 | `d800b5b62` | kv-stream : admit ngram drafters alongside MTP with the vision arena | replay |
| 20 | `ebe3f1e0c` | cuda : bound the CUDA-graph cache so a rebuilt-per-batch graph cannot exhaust VRAM | replay |

A mechanical probe — replaying the series in order onto his tip with `git apply --3way` —
measured **10/36 applying without conflict**. That number bounds rather than measures the
conflict surface: `--3way` leaves conflict markers on failure, so once one commit
conflicts, later commits in the same files fail spuriously. The genuine signal is that the
conflicts concentrate exactly where the architecture moved.

## 6. Architecture mapping (the re-author commits)

### 6a · `alloc_size` on execution ops (commits 3–7, 12)

His `ggml_backend_execution_ops` has five callbacks (`supports`, `compute`, `can_write`,
`modified`, `destroy`) and **no sizing hook**. Add
`size_t (*alloc_size)(void *, ggml_backend_buffer_type_t, const ggml_tensor *)` plus
`ggml_backend_execution_alloc_size(owner, buft, op)`. Existing owners answer or decline
(stream model owner, MTP draft proxy owner, recurrent-spill literal); the CUDA attention
sizing asks the owner first, falls back to stock, and **never sizes below the stock
figure**. Rationale: a wide streamed verify must size its workspace from the capped span
geometry, which his `mma_span_launch` computes — so the answer must come from the owner,
not from a global width constant.

### 6b · width-8 streamed verify (commit 13)

His dispatch is `plan.ncols1 ∈ {1,2,4}` in `ggml_cuda_flash_attn_ext_mma_f16_spans`; there
is no 8. The port becomes:

1. **Check first** whether his plan can already reach width > 4 by another route; if so,
   this commit shrinks or disappears.
2. Let `mma_span_launch_make` emit `ncols1 == 8` when the query width warrants it.
3. Add `ncols1 == 8` arms (`<8,8>`, `<8,2>`).
4. Instantiate the kernel for `ncols1 = 8` in the `.cu.in` instance templates and confirm
   the nthreads table covers width 8.
5. Relax the geometry admissions in `ggml-cuda.cu` / `kv-stream-partial.cu` that cap
   queries at 4.

### 6c · MTP draft depth 5, cap and residency (commits 2, 17)

`LLAMA_KV_STREAM_MTP_DRAFT_MAX` in `llama.h`, per-round `n_max` in the speculator,
`llama_kv_stream_residency`, clamps moved into the MTP draft path, sub-span MTP contexts
off the adaptive lease. His `src/**` file layout is identical to ours, so this is mostly
replay.

### 6d · ngram + MTP coexistence (commits 1, 18, 19)

Share the streamed verify round between ngram drafters and MTP; admit both through the
span path and the server's vision arena.

### 6e · CUDA-graph cache bound (commit 20)

`ggml-cuda/common.cuh`, additive; likely clean replay.

## 7. Improvement gate

No feature is ported merely because it is ours. It must be **shown to improve on
Raymond's untouched tip by measurement**, and is dropped if neutral or worse.

Harness: same model (`Qwen3.8-27B-IQ4_XS-ASCII-Condensed-MTP.gguf`), same preset, same
prompt set, fixed context ladder, repeated runs. Compare **his untouched `fuss`**
against **`fuss` + our feature**. Metrics from the existing pipeline (`performance.log` +
`plot_performance.py`, `memory_phase:` lines, `llama-gpu`).

| feature | metric | A/B how | drop rule |
|---|---|---|---|
| width-8 verify | TG/s, draft acceptance at 32k/64k/100k+ | build variant width 4 vs 8 | no TG gain, or VRAM/OOM regression → drop |
| MTP depth 5 + cap | TG/s, acceptance, per-round cost | draft depth 4 vs 5 | worse TG or acceptance → drop |
| ngram+MTP coexistence | TG/s with both drafters | both-on vs MTP-only | no gain over MTP-only → drop |
| `alloc_size` owner sizing | peak VRAM, preflight accuracy, OOM/rebuild count | stock sizing vs owner sizing | mis-sizing or OOM → drop or keep behind a flag |
| CUDA-graph bound | VRAM over long runs, OOM | bound on/off | keep if free (stability) even if neutral |
| tests / CPY guard | correctness only | — | always keep |

Because `alloc_size` is a prerequisite for a capped wide verify, some measurements can
only be isolated after the owner-sizing foundation lands. The gate therefore runs at group
boundaries, with targeted reverts for isolation. Each feature stays in its own commit(s)
so a single feature can be reverted without unpicking the rest.

## 8. Verification plan

1. **Baseline** on untouched `fuss`: `llama-build`, `llama-build-tests`,
   `llama-build-tests-cuda`, then `llama-test` + `llama-test-cuda`; record the failure set.
2. **After each group**: rebuild, rerun, diff failures against the baseline. A failure
   present at baseline is not ours.
3. **Final**: full build + test set (modulo baseline), then `llama-restart`, a live
   request, a benchmark run, and read `memory_phase` + `llama-gpu`.
4. The improvement gate (section 7) runs against the same baseline.

## 9. Risks and open questions

- **His tip may not build in our environment** (CUDA 13, our GPU arch). Step 0 answers
  this before any porting; if it does not build, the whole plan is blocked and must be
  re-scoped.
- **6b may be moot** if his plan already generalises past width 4.
- **Measurement noise** requires repeated runs and an identical preset/context; the live
  router preset differs from a synthetic benchmark.
- **`build-tests-cuda` stale cache** can fail with "CUDA_ARCHITECTURES is empty"; delete
  the directory and reconfigure.
- **Feature interdependence** limits isolation; the gate needs deliberate revert variants.
- His re-port may have *improved* something ours also touched (for example his plan-based
  sizing). The improvement gate is what catches this, and the correct answer may be
  "take his".

## 10. Non-goals

- Not merging his branch; not rewriting his history.
- Not refreshing to ggml-org master in this effort (his tip is the base; a ggml refresh
  is a separate later step).
- Not porting the throwaway sibling branches.
- Not touching the benchmark/data/preset infrastructure.
