# Brainstorm checklist — port our v2 features onto Raymond's rewritten branch

Path: **architectural** (announced 2026-10-11)

1. [x] Explore project context (topology, delta, trial merge, feature gaps)
2. [x] Visual companion not needed (no visual question arose)
3. [x] Ask clarifying questions (endgame, scope, target base, done criteria, approach)
4. [x] Propose approaches with trade-offs (replay / re-author / hybrid; hybrid chosen)
5. [x] Present design in sections, approved
6. [x] Write design doc -> docs/superpowers/specs/2026-10-11-fuss-port-design.md, committed
7. [x] Spec self-review (placeholders, consistency, scope, ambiguity)
8. [x] User reviewed and approved the written spec
9. [x] Invoke writing-plans -> docs/superpowers/plans/2026-10-11-fuss-port.md

- Branch name: **fuss** (user, 2026-10-11).
- IMPROVEMENT GATE (user, 2026-10-11): do not port our work blindly; each feature must be shown to
  improve on Raymond's untouched tip by measurement (many runs available). Drop it if neutral/worse.

## Facts established
- His branch force-pushed: 05c0a5e45 -> 1240f34c7 (history rewritten).
- His ggml base 10a60cf30 (18 behind ggml-org/master); ours 436f6f89e (182 behind).
- Shared merge-base: 436f6f89e. Trial 3-way merge: 754 conflicted files.
- Our local delta: 36 commits / 93 files / +987 -572 (40% docs).
- src/llama-kv-stream-*.cpp layout identical both sides (32 files). ggml-cuda restructured (his: *.cu.in templates).
- Features absent from his tree: alloc_size execution-ops, width-8 span verify (his uses plan.ncols1 1/2/4),
  LLAMA_KV_STREAM_MTP_DRAFT_MAX + draft depth 5 + llama_kv_stream_residency, ngram+MTP coexistence,
  CUDA-graph cache bound, tests/test-cuda-tile-{spans,access}.cu.
- Droppable: de0aad10a (upstream 2026-10-03 API follow), f4bd82c47 (graph_optimize sig), docs/meta commits.
