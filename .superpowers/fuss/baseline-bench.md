# Performance baseline on Raymond's untouched tip - NOT CAPTURED (blocked)

Status: blocked on the measurement endpoint, 2026-10-11. Task 0 Step 6 did not complete.

## What was established

- The router that serves Qwen3.8-27B spawns the model server from **our fork's build**:
  `/home/troed/llama.cpp-adaptive-kv-streaming/build/bin/llama-server` (verified from the
  router's own `/v1/models` args, and the router log reports `build 11689 (572fe0867)`,
  which is the `fuss` HEAD). So `llama-build` + a model (re)load does measure our code.
- The probe log `/home/troed/llm-models/probe-server.log` is live and receives that server's
  output (slot timings, `spec statistics draft-mtp`, `memory_phase`).
- The preset was set to arm A (`spec-type=draft-mtp`, `spec-draft-n-max=3`,
  `kv-stream-arena-mib=4000`, probe log file) via `ngram-probe/mtp_width_probe.py set`.

## The blocker

Every probe request fails immediately with `curl failed (52): curl: (52) Empty reply from
server` - for the 462800-char streamed prompt AND for a 40000-char smoke prompt, so it is not
payload size. The server does receive and process the work: the probe log shows the exact
probe prompt being prefilled (130166 tokens, ~571-900 t/s, finishing at progress 1.0).
So requests reach the model server but the HTTP response comes back empty.

Contributing fact: on the same `:8100` router, `gemma-4-12b` is served from
`/home/troed/llama.cpp/build/bin/llama-server` (a different checkout) while Qwen3.8-27B is
served from our repo build. `:8100` may therefore be an older router instance relaying to a
newer model server.

Also: the systemd `llama` unit now listens on **9931** (llama.cpp changed its default port),
while the probe's `PROBE_HOST` defaults to `192.168.0.2:8100`. The sandbox egress policy
allows `192.168.0.2:8100` but has no rule for `:9931`, so the freshly started unit is not
reachable from here.

## Consequence

The improvement gate (Task 8) cannot run until one of these is true:

a) a sandbox egress rule for `192.168.0.2:9931` exists, so the unit we restart (verified our
   build) can be measured directly; or
b) the `:8100` router is restarted/repaired so it relays responses from the current build.

Until then the port proceeds; the perf baseline is re-established later from the same tip
(the tip is tagged `trunk-pre-fuss` and preserved as `fuss`'s parent, so the tip build is
reproducible with one rebuild).
