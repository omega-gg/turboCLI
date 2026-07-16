# Plan: `implementation.md` — turboCLI design document

## Context

turbo-offloader has `implementation.md` (dense design doc) + `dummy.md` (plain English). turboCLI
— the LGPL runner that drives it — had only a README and three per-feature plan docs in `doc/`.
`implementation.md` details the architectural and implementation choices so an experienced
developer or an LLM can dig into specifics quickly: same role, tone and conventions as the
offloader's doc (dense, rationale-first, file:line references, 99-column wrap), and explicitly
kept up to date along iterations.

## Files

1. `implementation.md` (new, top level beside README.md).
2. `doc/IMPLEMENTATION_PLAN.md` — this plan, archived per repo convention.
3. `README.md` — one pointer line to `implementation.md`.
4. `turboCLI.pro` — `implementation.md` added to `OTHER_FILES`.

## Content outline

Header: what turboCLI is (LGPL runner, two front-ends over one core, string-named backend seam),
the 99-column style line, the kept-up-to-date line, cross-links to the offloader's
`implementation.md`/`dummy.md`. Then:

1. **Layout** — repo tree + the deployed tree under `$SKY_PATH_BIN/gg.omega/turbo`.
2. **The three-stage install** — python build, turbo build (pinned clones, offloader graft into
   `backend/`, relocatable venv, inline pip pins, model/engine detach-reattach), engine install.
3. **The runner package** — cli.py (argv contract, core-after-argparse), server.py (endpoints,
   always-200 streamed body, latest-wins preemption, gpu_lock, idle watcher, port scan),
   install.py (registry, stock vs comfy paths, HF-cache trim, refcounted removal, torch-free),
   check.py.
4. **The engine system** — the declaration contract table, no-torch-at-top-level rule,
   discovery, BASE inheritance (one-time setattr fold, hooks keep base `__globals__`, loud
   errors), stock vs ComfyUI-reuse, the 9-engine table.
5. **The backend seam (runner side)** — discovery before `import torch` and why, seam methods,
   native offload modes, the validation ladder, the licensing firewall, `Ctx`.
6. **One generation** — params dict, pipe cache key, seed policy, i2i input fit, the progress
   protocol (tqdm mirrored into a sink), the wire contract block, error-handling split.
7. **Caches** — resident pipeline + the prompt-encode cache (keying, CPU parking, eviction).
8. **Determinism and reproducibility** — pinned everything, CPU generator.
9. **Platform notes** — Windows/MPS/dtype/offline specifics.
10. **Notes** — server-mode curl path, .pro role, links to `doc/*.md` and the offloader docs.

## Style rules

- 99-column wrap (markdown table rows may exceed, matching the offloader doc's practice).
- Dense, rationale-backed tone; file:line refs verified against the sources before writing.
- No benchmarks (they live in the offloader doc and `doc/comfy-krea2-turbo-plan.md`).

## Verification

- `awk 'length > 99'` over the new/edited files → only table-row exceptions.
- file:line references spot-checked against `runner/core.py`, `runner/server.py`,
  `runner/install.py`, `runner/cli.py`, `runner/engine/_inherit.py`, `bash/turbo/build.sh`.
- Relative links resolve; cross-repo references are prose, not links (sibling repo).
