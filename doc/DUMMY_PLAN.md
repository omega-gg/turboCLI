# Plan: `dummy.md` — plain-English turboCLI documentation

## Context

turboCLI has `implementation.md`, the dense design doc. Same as was done for turbo-offloader, it
needs a `dummy.md`: a brief-but-complete plain-English explanation of what turboCLI does, step by
step through the architecture, gradually getting more technical, readable by technical people
without diffusion background — kept up to date along iterations, with that mentioned in
`implementation.md`.

## Files

1. `dummy.md` (new, top level beside `implementation.md` — mirrors the turbo-offloader pairing).
2. `implementation.md` — a "New here?" pointer after the style line.
3. `doc/DUMMY_PLAN.md` — this plan, archived per repo convention.
4. `turboCLI.pro` — `dummy.md` and `doc/DUMMY_PLAN.md` added to `OTHER_FILES`.

## Content outline

Progressive disclosure, ASCII diagrams, 99-column wrap. This dummy focuses on what the *runner*
does (orchestration, engines, server, install); the diffusion/offloading deep-dive is delegated
to turbo-offloader's `dummy.md` to avoid duplicating it.

1. **What is this?** — run image-generation models from a CLI or a local HTTP server; install an
   engine once (the only online step), then generate offline on cpu/cuda/mps; the LGPL runner /
   pluggable (possibly GPL) backend split.
2. **60-second primer** — the diffusion pipeline in three sentences + a small vertical diagram;
   the heavy model may not fit — the backend streams it (see the offloader's `dummy.md`).
3. **The big picture** — layered diagram: shell wrappers → front-ends (cli.py / server.py) →
   core.py → engines + backend seam; one line per layer on what it owns.
4. **Step by step: from command to PNG** — wrapper env + cd, the flat params dict, backends
   discovered before torch, engine lookup, the resident-pipe cache, per-step progress lines,
   `Saved:` as THE success sentinel.
5. **Engines: a model is a page of declarations** — the z-image recipe verbatim, string-named
   classes / no torch at import, BASE inheritance (variants write only their delta), stock vs
   comfy-reuse.
6. **The server** — keep the model warm, endpoints in plain words, latest-wins preemption, idle
   release, the streamed-text / always-200 contract.
7. **Installing and removing** — three stages, everything pinned, the engine registry and
   reference-counted removal (external ComfyUI files never deleted).
8. **Technical details worth knowing** — one sentence each, pointing to `implementation.md`:
   backend-before-torch, the seam as licensing firewall, the pipe cache key, the encode cache,
   CPU-generator determinism, path-deduced folders.
9. **Where to go next** — `implementation.md`, `doc/*.md`, `bash/README.md`, the offloader docs.
10. **Keeping this document up to date** — revised with each iteration; on disagreement
    `implementation.md` is right.

## Style rules

- 99-column wrap everywhere, including diagrams.
- Plain English; every term (VRAM, engine, LoRA, seed) defined in one clause at first use.
- No benchmarks, no line-number archaeology — *what and why* here, *exactly how* in
  `implementation.md`.

## Verification

- No new line over 99 columns; diagrams column-aligned.
- Relative links resolve; facts checked against the sources (core.py, server.py, cli.py,
  install.py, `engine/_inherit.py`, `bash/turbo/*.sh`).
