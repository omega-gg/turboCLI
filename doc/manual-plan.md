# Plan: a user manual for turboCLI (MANUAL.md)

## Context

turbopixel has a user manual (its `MANUAL.md`): second person, task first, a
first-run walkthrough, then one section per area. turboCLI has none. What a user needs is spread
over the README (quickstart), `bash/turbo/README.md` (each script's usage, verbatim), `dummy.md`
(how it works, for a newcomer), `engine.md` (which engine to pick) and `test/README.md` (mask
recipes). We add `turboCLI/MANUAL.md`, written like turbopixel's, for someone running turboCLI from
the bash scripts.

Decisions taken with the user:
- **Audience: script users.** The HTTP server and Python embedding get one short section each,
  pointing to `implementation.md`; no full API reference.
- **Explain, link usage.** Each task is prose plus one or two real commands; the full argument
  lists stay in `bash/turbo/README.md` and the manual links to the script's section there.

## The file: `turboCLI/MANUAL.md` (~250-300 lines)

Style as turbopixel's manual: "you", short paragraphs, small tables where a choice is made, code
blocks for commands. Lines ≤99 columns, American spelling, no `--` or spaced ` - ` joins.

1. **Intro** (3 short paragraphs): what turboCLI does (generates images locally from the command
   line, or behind a small local server), that it runs entirely offline once an engine is
   installed, and the four kinds of work: text-to-image, image-to-image, masks, applying masks.
   Link `dummy.md` for how it works inside.
2. **First run**: set `SKY_PATH_BIN`, build python (`bash/python/build.sh default`), build turboCLI
   for a renderer (`bash/turbo/build.sh <cpu|cuda|mps>`), `check.sh`. A table for the renderer
   choice (CPU / CUDA / Apple MPS, when to pick each), as turbopixel's first run has. `latest`
   and `clean` one line each.
3. **Choosing and installing an engine**: point to `engine.md` to choose; `install.sh <engine>
   <renderer>` with the arguments that matter in prose: dtype (default keeps the file as
   published; bfloat16 for CUDA, float16 for MPS; recast needs a remove first), offload and
   slicing recorded with the install, the ComfyUI folder to reuse a ComfyUI install's files
   (`comfy-` engines), and the token for gated engines (`HF_TOKEN` works too, never recorded).
   Then `check-model` (list, one engine, `SETTINGS:<engine>`) and `remove.sh` (reference-counted,
   never deletes a real ComfyUI install's files).
4. **Generating an image**: `text-to-image.sh` with the arguments in order (prompt, output, width,
   height, seed, steps); `-1` means "the engine's own" for seed and steps; what the output lines
   mean (`loading ... model`, `model ready`, `generating "..." (N steps)`, the `step k/N` progress,
   `Saved: <path>`, `ERROR: ...`). Image-to-image: up to 4 comma-separated references, the first
   one leading, the output size its own. LoRAs: `<path>@<weight>`, comma separated.
5. **Masks and mattes**: the two families in one table (`mask`: what an edit changed, from an
   input and its reference, modes `default` / `region`; `mask-birefnet` / `-lucida` / `-inspyrenet`:
   a subject's matte, with an optional plate to keep the cast shadow). Options in prose (`cutoff`,
   `glow`, `fade`). Then `image-apply-mask.sh`: composite, putalpha (an RGBA cutout), trim. One
   end-to-end example (matte a photo, cut it out, trim it), and a link to `test/README.md` for the
   longer recipes.
6. **Running a server**: why (the model stays loaded between images, so only the first one pays
   the load); `server.sh start [port] [scan]`; passing `host:port` (or a port) as the last
   argument of any generating script; latest request wins (a newer one stops the one running);
   `cancel`, `clear`, `stop`; an idle model is released after 600 s. One line pointing to
   `implementation.md` for the HTTP endpoints, for a host app.
7. **Memory and speed**: a table of offload modes (`offloader` the default and what it does in
   one line: streams weights to the GPU so large models fit small cards; `none`; `model_cpu`;
   `sequential_cpu`) and when to pick each; `slice` for tight memory; dtype's effect on memory.
   Note that `offloader` is required by the `comfy-` engines marked so in `engine.md`.
8. **When something goes wrong**: short entries: a gated engine without a token; a download that
   fails behind a VPN or proxy that intercepts HTTPS (turn it off for the download); an engine not
   installed (`check-model`); out of memory (offloader, slicing, a smaller size); a model that will
   not load after an update (`check.sh`, reinstall).
9. **Where files live**: under `SKY_PATH_BIN`: the turboCLI install, `model/`, `engine/`, `lora/`;
   `comfy-` engines' files in the ComfyUI folder when one was given.
10. **Using it from Python**: one paragraph: `runner.core.generate(params, emit)` exists for
    embedding, see `implementation.md`.

Every command, argument name and default is taken from the scripts' usage text
(`bash/turbo/*.sh`) and `runner/` (server port 8080, idle 600 s `--timeout`, output lines in
`runner/core.py`), not from memory.

## Other edits

- `README.md`: add `- [Manual](MANUAL.md)` to the top link list, first, in the list's own style.
- `dummy.md`, "Where to go next": one line pointing to `MANUAL.md` for using it.
- `turboCLI.pro`: add `MANUAL.md` to `OTHER_FILES` next to `engine.md`.

## Verification

- Every script invocation in the manual is checked against that script's usage block (engine
  names, argument order, option names, defaults): read each example back against
  `bash/turbo/README.md`.
- Every link resolves (relative paths to `engine.md`, `dummy.md`, `implementation.md`,
  `test/README.md`, `bash/turbo/README.md` anchors).
- Lines ≤99 columns; no `--` or spaced ` - ` joins; American spelling.
- No commit until the user says so; no attribution.
