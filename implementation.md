# turboCLI implementation

LGPL runner for diffusion image generation: two thin front-ends (`cli.py` one-shot, `server.py`
resident HTTP) over one shared core, engines as **pure declaration modules** discovered at
startup, and weight-placement strategy behind a **string-named backend seam** (`backend/<mode>/`,
external, may be GPL — e.g. `turbo-offloader`). The design goals are fast cold and warm starts,
one code path for CLI and server, and adding a model in a handful of declaration lines.

**Style:** code and comments wrap at 99 columns.

**This document is updated with each turboCLI iteration.** It details the implementation so an
experienced developer (or an LLM) can dig into specifics quickly. For the offload backend's own
internals read `turbo-offloader`'s `implementation.md`; its `dummy.md` explains diffusion and
offloading in plain English.

## Layout

```
turboCLI/
  runner/            the LGPL Python package
    cli.py           one-shot argv front-end
    server.py        HTTP front-end (threads, preemption, idle release)
    core.py          shared engine: discovery, resident-pipe cache, one generation
    install.py       online installer + engine registry + reference-counted remove
    check.py         install verifier (torch-free)
    engine/          one declaration module per engine (+ _inherit.py, not an engine)
  bash/
    python/          build.sh / check.sh -- bundled standalone CPython + uv
    turbo/           build/install/remove/check/check-model/server/text-to-image/image-to-image
  backend/           EMPTY in-repo (a .gitignore placeholder); build.sh grafts the offloader here
  doc/               plan docs, kept as records after implementation
```

Deployed layout (created by `bash/turbo/build.sh` under `$SKY_PATH_BIN/gg.omega`):

```
gg.omega/
  python/                          standalone CPython + uv (bash/python/build.sh)
  cache/uv, cache/huggingface      build/download caches (HF cache trimmed after install)
  turbo/
    .commit                        pinned turboCLI commit marker (the check.sh contract)
    .venv/                         relocatable uv venv
    runner/  bash/                 the pinned turboCLI clone
    backend/offloader/             turbo-offloader's package, grafted at build time
    model/<Name>/                  canonical diffusers copies, shared across engines
    model/<Name>/lora/             engine-declared LoRAs beside the base
    engine/<id>/engine.json        per-engine registry (+ scaffold for comfy engines)
```

## The three-stage install

1. **`bash/python/build.sh`** — downloads python-build-standalone + uv into `gg.omega/python`.
   macOS detects the *hardware* arch (`sysctl hw.optional.arm64`) so Rosetta never picks x86_64.
   `bash/python/check.sh` verifies the exact interpreter version.
2. **`bash/turbo/build.sh <cpu|cuda|mps> [latest]`** — deploys turboCLI itself. Shallow-clones
   the pinned turboCLI commit (writes it to `.commit`, strips `.git`), shallow-clones the pinned
   turbo-offloader commit and moves its `offloader/` package to `backend/` (build.sh:203-207),
   creates a relocatable uv venv, then installs the pinned stack: torch from the per-renderer
   wheel index (cu130 / PyPI-mps / cpu), the HF stack, diffusers from a pinned git commit, and
   the comfy wheels (`comfy-kitchen` everywhere, `comfy-aimdo` CUDA-only). Every pin lives
   inline in build.sh — no requirements.txt — "so a build six months from now resolves the same
   stack" (build.sh:43-44). `latest` drops all pins with a "not reproducible" warning.
   `model/` and `engine/` are detached to `.turbo-model`/`.turbo-engine` before the wipe and
   reattached after (build.sh:178-211), so a rebuild never re-downloads ~20 GB of weights.
3. **`bash/turbo/install.sh <engine> [dtype] [ComfyUI dir]`** → `python -m runner.install` — the
   only ONLINE step (exports `HF_HOME`, `hf-transfer`); generation itself runs with
   `HF_HUB_OFFLINE=1`. Details under `install.py` below.

`bash/turbo/check.sh` (turboCLI installed?) and `check-model.sh` (engine installed? `list`) are
the machine contracts a host app polls: fixed one-line outputs + exit code 0/1, both torch-free
and venv-free so they run under the bundled python.

## The runner package

### `cli.py` — one-shot front-end

Builds a flat string-valued params dict from 13 argv flags (`cli.py:42-73`; defaults: mode
`text-to-image`, 512x512, seed -1, inference -1, renderer `cpu`, offload `offloader`, slicing
`none`) and calls `core.generate(params, emit)`. Every input arrives via argv flags — no shell
interpolation; the wrappers pass `--prompt="$1"` in equals form so a prompt starting with `-` is
not read as a flag (`cli.py:26-28`). Two deliberate orderings:

- `from runner import core` happens **after** argparse (`cli.py:78-80`): importing core runs
  offload-backend discovery before torch (below), and deferring it keeps `--help` instant.
- Exit code is 0 only when `generate` returned True (image saved); an unexpected exception prints
  `ERROR: <traceback>` and exits 1 (`cli.py:82-89`).

Must run from the deployed dir — `backend/` is discovered relative to cwd and `model/`/`engine/`
are deduced from the runner's own path; the wrappers `cd` there (`cli.py:30-32`).

### `server.py` — HTTP front-end

A long-lived host that caches the model across generations. Owns only the server-specific
concerns — HTTP, locks, preemption, the idle watcher — and delegates the work to `core.generate`
(`server.py:23-27`). Importing core (`server.py:41-43`) preserves the pre-torch invariant:
nothing above it touches torch.

| endpoint | behavior |
|---|---|
| `GET /health` | `ok` (server.py:123-127) |
| `POST /generate` | urlencoded body → the SAME params dict cli builds (`parse_qs`, server.py:177-183); streams emit lines as a plain-text body |
| `POST /cancel` | bumps `latest_id` + `last_was_cancel`; never blocks (server.py:139-149) |
| `POST /clear` | cancel + acquire `gpu_lock` (30s timeout) + `release_pipe` (server.py:151-170) |
| `POST /shutdown` | responds, then shuts down on a thread (server.py:132-137) |

- **Always-200 + streamed body**: the status line is sent before generating, so failure cannot be
  an HTTP code — "the client treats a `Saved:` line as success and anything else as failure"
  (server.py:185-190). TCP_NODELAY is set so each flushed progress line goes out at once
  (server.py:107-110); emit swallows client-gone write errors so the image still saves
  (server.py:192-198).
- **Latest-wins preemption**: requests run in their own threads (`ThreadingHTTPServer`, daemon,
  queue 64) but one generation touches the GPU at a time (`gpu_lock`). Each `/generate` claims
  `my_id = ++latest_id` (server.py:200-204); the in-flight job's `should_stop()` closure compares
  ids each step and returns `"cancel"` or `"supersede"` (server.py:221-226); a request that went
  stale while waiting for the lock emits `SUPERSEDED:` without running (server.py:211-217).
- **Idle watcher**: a daemon thread ticks every 30s and releases the resident pipe once idle
  longer than `--timeout` (default 600s); it *skips* (non-blocking acquire) when busy
  (server.py:78-99).
- **Port scan** (`--scan`): binds the first free port in `[--port, --port + --range - 1]`; the
  probe socket deliberately has no SO_REUSEADDR "so an in-use port reliably fails, including on
  Windows" (server.py:266-284).

### `install.py` — installer and registry

Deliberately does NOT import core: install is online and needs no GPU, so no backend discovery,
no CUDA init (`install.py:52-54`); it duplicates `default_folder()`/`engine_folder()` so
install/check stay torch-free (`install.py:68-82`). Heavy torch/diffusers imports happen only
inside the base-reinstall branch (`install.py:490-493`).

- **Registry**: `engine/<id>/engine.json` is the source of truth for "installed" — model name,
  pinned revision, LoRA list; for a comfy engine its component refs + scaffold source
  (`install.py:40-50`). No per-model-dir markers: a shared dir's revision is derived from any
  engine record referencing it (`_model_revision`, install.py:168).
- **Stock install** (`install.py:444-557`): `from_pretrained(repo, revision=pin)` →
  `save_pretrained` into `model/<name>` (a self-contained canonical copy), fetch the engine's
  declared LoRAs into `model/<name>/lora/`, then trim the HF cache for everything pulled — disk
  holds one copy, not cache+copy (install.py:540-551). Idempotent and selective: `base_ok` skips
  the base re-download when the registry already records the pinned revision, and only missing
  LoRAs are fetched (install.py:461-476) — installing `-lightning` over an existing base pulls
  just the LoRA. `--dtype default` maps to `torch_dtype=None` (diffusers coerces non-torch
  values like `"auto"` to float32, so None is what keeps the stock dtype, install.py:501-504).
- **Comfy install** (`_install_comfy`, install.py:221-304): reuse each COMFY component already
  under the ComfyUI install's `models/`, `hf_hub_download` only the missing ones *into that
  tree*, fetch the tiny SCAFFOLD (configs/tokenizer/scheduler, no weights) into `engine/<id>`,
  and write engine.json with `external` flagging weights outside our model folder "so remove.sh
  only unregisters them, never deletes them" (install.py:281-291). Without `--comfy` a
  self-contained `model/ComfyUI/` layout is used (install.py:433-436).
- **Removal is reference-counted GC** (`_remove`, install.py:320-389): drop the registry entry
  first, then delete only the models / LoRAs / comfy components no surviving engine references —
  and only files under our model folder; a user's external ComfyUI is never touched.

### `check.py` — verifier

Reuses `install._discover` + `install._engine_installed`. "Light by design: no torch/diffusers
import, so it runs under the bundled python without the venv" (check.py:33-35).

## The engine system

An engine is a plain module of constants plus up to three optional hooks. Hard rule: **no
torch/diffusers imports at top level** — pipeline classes are `"module:Class"` strings resolved
lazily (`_resolve`, core.py:110), "so discovery stays cheap and an unused engine costs nothing"
(engine/__init__.py:23-26). Discovery globs `engine/*.py`, skips `_`-prefixed files, keys by
`ID` (core.py:81-89; mirrored torch-free in `install._discover`).

| symbol | meaning |
|---|---|
| `ID` | registry key + `--engine` value (never inherited) |
| `TYPE` | the backend-seam identity; variants sharing a pipeline family keep the same TYPE (`engine_type`, core.py:262) |
| `BASE` | base engine ID to inherit from (below) |
| `PIPELINE` | `"diffusers:XxxPipeline"` string, resolved lazily |
| `TRANSFORMER` | `"diffusers:XxxTransformer2DModel"`; **presence = offload-wired** (core.py:553-558) |
| `MODES` | wire modes tuple: `"text-to-image"` / `"image-to-image"` |
| `CFG` | one `(kwarg, value)` pair injected into the pipe call (core.py:599-600) |
| `INFERENCE` | default step count when the caller passes `-1` (fallback 4, core.py:586-587) |
| `MODEL` | install spec `{repository, model, revision}` (stock) |
| `COMFY` | ComfyUI-reuse spec `{repository?, revision, components: [{role, path, ...}]}`; presence dispatches both install and `resolve_model` (core.py:177) |
| `SCAFFOLD` | tiny config-only snapshot spec (`allow_patterns`, no weights) |
| `LORAS` | install-time LoRA list `[{repository, file, revision}]` |
| `loras(params)` | runtime preset hook → `[(filename, weight)]`, may be prompt-dependent |
| `extra_key(params)` | extra tuple folded into the pipe cache key (core.py:297-301) |
| `load(ctx, params)` | full custom loader, bypasses `_default_load` |

**Inheritance** (`engine/_inherit.py`, design in `doc/engine-inheritance-plan.md`): a variant
declares `BASE = "<base id>"` and writes only its delta. `resolve()` folds each BASE chain once
right after discovery — missing contract symbols are copied with one-time `setattr`, so
consumers read plain module attributes at zero per-access cost (`_inherit.py:41-71`). Hooks are
copied as function **objects**, so an inherited `load()`/`loras()` keeps the base module's
`__globals__` and still resolves the base's private helpers without copying them
(`_inherit.py:29-33`). Unknown base → `KeyError`, cycle → `ValueError`, both loud at discovery.
To *extend* rather than replace, import the base module (cheap, no torch) and compute:
`LORAS = base.LORAS + [...]` (engine/__init__.py:28-37). A stock variant can never accidentally
gain `COMFY` — its stock base has none.

**Stock vs ComfyUI-reuse**: stock engines resolve to `model/<MODEL["model"]>` (a canonical
diffusers repo, shared and reference-counted across engines); COMFY engines resolve to
`engine/<id>` (scaffold + engine.json pointing at ComfyUI-owned split single files — no second
multi-GB download for a ComfyUI user). Dispatch is purely `hasattr(mod, "COMFY")`.

| engine | notes |
|---|---|
| `flux2-4b` | TYPE `flux2`, Flux2KleinPipeline, t2i+i2i, 4 steps; pure declaration |
| `z-image-turbo` | TYPE `z-image`, ZImagePipeline, t2i, 8 steps; pure declaration (the README's "sample model": ~9 lines) |
| `qwen-image-edit-2511` | TYPE `qwen-image-edit`, QwenImageEditPlusPipeline, i2i, `true_cfg_scale 1.0`, 40 steps |
| `qwen-image-edit-2511-lightning` | BASE = the above; delta: 4 steps + the lightning LoRA (install `LORAS` + runtime `loras()`) |
| `qwen-image-edit-2511-lightning-angles` | BASE = lightning; adds the angles LoRA for `<sks>` prompts; `extra_key` = the `<sks>` flag so flipping it reloads the pipe |
| `comfy-z-image-turbo` | same weights as z-image-turbo from ComfyUI's 3 single files; custom `load()`, works on the native path too |
| `comfy-qwen-image-edit-2511` | 39 GB bf16 DiT streamed + scaled-fp8 TE (`quant: True`); components span three Comfy-Org repos; **offloader-only** (`load()` raises without a backend) |
| `comfy-qwen-image-edit-2511-lightning` | BASE = the above; entire delta = one extra COMFY component (the LoRA) + 4 steps |
| `comfy-krea2-turbo` | both DiT and TE scaled-fp8; hand-written key converter (validated 1:1, 430/430); offloader-only; deliberately standalone — it differs on transformer, TE and pipeline, so BASE would override nearly everything (`doc/comfy-krea2-turbo-plan.md`) |

## The backend seam (runner side)

- **Discovery precedes `import torch`** — module-level code at the top of core.py
  (core.py:39-73): every `backend/<mode>/__init__.py` under cwd is a candidate; the dir name IS
  the offload-mode string; `pre_torch_init()` must run before torch initialises (the offloader's
  CUDA allocator hooks can't install afterwards); a failed init is skipped silently. Nothing in
  turboCLI names a specific backend.
- **Seam methods**: `pre_torch_init() / available() / supports(engine_type) /
  load_pipe(model, dtype, pipeline_cls, transformer_cls, device, lora_files) /
  load_pipe_comfy(...)` plus optional per-generation `prepare(pipe)` (core.py:727-733),
  `reclaim(pipe)` (core.py:744-752) and `release(pipe)` (core.py:311-327). Class objects flow
  runner→backend, resolved lazily from the engine's declared strings, "so the class table lives
  here, not duplicated in the backend" (core.py:281-289). The backend stays model-agnostic;
  comfy engines hand it data specs (meta-builders + file paths + key-remap callables).
- **Native modes** `none | model_cpu | sequential_cpu` are built in (`Ctx.apply_offload`,
  core.py:239-259: `.to(device)` / `enable_model_cpu_offload` / `enable_sequential_cpu_offload`,
  cuda and mps variants). `Ctx` (core.py:189) is "the firewall handed to an engine's load():
  core helpers only, never backend internals".
- **Validation ladder** in `generate` (core.py:537-567): backend dir present but uninitialised →
  `ERROR: ... unavailable (backend not installed)`; `supports(TYPE)` false → `ERROR: ... does
  not support engine`; engine declares no `TRANSFORMER` → `ERROR: ... not wired for offload` —
  "offload eligibility is a turboCLI-side decision: the (model-agnostic) backend claims every
  engine, but the disk-stream path needs the engine's transformer class to meta-load"
  (core.py:553-555); anything else → `ERROR: unknown offload` listing what IS available. The CLI
  default offload is `offloader`, so a missing backend errors rather than silently degrading.
- **Licensing firewall**: the runner is LGPLv3 (+ private dual license); the backend may be GPL.
  Core reaches it "only by string name + the fixed seam methods, never by importing a backend"
  (core.py:28-30). All GPL-derived code stays in the backend package.

## One generation (`core.generate`)

The **params dict** is flat and string-valued, and identical for CLI and HTTP (the server's
urlencoded fields decode to exactly the dict cli builds): `engine, mode, prompt, images, loras,
output, width, height, seed, inference, renderer, offload, slicing`. The model folder is
deliberately NOT a param — `resolve_model()` derives it from the runner's own path
(core.py:515-517); nothing is path-passed, everything is path-deduced.

1. Validate engine + mode, then the offload ladder above (core.py:525-567).
2. `get_pipe` (core.py:447): compare `_engine_key` to the resident key — `(ID, model dir,
   renderer, offload, slicing) + extra_key(params) + (user loras,)` (core.py:297-308); reuse on
   match, else `release_pipe("config changed")` and reload via the engine's `load()` or
   `_default_load`. Slicing `slice` enables attention slicing + VAE slicing/tiling
   (core.py:492-496). The safety checker is stubbed out (core.py:489-490).
3. **Seed** (core.py:573-578): `-1` → nondeterministic; else `torch.Generator(device="cpu")` —
   a CPU generator regardless of renderer, for device-independent determinism.
4. **image-to-image** (core.py:602-631): lazy PIL import; per input image, downscale-only fit
   into the requested WxH preserving aspect (`scale = min(w/W, h/H, 1.0)`, LANCZOS), then a
   gc + empty_cache "before the heavy lifting".
5. **Progress** (core.py:661-725): diffusers' tqdm bar is kept fully alive but rendered into a
   sink ("a disabled bar computes no stats"), and `progress_bar` is hooked per generation: a
   heartbeat `  0%|step 0/N (00:00)` at bar creation (real loop start, after text-encoding, so
   its 00:00 is truthful), then one line per step emitted *after* `bar.update` so the mirrored
   `format_dict` figures reflect the step just completed. The original method is restored in
   `finally`.
6. **Interruption**: `callback_on_step_end` checks `should_stop()` each step and sets
   `pipeline._interrupt` (core.py:643-654); a partial result is discarded and reported as
   `CANCELLED:` or `SUPERSEDED:` (core.py:754-763).
7. **Backend hooks**: `prepare(pipe)` before the call (per-generation load boundary),
   `reclaim(pipe)` in `finally` (reclaim errors logged, not raised). The call itself runs under
   `torch.inference_mode()` (core.py:735-736).
8. Save, then emit `Saved: <output>` immediately "so the client gets the result as early as
   possible" (core.py:769-770).

**The wire contract** (what callers parse, identical on stdout and the HTTP body):

```
loading <id> model (<renderer> / <offload>)...   informational
model ready
loading input: <path>                            i2i only
generating "<prompt≤60>" (<N> steps)...
  0%|step 0/N (00:00)                            heartbeat
 42%|step 5/12 (00:31, 3.50s/it)                 one line per step, tqdm-native figures
Saved: <path>                                    THE success sentinel (exit 0 / HTTP success)
ERROR: <message or traceback>                    validation or unexpected failure
CANCELLED: stopped on request, server is idle
SUPERSEDED: a newer request took over, this one was cancelled
```

Error-handling split: expected/validation failures emit `ERROR:` and return False; unexpected
exceptions propagate — cli prints the traceback and exits 1, the server logs it and streams it
into the body (cli.py:82-89, server.py:237-246).

## Caches

- **Resident pipeline**: module globals `pipe`/`pipe_key` (core.py:104-107). The key holds
  everything that changes pipe identity (see above), so changing a LoRA or flipping `<sks>` on
  the angles engine reloads. `release_pipe` (core.py:311) calls `backend.release(pipe)` first —
  freeing host-pinned weights, file handles and the offloader↔module cycle — then gc + per-device
  `empty_cache`. The server adds idle release and `/clear`.
- **Prompt-encode cache** (`install_encode_cache`, core.py:343-444): wraps `pipe.encode_prompt`,
  keyed on the call's FULL (args, kwargs); an unhashable argument (tensor/image — e.g. an
  image-conditioned edit encode) makes the call uncacheable so it never false-hits. Entries are
  parked on CPU and moved back on a hit (frees the VRAM they would pin); eviction triggers when
  available host RAM drops below `min(10GB, max(2GB, 10% of RAM))`, worst entry first by
  `1.3**generation_age * bytes` with an LRU tiebreak, never evicting the current generation's
  entries. With an idempotent model loader a cached encode means the text encoder is never
  streamed in at all. Installed only for non-backend pipes; a backend may install its own in
  `prepare()` (core.py:498-501).

## Determinism and reproducibility

- A fixed seed is bit-identical across runs (the CPU generator + the offloader's node-boundary
  work; see the offloader doc and `doc/comfy-krea2-turbo-plan.md` for the war stories).
- Everything is pinned: engine `MODEL["revision"]` / LoRA revisions / SCAFFOLD revisions
  (mutable HF repos → reproducible installs, validated by check), every wheel + the diffusers
  git commit + both repo commits in build.sh, the `.commit` marker in the deployed tree
  (`check.sh` compares it to its own pin — "Also update in check.sh", build.sh:35).

## Platform notes

- **Windows**: Git Bash required; `getPath()` runs `cygpath -w` + `\`→`/` ("Python does not
  handle backslash"); venv activation is `.venv/Scripts` vs `.venv/bin`;
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` "so large VAE decodes fit and avoid the WDDM
  RAM spill" (text-to-image.sh:282-284); LoRA weights parse after the LAST `@` because Windows
  paths contain `:` but never `@` (core.py:127-153); `core.longpaths=true` for clones; the
  installer drops the pipe before touching the HF cache to avoid a permission error
  (install.py:518-519).
- **MPS**: fp16 preferred ("bfloat16 support is patchy"), `PYTORCH_ENABLE_MPS_FALLBACK=1` +
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` set by the wrappers.
- **Run-time dtype** is independent of the install dtype: cuda→bf16, mps→fp16, cpu→fp32
  (`_device_dtype`, core.py:117-124); install.sh coerces a float32 install request to bfloat16.
- **Offline generation**: the wrappers export `HF_HUB_OFFLINE=1` (+ datasets/transformers
  variants) for every run; install is the only online step.

## Notes

- The wrappers' server mode never touches python: `curl -sS -N` with one `--data-urlencode` per
  field, tee the streamed body, grep `^Saved: ` for success (text-to-image.sh:227-267).
- `turboCLI.pro` is a Qt Creator subdirs project listing files for IDE navigation; nothing is
  compiled.
- Doc records: `doc/engine-inheritance-plan.md` (the BASE mechanism),
  `doc/comfy-z-image-turbo-plan.md` (the first ComfyUI-reuse engine),
  `doc/comfy-krea2-turbo-plan.md` (fp8 engines, the `(1 + weight)` RMSNorm trap, per-step parity
  with ComfyUI), `doc/IMPLEMENTATION_PLAN.md` (this document's plan). Script usage blocks live
  in `bash/README.md`, `bash/turbo/README.md`, `bash/python/README.md`.
- The offload backend's internals — vendored ComfyUI subsystem, native vs VBAR paths, CPU
  stream mode, benchmarks — are documented in `turbo-offloader`'s `implementation.md`.
