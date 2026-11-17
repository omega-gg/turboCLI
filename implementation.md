# turboCLI implementation

LGPL runner for diffusion image generation: two thin front-ends (`cli.py` one-shot, `server.py`
resident HTTP) over one shared core, engines as **pure declaration modules** discovered at
startup, and weight-placement strategy behind a **string-named backend seam** (`backend/<mode>/`,
external, may be GPL — e.g. `turbo-offloader`). The design goals are fast cold and warm starts,
one code path for CLI and server, and adding a model in a handful of declaration lines.

**Style:** code and comments wrap at 99 columns.

**New here?** `dummy.md` is the plain-English introduction — what turboCLI does and why, step by
step, no diffusion background required. It is kept up to date with each iteration of this doc.
To use turboCLI rather than read about it, see `MANUAL.md`.

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
    engine/          one declaration module per engine (+ _*.py helpers, not engines): the mask /
                     mask-apply / mask-birefnet / mask-lucida / mask-inspyrenet "compute" engines
                     live here too, over the _mask / _apply / _segment / _options helpers
  bash/
    python/          build.sh / check.sh -- bundled standalone CPython + uv
    turbo/           build/install/remove/check/check-model/server/text-to-image/image-to-image/
                     image-to-mask/image-apply-mask
  backend/           EMPTY in-repo (a .gitignore placeholder); build.sh grafts the offloader here
  doc/               plan docs, kept as records after implementation
  test/
    run.sh           the suite: same images, no slower, per engine or all (test/README.md)
    comfy.py         its ComfyUI side: a template graph per comfy engine, PSNR
    modal.sh         the suite on Modal GPUs, one container per engine at once
    modal_app.py     the Modal app modal.sh starts: the image, the CPU and GPU containers
    comfyui.sh       ComfyUI as the image holds it, on turboCLI's torch
    reference/       each machine's md5s and timings, committed
    results/         a run's images, logs and summary (not committed)
    old/             the suite's input images, and a README for the mask checks by hand
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
3. **`bash/turbo/install.sh <engine> <renderer> [dtype] [inference] [offload] [slicing]
   [ComfyUI dir] [token]`** → `python -m runner.install` — the only ONLINE step (exports `HF_HOME`,
   `hf-transfer`); generation itself runs with `HF_HUB_OFFLINE=1`. The renderer and the three
   options after the dtype download nothing: they are recorded in the engine's registry entry so
   a host can offer them back, see `install.py` below. The token (for a gated repo, e.g. the FLUX.2
   klein 9B ones) is exported as `HF_TOKEN` for that run only: it never reaches the Python argv
   or the record. A gated download without it ends in one actionable message (`_gated_repo`),
   not a traceback.

`bash/turbo/check.sh` (turboCLI installed?) and `check-model.sh` (engine installed? `list`) are
the machine contracts a host app polls: fixed one-line outputs + exit code 0/1, both torch-free
and venv-free so they run under the bundled python. `check-model SETTINGS:<engine>` and
`ENGINES:<mode>` answer in several lines — `key: value` per recorded setting, plus `comfy:
<folder>` for a comfy engine installed inside an existing ComfyUI (so a reinstall passes it again
rather than moving the engine into our own folder), `<id> installed|absent` per engine of the
mode, plus `gated` when its download needs a token (`GATED`) so a host asks before it installs —
which is why they are queries of their own rather than more output on the installed check.

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
inside the base-reinstall branch (`install.py:561-562`).

- **Registry**: `engine/<id>/engine.json` is the source of truth for "installed" — model name,
  pinned revision, LoRA list; for a comfy engine its component refs + scaffold source
  (`install.py:40-50`). No per-model-dir markers: a shared dir's revision is derived from any
  engine record referencing it (`_model_revision`, install.py:177). A comfy record whose
  components (role and path) no longer match the engine's `COMFY` is not installed, so an update
  that swaps a file is installed again rather than loading the old one (`_engine_installed`).
- **Run settings ride the record**: the renderer, dtype, inference, offload and slicing the
  install was given land in `settings`, written by `_write_engine` whatever record shape it is
  handed — so every path records them and a reinstall re-assigns them, download or not. They
  download nothing; `check-model SETTINGS:<engine>` reads them back for a host, and a record
  written before the block reads as `SETTINGS_DEFAULT` (`settings()`, install.py:71-90). The
  dtype is a *wish* once a base is present: `base_ok` skips the re-download, so nothing is
  recast and install says to `remove` first.
- **Stock install** (`install.py:655-760`): `--dtype default` copies the pinned repo as
  published into `model/<name>` (`snapshot_download` of `model_index.json` + the component
  folders, or the engine's `MODEL["allow_patterns"]`, no cast, no load: root-level single files
  and legacy formats skipped); a concrete dtype instead runs
  `from_pretrained(repo, revision=pin, torch_dtype=...)` → `save_pretrained` (the re-cast copy),
  for a declaration engine only: one with its own `load()` keeps the published files. Either way `model/<name>` is a self-contained canonical copy. Then fetch the
  engine's declared LoRAs into `model/<name>/lora/` and trim the HF cache for everything pulled —
  disk holds one copy, not cache+copy (install.py:747-760). Idempotent and selective: `base_ok`
  skips the base re-download when the registry already records the pinned revision, and only
  missing LoRAs are fetched (install.py:533-546) — installing `-lightning` over an existing base
  pulls just the LoRA.
- **Comfy install** (`_install_comfy`, install.py:237-324): reuse each COMFY component already
  under the ComfyUI install's `models/`, `hf_hub_download` only the missing ones *into that
  tree*, fetch the tiny SCAFFOLD (configs/tokenizer/scheduler, no weights) into `engine/<id>`,
  and write engine.json with `external` flagging weights outside our model folder. `external` is
  *diagnostic only* — nothing reads it; what actually protects a user's ComfyUI is `_gc`'s live
  `_under(path, default_folder())` guard, recomputed from the real path. Without `--comfy` a
  self-contained `model/ComfyUI/` layout is used - so a reinstall that forgets the folder moves the
  engine there and downloads every component again, which is why `check-model SETTINGS:` hands
  the folder back as `comfy:`. The scaffold is pruned after the fetch
  (`_prune_scaffold`) so narrowing `SCAFFOLD.allow_patterns` cannot leave orphans behind.
- **Removal and reinstall share one GC** (`_referenced` + `_gc`): delete what a record referenced
  that the registry no longer names — only models / LoRAs / comfy components no surviving engine
  references, and only files under our model folder. `_gc` is the single place expressing that
  policy. The two callers differ only in *when* they ask: `_remove` after dropping the entry (so
  `_registry()` = survivors), `_write_engine` after writing the new record (survivors *plus* it,
  so a plain reinstall never GCs what it just downloaded). Hooking `_write_engine` is what closes
  the old hole where a component dropped from `COMFY` vanished from the registry and could never
  be found again — see `doc/reinstall-prune-plan.md`.

### `check.py` — verifier

Reuses `install._discover` + `install._engine_installed`. "Light by design: no torch/diffusers
import, so it runs under the bundled python without the venv" (check.py:33-35).

### Mask engines — generate + apply as compute engines (image-to-mask / image-apply-mask)

Masks fold into the engine system as **compute engines**: engines that declare a
`run(ctx, params, emit) -> image` hook instead of a diffusers pipeline. `core.generate`
short-circuits to `run` right after mode validation — before the offload ladder and `get_pipe` —
so a mask call never touches the diffusion machinery (no offloader, no `progress_bar`) and leaves
the resident pipe untouched. `"run"` is in `_inherit._INHERITED`, so a variant inherits it. Two
modes, six engines:

- **`image-to-mask`** generates an 8-bit grayscale mask/matte. `mask` (torch-free pixel-diff or
  grown-box masks via `mode=default|region`, `_mask.build_mask`) diffs an edit against its
  reference; the three matte engines segment a subject — `mask-birefnet` + `mask-lucida` via
  transformers' `AutoModelForImageSegmentation`, `mask-inspyrenet` via `transparent-background`'s
  `Remover`; all in `_segment`, torch imported inside the functions. A second image is an optional
  clean **plate** whose cast shadow is recovered by luminance diff and merged in (`build_matte`).
- **`image-apply-mask`** (engine `mask-apply`, `_apply.apply`) lays a mask onto an image:
  `composite` pastes the masked region onto a reference (restore the scene, or a new backdrop);
  `putalpha` writes the mask as the alpha channel (an RGBA cutout). The mask source is irrelevant,
  so a diff mask or a model matte feeds either op. `trim` (`_apply.trim`) takes no mask: it crops
  an image to the pixels that show, keeping at most `pad=N` pixels of transparency around them
  (32 by default). `speck=N` leaves out separate blobs under N pixels, the specks a noisy mask
  scatters (0 by default: off, since a particle could matter), see `doc/image-trim-plan.md`. An
  image where nothing shows is saved whole and logs `apply[trim]: nothing shows`, so a caller
  can drop it (turbopixel does for a selection).
  `putalpha` and `trim` leave every fully transparent pixel blank white, `(255, 255, 255, 0)`
  (`_apply._clear`): a cutout no longer carries the picture it was cut from, a reader that drops
  the alpha sees the subject on white, and the PNG is about 7x lighter.

Inputs ride the comma-separated **`images`** param (ordered), the input always first:
`input,reference` for mask, `input[,plate]` for the matte engines, `input,mask[,reference]` for
apply. Per-call scalars ride a general-purpose **`--options`** field (`key=value,...`): `cutoff=N`
(0-255, higher = fewer pixels/shadow — the mask change threshold / matte shadow floor), `mode`
(`default|region` for mask, `composite|putalpha` for apply), `glow=N` (pixels, `mask`
`mode=default` only, how much thicker the mask gets) and `fade=<l>:<t>:<r>:<b>` (pixels, `mask`
either mode, how far each border of a crop softens; the parser splits on commas before it
reads `=`, so a value that carries several numbers separates them with colons). The engine modules
stay torch-free at
import (heavy imports live in `run()`), so discovery and the diffusion path never load the
segmentation stack — the load-bearing zero-regression guarantee.

**Install** (`python -m runner.install <engine>`): `mask` / `mask-apply` are
**register-only** (they own no model — install just writes the engine.json record so check/remove
is uniform). The matte engines add a MODEL `"kind"`: **`snapshot`** (the whole HF repo verbatim,
incl. `trust_remote_code`, into `model/<name>` — birefnet `ZhengPeng7/BiRefNet`, lucida
`egeorcun/lucida`) and **`url`** (a raw GitHub release asset — inspyrenet's `ckpt_base.pth`, pinned
by tag not commit). `mask-lucida` BASE-inherits `mask-birefnet`'s `run` and only swaps the MODEL.
The reference-counted GC is unchanged (it keys off the model-dir name). The turbo venv gains
`timm`/`einops`/`kornia`/`transparent-background` (`bash/turbo/build.sh`); uv keeps the pinned
cu130 torch, so the diffusion stack is unperturbed.

This replaces the old standalone `runner/mask.py`/`apply.py` and the `bash/remove-background` tool,
and is byte-exact with them: `mask`→`mask-apply composite` reproduces the old fused merge and a
matte→`putalpha` the old cutout (verified via `cmp`, including through the server).

**Benchmark** (RTX A1000 laptop; BiRefNet always infers at 1024², so input size barely matters):
matte inference ~0.45s on cuda (half) / ~19s on cpu; a full `image-to-mask mask-birefnet` call is
~7s (cuda) / ~24s (cpu), dominated by per-process ~4.5s torch import + ~1–2s model load.
`mask-birefnet`/`mask-lucida` share those numbers; `mask-inspyrenet` is slower (~2s cuda
inference). The torch-free engines (`mask` / `mask-apply`) are ~1.7s wall-clock —
almost all python + PIL startup, the pixel work is milliseconds. So a cutout is the ~7s matte call
+ ~1.7s apply; a mask/region restore is two ~1.7s calls. Prefer cuda; cpu is a slow fallback.

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
| `MODES` | wire modes tuple: `"text-to-image"` / `"image-to-image"` / `"image-to-mask"` / `"image-apply-mask"` |
| `CFG` | one `(kwarg, value)` pair injected into the pipe call (core.py:599-600); the value may be a dict, one per mode (`by_mode`: comfy-moss-image's templates differ) |
| `INFERENCE` | default step count when the caller passes `-1` (fallback 4, core.py:586-587); a dict too, one per mode |
| `MODEL` | install spec `{repository, model, revision}` (stock); optional `kind` = `snapshot`\|`url` picks a non-diffusers install (verbatim HF repo / raw asset); optional `allow_patterns` narrows the diffusers snapshot (default `model_index.json` + every folder; `flux3-6b` takes its components and `modular_model_index.json` only); it only *renames* the folder: omitted, `resolve_model` falls back to `ID` (core.py:173), which is why COMFY engines declare none |
| `COMFY` | ComfyUI-reuse spec `{repository?, revision, components: [{role, path, ...}]}`; presence dispatches both install and `resolve_model` (core.py:177) |
| `SCAFFOLD` | tiny config-only snapshot spec (`allow_patterns`, no weights) |
| `LORAS` | install-time LoRA list `[{repository, file, revision}]` |
| `GATED` | `True` when the download needs a Hugging Face token: `check-model ENGINES:` adds `gated`. Declared by each gated engine, not inherited |
| `loras(params)` | runtime preset hook → `[(filename, weight)]`, may be prompt-dependent |
| `extra_key(params)` | extra tuple folded into the pipe cache key (core.py:297-301) |
| `load(ctx, params)` | full custom loader, bypasses `_default_load` |
| `run(ctx, params, emit)` | **compute engine**: owns generation, returns a PIL image; short-circuits before the offload ladder / pipeline (core.py:537-549) — mask / matte / apply engines |

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

**Two families on purpose, the best of both.** `comfy-*` engines are for going fast on a
reasonable computer: ComfyUI's own models (`load_pipe_comfy`), its quantized files and its
offloading, at ComfyUI's speed or better. Stock engines (no `comfy-` prefix) are for powerful
hardware: the published diffusers models, full precision, run natively (`none` / `model_cpu` /
`sequential_cpu`) or through the offloader's `load_pipe` (its diffusers-side path, kept for
them); they also keep turboCLI working without ComfyUI. So stock engines never run ComfyUI's
models: never port one onto them to share code or speed, and never remove `load_pipe` or its
diffusers-side helpers. flux3-6b was once moved onto ComfyUI's FLUX.3 and made offloader-only,
and had to be rebuilt on diffusers. Sharing goes one way only: a comfy engine may build on its
stock sibling (comfy-flux3-6b subclasses flux3-6b's `Flux3Pipe`, comfy-moss-image
moss-image's `MOSSPipe`), never the reverse.

**Adding an engine — keep the hand-maintained satellites in sync.** Discovery itself is automatic
(drop the module in `engine/`), but four things are written by hand and rot silently otherwise:

- **`turboCLI.pro`** lists every runner file in `OTHER_FILES`, and `bash/*.pri` lists the scripts —
  a new module missing there ships in no package. Keep the deployed copy in step too.
- **The usage engine lists** in `bash/turbo/install.sh` (everything installable) and the mode
  wrappers `text-to-image.sh` / `image-to-image.sh` / `image-to-mask.sh` (each mode-filtered).
- **`bash/turbo/README.md`**, which mirrors each script's usage block verbatim — update both or
  they drift.
- **This table**, below, and **[engine.md](engine.md)**, the user-facing list: the engine's
  line under each task it does, with its size, steps and requirements.

`check-model.sh` and `remove.sh` deliberately carry *no* list: `runner.check` with no `--engine`
enumerates the registry, so they point at it instead and cannot go stale. Prefer that whenever the
answer is derivable.

| engine | notes |
|---|---|
| `flux2-4b` | TYPE `flux2`, Flux2KleinPipeline, t2i+i2i, 4 steps; pure declaration |
| `flux2-9b` | BASE = flux2-4b; delta: the FLUX.2-klein-9B repo (Qwen3-8B TE, ~35 GB bf16). **Gated** (FLUX Non-Commercial License): install takes an HF token |
| `flux3-6b` | FLUX.3 Image Klein from BFL's diffusers repo, its weights as published (bf16: transformer and Qwen3-VL-4B TE in shards, the FLUX.2 VAE), ~20 GB, `MODEL.allow_patterns` skipping the repo's root BFL copies and fp8/nvfp4 variants. **Gated**. A stock engine: diffusers' own models, with or without the offloader. diffusers ships FLUX.3 as a modular pipeline only, so `Flux3Pipe` (a `DiffusionPipeline` with `Flux3LoraLoaderMixin`) gives it the classic face: `from_pretrained` over the repo's components, an `encode_prompt` on diffusers' Qwen3-VL (the offloader wraps it), a `progress_bar` the runner hooks, the step callback for cancel, `.images`. `load()` takes the offloader's `load_pipe`, as the other stock engines, or `from_pretrained` + `apply_loras` + `apply_offload` without it. comfy-flux3-6b builds on its pipe (`doc/flux3-6b-plan.md`) |
| `z-image-turbo` | TYPE `z-image`, ZImagePipeline, t2i, 8 steps; pure declaration (the README's "sample engine": ~9 lines) |
| `moss-image` | MOSS-Image SFT (~5B) from OpenMOSS's repo, its weights as published (bf16 transformer, Qwen3.5 9B TE, the FLUX.2 small VAE in float32), ~27 GB, t2i only (OpenMOSS publishes no edit model). A stock engine, with or without the offloader. Its pipeline is the repo's own `pipeline_moss_image.py` (Apache-2.0), imported from the pinned install (`official`); `MOSSPipe` keeps its components, encode and decode and gives it the runner's loop: a `progress_bar` the runner hooks, the step callback for cancel, the official CFG in float32 with the uncond pass skipping block 22. Seams for comfy-moss-image: the skip window (`skip_sigmas`), `negative_prompt`, `_decode` and `_before_sampling`. 50 steps, CFG 3, as the README runs it; on the L40S with the offloader, 33.4 s of sampling at 1024² (`doc/moss-image-plan.md`) |
| `qwen-image-edit-2511` | TYPE `qwen-image-edit`, QwenImageEditPlusPipeline, i2i, `true_cfg_scale 1.0`, 40 steps |
| `qwen-image-edit-2511-lightning` | BASE = the above; delta: 4 steps + the lightning LoRA (install `LORAS` + runtime `loras()`) |
| `qwen-image-edit-2511-lightning-angles` | BASE = lightning; adds the angles LoRA for `<sks>` prompts; `extra_key` = the `<sks>` flag so flipping it reloads the pipe |
| `comfy-z-image-turbo` | same weights as z-image-turbo from ComfyUI's 3 single files. The transformer and the Qwen3-4B text encoder are ComfyUI's own models, which the offloader runs behind the pipeline's calls (`{model: "lumina2", file, config}`, `{model: "lumina2", file}`), the files loaded as they are; `ZImagePipe` encodes through ComfyUI's CLIPTextEncode. The VAE is diffusers' `AutoencoderKL` on ComfyUI's `ae.safetensors`. Offloader-only. Against ComfyUI at 1024² (seed 42, euler): 49.7 dB, sampling at parity (54-58 s against 52.5-58 s) |
| `comfy-flux2-4b` | same weights (and pinned revision) as flux2-4b from ComfyUI's single files: the bf16 DiT (root-level `filename`) and the Qwen3-4B TE, the *same file* comfy-z-image-turbo reuses, so it is often already on disk. The transformer and the text encoder are ComfyUI's own models, which the offloader runs behind the pipeline's calls (`{model: "flux2", file, config}`, `{model: "flux2", file}`), the files loaded as they are; `Flux2Pipe` encodes through ComfyUI's CLIPTextEncode. ComfyUI's `flux2-vae.safetensors` is already in diffusers key layout (251/251 keys), so `_build_vae` loads it straight onto a config-built `AutoencoderKLFlux2`. Offloader-only. Against ComfyUI at 1024² (seed 42): 48.5 dB, sampling 13 s (21 s) |
| `comfy-flux2-9b` | BASE = comfy-flux2-4b; delta: the files (and pinned revision) of flux2-9b as ComfyUI single files: bf16 DiT (`flux-2-klein-9b`, root-level `filename`) + bf16 Qwen3-8B TE (`qwen_3_8b`) + the shared `flux2-vae`, ~35 GB. The offloader tells the 8B TE from the 4B by its width, as ComfyUI does. **Gated** (DiT + scaffold): install takes an HF token |
| `comfy-flux2-9b-distilled` | BASE = comfy-flux2-9b (scaffold, pipeline); delta: its files + `_build_vae`. ComfyUI's distilled Klein 9B template, file for file: fp8 DiT (`flux-2-klein-9b-fp8`) + fp8-mixed Qwen3-8B TE, loaded by ComfyUI's own models as they are, + BFL's small decoder VAE (`full_encoder_small_decoder`, decoder widths 96/192/384/384, same encoder and latents; BFL layout, remapped by diffusers' `convert_ldm_vae_checkpoint`), ~18 GB; offloader-only. Against ComfyUI at 1024² (seed 42): 48.3 dB, sampling at parity (25.0 s against 25.5 s). **Gated** (DiT + scaffold): install takes an HF token |
| `comfy-flux3-6b` | FLUX.3 Image Klein (5.6B), its 16-step distilled flow-map checkpoint (no CFG): text-to-image + image-to-image as ComfyUI's distilled templates, file for file: int8 ConvRot DiT + int8 ConvRot Qwen3-VL 4B TE (both ComfyUI's own models on the offloader) + the FLUX.2 VAE shared with comfy-flux2-4b, ~11 GB. **Gated** (FLUX Kommunity License, both BFL repos). its pipeline is flux3-6b's `Flux3Pipe`, the classic face over diffusers' modular FLUX.3 (`Flux3DistilledAutoBlocks`), encoding through ComfyUI's text encoder: an `encode_prompt` the offloader wraps (encode bridge, node boundary, cache; FLUX.3's own 80-token padding is kept, `install_unpadded_encode` only touching an (embeds, mask) pair), a `progress_bar` the runner hooks, the step callback for cancel, `.images`. The transformer is ComfyUI's own model, which the offloader runs behind the pipeline's calls (`{model: "flux3", file, config}`), its `comfy_attention.config` keys keeping ComfyUI's int8 attention. LoRAs bind as in ComfyUI (its FLUX.3 key map, q/k/v LoRAs on thirds of the fused qkv), and by their diffusers names too (`_lora_names`, BFL's table inverted). References at their own size (the pipeline's, up to 4096²), not ~1 MP as `ReferenceMultiLatent` does. No prompt upsampler yet (BFL recommends one: short edit prompts barely edit, in ComfyUI too). The scaffold is a modular repo (`modular_model_index.json`, which `_engine_installed` accepts). Needs diffusers 7ecb1e1+; offloader-only (`doc/comfy-flux3-6b-plan.md`) |
| `comfy-qwen-image-edit-2511` | 39 GB bf16 DiT + scaled-fp8 Qwen2.5-VL TE, ComfyUI's own models on the offloader (`{model: "qwen_image", ...}`); components span three Comfy-Org repos; the VAE is ComfyUI's own (`comfy_vae`). The edit is ComfyUI's TextEncodeQwenImageEditPlus: each image seen by the TE at about 384² with the prompt and VAE-encoded at about 1024² as a reference (the transformer's `edit`), on the stock `QwenImagePipeline`'s denoise. **Offloader-only**. Not yet run on ComfyUI's models (files not installed here) |
| `comfy-qwen-image-edit-2511-lightning` | BASE = the above; entire delta = one extra COMFY component (the LoRA) + 4 steps |
| `comfy-krea2-turbo` | both DiT and TE scaled-fp8, both ComfyUI's own models, which the offloader runs behind the pipeline's calls (its "krea2" model), the files loaded as they are; the pipeline encodes through ComfyUI's text encoder as `CLIPTextEncode` does (`_pipeline_class`), unpadded, and LoRAs bind through ComfyUI's own key map (every published naming); the VAE is ComfyUI's own (the offloader's `comfy_vae`); offloader-only; **Gated** (only the scaffold's krea/Krea-2-Turbo, the Comfy-Org weights are open); the scaffold is the scheduler and configs (`doc/comfy-krea2-turbo-plan.md`) |
| `comfy-krea2-turbo-edit-identity` | BASE = the above, image-to-image; delta = conradlocke's Identity Edit V2 LoRA as a COMFY component (1.83 GB, ComfyUI-native keys; prepended at 1.0) + `Krea2EditPipe` on the base's pipeline: the size snapped down to the stock call's 16 grid, which it requires, the references to the offloader's text encoder with the prompt (the node pack's grounded encode) and to its transformer (`edit`: the pack's pixel fit, VAE encode and edit forward), the subject beside the target by default (`REF_POSITION`, since `aligned` holds the source's pose) and boosted by 2; the denoise is stock (`doc/comfy-krea2-turbo-edit-identity-plan.md`) |
| `comfy-krea2-turbo-edit-general` | BASE = the above; delta = the General Edit V2 LoRA in place of the identity one (same repo and revision) and the node pack's General Edit presets handed to the base's `load`: `ref_boost` 4, the reference aligned, so what the instruction does not touch keeps its place |
| `comfy-qwen-image-2-1` | Qwen-Image 2.1 (7B, generates and edits with the same weights): text-to-image + image-to-image on `QwenImage21Pipeline`, ComfyUI's three templates file for file: int8 ConvRot DiT + int8 ConvRot Qwen3-VL 8B TE (comfy-kitchen int8 kernels) + the RGBA VAE, ~17.3 GB, ungated. The transformer is ComfyUI's own model, which the offloader runs behind the pipeline's calls (`{model: "qwen_image21", file, config}`), its prefix K/V cache included, so sampling matches ComfyUI's speed; TE flat to nested with an `Identity` lm_head; the VAE is ComfyUI's own Wan 2.2 model, opted in through the offloader's `comfy_vae`, so it decodes as ComfyUI does (1024² whole on a 4 GB card, in row strips). Scheduler pinned to ComfyUI's shift 0.69 + simple, 25 steps; RGBA output (PNG). References sized to the output's area, not ~1 MP (`load()`'s `Pipe` sets `output_resolution`): a reference of the output's size stays as it is, since another size shifts the edit (ComfyUI's `TextEncodeQwenImage21` says so); under 288² a whole factor (2x...) keeps it above the Qwen3-VL processor's 256² minimum, whose enlarging would no longer match the latents. Needs diffusers da1d3829+; offloader-only (`doc/comfy-qwen-image-2-1-plan.md`) |
| `comfy-qwen-image-2-1-turbo` | BASE = the above on Qwen-Image-2.1-Turbo, Qwen's 8-step distillation (CFG 1); entire delta = the transformer component swapped for Comfy-Org's int8 ConvRot Turbo file (same repo, revision-pinned) + `INFERENCE = 8`. TE, VAE and scaffold are the base's (Turbo's transformer config is identical); the schedule is Turbo's own: `load()` wraps the base's and samples `SAMPLE_SIGMAS` unshifted, as Qwen's Turbo `model_index.json` and scheduler config set it and ComfyUI's Turbo templates pass it (`ManualSigmas`), so it always runs those 8 steps (diffusers ignores a requested step count once `sample_sigmas` is set). The base's shifted "simple" schedule left grain: fine-grain energy 11.7 against ComfyUI's 8.8, now 8.8 and 33.4 dB from ComfyUI's render. Its file sets ComfyUI's int8 attention on blocks 1-31, which ComfyUI's own model honors |
| `comfy-moss-image` | MOSS-Image SFT as ComfyUI's two templates run it, file for file: int8 ConvRot DiT + Qwen3.5 9B w4a8 TE (both ComfyUI's own models on the offloader, `{model: "moss_image", ...}`) + the FLUX.2 small VAE, ComfyUI's own too (`comfy_vae`, fp16 where ComfyUI picks it), and Comfy-Org's edit LoRA for image-to-image, ~13.3 GB, ungated. Its pipeline is moss-image's `MOSSPipe`, encoding and decoding through ComfyUI's models, sampled by ComfyUI's own sampler (the offloader's `comfy_sampler`, bit-identical to ComfyUI's on CUDA). An edit is chaotic: a 1% difference in its first step's inputs (diffusers' VAE in bf16 against ComfyUI's in fp16) became another picture after 30 steps, ComfyUI's own included, so the reference is ComfyUI's encode exactly; its keys and values are computed before sampling, as `extra_conds` does (`_before_sampling`). Each mode is its template: t2i 25 steps, CFG 2.5, er_sde + simple, block 22 skipped for the first half; i2i 30 steps, CFG 4, euler + normal, block 22 skipped throughout, a fixed negative prompt, the source at ~1 MP as the reference (its keys and values computed once, the transformer's `edit`), the LoRA. `INFERENCE` and `CFG` are per mode, `extra_key` reloads the pipe when the mode changes (only edits carry the LoRA). No prompt upsampler. By design, CFG combines the velocities as the official loop does (ComfyUI the denoised, equal but for rounding), and an edit renders at the requested size (the template's latent takes the scaled source's). Against ComfyUI on the L40S at 1024² (seed 42): t2i 40.4 dB, sampling 9.17 s (9.28 s); i2i 30.8 dB, 11.17 s (11.13 s). Offloader-only (`doc/moss-image-plan.md`) |
| `mask` | **compute engine** (declares `run`, not a pipeline); mode `image-to-mask`; torch-free pixel-diff / grown-box mask (options `mode=default\|region`); register-only install; options `cutoff=N` (0-255), `glow=N` (pixels, `default` only: the mask grows by about N, graded, which covers the outline a removal leaves where `region` would have boxed the whole footprint. It does not replace `region`, see `doc/mask-glow-plan.md`) and `fade=<l>:<t>:<r>:<b>` (pixels per border, either mode: the borders of a crop ramp to 0 over that many pixels, never more than a quarter of a side, since a grown box fills a crop to its edge, or a change runs off it, and the caller pads it back into a seam. 0 keeps a border hard and the caller owns the distance, see `doc/mask-fade-plan.md`). Both are off unless sent |
| `mask-birefnet` / `mask-lucida` / `mask-inspyrenet` | compute matte engines, mode `image-to-mask`; MODEL `kind` snapshot (birefnet/lucida) or url (inspyrenet); `mask-lucida` BASE = `mask-birefnet`; optional plate keeps the cast shadow; options `cutoff=N` (shadow floor) |
| `mask-apply` | compute engine, mode `image-apply-mask`; applies a mask via `composite` or `putalpha`, or trims an image down to what shows via `trim`; register-only; options `mode=composite\|putalpha\|trim`, `pad=N` and `speck=N` for `trim` |

## The backend seam (runner side)

- **One package scan** — diffusers and transformers each run `importlib.metadata`'s
  `packages_distributions` as they import (~1.7 s on Windows); the top of core.py caches it, so
  one scan serves both (core.py:42-45).
- **Discovery precedes `import torch`** — module-level code at the top of core.py
  (core.py:46-80): every `backend/<mode>/__init__.py` under cwd is a candidate; the dir name IS
  the offload-mode string; `pre_torch_init()` must run before torch initialises (the offloader's
  CUDA allocator hooks can't install afterwards); a failed init is skipped silently. Nothing in
  turboCLI names a specific backend.
- **Seam methods**: `pre_torch_init() / available() / supports(engine_type) /
  load_pipe(model, dtype, pipeline_cls, transformer_cls, device, lora_files) /
  load_pipe_comfy(...) / comfy_vae(...)` plus optional per-generation
  `prepare(pipe)`
  (core.py:727-733),
  `reclaim(pipe)` (core.py:744-752) and `release(pipe)` (core.py:311-327). Class objects flow
  runner→backend, resolved lazily from the engine's declared strings, "so the class table lives
  here, not duplicated in the backend" (core.py:281-289). The backend stays model-agnostic;
  comfy engines name ComfyUI's models as data (model names + file paths + LoRA name maps).
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
options, output, width, height, seed, inference, renderer, offload, slicing`. The model folder is
deliberately NOT a param — `resolve_model()` derives it from the runner's own path
(core.py:515-517); nothing is path-passed, everything is path-deduced. `loras` is
comma-separated `<path>@<weight>` entries; `parse_loras` accepts any finite float (no `[0,1]`
clamp — diff-style patches use >1 and negative, matching ComfyUI).

1. Validate engine + mode, then the offload ladder above (core.py:525-567). An image-to-image
   run's `align=<n>` option must name one of its inputs, checked before any model loads.
2. `get_pipe` (core.py:447): compare `_engine_key` to the resident key — `(ID, model dir,
   renderer, offload, slicing) + extra_key(params) + (user loras,)` (core.py:297-308); reuse on
   match, else `release_pipe("config changed")` and reload via the engine's `load()` or
   `_default_load`. Slicing `slice` enables attention slicing + VAE slicing/tiling
   (core.py:492-496). The safety checker is stubbed out (core.py:489-490).
3. **Seed** (core.py:573-578): `-1` → nondeterministic; else `torch.Generator(device="cpu")` —
   a CPU generator regardless of renderer, for device-independent determinism.
4. **image-to-image** (core.py:638-651): lazy PIL import; each input image goes to the
   pipeline as it is, which sizes it its own way (Qwen and Flux2 to ~1 MP, Flux3 only past
   4096²).
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
   `reclaim(pipe)` in an outer `finally`, after the save and `Saved:` (so the client does not wait
   on it, ~0.2 s) yet on every path (reclaim errors logged, not raised). The same `finally` then
   collects garbage as ComfyUI does after a prompt (`_collect`: gc + `empty_cache`, at most every
   10 s); the edit used to collect before sampling, which delayed its result ~0.8 s on an L4 (the
   emptied cache also made the first step reallocate). The call itself runs under
   `torch.inference_mode()` (core.py:735-736).
8. **Align** (image-to-image with `align=<n>`, `runner/align.py`, torch free): the zoom and
   shift an engine adds to its result (Qwen-Image 2.1 does, by a different amount for each
   seed) are measured against input `n` and undone. Edge maps of both, so a relit frame still
   measures, matched in 64 px tiles by phase correlation on a copy of at most 1 MP; a zoom and
   shift per axis through the tile shifts, outliers dropped, fitted 3 times on the result warped
   back. The warp (Lanczos, the edge repeated) is applied only past 0.75 px at the corners and
   under a 3% zoom (past it the zoom was asked for, "zoom out" or a camera that moves), and kept
   only if it takes a fifth off the tile shifts: less, and the engine redrew the frame. One
   `align:` line says which; any failure is logged and the result saved as it came. 0.1 to 0.4 s
   on the CPU. OpenCV 5's `phaseCorrelate` windows its inputs in place, so the overlapping tiles
   go as copies (`doc/image-align-plan.md`).
9. Save (PNG at ComfyUI's SaveImage `compress_level=4`, lossless), then emit
   `Saved: <output>` immediately "so the client gets the result as early as possible"
   (core.py:769-770).

**The wire contract** (what callers parse, identical on stdout and the HTTP body):

```
loading <id> model (<renderer> / <offload>)...   informational
model ready
loading input: <path>                            i2i only
generating "<prompt≤60>" (<N> steps)...
  0%|step 0/N (00:00)                            heartbeat
 42%|step 5/12 (00:31, 3.50s/it)                 one line per step, tqdm-native figures
align: zoom <x>% x <y>%, ..., aligned            i2i with align only, informational
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

## Tests

`test/run.sh` turns that determinism into a check for every update of turboCLI or
turbo-offloader. It runs each installed engine, stock and comfy, in each of its modes at seed 42,
and compares the image md5 and the sampling time with this machine's reference
(`test/reference/<renderer>-<device>-<size>.txt`, committed, `--record` to write it). Images are
only identical within one machine and stack, hence a reference per machine. Timings come from the
runner's own lines (`model ready`, `generating`, `step 0/N`, `step N/N`, `Saved:`), each stamped
on arrival to the millisecond by perl: load, encode, sampling and decode, sampling deciding (a
mask engine has no steps; its time, mostly start-up, is shown, not judged). Each run also prints
its time from launch to `Saved:`, where a client takes the image, not to the process's exit.
`--cool` and `--repeat` keep a throttling laptop from failing a run that is not slower.

`--comfy <dir>` also runs each comfy engine through a ComfyUI install (`test/comfy.py`: the
template graph, its files from the engine's `COMFY` declaration), alternating with turbo, each
repeat swapping which goes first, so neither always reads the files cold. ComfyUI's time runs
from its server's launch to the image it returns. It compares the two; `--comfy-only <dir>` runs
ComfyUI alone, for a re-sync of the vendored ComfyUI.
ComfyUI's progress bar shifts its start after the second step (`model_trange`), so its sampling
is timed by the same stamps, from the bar's `0/N` to its `N/N`. `test/README.md` has the usage
and `doc/test-suite-plan.md` the design.

`test/modal.sh` runs it on Modal L40S GPUs, one container per engine at once. Its results,
turbo against ComfyUI engine by engine, are in turbo-offloader's `implementation.md`
(Benchmarks, the L40S machine), with what to read them with.

## Platform notes

- **Windows**: Git Bash required; `getPath()` runs `cygpath -w` + `\`→`/` ("Python does not
  handle backslash"); venv activation is `.venv/Scripts` vs `.venv/bin`;
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` "so large VAE decodes fit and avoid the WDDM
  RAM spill" (text-to-image.sh:282-284); LoRA weights parse after the LAST `@` because Windows
  paths contain `:` but never `@` (core.py:127-153); `core.longpaths=true` for clones; the
  installer drops the pipe before touching the HF cache to avoid a permission error
  (install.py:588-589).
- **MPS**: fp16 preferred ("bfloat16 support is patchy"), `PYTORCH_ENABLE_MPS_FALLBACK=1` +
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` set by the wrappers.
- **Run-time dtype** is independent of the install dtype: cuda→bf16, mps→fp16, cpu→fp32
  (`_device_dtype`, core.py:117-124); install.sh coerces a float32 install request to bfloat16.
  The matte engines skip that seam - `AutoModelForImageSegmentation` keeps the dtype of the
  checkpoint, and BiRefNet ships fp16 - so `_segment.birefnet_alpha` halves on cuda and calls
  `net.float()` everywhere else, a cpu conv refusing a half bias against a float input.
- **Offline generation**: the wrappers export `HF_HUB_OFFLINE=1` (+ datasets/transformers
  variants) for every run; install is the only online step.

## Notes

- The wrappers' server mode never touches python: `curl -sS -N` with one `--data-urlencode` per
  field, tee the streamed body, grep `^Saved: ` for success (text-to-image.sh:227-267).
- `turboCLI.pro` is a Qt Creator subdirs project listing files for IDE navigation; nothing is
  compiled.
- Doc records: `doc/engine-inheritance-plan.md` (the BASE mechanism),
  `doc/comfy-z-image-turbo-plan.md` (the first ComfyUI-reuse engine),
  `doc/comfy-flux2-9b-plan.md` (the klein 9B engines: gated install token, fp8 key companions,
  the small decoder VAE),
  `doc/comfy-krea2-turbo-plan.md` (fp8 engines, the `(1 + weight)` RMSNorm trap, per-step parity
  with ComfyUI), `doc/image-mask-plan.md` + `doc/image-mask-split-plan.md` (the earlier standalone
  mask tool) then `doc/mask-engines-plan.md` (folding mask/matte/apply in as engines) then
  `doc/mask-tolerance-plan.md` (the `cutoff` knob + the mask/region merge),
  `doc/mask-glow-plan.md` (the `glow` thickening, and why it does not retire `region`),
  `doc/mask-fade-plan.md` (the soft border a cropped `region` mask needs),
  `doc/image-trim-plan.md` (the `trim` of image-apply-mask, a cutout cut down to what shows),
  `doc/image-area-plan.md` (each input image keeps its ratio; `IMAGE_AREA` since removed),
  `doc/image-align-plan.md` (the `align` option of image-to-image, an edit put back in place),
  `doc/moss-image-plan.md` (the stock and comfy MOSS-Image pair, ComfyUI's sampler and
  templates),
  `doc/IMPLEMENTATION_PLAN.md` (this document's plan). Script usage blocks live
  in `bash/README.md`, `bash/turbo/README.md`, `bash/python/README.md`.
- The offload backend's internals — vendored ComfyUI subsystem, native vs VBAR paths, CPU
  stream mode, benchmarks — are documented in `turbo-offloader`'s `implementation.md`.
