# turboCLI, explained simply

This is the plain-English companion to `implementation.md`. It explains what turboCLI does and
why, step by step, starting from zero — no image-diffusion background required. Each section is
a little more technical than the one before. For the full design and mechanisms, read
`implementation.md`; this document is kept up to date with it at every turboCLI iteration.

## What is this?

turboCLI runs image-generation models (FLUX.2, Z-Image, Qwen-Image-Edit, Krea...) from a command
line or behind a small local HTTP server. You install an *engine* once — the only step that
needs the internet — and from then on you generate images fully offline, on a plain CPU, an
NVIDIA GPU (CUDA) or an Apple GPU (MPS):

```
sh text-to-image.sh z-image-turbo cuda "a beautiful knight" out.png
```

It is designed for embedding in client applications: fast cold and warm starts, one simple wire
protocol for both the CLI and the server, and adding a new model in a handful of lines. The
runner itself is LGPL; the heavy memory tricks live in a pluggable, possibly-GPL *backend*
(turbo-offloader) kept behind a tiny interface, so the licenses never mix.

## 60-second primer: how an image gets generated

If you have never looked at image diffusion:

```
 "a cat in the snow"
         │
         ▼
 [ text encoder ]           words → numbers the model can work with ("embeddings")
         │
         ▼
 [ diffusion model ]        starts from pure random noise and removes a bit of it at
         │    × N steps     each step, steered by the embeddings — the heavy part
         ▼
 [ VAE decoder ]            expands the model's compact internal image into pixels
         │
         ▼
     image.png
```

Three neural networks run in sequence; the diffusion model holds almost all of the gigabytes and
often does not fit in the GPU's memory (VRAM). Making it fit anyway — by streaming its weights
just in time — is the backend's job, and `turbo-offloader`'s own `dummy.md` explains that part
in plain English. turboCLI is everything *around* it: the part that turns a one-line command
into a running pipeline.

## The big picture

```
  you, or a host application
                 │
                 ▼
  ┌─────────────────────────────┐   POSIX shell wrappers: pick the environment (offline
  │  bash/turbo/*.sh            │   flags, GPU allocator), cd into the install, then call
  └──────────────┬──────────────┘   python — or just curl the server
                 ▼
  ┌─────────────────────────────┐   two thin front-ends that build the SAME flat params
  │  runner/cli.py  server.py   │   dict: cli.py (one shot, argv) and server.py (resident
  └──────────────┬──────────────┘   HTTP, keeps the model warm)
                 ▼
  ┌─────────────────────────────┐   the shared core: discovers engines and backends,
  │  runner/core.py             │   caches the loaded model, runs one generation
  └──────┬───────────────┬──────┘
         ▼               ▼
  ┌─────────────┐   ┌─────────────────┐   engines: one small declaration file per model
  │ runner/     │   │ backend/<mode>/ │   backend: pluggable weight-placement strategy
  │ engine/*.py │   │ (offloader)     │   (turbo-offloader), behind a tiny fixed interface
  └─────────────┘   └─────────────────┘
```

Everything below the wrappers is one Python package; everything the CLI can do, the server does
through the exact same code path.

## Step by step: from command to PNG

1. **The wrapper prepares the ground.** `text-to-image.sh` validates arguments, exports the
   environment (offline HuggingFace flags — generation never goes online; the right GPU
   allocator settings per platform), cd's into the deployed install and activates its private
   Python environment. With a `server` argument it skips Python entirely and curls the server.
2. **The front-end builds a params dict.** `cli.py` parses the flags into one flat dictionary of
   strings (engine, mode, prompt, size, seed, ...). The HTTP server decodes its request body
   into the *identical* dictionary — one format everywhere.
3. **Backends are found before torch wakes up.** Importing the core first scans `backend/*/`
   folders and initialises each one *before* importing PyTorch — the offloader must install its
   GPU-allocator hooks before torch starts, or it can't at all. Nothing in turboCLI names a
   specific backend; the folder name is the `offload` mode you pass on the command line.
4. **The engine is looked up by name.** Every file in `runner/engine/` was discovered at startup
   (cheap — they contain only declarations). Your `--engine z-image-turbo` selects one.
5. **The model loads — or is already there.** The loaded pipeline stays resident, keyed by
   everything that defines it (engine, renderer, offload mode, LoRAs...). Same key → reuse
   instantly; changed key → release and reload. This is what makes warm runs fast.
6. **The denoise loop runs.** One progress line per step is printed, mirroring the exact timing
   figures of the underlying progress bar (` 42%|step 5/12 (00:31, 3.50s/it)`), so a host app
   can show real progress.
7. **`Saved: out.png`.** That line is *the* success signal: the wrapper, the server client and
   any host application all just look for `Saved:`. Anything else — `ERROR:`, `CANCELLED:`,
   `SUPERSEDED:` — means no image.

## Engines: a model is a page of declarations

Adding a model does not mean writing a loader. An engine is a small Python file of constants —
here is the complete z-image recipe:

```
ID          = "z-image-turbo"
TYPE        = "z-image"
PIPELINE    = "diffusers:ZImagePipeline"
TRANSFORMER = "diffusers:ZImageTransformer2DModel"
MODES       = ("text-to-image",)
CFG         = ("guidance_scale", 0.0)
INFERENCE   = 8
MODEL       = {"repository": "Tongyi-MAI", "model": "Z-Image-Turbo",
               "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021"}
```

Classes are named by *string* and imported only when the engine actually loads, so discovering
every engine at startup costs nothing. A variant declares `BASE = "<other engine>"` and writes
only its delta — the 4-step "lightning" flavour of the Qwen editor is essentially one LoRA
declaration (a LoRA is a small add-on weight file that changes a model's behaviour) plus a step
count. Two families exist:

- **stock** engines download their model once into the install's own `model/` folder;
- **comfy-reuse** engines point at the split model files of an existing ComfyUI installation
  instead, so a ComfyUI user never downloads 40 GB twice.

## The server

Starting a fresh process for every image repeats work — importing torch, discovering engines,
loading and placing the model — so `server.sh start` keeps a resident process that holds the
model warm between requests: back-to-back generations pay none of that again. In plain words:

- `POST /generate` — same fields as the CLI; the reply streams the same progress lines, ending
  in `Saved:` (or not — that IS the failure signal, the HTTP status is always 200).
- **Latest wins** — a new request politely asks the one in progress to stop at its next step and
  takes over; generations never run in parallel on the GPU.
- `POST /cancel`, `POST /clear`, `POST /shutdown`, `GET /health` — stop the current job, drop
  the resident model, stop the server, liveness probe.
- After 10 idle minutes the model is released on its own, giving the machine its memory back.

## Installing and removing

Three stages, from nothing to generating:

1. **`bash/python/build.sh`** — a self-contained Python (no system Python touched).
2. **`bash/turbo/build.sh cpu|cuda|mps`** — deploys turboCLI itself: pinned copies of turboCLI
   and turbo-offloader, a private virtualenv, and the exact pinned versions of every dependency
   — a build six months from now resolves the same stack. Your downloaded models survive
   rebuilds untouched.
3. **`bash/turbo/install.sh <engine>`** — downloads what that engine needs (base model at a
   pinned revision + its LoRAs) and records it in a small per-engine registry file.

Removal is garbage collection: `remove.sh <engine>` drops the engine's registry entry, then
deletes only the files no *other* installed engine still references — and never anything inside
a user's own ComfyUI folder.

## A few technical details worth knowing

Each of these is one sentence here and a full section in `implementation.md`:

- **Backend-before-torch** — backend discovery is deliberately module-level code above
  `import torch`; both front-ends import the core only after argument parsing to preserve that.
- **The seam is a licensing firewall** — the LGPL runner reaches the (possibly GPL) backend only
  by folder name and a fixed set of functions, never by importing it directly.
- **The pipe cache key** — engine, model dir, renderer, offload, slicing, engine extras and user
  LoRAs; anything that changes model identity triggers a clean reload, nothing else does.
- **The prompt-encode cache** — repeated prompts skip the text encoder entirely; entries are
  parked in RAM and evicted under memory pressure.
- **Determinism** — a fixed seed uses a CPU random generator regardless of the renderer, so the
  same seed gives the same image on any device.
- **Everything is path-deduced** — model and engine folders derive from the runner's own
  location; no caller ever passes a path to them.

## Where to go next

- `implementation.md` — the full design document: every mechanism and its rationale.
- `doc/engine-inheritance-plan.md`, `doc/comfy-z-image-turbo-plan.md`,
  `doc/comfy-krea2-turbo-plan.md` — how the engine features were designed and validated.
- `bash/README.md` — usage blocks for every script.
- turbo-offloader's `dummy.md` and `implementation.md` — the offloading story, plain and deep.

## Keeping this document up to date

This file describes the current architecture and is revised with each turboCLI iteration:
whenever behaviour documented in `implementation.md` changes, the matching plain-English section
here changes with it. If the two ever disagree, `implementation.md` is right and this file has
a bug.
