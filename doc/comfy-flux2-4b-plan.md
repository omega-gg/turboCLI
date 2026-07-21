# Add `comfy-flux2-4b` engine — reuse a ComfyUI install's FLUX.2-klein-4B files

## Context

turboCLI ships a diffusers-based `flux2-4b` engine that downloads the full
`black-forest-labs/FLUX.2-klein-4B` HF repo (~15 GB: 7.3 GB transformer + 7.5 GB text encoder +
161 MB VAE) into `turbo-model/FLUX.2-klein-4B/`. ComfyUI stores the same model as split single-file
safetensors. We want **`comfy-flux2-4b`**: the same pipeline, built from a ComfyUI install's files,
so a user who already runs ComfyUI does not re-download the big weights.

This mirrors the existing `z-image-turbo` → `comfy-z-image-turbo` pair exactly, and lands as the
same shape of engine: one auto-discovered module, a `COMFY` component list, a tiny `SCAFFOLD`, and
a `load()` that supports both the native path and the offloader disk-stream path.

### The reuse win (verified, not assumed)

FLUX.2-klein-4B's text encoder **is** Qwen3-4B — the very file ComfyUI already has on disk for
Z-Image, `models/text_encoders/qwen_3_4b.safetensors`. Compared byte-for-byte against
`FLUX.2-klein-4B/text_encoder/model.safetensors`:

```
comfy keys 398   klein keys 398
only comfy: []   only klein: []   shape mismatches: 0
```

All 398 keys carry the `model.` prefix, which is exactly what `Qwen3ForCausalLM` expects — so
unlike `comfy-z-image-turbo` (which targets the bare `Qwen3Model` and must strip the prefix),
**this engine needs no text-encoder key conversion at all**. `tie_word_embeddings: True`, so the
absent `lm_head.weight` is correct, not missing.

On a box that already ran the ComfyUI Klein workflow, that is 7.5 GB reused for free.

### Verified feasibility (deployed diffusers 0.40.0.dev0, `Sky-runtime-bin/gg.omega/turbo/.venv`)

- `Flux2KleinPipeline`, `Flux2Transformer2DModel`, `AutoencoderKLFlux2` all present.
- `Flux2KleinPipeline.__init__` args: `scheduler, vae, text_encoder, tokenizer, transformer,
  is_distilled` — same shape as `ZImagePipeline` plus `is_distilled`.
- `Flux2Transformer2DModel` **is** in `SINGLE_FILE_LOADABLE_CLASSES`, mapped to
  `convert_flux2_transformer_checkpoint_to_diffusers` (`default_subfolder: transformer`). The
  converter is rename-only plus fused-QKV splits, i.e. the same mmap-preserving shape the offloader
  already streams for Z-Image. **No hand-written converter needed** (unlike `comfy-krea2-turbo`).
- `AutoencoderKLFlux2` is **not** in `SINGLE_FILE_LOADABLE_CLASSES` — see the VAE decision below.

### Component sources

ComfyUI's own `image_flux2_klein_text_to_image` template names the files; each was confirmed to
exist on HF:

| role | ComfyUI path | repo / file |
|---|---|---|
| transformer | `diffusion_models/flux-2-klein-4b.safetensors` | `black-forest-labs/FLUX.2-klein-4B`, root-level `flux-2-klein-4b.safetensors` |
| text_encoder | `text_encoders/qwen_3_4b.safetensors` | `Comfy-Org/z_image_turbo`, `split_files/text_encoders/…` |
| vae | `vae/flux2-vae.safetensors` | `Comfy-Org/flux2-dev`, `split_files/vae/…` |

The transformer single file was probed remotely at the pinned revision:

```
tensors 149   dtypes {BF16: 149}   quant/scale keys 0
double blocks 5   single blocks 20        (= klein-4B topology, BFL key layout)
```

Two consequences that keep this engine lean:

1. It is plain **bf16**, not scaled-fp8 — so no `"quant": True`, no comfy quant path, and the
   engine is **not** offloader-only the way `comfy-krea2-turbo` is. Both the native and disk-stream
   paths work.
2. It is the **distilled** klein (`is_distilled: true`, 4 steps, guidance 0) — identical sampling
   to the existing `flux2-4b`, so `CFG`/`INFERENCE` carry over unchanged. (The fp8 variants
   `FLUX.2-klein-base-4b-fp8` / `FLUX.2-klein-4b-fp8` exist and ComfyUI's *edit* template uses the
   base one at 20 steps / CFG 5 — deliberately **not** what we mirror here, since it is a different
   model with a different sampling profile from `flux2-4b`.)

Note the transformer lives at the **root** of the BFL repo, not under `split_files/`, so its
component sets `filename` explicitly — the `comfy-krea2-turbo` idiom. It is pinned to
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, the same revision `flux2-4b` already pins, so both
engines are reproducibly the same weights.

### VAE

`AutoencoderKLFlux2` is **not** in `SINGLE_FILE_LOADABLE_CLASSES`, so `from_single_file` is
unavailable. That initially read as "reusing ComfyUI's VAE needs a hand-written converter" — it
does not. Comparing ComfyUI's `flux2-vae.safetensors` (`Comfy-Org/flux2-dev`) against the
diffusers klein VAE:

```
comfy 251 tensors   diffusers 251 tensors
exact key overlap: 251        shape multisets identical: True
```

The ComfyUI file is **already in diffusers key layout**. So the missing mapping only rules out
`from_single_file`, not reuse: `_build_vae` instantiates `AutoencoderKLFlux2` from the scaffold
`vae/config.json` and `load_state_dict`s the ComfyUI file directly (`strict=True`), the same shape
as `_build_text_encoder`. `.to(dtype)` is safe — it skips the integer `bn.num_batches_tracked`
buffer.

This keeps the engine true to the comfy-engine premise: **all three big files come from the
ComfyUI install**, and the scaffold stays weight-free (`vae/config.json`, not `vae/*`).

### Install cost summary

| | fresh box | box that already ran ComfyUI's Klein workflow |
|---|---|---|
| transformer | 7.75 GB | reused |
| text encoder | 7.5 GB | reused |
| VAE (`flux2-vae.safetensors`, fp32) | 336 MB | reused |
| scaffold (configs + tokenizer, weight-free) | 16 MB | 16 MB |

Measured on a box that had already run ComfyUI's Z-Image workflow: the transformer downloaded, the
text encoder reported `Component present` (0 bytes), and the scaffold was 16 MB.

> **Note for `SCAFFOLD.allow_patterns` changes:** `snapshot_download` does not prune files that no
> longer match, so narrowing the patterns leaves orphans in `engine/<id>/` from earlier installs
> (dropping `vae/*` left a stale 161 MB `vae/diffusion_pytorch_model.safetensors` behind). Delete
> them by hand, or reinstall into a clean engine dir.

## Reference files (existing, to mirror)

- Closest template — same bf16 / no-quant / both-paths shape:
  [comfy_z_image_turbo.py](../runner/engine/comfy_z_image_turbo.py).
- Stock twin, for the declarations: [flux2_4b.py](../runner/engine/flux2_4b.py).
- Root-level `filename` idiom + multi-repo components:
  [comfy_krea2_turbo.py](../runner/engine/comfy_krea2_turbo.py).
- Engine contract: [engine/\_\_init\_\_.py](../runner/engine/__init__.py) — **no torch/diffusers
  import at top level**.
- Install/check dispatch (`hasattr(mod, "COMFY")`): [install.py](../runner/install.py)
  (`_install_comfy`), [check.py](../runner/check.py).
- Offloader seam: `load_pipe_comfy` in [offloader/\_\_init\_\_.py](../../turbo-offloader/offloader/__init__.py)
  — `convert` is read with `.get()`, so the text-encoder spec simply omits it.

## Implementation

### 1. Engine — `runner/engine/comfy_flux2_4b.py` (new)

Auto-discovered by the `engine/*.py` glob; no registry edit. Declares:

- `ID = "comfy-flux2-4b"`, `TYPE = "flux2"` (share the offloader seam vocabulary with `flux2-4b`),
  `PIPELINE = "diffusers:Flux2KleinPipeline"`,
  `TRANSFORMER = "diffusers:Flux2Transformer2DModel"`,
  `MODES = ("text-to-image", "image-to-image")`, `CFG = ("guidance_scale", 0.0)`, `INFERENCE = 4`
  — all carried over from `flux2-4b`.
- `COMFY` — the two reused single files (transformer with explicit root-level `filename`, text
  encoder from `Comfy-Org/z_image_turbo` under the default `split_files/` layout).
- `SCAFFOLD` — `black-forest-labs/FLUX.2-klein-4B` at the pinned revision, `allow_patterns` =
  `model_index.json`, `scheduler/*`, `tokenizer/*`, `transformer/config.json`,
  `text_encoder/config.json`, `vae/*`.
- Helpers: `_by_role()` (verbatim from `comfy_z_image_turbo`), `_transformer_meta()`,
  `_text_encoder_meta()`, `_build_text_encoder()` — the last three targeting
  `Flux2Transformer2DModel` / `Qwen3ForCausalLM`, with **no key remap** on the TE.
- `load(ctx, params)`:
  - **Disk-stream** (`ctx.backend is not None`): hand `load_pipe_comfy` the transformer spec
    (`meta` + `file` + `convert=convert_flux2_transformer_checkpoint_to_diffusers`) and the TE spec
    (`meta` + `file`, no `convert`), plus prebuilt `scheduler`/`tokenizer`/`vae`/`is_distilled`.
  - **Native**: `Flux2Transformer2DModel.from_single_file(...)` +
    `_build_text_encoder(...)`, assemble `Flux2KleinPipeline`, then `ctx.apply_loras` +
    `ctx.apply_offload`.

Because `is_distilled` is a `Flux2KleinPipeline.__init__` arg rather than a module, it is passed
through the `components` dict on the offloader path.

### 2. Hand-maintained satellites (per `implementation.md` — these rot silently)

1. `turboCLI.pro` → add `runner/engine/comfy_flux2_4b.py` to `OTHER_FILES` (alphabetical).
2. `bash/turbo/install.sh` usage engine list.
3. `bash/turbo/text-to-image.sh` and `bash/turbo/image-to-image.sh` usage lists (engine declares
   both modes, so both scripts list it).
4. `bash/turbo/README.md` — mirrors those usage blocks verbatim.
5. The engine table in `implementation.md`.

`check-model.sh` / `remove.sh` are deliberately list-free — do not touch.

### 3. Deployed mirror

`implementation.md` requires keeping `Sky-runtime-bin/gg.omega/turbo/runner/engine/` in step with
the source tree; copy the new module there.

## Validation

1. `check-model.sh` lists `comfy-flux2-4b` as not-installed (registry discovery works).
2. `install.sh comfy-flux2-4b --comfy C:/dev/test/ComfyUI_windows_portable/ComfyUI` — expect
   `Component present: text_encoders/qwen_3_4b.safetensors` (the reuse win, no download) and a
   download for the transformer only.
3. `text-to-image.sh comfy-flux2-4b "…"` on both the native and offloader paths.
4. Parity against `flux2-4b` at a fixed seed — same weights, same revision, so outputs should match
   modulo load-path numerics (see `text-to-image.sh` seed arg 7 for determinism).
