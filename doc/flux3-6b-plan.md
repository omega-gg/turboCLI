# Plan: flux3-6b, FLUX.3 Image Klein from the diffusers repo

## Context

turboCLI ships each model twice where it can: a stock engine on the diffusers repo (`flux2-4b`)
and a `comfy-` twin on ComfyUI's files (`comfy-flux2-4b`), with similar images and speed.
FLUX.3 Image Klein has only the twin so far, `comfy-flux3-6b` (BFL's int8 ConvRot ComfyUI files,
committed in turboCLI `5d70577`). This adds `flux3-6b` on `black-forest-labs/flux-3-image-klein`
(gated), the 16-step distilled checkpoint as published: **bf16** (decided with the user), so the
two engines share one model and one fast path, the stock one streaming twice the bytes.

What the repo holds (revision `337337621b8526e120f2275d250242f4a6bedd75`, read from its headers):
- `transformer/`: 2 shards, 11.1 GB bf16, diffusers layout, **split** q/k/v, `norm_out` in
  diffusers' (scale, shift) order.
- `text_encoder/`: 2 shards, 8.9 GB bf16, Qwen3-VL-4B (`model.language_model.*`, `model.visual.*`,
  no `lm_head`), plus the Qwen3-VL processor files.
- `vae/`: AutoencoderKLFlux2, diffusers layout (0.3 GB).
- Root: `modular_model_index.json` only (no `model_index.json`), no `scheduler/` or `tokenizer/`;
  also BFL-native copies (`model.safetensors`, `ae.safetensors`) and `variants/` (fp8r, nvfp4,
  11.5 GB), none of which we need.

Why it is not a pure declaration like `flux2-4b`: diffusers has no classic FLUX.3 pipeline
(`ModularPipeline.from_pretrained` loads no weights), and the offloader's dir-based `load_pipe`
has no hook for the speed layer. So the engine has its own `load()`, like the comfy engines, and
reuses `comfy_flux3_6b`'s pieces.

## Design

### Engine `runner/engine/flux3_6b.py` (new, short)

Imports `comfy_flux3_6b as cf` (as `comfy_flux3_6b` imports `comfy_flux2_4b`) and declares:
- `ID = "flux3-6b"`; `TYPE, PIPELINE, TRANSFORMER, MODES, CFG, INFERENCE, IMAGE_AREA` from `cf`;
  `GATED = True`.
- `MODEL = {"repository": "black-forest-labs", "model": "flux-3-image-klein", "revision": ...,
  "allow_patterns": ["modular_model_index.json", "transformer/*", "text_encoder/*", "vae/*"]}`
  (20.3 GB; skips the root BFL copies and `variants/`).
- `load(ctx, params)`, offloader-only like `comfy-flux3-6b` (raises without a backend):
  tokenizer `Qwen3VLProcessor` from `model/text_encoder`; VAE
  `AutoencoderKLFlux2.from_pretrained(model, subfolder="vae").to(ctx.dtype).eval()` (as
  `comfy_flux2_4b._build_vae` casts); then `ctx.backend.load_pipe_comfy(cf.Flux3Pipe, ...)` with
  the shard lists as `"file"`:
  - transformer: `meta = cf._transformer_meta(model, d, comfy, fused_qkv=False)`,
    `convert = _transformer_convert`, `lora_keys = cf._lora_keys(model, names)`;
  - text encoder: `meta = cf._text_encoder_meta(model, d)` (same model, same layout, no convert).
- `_transformer_convert(sd)`: one line of work. The speed layer reads the final modulation in
  the file's (shift, scale) order (ComfyUI's LastLayer), so `norm_out.linear.weight` goes back
  through diffusers' own `swap_scale_shift` (it is its own inverse).
- LoRAs as in ComfyUI: `names` = {BFL key: diffusers key} for every weight, from a short table of
  BFL's names (the repo's root `model.safetensors`: `single_blocks.N.q_proj` for
  `single_transformer_blocks.N.attn.to_q`, `mlp_in` for `ff.linear_in`, ...). `cf._lora_keys`
  then gives ComfyUI's map, exactly what ComfyUI builds for a split-qkv FLUX.3 file;
  diffusers-named LoRAs map through the offloader's generic entries (`add_lora`), as on
  `flux2-4b`. The table is checked offline: diffusers' converter maps every BFL name back to its
  diffusers key, 359/359.

### Shared code `runner/engine/comfy_flux3_6b.py` (small edits)

- `_transformer_meta(scaffold, dtype, comfy, fused_qkv=True)` passes it to `_comfy_forward`.
- `_comfy_forward(model, comfy, fused_qkv)`: ComfyUI's `Block` has both forms (`fused_qkv` in
  comfy/ldm/flux3/model.py, chosen by the file): fused, the blocks get `attn.to_qkv` as today;
  split, they keep `to_q/to_k/to_v` and `block_forward` runs the three GEMMs. Everything else
  (one modulation vector per mode, rms_rope, adaln, SwiGLU folded into `linear_input_act`, which
  takes plain bf16 weights too) is shared.

### turbo-offloader: shards for the comfy assembly (small)

`load_pipe_comfy`'s `"file"` may be a list of files, the shards of one model:
- `adapter.stream_single_file`: mmap each file with `load_torch_file`, merge the state dicts,
  `_offloader_files` = all of them (fast_disk);
- `load_pipe_comfy._fits_full_load`: sum every file's size.
Docstrings say "a single file or a model's shards". No change to the quant path (int8 files are
single files).

### `runner/install.py` (small)

- Stock snapshot uses `MODEL["allow_patterns"]` when the engine gives it, else today's
  `["model_index.json", "*/*"]`.
- One `_has_index(folder)` (`model_index.json` or `modular_model_index.json`) for `base_ok`,
  the stock branch of `_engine_installed` and its comfy branch (which inlines it today).
- A concrete `--dtype` re-saves through `PIPELINE.from_pretrained`, which only declaration
  engines support: an engine with its own `load()` keeps the weights as published, with a one-line
  notice (hosts may pass `bfloat16`, which is what FLUX.3 ships).

### Lists and docs

- `engine.md`: a `flux3-6b` row in both tables ("FLUX.3 Image Klein 5.6B, 16-step distilled
  (diffusers weights, bf16)", 20 GB, 16, `gated, offloader`), next to `comfy-flux3-6b`.
- `bash/turbo/install.sh` (with `(gated: needs a token)`), `text-to-image.sh`,
  `image-to-image.sh` usage lists; their copies in `bash/turbo/README.md`; the token help naming
  the gated models.
- `implementation.md`: the engine notes row; the `IMAGE_AREA`/`MODEL` contract row gains
  `allow_patterns`. turbo-offloader `implementation.md` / `README.md`: `load_pipe_comfy` takes
  shards.
- `turboCLI.pro`: the engine file and this plan. Plan copied to `doc/flux3-6b-plan.md`, results
  added there. Deployed mirror synced.

## Critical files

- `turboCLI/runner/engine/flux3_6b.py` (new), `runner/engine/comfy_flux3_6b.py`,
  `runner/install.py`
- `turbo-offloader/offloader/adapter.py` (`stream_single_file`), `offloader/__init__.py`
  (`load_pipe_comfy`)
- Reused: `cf.Flux3Pipe`, `cf._comfy_forward`, `cf._text_encoder_meta`, `cf._lora_keys`,
  diffusers' `swap_scale_shift`, the offloader's `load_pipe_comfy` / `wire_streamed` / `add_lora`.

## Verification

- Offline first: meta-build with `fused_qkv=False` against the shard headers, 359/359 keys bind,
  shapes match; the BFL name table round-trips through diffusers' converter; `_lora_keys` covers
  ComfyUI's generic map for split qkv.
- `install.sh flux3-6b cuda default -1 offloader none none <token>`: downloads 20.3 GB only;
  `check-model.sh` lists it installed and gated; re-running install does not re-download.
- comfy-flux3-6b still renders the same md5 (the shared code moved only behind a flag).
- Images: `flux3-6b` against `comfy-flux3-6b`, t2i at 1024² and the knight edit, seed 7: same
  composition, PSNR at the int8-vs-bf16 level; a synthetic LoRA makes the same change on both.
- Speed, by the throttling protocol (interleaved, cool-downs, repeated, 512² and 1024²): same
  per-step rate as `comfy-flux3-6b` where the compute dominates; the extra streaming of bf16 on
  the 4 GB card measured and reported, not hidden.
- Style: 99 columns, no `--`, spaced ` - ` or em dash in added lines, American spelling, no local
  paths in the doc. No commit until the user says so, no attribution, no pin bumps.

## Results

Measured on the RTX A1000 laptop (4 GB), torch 2.14.1+cu130, ComfyUI v0.41.2, seed 7, 16 steps.

- **Install**: 19 GB (the three diffusers folders and the modular index); a second install
  finds it in place and downloads nothing. `check-model.sh` lists it installed and gated.
- **Offline**: the split-qkv model binds the repo's 359 transformer tensors, shapes matching;
  the table of BFL names round-trips through diffusers' converter and equals the names in BFL's
  ComfyUI file, 359/359.
- **comfy-flux3-6b unchanged**: the same image md5 after the shared code took `fused_qkv`.
- **Images against comfy-flux3-6b** (bf16 against int8 weights): text to image at 1024²,
  21 dB, the same composition; the knight edit at 1024², 46.7 dB. A test LoRA makes the same
  change on both engines.

### Render times

Interleaved, 90 s cool-down between runs, two rounds, a fresh process each. Steady steps are
3 to 16; ComfyUI runs BFL's bf16 single file (the repo's root `model.safetensors`, the same
weights as `transformer/`) with its int8 text encoder.

| t2i | flux3-6b | ComfyUI, bf16 file | comfy-flux3-6b (int8) |
|---|---|---|---|
| 512², steady | 1.21 s/step | 1.21 s/step | 0.79 s/step |
| 1024², steady | 4.47 s/step | 4.79 s/step | 3.07 s/step |
| 1024², end to end | 120 s | 117 s (server start included) | 83 s |

flux3-6b runs the bf16 weights at ComfyUI's speed or better. Against comfy-flux3-6b it is about
1.5 times slower per step on this 4 GB card, the cost of bf16 itself (twice the bytes to stream,
no int8 matmul), which ComfyUI pays the same way. The edit: 10.8 against 7.2 s/step.
