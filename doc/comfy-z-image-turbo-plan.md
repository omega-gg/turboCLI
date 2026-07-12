# Add `comfy-z-image-turbo` engine — reuse ComfyUI's Z-Image-Turbo models

## Context

turboCLI ships a diffusers-based `z-image-turbo` engine that downloads the full
`Tongyi-MAI/Z-Image-Turbo` HF repo (~20 GB) into `turbo-model/Z-Image-Turbo/`. Users who already run
ComfyUI have the *same* weights on disk, but stored as three split single-file safetensors
(`diffusion_models/z_image_turbo_bf16.safetensors`, `text_encoders/qwen_3_4b.safetensors`,
`vae/ae.safetensors`). We want a new engine, **`comfy-z-image-turbo`**, that *reuses* those ComfyUI
files (no second 20 GB download), downloading only what's missing into ComfyUI's own model hierarchy,
and generates through turboCLI's existing diffusers implementation so `text-to-image comfy-z-image-turbo`
produces a proper image.

Confirmed with the user:
- **Generation backend:** load ComfyUI's single files into turboCLI's own diffusers `ZImagePipeline`
  (self-contained; reuses turboCLI's server, offloader, encode-cache, progress). Not an external
  ComfyUI server.
- **Offload:** support **both** native placement (`none`/`model_cpu`/`sequential_cpu`) **and** the
  turbo-offloader disk-stream path.

### Verified feasibility (deployed diffusers 0.39, `Sky-runtime-bin/gg.omega/turbo/.venv`)
- `ZImageTransformer2DModel` and `AutoencoderKL` both support `from_single_file` — the ComfyUI
  transformer + VAE load directly.
- The comfy→diffusers transformer conversion
  (`diffusers/loaders/single_file_utils.py::convert_z_image_transformer_checkpoint_to_diffusers`) is
  **rename-only + one fused-QKV `torch.chunk`** (chunk returns views), so it is mmap-preserving — the
  offloader can stream it from the single file without materialising 12 GB.
- `ZImagePipeline` needs `scheduler` (FlowMatchEulerDiscreteScheduler), `vae`, `text_encoder`
  (Qwen3-4B `PreTrainedModel`), `tokenizer` (AutoTokenizer), `transformer`. ComfyUI's files carry
  only weights — the tiny **configs + tokenizer + scheduler** (a few MB) must come from the diffusers
  repo. That is the only network fetch when the comfy weights are already present.
- Component download URLs (from ComfyUI's own template): `Comfy-Org/z_image_turbo`,
  `.../resolve/main/split_files/{diffusion_models,text_encoders,vae}/<file>`.

## Reference files (existing, to reuse/mirror)
- Engine recipe pattern: [z_image_turbo.py](turboCLI/runner/engine/z_image_turbo.py) — copy its
  declarations; add a custom `load()` + comfy component/scaffold metadata.
- Engine discovery / load / generate seam: [core.py](turboCLI/runner/core.py)
  (`ENGINES` glob, `resolve_model`, `Ctx`, `_default_load`, `get_pipe`, `generate`).
- Install/check: [install.py](turboCLI/runner/install.py), [check.py](turboCLI/runner/check.py)
  (`_discover`, `_installed`, `.commit` manifest).
- Offloader disk-stream: [offloader/__init__.py](turbo-offloader/offloader/__init__.py) (`load_pipe`,
  lines 119-248) and [offloader/adapter.py](turbo-offloader/offloader/adapter.py)
  (`load_streamed`, `assign_streamed_weights`, `_assign_sd`, `_shards`).
- Canonical Z-Image ComfyUI graph + sampler settings (cfg 1.0, res_multistep, simple,
  ModelSamplingAuraFlow shift 3): [generate.sh](../../dev/test/ComfyUI_windows_portable/generate.sh)
  and `sandbox/aimdo/diffusion/server-comfy.sh`.
- Bash wrapper pattern: [install.sh](turboCLI/bash/turbo/install.sh),
  [check-model.sh](turboCLI/bash/turbo/check-model.sh),
  [text-to-image.sh](turboCLI/bash/turbo/text-to-image.sh).

## Implementation

### 1. Engine recipe — `turboCLI/runner/engine/comfy_z_image_turbo.py` (new)
Auto-discovered by the `engine/*.py` glob. Declares:
- `ID = "comfy-z-image-turbo"`, `TYPE = "z-image"` (reuse the offloader seam vocabulary — same
  pipeline family), `PIPELINE = "diffusers:ZImagePipeline"`,
  `TRANSFORMER = "diffusers:ZImageTransformer2DModel"`, `MODES = ("text-to-image",)`,
  `CFG = ("guidance_scale", 0.0)`, `INFERENCE = 8`.
- `MODEL = {"model": "comfy-z-image-turbo"}` so `resolve_model()` points at
  `turbo-model/comfy-z-image-turbo/` (holds scaffolding + a `comfy.json` manifest, not the big
  weights).
- `COMFY` metadata: base URL `Comfy-Org/z_image_turbo/.../split_files` + per-component
  `{dir, file, role}` for transformer/text_encoder/vae (drives download + presence checks).
- `SCAFFOLD = {"repository": "Tongyi-MAI", "model": "Z-Image-Turbo", "revision": "04cc4abb…"}` — the
  tiny configs/tokenizer/scheduler source.
- A custom **`load(ctx, params)`** hook (core calls `mod.load` when present, so the diffusers-dir
  default path is bypassed). It reads the `comfy.json` manifest for the three single-file paths, then:
  - **Native path (`ctx.backend is None`):** build `ZImagePipeline` from the scaffolding
    (`scheduler`, `tokenizer`) + `ZImageTransformer2DModel.from_single_file(transformer_path)` +
    `AutoencoderKL.from_single_file(vae_path)` + a Qwen3 text-encoder instantiated from the scaffold
    `text_encoder/config.json` with weights loaded from `qwen_3_4b.safetensors`; then
    `ctx.apply_loras`/`ctx.apply_offload`.
  - **Offloader path (`ctx.backend` set):** call the new single-file entry point (step 4) so the
    transformer/text-encoder stream from the comfy files.

### 2. Install — `install-comfy.sh` + `runner/install_comfy.py` (new)
- `turboCLI/bash/turbo/install-comfy.sh`: mirror `install.sh` (same `getSky/getOs/getPath`, venv
  activate, HF env). Signature: `install-comfy <engine> <comfyUI-folder> [dtype]`. Passes
  `--comfy "$comfy"` and `--folder "$folder"` to `python -m runner.install_comfy`.
- `runner/install_comfy.py`: reuses `install._discover()`. For the engine's `COMFY` components:
  1. Resolve ComfyUI models root = `<comfy>/ComfyUI/models`. For each component check
     `models/<dir>/<file>`.
  2. **Reuse** present files; **download** missing ones from the component URL into the correct comfy
     subdir (partial installs supported — only fetch gaps).
  3. Fetch the small `SCAFFOLD` pieces (`model_index.json`, `scheduler/`, `transformer/config.json`,
     `vae/config.json`, `text_encoder/config.json`, `tokenizer/*`) into
     `turbo-model/comfy-z-image-turbo/` (via `hf_hub_download`/`snapshot_download`, config/tokenizer
     files only — no weights).
  4. Write `turbo-model/comfy-z-image-turbo/comfy.json`: the ComfyUI models root + absolute path of
     each component file + its role. This manifest is what generation and check read.

### 3. Check — extend `runner/check.py` (+ `install.py` helper)
Add comfy-aware presence: when an engine declares `COMFY`, "installed" ==
`comfy.json` exists AND every referenced component file exists AND the scaffold dir is present
(instead of the `.commit`/revision test). Add an `_installed_comfy(path)` helper beside `_installed`
in `install.py`; branch on `hasattr(mod, "COMFY")` in both `check.py` and the list path. `check-model.sh`
needs no change (already forwards `--engine`); just add `comfy-z-image-turbo` to its usage list.

### 4. Offloader single-file streaming — `turbo-offloader/offloader/`
`load_pipe`/`load_streamed` today require a diffusers-format component dir (`_shards` +
`model_cls.load_config(dir)` + name-matched `_assign_sd`). Add a single-file-aware variant so the
comfy engine's `load()` can stream:
- New entry point `load_pipe_single_file(components, scaffold_dir, dtype, pipeline_cls,
  transformer_cls, device, lora_files)` (or extend `load_pipe` to accept single-file descriptors).
- A `load_streamed_single_file(model_cls, config_dir, weight_file, converter, dtype, operations)`:
  meta-load `model_cls.from_config(load_config(config_dir))`, mmap the weights via
  `cu.load_torch_file(weight_file)`, apply `converter` (for the transformer:
  `convert_z_image_transformer_checkpoint_to_diffusers` — rename + chunk-views, mmap-preserving),
  then `_assign_sd` to rebind file-sliced tensors. Report `missing` for verification.
- Build the pipeline from the scaffold (`scheduler`, `tokenizer`, config-instantiated text-encoder
  streamed from `qwen_3_4b.safetensors`) + streamed transformer + `from_single_file` VAE (small,
  materialise), keeping `TYPE == "z-image"` so `supports()` and dtype logic are unchanged.
- Because the engine declares `TRANSFORMER`, `core.generate`'s offload gate already accepts it.

### 5. Bash usage lists
Add `comfy-z-image-turbo` to the engine usage lists in `install.sh`, `check-model.sh`,
`text-to-image.sh` (echo blocks only — the wrappers forward `--engine` verbatim, so no logic change).

## Verification (end-to-end)
Deploy the new/changed files into `Sky-runtime-bin/gg.omega/turbo/` (runner engine + install_comfy)
and `.../turbo/backend/offloader/` (adapter), then from `turboCLI/bash/turbo`:
1. `./install-comfy.sh comfy-z-image-turbo C:/dev/test/ComfyUI_windows_portable` — expect it to detect
   the 3 present component files, download nothing big, fetch only the scaffold, and write
   `comfy.json`. (To test the download path, temporarily point `--comfy` at an empty dir and confirm
   it fetches into `models/{diffusion_models,text_encoders,vae}/`.)
2. `./check-model.sh comfy-z-image-turbo` → prints "installed", exit 0; `./check-model.sh list`
   includes it.
3. Native: `./text-to-image.sh comfy-z-image-turbo cuda "a knight in armor" out.png 1024 768 42 8 none`
   → saves `out.png`. Also verify `cpu`.
4. Offloader: same command with `offloader` as the offload arg → confirm the log shows streamed
   transformer weights with **0 missing** (or only expected buffers), and a valid image. Sanity-check
   the result against ComfyUI's own `generate.sh` output at the same prompt/seed/steps.
5. Confirm the existing diffusers `z-image-turbo` engine still installs/checks/runs unchanged (shared
   `install.py`/`check.py` code paths).

## Risks / notes
- **Text-encoder key mapping:** ComfyUI's `qwen_3_4b.safetensors` keys must map onto the diffusers/
  transformers Qwen3 module. `_assign_sd`'s shape+suffix fallback already bridges prefix renames
  (e.g. `model.`); verify 0 missing at step 4 and add a small remap if needed.
- **Offloader `missing` report** is the correctness signal — a non-empty list means a component's keys
  didn't bind; the converter/remap must cover them before shipping.
- **VAE** loads fully (335 MB) on all paths — negligible, no streaming needed.
- Keep the recipe's HF revisions pinned (scaffold + a component revision on the Comfy-Org repo) for
  reproducible installs, mirroring the existing `.commit` discipline.
