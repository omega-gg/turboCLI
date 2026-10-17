# Plan: comfy-flux3-6b, FLUX.3 Image Klein in turboCLI

## Context

Black Forest Labs released FLUX.3 Image Klein on 2026-10-15: open weights, 5.6B, text-to-image
and editing with up to 10 references, a 16-step distilled checkpoint (no CFG) and a 50-step base.
ComfyUI supports it (PR #17064, after the v0.39.1 we vendor) and so does diffusers main (commit
`7ecb1e1`, "Flux3 image (#15046)"). We add it to turboCLI the way the other `comfy-` engines are
built: diffusers model classes, filled from ComfyUI's single files and streamed by turbo-offloader.

Decided with the user:
- **Name `comfy-flux3-6b`**, the size in the name like `comfy-flux2-4b` / `-9b` (5.6B rounded).
  It ships the **distilled** checkpoint; a base engine can follow as `comfy-flux3-6b-base`.
- **No prompt upsampler** in this step (the optional 9.8 GB Qwen3.5 9B rewriter); prompts are used
  as written.
- **Render times compared with ComfyUI** on the same files, as for every comfy engine.

## What it needs

- **diffusers** (turboCLI's pin, `bash/turbo/build.sh` `diffusers_ref`): move to `7ecb1e1` or
  later. That brings `Flux3Transformer2DModel`, the modular `Flux3DistilledAutoBlocks` /
  `Flux3DistilledModularPipeline` (no classic `Flux3Pipeline` exists), `Flux3ImageProcessor` and
  `convert_flux3_transformer_checkpoint_to_diffusers` (`loaders/single_file_utils.py`).
  transformers 5.12.1 already has `Qwen3VLForConditionalGeneration`.
- **turbo-offloader: no change expected.** It holds no model code; int8 ConvRot already loads
  through `load_quant_single_file` (`adapter.py:717`), proven by Qwen-Image 2.1. Anything found
  missing during the work (a hook, a speed layer) is added in `adapter.py` only, never in the
  vendored `offloader/comfy/`.

## Model sources (pinned at implementation time)

From `black-forest-labs/flux-3-image-klein-comfy` (gated), the VAE excepted: it is the FLUX.2
component, shared with `comfy-flux2-4b`.

| Role | File | Size |
|---|---|---|
| transformer | `diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors` | 5.8 GB |
| text encoder | `text_encoders/qwen3vl_4b_int8_convrot.safetensors` | 4.9 GB |
| vae | `vae/flux2-vae.safetensors` | 0.3 GB |

Scaffold (configs, processor, scheduler, no weights): `black-forest-labs/flux-3-image-klein`
(gated), `allow_patterns` on its `model_index.json`, `scheduler/*`, `tokenizer/*` or `processor/*`,
and the `transformer/` / `text_encoder/` / `vae/` `config.json`. One token covers both repos.

## Steps

### 1. Read the checkpoint layout first

With the token, read the two safetensors headers (`huggingface_hub.get_safetensors_metadata`, no
full download) and settle the conversions before writing the engine:
- **Transformer:** prefix (`model.diffusion_model.` or none), and whether attention is fused `qkv`
  (ComfyUI's PR maps LoRAs onto a fused qkv) or split `q/k/v` (what the diffusers converter
  expects). A fused one is split before the converter, each quant companion (`.weight_scale`,
  `.comfy_quant`) moved with its weight, as `comfy_flux2_9b_distilled._transformer_convert`
  (`runner/engine/comfy_flux2_9b_distilled.py:81-111`) already does for FLUX.2.
- **Text encoder:** the key layout against `Qwen3VLForConditionalGeneration`, mapped the way
  `comfy_qwen_image_2_1._text_encoder_convert` (`comfy_qwen_image_2_1.py:292-318`) does for the 8B.

### 2. diffusers pin

Bump `diffusers_ref`, rebuild the venv, and run one image on each installed engine
(flux2-4b, comfy-flux2-4b, comfy-flux2-9b-distilled, z-image-turbo, comfy-z-image-turbo,
comfy-krea2-turbo, comfy-qwen-image-2-1(-turbo)) at a fixed seed: same image as before the bump,
same speed. A regression stops here.

### 3. The engine, `runner/engine/comfy_flux3_6b.py`

Declared like its siblings:
- `ID = "comfy-flux3-6b"`, `TYPE = "flux3"`, `TRANSFORMER = "diffusers:Flux3Transformer2DModel"`,
  `MODES = ("text-to-image", "image-to-image")`, `INFERENCE = 16`, no CFG (the distilled model
  takes none), `IMAGE_AREA = None` (the image processor sizes the references, checked against
  ComfyUI's graph in step 5), `GATED = True` (not inherited, declared here).
- `COMFY` components and `SCAFFOLD` as above, pinned revisions.
- Transformer: meta-built from the scaffold config, `"quant": True`, `convert` from step 1 then
  diffusers' converter. Text encoder: `Qwen3VLForConditionalGeneration` meta-built, `lm_head`
  replaced by an identity, `"quant": True`. VAE: `comfy_flux2_4b._build_vae` as is.
- `load()` raises without the offloader, then `ctx.backend.load_pipe_comfy(Flux3Pipe, ...)`.

**`Flux3Pipe`**, a small classic face on the modular pipeline, in the engine file. The offloader
keys its encode bridge, its TE-to-sampler node boundary and its encode cache on `encode_prompt`
(`turbo-offloader/offloader/__init__.py:291-415`), and `core.generate` reads `result.images`
(`runner/core.py:788`), so:
- it takes `transformer`, `text_encoder`, `tokenizer`, `vae`, `scheduler` like any pipeline and
  keeps them as attributes;
- `encode_prompt(prompt, device)` runs diffusers' `Flux3DistilledTextEncoderStep` alone (layers
  4 to 32, 80-token padding) and returns the embeddings;
- `__call__(prompt, image, height, width, num_inference_steps, generator)` calls
  `self.encode_prompt` (so the offloader's wrappers run), then `Flux3DistilledAutoBlocks` without
  its `prompt_upsampler` and `text_encoder` steps, fed the embeddings, and returns
  `SimpleNamespace(images=...)`.

The 80-token padding is FLUX.3's own (ComfyUI pads the same way), so the offloader's
`install_unpadded_encode` must leave it alone: confirmed in step 4, the engine opting out if not.

### 4. Install, run, parity

- `install.sh comfy-flux3-6b cuda default -1 offloader none <ComfyUI> <token>`, reusing the
  ComfyUI folder's files.
- Text to image and image to image (1 and 2 references) at 1024×1024 and 1024×768, seed 7.
- Same files, seed and sampler in an up-to-date ComfyUI (step 5): the images match (PSNR, as in
  `doc/comfy-qwen-image-2-1-plan.md`).

### 5. Render times against ComfyUI

- Update the local ComfyUI to a build with PR #17064, pinned in the doc, and run its distilled
  FLUX.3 template headless at the same files, size, steps and seed.
- The laptop throttles, so: cool the GPU between runs, interleave turboCLI and ComfyUI, repeat,
  512² and 1024² (`turbo-offloader/doc/BENCHMARKING.md`). Note the torch and comfy-kitchen
  versions of each side; compare on the same torch.
- Report s/it and end to end, cold and warm. Parity or better is the target; a gap is a
  follow-up in `adapter.py` or an engine speed layer (as Qwen-Image 2.1's `_comfy_forward`), not a
  blocker.

### 6. Lists and docs

- `engine.md`: a row in both tables, "FLUX.3 Image Klein 5.6B, 16-step distilled, int8",
  ~11 GB, 16 steps, `gated, offloader`.
- `README.md` "Supported engines": FLUX.3.
- `implementation.md`: the engine notes table and the contract table.
- `bash/turbo/install.sh`, `text-to-image.sh`, `image-to-image.sh` usage lists (with
  `(gated: needs a token)`), their copies in `bash/turbo/README.md`, and the token help naming the
  gated engines.
- `turboCLI.pro`: the engine file and this plan.
- Deployed mirror (`gg.omega/turbo` in the runtime bin) synced; turbopixel picks the scripts up
  through `deploy.sh`.
- This plan copied to `doc/comfy-flux3-6b-plan.md`, with the results (parity, timings) added there.

## Critical files

- `turboCLI/runner/engine/comfy_flux3_6b.py` (new)
- `turboCLI/bash/turbo/build.sh` (diffusers pin)
- References: `runner/engine/comfy_flux2_9b_distilled.py`, `comfy_flux2_4b.py`,
  `comfy_qwen_image_2_1.py`, `runner/install.py` (`_install_comfy`), `runner/core.py`
  (`generate`), `turbo-offloader/offloader/__init__.py` (`load_pipe_comfy`, `_finalize_pipe`)

## Verification

- Every existing engine unchanged after the diffusers bump (image and speed).
- `check-model.sh` lists the engine; `install.sh` without a token fails with the gated message.
- t2i and i2i images correct and matching ComfyUI at the same seed.
- The timing table against ComfyUI in the doc, by the throttling protocol.
- Style: 99 columns, no double or spaced dashes, American spelling, no local paths in the doc.
- No commit until the user says so, no attribution.

## Results

Measured on the RTX A1000 laptop (4 GB), torch 2.14.1+cu130 on both sides, against ComfyUI
v0.41.2 running its distilled FLUX.3 templates headless (no upsampler) on the same three files:
16 steps, euler / simple, seed 7.

### Departures from the plan

- turbo-offloader did change: re-synced to ComfyUI v0.41.2 so both sides run the same comfy code.
- The fused qkv is kept, not split: the speed layer runs it as one GEMM, LoRAs address its thirds.
- The final modulation keeps the file's (shift, scale) order (ComfyUI's LastLayer reads it so).
- No scheduler or tokenizer folders: the modular blocks build their own scheduler and the
  Qwen3-VL processor sits in `text_encoder/`. The scaffold has `modular_model_index.json`,
  which `install.py` now accepts.
- `IMAGE_AREA` is 1024²: ComfyUI's `ReferenceMultiLatent` resizes each reference to 1 MP.

### What the work found

- **diffusers bump** (`da1d3829` to `7ecb1e1`): the seven installed engines render bit-identical
  images at a fixed seed.
- **Checkpoint layout.** The transformer is BFL's own layout (no prefix) with a fused `qkv`, int8
  ConvRot (per-row scales, `.comfy_quant` markers) and a `comfy_attention.config` on the image and
  joint blocks. The text encoder file is already in transformers' layout, its embedding int8 too.
- **turbo-offloader re-synced to ComfyUI v0.41.2** (`comfy/resync.md`), so both sides run the same
  comfy code: five vendored files re-copied plus `configurable.py`; a list of attention configs and
  comfy-kitchen's `sol` attention mirrored; comfy-aimdo 0.5.6. flux2-4b (CUDA, CPU) and
  comfy-krea2-turbo give the same md5 before and after.
- **Speed layer.** diffusers' FLUX.3 transformer ran a modulation GEMM over every token, split
  q/k/v and separate fp32 ops: 2.2x slower per step than ComfyUI at 512². `_comfy_forward` runs
  the diffusers modules as ComfyUI's `Flux3` does (one modulation vector per mode, the file's fused
  qkv, `rms_rope`, SwiGLU folded into the int8 `mlp_out`, `adaln`, in-place gated residuals).
- **Images** keep the composition of ComfyUI's at the same seed (schedule, position ids and noise
  are identical), 25-27 dB at the pixel level; the remaining drift is in the text encoder or the
  VAE implementation, not the sampler.
- **LoRAs** bind as in ComfyUI: `_lora_keys` is its FLUX.3 key map (`model_lora_keys_unet`,
  generic entries and the Flux3 branch) on the diffusers names the convert records. Checked
  against ComfyUI's own map for this file: every weight entry resolves to the same tensor and
  slice (q/k/v LoRAs on thirds of the fused qkv); diffusers' to_q/to_k/to_v names are added.
  A test LoRA makes the same change on both sides; both re-quantize the patched int8 weight,
  with stochastic rounding seeded by the module's name, which differs between the layouts.
- **Edits** follow explicit instructions; short ones barely edit, in ComfyUI too (BFL recommends
  its prompt upsampler).

### Render times, 1024²

Interleaved turbo / ComfyUI, 90 s cool-down between runs, three rounds, a fresh process each
(ComfyUI's server start included). Steady steps are 3 to 16: ComfyUI's bar restarts after its
first step and reports its second at once, so its totals are not comparable step for step.

| | turbo steady | ComfyUI steady | turbo end to end | ComfyUI end to end |
|---|---|---|---|---|
| text to image | 2.69 s/step | 3.00 s/step | 83 s | 87 s |
| edit, one 1 MP reference | 7.19 s/step | 7.19 s/step | 149 s | 152 s |

At 512² (two rounds, the first throttled on both sides) the clean round gives turbo 9 s of
sampling against ComfyUI's 11 s, 31 s against 33 s end to end.

Against comfy-flux2-9b-distilled (4 steps, fp8) on the same 1024² edit, interleaved: 144 s
against 103 s end to end. FLUX.3's steps are twice as fast (7.2 against 14.7 s) but four times
as many; BFL's template suggests 4 to 16 steps.
