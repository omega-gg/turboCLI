# Plan: Qwen-Image 2.1 in turboCLI (comfy-qwen-image-2-1)

## Context

Qwen-Image 2.1 (Alibaba, 2026-09-21) is a 7B single-stream DiT that generates AND edits with the
same weights (up to 10 reference images, native 2K, RGBA output). ComfyUI supports it natively
since v0.37.0 and its three official templates (text to image, image edit, background removal) all
load the same int8 files. turboCLI adds it as ONE ComfyUI-reuse engine, `comfy-qwen-image-2-1`,
declaring both text-to-image and image-to-image, reusing the template files, with images and
performance on par with ComfyUI.

Decisions: one engine for both modes (one pipeline, one set of weights); the int8 template
weights (~17.3 GB, fits a small card best).

What it needed:
- diffusers had no Qwen-Image 2.1 at the old pin. Upstream added `QwenImage21Pipeline`,
  `QwenImage21Transformer2DModel` and `AutoencoderKLQwenImage21` in 6256aa76; the pin moves to
  da1d3829 (2026-10-05, the newest commit touching them). That diffusers requires
  huggingface-hub >= 1.32, so huggingface_hub moves 1.21.0 -> 1.33.0 and hf_xet 1.5.1 -> 1.7.0.
  transformers 5.12.1 already has Qwen3-VL.
- The offloader's vendored ComfyUI ops already load `int8_tensorwise` with ConvRot and
  comfy-kitchen 0.2.37 runs the int8 kernels on CUDA. One small offloader addition,
  `kitchen_ops()`, hands an engine ComfyUI's fused building blocks (see Speed).

## Model sources (pinned)

| role | ComfyUI file (Comfy-Org/Qwen-Image-2.1 @ cb504a40, plain paths) | size |
|---|---|---|
| transformer | `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (649 keys) | 7.26 GB |
| text encoder | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (Qwen3-VL 8B) | 9.35 GB |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` (Wan 2.2 layout, 2D, RGBA head) | 0.68 GB |
| scaffold | `Qwen/Qwen-Image-2.1` @ d26bb612: model_index, scheduler/, processor/, configs | ~16 MB |

Ungated (license "qwen-research"), no token. The optional prompt enhancers stay unused (off in
the templates).

## The engine (`runner/engine/comfy_qwen_image_2_1.py`, offloader-only)

- `PIPELINE` QwenImage21Pipeline, both `MODES`, `CFG = ("true_cfg_scale", 1.0)` (no negative
  prompt, so no CFG, as ComfyUI's cfg 1), `INFERENCE = 25` (the template), `IMAGE_AREA = None`
  (the pipeline sizes references to ~1 MP itself).
- Transformer: the ComfyUI file already uses diffusers names except the fused SwiGLU `gate_up`.
  The engine keeps it fused (see Speed), so the file binds as is: 649/649 keys, no conversion.
- Text encoder: Qwen3-VL meta-built from the scaffold with an `Identity` lm_head (the pipeline
  reads pre-norm hidden states only); flat ComfyUI keys move under `model.language_model.`, the
  vision tower stays, the unused head is dropped. Its `embed_tokens` is int8 ConvRot too.
- VAE: diffusers has no converter for the Wan 2.2 layout, so a small rename table maps it
  (residual indices, middle/head, down/up blocks and their resamplers) and drops the unit time
  axis of the 3D kernels; strict load.
- Scheduler pinned to ComfyUI's ModelSamplingFlux(shift 0.69) + "simple":
  `base_shift = max_shift = 0.69` (a fixed mu, independent of resolution) and no terminal
  stretch, so the pipeline's default `linspace(1, 1/N, N)` lands on ComfyUI's sigmas.
- Output is RGBA (4-channel VAE); save as PNG, as ComfyUI's SaveImage does.
- Image-to-image: core's width/height set the output (pass image_1's size to mirror the edit
  template); references use the pipeline's 1024² area (the node default; the edit template uses 0,
  native size, which coincides for a 1 MP multiple-of-32 input).

## Verification

Offline (CPU, on the real files):
- Transformer (fused layout): 649 keys, 0 missing, 0 unexpected, 0 shape mismatches.
- Text encoder: 0 missing, 0 unexpected, 0 shape mismatches.
- VAE: 238/238 tensors equal to the official diffusers VAE cast to bf16.
- Sigmas equal ComfyUI's `simple` schedule at shift 0.69 within 6e-8 (8 and 25 steps, 512² and
  1024²).

diffusers bump regression (512², seed 42, md5 before/after): comfy-z-image-turbo, comfy-krea2-turbo
and comfy-flux2-4b are byte-identical. comfy-qwen-image-edit-2511-lightning changed
deterministically (same edit, the framing shifted a few pixels); the Qwen-Image transformer itself
is bit-identical across the two commits (random weights, padded mask), so the change sits in
pipeline-level code (open, see Follow-ups).

Against ComfyUI v0.39.1 (one L4, same files, 25 steps, euler/simple, seeds 42/43/44; turboCLI
fed ComfyUI's exact CPU noise for each seed):

| config | RGB PSNR vs ComfyUI (seeds 42 / 43 / 44) |
|---|---|
| text-to-image 512² | 38.4 / 37.3 / 36.1 dB |
| text-to-image 1024² | 30.8 / 36.9 / 41.9 dB |
| edit 1024² (attic room, crisp) | 44.4 / 44.2 / 46.8 dB |

## Speed

A first version ran the stock diffusers transformer and was on par at 512² but 26-33% slower at
1024² (1.01 s/step against ComfyUI's 0.76). Op-level profiles of both on one L4 (4 steps) showed
the int8 GEMMs and the flash attention identical, and the whole gap in the glue around them:
ComfyUI's model runs it as four comfy-kitchen fusions (QK RMSNorm + RoPE in `rms_rope`, LayerNorm
modulation in `adaln`, in-place `addcmul_` gated residuals, the fused `gate_up` with its SiLU gate
folded into the int8 down projection's input quantizer), about 0.3 s of elementwise work, where
diffusers' separate fp32 ops (complex RoPE, per-token `torch.where` modulation copies, `tanh`,
`silu * mul`) cost about 1.7 s.

So the engine runs the diffusers transformer the way ComfyUI's model does: its `_comfy_forward`
(classes defined inside it, so discovery stays torch-free) swaps in ComfyUI's fused MLP, its
block forward and the fused QK norm + RoPE, through the offloader's new `kitchen_ops()`
(comfy_kitchen as ComfyUI configures it, and ComfyUI's `linear_input_act`, which stays in the
GPL backend).
Without comfy-kitchen's kernels only the MLP stays fused.

| config (L4, warm, 25 steps) | ComfyUI (prompt exec) | turboCLI (generate) | turboCLI sampling |
|---|---|---|---|
| text-to-image 512² | 4.6 s | 5.4 s | 4.6 s |
| text-to-image 1024² | 19.6 s | 21.3 s | 19 s (0.76 s/step, ComfyUI 0.76) |
| edit 1024² | 25.3 s | 28.0 s | 24 s |

Sampling is on par. ComfyUI's warm edit reruns skip the prompt and reference encode (its node
cache reuses identical inputs), so its edit figure is a floor; the rest of the end-to-end gap
(~1.5 s at 1024²) is outside sampling (see Follow-ups).

## Follow-ups

- End to end at 1024², phase by phase (warm, GPU-synced): the PNG save was ~0.4 s slower (PIL's
  level 6 against ComfyUI's SaveImage level 4; core now saves at 4), the offloader's per-run
  `reclaim` cost ~0.2 s before the result went out (core now runs it after `Saved:`), and the
  VAE decode stays 0.3 s behind (1.0 s against ComfyUI's 0.69 s, diffusers' causal-conv padding
  copies; left as is, closing it means replacing diffusers' decoder).
- The 4 GB laptop run (needs the local download).
- Bisect the comfy-qwen-image-edit-2511 pipeline-level change across the diffusers bump.
