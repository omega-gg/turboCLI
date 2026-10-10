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
  `comfy_api()`, hands an engine ComfyUI itself (comfy_kitchen, comfy.ops, comfy.model_management;
  see Speed).

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
- VAE: ComfyUI's own, through the offloader's `comfy_vae` (see Revision): the engine hands it the
  file, the offloader builds the `comfy.ldm.wan.vae2_2.WanVAE` comfy/sd.py builds for it and runs
  it as sd.py does. It loads the file as is, strictly.
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
block forward and the fused QK norm + RoPE, through the offloader's new `comfy_api()`
(comfy_kitchen as ComfyUI configures it, and ComfyUI's modules, which stay in the GPL
backend).
Without comfy-kitchen's kernels only the MLP stays fused.

| config (L4, warm, 25 steps) | ComfyUI (prompt exec) | turboCLI (generate) | turboCLI sampling |
|---|---|---|---|
| text-to-image 512² | 4.6 s | 5.4 s | 4.6 s |
| text-to-image 1024² | 19.6 s | 21.3 s | 19 s (0.76 s/step, ComfyUI 0.76) |
| edit 1024² | 25.3 s | 28.0 s | 24 s |

Sampling is on par. ComfyUI's warm edit reruns skip the prompt and reference encode (its node
cache reuses identical inputs), so its edit figure is a floor; the rest of the end-to-end gap
(~1.5 s at 1024²) is outside sampling (see Follow-ups).

## Revision: ComfyUI's VAE

A layer of this engine showed rows of dots and tints in turbopixel. Measured on neutral
1024×768 text-to-image runs (seed 42, same files as ComfyUI, the 4 GB card): the image was not
fully opaque, alpha falling to 181-215 along a grid at the tile spacing, which turbopixel blends
with what lies under the layer. The diffusers VAE needed 5.5 GB there, so the offloader tiled it
(256 px tiles, 64 overlap) and each tile's border bent the alpha; decoded whole the alpha matched
ComfyUI's within a level. ComfyUI decodes the same image whole on that card: its Wan 2.2 decoder
runs a single image in row strips, exact and in a quarter of the memory (1.4 GB).

So the engine now runs ComfyUI's own VAE through the offloader's `comfy_vae`: comfy/ldm/wan/
vae2_2.py, vendored verbatim, built and run as sd.py's VAE class does (its settings, its memory
steps under dynamic VRAM, the clamp, its tiled fallbacks). The key-rename tables are gone.
Results, same runs:

- Alpha against ComfyUI: within 1-6 levels (text-to-image and edit), from 40-74 before; its
  lowest value 247-254, as ComfyUI's.
- The faint pixel checker on flat areas is the model's: ComfyUI shows it as strongly.
- Text-to-image 1024×768 on the 4 GB card: 93-109 s against 128-142 s before (no tiling and no
  VRAM freeing for the old estimate; one run each, the laptop throttles).
- The decode alone, warm, against ComfyUI's on the same latent shape. L4 at 1024²: 0.73 s
  against 0.70 s, from 1.0 s, so the 0.3 s noted under Speed is gone (the same kernels profile on
  both). 4 GB card at 1024×768, interleaved, untiled on both: 1.50 s against 1.52 s, and 2.57 s
  against 2.61 s once throttled.
- Colors stay where they were against ComfyUI (27-31 dB text-to-image, 21 dB on the edit, where
  the reference sizing differs as noted above): the sampling drift, not the decode.

## Revision: Turbo, int8 attention, cache placement

`comfy-qwen-image-2-1-turbo` inherits this engine for Qwen-Image-2.1-Turbo, Qwen's 8-step
distillation: only the transformer differs (Comfy-Org's `qwen_image_2.1_turbo_int8_convrot`,
7,257 MB; Turbo's transformer config is identical), at 8 steps on the same schedule, since
ComfyUI's shift 0.69 belongs to the model (`supported_models.py`), not the template.

- Turbo's file sets `comfy_kitchen_int8` attention on blocks 1-31, which ComfyUI runs through
  comfy-kitchen's int8 kernel. The offloader now reads that config (`use_comfy_attention_config`):
  31-39 dB against ComfyUI on text-to-image, from 29-34 dB with bf16 attention.
- The prefix cache's "auto" rule read raw free VRAM. Under dynamic VRAM that looks nearly full
  even on an L4, so the K/V went to host RAM and came back every step: the edit ran ~0.35 s per
  step behind ComfyUI. It now asks the model's patcher (`current_patcher.get_free_memory`, handed
  over by the offloader's `pre_run`), once per forward before the blocks run, as
  `select_prefix_cache` does; from inside a block, aimdo warns about the block's pinned pages.

L4 at 1024², 8 steps, warm (cold): text-to-image 7.3 s (20.9 s) against ComfyUI's 7.1-7.2 s
(33.4 s); edit 9.3-9.4 s (11.9 s) against 8.9-9.2 s (17.5 s), from 12.4-12.7 s. On the 4 GB card
at 1024×768 speed is on par (warm 22-31 s on both, the laptop throttles).

## Follow-ups

- End to end at 1024², phase by phase (warm, GPU-synced): the PNG save was ~0.4 s slower (PIL's
  level 6 against ComfyUI's SaveImage level 4; core now saves at 4), the offloader's per-run
  `reclaim` cost ~0.2 s before the result went out (core now runs it after `Saved:`), and the
  VAE decode was 0.3 s behind (1.0 s against ComfyUI's 0.69 s, diffusers' causal-conv padding
  copies), since closed by running ComfyUI's own decoder (see Revision).
- On the 4 GB card (RTX A1000 laptop, interleaved runs, same files and graphs as ComfyUI):
  text-to-image 512² 55-67 s cold against 66-76 s; the 1024² edit 190-209 s cold against
  232-237 s, and 166 s warm (same image, new seed) against 169 s. Three fixes got it there, all
  ComfyUI's own: the edit's prefix K/V (~2 GB) go to host RAM when VRAM is short, by ComfyUI's
  "auto" rule and pinned within its budget (28 to 6.1 s/step); the engine gives the offloader its
  VAE's real working memory (diffusers' decoder needs 7 GB at 1024²), so the decode tiles up
  front instead of failing first; and the encode cache now keys an edit's images by content and
  covers the reference VAE encode. Sampling stays ~3% behind there (6.1 against 5.9 s/step):
  ComfyUI's K/V prefetch on its offload stream measured slower on that card (6.49 s/step, the
  stream is busy with the weights), so the plain copy stays.
- Bisect the comfy-qwen-image-edit-2511 pipeline-level change across the diffusers bump.
- RAM on the 4 GB card at 1024×768: pins now match ComfyUI's (13.05 GB; the offloader loads each
  model at its own node, see its implementation.md), but the rest of private RAM stays ~0.5 GB
  above (9.4 against 8.8-9.1 GB between images, peak 24.6 against 23.5 GB). Both start at
  2.7-2.8 GB and keep the model files mapped, so it is in what turboCLI allocates while running.
  Speed is on par once the laptop's heat is controlled (GPU cooled to 55 °C before each image):
  12.7-12.8 s of sampling on both, steady steps 1.75 against 1.77 s.
