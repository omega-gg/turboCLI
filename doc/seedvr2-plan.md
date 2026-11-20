# SeedVR2 upscalers: comfy-seedvr2-3b-upscale and comfy-seedvr2-7b-upscale

## Context

SeedVR2 (ByteDance, Apache-2.0) is a one-step diffusion restorer: it takes an image already
enlarged to the target size and redraws it sharp in a single step, where ESRGAN only sharpens.
turboCLI's upscalers are image-to-image engines (`doc/upscalers-plan.md`); SeedVR2 joins them, as
fast as ComfyUI or faster, measured on Modal's L40S.

Decisions (asked):
- **Comfy engines only**: `comfy-seedvr2-3b-upscale` and `comfy-seedvr2-7b-upscale`. Diffusers has
  no SeedVR2 and ByteDance's code needs flash-attn and apex, so stock engines wait for a release.
- **The template's files**, Comfy-Org/SeedVR2 `diffusion_models/seedvr2_{3b,7b}_int8_convrot`
  (3.2 / 7.8 GB) and `vae/seedvr2_ema_vae_fp16` (0.5 GB), so the comparison is like for like.
- **3B and 7B**; the 7B "sharp" variant can come later as one more file.
- **seedvr2** in the names: it is the one-step successor, with its own weights, of SeedVR.

What ComfyUI runs (v0.45.4, `comfy/ldm/seedvr/`, `comfy_extras/nodes_seedvr.py`, template
`utility_seedvr2_3b_int8_upscale_image.json`): the input resized with lanczos, `SeedVR2Preprocess`
(alpha dropped, zero padded right and bottom to the 16 grid), `VAEEncodeTiled` 512/128 (a causal
3D VAE, the posterior mean x 0.9152), `SeedVR2Conditioning` (the latent plus a ones mask, 17
channels; the text is two embeddings stored in the DiT, no text encoder), `KSampler` 1 step, cfg
1, euler, simple, shift 1 (sigma 1 to 0 on a seeded noise latent), `VAEDecodeTiled` 512/128,
`SeedVR2PostProcessing` (crop to the resized input, color "none"), the alpha joined back.

## turbo-offloader: SeedVR2 on ComfyUI's own model

Following `resync.md`'s recipe for a new family, as MiniMax-H3 and MOSS were added:
- Vendor, byte for byte from v0.45.4, `comfy/ldm/seedvr/{model,vae,attention,constants}.py` and
  whatever they import that is not there yet (`color_fix.py` stays out: the template uses none).
- `offloader/seedvr2.py` (new), after `minimax_h3_image.py`: `detect(sd)` mirroring
  `model_detection.py:714-750` (3B, 7B), an `adapter.BaseModel` subclass whose `forward` maps the
  pipeline's call onto `NaDiT` (noise latent, timestep, `condition`, the stored
  `positive_conditioning` as context), `load(...)`, `SHIFT = 1.0`; registered in `_COMFY_MODELS`.
- `adapter._comfy_vae_model`: the SeedVR2 VAE branch mirroring `sd.py:609-624` (latent_dim 3,
  `handles_tiling`, the 4n+1 frame ratios, `x*2-1` input, its own scaling, the fp16 estimates).
  `ComfyVAE` gains a tiled encode/decode at the template's 512/128, which the pipeline uses.
- `load_pipe_comfy` takes `text_encoder=None` for a model with none: no encoder built, the
  sampling patchers alone (the rest of `_finalize_pipe` already copes with no encoder).
- `resync.md` (the files, the mirrored sd.py and model_detection lines), `implementation.md`,
  `turbo-offloader.pro`.

## turboCLI

**Engines** (`runner/engine/`):
- `comfy_seedvr2_3b_upscale.py`: `ID`, `TYPE = "seedvr2"`, `TRANSFORMER` (offload-wired: the
  diffusion path, so the pipe stays resident, the progress, cancel and reclaim come for free),
  `MODES = ("image-to-image",)`, `INFERENCE = 1`, `CFG = ("guidance_scale", 1.0)`, `UPSCALE =
  True`, `COMFY` (Comfy-Org/SeedVR2 at a pinned revision, the transformer and the VAE; no
  `SCAFFOLD`, as comfy-esrgan-upscale), and a short pipeline class built in `_pipeline_class()`:
  the first input resized to the output size (lanczos), padded to the 16 grid, `vae.encode`, the
  17 channel condition, a seeded noise latent, one Euler step through `comfy_sampler("seedvr2",
  "euler", "simple")` with the progress bar and `callback_on_step_end` as the other comfy
  pipelines, `vae.decode`, cropped back. The prompt is ignored. `load()` calls
  `ctx.backend.load_pipe_comfy(cls, {"model": "seedvr2", ...}, None, {"scheduler", "vae"})`,
  raising without a backend as the other comfy engines do.
- `comfy_seedvr2_7b_upscale.py`: `BASE` the 3B, its own `COMFY` (the 7B file).
- **The upscaler tag becomes explicit**: `UPSCALE = True` (in `_inherit._INHERITED`), set on the
  two ESRGAN engines too; `check._upscaler` reads it rather than `weights()`.
- **Alpha**: an upscaler on the diffusion path keeps its first input's alpha as the ESRGAN ones
  do and ComfyUI's template does: core reads the first input's alpha before the RGB conversion
  and puts it back, resized, on the result when the engine declares `UPSCALE`.

**Tests**:
- `test/comfy.py`: a `seedvr2` graph mirroring the template (UNETLoader, VAELoader, LoadImage,
  ImageScale lanczos to the size, SeedVR2Preprocess, VAEEncodeTiled 512/128, SeedVR2Conditioning,
  KSampler 1/1/euler/simple, VAEDecodeTiled 512/128, SeedVR2PostProcessing "none", out at node
  "12"), keyed for both engines. `test/comfyui.sh` needs nothing: the nodes are core.
- `test/run.sh` needs nothing: `*-upscale` already takes `knight-small.png`, no prompt,
  `$upscaled`.

**Lists and docs**: `install.sh` and `image-to-image.sh` usage (and their `bash/turbo/README.md`
copies), `engine.md` (Upscaling rows: size, 1 step, offloader), `implementation.md` (engine rows,
the `UPSCALE` contract row, the diffusion-path alpha), `MANUAL.md` (Upscaling: ESRGAN sharpens,
SeedVR2 redraws), `README.md`, `turboCLI.pro`, `test/README.md`; `doc/seedvr2-plan.md` (this plan,
copied first, Results after). turbopixel lists them under Image to Image by the tag, with the
canvas at its own size and no prompt, so only its MANUAL names them; `build.sh` pins the new
turbo-offloader commit.

Conventions: 99 columns, CRLF where the file has it, license headers, `# NOTE:`, no top-level
torch in engines. Commits on approval only, no Claude attribution; check what is pushed first.

## Verification (Modal L40S: Hugging Face is blocked on this machine)

1. Lint: `python -m pyflakes runner test`; the offloader's import smoke test (`resync.md` step 4).
2. `test/modal.sh install comfy-seedvr2-3b-upscale comfy-seedvr2-7b-upscale`, then `test/modal.sh
   exec --gpu` an `image-to-image.sh` run of each at 2x (`knight-small.png` to 736x1024): an image,
   one step, `Saved:`; a transparent input keeps its alpha; align and color lines with
   `align,color`.
3. Against ComfyUI, the suite's way: `test/modal.sh run --record --warmup --repeat 2 --comfy
   /opt/ComfyUI comfy-seedvr2-3b-upscale comfy-seedvr2-7b-upscale` at `--size 1024`, `2048` and
   `4096`: turbo's time at most ComfyUI's (launch to image and sampling), the PSNR high (same
   files, same graph). If slower, profile the VAE tiles and the load first.
4. `check-model MODES:image-to-image` tags both `upscaler`; turbopixel lists them in Image to
   Image and runs one with an empty prompt.
5. Results (timings, PSNR, memory) into `doc/seedvr2-plan.md` and `implementation.md`; the L40S
   references recorded; `modal app list` empty at the end.

## As built

- The plan as written, plus one offloader fix found on the way. A fresh process sampled in 260 to
  1340 s against ComfyUI's 3: SeedVR2's transformer ran on the CPU. It declares a memory estimate,
  so it loads only once sampling starts (BaseModel.prepare_sampling), and diffusers'
  `_execution_device` reads the modules' placement: when the pipeline asked, the transformer was
  still off the GPU, so the noise and the condition were built on the CPU and the whole forward ran
  there (comfy_kitchen refused its CUDA kernels, `device cpu not in {'cuda'}`). `_finalize_pipe`'s
  assignment of `_execution_device` is a no-op on diffusers' property; it now pins the device with
  the per-instance subclass it used for a text encoder off the compute device, for every pipeline,
  so no engine can fall into it again. On the A1000 the step went from 335 s to 2.9 s.
- A review against ComfyUI found the last difference: `ComfyVAE.encode` scaled the latent
  (`format_encoded`) in fp16, where ComfyUI casts it to float32 first. With that cast, the 3B's
  image is identical to ComfyUI's on the A1000 (it was 39.9 dB), sampling in 2.93 s (3.02 s,
  `--warmup --repeat 2`).
  The dB below came before it.
- Results on the L40S (knight-small.png to 736x1024, seed 42, `--warmup --repeat 2`), against
  ComfyUI on the same files and graph: comfy-seedvr2-3b-upscale sampled in 1.88 s (1.94 s, -3%),
  its image in 11.0 s from launch (11.7 s), 40.3 dB; comfy-seedvr2-7b-upscale 3.80 s (4.01 s, -5%),
  14.1 s (14.1 s), 45.2 dB. On the A1000 the 3B sampled in 2.58 s (2.54 s), its image in 18.1 s
  (25.4 s), 39.9 dB. Both engines' L40S references are recorded at 1024.
- Measured lean on purpose: one Modal run of both engines at 1024 (2 minutes); 2048 and 4096 were
  left out.
