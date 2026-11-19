# Image upscale: `image-upscale` mode, ESRGAN and Z-Image Turbo upscale engines

## Context

turboCLI has no upscaler. ComfyUI ships model upscalers through spandrel (`ImageUpscaleWithModel`)
and a generative Z-Image Turbo upscale (blueprint "Image Upscale (Z-image-Turbo)"). This adds both
behind a new `image-upscale` mode and an `image-upscale.sh` script; SeedVR2 comes later on the same
mode. Bar: simple code, tested, on par with or faster than ComfyUI on the same torch.

| engine | kind | weights | upscaler |
|---|---|---|---|
| `esrgan` | compute (`run()`) | url: xinntao release `v0.1.0` `RealESRGAN_x4plus.pth` | |
| `comfy-esrgan` | compute (`run()`) | COMFY: `Comfy-Org/Real-ESRGAN` `RealESRGAN_x4plus.safetensors` | |
| `z-image-turbo-upscale` | diffusion, BASE `z-image-turbo` | Tongyi repo (shared) | `esrgan` |
| `comfy-z-image-turbo-upscale` | diffusion, BASE `comfy-z-image-turbo` | the 3 Z-Image files | `comfy-esrgan` |

One mechanism ties them: a Z-Image upscale engine declares `UPSCALER = "<esrgan engine id>"`;
install installs it first, check requires it, core runs it. Stock engines stay ComfyUI free.
Licenses: spandrel MIT, RealESRGAN BSD-3, Z-Image Apache-2.0; no `spandrel_extra_arches`.

## Script: `bash/turbo/image-upscale.sh` (new)

`image-to-image.sh` with one input, a scale for width/height and the prompt optional:

```
image-upscale <engine> <renderer> <input image> <output image>
              [scale = 2.0] [prompt = none]
              [seed = -1] [inference = -1]
              [offload = offloader] [slicing = none]
              [loras = none] [server]
```

Sends `--mode image-upscale --scale "$scale" --prompt="$prompt"` (`none` sends empty), same fields
in the server `curl`.

## Upscale chain

- **ESRGAN**: native x4 in tiles (512, overlap 32, feathered blend, tile halved on OOM down to 128:
  ComfyUI's numbers), clamp, lanczos to `round(w * scale) x round(h * scale)`. Our own tiler
  (ComfyUI's `tiled_scale` is GPL, turboCLI is LGPL). One `%3d%%|step i/n` line per tile.
- **Z-Image upscale** (the blueprint without its 1 MP prescale, the scale sets the size): ESRGAN to
  the target on the 16 grid, then `ZImageImg2ImgPipeline`, strength 0.33, 5 effective steps
  (`num_inference_steps = int(5 / 0.33)`, which diffusers cuts to 5 like ComfyUI's denoise).
  Empty prompt means `"masterpiece, 8k"`.

## Performance

- **Resident ESRGAN**: `_upscale` keeps the last loaded model (keyed by file and device, 67 MB), as
  ComfyUI caches its loader node: a warm call pays no load.
- **Half precision on GPU**: fp16 on cuda/mps, `channels_last`, under `torch.inference_mode`;
  ComfyUI runs fp32. Kept only if it stays within 0.5 dB PSNR of our fp32 output, else fp32.
- **Tile size**: start at ComfyUI's 512; try 768 and 1024 once; keep the fastest seamless one.
- **Z-Image**: the existing pipe path (offloader, encode cache, cudaMallocAsync); the ESRGAN pass
  runs before `backend.prepare`, so the transformer is not resident at the same time.
- Target: per call time at most ComfyUI's, warm and cold, on the same torch (tests below).

## Implementation

### `runner/engine/_upscale.py` (new, torch free at import)
- `upscale_size(image, scale, grid=1)`, the target `(w, h)`.
- `esrgan(image, size, renderer, file, emit)`: spandrel `ModelLoader().load_from_file` (takes
  `.pth` and `.safetensors`), `ImageModelDescriptor` required, device from
  `_segment.pick_device`, resident cache, tiled run, lanczos to `size`.
- `run(ctx, params, emit, file)`: the compute engines' body (first input, `params["scale"]`).

### Engines (`runner/engine/`)
- `esrgan.py`: `MODES = ("image-upscale",)`, `MODEL` kind `url` (as `mask_inspyrenet.py`),
  `weights(model_dir)` (the `.pth` in it), `run()` calling `_upscale.run`.
- `comfy_esrgan.py`: `COMFY` with one component (`role` upscaler, its own `repository`, pinned
  `revision` `5fd49b7b278836f48af63ecd314d0f98ab336105`, `path`
  `upscale_models/RealESRGAN_x4plus.safetensors`, root `filename`), so an existing ComfyUI file
  is reused; no SCAFFOLD; `esrgan`'s `run` and `weights`, which reads either install.
- `z_image_turbo_upscale.py`: `BASE`, `PIPELINE = "diffusers:ZImageImg2ImgPipeline"`,
  `MODES = ("image-upscale",)`, `INFERENCE = 5`, `STRENGTH = 0.33`,
  `PROMPT = "masterpiece, 8k"`, `UPSCALER = "esrgan"`. Core's default load covers it.
- `comfy_z_image_turbo_upscale.py`: the same delta with `UPSCALER = "comfy-esrgan"` and a
  `load()` calling the base builder with `_pipeline_class("ZImageImg2ImgPipeline")`.
- `comfy_z_image_turbo.py`: `load()` body moves into `_build(ctx, pipeline)`; `_pipeline_class`
  takes the diffusers base class (`ZImagePipeline` by default, no behavior change). On the img2img
  base, `prepare_latents` encodes to the latent's mean as ComfyUI's VAEEncode does, and empties
  the CUDA cache before sampling (see Results).
- `__init__.py`: contract comment lists `STRENGTH`, `PROMPT`, `UPSCALER`, `weights()`.

### `runner/core.py`
- `generate()`: an `image-upscale` branch beside `image-to-image`: target size on the 16 grid,
  `up = ENGINES[mod.UPSCALER]`, `kwargs["image"] = esrgan(..., up.weights(resolve_model(up,
  params)), emit)`, `strength`, `num_inference_steps`, empty prompt takes `mod.PROMPT`.
  Width/height are read only for the other modes; `scale` is `params.get("scale", "2.0")`.

### `runner/cli.py`
- `--scale` (default `"2.0"`); the server passes every field already.

### `runner/install.py`
- `_install_comfy` / `_engine_installed`: SCAFFOLD optional.
- `main()`: the per engine body becomes `_install(mod, args)`; an `UPSCALER` is installed first
  with the same args (`--comfy` included). `_engine_installed` requires the `UPSCALER` too.

### Packaging and docs
- `bash/turbo/build.sh`: `spandrel` 0.4.2 through `require`. `bash/turbo/turbo.pri`: the script.
- `bash/turbo/README.md`, `MANUAL.md`, `engine.md`, `implementation.md`, `test/README.md`:
  the mode, the four engines, the tests.
- `bash/turbo/install.sh`, `check-model.sh`: the engines and the mode in their usage.
- `doc/image-upscale-plan.md`: this plan.

## Tests

No separate bench: the suite takes the mode. `test/run.sh` runs `image-upscale` on
`test/old/knight-small.png` (`knight.png` halved, 368x512) to the size's height (x2 at 1024,
x4 at 2048, x8 at 4096, x2 at least), and
`test/comfy.py` has the ComfyUI graphs: `comfy-esrgan` (`UpscaleModelLoader` +
`ImageUpscaleWithModel` + lanczos `ImageScale`) and `comfy-z-image-turbo-upscale` (the blueprint
at the same size and seed, euler/simple as the `z_image` graph). `files()` follows BASE and adds
the UPSCALER's files. With `--comfy`, each runs alternately against ComfyUI and fails `SLOWER` past
the tolerance; `esrgan` has no steps, so its wall time is shown, not judged.

## Results

ESRGAN on cuda: fp16 is 54.4 dB from fp32 and 3.3x faster; `channels_last` 1.5x; cudnn autotune
and tile 768 / 1024 gain nothing over 512; tile 512 against 256 is 43.5 dB (no seams); a forced
out of memory falls back 512, 256, 128.

The comfy Z-Image upscale first sampled +24% against ComfyUI at 22.7 dB. Two causes, both fixed in
`comfy_z_image_turbo._pipeline_class`: diffusers samples the VAE latent from the generator (ComfyUI
takes its mean), which also shifts the noise, and the ESRGAN tiles and the VAE encode left their
buffers in the allocator, so the offloader streamed more each step. Then 37.9 dB, the same
picture as ComfyUI's. The allocator part was first fixed with `torch.cuda.empty_cache()` after
each; once the offloader loaded Z-Image with KSampler's memory estimate (below) an A/B on the
A1000 ran faster without them (43.4 and 39.8 s against 46.6 and 41.6 s, the image identical), so
they went.

| engine | RTX A1000 4 GB, against ComfyUI | L40S, against ComfyUI |
|---|---|---|
| `comfy-esrgan` (launch to image) | 6.8 s / 34.4 s, -80%, 50.9 dB | 3.5 s / 8.4 s, -57%, 50.9 dB |
| `comfy-z-image-turbo-upscale` (sampling) | 41.6 s / 44.3 s, -5%, 37.9 dB | 3.96 s / 5.22 s, -24%, 37.9 dB |
| `esrgan` (launch to image) | 7.9 s | 4.2 s |
| `z-image-turbo-upscale` (sampling) | 34.4 s | 11.9 s |

A stock engine takes the dtype its Hugging Face page publishes: Z-Image-Turbo's pinned revision
publishes its transformer in fp32, which the offloader streams as it is and computes in fp32,
z-image-turbo-upscale 11.9 s against ComfyUI's 4.1 s on the L40S. Installed with `bfloat16`
(install.sh's dtype) it takes 4.17 s, z-image-turbo about 5.4 s against 20.05 s. A Z-Image gap
outside the upscale code, fixed after: at 4096 the comfy engines lost 2 to 3% a step streaming
part of the model back: the offloader now loads a model declaring
`memory_usage_factor` with KSampler's memory estimate, right after `prepare_latents`, before the
first step as KSampler does. After both, on the L40S against ComfyUI: text-to-image +2% at 1024
(the image SAME) and -4% at 4096, the upscale +1% and +0%, the steady steps equal. The +1 to 2%
at 1024 was the test's order: the side running last gets the warmest first step, ComfyUI with
three repeats. With `test/run.sh --warmup` turbo's first step is the faster (3.15 s against
3.20 s), and cold against cold, each side alone in a fresh container, its sampling too (7.20 s
against 7.51 s).

Every output is
736x1024 RGB, within 1.5 levels of the original's mean, the L40S's 59 dB (ESRGAN) and 39.4 dB
(Z-Image) from the laptop's. `comfy-z-image-turbo` text-to-image stays byte identical on the L40S
after the `_build` change (SAME, -1% against ComfyUI).
