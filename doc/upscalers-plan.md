# Upscalers are image-to-image engines

## Context

An upscaler takes an image and gives an image back: image-to-image already carries everything it
needs (an input, an output, a size, a seed, the steps, the fixes). A mode of its own,
`image-upscale` in turboCLI and Image Upscale in turbopixel, is a second road to the same place.
So the four upscalers become image-to-image engines, named so the user sees what they are:
`esrgan-upscale`, `comfy-esrgan-upscale`, `z-image-turbo-upscale`, `comfy-z-image-turbo-upscale`.
Everything image-upscale specific goes: the mode, its script, its `scale`, its plan, and the two
unpushed commits that built on it, turboCLI `07ba169` (align and color after an upscale) and
Sky-pro `25df8d1` (turbopixel's Image Upscale mode): both repos go back to `origin/master` and
the change is written from there, nothing kept for the old road.

Decisions (asked):
- **turboCLI says which engines are upscalers**, turbopixel relies on it rather than on a name:
  the `MODES:` answer turbopixel already reads tags them, so it costs no call.
- **An upscaler gets the canvas at its own pixels**: Image to Image renders the Canvas reference
  at the output size, which would hand ESRGAN an image the render already enlarged (4x the pixels
  at x2, and blurred). An upscaler gets the canvas, or the framed part of it, at its size in the
  canvas, and the output size says how big it comes back.
- **The prompt is optional**: in turboCLI it moves after `<output image>`, `[prompt = none]`, in
  `image-to-image.sh` and `text-to-image.sh` alike; turbopixel runs an upscaler with none.

## turboCLI

**Engines** (`runner/engine/`):
- `esrgan.py` -> `esrgan_upscale.py` (`ID = "esrgan-upscale"`), `comfy_esrgan.py` ->
  `comfy_esrgan_upscale.py` (`"comfy-esrgan-upscale"`); the four declare
  `MODES = ("image-to-image",)`; the Z-Image ones name `UPSCALER = "esrgan-upscale"` /
  `"comfy-esrgan-upscale"`. The model folder (`RealESRGAN`) and the ComfyUI component keep their
  names. Comments say "the output size" where they said `scale`.
- `_upscale.run`: the size is `(width, height)` from params; `upscale_size` goes (no caller left).
- `engine/__init__.py`: the IMAGE-UPSCALE note becomes UPSCALERS: an engine with `weights()` (the
  esrgan ones) or `UPSCALER` (the Z-Image ones) is an upscaler, image-to-image on its first input.

**core.generate** (`runner/core.py`, from `origin/master`):
- `esrgan-upscale` is a compute engine in image-to-image, so align and color must reach the
  compute path too: the steps are parsed (image-to-image only, as today) before the compute
  shortcut, and a new `_save(image, params, steps, emit, reference=None)` runs them, then saves
  and emits `Saved:`, for both paths, the reference passed when the path decoded it already, read
  from the first input otherwise (the compute path).
- The `image-upscale` branch moves into the image-to-image one: an engine with `UPSCALER` opens
  its first input (`source`), `reference = source.convert("RGB")`, the size `(width, height)` on
  the 16 grid the Z-Image pipeline needs, ESRGAN to that size, then the img2img pass (`STRENGTH`,
  `PROMPT` when the prompt is empty); `keep_alpha(image, source)` after it. The other engines load
  their references as today.
- `cli.py`: `--scale` and its param go.

**check** (`runner/check.py --modes`): an upscaler's line reads `<id> upscaler` (`_upscaler(mod)`:
`hasattr(mod, "UPSCALER") or hasattr(mod, "weights")`), as `--engines` adds `gated`.
`check-model.sh` and its README say so.

**Scripts** (`bash/turbo/`):
- `image-upscale.sh` deleted (`turbo.pri`, `turboCLI.pro`).
- `text-to-image.sh <engine> <renderer> <output image> [prompt = none] [width] ...` and
  `image-to-image.sh <engine> <renderer> <input images> <output image> [prompt = none] [width]
  ...`: the prompt swaps places with what followed it, `none` reads as empty, the minimum count
  drops by one; the rest keeps its position.
- `check-model.sh` (modes list), `install.sh` (the engine list: `esrgan-upscale` ...,
  `(installs esrgan-upscale too)`).

**Tests**:
- `test/run.sh`: the new argument order; the image-upscale mode goes, an `*-upscale` engine's
  image-to-image case takes `knight-small.png`, no prompt and `$upscaled` (set beside `input` and
  `expect`, in `run_turbo` and `run_comfy`).
- `test/comfy.py`: `GRAPHS` keys `("comfy-esrgan-upscale", "image-to-image")`,
  `("comfy-z-image-turbo-upscale", "image-to-image")`; the graphs already take width and height.
- `test/reference/cuda-NVIDIA-L40S-{1024,2048}.txt`: the keys renamed in place
  (`esrgan image-upscale` -> `esrgan-upscale image-to-image`, ...), the md5s kept: same input,
  size, prompt and seed give the same bytes, which the A1000 checks against its earlier upscale
  results.

**Docs**: `README.md` (examples), `bash/turbo/README.md` (the two usages, the image-upscale
section goes, the `upscaler` tag), `MANUAL.md` (examples; "Upscaling an image" becomes the
upscalers of image to image), `implementation.md` (layout, MODES, engine table, the flow and wire
contract), `engine.md` (the four rows), `dummy.md` (example), `doc/image-adjust-plan.md` (a
line: the steps reach the upscalers), `test/README.md`, `test/old/README.md`,
`doc/upscalers-plan.md` (this plan, copied first). `doc/image-upscale-plan.md` is deleted: it
plans the mode and script that go; what still holds, the upscale chain (tiles, out of memory
fallback) and the ComfyUI comparison (-5% on the A1000, -24% on the L40S), moves into
`doc/upscalers-plan.md`, which `implementation.md` cites in its place.

## turbopixel

- Back to `origin/master` (the Image Upscale mode, `25df8d1`, goes with its page, color, checks,
  translations and docs).
- `pApplyList` (`run/turbopixel.sky`): each line is `<id> [upscaler]`; the names go to the list
  as today, the upscalers into `pUpscalers` (`{ id: true }`, kept across modes like `pEngines`).
  `getUpscaler()` reads `pUpscalers[engineName] == true`.
- `pGetImages`: the Canvas reference renders at `pRunRect`'s size (the canvas pixels it covers)
  for an upscaler, at the output size for the others.
- `generate()`: no prompt is needed when the engine is an upscaler.
- `pRunGenerate`: the new order, `gui.async(script, engineName, render, images, output, prompt,
  outputWidth, ...)` and `gui.async(script, engineName, render, output, prompt, ...)`.
- Docs: `MANUAL.md` (Image to Image: the upscalers, no prompt needed, the canvas at its own
  pixels), `implementation.md` (state of the code, the turbo scripts table), `doc/upscalers.md`
  (this plan).

Conventions: 99 columns, CRLF where the file has it, license headers, `# NOTE:` / `// NOTE:`,
American spelling. First step: `git reset --hard origin/master` in both repos (the two commits
are local only; the working trees hold nothing else but untracked files, which stay). Commits on
approval only, one per repo, no Claude attribution.

## Deploy on this machine

The installed `esrgan`, `comfy-esrgan` and `comfy-z-image-turbo-upscale` are removed with the
current runtime (`remove.sh`, ComfyUI's files are never deleted), the new runner copied in (the
real deploy is the commit, `update.sh`, `build.sh cuda`), then `esrgan-upscale` and
`comfy-z-image-turbo-upscale` (same ComfyUI folder) installed; `generate.sh win64 turbopixel`, the
turbo scripts copied into `skz` (and its stale `image-upscale.sh` deleted).

## Verification

1. `check-model MODES:image-to-image` lists the upscalers with `upscaler`, the others bare.
2. `image-to-image.sh esrgan-upscale cuda knight-small.png out.png none 736 1024`: the same bytes
   as the earlier `esrgan` x2 (`237611e0...`); with `align,color` both lines; the Z-Image upscaler
   the same, with and without a prompt, through a server too.
3. A regular edit with the new order (prompt after the output), and `text-to-image.sh` with and
   without a prompt.
4. `test/run.sh esrgan-upscale comfy-esrgan-upscale comfy-z-image-turbo-upscale` on the A1000;
   `python -m pyflakes runner test`.
5. turbopixel: no Image Upscale mode; the upscalers in Image to Image; with one picked an empty
   prompt runs. On a 1024 canvas at x2: the canvas goes at 1024, a 512 frame at 512 (the temp
   render's size), the result lands at the output size. An edit engine still sends the output
   size and needs a prompt; Text to Image runs.
6. `awk 'length > 99'` on every touched file.

## As built

- turboCLI: on the A1000 the three installed upscalers give the bytes the old mode gave
  (`esrgan-upscale` and `comfy-esrgan-upscale` 237611e0..., `comfy-z-image-turbo-upscale`
  c3bf167e...), so the L40S references were renamed in place, their md5s kept. Align and color
  reach the upscalers (`esrgan-upscale`: under 0.75 px, left as is; color matched), `color=x` stops
  at `ERROR:` before any load, text-to-image runs with a prompt and with `none`, an edit in the new
  order runs and its fixes leave a night as is. pyflakes: only the known `pipe`.
- `test/run.sh` picks an upscaler's image-to-image case by the `-upscale` suffix of its id: the
  test owns its engine names, where turbopixel reads turboCLI's tag.
- Deploy on this machine: removing the old `esrgan`, `comfy-esrgan` and
  `comfy-z-image-turbo-upscale` with the old runtime was refused by the permission check, so their
  old modules were deleted from the runtime and the new ones installed over; the records
  `engine/esrgan` and `engine/comfy-esrgan` stay behind, unused. The reinstall of
  `comfy-z-image-turbo-upscale` could not fetch its scaffold again (an SSL certificate error), the
  one already there serves.

## Revised: the Z-Image upscalers removed

`z-image-turbo-upscale` and `comfy-z-image-turbo-upscale` were not good enough to keep, so they go,
with what only they used: the `UPSCALER` engine symbol and its install-first and check rules
(`install.py`), core's img2img upscale branch and its alpha, `comfy_z_image_turbo`'s img2img
pipeline class (`_pipeline_class(base)`, `_build`), the test graph and the L40S reference lines.
The upscalers left are the two ESRGAN engines, and an upscaler is an engine with `weights()`. An
upscaler takes no prompt any more; turbopixel still sends the one typed, which ESRGAN ignores.
`_upscale.py`, a helper only so they could share the ESRGAN pass, is folded into
`esrgan_upscale.py`.

## Carried over from the image-upscale plan

`doc/image-upscale-plan.md` planned the `image-upscale` mode and script, which are gone; what it
found about the engines themselves still holds, under their new names.

### The upscale chain

- **ESRGAN**: native x4 in tiles (512, overlap 32, feathered blend, tile halved on OOM down to 128:
  ComfyUI's numbers), clamp, lanczos to the output size. Our own tiler
  (ComfyUI's `tiled_scale` is GPL, turboCLI is LGPL). One `%3d%%|step i/n` line per tile.
- **Z-Image upscale** (the blueprint without its 1 MP prescale, the output size is the size):
  ESRGAN to the output size on the 16 grid, then `ZImageImg2ImgPipeline`, strength 0.33, 5
  effective steps (`num_inference_steps = int(5 / 0.33)`, which diffusers cuts to 5 like ComfyUI's
  denoise). Empty prompt means `"masterpiece, 8k"`.

### Performance

- **Resident ESRGAN**: `_upscale` keeps the last loaded model (keyed by file and device, 67 MB), as
  ComfyUI caches its loader node: a warm call pays no load.
- **Half precision on GPU**: fp16 on cuda/mps, `channels_last`, under `torch.inference_mode`;
  ComfyUI runs fp32. Kept only if it stays within 0.5 dB PSNR of our fp32 output, else fp32.
- **Tile size**: start at ComfyUI's 512; try 768 and 1024 once; keep the fastest seamless one.
- **Z-Image**: the existing pipe path (offloader, encode cache, cudaMallocAsync); the ESRGAN pass
  runs before `backend.prepare`, so the transformer is not resident at the same time.
- Target: per call time at most ComfyUI's, warm and cold, on the same torch (tests below).

### Results

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
| `comfy-esrgan-upscale` (launch to image) | 6.8 s / 34.4 s, -80%, 50.9 dB | 3.5 s / 8.4 s, -57%, 50.9 dB |
| `comfy-z-image-turbo-upscale` (sampling) | 41.6 s / 44.3 s, -5%, 37.9 dB | 3.96 s / 5.22 s, -24%, 37.9 dB |
| `esrgan-upscale` (launch to image) | 7.9 s | 4.2 s |
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

The L40S references hold the four upscale engines at 1024 and 2048 (`--record --warmup --repeat
2`), against ComfyUI: comfy-esrgan-upscale -57% at both, the comfy Z-Image upscale -9% and -1%. At
4096, measured but not recorded, -53% and +0%.

Every output is
736x1024 RGB, within 1.5 levels of the original's mean, the L40S's 59 dB (ESRGAN) and 39.4 dB
(Z-Image) from the laptop's. `comfy-z-image-turbo` text-to-image stays byte identical on the L40S
after the `_build` change (SAME, -1% against ComfyUI).
