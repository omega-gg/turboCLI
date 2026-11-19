# Plan: MiniMax-H3-Image, comfy-minimax-h3-image and minimax-h3-image

## Context

MiniMax-H3-Image (released 2026-11-10) is a 33.1B single-stream DiT, the H3 video architecture
without its audio branch, on H3's Qwen3-VL 32B text encoder (layer 50) and a 24-channel image VAE
(16x down, 2D ViT decoder). It is distilled to 6 steps without CFG: one output head per step (PDD
heads) on a fixed sigma grid. Text to image and multi-reference editing share one checkpoint, and
it works best at 4 to 20 MP. ComfyUI v0.45.4, the tag turbo-offloader vendors, runs it natively,
and diffusers v0.43.0 ships it as a modular pipeline. As with MOSS-Image, we add two engines:

- **comfy-minimax-h3-image**: ComfyUI's own files (49.5 GB), faithful to ComfyUI's two templates,
  text to image and image to image, at ComfyUI's speed or better, tested against ComfyUI.
- **minimax-h3-image**: the stock engine on the official `MiniMaxAI/MiniMax-H3-Image` repo (bf16,
  ~135 GB), ComfyUI-free, with or without the offloader, text to image and image to image (the
  official pipeline edits too, unlike MOSS).

Neither fits a laptop: both are installed and tested on Modal only, like moss-image, and a laptop
suite skips them.

**License:** MiniMax-H3-Image Community License. Its "Excluded Territories" are the EU, the UK,
South Korea and the US (sections I.5 and V.4). Above $1M annual revenue or for a hosted service,
written authorization is needed, and a commercial UI must show "MiniMax-H3-Image" (IV).
engine.md and both engines' headers state the license.

## What the references do

| | ComfyUI t2i template | ComfyUI edit template | Official (diffusers 0.43) |
|---|---|---|---|
| Transformer | `minimax_h3_image_pruned_int8_convrot` (21 GB) | the same | `transformer_diffusers` bf16 (66 GB) |
| Text encoder | `qwen3vl_32b_minimax_h3_int8_convrot` (27 GB) | the same | Qwen3-VL 32B bf16 (66.7 GB) |
| VAE | `minimax_h3_image_vae_fp16` (1.2 GB) | the same | `vae_diffusers` fp32 (2.3 GB) |
| Sampling | KSampler 6 steps, CFG 1, euler, simple | the same | 6 steps on `sample_sigmas`, no guider |
| Negative | ConditioningZeroOut (unused at CFG 1) | the same | none |
| Size | ResolutionSelector, 1:1 at 4 MP (2048²) | the source at 4 MP, then the 32 grid | requested; under 4 MP it renders on a 4 MP canvas, LANCZOS back |
| References | | the scaled source as `images.image_1` | `image`, each kept within a 1024 to 2048 short edge |

ComfyUI's `MiniMaxH3ImageConditioning` (nodes_minimax_h3.py:711-747) tokenizes
`"Picture 1: " <vision block> ... <prompt>` (vision at a 1024 short edge), VAE-encodes each
reference at a short edge within 1024 to 2048, and returns the positive and an empty
`[1, 24, 1, H/16, W/16]` latent. The sigma grid is the file's metadata (`pdd_steps: 6`,
`pdd_sigmas` 1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0.0), and
`MiniMaxH3ImageModelSampling` (sampling.py) interpolates it linearly, so "simple" at 6 steps lands
on the grid. Each step's head is picked from the sampler's schedule (`sample_sigmas`).

Both templates carry a prompt expander (TextGenerateV2 on Qwen3.5 9B): off in t2i, on in the
edit. We leave it out, as for MOSS: turboCLI has no such feature, and the prompt is passed
verbatim.

## Shared runtime

- `bash/turbo/build.sh`: bump `diffusers` from `7ecb1e1b` to v0.43.0
  (`423ab90580b2847998591669b8326f47b6d14d8b`). Our pin predates H3-Image (875373ac, #15230). It
  has the H3 video blocks but not `MiniMaxH3ImageTransformer2DModel`,
  `AutoencoderKLMiniMaxH3Image` or `MiniMaxH3ImageBlocks`. transformers 5.12.1 already has
  `Qwen3VLForConditionalGeneration`. The bump affects every stock engine: see Verification 5.

## minimax-h3-image (stock)

`runner/engine/minimax_h3_image.py`, on the pattern of `flux3_6b.py` (a classic face over
diffusers' modular blocks):

- `MODEL`: `MiniMaxAI/MiniMax-H3-Image` at `8ced60068623def5b74b4d10558117cb61d6482d`,
  `allow_patterns`: `modular_model_index.json`, `transformer_diffusers/*`, `vae_diffusers/*`,
  `text_encoder/*`, `processor/*`, `tokenizer/*`, `scheduler/*`. It does not take the root
  `transformer/` or `vae/` (MiniMax's own layout) or the PDD heads file.
- `PIPELINE = "diffusers:MiniMaxH3ImageModularPipeline"`,
  `TRANSFORMER = "diffusers:MiniMaxH3ImageTransformer2DModel"`.
- `MODES = ("text-to-image", "image-to-image")`, `INFERENCE = 6`, and
  `CFG = ("guidance_scale", 1.0)`, guidance-distilled, so the pipe ignores it.
- `H3Pipe(DiffusionPipeline)` takes the same steps as `Flux3Pipe`:
  - `MiniMaxH3ImageBlocks().init_pipeline()`;
  - the denoise loop's `progress_bar` (`MiniMaxH3ImageDenoiseLoopWrapper`) routed through ours,
    for the runner's step lines, the step callback and cancel;
  - `_execution_device` follows the pipe's;
  - `from_pretrained` loads the components from the subfolders the modular index names (a
    streamed transformer may come in its place).
  - `__call__(prompt, image=None, height, width, generator, ...)` calls the modular pipeline with
    `image` for edits and keeps diffusers' defaults (`min_generation_area` included), as the
    official pipeline renders.
  - `encode_prompt` stays inside the blocks: the text encoder sees an edit's reference images, so
    the offloader's prompt-keyed encode cache must not serve it. Check `install_encode_cache`'s
    key, and keep the text encoder block if the key covers the prompt only.
- `load()`: `ctx.backend.load_pipe(...)` with the offloader, else `from_pretrained` plus
  `apply_offload`, as in `flux3_6b.load`.

## comfy-minimax-h3-image

### turbo-offloader

- **Vendor**, byte for byte from v0.45.4: `comfy/ldm/minimax/model.py`,
  `comfy/ldm/minimax/vae.py`, `comfy/text_encoders/minimax.py`. Everything they import is already
  vendored (`ldm/modules/attention.py` with `ComfyAttention`/`AttentionTensorContainer`,
  `common_dit`, `model_prefetch`, `quant_ops`, `rmsnorm`, `text_encoders/qwen3vl.py`,
  `qwen_vl.py`). `sampling.py` imports the unvendored `model_sampling`, so it is mirrored instead
  (below).
- New **`offloader/minimax_h3_image.py`**, on the pattern of `moss_image.py`, each part citing its
  v0.45.4 lines:
  - `detect(sd, metadata)`: model_detection.py:401-434 for the image branch, with `pdd_steps` and
    `pdd_sigmas` from the file's `config` metadata.
  - `sampling(file)`: `MiniMaxH3ImageModelSampling.sigma` (sampling.py) on the file's grid, for
    `FlowSampler`.
  - `MiniMaxH3Image(adapter.BaseModel).forward(latents, timestep, context)` mirrors
    model_base.MiniMaxH3Image and MiniMaxH3.extra_conds (model_base.py:2254-2300, 2429-2441):
    - `preprocess_text_embeds` once per generation (the condition projection plus the token
      refiner, which ComfyUI runs once per sampling);
    - the `minimax_payload`: token tags, refs, `cond_video_latents`, seed;
    - the `PackedLayout`;
    - the timestep as sigma × 1000;
    - `transformer_options["sample_sigmas"]` = the sampler's sigmas, for the PDD heads.
  - `conditioning(prompt, images, vae)` context: `MiniMaxH3ImageConditioning`
    (nodes_minimax_h3.py:673-747, `_size_short_edge`, `_resize_to`, `_image_refs`). The source
    is first scaled as the edit template does: `ResizeImageMaskNode` "scale total pixels" 4 MP,
    then "scale to multiple" 32, both "area". The references are VAE-encoded before sampling,
    as MOSS's are.
  - `TextEncoder(adapter.CLIP)`: CLIPLoader type minimax on the Qwen3-VL 32B file
    (sd.py:1842-1850, 2127-2128). It tokenizes with `images=` and `picture_label="Picture %d: "`
    and returns the conditioning plus `minimax_token_tags` (`encode_from_tokens` already returns
    the extra dict).
  - `lora_keys`: the MiniMaxH3 branch of lora.py:426.
- `adapter.py`, small changes:
  - `load_comfy_single_file` passes the file's metadata to `detect` (it already reads it);
  - `FlowSampler` takes an optional `sigma` function in place of the shift. Its table,
    `percent_to_sigma` and the simple scheduler then run on it, as ComfyUI's do on any model
    sampling;
  - `_comfy_vae_model` gains sd.py's H3 image VAE branch (sd.py:1050-1076 for
    `MiniMaxH3ImageVAE`, latent statistics included). `ComfyVAE` takes its 5D latent, and the
    memory estimates follow that branch.
- `__init__.py`: register `"minimax_h3_image"` in `_COMFY_MODELS`. `comfy_sampler` passes the
  model's `sampling(file)` when the module defines one.
- Update `resync.md` (the vendored files and the mirrored lines), `implementation.md` and
  `turbo-offloader.pro`.

### turboCLI

`runner/engine/comfy_minimax_h3_image.py`:

- `MODES` both, `INFERENCE = 6`, `CFG = ("guidance_scale", 1.0)`.
- `COMFY`: `Comfy-Org/MiniMax-H3-Image` at `89c635085f539d03ad35485ed542c5a286e3c6d6`, three
  components: `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors`,
  `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (the template points to
  Comfy-Org/MiniMax-H3, and this repo holds the same file at the same size), and
  `vae/minimax_h3_image_vae_fp16.safetensors`.
- `SCAFFOLD`: `model_index.json` only, from the official repo (`stock.MODEL`, as MOSS's scaffold
  comes from OpenMOSS's), which the suite and the install check look for. ComfyUI detects
  everything else from its files.
- Sources, as for MOSS: the stock engine's weights from the official repo
  (`MiniMaxAI/MiniMax-H3-Image`), the comfy engine's from Comfy-Org's
  (`Comfy-Org/MiniMax-H3-Image`).
- `ComfyH3Pipe(DiffusionPipeline)`, a short loop of its own, as MOSSPipe's:
  1. encode;
  2. draw the noise as ComfyUI does (the seed's CPU float32 draw, shape `[1, 24, 1, h, w]`);
  3. `scheduler.set_timesteps(6)`, then one transformer pass per step (CFG 1 runs no uncond);
  4. `scheduler.step`, the progress bar update and the step callback;
  5. decode through ComfyUI's VAE.

  An edit wraps the loop in `transformer.conditioning(image, vae)`. It doesn't build on the stock
  H3Pipe: diffusers' transformer takes packed rows and position ids, while ComfyUI's takes
  `(x, timestep, context, payload)` and packs them itself. A translation layer would be more code
  and more risk than this loop. As for MOSS, an edit renders at the requested size (the template
  takes the scaled source's).
- `load()`: `comfy_sampler("minimax_h3_image", "euler", "simple")` on the transformer's file,
  `load_pipe_comfy(...)` with `comfy_vae(files["vae"], {})`. Offloader only, as the other comfy
  engines.

## Tests

- `test/comfy.py`: an `h3_image` builder for both modes, the templates without the expander:
  - UNETLoader, CLIPLoader (type `minimax`), VAELoader;
  - for edits, LoadImage, then ResizeImageMaskNode (scale total pixels 4, area), then
    ResizeImageMaskNode (scale to multiple 32, area) into `images.image_1`;
  - `MiniMaxH3ImageConditioning` at the output size, ConditioningZeroOut;
  - `ksampler(6, cfg=1.0, "euler", "simple")` on the conditioning's latent, VAEDecode.

  Register it in `GRAPHS` for both modes.
- `test/modal_app.py`: a `gpu_big` function (L40S, 224 GiB RAM). Both H3 engines hold more than
  the 64 GiB the regular function has: ComfyUI's 49.5 GB plus a process, the stock's 135 GB.
  Without it, every step streams off the Volume and the timings mean nothing. `test/modal.sh run`
  and `exec` gain `--big` to pick it, so the other engines keep the cheaper container.
- `test/run.sh` needs nothing: the cases come from `MODES` and `COMFY`, and an engine that isn't
  installed is skipped.

## Docs and lists

- `engine.md` (both tables, sizes, the license note), `README.md` (supported engines),
  `implementation.md` (engine notes, the stock/comfy pair, why the comfy loop is its own, the
  diffusers bump, Doc records), `test/README.md` (`--big`).
- `bash/turbo/install.sh`, `text-to-image.sh`, `image-to-image.sh` and `bash/turbo/README.md`
  usage lists, and `turboCLI.pro` (the two engines and the plan).
- `doc/minimax-h3-image-plan.md`: this plan. A "Changed after" section is added at the end, as in
  the MOSS plan.

## Changed after

- **The PDD config is read off the file's header** in `minimax_h3_image._config` (safetensors'
  `metadata()`, no tensor read), for `detect` and `sampling` alike. `load_comfy_single_file` and
  every other adapter's `build(sd, operations)` stay as they were.
- **`load_pipe` streams the transformer from the folder the pipeline names**
  (`transformer_folder`, `transformer` by default): the official repo keeps MiniMax's own
  transformer in `transformer/` and diffusers' in `transformer_diffusers/`. `cpu_fits_full_load`
  sums the same folder.
- **The stock encode is diffusers' own blocks**: a two-block pipeline (the edit's resize, then the
  text encoder) behind `encode_prompt`, the main blocks taking `prompt_embeds` and
  `text_token_tags`. The offloader's encode cache keys on the call's full arguments, images by
  their content, so an edit's references are part of the key.
- **The VAE's out-of-memory fallback** calls the model's own `decode_tiled` / `encode_tiled`
  (`handles_tiling`, sd.py's `_decode_tiled_owned`), not the generic tilers.
- **16 MP is tested too**, as MiniMax recommends, beside 1024 and the template's 2048. Its first
  run had one repeat, so turbo went first and alone read the 49 GB of files off the Volume cold:
  23.27 s of sampling against ComfyUI's 18.90 s (+23%), its load and encode 15.8 s and 20.0 s
  (8.3 s each warm). With two repeats, each side first in turn, 19.30 s against 19.32 s.
  `test/README.md` says to judge against ComfyUI with `--repeat 2` or more.
- **The offloader archives a ComfyUI model's dtypes as it builds it** (model_base.py:197,
  `load_comfy_single_file`). The first run against ComfyUI gave 20.0 dB (text to image) and
  24.9 dB (edit): same scene, another fox. Dumping both sides' tensors showed identical inputs to
  the transformer (latent, timestep, conditioning, token tags, sigmas, the edit's reference), and
  the first difference in its float32 layers' biases (`video_patch_proj`, the PDD heads), off by
  bf16 rounding: ComfyUI's patcher loads a file's float32 tensor in its layer's archived dtype,
  and without the archive it stayed float32. The 33B model's 50 int8 layers and 6 steps turned
  that 2e-3 into another picture. With the archive the transformer is bit-identical to ComfyUI's
  at every step (51 dB, the rest being the 8-bit conversion). Every comfy engine loads through
  it, so their references are recorded again.
- **The comfy engine saves as SaveImage does**: each value times 255, truncated, not rounded.
- **An edit's source is sized once**, as the template's two ResizeImageMaskNode feed
  MiniMaxH3ImageConditioning: `MiniMaxH3Image.edit` sizes it and yields it, `TextEncoder.encode`
  makes its 1024 copy, and `extra_conds` VAE-encodes the reference after the text encoder, before
  the transformer. Encoding the reference first, the node's own order, put the edit one level off
  on 4% of its pixels: the H3 VAE sizes its tiles to the free VRAM, and only this order gives
  ComfyUI's images exactly. One path serves both modes (no image, no references).
- **flux3-6b drops diffusers 0.43's prompt upsampler step**, as its docstring says to encode the
  prompt as given: the bump put it first in FLUX.3's blocks, and it asked for the `prompt` that
  `Flux3Pipe` hands over as embeddings. flux3-6b and comfy-flux3-6b are `SAME` again. Every other
  installed engine's images were `SAME` after the bump and the offloader fixes (mask-inspyrenet
  has never run on Modal: its OpenCV needs libglib, not in the image).
- **`load_streamed` keeps `_keep_in_fp32_modules` in float32**, as `from_pretrained` does: it cast
  MiniMax-H3-Image's rope buffers to bf16 (FLUX.3's kept modules are file weights, unchanged).
- **`install_prefetch` prefetches blocks only, not a list of bare layers.** The stock engine's text
  to image changed between runs (28 dB), its text embeddings bit-identical: diffusers keeps the
  six PDD heads in a top-level `ModuleList`, which the offloader prefetched as a block sequence,
  though a step runs one head. The head a step ran was not the one the queue streamed, and its
  weights raced the copy (the 66 GB transformer streams through the 45 GB L40S every step). A
  sequence is now a list of blocks, each with layers of its own; three runs are byte-identical.

Results on the L40S (224 GiB container, seed 42), comfy-minimax-h3-image against ComfyUI, the
images identical at every size:

| Size | Mode | Sampling | From launch to image |
|---|---|---|---|
| 1024² | text to image | 5.22 s (5.19 s) | 22.3 s (23.1 s) |
| 1024² | image to image | 8.81 s (8.76 s) | 28.3 s (29.3 s) |
| 2048² | text to image | 8.07 s (8.11 s) | 26.5 s (27.2 s) |
| 2048² | image to image | 12.29 s (12.32 s) | 31.9 s (32.6 s) |
| 4096² | text to image | 19.30 s (19.32 s) | 42.1 s (43.5 s) |
| 4096² | image to image | 25.94 s (26.53 s) | 52.1 s (54.6 s) |

Cold against cold, each side alone in a fresh container reading the files off the Volume, turbo
reaches the first step in 31.5 s against 33.9 s and the image in 45.8 s against 48.4 s (1024²).

minimax-h3-image samples in 28.1 s at 1024² (on its 4 MP canvas), 23.3 s at 2048² and 86.4 s at
4096² (a single cold run); edits 34.3, 33.6 and 119.7 s. At 2048² both sides' text to image move
by one level on a few pixels between a cold and a warm run (ComfyUI's VAE sizes its tiles to the
free VRAM), the same on both.

## Verification (Modal: CPU for installs, a few short GPU runs, nothing left running)

1. **Install** on CPU: `test/modal.sh install comfy-minimax-h3-image minimax-h3-image`, about
   185 GB onto the Volume.
2. **Against ComfyUI** at the suite's 1024, for both modes: PSNR against ComfyUI in the 40s dB for
   text to image (an edit amplifies differences, as MOSS's 30.8 dB did), and sampling and launch
   to image at parity or better. This records the L40S reference lines:

   ```
   test/modal.sh run --big --record --repeat 2 --comfy /opt/ComfyUI \
                 comfy-minimax-h3-image minimax-h3-image
   ```

3. **Determinism**: a second run without `--record` is `SAME` and within the tolerance.
4. **At the template's size and MiniMax's**: `--size 2048` (the template's 4 MP) and `--size 4096`
   (16 MP, MiniMax's recommended 1:1 size) runs of both engines against ComfyUI, each recording its
   own L40S reference (`cuda-NVIDIA-L40S-2048.txt`, `-4096.txt`), the images checked by eye. Their
   numbers go into turbo-offloader's Benchmarks table. The stock engine is also checked against the
   official README's examples by eye.
5. **Regression for the diffusers bump**: `test/modal.sh run` on the installed stock engines
   (flux3-6b, flux2-4b, z-image-turbo, qwen-image-edit-2511, moss-image) stays `SAME` against
   the L40S reference, and so does a laptop's `test/run.sh` on its installed stock engines. The
   comfy engines and the FlowSampler change are checked by `comfy-moss-image` staying `SAME`.
6. `modal app list` shows nothing running afterwards.
