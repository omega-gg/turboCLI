# Plan: MOSS-Image, comfy-moss-image and moss-image

## Context

MOSS-Image (OpenMOSS, Apache-2.0) is a ~5B single-stream DiT on Qwen3.5 9B and the FLUX.2 small
VAE. ComfyUI v0.45.4, the tag turbo-offloader vendors, runs it natively. We add two engines, in
line with "two families on purpose":

- **comfy-moss-image**: ComfyUI's own files (13.3 GB), faithful to ComfyUI's templates, at
  ComfyUI's speed or better, text to image and image to image. It runs on the laptop and is part of
  the test suite, against ComfyUI.
- **moss-image**: the stock engine, on the official `OpenMOSS-Team/MOSS-Image-SFT` repo (bf16,
  ~27 GB), ComfyUI-free, with or without the offloader, **text to image only**. OpenMOSS
  publishes no edit model. ComfyUI's edit LoRA is a community one (`chinoll/Moss-Image-Edit`, made
  for a ComfyUI node), and the official pipeline excludes reference images. It is only installed
  and tested on Modal.

Both sides run torch 2.14.1+cu130, comfy_kitchen 0.2.41 and ComfyUI v0.45.4.

## What the references do

| | ComfyUI t2i template | ComfyUI edit template | Official pipeline |
|---|---|---|---|
| Transformer | `moss_image_sft_int8_convrot` | the same, plus the edit LoRA (rank 64) | SFT bf16 |
| Text encoder | `qwen3.5_9b_w4a8_gptq` | the same (text only, `Moss-Image-Edit` format) | Qwen3.5 9B bf16 |
| Steps, CFG | 25, 2.5 | 30, 4 | 50, 3 |
| Sampler, schedule | er_sde, simple (shift 5) | euler, normal | Euler, its nearest-grid shift 5 |
| Negative prompt | "" | "nsfw, worst quality, bad anatomy, watermark, jpeg artifact" | "" |
| Uncond skips layer 22 | for the first 50% (by sigma) | for the whole run | for the whole run |
| References | | the source at ~1 MP on the 16 grid, VAE encoded, `index_timestep_zero_text_start`, ref K/V cached once | |

We leave out ComfyUI's prompt upsampler (an optional TextGenerate pass), since turboCLI has no such
feature.

## Shared pipeline (stock first, comfy reuses it)

`runner/engine/moss_image.py` (stock), on the pattern of `flux3_6b.py`:

- `MODEL`: `OpenMOSS-Team/MOSS-Image-SFT` at `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`.
  `PIPELINE` / `TRANSFORMER` come from the official `pipeline_moss_image.py`, which is in the
  pinned snapshot. It is imported from the install folder (Apache-2.0, read and pinned), with no
  `trust_remote_code` round trip.
- `MODES = ("text-to-image",)`, `INFERENCE = 50`, `CFG = ("guidance_scale", 3.0)`.
- `_pipeline_class(folder)` subclasses the official `MOSSImagePipeline`. It keeps the official
  `encode_prompt`, transformer, scheduler and decode, and replaces only `__call__` with a short
  loop that core can drive:
  - `progress_bar(total=...)` plus `update()`, `callback_on_step_end` and `_interrupt`, as
    `Flux3Pipe` does. The official loop iterates a tqdm, which never calls the `update` that core
    hooks for its step lines.
  - The official CFG in fp32 with uncond `skip_layers`, behind two small hooks a subclass can
    override: `_skip(step, sigma)` (official: always) and the scheduler.
- `load()`: `ctx.backend.load_pipe(...)` with the offloader, else `from_pretrained` plus
  `apply_offload`, as in `flux3_6b.load`.

## comfy-moss-image

### turbo-offloader

- Vendor, byte for byte from v0.45.4: `comfy/ldm/moss_image/model.py`,
  `comfy/text_encoders/moss_image.py` and the `comfy/text_encoders/qwen35_tokenizer/` data folder.
  Every module they import is already vendored, and comfy_kitchen 0.2.41 has `rms_adaln` and
  `rms_rope_split_half`.
- New `offloader/moss_image.py`, on the pattern of `lumina2.py` and `qwen_image.py`:
  - `detect`: mirrors `model_detection.py:1050-1058` and uses `adapter.count_blocks`. `process=`
    pops `rope.bands`, as `supported_models.py` does.
  - `MOSSImage(adapter.BaseModel).forward(hidden_states, timestep, encoder_hidden_states,
    encoder_attention_mask=None, skip_layers=(), return_dict=False)`, taking the official
    transformer's arguments. Timestep is ×1000 → sigma (multiplier 1.0). The skipped layers
    become `patches_replace["dit"][("single_block", i)]` identities, as
    `SkipLayerGuidanceDiTSimple` does.
  - `edit(images, vae)` context: refs VAE-encoded, then `ref_kv = preprocess_refs(refs)` computed
    once and shared by cond and uncond, then `ref_latents_method="index_timestep_zero_text_start"`.
    This mirrors `model_base.py:2947-2994` and `nodes_moss_image.py`.
  - `lora_keys`: mirrors `lora.py:361-371`. `TextEncoder(adapter.CLIP)` uses
    `text_encoders/moss_image` (the last hidden state, its user/assistant/think template).
- `adapter.py`: a `FlowSampler`, ComfyUI's KSampler for a flow model behind the diffusers scheduler
  calls (`set_timesteps`, `timesteps`, `step`). It covers:
  - the `simple` and `normal` schedulers (`samplers.py:645-693`, shift from the model);
  - `euler`, and `er_sde` (`k_diffusion/sampling.py:1593-1656`) with its first-sigma offset;
  - ComfyUI's per-step noise: `torch.Generator(device=x.device)` seeded with the seed (+1 on CPU).

  It is model-agnostic, about 50 lines, each part cited. The vendored tree has no k_diffusion and
  doesn't need all of it.
- Register `"moss_image"` in `_COMFY_MODELS`. Update `resync.md` (the new files and the mirrored
  lines), `implementation.md` and `turbo-offloader.pro`.

### turboCLI

`runner/engine/comfy_moss_image.py`:

- `MODES = ("text-to-image", "image-to-image")`.
- `INFERENCE = {"text-to-image": 25, "image-to-image": 30}` and
  `CFG = ("guidance_scale", {"text-to-image": 2.5, "image-to-image": 4.0})`.
- `COMFY` from `Comfy-Org/MOSS-Image` (transformer, text_encoder, vae, lora).
- `SCAFFOLD`: the official repo's `pipeline_moss_image.py`, `vae/config.json` and `scheduler/*`.
  These are small files, no weights.
- `load(ctx, params)`:
  - `stock._pipeline_class(scaffold)` subclassed, as `comfy_flux3_6b.py` does;
  - `encode_prompt` uses ComfyUI's text encoder;
  - the sampler per mode: er_sde + simple, or euler + normal;
  - `_skip` by sigma (first 50% for text to image);
  - the negative prompt per mode;
  - `__call__(image=...)` wraps the loop in `transformer.edit(refs, self.vae)`, with the source
    scaled to ~1 MP on the 16 grid as `ImageScaleToTotalPixels` does.
- **The edit LoRA only for image to image:** prepended in `load()` when
  `params["mode"] == "image-to-image"`, with `extra_key(params)` returning the mode so the pipe
  reloads on a mode switch. This follows the `_lightning_angles` precedent.
- **VAE:** `AutoencoderKLFlux2.from_config(scaffold vae config)` with ComfyUI's
  `flux2-vae_small` file. Its LDM keys (`encoder.quant_conv`, `decoder.post_quant_conv`, `bn.*`)
  are mapped through diffusers' `convert_ldm_vae_checkpoint`, extending
  `comfy_flux2_4b._build_vae` rather than writing a second loader.

`runner/core.py` (a few lines): `INFERENCE` and the `CFG` value may be a dict keyed by mode. Every
existing engine keeps its plain value.

## Tests

- `test/comfy.py`: `ksampler()` gains `cfg`, `sampler` and `scheduler` (defaults as today). A
  `moss` builder covers both modes: UNET/CLIP (`moss_image`)/VAE loaders,
  `SkipLayerGuidanceDiTSimple` (layer 22; 0 to 0.5, or 0 to 1), `CLIPTextEncode`, or
  `TextEncodeMOSSImageEdit` with `LoraLoaderModelOnly` and `ImageScaleToTotalPixels` for edits, and
  `EmptyFlux2LatentImage` at the output size. Register it in `GRAPHS` for both modes.
- `test/run.sh` and `test/modal.sh` need nothing: the cases come from `MODES` and `COMFY`, and an
  engine that isn't installed is skipped. moss-image is installed on the Modal Volume only
  (`modal.sh install moss-image`), so the laptop skips it.

## Docs and lists

- `engine.md` (both tables), `README.md` (supported engines), and `implementation.md` (engine
  notes, the stock/comfy pair, the core per-mode note, Doc records).
- `bash/turbo/install.sh`, `text-to-image.sh`, `image-to-image.sh` and `bash/turbo/README.md`
  usage lists.
- `turboCLI.pro`: the two engines and the doc. Fix the stale entries while there (it lists a
  removed `comfy_krea2_turbo_realism.py` and misses the two Krea 2 edit engines).
- `doc/moss-image-plan.md`: this plan, without local paths.

## Changed after

- **The VAE is ComfyUI's own (`comfy_vae`), not diffusers' on ComfyUI's file.** The edit came out
  as another picture (16.7 dB from ComfyUI's), though one transformer step was bit-identical with
  ComfyUI's, the LoRA and a reference included. The first step's inputs differed by 1%: the
  reference latent from diffusers' VAE in bf16 against ComfyUI's in fp16. The edit amplifies that
  (26% by its fourth step), as ComfyUI's own does when its VAE changes precision. The offloader
  now mirrors sd.py's FLUX.2 VAE branch (vendoring `ldm/models/autoencoder.py`, what it imports,
  and `diffusers_convert.py`, since Comfy-Org's file is in diffusers' layout), bit-identical to
  ComfyUI's VAE. `comfy_flux2_4b._build_vae` stays as it was.
- **The reference's keys and values are computed before sampling**, as `extra_conds` does
  (`_before_sampling`), not in the first step, which ComfyUI's sampling time leaves out.
- `MOSSPipe`'s seams are a skip window by sigma (`skip_sigmas`, the plan's `_skip(step, sigma)`),
  `negative_prompt`, `_decode` and `_before_sampling`. `TEMPLATE` names its fields (sampler,
  scheduler, negative, skip), and `FlowSampler` is about 100 lines, `er_sde`'s stages included.
- `SCAFFOLD` is `model_index.json` (which the test suite looks for), the pipeline file and the
  transformer's config. With ComfyUI's VAE and sampler, the VAE config and `scheduler/*` aren't
  read. `MOSSImage.forward` takes no `encoder_attention_mask`, since `MOSSPipe` passes none.
- Two differences from ComfyUI, by design: CFG combines the velocities, as the official loop does
  (ComfyUI combines the denoised, equal but for rounding), and an edit renders at the requested
  size (the template's latent takes the scaled source's).
- `test/modal.sh` brings each run's images back with its logs, and `exec --gpu` runs a command on
  an L40S.

Results on the L40S at 1024² (seed 42), turbo against ComfyUI: text to image 9.17 s (9.28 s),
40.4 dB; image to image 11.17 s (11.13 s), 30.8 dB. moss-image samples in 33.4 s.

## Verification

1. **Laptop:** `install.sh comfy-moss-image` (13.3 GB). Then
   `test/run.sh --size 512 --comfy <ComfyUI> --repeat 2 comfy-moss-image`, for both modes against
   ComfyUI: PSNR in the 40s dB and sampling at parity or better. The edit's ~2 GB ref K/V cache is
   the VRAM risk to watch on 4 GB.
2. **Determinism:** a second run is `SAME`, and `--record` adds the laptop reference lines.
3. **Regression:** `test/run.sh comfy-flux2-4b comfy-z-image-turbo` stays `SAME` after the core
   and adapter changes.
4. **Modal:** `modal.sh install comfy-moss-image moss-image`, then
   `modal.sh run --record --repeat 2 --comfy /opt/ComfyUI comfy-moss-image moss-image` on L40S. The
   stock engine renders text to image with and without the offloader; its image is checked by eye
   against the official README's settings. Then CPU at 256² for comfy-moss-image once.
5. Add the L40S numbers to turbo-offloader's Benchmarks table.
