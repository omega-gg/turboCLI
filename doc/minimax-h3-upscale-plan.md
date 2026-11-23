# MiniMax H3 upscale engines: a saved take re-rendered at 2x

## Context

The experimental H3 (`msvc/experimental/h3`, `remote/gen.py --from-latent ... --hires 2`) upscales
a take it liked from its saved latents: the community **MiniMax H3 Latent Upscaler (3D)** node
(`MinimaxH3LatentUpscaler3D`, LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler @ 40316cf, MIT)
enlarges the video latent x2, then H3 refines it at the new size (same conditioning re-applied, 8
steps, denoise 0.45); the soundtrack stays the take's. turboCLI now saves a take's latents
(`latent=save`: `<stem>.video.latent` / `.audio.latent`, SaveLatent's format, the settings in the
metadata). Goal: engines that take those latents through image-to-video's input references and
render the take again at a higher resolution, identical to that graph in ComfyUI on the same GPU.
Faithful (ComfyUI v0.45.4 + the pinned node), efficient, clean, simple.

**Engines** (each a declaration, BASE the take's engine, so the refine runs the same files):
- `comfy-minimax-h3-upscale`, BASE comfy-minimax-h3;
- `comfy-minimax-h3-quality-upscale`, BASE comfy-minimax-h3-quality (the experiment's);
- `minimax-h3-upscale`: **not built** (see "minimax-h3-upscale" below).

**Versions** (checked 2026-11-23): the node's latest commit is f34cadb (2026-11-10); its 3D
node is byte for byte the pinned 40316cf's (the later commits are the 2D node and the READMEs).
`3d_conv_v1` is still the only 3D model. A 2D one (`2d_conv_v1`, 2026-11-10) is lighter and
better at most scales by its author's comparison, but the 3D leads at its trained scales, 2x
among them (84.7-89.1% less error than trilinear against 83.0-86.4%), and is a little steadier
in time; the experiment's approved results are 3D's: the 3D model at 2x. Its Hugging Face file
is the same at the latest revision (85a0187) as at the experiment's (3f941d5): sha256 043e5a48...,
690,592,672 bytes; the latest is pinned.

## CLI

`image-to-video.sh comfy-minimax-h3-quality-upscale cuda latent=take.video.latent take-2x.mp4`

- A new input role **`latent=<stem>.video.latent`**: the take's latents (its `.audio.latent` beside
  it, by name, as gen.py finds it). Required by the upscaler engines, refused by every other.
- **The latent alone** (decided after the first draft): core makes the request of the take's
  settings, saved in the latents' metadata (`core._take`), as gen.py's `--from-latent` reads its
  `.json`: the prompt, inputs (frames, references, guide), LoRAs, duration and seed are the
  take's; its options too (the scheduler, `attention=cudnn`), under the request's, `latent=save`
  the request's alone; the output twice the take's size (the factor the experiment tried). The
  take's engine and `-upscale` must name the engine (`ValueError` otherwise, before loading).
  The script's defaults (duration 5, 512x512, seed -1) cannot be told from typed values, so the
  command's are not used.
- Steps default to 8 (`INFERENCE`), the scheduler is the engine's (simple; beta with references on
  -quality), denoise 0.45: gen.py's `--hires` defaults.

## turbo-offloader

- **`offloader/minimax_h3_upscaler.py`** (new, ~120 lines): the node's model and call, mirrored
  from `nodes/minimax_h3_latent_upscaler_3d.py` @ 40316cf as the experiment runs it:
  `LatentResizer3D` (+ `ResBlockEmb3D`, `TemporalConv`; same module tree, so the state dict loads
  strict), the arch read off the file (`_detect_arch`), H3's latent mean/std normalization, the
  target size and effective scale as `execute` computes them (align 32), one pass (chunking off),
  fp16. `LatentUpscaler(path)`: weights on CPU in fp16; `__call__(latent, width, height)` moves to
  the compute device, runs, returns float32 on CPU and moves itself back (`force_unload`), then
  `mm.soft_empty_cache()`. Places itself, as `ComfyVAE` does (`_finalize_pipe` leaves it alone).
- **`offloader/__init__.py`**: `comfy_latent_upscaler(path)` beside `comfy_vae`.
- **`adapter.FlowSampler.set_timesteps(steps, device=None, denoise=1.0)`**: BasicScheduler's
  denoise (nodes_custom_sampler.py: `int(steps / denoise)` sigmas, the last `steps + 1`).
- `resync.md` (the node as a pinned external source, its file:lines), `implementation.md`, `.pro`.

## turboCLI

- **`runner/engine/_options.py`**: `latent` in `ROLES`; `parse_references` returns
  `out["latent"]` (one at most), apart from `references` (so `extra_key`'s workflow, picking the
  ref2va transformer, is unchanged).
- **`runner/video.py`**: `load_latents(path)` -> (video, audio) float32 arrays, LoadLatent's read
  (nodes.py:565-582: `latent_tensor`, x 1/0.18215 without `latent_format_version_0`), the audio
  file found from the video's name. Torch free, beside `save_latents`.
- **`runner/core.py`**: `_take(params)`, the request made of the take's settings before anything
  loads (see CLI), one rule refusing a take on any other engine; `_video_inputs`:
  `take=load_latents(path)` when given (`loading input:`).
- **`runner/engine/comfy_minimax_h3.py`** (the pipe; no new class):
  - `__init__(..., upscaler=None)`; `load` adds `upscaler=ctx.backend.comfy_latent_upscaler(...)`
    when the engine's files have the `upscaler` role (no new load parameter).
  - `_generate(..., take=None)`: with it, the video latent through `self.upscaler` to
    `width x height`; shapes as now from size and duration, checked against the upscaled video
    and the audio latent (`ValueError`: the latents do not match their take's settings);
    the conditioning as now at the new size; `set_timesteps(steps, denoise=REFINE_DENOISE)`; the
    initial latents CONST's `noise_scaling`: `s0 * noise + (1 - s0) * pack(video, audio x
    audio_scale)` (samplers.py KSAMPLER, model_base process_latent_in), the noise RandomNoise's as
    now; after sampling the video decoded, the **audio decoded from the take's audio latent**
    (gen.py: stage 1's soundtrack). `latent=save` keeps working (the refined video latent, the
    take's audio).
  - A `_pack(tensors)` helper shared with `prepare_latents`; `REFINE_DENOISE = 0.45`.
- **`runner/engine/comfy_minimax_h3_upscale.py`** (new, declaration):
  `BASE comfy-minimax-h3`; `COMFY = dict(base.COMFY, components=base.COMFY["components"] +
  [UPSCALER])` with `UPSCALER = {"role": "upscaler", "repository":
  "LBH-123-AI/Minimax_h3_latent_Upscaler", "revision": "3f941d5d...", "filename":
  "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors", "path": "latent_upscale_models/..."}`
  (pinned as h3_modal.py pins it); `MODES = ("image-to-video",)`; `INFERENCE = 8`;
  `UPSCALE = True` (named and tagged as the image upscalers are: `check --modes` lists it
  `<id> upscaler`, so a host tells it from a generator); `check(params)` requiring `latent=`.
- **`runner/engine/comfy_minimax_h3_quality_upscale.py`** (new): the same on BASE
  comfy-minimax-h3-quality, reusing the first's `UPSCALER`, `MODES`, `INFERENCE`, `UPSCALE`,
  `check`.
- **`runner/engine/__init__.py`**: the UPSCALERS note gains the image-to-video engines that
  enlarge a take's latents.
- Satellites: `bash/turbo/install.sh` list, image-to-video's usage (`latent=` role, an example),
  bash README, `turboCLI.pro`.

## minimax-h3-upscale

Not built: no faithful way to do it.
- diffusers' modular MiniMax H3 (0.43.0) has no strength / denoise / sigmas input:
  `MiniMaxH3SetTimestepsStep` only takes `num_inference_steps`, and `latents` is "pre-generated
  noise" used verbatim. A refine needs a patched `set_timesteps` per workflow slicing `timesteps`,
  `audio_timesteps` and `row_timestep_plan`, plus hand-noised video and audio latents: a pipeline
  of our own, with no reference (ComfyUI or diffusers) to check it against.
- Its latents are diffusers' normalized ones ((raw - mean) / std), not the raw VAE latents the
  community upscaler was trained on and that `latent=save` writes; minimax-h3 saves no latents.
- A take from minimax-h3 can still be upscaled by the comfy upscalers only if re-rendered with a
  comfy engine: out of scope. Revisit if diffusers adds a v2v / strength path.

## Docs

`doc/minimax-h3-upscale-plan.md` (this plan; results after); MANUAL ("Upscaling a take": save,
then the upscaler engine with `latent=`, cost); engine.md (two rows, sizes: the base's + the
upscaler file); implementation.md (the role, the refine path, engine rows); the offloader's
resync.md and implementation.md.

Conventions: 99 columns, CRLF where the file has it, no top-level torch in engines, commits on
request only, no Claude attribution, no build.sh pin bump.

## Verification (incremental, lean on Modal; files from Hugging Face fetched on Modal)

Each step runs only once the one before passed; about $1.80 in all. Performance is compared
with ComfyUI on the same settings: turbo must be on par or faster. Both sides start cold in
their own process, without the pinned copy (`OFFLOADER_DISABLE_PINNED_MEMORY=1`,
`--disable-pinned-memory`), and are timed from launch to the saved video, with the sampling
per step, the latent upscale and the decode apart.

0. Copy this plan to `turboCLI/doc/minimax-h3-upscale-plan.md` first.
1. **Locally** (done): lint; `parse_references` with `latent=`; the engines' inheritance (files,
   options, `load`, crf); the request made of a take's settings (`_take`, torch and the file
   read stubbed): the take's prompt, inputs, seed and duration, twice its size, its options under
   the request's; a take of another engine and a latent without settings refused.
2. **Modal CPU** (a few cents), the parts before any GPU time:
   - install `comfy-minimax-h3-quality-upscale` (only the 0.7 GB upscaler file is new);
   - the weights' latest Hugging Face revision against the experiment's 3f941d5, the latest
     pinned if its 3D file is the same;
   - the mirrored `LatentUpscaler` against the node's own `execute` on a small random latent:
     identical;
   - ComfyUI's `BasicScheduler`, `CONST` and `ModelSamplingAV` read, to confirm the refine's
     start (the noise mixed into a given latent): fixed before going on if H3 does it otherwise.
3. **Small identity on a B200** (~$0.40), turbo against ComfyUI, on comfy-minimax-h3 (its 40 GB
   int8 files load faster than -quality's 98 GB, three cold loads; the same code, the int8
   upscale engine covered too, -quality's identity checked at full size in step 4):
   - a 512x512, 1 s comfy-minimax-h3 take with `latent=save`, in the Indy workflow (a first
     frame, the knight, and a voice reference);
   - comfy-minimax-h3-upscale on `latent=` alone, to 1024x1024;
   - ComfyUI's graph on the same latents (gen.py's `--from-latent --hires 2`: two LoadLatent,
     LTXVConcatAVLatent, the node cloned at 40316cf into ComfyUI's custom_nodes with its model
     from the Volume, the Ref2VA conditioning at 2x, BasicScheduler beta 8 / 0.45, VAEDecode,
     stage 1's audio): frames and sound identical. It also runs `latent=save` and `_take` for
     real;
   - both on ComfyUI's default attention (flash), timed: a first read of the speeds.
4. **The Indy 2x on a B200** (~$1.50), the look, the speed and the cost:
   - the Indy take again with `latent=save` and `attention=cudnn`, 1376x576;
   - comfy-minimax-h3-quality-upscale on its `latent=`: 2752x1152;
   - ComfyUI's graph of step 3 on the same latents at 2752x1152, with the experiment's
     `cudnn_attention.py` node (cuDNN first, as `attention=cudnn`): identical at full size, and
     turbo's time against it, launch to saved video, the steps, the upscale and the decode;
   - both copied to Desktop\image\minimax-h3-video, next to `indy_experiment-2x.mp4` (the
     experiment's approved `20261005-215715-scope1376-native-c-2x-fp16-1pass.mp4`), compared by
     PSNR and by eye: close, not the same bits (ComfyUI v0.38.0 against v0.45.4);
   - the cold cost, launch to saved video (the experiment measured ~$0.65 for 5 s).
5. Results into this doc and the offloader's implementation.md.

Not tested, to keep the spend down (each can be added):
- the first and last frame workflow without references: the same refine, the conditioning
  already identical in a normal take;
- a warm second upscale in the same process (~$0.30), for the warm price alone.
- a 10 s take at 2x (~$1.10, estimated from the 5 s steps): whether the upscaler's single
  pass and the decode at 241 frames of 2752x1152 fit on the B200.
