# MiniMax H3 video: text-to-video and image-to-video

## Context

turboCLI makes images only. Video is the next step: two new modes, `text-to-video` and
`image-to-video`, with MiniMax H3 as their first engines. The model has already been run through
ComfyUI in `msvc/experimental/h3`, and its `remote/gen.py` "base" graph is the reference.
`comfy-minimax-h3-quality` must give the same video as that graph for the same seed. Every comfy
engine must match its ComfyUI graph and be as fast as ComfyUI. Upscaling comes later. Measured on
Modal, keeping the spend low.

Decisions (asked). Each engine defaults to its official source:
- **`minimax-h3`:** what MiniMax publishes on `MiniMaxAI/MiniMax-H3`, through diffusers'
  `MiniMaxH3ModularPipeline` (v0.43.0, already pinned): their files, dtype, default steps (50) and
  limits. On Modal: installed, then one short run per mode, with no ComfyUI comparison.
- **`comfy-minimax-h3`:** what ComfyUI's official template ships (`video_minimax_h3_t2v` / `_i2v`,
  templates package 0.1.135), from `Comfy-Org/MiniMax-H3`:
  `minimax_h3_fl2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`,
  `minimax_h3_video_vae_int8_convrot` and `minimax_h3_audio_vae_fp32`. Template settings: 20 steps
  with the Turbo LoRA off, res_multistep, simple, the first frame given straight to the node,
  SaveVideo's default quality (crf 18). Identical to ComfyUI running that template.
- **`comfy-minimax-h3-quality`:** the experiment's setup. `BASE comfy-minimax-h3` with the bf16
  files (`minimax_h3_fl2va_pruned_bf16` ~40 GB, `qwen3vl_32b_minimax_h3_bf16` 51 GB,
  `minimax_h3_video_vae_fp16` 5.2 GB, audio fp32), crf 12, and the keyframes centre-cropped first,
  as gen.py does. Identical to ComfyUI running the experiment's base graph.
- **Identity:** same graph, same GPU. Each comfy engine against its ComfyUI graph, re-run in
  ComfyUI v0.45.4 on the same Modal L40S as turbo, with the same files, seed, prompt, size and
  duration. The decoded frames and audio must be identical. No B200: the existing base mp4s came
  from ComfyUI v0.38.0 on a B200 with cuDNN attention first, so their bits can't be reproduced
  elsewhere.
- **The soundtrack stays:** AAC 32 kHz stereo, muxed as ComfyUI's CreateVideo/SaveVideo do.
- **Main risk:** the template's NVFP4 text encoder and int8 video VAE on an L40S. The offloader
  runs quantized files through ComfyUI's own `mixed_precision_ops`, which dequantizes in software
  where the GPU can't run a format, as ComfyUI does. The first Modal run checks it.

What the experiment's base graph runs (FL2VA, no references, no hires), and what we mirror:
- `UNETLoader` + `CLIPLoader(type minimax)` + video and audio `VAELoader`.
- `MiniMaxH3ImageToVideo(prompt, width, height, length)`, with optional `first_frame` /
  `last_frame`, each through `ImageScale(lanczos, w, h, crop=center)`.
- `BasicScheduler(simple, steps 20, denoise 1)`, `BasicGuider` (no CFG),
  `RandomNoise(seed)`, `KSamplerSelect(res_multistep)` into `SamplerCustomAdvanced`.
- `VAEDecode` + `VAEDecodeAudio`, then `CreateVideo(24 fps)` and `SaveVideo(mp4, h264, crf 12)`.
- Length is 24 fps snapped up to the 17k+5 grid: `n = max(5, round(s*24)); n + (5 - n%17)%17`.

## CLI: the two modes

- **`bash/turbo/text-to-video.sh <engine> <renderer> <output video> [prompt] [duration] [width]
  [height] [seed] [inference] [offload] [slicing] [loras] [server]`**: text-to-image.sh's
  arguments, with `duration` (seconds, default 5) right after the prompt.
- **`bash/turbo/image-to-video.sh <engine> <renderer> <input images> <output video> [prompt]
  [duration] [width] [height] [seed] [inference] [offload] [slicing] [loras] [options] [server]`**:
  image-to-image.sh's arguments plus `duration`. The first input image is the first frame, an
  optional second one the last frame. The `options` slot is kept for parity; no option exists
  yet, so a given one is refused.
- Both are copies of their image counterparts. The diff is the `duration` argument (curl field and
  `--duration`), the mode value and the usage text.
- `runner/cli.py` adds `--duration` (default `"5"`) to the params dict; the server passes it
  through as it is.

## turboCLI core (`runner/core.py`), video out

- **Inputs:** `image-to-video` loads its input images as `image-to-image` does, through one shared
  branch (`mode in IMAGE_INPUT`).
- **Pipe call:** a video mode passes `duration=float(params["duration"])`. The engine turns it into
  frames, since the frame grid is the model's.
- **Result:** a video engine returns `SimpleNamespace(videos=[frames], audio=(waveform, rate),
  fps=24)`, with frames as uint8 `[T, H, W, 3]` and the waveform as `[2, L]`. Core hands it to a
  new torch-free **`runner/video.py`** `save(path, frames, fps, audio, rate)`, then emits `Saved:`.
  The wire contract stays the same.
- **`runner/video.py`** mirrors ComfyUI's `VideoFromComponents.save_to`
  (`comfy_api/latest/_input_impl/video_types.py:283-415`) with PyAV:
  - mp4 with `movflags use_metadata_tags+faststart`; h264 at rate 24, `yuv420p`, at the engine's
    `CRF`. That is an optional engine symbol, inherited, defaulting to 18 (SaveVideo's h264
    default in v0.45.4); comfy-minimax-h3-quality sets 12, the experiment's.
  - BT709 primaries and colorspace, IEC61966-2-1 transfer, limited range; frames reformatted
    rgb24 → yuv420p with BT709.
  - AAC at the source rate, stereo, trimmed to `ceil(rate/24 * frames)` samples, then flushed.
  - No metadata tags (ComfyUI writes the prompt there), so turbo's files are byte-stable, and the
    comparison with ComfyUI decodes the frames and audio.
- **`build.sh`** pins PyAV (`av`, the version ComfyUI v0.45.4 resolves). `test/comfyui.sh` installs
  that same pin into ComfyUI's venv, so both sides encode with one build.

## turbo-offloader: H3 video as ComfyUI runs it (v0.45.4)

The H3 audio-video DiT (`comfy/ldm/minimax/model.py`), the video VAE class (`vae.py`) and the text
encoder (`text_encoders/minimax.py`) are already vendored, byte-identical to v0.45.4. Missing:
- **Vendor** `comfy/ldm/minimax/audio_vae.py` byte for byte, plus whatever it imports that isn't
  there (the BigVGAN vocoder pieces).
- **`offloader/minimax_h3.py`** (new), after `minimax_h3_image.py`, which stays untouched:
  - `detect(sd)`: model_detection.py:401-434, the audio branch (`image_model minimax_h3`,
    `audio_latents_dim`).
  - `sampling()`: shift 12, plus the AV settings (`audio_scale = 12/3`), from
    supported_models.MiniMaxH3:986-1012 and model_sampling.ModelSamplingAV:338-357.
  - `class MiniMaxH3(adapter.BaseModel)`, mirroring model_base.MiniMaxH3:2226-2427:
    - `memory_usage_factor 0.114`.
    - `extra_conds` builds the payload: `audio_scale`, the layout with the audio length, the
      keyframes and `cond_video_latents`, `seed`, `latent_shapes`.
    - `forward` takes the video and audio latents, scales the audio by `audio_scale` on the way
      in and out (process_latent_in/out), and returns both velocities.
    - `load` builds `MiniMaxH3Model`.
  - The text encoder reuses `minimax_h3_image.TextEncoder` with the video path's settings, which it
    gains as arguments: label `"<Picture %d>: "`, images at canvas size (text_encoders/minimax.py,
    nodes_minimax_h3.py:172-173). The image engine keeps its own settings.
  - Registered in `_COMFY_MODELS`.
- **`adapter._comfy_vae_model`:**
  - The H3 video VAE branch (sd.py:1050-1125): latent 24 channels, 3 dims, ratios, chunked I/O,
    fp16/fp32, `handles_tiling`, identity `process_output`, its memory estimates.
  - The audio VAE branch (sd.py:1127-1147): 32 channels, 2 dims, 32 kHz, ratio 800, fp32, its
    estimates.
  - `ComfyVAE.decode` passes `output_buffer` to a `comfy_has_chunked_io` model (sd.py:1384-1394).
- **`adapter.FlowSampler`:** `res_multistep` at eta 0 (k_diffusion/sampling.py:1462-1528). Euler on
  the first step and onto sigma 0, then the 2nd-order phi step with `old_denoised`.
- `resync.md` (vendored files, mirrored lines), `implementation.md`, `turbo-offloader.pro`.

## turboCLI engines

**`runner/engine/comfy_minimax_h3.py`** (the template):
- Symbols:
  - `ID comfy-minimax-h3`, `TYPE minimax-h3`, `TRANSFORMER` (offload-wired marker).
  - `MODES ("text-to-video", "image-to-video")`, `CFG ("guidance_scale", 1.0)`, `INFERENCE 20`.
  - `COMFY`: Comfy-Org/MiniMax-H3 at a pinned revision, four roles (transformer, text_encoder,
    vae, audio_vae), the template's files.
- `_pipeline_class(crop=None)` builds a `ComfyH3VideoPipe`. Each step mirrors one node:
  - `frames(duration)`: the template's length expression. The latents follow `_av_latent`
    (nodes_minimax_h3.py:60-63, 97-102): video `[1, 24, lt, H/16, W/16]`, audio `[1, 32, 2, at]`.
  - Keyframes: the first input as the first frame, and an optional second as the last frame.
    With `crop="center"` (the bf16 engine) each goes first through `ImageScale` lanczos with
    centre crop, as gen.py does. Then the node's own resize (first frame stretched, last frame
    centre-cropped, nodes_minimax_h3.py:159-170) and the video VAE encode of each, at index 0
    and `frames - 1`.
  - Noise as `prepare_noise` (sample.py:22-37): one CPU generator on the seed, float32 `randn`,
    video then audio.
  - Sampling: video and audio packed into one flat latent, as samplers.py:1281-1284 does, so
    `res_multistep` keeps one history. `simple` sigmas at shift 12. Progress, callback and cancel
    as the other comfy pipes.
  - Decode: the video VAE gives frames truncated to uint8 as `save_to` does. The audio VAE gives a
    waveform, which `VAEDecodeAudio` normalizes: std × 5, floored at 1 (nodes_audio.py:99-113).
- `load(ctx, params, crop=None)` calls `load_pipe_comfy(_pipeline_class(crop), {"model":
  "minimax_h3", ...}, {"model": "minimax_h3", ...}, {"scheduler": comfy_sampler("minimax_h3",
  "res_multistep", "simple"), "vae": comfy_vae(video), "audio_vae": comfy_vae(audio)})`. It raises
  without a backend, as the other comfy engines do.

**`runner/engine/comfy_minimax_h3_quality.py`** (the experiment):
- `BASE comfy-minimax-h3`, its own `COMFY` (the bf16 transformer and text encoder, the fp16 video
  VAE, the same audio VAE) and `CRF 12`.
- `load(ctx, params)`: `base.load(ctx, params, crop="center")`.

**`runner/engine/minimax_h3.py`** (stock, MiniMax's release):
- `MODEL MiniMaxAI/MiniMax-H3` at a pinned revision, in the dtype they publish. `allow_patterns`
  take the t2va/fl2va folders only, not `transformer_ref/`.
- `PIPELINE diffusers:MiniMaxH3ModularPipeline`,
  `TRANSFORMER diffusers:MiniMaxH3Transformer3DModel`.
- `INFERENCE 50`, MiniMax's and diffusers' default.
- An `H3VideoPipe` modeled on `minimax_h3_image.H3Pipe` (`minimax_h3_image.py:69-223`): the
  `text_encoder` sub-block split out behind `encode_prompt`, the denoise progress bar rerouted,
  `_execution_device` following, and `load_pipe` on the offloader.
- It returns `videos` (frames), `audio` and fps 24.
- diffusers' own limits stand: 5 to 15 s, and its canvas. Its errors come back as they are.

**Satellites:** `install.sh` and the two new wrappers' engine lists; `check-model.sh` and
`bash/turbo/README.md` MODES/ENGINES lists; `turbo.pri`; `turboCLI.pro`; `engine/__init__.py`
(the MODES note).

## Tests (`test/`)

- **`run.sh`:**
  - `turbo_cases` adds the two modes.
  - Video cases write `.mp4`, prompt a short motion and use `knight.png` as the first frame.
  - They run 5 s at the suite's `--size`; the md5 is of the mp4 file.
  - Timings come from the same lines.
- **`comfy.py`:**
  - A `h3_video` graph, keyed for both modes. For comfy-minimax-h3 it is the template's nodes
    (the first frame straight to the node, SaveVideo crf 18). For comfy-minimax-h3-quality it is
    gen.py's base graph node for node (ImageScale centre crop, SaveVideo crf 12).
  - `compare()` decodes both mp4s with PyAV and reports `identical` when every frame and audio
    sample match, else the frames' PSNR.
- **`modal.sh`:** copies `*.mp4` back with the PNGs.
- References are recorded for the new lines.

## Docs

- `doc/minimax-h3-video-plan.md` (this plan, copied first; results after).
- `MANUAL.md`: "Generating a video".
- `engine.md`: Text-to-video and Image-to-video tables, with size, steps and requirements (the
  MiniMax license terms checked on the repos).
- `implementation.md`: the modes, the params `duration`, the generate steps, `runner/video.py`, the
  engine rows, the tests.
- `README.md` supported engines, `feature.md` (engine count, video), `test/README.md`,
  `dummy.md` where it lists modes.
- turbo-offloader's `resync.md` and `implementation.md`.
- turbopixel stays out: video in its interface comes later.

Conventions: 99 columns, CRLF where the file has it, license headers, `# NOTE:`, no top-level torch
in engines or `runner/video.py`. Commits on request only, no Claude attribution, pushed state
checked first.

## Verification (Modal, lean: ask before anything beyond step 3)

1. **Locally:** pyflakes, the offloader's import smoke test, and `runner/video.py` writing a
   synthetic clip with audio that PyAV reads back (frames, rate, samples).
2. **Install on Modal** (CPU container): `test/modal.sh install comfy-minimax-h3
   comfy-minimax-h3-quality minimax-h3` into the `turbo-models` Volume. That is about 43 GB for the
   template's files, 97 GB for bf16 and 135 GB for MiniMax's release. Revisions are pinned from
   the HF API there, since Hugging Face is blocked on this machine.
3. **Identity and speed, comfy-minimax-h3**, on the big L40S container (224 GiB):
   - First `test/modal.sh run --big --size 512 --comfy /opt/ComfyUI comfy-minimax-h3`: one run per
     side per mode, 5 s. It must report `identical` against ComfyUI running the template, with
     sampling at parity.
   - Then once with `--warmup --repeat 2 --record` for the references.
   - Spend is reported after each run (`modal billing report`); `modal app list` is empty at the
     end.
4. **comfy-minimax-h3-quality:** the same first run, one per side per mode. It must report
   `identical` against ComfyUI running the experiment's base graph.
5. **minimax-h3:** one `test/modal.sh exec --big` run per mode at 512, 5 s, at its 50 steps. It
   must write a playable mp4 with sound and print `Saved:`.
6. Results (timings, identity, cost) go into `doc/minimax-h3-video-plan.md` and the offloader's
   `implementation.md`.

## As built

- The plan as written, measured on Modal's L40S with 224 GiB of RAM (`run --big`), 512x512, 5 s
  (124 frames), seed 42, the knight as image-to-video's first frame.
- **comfy-minimax-h3 against ComfyUI running the template**: identical in both modes, frames and
  sound decoded. With `--warmup --repeat 2`, sampling 38.36 s against 38.86 s (-1%) for text to
  video and 41.69 s against 41.39 s (+0%) for image to video; from launch to the saved video 60.3 s
  against 61.6 s and 64.8 s against 64.8 s. The first single run read +12% for text to video:
  turbo goes first and its first step read the weights off the Volume (10.9 s, then 1.75 s per
  step). Its L40S reference was dropped later: H3 is the `--video` suite's, on a B200.
- **comfy-minimax-h3-quality against the experimental H3's base graph in ComfyUI** (gen.py's nodes,
  `--bf16-text-enc`, the first frame center cropped, crf 12): identical in both modes. Sampling
  75.9 s and 71.9 s, ComfyUI's 84.7 s and 94.1 s, in separate containers: not a speed comparison.
- **minimax-h3** (diffusers, MiniMax's weights): an mp4 with its soundtrack in both modes, 49
  steps of 3.3 to 3.6 s after a first step reading the weights (73 s, 35 s), 3 min 50 s and
  3 min 22 s of sampling.
- **runner/video.py** writes the same bytes as ComfyUI's `VideoFromComponents.save_to` for the same
  frames and sound (crf 18 and 12, PyAV 19.0.1); res_multistep and the shift-12 schedule match
  ComfyUI's code bit for bit; the keyframe sizing matches `common_upscale` pixel for pixel.
- Later, comfy-minimax-h3-quality's center crop was removed (the frames go to the node as
  given, cropped beforehand if need be), and with it gen.py's ImageScale in `test/comfy.py`.
- Two fixes on the way: the offloader places the audio VAE beside the VAE (`_finalize_pipe`), and
  `test/comfy.py` reads a variant's frames and steps off its BASE.
- The installs are 40 GB (the template's files), 98 GB (bf16) and 143 GB (MiniMax's diffusers
  layout); the plan's estimates were 43, 97 and 135. Both repos are under the MiniMax H3 Community
  License: the EU, the UK, South Korea and the US excluded, written authorization above $20M of
  yearly revenue.
- Modal: about $5 for the day's runs (the installs, five runs and two repeats), nothing left
  running.
