# MiniMax H3 Ref2VA: references in image-to-video, and the audio guide

## Context

The H3 video engines do the experiment's FL2VA workflow (text, first and last frames). The
experiment's Indy take, `msvc/experimental/h3/out/20261005-195354-scope1376-native-c.mp4`
(`scope_native_c_job.json`), is its other workflow, Ref2VA: the `minimax_h3_ref2va_pruned_bf16`
transformer, a voice reference read as `<Audio 1>` (`neutral_30s_10s_clean.wav`, 44.1 kHz),
`MiniMaxH3ReferenceToVideo` (ref_image_size match), the first frame pinned by `MiniMaxH3AddGuide`,
the `beta` scheduler, 20 steps, 1376x576, 5 s, seed 1791222669. Goal: reproduce it with
comfy-minimax-h3-quality, identical to that graph in ComfyUI on the same GPU, faithful to ComfyUI
and the codebase, lean on Modal.

Decisions (asked):
- **No new mode: image-to-video's input images become input references**, each tagged by its
  role (`first=toto.png,last=titi.png,audio=audio.mp3`). A reference (`image=`, `audio=`) switches
  the engine to its Ref2VA workflow, as gen.py does on `--ref`.
- **Every H3 engine gets it**: comfy-minimax-h3 (ComfyUI's `video_minimax_h3_r2v` template),
  comfy-minimax-h3-quality (the experiment's Ref2VA graph), minimax-h3 (diffusers' `ref2va`
  workflow).
- **The audio guide is included**: an exact recording pinned at a time of the soundtrack
  (`MiniMaxH3AddGuide`'s audio), for lip sync.

What ComfyUI runs (v0.45.4, `comfy_extras/nodes_minimax_h3.py`):
- `MiniMaxH3ReferenceToVideo` (:259-381): each image resized ("match": down to the output's area,
  32 grid, lanczos) and VAE-encoded; each audio loaded (`LoadAudio`, nodes_audio.py:372-396),
  resampled to 32 kHz (`comfy.audio.resample`) and audio-VAE-encoded (`_encode_ref_audio`, :87-95);
  the items to the tokenizer (`minimax_ref_items`, `<Picture i>` / `<Audio j>` per type, in order),
  the latents to the DiT as `minimax_refs`.
- `MiniMaxH3AddGuide` (:182-256): an image center-cropped to the canvas and encoded at
  `frame_idx` (-1: the last frame), or an audio encoded at `frame_idx = round(at * 24)` and cut to
  what is left of the soundtrack; appended to `minimax_keyframes`.
- The template r2v (`video_minimax_h3_r2v.json`): `minimax_h3_ref2va_pruned_int8_convrot`, the
  NVFP4 text encoder, the int8 video VAE, res_multistep + **simple**, 20 steps, the Turbo LoRA off,
  reference images only. The experiment (gen.py `conditioning()`): the bf16 ref2va file,
  **beta**, AddGuide for the first and last frames and the audio guide, chained in that order.
- diffusers (`MiniMaxH3Blocks`, `ref2va`): `references` only (`MiniMaxH3ImageReference`,
  `MiniMaxH3AudioReference` with `from_file`), at most 9 images and 3 audios, and audio cannot be
  alone; no pinned frame or audio guide in that workflow.

## CLI: image-to-video's `<input references>`

- **`image-to-video.sh <engine> <renderer> <input references> <output video> [prompt] ...`**, the
  arguments unchanged but the third: comma separated, each tagged by its role:
  - `first=<image>`, `last=<image>`: the frames the video starts and ends on;
  - `image=<image>`: a picture reference, `<Picture 1>`, `<Picture 2>`... in the prompt, in order;
  - `audio=<audio>`: a voice or sound reference, `<Audio 1>`...;
  - `guide=<audio>@<seconds>`: an exact recording pinned at that time of the soundtrack;
  - an untagged path is a frame, as today: the first, then the last.
- With an `image=` or `audio=` the engine runs its Ref2VA workflow, else its FL2VA one; the frames
  and the guide go with either. The Indy take: `image-to-video.sh comfy-minimax-h3-quality cuda
  first=in_fedora.png,audio=voice.wav out.mp4 "..." 5 1376 576 1791222669`.
- The wrapper converts each path as it does the LoRAs' (keeping the `role=` and `@seconds`);
  usage, README and MANUAL say "input references". No new script, no new mode; text-to-video
  unchanged.

## turboCLI core

- image-to-video parses its references once (`first`, `last`, `image`, `audio`, `guide`; an
  untagged path a frame), loads the images (PIL) and passes the pipe `image=[first, last]` (None
  where absent), `references=[image or audio path, in order]` and `guide=(path, seconds)`.
- The workflow changes the transformer, so `extra_key(params)` folds in whether the request has
  references: the resident pipe reloads with the other transformer when it changes.

## turbo-offloader (v0.45.4, mirrored)

- **Vendor** `comfy/audio.py` byte for byte (resample; numpy, scipy, torch).
- **`adapter.load_audio(path)`**: `LoadAudio`'s load (PyAV decode, `f32_pcm`), beside
  `adapter.load_image`.
- **`minimax_h3.MiniMaxH3.references(...)`**: `MiniMaxH3ReferenceToVideo` and `MiniMaxH3AddGuide`
  on the pipeline's inputs, as `minimax_h3_image.MiniMaxH3Image.edit` mirrors its node: the
  reference images sized and encoded, the audio loaded, resampled and encoded, the ref items for
  the tokenizer, the ref blocks, the guide keyframes (image and audio).
- **`MiniMaxH3.extra_conds`** takes the refs: `payload["refs"]`, `_cond_latents`' reference and
  audio latents (`cond_audio_latents`), `PackedLayout(..., refs=)` (model_base.py:2254-2307).
- **`minimax_h3.TextEncoder.encode`** takes `minimax_ref_items` (text_encoders/minimax.py:186-243).
- **`adapter.FlowSampler`**: the `beta` scheduler (samplers.py:696-708, scipy's beta ppf).
- `resync.md`, `implementation.md`, `.pro`.

## turboCLI engines

- **comfy-minimax-h3**: a `transformer_ref` COMFY component
  (`minimax_h3_ref2va_pruned_int8_convrot`, +21 GB); `extra_key` = whether there are references;
  `load` picks the transformer and the scheduler (simple, the template's); the pipe's ref2va
  branch: `transformer.references(...)` then the same sampling and decodes as FL2VA; an audio
  guide through AddGuide in either workflow. The pipe also skips a None keyframe (a last frame
  alone).
- **comfy-minimax-h3-quality**: `transformer_ref` = `minimax_h3_ref2va_pruned_bf16` (+40 GB), the
  scheduler `beta` with references (gen.py), the frames pinned through AddGuide there.
- **minimax-h3**: `"transformer_ref/*"` in `allow_patterns` (+66 GB); a ref2va pipe class
  (`transformer_folder = "transformer_ref"`, `update_components(transformer_ref=...)`), the
  references as `MiniMaxH3ImageReference` / `MiniMaxH3AudioReference.from_file`; `first=`, `last=`
  and `guide=` refused with diffusers' reason.
- Satellites: image-to-video.sh's usage (and bash/turbo/README.md's copy), engine/__init__.py's
  VIDEO note, turboCLI.pro.

## Tests

- `test/run.sh`: an `image-to-video-ref` case of image-to-video's engines (as `image-adjust-night`
  is a second case of image-adjust), `image=knight.png` as `<Picture 1>` (no audio asset in the
  suite).
- `test/comfy.py`: the r2v graph for that case, the template's for comfy-minimax-h3
  (ReferenceToVideo, simple) and gen.py's Ref2VA for comfy-minimax-h3-quality (beta, AddGuide).

## Docs

`doc/minimax-h3-ref2va-plan.md` (this plan, results after); MANUAL ("Generating a video":
references, pinned frames, the audio guide); engine.md (image-to-video's references, sizes with
the ref2va files); implementation.md (the input references, engine rows); feature.md, test/README,
bash README; the offloader's resync.md and implementation.md.

Conventions: 99 columns, CRLF where the file has it, faithful to ComfyUI v0.45.4, no top-level
torch in engines. Commits on request only, no Claude attribution.

## Verification (Modal, lean, incremental: one run per failure, ask before step 4)

1. **Locally**: lint; `comfy.audio.resample` and `adapter.load_audio` against ComfyUI's on the
   Indy wav (bit for bit, ComfyUI's python); the beta schedule against `beta_scheduler`; the
   wrapper and core parsing of `<inputs>` with a fake engine.
2. **Install** the three ref2va files on Modal (`test/modal.sh install`, CPU, +127 GB).
3. **Identity at 512x512, 5 s**, one run per side: comfy-minimax-h3 against the r2v template with
   the knight as reference; comfy-minimax-h3-quality against gen.py's Ref2VA graph with the knight
   as first frame, a short voice reference and an audio guide. Both `identical`.
4. **The Indy take** with comfy-minimax-h3-quality at its own settings (1376x576, 5 s, seed
   1791222669, `first=in_fedora.png,audio=neutral_30s_10s_clean.wav`): turbo once, next to the
   original B200 clip (same scene and voice, not the same bits: B200, v0.38.0, cuDNN attention).
   Copied to Desktop\image\minimax-h3-video.
5. **minimax-h3** with references once at 512 (`image=knight.png,audio=<voice>`).
6. Results into the plan doc and the offloader's implementation.md.

## As built

- As planned: `<input references>` in image-to-video, the Ref2VA workflow on each H3 engine,
  the audio guide. `comfy-minimax-h3-quality` (first `comfy-minimax-h3-bf16`, the name this plan
  was written with): past its bf16 files it sets the bf16 text encoder, beta with references
  and crf 12.
- **Identity (step 3)**: comfy-minimax-h3 against the r2v template with the knight as
  reference, and comfy-minimax-h3-quality against gen.py's Ref2VA graph (the knight as first frame,
  a voice reference, an audio guide): both identical. Loading, resampling and the beta schedule
  are bit for bit ComfyUI's on the Indy voice (step 1).
- **The Indy take (step 4)**: 13.7 s a step on the L40S, 4.6 on a B200. The experiment loaded
  a node putting cuDNN attention first: with it (`attention=cudnn`, a per-video option that
  replaced a planned separate engine) 2.23 s a step, and the soundtrack bit-identical to the
  experiment's original B200 clip, the picture 45.2 dB from it (37.3 dB with flash, the sound
  differing). The gap is after sampling (ComfyUI v0.38.0 against v0.45.4): `--highvram`, as
  the experiment launched ComfyUI, gives 45.1 dB.
- **Options**: `attention=cudnn` and `scheduler=simple|beta` for the comfy engines, each in
  place of the engine's for one video; minimax-h3 refuses them. A `crop=none|center` option
  and comfy-minimax-h3-quality's center crop were removed: both engines give the node the
  frames as they are, a frame cropped to the video's proportions beforehand if need be.
  The Indy take from a pre-cropped frame (2912x1220, rows 206 to 1426) is the same file as
  from in_fedora.png: AddGuide center-crops it again, 1 row off each edge, onto the region
  it cut from the original (rows 207 to 1425; a 2912x1218 file would lose a column a side).
- **Cost on a B200** (Indy, cuDNN): 138 s for the first video in a process, 52.5 s for the
  next, the RAM at 189 GiB (the weights mapped and comfy-aimdo's pinned copy of them).
  `--disable-pinned-memory` (`OFFLOADER_DISABLE_PINNED_MEMORY=1`): 94 s and 51 s, 100 GiB,
  the same videos. `--highvram` (10 GiB once loaded, 140 s cold, other videos) is not
  mirrored.
- **Tests**: the video cases became a suite of their own, `run.sh --video`, on B200 containers
  (`modal.sh run --video`): a plain run stays on the image engines. The L40S reference was
  dropped; the B200's is not recorded yet.
- **Not done**: step 5, minimax-h3 with references.
