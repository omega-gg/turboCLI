# turboCLI features

turboCLI owes a lot to ComfyUI. Its `comfy-` engines run ComfyUI's own models, its quantized files
and its offloading, vendored in [turbo-offloader](https://omega.gg/turbo-offloader), and most of
them are measured side by side with ComfyUI's own templates. The two projects answer different
needs; this page says what turboCLI does, and when each one is the better pick.

## Top features

- As fast as ComfyUI or faster on ComfyUI's own models and files: on an L40S, sampling at parity
  or up to 16% faster, and the image sooner from launch. MiniMax-H3-Image and SeedVR2 give images
  bit-identical to ComfyUI's.
- Models larger than the graphics card, their weights streamed as needed (turbo-offloader).
- 36 engines for text to image, image editing, text and image to video, upscaling and masking:
  FLUX.2 Klein, FLUX.3 Image Klein, Z-Image-Turbo, Krea 2 Turbo, Qwen-Image 2.1,
  Qwen-Image-Edit-2511, MOSS-Image, MiniMax-H3-Image, MiniMax H3, SeedVR2, Real-ESRGAN, BiRefNet
  and InSPyReNet.
- Video with its soundtrack: MiniMax H3 films a prompt, from a first and a last frame, or from
  picture and voice references, an exact recording pinned for lip sync, and makes the sound with
  the picture, written to an mp4 exactly as ComfyUI writes it. Per video, cuDNN attention (about
  twice as fast on a B200), the scheduler and the frames' crop.
- Fast cold starts, with no torch until it is needed, and fast warm ones: a local server keeps the
  model loaded and caches prompt encodings.
- Offline once an engine is installed: nothing is uploaded and nothing is tracked.
- A ComfyUI install's model files reused, so nothing is downloaded twice.
- Reproducible: Python, every package and each model revision are pinned, so a build months later
  resolves the same stack.
- Fixes for edits: `align` undoes the zoom and shift an engine adds, `color` matches the colors
  back to the source without undoing what the edit changed.
- Upscaling with Real-ESRGAN, or with SeedVR2 to redraw the detail in one step, alpha kept.
- Background removal that can keep the subject's cast shadow, given a clean plate; edit masks,
  compositing, cutouts and trimming.
- LoRAs, several per run at any strength; RGBA output with Qwen-Image 2.1.
- From the command line, behind a local HTTP server (streamed progress, cancel, the latest request
  wins), or embedded in Python.
- A test suite that checks every engine's image stays byte-identical and no slower across
  updates, side by side with ComfyUI, locally or on Modal GPUs; the videos in a suite of their
  own, on a B200.
- Self-contained in one folder with its own Python; engines share their files, and a ComfyUI
  install's files are never deleted.
- An engine in a few declarations, a variant inheriting from its base.
- NVIDIA CUDA, Apple Silicon (MPS) and CPU, on Windows, macOS and Linux.

## When to use turboCLI

- **An application or a script that runs known image models.** One command, or one Python call,
  per task: no workflow to export and keep in step with each model. Each engine is a template's
  recipe, pinned.
- **Embedding in a product.** The runner is LGPL, with a private license available, and talks to
  its host through a Python call or a small HTTP server whose progress streams as plain lines.
  turbopixel is built on it. Note that the `comfy-` engines run on turbo-offloader, which is GPL
  like the ComfyUI code it vendors; the runner reaches it by name only.
- **Time to the image matters.** Fast starts, a model kept loaded between images, and sampling at
  ComfyUI's speed or faster.
- **The same result months later.** The whole stack is pinned, and the test suite checks that an
  update changes neither the images nor the speed.
- **Edits that must stay in place**, with their drift and color shifts corrected, and cutouts or
  masks in the same tool.
- **No ComfyUI on the machine.** The stock engines run the published diffusers models at full
  precision; with ComfyUI installed, the `comfy-` engines reuse its files.

## When to use Comfy CLI

[comfy-cli](https://github.com/Comfy-Org/comfy-cli) installs, launches and drives ComfyUI from the
command line: its custom nodes, its models and the workflows it runs.

- **Any model, any pipeline.** ComfyUI runs video, audio, 3D, ControlNet, inpainting and whatever
  a graph can express, with thousands of custom nodes. turboCLI has 36 engines, one model of
  them for video.
- **New models on day one.** Models reach ComfyUI first; turboCLI follows its code and adds an
  engine once it is worth it.
- **Building and tweaking workflows.** ComfyUI's visual editor is the place to explore, and its
  community shares workflows for almost everything.
- **Running workflows already made in ComfyUI**, as they are, from a script or on a server.
- **Its ecosystem.** Custom nodes, model management, and a large community that turboCLI benefits
  from too.
