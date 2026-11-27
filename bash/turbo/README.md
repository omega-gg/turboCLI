# [Bash](../README.md) turboCLI

### [build.sh](build.sh) - Install turboCLI in the SKY_PATH_BIN folder

```
Usage: build <cpu | cuda | mps | clean> [latest]

latest: install the newest releases, ignoring the pinned versions (not reproducible)

example:
    build cuda
    build cuda latest
```

### [install.sh](install.sh) - Install a model into the model folder

```
Usage: install <engine> <renderer> [dtype = default] [inference = -1]
               [offload = offloader] [slicing = none]
               [ComfyUI folder] [token]

engine: flux2-4b
        flux2-9b                 (gated: needs a token)
        flux3-6b                 (gated: needs a token)
        z-image-turbo
        moss-image
        minimax-h3-image
        comfy-flux2-4b
        comfy-flux2-9b           (gated: needs a token)
        comfy-flux2-9b-distilled (gated: needs a token)
        comfy-flux3-6b           (gated: needs a token)
        comfy-z-image-turbo
        comfy-krea2-turbo        (gated: needs a token)
        comfy-krea2-turbo-edit-identity (gated: needs a token)
        comfy-krea2-turbo-edit-general (gated: needs a token)
        comfy-qwen-image-2-1
        comfy-qwen-image-2-1-turbo
        comfy-moss-image
        comfy-minimax-h3-image
        comfy-qwen-image-edit-2511
        comfy-qwen-image-edit-2511-lightning
        qwen-image-edit-2511
        qwen-image-edit-2511-lightning
        qwen-image-edit-2511-lightning-angles
        mask                     (no download, registers a compute engine)
        mask-apply               (no download)
        adjust                   (no download)
        mask-birefnet            (BiRefNet matte model)
        mask-lucida              (Lucida matte model)
        mask-inspyrenet          (InSPyReNet matte model)
        esrgan-upscale           (Real-ESRGAN x4plus upscale model)
        comfy-esrgan-upscale
        comfy-seedvr2-3b-upscale (SeedVR2 3B, ComfyUI's files)
        comfy-seedvr2-7b-upscale (SeedVR2 7B, ComfyUI's files)
        minimax-h3               (video, MiniMax's weights)
        comfy-minimax-h3         (video, ComfyUI's files)
        comfy-minimax-h3-quality (video, ComfyUI's bf16 files)
        comfy-minimax-h3-upscale
        comfy-minimax-h3-quality-upscale

renderer: cpu, cuda, mps

dtype: default, bfloat16, float16, float32
       (default copies the model as published, no cast)
       (bfloat16 is recommended for CUDA, float16 for Apple MPS)
       (the weights are cast on a fresh install alone, remove first to recast)

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

ComfyUI folder: reuse an existing ComfyUI install's model files (comfy-* engines).
                Missing components are fetched into ComfyUI's own models hierarchy.
                Pass none to skip it and still give a token.

token: optional Hugging Face access token, for a gated model (flux2-9b and the comfy-flux2-9b
       engines: FLUX Non-Commercial License; the flux3-6b engines: FLUX Kommunity License;
       accept it on huggingface.co first). Used for this download only, never recorded. An HF_TOKEN
       environment variable works too.

NOTE: The renderer and the options after it are recorded with the install, so a host
      reads them back with 'check-model SETTINGS:<engine>'. Installing again over an
      installed engine re-assigns them.

examples:
    install flux2-4b cuda
    install comfy-z-image-turbo cuda bfloat16 -1 offloader none
    install comfy-z-image-turbo cuda default -1 offloader none C:/dev/ComfyUI_portable
    install comfy-flux2-9b-distilled cuda default -1 offloader none none hf_xxxxxxxx
```

### [remove.sh](remove.sh) - Remove an installed engine (reference-counted)

```
Usage: remove <engine>

Remove an engine and garbage-collect its model / LoRAs / comfy components once no other
installed engine references them (a real ComfyUI install's files are never deleted).

engine: an installed id (see check-model)

example:
    remove comfy-z-image-turbo
```

### [check.sh](check.sh) - Check the install validity

```
Usage: check
```

### [check-model.sh](check-model.sh) - Check the installed models

```
Usage: check-model [engine | MODES:<mode,...> | ENGINES:<mode> | SETTINGS:<engine>]

no argument (or 'list'): list the installed engine id(s)

engine: an installed id, reports whether it is installed

MODES: list the installed engine id(s) supporting ANY of the listed modes, an
       upscaler's followed by 'upscaler'
       (text-to-image, image-to-image, text-to-video, image-to-video)

ENGINES: every engine supporting the mode, with 'installed' or 'absent', and 'gated'
         when its download needs a Hugging Face token
         (text-to-image, image-to-image, text-to-video, image-to-video,
         image-to-mask, image-adjust)

SETTINGS: the run settings an engine was installed with, one 'key: value' per line
          (renderer, dtype, inference, offload, slicing), and 'comfy: <folder>'
          for a comfy engine installed inside an existing ComfyUI
```

### [server.sh](server.sh) - Start and control the rendering server

```
Usage: server <action> [port = 8080] [scan]

actions:
    start:  start the server
    stop:   stop the server
    cancel: stop the current task
    clear:  stop the current task and release the loaded model

scan: with 'start', bind the first free port in [port, port + 19]

examples:
    server start
    server start  9000
    server start  9000 scan
    server stop   9000
    server cancel 9000
    server clear  9000
```

### [text-to-image.sh](text-to-image.sh) - Generate an image from a text prompt

```
Usage: text-to-image <engine> <renderer> <output image>
                     [prompt = none] [width = 512] [height = 512]
                     [seed = -1] [inference = -1]
                     [offload = offloader] [slicing = none]
                     [loras = none]
                     [server]

engine: flux2-4b
        flux2-9b
        flux3-6b
        z-image-turbo
        moss-image
        minimax-h3-image
        comfy-flux2-4b
        comfy-flux2-9b
        comfy-flux2-9b-distilled
        comfy-flux3-6b
        comfy-z-image-turbo
        comfy-krea2-turbo
        comfy-qwen-image-2-1
        comfy-qwen-image-2-1-turbo
        comfy-moss-image
        comfy-minimax-h3-image

renderer: cpu, cuda, mps

prompt: what to draw, none for an empty one

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    text-to-image flux2-4b cpu  output.png "knight in armor"
    text-to-image flux2-4b cuda output.png "knight in armor" 512 512 -1 -1 offloader none none 8080
```

### [image-to-image.sh](image-to-image.sh) - Generate an image from a text prompt and reference images

```
Usage: image-to-image <engine> <renderer> <input images> <output image>
                      [prompt = none] [width = 512] [height = 512]
                      [seed = -1] [inference = -1]
                      [offload = offloader] [slicing = none]
                      [loras = none]
                      [options] [server]

engine: flux2-4b
        flux2-9b
        flux3-6b
        minimax-h3-image
        comfy-flux2-4b
        comfy-flux2-9b
        comfy-flux2-9b-distilled
        comfy-flux3-6b
        comfy-krea2-turbo-edit-identity
        comfy-krea2-turbo-edit-general
        comfy-qwen-image-2-1
        comfy-qwen-image-2-1-turbo
        comfy-moss-image
        comfy-minimax-h3-image
        comfy-qwen-image-edit-2511
        comfy-qwen-image-edit-2511-lightning
        qwen-image-edit-2511
        qwen-image-edit-2511-lightning
        qwen-image-edit-2511-lightning-angles
        esrgan-upscale                  upscalers: the first input to the output size
        comfy-esrgan-upscale
        comfy-seedvr2-3b-upscale
        comfy-seedvr2-7b-upscale

renderer: cpu, cuda, mps

input images: separated by a comma, 4 maximum

prompt: what to draw, none for an empty one (an upscaler takes none)

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

options: key=value,... both fixes below work against the first input image:
         align undoes the zoom and shift an engine adds to its result, unless over
         3% (align=<percent> for another limit).
         color matches the result's colors to the input, the detail kept, unless they
         moved over 16 levels (color=<levels> for another limit).

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-to-image flux2-4b cpu  shield.png,helmet.png output.png "knight in armor"
    image-to-image flux2-4b cuda shield.png,helmet.png output.png "knight in armor" 512 512 -1 -1 offloader none none "" 8080
    image-to-image comfy-qwen-image-2-1-turbo cuda frame.png output.png "make it night" 1024 768 -1 -1 offloader none none align
    image-to-image comfy-minimax-h3-image cuda frame.png output.png "clean and crisp" 4096 2304 -1 -1 offloader none none align,color
    image-to-image esrgan-upscale cuda photo.png big.png none 2048 1536
```

### [text-to-video.sh](text-to-video.sh) - Generate a video from a text prompt

```
Usage: text-to-video <engine> <renderer> <output video>
                     [prompt = none] [duration = 5]
                     [width = 512] [height = 512]
                     [seed = -1] [inference = -1]
                     [offload = offloader] [slicing = none]
                     [loras = none]
                     [options] [server]

engine: minimax-h3
        comfy-minimax-h3
        comfy-minimax-h3-quality

renderer: cpu, cuda, mps

prompt: what to film, none for an empty one

duration: in seconds, each engine snapping it to its frame grid

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

options (comfy-minimax-h3 engines), comma separated:
         attention=cudnn puts cuDNN attention ahead of flash: about twice as fast
         on a B200
         scheduler=simple|beta in place of the engine's: simple, or beta with
         references on comfy-minimax-h3-quality
         latent=save also writes the video's latents next to it, for a later
         pass: <output>.video.latent and .audio.latent, as ComfyUI's SaveLatent

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    text-to-video comfy-minimax-h3 cuda output.mp4 "a knight walks into the mist"
    text-to-video comfy-minimax-h3 cuda output.mp4 "a knight walks into the mist" 5 1344 768 -1 -1 offloader none none "" 8080
```

### [image-to-video.sh](image-to-video.sh) - Generate a video from a text prompt and its references

```
Usage: image-to-video <engine> <renderer> <input references> <output video>
                      [prompt = none] [duration = 5]
                      [width = 512] [height = 512]
                      [seed = -1] [inference = -1]
                      [offload = offloader] [slicing = none]
                      [loras = none]
                      [options] [server]

engine: minimax-h3
        comfy-minimax-h3
        comfy-minimax-h3-quality
        comfy-minimax-h3-upscale
        comfy-minimax-h3-quality-upscale

renderer: cpu, cuda, mps

input references: separated by a comma, each tagged by its role:
                  first=<image> last=<image>   the frames the video starts and ends on
                  image=<image>                a picture reference, <Picture 1>...
                  audio=<audio>                a voice or sound reference, <Audio 1>...
                  guide=<audio>@<seconds>      a recording pinned at that time
                  latent=<take>.video.latent   a take's latents (latent=save), alone:
                                               an upscale engine renders the take again
                                               at 2x, the rest of it the take's
                  an untagged image is a frame, the first then the last. With a
                  picture or audio reference the engine runs its reference workflow

prompt: what to film, none for an empty one

duration: in seconds, each engine snapping it to its frame grid

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

options (comfy-minimax-h3 engines), comma separated:
         attention=cudnn puts cuDNN attention ahead of flash: about twice as fast
         on a B200
         scheduler=simple|beta in place of the engine's: simple, or beta with
         references on comfy-minimax-h3-quality
         latent=save also writes the video's latents next to it, for a later
         pass: <output>.video.latent and .audio.latent, as ComfyUI's SaveLatent
         denoise=0-1 the upscale engines' refine strength, 0.45 by default:
         lower keeps more of the take's fine detail, 0 the upscaler's alone

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-to-video comfy-minimax-h3 cuda frame.png output.mp4 "the camera pushes in"
    image-to-video comfy-minimax-h3 cuda first.png,last.png output.mp4 "the camera pushes in" 10 1344 768
    image-to-video comfy-minimax-h3-quality cuda first=fedora.png,audio=voice.wav output.mp4 "he speaks in the voice of <Audio 1>" 5 1376 576
    image-to-video comfy-minimax-h3-quality-upscale cuda latent=output.video.latent output-2x.mp4
```

### [image-to-mask.sh](image-to-mask.sh) - Generate a mask / matte

```
Usage: image-to-mask <engine> <renderer> <input images> <mask output> [options] [server]

Generate a mask / matte (an 8-bit grayscale PNG). Apply it with image-apply-mask.

engine: mask            diff / region mask (options mode=default|region)
        mask-birefnet   subject matte via BiRefNet
        mask-lucida     subject matte via Lucida (glass / camouflage / text / print)
        mask-inspyrenet subject matte via InSPyReNet

renderer: cpu, cuda, mps (mask ignores it; the matte engines use it)

input images: separated by a comma, the input first. mask: input,reference. matte engines:
              input, or input,plate (a plate keeps the cast shadow).

options: key=value,... -- cutoff=N (0-255, higher = fewer pixels/shadow); mask also takes
         mode=default|region (default = diff mask, region = grown boxes), glow=N (pixels,
         mode=default only: the mask reaches that far past the change, faded, which covers the
         outline a removal leaves behind) and fade=<left>:<top>:<right>:<bottom> (pixels,
         either mode: softens the mask on the borders the caller cut, for an input that is a
         crop of a bigger picture. A side given 0 keeps its hard edge)

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-to-mask mask          cpu  edited.png,original.png mask.png cutoff=40
    image-to-mask mask          cpu  edited.png,original.png mask.png mode=region
    image-to-mask mask          cpu  edited.png,original.png mask.png glow=16
    image-to-mask mask          cpu  crop.png,crop_plate.png mask.png mode=region,fade=0:0:24:0
    image-to-mask mask-birefnet cuda photo.png matte.png
    image-to-mask mask-birefnet cuda photo.png,plate.png matte.png cutoff=40
```

### [image-apply-mask.sh](image-apply-mask.sh) - Apply a mask (composite or putalpha), or trim

```
Usage: image-apply-mask <mode> <input images> <output image> [options] [server]

Apply a precomputed mask (from image-to-mask). Torch-free (PIL, no GPU).

mode: composite  paste the input's masked region onto a reference (needs a reference)
      putalpha   write the mask as the input's alpha channel (an RGBA cutout)
      trim       crop the input to what shows, with a margin of transparency around it

input images: separated by a comma -- input,mask for putalpha; input,mask,reference for
              composite (the reference is shown where the mask is black); input alone
              for trim.

options: key=value,... -- trim takes pad=N, the most transparency it keeps on a side
         (pixels, 32 by default), and speck=N, the size under which a separate blob is
         left out as a speck of the mask (pixels, 0 by default: off, all of it stays)

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-apply-mask putalpha  photo.png,matte.png cutout.png
    image-apply-mask composite edited.png,mask.png,original.png output.png
    image-apply-mask trim      cutout.png trimmed.png pad=32,speck=1024
```

### [image-adjust.sh](image-adjust.sh) - Align an existing image on its reference, or match its colors

```
Usage: image-adjust <base image> <reference image> <output image> <fixes> [server]

Fix an existing image against its reference, an engine's result against its input.
Torch-free, no GPU.

fixes: align        the zoom and shift an engine added are measured and undone,
                    unless over 3% (align=<percent> for another limit)
       color        the colors are matched to the reference, the detail kept,
                    unless they moved over 16 levels (color=<levels> for another)
       align,color  both, in that order
       transfer     graded toward the reference, which need not show the same
                    picture: each L*a*b* channel takes its mean and spread
                    (transfer=<strength> from 0 to 1, part of the way)

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-adjust clean.png frame.png fixed.png color
    image-adjust edit.png frame.png fixed.png align,color
    image-adjust edit.png frame.png fixed.png align,color=24 8080
    image-adjust shot.png other-shot.png graded.png transfer=0.5
```
