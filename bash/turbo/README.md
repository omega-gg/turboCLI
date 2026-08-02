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
Usage: install <engine> [dtype = default] [ComfyUI folder]

engine: flux2-4b
        z-image-turbo
        comfy-flux2-4b
        comfy-z-image-turbo
        comfy-krea2-turbo
        comfy-krea2-turbo-realism
        comfy-qwen-image-edit-2511
        comfy-qwen-image-edit-2511-lightning
        qwen-image-edit-2511
        qwen-image-edit-2511-lightning
        qwen-image-edit-2511-lightning-angles

dtype: default, bfloat16, float16, float32
       (bfloat16 is recommended for CUDA, float16 for Apple MPS)

ComfyUI folder: reuse an existing ComfyUI install's model files (comfy-* engines).
                Missing components are fetched into ComfyUI's own models hierarchy.

examples:
    install flux2-4b
    install comfy-z-image-turbo default C:/dev/test/ComfyUI_windows_portable
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
Usage: check-model [engine | MODES:<mode,...>]

no argument (or 'list'): list the installed engine id(s)

engine: an installed id, reports whether it is installed

MODES: list the installed engine id(s) supporting ANY of the listed modes
       (text-to-image, image-to-image)
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
Usage: text-to-image <engine> <renderer> <prompt> <output image>
                     [width = 512] [height = 512]
                     [seed = -1] [inference = -1]
                     [offload = offloader] [slicing = none]
                     [loras = none]
                     [server]

engine: flux2-4b
        z-image-turbo
        comfy-flux2-4b
        comfy-z-image-turbo
        comfy-krea2-turbo
        comfy-krea2-turbo-realism

renderer: cpu, cuda, mps

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    text-to-image flux2-4b cpu  "knight in armor" output.png
    text-to-image flux2-4b cuda "knight in armor" output.png 512 512 -1 4 offloader none none 8080
```

### [image-to-image.sh](image-to-image.sh) - Generate an image from a text prompt and reference images

```
Usage: image-to-image <engine> <renderer> <prompt> <input images> <output image>
                      [width = 512] [height = 512]
                      [seed = -1] [inference = -1]
                      [offload = offloader] [slicing = none]
                      [loras = none]
                      [server]

engine: flux2-4b
        comfy-flux2-4b
        comfy-qwen-image-edit-2511
        comfy-qwen-image-edit-2511-lightning
        qwen-image-edit-2511
        qwen-image-edit-2511-lightning
        qwen-image-edit-2511-lightning-angles

renderer: cpu, cuda, mps

input images: separated by a comma, 4 maximum

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-to-image flux2-4b cpu  "knight in armor" shield.png,helmet.png output.png
    image-to-image flux2-4b cuda "knight in armor" shield.png,helmet.png output.png 512 512 -1 -1 offloader none none 8080
```

### [image-mask.sh](image-mask.sh) - Generate a diff/region mask

```
Usage: image-mask <mode> <reference image> <input image> <mask output> [threshold]

Generate a soft mask of where an edit differs from its reference -- no compositing. Pure image
processing (PIL + numpy, no GPU). Apply it with image-mask-apply.

mode: mask   soft pixel diff, best for adding an object / recoloring
      region grown bounding boxes, best for removal / replace (ghost-free)

reference: the base canvas (the original scene)

input: the edited / generated image

threshold: pixels differing from the reference beyond this are kept in the mask. Default 24;
           higher = tighter, lower keeps more. Raise it when the generator drifts the whole frame.

The mask is an 8-bit grayscale PNG at the input resolution.

examples:
    image-mask mask   original.png edited.png mask.png
    image-mask mask   original.png edited.png mask.png 40
    image-mask region original.png edited.png mask.png
```

### [image-mask-background.sh](image-mask-background.sh) - Generate a subject matte

```
Usage: image-mask-background <model> <renderer> <input> <mask output> [plate] [shadow threshold]

Generate a subject matte (8-bit grayscale PNG, same size and placement) from the input. Delegates
to the remove-background tool (own venv; see bash/remove-background). Apply the matte with
image-mask-apply (putalpha to cut out, composite onto a new background).

model: birefnet   (ZhengPeng7/BiRefNet) -- strong on thin glows like a neon sign or a lightsaber
       lucida     (egeorcun fine-tune) -- better on glass / camouflage / text / print
       inspyrenet (transparent-background) -- InSPyReNet, also strong on thin glows

renderer: cpu, cuda or mps (cuda/mps fall back to cpu if the build lacks them,
          bash/remove-background/build.sh <cpu|cuda|mps>; cpu is slow)

plate: a clean background (the same scene without the subject); where the input is darker than the
       plate is the cast shadow, recovered as soft alpha so it is kept. Omit it for subject only.

shadow threshold: darkening floor for the plate shadow (default 12); raise it when a drifted plate
                  ghosts the background back in. Only used with a plate.

examples:
    image-mask-background birefnet cuda photo.png matte.png
    image-mask-background lucida   cuda photo.png matte.png plate.png
    image-mask-background lucida   cuda photo.png matte.png plate.png 40
```

### [image-mask-apply.sh](image-mask-apply.sh) - Apply a mask (composite or putalpha)

```
Usage: image-mask-apply <mode> <input image> <mask image> <output image> [reference]

Apply a precomputed mask (from image-mask or image-mask-background). Torch-free (PIL, no GPU).

mode: composite  paste the input's masked region onto a reference (needs a reference)
      putalpha   write the mask as the input's alpha channel (an RGBA cutout)

input: the source image the mask was computed for

mask: an 8-bit grayscale mask / matte (255 = kept)

reference: base canvas shown where the mask is black -- REQUIRED for composite (5 args), omit for
           putalpha (4 args). The original scene to restore, or a new backdrop to composite onto.

examples:
    image-mask-apply composite edited.png mask.png output.png original.png
    image-mask-apply putalpha  photo.png  matte.png cutout.png
```
