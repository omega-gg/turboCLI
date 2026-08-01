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

### [image-mask.sh](image-mask.sh) - Merge an edited image back onto its reference

```
Usage: image-mask <mode> <reference image> <input image> <output image> [threshold]

Keep the changed region from the input and restore the byte-exact reference everywhere else. No
generation -- pure image processing (PIL + numpy, no GPU). Run it after an image-to-image edit to
undo the whole-frame color/tone drift outside the part you changed.

mode: mask   soft pixel diff, best for adding an object / recoloring
      region grown bounding boxes, best for removal / replace (ghost-free)

reference: the base canvas (the original scene)

input: the edited / generated image

threshold: positive integer; a pixel differing from the reference by more than this is kept from
           the input. Higher = tighter (restores more reference), lower = keeps more. Omit for the
           default; raise it when the generator drifts the whole frame (e.g. a flux2 img2img edit).

The output is at the reference resolution: a full-res reference with a smaller input merges back at
full resolution. To cut a subject onto transparency instead, see image-remove-background.

examples:
    image-mask mask   original.png edited.png output.png
    image-mask mask   original.png edited.png output.png 40
    image-mask region original.png edited.png output.png
```

### [image-remove-background.sh](image-remove-background.sh) - Cut a subject onto transparency

```
Usage: image-remove-background <model> <renderer> <input> <output> [plate] [shadow threshold]

Cut the subject out of the input onto a transparent background (RGBA PNG, same size and placement).
Delegates to the remove-background tool (own venv; see bash/remove-background).

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
    image-remove-background birefnet cuda photo.png cutout.png
    image-remove-background lucida  cuda photo.png cutout.png plate.png
    image-remove-background lucida  cuda photo.png cutout.png plate.png 40
```
