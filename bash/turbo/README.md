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
               [ComfyUI folder]

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
        mask                 (no download -- registers a compute engine)
        mask-apply           (no download)
        mask-birefnet        (BiRefNet matte model)
        mask-lucida          (Lucida matte model)
        mask-inspyrenet      (InSPyReNet matte model)

renderer: cpu, cuda, mps

dtype: default, bfloat16, float16, float32
       (bfloat16 is recommended for CUDA, float16 for Apple MPS)
       (the weights are cast on a fresh install alone, remove first to recast)

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

ComfyUI folder: reuse an existing ComfyUI install's model files (comfy-* engines).
                Missing components are fetched into ComfyUI's own models hierarchy.

NOTE: The renderer and the options after it are recorded with the install, so a host
      reads them back with 'check-model SETTINGS:<engine>'. Installing again over an
      installed engine re-assigns them.

examples:
    install flux2-4b cuda
    install comfy-z-image-turbo cuda bfloat16 -1 offloader none
    install comfy-z-image-turbo cuda default -1 offloader none C:/dev/ComfyUI_portable
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

SETTINGS: the run settings an engine was installed with, one 'key: value' per line
          (renderer, dtype, inference, offload, slicing)
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
         mode=default|region (default = diff mask, region = grown boxes)

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-to-mask mask          cpu  edited.png,original.png mask.png cutoff=40
    image-to-mask mask          cpu  edited.png,original.png mask.png mode=region
    image-to-mask mask-birefnet cuda photo.png matte.png
    image-to-mask mask-birefnet cuda photo.png,plate.png matte.png cutoff=40
```

### [image-apply-mask.sh](image-apply-mask.sh) - Apply a mask (composite or putalpha)

```
Usage: image-apply-mask <mode> <input images> <output image> [server]

Apply a precomputed mask (from image-to-mask). Torch-free (PIL, no GPU).

mode: composite  paste the input's masked region onto a reference (needs a reference)
      putalpha   write the mask as the input's alpha channel (an RGBA cutout)

input images: separated by a comma -- input,mask for putalpha; input,mask,reference for
              composite (the reference is shown where the mask is black).

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    image-apply-mask putalpha  photo.png,matte.png cutout.png
    image-apply-mask composite edited.png,mask.png,original.png output.png
```
