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
    image-to-image flux2-4b cpu "knight in armor" shield.png,helmet.png output.png
    image-to-image flux2-4b cuda "knight in armor" shield.png,helmet.png output.png 512 512 -1 4 offloader none none 8080
```
