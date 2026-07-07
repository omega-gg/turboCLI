# [Bash](../README.md) Z-Image

### [build.sh](build.sh) - Install a model into the model folder

```
Usage: build <engine> [dtype = default] [fast]

engine: z-image-turbo

dtype: default, bfloat16, float16, float32

example:
    build z-image-turbo
```

### [check.sh](check.sh) - Check the installed models

```
Usage: check [engine = z-image-turbo]

engine: z-image-turbo
```

### [run.sh](run.sh) - Generate an image from a text prompt

```
Usage: run <prompt> <output image> [width = 512] [height = 512]
           [renderer = cpu] [seed = -1] [inference = 8]
           [offload = offloader] [slicing = none]
           [loras = none]
           [server]

renderer: cpu, cuda, mps

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    run "knight in armor" output.png
    run "knight in armor" output.png 512 512 cuda -1 8 offloader none none 8080
```
