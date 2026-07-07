# [Bash](../README.md) flux2

### [build.sh](build.sh) - Install a model into the model folder

```
Usage: build <engine> [dtype = default] [fast]

engine: flux2-4b

dtype: default, bfloat16, float16, float32

example:
    build flux2-4b
```

### [check.sh](check.sh) - Check the installed models

```
Usage: check [engine = flux2-4b]

engine: flux2-4b
```

### [run.sh](run.sh) - Generate an image from a text prompt

```
Usage: run <prompt> <output image> [width = 512] [height = 512]
           [renderer = cpu]
           [engine = flux2-4b] [seed = -1] [inference = 4]
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
    run "knight in armor" output.png 512 512 cuda flux2-4b -1 4 offloader none none 8080
```

### [run-image.sh](run-image.sh) - Generate an image from a text prompt and reference images

```
Usage: run-image <prompt> <input images> <output image>
                 [width = 512] [height = 512]
                 [renderer = cpu]
                 [engine = flux2-4b] [seed = -1] [inference = 4]
                 [offload = offloader] [slicing = none]
                 [loras = none]
                 [server]

input images: separated by a comma, 4 maximum

renderer: cpu, cuda, mps

offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)

slicing: none, slice

loras: none, comma separated <path>@[weight]

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    run "knight in armor" shield.png,helmet.png output.png
    run "knight in armor" shield.png,helmet.png output.png 512 512 cuda flux2-4b -1 4 offloader none none 8080
```
