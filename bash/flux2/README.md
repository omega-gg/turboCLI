# [Bash](../README.md) flux2 tools

## Configuration

Place your flux2 standalone folder into the SKY_PATH_BIN/flux2 folder or set SKY_PATH_FLUX2. Your
models should be in SKY_PATH_BIN/flux2-model or SKY_PATH_FLUX2_MODEL.

## Tools

### [build.sh](build.sh): Install flux2 in the SKY_PATH_BIN folder

```
Usage: build <cpu | cuda | mps | clean>

example:
    build cuda
```

### [check.sh](check.sh): Check the install validity

```
Usage: check
```

### [check-model.sh](check-model.sh): Check the installed models

```
Usage: check-model [model]
```

### [model.sh](model.sh): Download a model into the model folder

```
Usage: model <model> [dtype = bfloat16] [fast]

models:
  FLUX.2-klein-4B
  FLUX.2-klein-4B-base
  FLUX.2-klein-9B
  FLUX.2-klein-9B-base

dtype: bfloat16, float16, float32

example:
    model FLUX.2-klein-4B
```

### [run.sh](run.sh): Generate an image from a text prompt

```
Usage: run <prompt> <output image> [width = 512] [height = 512]
           [renderer = cpu] [seed = -1] [inference = 4]
           [cuda_offload = sequential_cpu] [slicing = none]
           [server]

renderer: cpu, cuda, mps

cuda_offload: none, model_cpu, sequential_cpu

slicing: none, slice

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    run "knight in armor" output.png
    run "knight in armor" output.png 512 512 cuda -1 4 sequential_cpu none 8080
```

### [run-image.sh](run-image.sh): Generate an image from a text prompt and reference images

```
Usage: run-image <prompt> <input images> <output image>
                 [width = 512] [height = 512]
                 [renderer = cpu] [seed = -1] [inference = 4]
                 [cuda_offload = sequential_cpu] [slicing = none]
                 [server]

input images: separated by a comma, 4 maximum

renderer: cpu, cuda, mps

cuda_offload: none, model_cpu, sequential_cpu

slicing: none, slice

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    run "knight in armor" shield.png,helmet.png output.png
    run "knight in armor" shield.png,helmet.png output.png 512 512 cuda -1 4 sequential_cpu none 8080
```
