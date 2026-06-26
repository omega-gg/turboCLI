# [Bash](../README.md) Z-Image tools

## Configuration

Place your z-image standalone folder into the SKY_PATH_BIN/z-image folder or set SKY_PATH_Z_IMAGE.
Your models should be in SKY_PATH_BIN/z-image-model or SKY_PATH_Z_IMAGE_MODEL.

## Tools

### [build.sh](build.sh): Install z-image in the SKY_PATH_BIN folder

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
  Z-Image-Turbo

dtype: bfloat16, float16, float32

example:
    model Z-Image-Turbo
```

### [run.sh](run.sh): Generate an image from a text prompt

```
Usage: run <prompt> <output image> [width = 512] [height = 512]
           [renderer = cpu] [seed = -1] [inference = 8]
           [cuda_offload = sequential_cpu] [slicing = none]
           [server]

renderer: cpu, cuda, mps

cuda_offload: none, model_cpu, sequential_cpu

slicing: none, slice

server: host:port (or port for 127.0.0.1) of a rendering server

examples:
    run "knight in armor" output.png
    run "knight in armor" output.png 512 512 cuda -1 8 sequential_cpu none 8080
```
