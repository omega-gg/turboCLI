# [Bash](../README.md) qwen-image tools

## Configuration

Place your qwen-image standalone folder into the SKY_PATH_BIN/qwen-image folder or set
SKY_PATH_QWEN_IMAGE. Your models should be in SKY_PATH_BIN/qwen-image-model or
SKY_PATH_QWEN_IMAGE_MODEL.

## Tools

### [build.sh](build.sh): Install qwen-image in the SKY_PATH_BIN folder

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
  Qwen-Image-Edit-2511

dtype: bfloat16, float16, float32

example:
    model Qwen-Image-Edit-2511
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
