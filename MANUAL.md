# turboCLI manual

turboCLI generates images on your own machine, from the command line or behind a small local
server. It turns a prompt into a picture, edits a picture from reference images, and cuts a subject
out of its background. Once an engine is installed it needs no internet at all: nothing is
uploaded and nothing is tracked.

You drive it with a handful of shell scripts in `bash/turbo/`. This manual goes through them task
by task, with a command or two for each; the full list of every script's arguments is in
[bash/turbo/README.md](bash/turbo/README.md). To understand how it works inside, read
[dummy.md](dummy.md).

## First run

turboCLI installs into the folder the `SKY_PATH_BIN` variable names, so set it first, in every
shell you run the scripts from. On Windows, run them from Git Bash
([Git for Windows](https://git-for-windows.github.io)).

```sh
export SKY_PATH_BIN="$PWD/bin"
```

Then build python, and turboCLI for your renderer:

```sh
cd bash/python
sh build.sh default

cd ../turbo
sh build.sh cuda
```

The renderer is what turboCLI is built for, so pick the one that matches your machine:

| Renderer | When to pick it |
| --- | --- |
| `cpu` | No dedicated graphics card. Works everywhere, and is the slowest. |
| `cuda` | An NVIDIA graphics card. |
| `mps` | An Apple Silicon Mac (M1 and later). |

`sh check.sh` says whether turboCLI is installed and up to date. Building again reinstalls turboCLI
from scratch and keeps the models you downloaded; `build.sh clean` removes it, and the next build
puts the models back. `build.sh cuda latest` takes the newest releases of everything rather than
the tested versions, which is quicker to new features and no longer reproducible.

## Choosing and installing an engine

An engine is one model, ready to run: FLUX.2 Klein, Z-Image-Turbo, Qwen-Image and the others, each
in a few variants. [engine.md](engine.md) lists them all by task, with what each one downloads, how
many steps it takes and what it needs. Install one with its id and your renderer:

```sh
sh install.sh z-image-turbo cuda
```

The arguments after the renderer are optional, in this order:

- **dtype**: `default` copies the weights as they were published. `bfloat16` suits CUDA and
  `float16` Apple MPS, and both take half the memory of `float32`. The weights are converted on a
  fresh install only, so remove an engine before installing it again in another dtype.
- **inference, offload, slicing**: recorded with the install as the engine's settings, which a
  host such as turbopixel reads back with `check-model SETTINGS:<engine>`. The scripts that
  generate take their own, see below.
- **ComfyUI folder**: for a `comfy-` engine, the folder of a ComfyUI install whose model files you
  want to reuse, so nothing is downloaded twice. A file it does not have yet is downloaded into
  ComfyUI's own `models` folder. Pass `none` to skip it and still give a token.
- **token**: a Hugging Face token, for a gated engine (marked so in [engine.md](engine.md)). Accept
  the model's license on huggingface.co first, with the account the token belongs to. An `HF_TOKEN`
  environment variable works too. The token is used for that download and never recorded.

```sh
sh install.sh comfy-z-image-turbo cuda default -1 offloader none C:/dev/ComfyUI_portable
sh install.sh comfy-flux2-9b-distilled cuda default -1 offloader none none hf_xxxxxxxx
```

To see what is installed, and to take an engine out:

```sh
sh check-model.sh                         # the installed engines
sh check-model.sh z-image-turbo           # whether this one is installed
sh check-model.sh SETTINGS:z-image-turbo  # what it was installed with
sh remove.sh z-image-turbo
```

`remove.sh` deletes the files that no other installed engine uses, so two variants of a model
can share theirs. It never deletes the files of a real ComfyUI install.

## Generating an image

```sh
sh text-to-image.sh z-image-turbo cuda "a knight in a misty forest" knight.png 1024 768
```

After the engine, the renderer, the prompt and the output file come, in order: the width and the
height (512 by default), the seed, the number of steps, the offload, the slicing, the LoRAs and a
server, see [bash/turbo/README.md](bash/turbo/README.md).
A seed of `-1` picks a new one each time; the same seed, prompt and settings give the same image
again. Steps at `-1` are the engine's own count, the one [engine.md](engine.md) gives.

What turboCLI prints tells you where it is:

| Line | What it means |
| --- | --- |
| `loading <engine> model (<renderer> / <offload>)...` | The model is loading. |
| `model ready` | It is loaded. |
| `generating "<prompt>" (N steps)...` | The image is being made. |
| `40%\|step 2/5 (00:06, 3.10s/it)` | Progress, step by step. |
| `Saved: <path>` | The image is written. This is the line that means it worked. |
| `ERROR: ...` | What went wrong. No image is written. |

**Image to image.** An edit takes up to four reference images, separated by a comma, before the
output file:

```sh
sh image-to-image.sh flux2-4b cuda "make it night" photo.png night.png 1024 1024
```

The output has the width and the height you give. References keep their shape and are sized by
the engine: Qwen-Image and FLUX.2 take them to about one megapixel, FLUX.3 keeps them at their
own size up to 4096x4096.

**LoRAs.** A LoRA adds a style or a subject to an engine that supports it. Give its file and a
strength after an `@`, several separated by a comma; a missing strength is 1.

```sh
sh text-to-image.sh comfy-krea2-turbo cuda "a portrait" out.png 1024 1024 -1 -1 offloader none \
                    style.safetensors@0.8
```

## Masks and mattes

A mask is an 8-bit grayscale image: white where something is kept, black where it is not. Two
kinds of engine make one, with `image-to-mask.sh`:

| Engine | Input images | What it gives |
| --- | --- | --- |
| `mask` | `edited.png,original.png` | Where the edit changed the original. `mode=region` fills grown boxes instead, best to take something out. |
| `mask-birefnet` | `photo.png`, or `photo.png,plate.png` | The subject's matte, as BiRefNet sees it. |
| `mask-lucida` | the same | A BiRefNet fine-tuned for glass, camouflage, text and print. |
| `mask-inspyrenet` | the same | The subject's matte, as InSPyReNet sees it. |

A plate is the same scene without the subject: given one, a matte keeps the shadow the subject
casts. Options go after the output, as `key=value` separated by commas: `cutoff=N` (0 to 255, how
much a pixel has to differ to count, higher keeps less), `glow=N` (the `mask` engine reaches N
pixels further, which covers the outline a removal leaves) and `fade=<left>:<top>:<right>:<bottom>`
(softens the borders of a mask made from a crop of a bigger picture).

`image-apply-mask.sh` then uses the mask:

| Mode | Input images | What it makes |
| --- | --- | --- |
| `putalpha` | `photo.png,mask.png` | A cutout: the mask becomes the image's transparency. |
| `composite` | `edited.png,mask.png,original.png` | The edit where the mask is white, the original elsewhere. |
| `trim` | `cutout.png` | The image cropped to what shows, `pad=N` pixels of margin around it. |

Cutting a subject out of a photo, then trimming it to its size:

```sh
sh image-to-mask.sh mask-birefnet cuda photo.png matte.png
sh image-apply-mask.sh putalpha photo.png,matte.png cutout.png
sh image-apply-mask.sh trim cutout.png trimmed.png pad=32
```

Keeping only what an edit added, over the untouched original:

```sh
sh image-to-mask.sh mask cpu edited.png,original.png mask.png
sh image-apply-mask.sh composite edited.png,mask.png,original.png result.png
```

The `mask` engine and `image-apply-mask.sh` need no download and run on the CPU; install them once
with `sh install.sh mask cpu` and `sh install.sh mask-apply cpu`. The matte engines download their
model. [test/README.md](test/README.md) has longer recipes, with the images to try them on.

## Running a server

Each script run starts python and loads the model, which can take longer than making the image. A
server keeps the model loaded between images, so only the first one pays for it:

```sh
sh server.sh start
sh text-to-image.sh z-image-turbo cuda "a cat" cat.png 1024 768 -1 -1 offloader none none 8080
```

Give any generating script the server's port, or `host:port`, as its last argument and it runs
there. The server listens on port 8080 unless you give another; `sh server.sh start 9000 scan`
takes the first free port from 9000 on.

- The model loads with the first request and loads again when the engine, the renderer, the
  offload or the slicing change.
- The latest request wins: a new one stops the one in progress. Images are never made in
  parallel.
- A model left idle for 600 seconds is released, which gives its memory back.
- `sh server.sh cancel` stops the image in progress, `clear` also releases the model, and `stop`
  shuts the server down. Each takes the port too when it is not 8080.

An application can talk to the server over HTTP, as turbopixel does; its requests are in
[implementation.md](implementation.md).

## Memory and speed

The offload decides where the model's weights sit while an image is made:

| Offload | What it does | When to pick it |
| --- | --- | --- |
| `offloader` | turbo-offloader moves the weights between memory and the graphics card as they are needed, the way ComfyUI does. | The default. A model larger than your card fits, and some engines run only this way (see [engine.md](engine.md)). |
| `none` | The whole model on the graphics card. | A card with memory to spare. |
| `model_cpu` | diffusers moves each part of the model to the card in turn. | A card a little short of memory, with an engine that runs without the offloader. |
| `sequential_cpu` | diffusers moves the model a layer at a time. | Very little graphics memory: the least memory, and the slowest. |

`slice` instead of `none` for the slicing computes in smaller pieces, the attention and the
decoding of the image both, for less memory at some cost in speed. Time and memory grow fast with
the size of the image: at about one megapixel an image takes seconds to a few minutes, at four
megapixels many times longer. To get a large image, generate around one megapixel and enlarge it
afterwards.

## When something goes wrong

- **`ERROR: unknown engine`, or a model that is not there.** The engine is not installed:
  `sh check-model.sh` lists what is, `sh install.sh` adds it.
- **A gated engine will not download.** It needs a token, from an account that accepted the
  model's license on huggingface.co, see the token in
  [Choosing and installing an engine](#choosing-and-installing-an-engine).
- **A download fails with a certificate error.** A VPN or a proxy that inspects secure traffic gets
  in the way: turn it off for the download.
- **Out of memory.** Keep the `offloader`, add `slice`, make a smaller image, or pick a lighter
  engine in [engine.md](engine.md).
- **`ERROR: offloader offload unavailable`.** turboCLI was built without its offloader: build it
  again with `build.sh`.
- **Something stopped working after an update.** `sh check.sh` says whether turboCLI is current;
  `build.sh` brings it back, and keeps your models.

## Where files live

Everything goes under `$SKY_PATH_BIN/gg.omega/`: python in `python/`, turboCLI in `turbo/`. Inside
`turbo/`, `model/` holds the downloaded models and `engine/` one record per installed engine. A
`comfy-` engine installed with a ComfyUI folder keeps its files in that ComfyUI's `models` folder
instead, where ComfyUI uses them too. Images are written where you tell each script to write them.

## Using it from Python

turboCLI can run inside a python application rather than through the scripts:
`runner.core.generate(params, emit)` takes the same settings as the scripts and calls `emit` with
each line they print. [implementation.md](implementation.md) describes it.
