# turboCLI manual

turboCLI generates images and videos on your own machine, from the command line or behind a
small local server. It turns a prompt into a picture or a clip with its sound, edits a picture
from reference images, and cuts a subject out of its background. Once an engine is installed it
needs no internet at all: nothing is uploaded and nothing is tracked.

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
sh text-to-image.sh z-image-turbo cuda knight.png "a knight in a misty forest" 1024 768
```

After the engine, the renderer and the output file come, in order: the prompt (`none` for an
empty one), the width and the height (512 by default), the seed, the number of steps, the
offload, the slicing, the LoRAs and a server, see [bash/turbo/README.md](bash/turbo/README.md).
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
output file, and the prompt after it:

```sh
sh image-to-image.sh flux2-4b cuda photo.png night.png "make it night" 1024 1024
```

The output has the width and the height you give. References keep their shape and are sized by
the engine: Qwen-Image Edit and FLUX.2 take them to about one megapixel, Qwen-Image 2.1 to the
output's area (an image of the output's size goes as it is, so the edit does not shift), FLUX.3
keeps them at their own size up to 4096x4096.

**Putting an edit back in place.** An edit of part of a picture has to come back where it was,
and Qwen-Image 2.1 zooms and shifts its result a little, differently for each seed. `align`,
after the LoRAs, measures that against the first input image, the picture being edited, and
undoes it:

```sh
sh image-to-image.sh comfy-qwen-image-2-1-turbo cuda frame.png night.png "make it night" \
                     1024 768 -1 -1 offloader none none align
```

A line `align: ...` says what it did: aligned, or left as is when the result was already in place,
was redrawn rather than moved, or was zoomed by more than 3%, which a prompt asks for ("zoom out")
rather than an engine drifts. `align=5` moves that limit to 5%.

**Keeping the colors.** An edit meant to keep the picture can still move its colors: cleaning up a
deep blue frame, MiniMax-H3-Image lifts the shadows and warms it, Qwen-Image 2.1 greys it.
`color` matches the result's colors to the first input image, one curve per channel, so the new
detail stays. The curve is measured on what the edit left alone, so a shirt the prompt made red
stays red. A change of more than 16 levels on average was asked for ("make it night") and is
left as is; `color=24` moves that limit to 24. It runs after `align` when both are given
(`align,color`), and its line `color: ...` gives the mean move of each channel and what it did.

**LoRAs.** A LoRA adds a style or a subject to an engine that supports it. Give its file and a
strength after an `@`, several separated by a comma; a missing strength is 1.

```sh
sh text-to-image.sh comfy-krea2-turbo cuda out.png "a portrait" 1024 1024 -1 -1 offloader none \
                    style.safetensors@0.8
```

## Generating a video

```sh
sh text-to-video.sh comfy-minimax-h3 cuda knight.mp4 "a knight walks into the mist" 5 1344 768
sh image-to-video.sh comfy-minimax-h3 cuda frame.png knight.mp4 "the camera pushes in" 5 1344 768
```

A video takes the arguments of an image, with its duration in seconds right after the prompt, 5
by default. The engine snaps it to its own frame grid: MiniMax H3 films at 24 fps, in 17k+5
frames, so 5 seconds is 124 frames. The video comes out as an mp4 (H.264) with its soundtrack
(AAC): MiniMax H3 makes the sound with the picture, from what the prompt says of it. Its width
and height are multiples of 32.

MiniMax H3 is large, and fastest on a card that holds it whole: `comfy-minimax-h3`'s files take
40 GB, `comfy-minimax-h3-hq`'s 98 GB, `minimax-h3`'s 143 GB. A smaller card works, the weights
streaming to it every step, but much slower: `comfy-minimax-h3-hq` takes 13.7 s a step on a
48 GB L40S, 4.6 on a 180 GB B200 (2.2 with `attention=cudnn`, below), at 1376x576.

`image-to-video` takes input references, separated by commas, each tagged by its role:

| Reference | What it does |
| --- | --- |
| `first=frame.png`, `last=end.png` | The frames the video starts and ends on, one of each at most. An untagged image is a frame too, the first then the last. |
| `image=hero.png` | A picture reference: `<Picture 1>`, `<Picture 2>`... in the prompt, in order. |
| `audio=voice.wav` | A voice or sound reference: `<Audio 1>`... ("he says, in the voice of `<Audio 1>`"). |
| `guide=line.wav@3.5` | An exact recording placed in the soundtrack at 3.5 seconds, the video made to match it (lip sync). |

With a picture or audio reference the engine switches to MiniMax H3's reference model; the frames
and the guide go with either:

```sh
sh image-to-video.sh comfy-minimax-h3-hq cuda first=fedora.png,audio=voice.wav take.mp4 \
                     "then he says, in the voice of <Audio 1>: ..." 5 1376 576
```

The three engines are one model: `minimax-h3` is MiniMax's release (50 steps, 5 to 15 seconds),
`comfy-minimax-h3` the files of ComfyUI's template (20 steps, much lighter), and
`comfy-minimax-h3-hq` ComfyUI's highest precision files, its text encoder in bf16, each frame
center-cropped, the beta scheduler with references and the video at crf 12 (finer than 18).
`minimax-h3` takes references alone, as diffusers runs them: no frame and no guide beside them,
and never an audio reference on its own.

The video scripts take these options for the `comfy-minimax-h3` engines, comma separated:

| Option | What it does |
| --- | --- |
| `attention=cudnn` | cuDNN attention ahead of flash: about twice as fast on a B200 (2.3 s a step against 4.6 at 1376x576), the same quality but not the same bits. |
| `scheduler=beta` | The beta scheduler in place of the engine's: more steps where the noise is high and where it is low, fewer between. `comfy-minimax-h3-hq` already uses it with references; `scheduler=simple` gives it the templates' instead. |
| `crop=center` | Each frame first fitted to the video by a center crop, as `comfy-minimax-h3-hq` does; `crop=none` leaves the frames to ComfyUI's node, as the templates do (the first frame stretched, the last cropped). With references, the frames are always center-cropped. |

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
model. [test/old/README.md](test/old/README.md) has longer recipes, with the images to try them on.

## Upscaling an image

An upscaler is an image-to-image engine that makes its first input as big as the width and the
height you give:

```sh
sh image-to-image.sh esrgan-upscale cuda photo.png big.png none 2048 1536
sh image-to-image.sh comfy-seedvr2-3b-upscale cuda photo.png big.png none 2048 1536
```

`esrgan-upscale` sharpens what is there and is done in seconds; `comfy-esrgan-upscale` is the same
on ComfyUI's file. `comfy-seedvr2-3b-upscale` and `comfy-seedvr2-7b-upscale` run SeedVR2, a
diffusion model that redraws the picture sharp at the output size in a single step, from ComfyUI's
files and on the offloader; the 7B gives more detail and needs more memory. They take no prompt
(`none`). Give a size of the input's shape, the input is stretched to it otherwise. The other
arguments and the `align` and `color` options are those of any image to image run. A transparent
input keeps its alpha. `check-model.sh MODES:image-to-image` marks the upscalers `upscaler`.

## Adjusting an existing image

`image-adjust.sh` gives an image made earlier the fixes of `align` and `color`, against the image
it was made from:

```sh
sh image-adjust.sh clean.png frame.png fixed.png color
sh image-adjust.sh edit.png frame.png fixed.png align,color
```

The image to fix, its reference and the output come first, then the fixes: `align`, `color`, or
both, each with its limit if you give one (`align,color=24`), and a server last if you use one.
It needs no download and no graphics card, and takes about a tenth of a second for the colors of
a 16 MP image. `align` leaves an image of another shape than
its reference as it is: it measures a drift, not a reframing.

## Running a server

Each script run starts python and loads the model, which can take longer than making the image. A
server keeps the model loaded between images, so only the first one pays for it:

```sh
sh server.sh start
sh text-to-image.sh z-image-turbo cuda cat.png "a cat" 1024 768 -1 -1 offloader none none 8080
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

On a card that holds the whole model, an H200 or a B200 for MiniMax H3, start the server or a
script with `OFFLOADER_DISABLE_PINNED_MEMORY=1` (or pass `--disable-pinned-memory` to
`runner.server` or `runner.cli`): the offloader then keeps no pinned copy of the weights in RAM,
as ComfyUI's flag of that name does. The same images and videos with half the RAM, and a faster
first load: on a B200, `comfy-minimax-h3-hq` peaks at 100 GiB instead of 190 and takes 94
seconds instead of 138 for its first video. Leave it off on a smaller card, where the weights
stream to it every step.

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

## Testing an update

`test/run.sh` checks that an update leaves every installed engine's image byte-identical and its
speed intact: `test/run.sh --record` before the update, `test/run.sh` after, for all engines or the
ones you name. It can compare the comfy engines with ComfyUI too. The video engines are a suite of
their own, `test/run.sh --video`, for a card that holds MiniMax H3.
[test/README.md](test/README.md) has the details, and how to run it on cloud GPUs (Modal), every
engine at once.

## Using it from Python

turboCLI can run inside a python application rather than through the scripts:
`runner.core.generate(params, emit)` takes the same settings as the scripts and calls `emit` with
each line they print. [implementation.md](implementation.md) describes it.
