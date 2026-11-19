# [turboCLI](../README.md) tests

`run.sh` runs each engine, stock and comfy, at a fixed seed and checks two things against this
machine's reference: its image is byte-identical, and its sampling is no slower beyond a
tolerance. Run it before and after a turboCLI or turbo-offloader update, locally or on a cloud
GPU, on CUDA, MPS or CPU. It needs bash and perl, which Git Bash, Linux and macOS have, and
`SKY_PATH_BIN` pointing at the install.

```
test/run.sh [options] [engine ...]
```

No engine runs every installed one; an engine that is not installed is skipped. Each runs the
modes it has: text to image (a fox in a snowy forest), image to image (an edit of
[old/knight.png](old/knight.png)), image to mask (the `mask` engine on
[old/chest.png](old/chest.png) against [old/courtyard.png](old/courtyard.png), the matte engines
on the knight) and image upscale ([old/knight-small.png](old/knight-small.png), 368x512, to
the size's height: x2 at 1024, x4 at 2048, x8 at 4096, x2 at least). `esrgan` has no steps:
like a mask, its time is shown, not judged.

| option | |
|---|---|
| `--renderer <cuda\|mps\|cpu>` | cuda when nvidia-smi answers, mps on macOS, else cpu |
| `--size <N\|WxH>` | output size, `N` square or `WxH` (e.g. `1024x768`); 1024 on cuda, 512 on mps and cpu |
| `--record` | write this machine's reference instead of checking it |
| `--repeat <N>` | run each case N times and keep the fastest; the image must not change |
| `--cool` | before each run, wait for the GPU to cool to 50 °C, or pause 60 s on mps and cpu |
| `--warmup` | before each case, one run of each side that is not kept, so no measured run reads the files cold |
| `--tolerance <P>` | percent slower still passing, 10 by default |
| `--comfy <dir>` | also run the comfy engines through the ComfyUI in `<dir>` |
| `--comfy-only <dir>` | run the comfy engines through that ComfyUI alone, no turbo |

## Record, update, check

Images are only byte-identical on the same machine and software stack, so each machine has its
own reference in [reference/](reference/), named by its renderer, device and size, and committed:

```
test/run.sh --record                # before the update: this machine's images and timings
...                                 # update turbo-offloader or turboCLI, deploy it
test/run.sh                         # after: SAME and within the tolerance, or it fails
```

Each run prints its time from launch to image: ComfyUI's from its server's launch to the image it
returns, turbo's from its script's launch to its `Saved:` line, where a client takes the image.
Each case prints its sampling time, then the load, encode and decode around it, and its result:
`SAME` or `DIFFERS` for the image, the time against the reference (`+3%`), and `SLOWER` past the
tolerance. The mask engines have no steps and their time is mostly start-up: it is shown, not
judged. A case the reference does not have yet is `NEW`. The run exits with 1 when any case
fails, and leaves its images, logs and summary under `results/` (not committed).

A laptop throttles as it heats up, which reads as a slowdown that is not one. `--cool` starts
every run cool, `--repeat 2` or more keeps each case's least throttled run, and prints `UNSTABLE`
if a case's image changes between its runs. Record with them too: a reference recorded hot makes
every later run look faster, and hides a real slowdown. Each repeat swaps which side goes first,
so neither always reads the files cold.

With `--repeat 1` turbo always goes first, so against ComfyUI it alone reads the model files cold
and ComfyUI finds them cached. The cold read lands in turbo's load and encode, and in its
sampling too, since a streamed model reads its weights during the first step: on Modal, where the
files are on a network Volume, comfy-minimax-h3-image at 4096² sampled in 23.27 s against
ComfyUI's 18.90 s (+23%, `SLOWER`) with one repeat, and in 19.30 s against 19.32 s with two.

Repeats alone leave a smaller bias: the first step keeps getting faster run after run as the files
warm, so the side running last gets the warmest one, and with an odd `--repeat` that is ComfyUI.
comfy-z-image-turbo at 1024² on the L40S, three repeats: turbo's first step 6.66, 3.79, 3.86 s
(first, second, first), ComfyUI's 3.86, 3.81, 3.61 s, a +2% that was the order. `--warmup` runs
each side once before measuring, every measured run then warm. Judge turbo against ComfyUI with
`--warmup --repeat 2`.

## Against ComfyUI

`--comfy <dir>` runs each comfy engine that has a ComfyUI graph in [comfy.py](comfy.py) through
the ComfyUI install in `<dir>` (the folder with `main.py`) too, alternating with turbo so heat
lands on both alike. Each side is checked against its own reference lines, and turbo's sampling
against ComfyUI's, with the PSNR between their images (reported only: the samplers differ by
design). `--comfy-only <dir>` runs ComfyUI alone, for a ComfyUI update (a re-sync of the
vendored code to a new tag). ComfyUI runs with its portable `python_embeded`, its `venv`, or
`COMFY_PYTHON`, on port 8199, a fresh server per run.

The model files come from each engine's `COMFY` declaration, so the graphs follow the engines;
an engine whose files are not in that ComfyUI is skipped.

## On Modal

`modal.sh` runs the same suite on an L40S GPU on Modal, for a second machine's reference, with
turboCLI's own scripts in the container:

```
test/modal.sh install comfy-flux2-4b z-image-turbo mask    # on CPU: install.sh per engine
test/modal.sh run --record comfy-flux2-4b                  # on the L40S: test/run.sh, options
test/modal.sh run --comfy /opt/ComfyUI                     # every engine, against ComfyUI
test/modal.sh run --renderer cpu --size 256 comfy-flux2-4b # on a CPU container, no GPU billed
```

The image holds ComfyUI too, in `/opt/ComfyUI` (`test/comfyui.sh`: the tag turbo-offloader
vendors, on turboCLI's torch, with the Krea 2 edit node pack), its models the Volume's. A CPU run
keeps the CPU path honest; it is slow, so a small size and a few engines, now and then.
`modal.sh exec <command>` runs a command in the same setup on CPU, for a look around, and
`modal.sh exec --gpu <command>` on an L40S. A run's logs and images come back into
`results/modal/`, and what an `exec` writes to `out/` into `results/modal/exec/`.

The L40S containers have 64 GiB of RAM. A model past that, MiniMax-H3-Image (ComfyUI's 49.5 GB
and a process, the stock 135 GB), would stream off the Volume every step, so its runs take
`--big`, a container with 224 GiB (`modal.sh run --big ...`, `modal.sh exec --big ...`). It is
made for 4 to 20 MP, best at 16 MP, so it runs at `--size 2048` and `--size 4096` too, each size
its own reference file.

The working trees of turboCLI and turbo-offloader (`../turbo-offloader`, or `TURBO_OFFLOADER`) go
up as they are, so it tests what is about to be committed. The image is turboCLI as its
`build.sh` builds it, made again when the build scripts change; the models and engine records stay
on the `turbo-models` Volume, so a run downloads nothing. A run brings the L40S's reference and its
summary back into `test/`. It needs the Modal CLI logged in (`pip install modal`, `modal setup`),
and a Modal Secret `hf-token` holding `HF_TOKEN` for the gated engines.
[modal_app.py](modal_app.py) only carries the script into a container (`modal shell` would need no
Python, but has no Windows support). Each call is an ephemeral `modal run`: nothing stays deployed
or running. A run is parallel: one L40S per engine (`MODAL_JOBS` at a time, 8 by default, under the
account's GPU limit), and `test/modal.sh run --list` prints the engines it would test without
starting any. Ctrl-C stops the containers a run started; a run killed otherwise can leave its apps
behind on Windows: `modal app list`, then `modal app stop -y <id>` for each.
