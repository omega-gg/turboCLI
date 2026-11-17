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
[old/knight.png](old/knight.png)) and image to mask (the `mask` engine on
[old/chest.png](old/chest.png) against [old/courtyard.png](old/courtyard.png), the matte engines
on the knight).

| option | |
|---|---|
| `--renderer <cuda\|mps\|cpu>` | cuda when nvidia-smi answers, mps on macOS, else cpu |
| `--size <N\|WxH>` | output size, `N` square or `WxH` (e.g. `1024x768`); 1024 on cuda, 512 on mps and cpu |
| `--record` | write this machine's reference instead of checking it |
| `--repeat <N>` | run each case N times and keep the fastest; the image must not change |
| `--cool` | before each run, wait for the GPU to cool to 50 °C, or pause 60 s on mps and cpu |
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

Each case prints its sampling time, then the load, encode and decode around it, and its result:
`SAME` or `DIFFERS` for the image, the time against the reference (`+3%`), and `SLOWER` past the
tolerance. The mask engines have no steps and their time is mostly start-up: it is shown, not
judged. A case the reference does not have yet is `NEW`. The run exits with 1 when any case
fails, and leaves its images, logs and summary under `results/` (not committed).

A laptop throttles as it heats up, which reads as a slowdown that is not one. `--cool` starts
every run cool, `--repeat 2` or more keeps each case's least throttled run, and prints `UNSTABLE`
if a case's image changes between its runs. Record with them too: a reference recorded hot makes
every later run look faster, and hides a real slowdown.

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
