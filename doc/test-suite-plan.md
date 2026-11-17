# Plan: a test suite, identical images and top-notch speed

## Context

Every change to turbo-offloader or turboCLI was verified by hand: ad hoc scripts ran each engine
at seed 42, compared md5s with earlier runs and timed sampling against ComfyUI. This turns that
into a committed suite under `test/` that anyone runs from the command line, per engine or for
all installed engines, stock and comfy (absent ones are skipped), on CUDA, CPU or MPS, on a
laptop or a cloud GPU. It fails when an output image is no longer byte-identical, or when an
engine got slower than its reference beyond a tolerance. It can also run each comfy engine
against a ComfyUI install, alongside turbo or alone. Options keep a throttling box from raising
false negatives.

Images are only byte-identical on the same machine and stack, so each machine keeps its own
reference (md5s and timings), committed under `test/reference/`, recorded before an update and
checked after it.

## Files

- `test/run.sh`: the suite (bash plus perl, present on Git Bash, Linux and macOS; no bash 4
  features, so macOS's bash 3.2 runs it).
- `test/comfy.py`: the ComfyUI side, run with ComfyUI's own python: the template graph per comfy
  engine, queued through ComfyUI's API, and the PSNR between two images.
- `test/reference/<renderer>-<device>-<size>.txt`: one committed reference per machine.
- `test/README.md`: usage, the record-then-check workflow, the options.
- `.gitignore`: `test/results/`, where each run leaves its images, logs and summary.
- Updated: `turboCLI.pro`, `implementation.md` (layout, a "Tests" section), `MANUAL.md`.

## test/run.sh

```
test/run.sh [options] [engine ...]        no engine: every installed one

  --renderer cuda|mps|cpu   default: cuda when nvidia-smi answers, mps on macOS, else cpu
  --size N                  square output size, default 1024 on cuda, 512 on mps and cpu
  --record                  write this machine's reference instead of checking against it
  --repeat N                run each case N times, keep the fastest (every md5 must agree)
  --cool                    before each run, wait for the GPU to cool to 50 C (nvidia-smi),
                            or pause 60 s on mps and cpu
  --tolerance P             percent slower than the reference (or ComfyUI) still passing, 10
  --comfy DIR               also run the comfy engines through the ComfyUI install in DIR
  --comfy-only DIR          run the comfy engines through ComfyUI alone, no turbo
```

| run | sides | what it guards |
|---|---|---|
| `test/run.sh [engine ...]` | turbo, every engine | a turboCLI or turbo-offloader update |
| `--comfy DIR` | turbo and ComfyUI, alternated | the same, plus turbo against ComfyUI |
| `--comfy-only DIR` | ComfyUI | a ComfyUI update (a re-sync to a new tag), without turbo |

The reference holds one line per (side, engine, mode), `turbo` or `comfy`, with its md5 and
timings, so a run checks the sides it ran against their own lines and `--record` writes only
those.

- **Cases.** `bash/turbo/check-model.sh ENGINES:<mode>` lists every engine of a mode, stock and
  comfy, `installed` or `absent`: one case per installed (engine, mode) among text to image,
  image to image and image to mask. With `--comfy-only` (no turbo install needed), the cases are
  those `comfy.py` has a graph for, each skipped unless its files are in that ComfyUI.
- **Fixed inputs.** Seed 42, the engine's default steps and its installed offload
  (`check-model.sh SETTINGS:<engine>`). Text to image: "a red fox sitting in a snowy forest,
  golden hour". Image to image: `test/old/knight.png` with one edit prompt. Masks: `mask` on
  `test/old/chest.png` against `test/old/courtyard.png`, the matte engines on the knight.
- **Runs.** The repo's own `bash/turbo/<mode>.sh`, which find the install through `SKY_PATH_BIN`
  (and `SKY_PATH_TURBOCLI` for a dev runner), so the suite tests whatever is installed.
- **Timing.** Every output line is stamped as it arrives (perl, `Time::HiRes`; the runner flushes
  each line, `runner/cli.py`), giving a millisecond breakdown off the runner's own lines
  (`runner/core.py` emit_step): load (start to `model ready`), encode (`generating` to
  `step 0/N`), sampling (`step 0/N` to `step N/N`), decode and save (to `Saved:`). Mask engines
  have no steps, and their wall time is mostly start-up: it is shown, not judged. Sampling is
  what passes or fails; the rest is reported.
- **Check.** Per case: the md5 must equal the reference's (`SAME` / `DIFFERS`), sampling must be
  within the tolerance (`+3%`, `SLOWER` past it); a case the reference lacks is `NEW`. A summary
  in `test/results/<machine>/summary.txt`; exit status 1 if any case fails.
- **Record.** `--record` replaces this machine's lines for the cases it ran, under a header with
  the date and the installed turboCLI commit (the install's `.commit`).
- **Machine name.** The renderer, the GPU (`nvidia-smi`), the chip (`sysctl` on macOS) or the CPU
  (`/proc/cpuinfo`, Git Bash too), and the size.

## Throttling

- `--cool` before every run, turbo's and ComfyUI's, recording included: a reference recorded
  hot makes every later run look faster.
- `--repeat N` keeps each case's fastest sampling, the least throttled run.
- With `--comfy`, turbo and ComfyUI alternate, so heat lands on both alike.

## test/comfy.py

- Graphs for the comfy engines checked against ComfyUI: comfy-flux2-4b, comfy-flux2-9b-distilled,
  comfy-flux3-6b, comfy-z-image-turbo, comfy-krea2-turbo and its two edits (their own boost and
  placement), comfy-qwen-image-2-1-turbo text to image and edit. Each is its ComfyUI template's;
  the model files come from the engine's `COMFY` declaration, so they follow the engine. Z-Image
  samples with euler, the sampler of diffusers' scheduler, in place of the template's
  res_multistep.
- `run.sh` starts a fresh ComfyUI server per run on port 8199 (`--cpu` on cpu), as a one-shot
  turbo run starts fresh, stamps its output the same way, and stops it after. ComfyUI's python:
  its portable `python_embeded`, its `venv`, or `COMFY_PYTHON`.
- ComfyUI's sampling: its progress bar from `0/N` to `N/N`, stamped on arrival (its own elapsed
  is shifted by `model_trange` after step 2, so the stamps are what count).
- With both sides: turbo's sampling against ComfyUI's (fails past the tolerance), and the PSNR
  between their images (reported, not a pass condition: the samplers differ by design).

## Verification

1. `test/run.sh --record mask comfy-flux2-4b` writes the laptop's reference; `test/run.sh mask
   comfy-flux2-4b` then passes with `SAME`.
2. `test/run.sh --record flux2-4b` covers a stock engine the same way.
3. `test/run.sh --record --comfy <ComfyUI> comfy-flux2-4b` alternates the two and reports the
   PSNR and the speed against ComfyUI; `--comfy-only <ComfyUI> comfy-flux2-4b` then checks
   ComfyUI alone against its `comfy` lines.
4. CPU and MPS: `--renderer cpu --size 256 mask-birefnet`, and the same on a Mac.
