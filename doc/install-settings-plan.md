# turboCLI: an install that remembers how a model runs

## Context

`install` takes a dtype and nothing else. Everything about *how* a model actually runs -- the
renderer it was meant for, the inference steps, the offload backend, whether to slice -- is typed
again on every `text-to-image` / `image-to-image` call, and nothing on disk says what a given
install was set up for. A host app that wants to offer sensible defaults has nowhere to read them
from, and a machine that installed for `cuda` looks exactly like one that installed for `cpu`.

So install takes the same four options the run scripts take, in the same order, records them with
the dtype in the engine's registry entry, and `check-model` hands them back.

## The command

```
install <engine> <renderer> [dtype = default] [inference = -1] [offload = offloader]
        [slicing = none] [ComfyUI folder]
```

- `<renderer>` is required and validated `cpu|cuda|mps`, the way
  [text-to-image.sh](../bash/turbo/text-to-image.sh) validates its own.
- `slicing` is validated `none|slice`, and `dtype` keeps the check it has - including the
  float32 -> bfloat16 coercion, which stays where it is.
- The argument count becomes 2 to 7, `[ComfyUI folder]` staying last.
- The four go to python as `--renderer --inference --offload --slicing`.

The usage block and its examples change with it, and so does
[bash/turbo/README.md](../bash/turbo/README.md), which mirrors each script's usage verbatim -
`implementation.md` lists that as one of the four things that rot silently.

## The record

`engine/<id>/engine.json` gains one block beside `model` / `revision` / `loras`:

```json
{ "id": "z-image-turbo", "model": "Z-Image-Turbo", "revision": "04cc4ab...", "loras": [],
  "settings": { "renderer": "cuda", "dtype": "bfloat16", "inference": "-1",
                "offload": "offloader", "slicing": "none" } }
```

- It is called `settings`, not `options`: `cli.py --options` already means the engine-specific
  `key=value` pairs (`cutoff=40`, `mode=composite`), and these are not those.
- **One place writes it.** Every install path ends in `_write_engine` - six call sites over four
  record shapes (`_stock_record`, `_kind_record`, the comfy dict, the register-only record) - so
  `main()` settles the block once and `_write_engine` merges it into whatever record it is
  handed. Nothing is threaded through `_install_snapshot`, `_install_url` or `_install_comfy`.
- **A reinstall re-assigns by construction.** The "already installed" early return writes the
  record too, so `install z-image-turbo cuda bfloat16` over a cpu install rewrites the block
  without downloading anything.
- A record written before this round has no block, and every reader fills it from
  `SETTINGS_DEFAULT`: `cpu`, `default`, `-1`, `offloader`, `none`. The engines already installed
  on a machine keep running exactly as they did - the block only says what a host should offer.
  On a CUDA machine those records are worth a one-off pass writing `cuda` into them, which is
  what was done here for the thirteen engines of this install.

## dtype is a preference, not a promise

Weights are cast only on a base (re)install. Once the model is present `base_ok` skips the
download, so a new dtype changes the record and not the files. Install says so on that path, in
one line naming `remove` as the way to recast - which is the behaviour asked for: a custom model
reinstalled with another dtype needs a plain `remove` first.

## Retrieval

`check-model` gains a query beside `MODES:`, so the fixed one-line outputs it is polled for stay
exactly as they are:

```
check-model SETTINGS:<engine>    ->  renderer: cuda
                                     dtype: bfloat16
                                     inference: -1
                                     offload: offloader
                                     slicing: none
```

- `runner.check --settings <engine>` behind it, reading the record with `_read_engine` and
  filling what an older record omits. An engine that is not installed prints nothing and exits 1,
  like every other check.
- `check-model <engine>` keeps `<id> is installed` and its exit code, and `MODES:` keeps its list
  - those are what turbopixel polls today.

## Files

- `bash/turbo/install.sh` - the syntax block, the validation, the four flags
- `runner/install.py` - the four arguments, the block, its merge in `_write_engine`, the dtype
  notice
- `runner/check.py` and `bash/turbo/check-model.sh` - the `SETTINGS:` query
- `bash/turbo/README.md` and `implementation.md` - the install line, the check-model contract, the
  hand-written lists table
- `doc/install-options-plan.md` - this plan, with `turboCLI.pro` following it

## Measured

On the deployed install, on engines already present so nothing downloads:

| command | what happened |
| --- | --- |
| `install z-image-turbo cuda bfloat16 -1 offloader none` | no download, the block written, and
  the dtype notice printed |
| `install z-image-turbo cpu` | the whole block re-assigned - renderer `cpu`, dtype back to
  `default` - the files untouched |
| `install mask cuda` | a register-only engine records the block too |
| `check-model SETTINGS:z-image-turbo` | the five `key: value` lines, exit 0 |
| `check-model SETTINGS:nope` | nothing, exit 1 |
| `check-model z-image-turbo` / `MODES:text-to-image` | unchanged, one line and the id list |
| `install z-image-turbo` / `... gpu` / `... none none sliced` | the usage block, exit 1 |

## Verification

Run from `Sky-runtime-bin/gg.omega/turbo` (the deployed install), on an engine already installed
so nothing downloads:

1. `sh install.sh z-image-turbo cuda bfloat16 -1 offloader none` - no download, `engine.json`
   holds the block, and the dtype notice prints because the weights keep what they were saved
   with.
2. Again with `cpu 4 model_cpu slice` - the block changes, the files do not, and a `git`-less
   diff of the two records shows only `settings`.
3. `sh check-model.sh SETTINGS:z-image-turbo` prints the five lines; on an engine that is not
   installed it prints nothing and exits 1.
4. `sh check-model.sh z-image-turbo` still prints one line, and `sh check-model.sh
   MODES:text-to-image` still lists ids - turbopixel's engine menu still fills.
5. `sh install.sh mask cuda` - a register-only engine records the block with no download.
6. `sh install.sh comfy-z-image-turbo cuda default -1 offloader none <ComfyUI dir>` - the folder
   is still read as the last argument, and the comfy record keeps its `comfy` block beside the
   new one.
7. `sh install.sh z-image-turbo` alone now fails with the usage, since the renderer is required.
8. A record from before this round: `SETTINGS:` prints the fallbacks rather than failing.
