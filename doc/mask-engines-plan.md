# Fold mask / segmentation / apply into the turboCLI engine system

## Context

Mask generation, background-matte generation, and mask application used to live OUTSIDE the engine
system: torch-free `runner/mask.py`/`apply.py` wrappers plus a standalone `bash/remove-background`
tool with its own venv/models. They could not be installed/checked/removed or driven by the server
the way the diffusion engines are. This folds all of it in as first-class engines behind two new
modes, so one uniform surface (`cli` + server, `install`/`check-model`/`remove`) covers everything,
and `bash/remove-background` is gone.

Design constraints (from the user): simple + readable + fast, and **zero regression** to the existing
diffusion engines. The load-bearing guarantee: the new `run` hook fires only for engines that declare
it, so diffusion engines take the exact current path; engine discovery stays torch-free and all heavy
imports live inside `run()`, so `text-to-image`/`image-to-image` never import the segmentation stack.

## Modes + engines

Modes: `image-to-mask`, `image-apply-mask`. Inputs ride the existing comma-separated `images` param;
the scalar option rides a new general-purpose `--options` (`key=value,...`).

| engine | mode | model install | images | options |
|---|---|---|---|---|
| `mask` | image-to-mask | register-only | input,reference | threshold=N |
| `mask-region` | image-to-mask | register-only | input,reference | threshold=N |
| `mask-birefnet` | image-to-mask | snapshot `ZhengPeng7/BiRefNet` | input[,plate] | threshold=N (shadow) |
| `mask-lucida` | image-to-mask | snapshot `egeorcun/lucida`, BASE=`mask-birefnet` | input[,plate] | threshold=N |
| `mask-inspyrenet` | image-to-mask | url `ckpt_base.pth` (tag 1.2.12) | input[,plate] | threshold=N |
| `mask-apply` | image-apply-mask | register-only | input,mask[,reference] | mode=composite\|putalpha |

## Implementation

- **core.py**: an engine hook `run(ctx, params, emit) -> PIL.Image`. `generate()` short-circuits right
  after mode validation, before the offload ladder / get_pipe: build a lightweight `Ctx` + resolved
  model dir, call `run`, save, emit `Saved:`. Skips all diffusion/pipe machinery (incl. the unguarded
  `progress_bar` patch that would crash on a non-pipe). `"run"` added to `_inherit._INHERITED`.
- **cli.py**: a `--options` flag -> `params["options"]`; server already passes any field.
- **runner/engine/_*.py** (torch-free at import, discovery-skipped): `_options.py` (parse), `_mask.py`
  (`build_mask` diff/region, from mask.py), `_apply.py` (`apply` composite/putalpha, from apply.py),
  `_segment.py` (`birefnet_alpha`/`inspyrenet_alpha`/`build_matte`, from extract.py -- torch imports
  inside functions).
- **6 engine modules** (`mask.py`, `mask_region.py`, `mask_apply.py`, `mask_birefnet.py`,
  `mask_lucida.py`, `mask_inspyrenet.py`): thin declare + `run()`. `mask-lucida` BASE=`mask-birefnet`
  (run + MODES inherited; its own snapshot MODEL). MODEL gains a `"kind"`: snapshot | url (absent =
  diffusers).
- **install.py**: register-only branch (no MODEL -> write the engine.json record, no download);
  `snapshot` (whole HF repo verbatim into model/<name>) and `url` (raw asset) install kinds;
  `_engine_installed` marker generalized per kind. Reference-counted GC unchanged (keys off the model
  name).
- **bash/turbo/build.sh**: add timm/einops/kornia/transparent-background to the turbo venv (uv keeps
  the pinned cu130 torch -- no re-pin needed, unlike pip).
- **Wrappers**: `image-to-mask.sh` (`<engine> <renderer> <input images> <output> [options] [server]`)
  and `image-apply-mask.sh` (`<mode> <input images> <output> [server]`), mirroring image-to-image.sh
  (local + server branches). `install.sh` usage lists the 6.
- **Deleted**: `bash/remove-background/`, `runner/mask.py`, `runner/apply.py`, the three
  `image-mask*.sh` wrappers; all satellites de-registered.

## Verification (done)

- Byte-exact vs goldens captured from the old fused commands: `mask`/`mask-region` -> `mask-apply
  composite` == the old merges (default + threshold 40, incl. full-res-ref); `mask-birefnet` ->
  `mask-apply putalpha` == the old cutout (with/without plate). Also byte-exact through the server.
- install/check/remove for all 6 (register-only + snapshot + url); segmentation E2E for
  birefnet/lucida/inspyrenet.
- No regression: `uv pip freeze` shows torch/torchvision/numpy/diffusers UNCHANGED after the dep add;
  a diffusion engine (flux2-4b t2i) still loads and saves right after mask requests via the server.
