# image-mask: split GENERATION from APPLICATION

## Context

`image-mask` (torch-free `runner/mask.py`) used to compute a diff/region mask **and** composite it
onto the reference in one `merge()`. `image-remove-background` (torch `extract.py`) computed a
subject alpha matte **and** `putalpha`d it in one pass. Both fused "make a mask" with "use the
mask".

Making the mask a first-class, reusable artifact lets a matte generated once be applied several
ways (transparency, or composite over the original / a new backdrop). The seam already existed
internally; this exposes it as three commands (pure split — the old fused commands are removed).
Torch-free stays torch-free, and it is as fast as before: no model re-run, apply is cheap PIL.

## Final CLI

```
image-mask            <mode> <reference> <input> <mask output> [threshold]         # gen diff/region
image-mask-background <model> <renderer> <input> <mask output> [plate] [shadow threshold]  # gen matte
image-mask-apply      <mode> <input> <mask> <output> [reference]                   # apply
```
- `image-mask` mode = `mask|region`; `image-mask-apply` mode = `composite|putalpha`
  (`composite` needs the reference = 5 args; `putalpha` = 4 args).
- `reference` (composite only) = the background shown where the mask is not opaque
  (`Image.composite` picks input where mask=white, reference where mask=black): restore the original
  scene after an edit, or drop a subject onto a new backdrop. `putalpha` has no reference (RGBA out).
- The mask/matte is always an **8-bit grayscale (`L`) PNG at the input resolution** — the one format
  both generators emit and `image-mask-apply` consumes.

## What changed

- **`runner/mask.py`** — generate-only. `merge()` → `build_mask(reference, edit, mode, thr=THR)`
  (the exact old mask-building lines); `main()` saves the mask. argparse + prints unchanged.
- **`runner/apply.py`** — NEW, torch-free (PIL, no numpy beyond `ImageStat`). `apply(source, mask,
  mode, reference=None)`: composite (upscale source+mask to reference, `Image.composite`) or putalpha
  (mask as source's alpha). `main()`: `--mode/--input/--mask/--output/--reference`.
- **`bash/remove-background/extract.py`** — saves an 8-bit `L` matte
  (`Image.fromarray((alpha*255)…,"L").save`) instead of `putalpha`. `_subject_alpha`, the plate/shadow
  fuse, `--shadow-threshold`, and prints unchanged.
- **Bash**: `image-mask.sh` help only; NEW `image-mask-apply.sh` (turbo venv, `runner.apply`);
  `image-remove-background.sh` → `image-mask-background.sh` (still delegates to `run.sh`, now emits a
  matte); `run.sh` help → matte.
- **Satellites**: `bash/turbo/turbo.pri` (order: image-mask, image-mask-background, image-mask-apply);
  `turboCLI.pro` (+`runner/apply.py`, +this doc); `bash/turbo/README.md`, `bash/remove-background/
  README.md`, `test/README.md`, `implementation.md`.

## Byte-exactness (decoded-array identity)

- **composite == old `merge`**: generate issues `reference.resize(edit.size)` + `_diff/_region_mask`
  and saves the `L` mask losslessly; apply issues `edit.resize(reference.size)` +
  `mask.resize(reference.size)` + `Image.composite` — the same calls on the same pixels. Holds for
  same-size and for full-res-reference + smaller-edit (mask at edit res, apply upsamples).
- **putalpha == old `extract`**: new extract saves the identical `L` matte; apply opens input RGB and
  `putalpha`s it (== `convert("RGBA")` + `putalpha`); matte already at input size ⇒ no resize.
- Verified `np.array_equal` EQUAL on: mask (default + threshold 40), region, full-res-ref+smaller-edit
  (composite path); and cutout with/without plate at default + non-default shadow threshold (putalpha).

## Risks
- Mask MUST be `.png` (8-bit `L`) — a `.jpg`/`"1"`-mode save breaks byte-exactness.
- The LIVE `gg.omega/remove-background/extract.py` install must be updated (it runs at runtime).
- Clear the mirror's `runner/__pycache__` so the removed `merge` bytecode isn't served.
