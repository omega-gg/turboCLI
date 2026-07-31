# image-mask.sh: standalone mask/merge, decoupled from image-to-image generation

## Context

Commit `88dedb2` folded a `--preserve`/`--threshold` post-process into the `image-to-image`
generation path (`core._postprocess_i2i`, plus `mask`/`region` helpers). The user has decided that
was the wrong seam: **`image-to-image` should be a pure image-to-image generator again** (no
preserve, like before the commit), and the mask/merge should be a **separate, generation-free
command** — `image-mask` — that one runs *after* an edit if desired.

`image-mask` takes a **reference** image (the original scene, the base canvas) and an **input**
image (an edited/generated frame), applies a mask, and merges the input's changed region back onto
the reference. No engine, no prompt, no GPU — pure PIL + numpy. Confirmed with the user:

- Args: **`image-mask <reference> <input> <output> [mode = mask]`**.
- **`mode = mask | region`** (no `none`): `mask` = soft pixel-diff (add/recolor), `region` = grown
  bounding boxes (remove/replace, ghost-free). Default `mask`.
- **No threshold on the CLI** — the change threshold (and feather/dilate/grow/min-area) default to
  fixed values inside the Python code.

## 1. Revert the generation-side of `88dedb2` (make image-to-image pure again)

Restore these to their pre-commit (HEAD~1) state — remove all preserve/threshold/mask code:
- `runner/core.py` — drop `_diff_mask`/`_region_mask`/`_blob_boxes`/`_postprocess_i2i` and the
  save-block hook. Cleanest: `git checkout HEAD~1 -- runner/core.py` (the helpers are re-created in
  `runner/mask.py` below, so nothing is lost).
- `runner/cli.py` — `git checkout HEAD~1 --` it (drops `--preserve`/`--threshold`).
- `bash/turbo/image-to-image.sh` — `git checkout HEAD~1 --` it (pristine, `[server]` back at slot 13).
- `bash/turbo/README.md`, `implementation.md`, `turboCLI.pro` — `git checkout HEAD~1 --` them, then
  re-apply only the image-mask additions in steps 4–5 (keeps the diff clean and removes the stale
  preserve docs + the `doc/image-edit-plan.md` `.pro` line in one step).
- `git rm doc/image-edit-plan.md` — it documented the abandoned preserve-in-i2i design.

## 2. New `runner/mask.py` — self-contained mask/merge (PIL + numpy, no torch)

A sibling runner entry invoked as `python -m runner.mask`, modeled on `runner/install.py`'s
independence: it does **not** import `core` (so no backend discovery, no torch) — it only needs PIL
+ numpy, already in the venv. Move the three helpers here verbatim from the committed `core.py`
(`_blob_boxes`, `_region_mask`, `_diff_mask`), add fixed default constants, a `merge()`, and
`main()`.

```python
THR, MIN_AREA = 24, 400          # change threshold; drop blobs smaller than this
DILATE, FEATHER_MASK   = 5, 3    # mask-mode margin + seam
GROW,   FEATHER_REGION = 60, 18  # region-mode box grow + seam

def merge(reference, edit, mode):
    """Composite `edit`'s changed region onto `reference`; return (result, mask). Output is at the
    reference's resolution: the mask is built at the edit's resolution against a downscaled
    reference (what the edit is a version of), then mask + edit are upscaled onto the full-res
    reference. Same-size inputs => the resizes are identities. Byte-exact outside the mask."""
    ref = reference.resize(edit.size, Image.Resampling.LANCZOS)
    if mode == "region":
        mask = _region_mask(edit, ref, THR, GROW, FEATHER_REGION, MIN_AREA)
    else:
        mask = _diff_mask(edit, ref, THR, DILATE, FEATHER_MASK)
    up = edit.resize(reference.size, Image.Resampling.LANCZOS)
    m  = mask.resize(reference.size, Image.Resampling.LANCZOS)
    return Image.composite(up, reference, m), mask
```

`main()`: argparse `--reference --input --output --mode` (default `mask`); load both RGB; call
`merge`; save the result; print `mask[<mode>]: masked N%` (coverage) then `Saved: <output>` — the
same success sentinel the wrappers grep. Errors print `ERROR: ...` and exit 1 (mirror cli.py).
≤99 cols, ASCII-only prints.

Note the resolution rule doubles as the deferred "full-res back-composite": pass a full-res
reference + a smaller edit and the merge lands at full resolution automatically.

## 3. New `bash/turbo/image-mask.sh` — generation-free wrapper

Model on `install.sh`'s skeleton (getSky/getOs/getPath, venv activate) minus HF/CUDA env, no server:
- Args: `reference` (1), `input` (2), `output` (3), `mode` (4, default `mask`).
- Guard: `[ $# -lt 3 -o $# -gt 4 ]`, and validate `mode` is `mask` or `region`.
- `getPath` all three image paths (Windows `cygpath -w` + `\`→`/`, same as the others).
- `cd "$bin"`, activate `.venv`, then:
  ```sh
  python -m runner.mask \
         --reference "$reference" \
         --input     "$input" \
         --output    "$output" \
         --mode      "$mode"
  ```
- Usage text documents `mask` vs `region` + two examples. Keep every line ≤99 cols.

## 4. Registration

- `bash/turbo/turbo.pri`: add `bash/turbo/image-mask.sh \` to OTHER_FILES.
- `turboCLI.pro`: add `runner/mask.py \` to the runner OTHER_FILES list; replace the removed
  `doc/image-edit-plan.md` line with `doc/image-mask-plan.md` (added in step 7).

## 5. Docs

- `bash/turbo/README.md`: image-to-image section is pristine again (from the revert); add an
  `### image-mask.sh` section — usage block, the mask/region note, examples.
- `implementation.md`: after the `check.py` subsection add a short **`mask.py`** note — a
  torch-free post-generation merge tool (reference + edit → composited output), `mask` vs `region`,
  reused by `image-mask.sh`; and add `image-mask.sh` to the scripts line in the Layout block.

## 6. Deploy mirror sync (after editing)

- `cp runner/core.py runner/cli.py runner/mask.py Sky-runtime-bin/gg.omega/turbo/runner/`
  (core/cli reverted to pristine; mask.py new).
- `cp bash/turbo/image-to-image.sh bash/turbo/image-mask.sh Sky-runtime-bin/gg.omega/turbo/bash/turbo/`

## 7. Copy the plan into doc/

Once implemented and verified, copy this plan file to `turboCLI/doc/image-mask-plan.md` (kept as a
record alongside the other `doc/*-plan.md` files; referenced from `turboCLI.pro` per step 4).

## Verification (manual)

Venv python: `/c/dev/workspace/msvc/Sky-runtime-bin/gg.omega/turbo/.venv/Scripts/python.exe`.

1. **image-to-image is pure again** — a normal i2i run prints no `preserve[...]`/`mask[...]` line;
   `image-to-image.sh` usage no longer mentions preserve; `git diff HEAD~1 -- runner/core.py
   runner/cli.py bash/turbo/image-to-image.sh` is empty.
2. **image-mask add (mask)** — reference = original courtyard, input = `add_raw.png` (chest edit):
   `image-mask courtyard.png add_raw.png out.png mask` → prints `mask[mask]: masked ~21%`, `Saved:`;
   pixels outside the chest byte-identical to the reference; result matches the prior `add_edit.png`.
3. **image-mask remove (region)** — reference = `flux2_klein_hu7avjv7.png`, input = `knight_edit.png`,
   `region` → `mask[region]: masked ~50%`, knight gone, no ghost, outside byte-exact (matches the
   validated `merge_region.py` / `knight_removed_region.png`).
4. **Full-res back-composite** — reference = a large original, input = a smaller edit of it → output
   at the reference's resolution, byte-exact outside the mask.
5. **Torch-free** — `python -m runner.mask ...` runs without importing torch (fast start, no GPU).
6. **99-col + ASCII** — new python and bash lines ≤99 cols; prints ASCII-only.

## Critical files
- `runner/mask.py` — NEW self-contained mask/merge + `main()` (helpers moved from committed core.py).
- `bash/turbo/image-mask.sh` — NEW generation-free wrapper.
- `runner/core.py`, `runner/cli.py`, `bash/turbo/image-to-image.sh` — reverted to HEAD~1 (pristine).
- `bash/turbo/turbo.pri`, `turboCLI.pro`, `bash/turbo/README.md`, `implementation.md` — register +
  document image-mask; drop `doc/image-edit-plan.md`.
- Deployed mirror under `Sky-runtime-bin/gg.omega/turbo/` — copy after editing.
