# image-edit.sh: a dedicated edit command with pixel-mask (add) + region-box (remove) preserve

## Context

`flux2-4b` image-to-image (`Flux2KleinPipeline`) regenerates the **whole frame from noise** and
VAE-decodes it, so every pixel drifts. Earlier (uncommitted) work added an opt-in `--preserve`
post-process to `core.py` that restores the original's pixels outside the edited region via a soft
difference mask ([core.py:341](turboCLI/runner/core.py#L341) `_postprocess_i2i` +
[core.py:311](turboCLI/runner/core.py#L311) `_diff_mask`), and wired `--preserve` into
`image-to-image.sh`.

Validated on real photos this session: the **pixel-mask** approach is right for **adding /
recoloring** (tight, high-contrast change → max preservation) but **ghosts on removals** — when a
dark object is replaced by a similar background, matching parts fall below threshold and survive as
a faint outline (the knight / TIE ghost). Proven fix (standalone `merge_region.py`): a **region-box**
mask — threshold → connected blobs → replace each blob's **grown bounding box** wholesale, so a
ghost is impossible inside the box (knight removal: clean, 0 ghost).

Goal + scope (confirmed with the user):
- **Separate command.** Restore `image-to-image.sh` to its pristine, README-documented form (raw
  reference generation, **no `--preserve`**). Put **all** preservation — mask *and* region — into a
  new **`image-edit.sh`**. Clean split: `image-to-image` = generate, `image-edit` = edit.
- **`image-edit.sh` exposes mode + threshold**; `mode = mask | region`. grow/feather/dilate stay
  **auto** (per-mode defaults); not exposed on the CLI.
- **Output stays at `width x height`** (same as today's post-process). Full-res back-composite for
  huge images is out of scope for now.

## 1. `core.py` — add the region helper + a mode switch (shared by the server & both fronts)

Both modes share the diff (`abs(g-o).max(2)`) and the final `Image.composite(edit, ref, mask)` at
generation size — only the **mask construction** differs. Add one helper beside `_diff_mask` and
branch in `_postprocess_i2i`. Port `merge_region.py` faithfully but single-threshold (its docstring:
`grow=60` at the noise floor is the workhorse, hysteresis "measured worse" — drop the low/high seed
split to keep it small). ≤99 cols, ASCII-only emits.

```python
def _blob_boxes(mask, min_area):
    """(x0,y0,x1,y1) bounding boxes of 4-connected blobs in a binary L mask; drop < min_area.
    Iterative flood fill (no scipy). Ported from merge_region.py `boxes`, single-threshold."""
    import numpy as np
    a = np.asarray(mask) > 127
    seen = np.zeros(a.shape, bool)
    h, w = a.shape
    out = []
    for y0, x0 in zip(*np.nonzero(a)):
        if seen[y0, x0]:
            continue
        stack, pix = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pix) < min_area:
            continue
        p = np.array(pix)
        out.append((p[:, 1].min(), p[:, 0].min(), p[:, 1].max(), p[:, 0].max()))
    return out


def _region_mask(generated, ref, thr, grow, feather, min_area):
    """Solid grown bounding boxes around changed blobs (255 = replace). For removal/replace: the
    box is filled solid so nothing of the old object can ghost inside it. See merge_region.py."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    g = np.asarray(generated, np.int16)
    o = np.asarray(ref, np.int16)
    diff = np.abs(g - o).max(2).astype(np.uint8)
    m = Image.fromarray(np.where(diff > thr, 255, 0).astype(np.uint8), "L")
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # open: despeckle
    m = m.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))   # close: holes
    out = Image.new("L", generated.size, 0)
    d = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in _blob_boxes(m, min_area):
        d.rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow), fill=255)   # grow clears clipped
    if feather > 0:                                                           # dark edges
        out = out.filter(ImageFilter.GaussianBlur(feather))
    return out
```

Mode switch inside `_postprocess_i2i` — keep the `preserve < 0` guard, first-input lookup, the
`ref = ImageOps.fit(original, image.size, LANCZOS)` build, and the final
`Image.composite(image, ref, mask)` **unchanged**; only choose the mask builder:

```python
    mode    = params.get("preserve_mode", "mask")
    feather = float(params.get("preserve_feather", -1))

    if mode == "region":
        grow    = int(float(params.get("preserve_grow", -1)))
        grow    = 60 if grow < 0 else grow
        feather = 18 if feather < 0 else feather
        mask = _region_mask(image, ref, preserve, grow, feather, 400)
    else:
        dilate  = int(float(params.get("preserve_dilate", -1)))
        dilate  = 5 if dilate < 0 else dilate
        feather = 3 if feather < 0 else feather
        mask = _diff_mask(image, ref, preserve, dilate, feather)

    emit("preserve[%s]: masked %.0f%% (thr=%g)"
         % (mode, np.asarray(mask).mean() / 255 * 100, preserve))

    return Image.composite(image, ref, mask)
```

`-1` = "auto per mode", so the wrappers only pass threshold + mode.

## 2. `cli.py` — keep the preserve flags, add mode/grow (used by image-edit.sh & server)

Existing `--preserve` (default `"-1"` = off) stays. Add `--preserve-mode` (default `"mask"`) and
`--preserve-grow` (default `"-1"`); change `--preserve-feather` / `--preserve-dilate` defaults to
`"-1"` (auto). Mirror all into `params`. See [cli.py:56-80](turboCLI/runner/cli.py#L56). Harmless to
`image-to-image.sh`: it won't pass any of them, so `preserve = -1` → `_postprocess_i2i` no-ops.

## 3. `image-to-image.sh` — REVERT to pristine (remove the preserve wiring)

Undo the earlier additions so it matches the README form
([README.md:130-163](turboCLI/bash/turbo/README.md#L130)):
- remove `preserve="-1"` from Settings ([:43](turboCLI/bash/turbo/image-to-image.sh#L43));
- restore the arg guard `-gt 14` → `-gt 13` ([:116](turboCLI/bash/turbo/image-to-image.sh#L116));
- remove the slot-14 `preserve` parse ([:189](turboCLI/bash/turbo/image-to-image.sh#L189)) and the
  `[preserve = ...]` / preserve-note usage lines ([:128,:150](turboCLI/bash/turbo/image-to-image.sh#L128));
- remove `--data-urlencode "preserve=$preserve"` ([:285](turboCLI/bash/turbo/image-to-image.sh#L285))
  and `--preserve "$preserve"` ([:359](turboCLI/bash/turbo/image-to-image.sh#L359)).

## 4. NEW `image-edit.sh` — modeled on the pristine `image-to-image.sh`

Copy the pristine `image-to-image.sh` structure (same getSky/getOs/getPath, server curl, env, venv,
run block) and insert the two new positionals **before** `server` (so `server` stays last). New
order after `loras` (slot 12): `preserve` (13), `preserve_mode` (14), `server` (15):
- slot 13 `preserve` — change threshold, default **`24`** (edit script → preservation ON by default);
- slot 14 `preserve_mode` — `mask | region`, default **`mask`**;
- slot 15 `server` — unchanged meaning, shifted from 13 → 15.

Wiring:
- Settings: `preserve="24"`, `preserve_mode="mask"`.
- Syntax guard: `-gt 15`, plus validate `preserve_mode` is `mask` or `region`. The slicing check
  stays keyed on slot 11 (`${11}`); the `server` parse moves to `${15}`, preserve/mode to `${13}`/
  `${14}`.
- Parse slots 13/14/15 like the other optionals.
- Server curl: add `--data-urlencode "preserve=$preserve"` and
  `--data-urlencode "preserve_mode=$preserve_mode"`.
- Local run: add `--preserve "$preserve" --preserve-mode "$preserve_mode"` to the `runner.cli` call.
- Usage text: list `[preserve = 24]` and `[preserve-mode = mask]` **before** `[server]`; document
  `mask` (add / recolor) vs `region` (remove / replace, ghost-free), same engine list as
  image-to-image. Keep every line ≤99 cols.

## 5. Registration & docs

- `bash/turbo/turbo.pri`: append `bash/turbo/image-edit.sh \` to OTHER_FILES
  ([turbo.pri:9](turboCLI/bash/turbo/turbo.pri#L9)).
- `bash/turbo/README.md`: add an `### image-edit.sh` section after image-to-image (usage block +
  the mask/region note + two examples: add-starfighter `mask`, remove-object `region`).
- `runner/server.py`: **no change** (generic urlencoded passthrough builds `params`).

## 6. Deploy mirror sync (after editing sources)

Per project memory, update the deployed copy:
- `cp turboCLI/runner/core.py turboCLI/runner/cli.py Sky-runtime-bin/gg.omega/turbo/runner/`
- `cp turboCLI/bash/turbo/image-to-image.sh turboCLI/bash/turbo/image-edit.sh \
     Sky-runtime-bin/gg.omega/turbo/bash/turbo/`  (the reverted i2i + the new script)

## Verification (manual, one specific image each)

Venv python: `/c/dev/workspace/msvc/Sky-runtime-bin/gg.omega/turbo/.venv/Scripts/python.exe`.
Output stays at `width x height`.

1. **Add (mask)** — `image-edit flux2-4b cuda "add a starfighter" base.png out.png 1280 720 -1 -1
   offloader none none 24 mask 8080`: `preserve[mask]` small coverage; pixels outside the new object
   byte-identical to `ImageOps.fit(input, out.size, LANCZOS)`.
2. **Remove (region)** — same on the knight image, `... none none 24 region 8080`, prompt "empty
   courtyard":
   `preserve[region]` ~50% coverage, knight gone, **no ghost**; outside the boxes byte-exact.
   Compare against the standalone `merge_region.py` result at the same working size.
3. **Replace dark-on-dark (region)** — TIE → TIE interceptor with `region`: no residual outline.
4. **i2i untouched** — `image-to-image` runs exactly as before (no `preserve[...]` log); its usage
   text no longer mentions preserve; text-to-image unaffected.
5. **99-col + ASCII** — new python and bash lines ≤99 cols; emits ASCII-only.

## Critical files
- `turboCLI/runner/core.py` — add `_blob_boxes` + `_region_mask`; add the mode switch in
  `_postprocess_i2i` (resolution/compositing unchanged).
- `turboCLI/runner/cli.py` — add `--preserve-mode`, `--preserve-grow`; `-1` auto defaults.
- `turboCLI/bash/turbo/image-to-image.sh` — revert preserve wiring (pristine).
- `turboCLI/bash/turbo/image-edit.sh` — NEW edit command (threshold + mode).
- `turboCLI/bash/turbo/turbo.pri`, `turboCLI/bash/turbo/README.md` — register + document.
- `turboCLI/runner/server.py` — verify only (no change).
- Deployed mirror under `Sky-runtime-bin/gg.omega/turbo/` — copy after editing.
