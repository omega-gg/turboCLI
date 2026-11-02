# turboCLI: align an edit on its reference (`align=<n>`)

## Context

An edit of part of a picture is pasted back where it came from, so it must come back in place.
Qwen-Image 2.1 Turbo zooms and shifts its result a little, differently for each seed (up to 0.9%
on an axis, up to 9 px at the corners of a 1000 px frame), which shows as a fringe once the edit
lands on the canvas. A better reference cannot fix it: the model adds it. FLUX.2 lands within
about half a pixel. Validated on 56 real edits of the Untitled project (local and day to night,
Qwen 2.1 Turbo, FLUX.2 4B and 9B), judged by SIFT matches against the canvas, canvas px:

| engine, edit | aligned | median drift, as is | aligned |
| --- | --- | --- | --- |
| Qwen, local (16) | 8 | 3.33 | 1.73 |
| Qwen, day to night (8) | 8 | 7.01 | 2.09 |
| FLUX.2 4B / 9B, local (16) | 0 | 0.71 / 0.72 | unchanged |
| FLUX.2, day to night (16) | 1 | redrawn, not judged | |

Every aligned Qwen result improved (worst 25.2 to 4.7). Cost: a 0.3 to 0.4 px blur equivalent,
0.2 to 0.8 s of CPU. Goal: an optional step after an image-to-image run, off unless a caller asks.
turbopixel turns it on later, not in this change.

## Design

**The option**: `align=<n>`, align the result on input image `n` (from 1). Rides `options`, which
`cli.py` (`--options`) and `server.py` (every POST field lands in `params`) already carry, so
neither front-end changes. image-to-image only; text-to-image ignores it.

**core.generate** (`runner/core.py`):
- Validation after the compute-engine short-circuit and before the offload checks, gated on
  `mode == "image-to-image"`: the `image_list` parse moves up from the image-to-image block and
  serves both. `options = parse_options(params.get("options", ""))` (lazy import from
  `runner/engine/_options.py`, as the mask engines do; an older client sends no `options`). A
  value that is not `isdigit()` in `1..len(image_list)`: `ERROR: align=<v> names no input image
  (1 to <N>)`, return False, no model load.
- After `image = result.images[0]` and before the save: `image = align.run(image,
  prompt_images[n - 1], emit)` inside one `try` with the lazy `from . import align`. Any failure
  (cv2 missing, MemoryError, anything): `log(traceback.format_exc())`, emit
  `align: failed, left as is`, keep the image and save as usual. A finished render is never lost.
- The docstring's params list gains `options`.

**The step** (`runner/align.py`, drafted, revised as below; torch free: cv2, numpy, PIL):
`run(image, reference, emit)` returns the image to save, with one informational line before
`Saved:` (the wire contract keeps `Saved:` as the only success sentinel).
1. Skips, each one line: a mode other than RGB, RGBA, L, LA; an input of another shape (ratios
   over 1% apart); a frame under 4 tiles on a side once at most 1 MP (`min(size) * scale <
   4 * TILE`), where the zoom on the short axis has no spread to be fitted.
2. The input is resized to the output; both go gray on a copy of at most 1 MP (`AREA`).
3. Edge maps (Sobel over the local mean, so relighting does not matter), computed once per
   image; 64 px tiles every 32 px matched by `cv2.phaseCorrelate` (NOTEs on the flat and
   response floors); zoom and shift per axis by least squares, the tiles that disagree dropped
   over 3 passes, the fit taken after the last; the whole fit repeated 3 times on the result
   warped back (`ROUNDS`), each adding what is left. Under 16 tiles: skipped.
4. Corner move under `MIN_SHIFT` (0.75 output px): `..., under 0.75 px, left as is`.
5. The warp: Lanczos, the edge repeated (`BORDER_REPLICATE`), alpha with the rest. Nothing is
   taken from the input: on a day to night edit its band would put daylight along the night
   result's edges (seen on the bench's strongest zoom).
6. Kept only if the median tile shift against the input drops by at least `MIN_GAIN` (a fifth),
   else `..., does not explain the change, left as is`; then
   `align: zoom <x>% x <y>%, shift <x> <y> px, aligned`.

**image-to-image.sh**: `[options]` before `[server]`, the order of image-to-mask and
image-apply-mask. Arg count `-gt 14`; always sent, as `options=$options` to `/generate` and
`--options "$options"` to the CLI. turbopixel ships its own copy of the script, so its call and
the script change together; an older call would have its server read as options and run locally.

**build.sh**: `opencv_version="5.0.0.93"`, `opencv-python` and `opencv-python-headless` both
pinned to it in the main install (the two wheels share `cv2/`; transparent-background and
albumentations pull them), with a NOTE.

## Files

- `runner/align.py` (drafted): the revisions above.
- `runner/core.py`, `bash/turbo/image-to-image.sh`, `bash/turbo/build.sh`.
- Docs: `doc/image-align-plan.md` replaced by this plan once approved (no local paths);
  `implementation.md` (params list with `options`, the step in "One generation", a wire contract
  line `align: ...  i2i with align only, before Saved:`, the doc index); `MANUAL.md` (image to
  image: the option); `bash/turbo/README.md` (the argument).

Conventions: 99 columns, the license header, `# NOTE:` comments, blank lines as the runner does,
no dash joins, American spelling. No commit until asked.

## Verification

1. **Synthetic** (turbo venv, `-B`, from the checkout): a known zoom and shift recovered within
   about 0.05% and 0.4 px; RGBA in, RGBA out; the narrow frame, tiny, flat and `P` images skip.
2. **Bench**: `runner.align.run` on the 56 results against their references gives the
   prototype's decisions, figures within rounding.
3. **Failure path**: `align.run` forced to raise: `align: failed, left as is`, `Saved:`, exit 0.
4. **Validation**: `align=0`, `x`, `1.5`, `3` with one input: `ERROR:` before any model load; no
   `options` field at all: a normal run.
5. **End to end** (runtime turbo dir backed up, `runner/` and the script deployed, restored
   after; a server on port 8200; no generation in turbopixel meanwhile): a Qwen edit (desk x0.8,
   seed 8) with `align=1` through the server and through the CLI; a call with empty options; a
   FLUX.2 run with `align=1` (left as is expected). Without `align`,
   no `align:` line (bytes compared only if two baseline runs match).
6. `awk 'length > 99'` on every touched file.

## As built

OpenCV 5's `phaseCorrelate` applies its window to its inputs in place. The tiles overlap, so each
one damaged its neighbors, and every later pass on the same edge map matched fewer tiles (610,
then 85, 43, 7). The prototype the table above comes from had it too. The tiles now go as copies,
and a known zoom and shift come back almost exactly (+0.80% x +0.50%, -3.97 +2.99 px for +0.8% x
+0.5%, -4 +3 px, at 1 and 3 MP).

The 56 results again, through `runner.align.run`:

| engine, edit | aligned | median drift, as is | aligned |
| --- | --- | --- | --- |
| Qwen, local (16) | 9 | 3.33 | 0.92 |
| Qwen, day to night (8) | 8 | 7.01 | 1.25 |
| FLUX.2 4B, local (8) | 0 | 0.72 | unchanged |
| FLUX.2 9B, local (8) | 1 | 0.72 | 0.73 |
| FLUX.2, day to night (16) | 3 | redrawn, not judged | |

No Qwen result got worse; the frame Qwen recomposed (11.6 px off) is now aligned, to 0.9. One
FLUX.2 9B edit just past the 0.75 px floor went from 0.73 to 0.80. 0.12 to 0.39 s per result.

End to end on a Qwen edit (desk x0.8, seed 8): through the server the line read the bench's
figures and the file matched `runner.align.run` on the raw result; without `align`, the server
and the CLI saved the same bytes as before the change; a forced failure gave `align: failed, left
as is`, `Saved:` and exit 0; `align=0`, `1.5`, `x` and `3` stopped at `ERROR:` in about 5 s, no
model loaded. The FLUX.2 run end to end was left to the bench above.

A zoom over 3% (`MAX_ZOOM`) is left as is: past it the zoom was asked for, "zoom out" or an engine
that moves the camera, and Qwen's own drift stayed under it on the bench (2.8% at most). Zooms of
3, 5 and 10% made on purpose are measured to 0.03% and left alone.

turbopixel sends `align=1` from its "Correct drift" box, under the seed of Image to Image, checked
by default for every engine and kept by the project.
