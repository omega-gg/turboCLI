# turboCLI: an edit's colors put back (`color`), an existing image adjusted (`image-adjust`)

## Context

An image-to-image run meant to keep the picture can still move its colors. Frame 140 of an H3
orbit (a deep blue attic), cleaned up:

| result | mean RGB | chroma | median dE, blurred, against the input |
| --- | --- | --- | --- |
| input frame | 14 / 53 / 76 | 17.2 | |
| MiniMax-H3-Image, 16 MP | 36 / 59 / 78 | 14.1 | 5.52 |
| same, histogram matched (prototype) | 14 / 53 / 75 | 17.3 | 2.22 |
| Qwen-Image 2.1 Turbo | 25 / 52 / 67 | 12.1 | 5.78 |
| same, histogram matched (prototype) | 14 / 53 / 75 | 17.3 | 1.98 |

H3 lifts the shadows and adds red (warm wood, white marble); Qwen loses blue and saturation. The
drift is about the same across the frame and depends on the tone, not the place, so one curve
per channel takes it back, and the detail stays the result's (checked on a 16 MP crop). What is
left (p95 about 6) is local: objects the model redrew.

`align=<n>` (doc/image-align-plan.md) undoes the zoom and shift, not the colors. Goals:
1. `color=<n>`, an option of image-to-image like `align`: off unless asked, never costs a render.
2. `image-adjust.sh`: either fix, or both, on an image that already exists, against a reference,
   with no model loaded.

## Design

As approved, with the color fit as built. Two changes came after, each in its section below: the
options (the first input always, the value a limit: `align`, `color`, `align=<percent>`,
`color=<levels>`) and the tests.

**The color step** (`runner/color.py`, new, beside `align.py`; PIL and numpy, torch free):
`run(image, reference, emit)` returns the image to save, with one line, the contract of
`align.run`.
1. Skip on a mode other than RGB, RGBA, L, LA (`color: skipped, <mode> images are not matched`).
   The reference is converted to the image's color mode (RGB or L).
2. Both go to a copy `SIDE` (256) px on the long side, the result's shape, `BOX` then a 1.5 px
   blur: a tone curve needs no detail, and the blur forgives what align left. The pixels are
   paired, result against reference; a cutout's transparent pixels are left out.
3. The fit (`_fit`): per channel, the median reference level in 16 bins of the result's levels
   (20 pixels at least per bin), made to rise, the shift at either end carried on past them.
   It runs on the pixels the edit left alone: first those within `TOLERANCE` (48 levels) of their
   reference, as if nothing had drifted; then, 3 times, those the last curve explains (within
   `max(8, 3 x the median residual)`). An edit's own colors (a shirt made red, a screen made
   blue) drop out, so they neither steer the curve nor get undone. Fewer than 20 pixels left:
   `color: skipped, nothing is left as it was`.
4. `change`: the mean absolute level move over the result's pixels. Over `MAX_CHANGE` (16
   levels) the change was asked for, "make it night": `..., over 16 levels, left as is`, as
   align leaves a zoom over 3%. Under `MIN_CHANGE` (1 level): `..., under 1 level, left as is`.
5. Applied with `image.point(table)`, the identity for alpha, in C at full size.
6. `color: R <+x> G <+y> B <+z>, matched`, the mean move per channel (one figure for L).

The cap lets turbopixel send `color=1` by default, as it does `align=1`. No strength: kept out
until someone needs it.

**Shared parse and dispatch** (`runner/engine/adjust.py`, new, a compute engine like
`mask_apply.py`; core imports it lazily, so all the adjust logic lives in one module):
- `STEPS = ("align", "color")`, the order they run in (geometry first).
- `parse(options, count)`: `[(name, n), ...]` for each step the options ask, in `STEPS` order;
  `ValueError("%s=%s names no input image (1 to %d)")` when a value is not `isdigit()` in
  `1..count`. The align message stays byte for byte the one core prints today.
- `step(name, image, reference, emit)`: `import_module(".." + name, __package__).run(...)`, so a
  color-only run never imports cv2.
- `ID = "adjust"`, `MODES = ("image-adjust",)`, `run(ctx, params, emit)`: images
  `base,reference[,...]`, options `align=<n>` and/or `color=<n>` with n naming a reference (1 for
  the first after the base). The base is opened (a `P` or `PA` file converted to RGBA or RGB),
  each step runs in turn, the image is returned; core saves it and prints `Saved:` as for any
  compute engine. No steps, or fewer than two images: `ValueError`. A failure here propagates (an
  `ERROR:` from cli/server): there is no render to protect.

**core.generate** (`runner/core.py`): the align block becomes the steps.
- Validation (same place, image-to-image only): `steps = parse(parse_options(...),
  len(image_list))`, a `ValueError` emitted as `ERROR: <message>`, return False, no model load.
- After the render: for each `(name, n)`, `image = step(name, image, prompt_images[n - 1],
  emit)` inside its own `try`; any failure is logged and `<name>: failed, left as is`, the next
  step still runs, the image is saved as usual.
- The NOTE names `engine/adjust.py`. Without `align` or `color`, nothing changes.

**image-to-image.sh**: the options help gains `color=<n> matches the result's colors to input
image n`, and an example with `align=1,color=1`. No argument changes, so turbopixel's copy of the
script keeps working.

**image-adjust.sh** (new, `bash/turbo/`, a copy of `image-apply-mask.sh`'s structure):
```
image-adjust <base image> <reference image> <output image> <fixes> [server]
fixes: align, color, or align,color
```
The images first and the output third, as image-upscale and image-to-mask take them; what to do
after, where the other scripts take their options; the server last.
`<fixes>` checked word by word (`case`), turned into `align=1,color=1`; sent as engine `adjust`,
mode `image-adjust`, `images=base,reference`, renderer `cpu`, offload `none`, to the CLI or to a
server's `/generate`, exactly as image-apply-mask does.

## Files

- New: `runner/color.py`, `runner/engine/adjust.py`, `bash/turbo/image-adjust.sh`,
  `doc/image-adjust-plan.md` (this plan, copied there first).
- Changed: `runner/core.py`, `bash/turbo/image-to-image.sh`, `bash/turbo/install.sh` (usage:
  `adjust (no download)`), `bash/turbo/turbo.pri`, `turboCLI.pro` (the new files).
- Docs: `MANUAL.md` (image to image: `color=1` beside `align=1`; a short "Adjusting an existing
  image" section with `image-adjust.sh`), `bash/turbo/README.md` (the option and the script),
  `implementation.md` (the step in "One generation", the wire line `color: ...  i2i with color
  only, before Saved:`, the engine table row, the doc index), `engine.md` (an `adjust` row like
  `mask-apply`'s).

Conventions: 99 columns, the license header, `# NOTE:` comments, the runner's blank lines, no
dash joins, American spelling. No commit until asked.

## Verification

1. **Bench**: `runner.color.run` on the two frame 140 results against the input frame: the
   table's figures or better; about 0.1 s on the 16 MP result. On the align plan's bench, every
   local edit matched and every night left as is.
2. **Synthetic** (turbo venv, `-B`): a known curve on a picture recovered within a level; RGBA in,
   RGBA out with alpha untouched; an image against itself: `under 1 level, left as is`; `P` skips
   in `color.run`, converts in `adjust.run`.
3. **image-to-image**: `align=1,color=1` on a Qwen 2.1 Turbo edit, CLI and server: both lines
   before `Saved:`; `color=0`, `x`, `3` with one input stop at `ERROR:` before any model load;
   `color.run` forced to raise: `color: failed, left as is`, `Saved:`, exit 0; no options: the
   same bytes as before the change.
4. **image-adjust.sh**: `color`, `align`, `align,color` on the H3 result (16 MP base, 1344x768
   reference), locally and through a server; a bad `<fixes>` prints the usage; the `color` output
   matches step 1.
5. `python -m pyflakes runner test`; `awk 'length > 99'` on every touched file.

## As built

**The fit changed: pixel against pixel, not histogram against histogram.** The first build
matched each channel's histogram to the reference's. On the clean-ups it did well, but the
bench of the align plan (the Untitled frames: 32 local edits, 24 "make it night", Qwen 2.1
Turbo and FLUX.2 4B and 9B) showed what it cannot tell apart: the local edits there change
colors on purpose (a shirt made red, the screens made blue sky, a tablet made metal), which
moves a histogram by 11 to 28 levels, as much as a drift, and a histogram match would have
pulled the red out of the whole frame to undo the shirt. Fitted on the pixels the edit left
alone, the curve sees only the drift. The first pass needs `TOLERANCE`: started on every pixel,
the blue screens, whose red sits near 8 where the grey screens' sat near 104, were most of the
darkest red bin, and the rising curve then lifted every red level to 104.

With it, the change separates (in levels):

| results | n | change |
| --- | --- | --- |
| local edits, their own colors left out | 32 | 1.6 to 11.7 |
| clean-ups of frame 140 (H3, Qwen) | 2 | 11.1, 7.3 |
| make it night | 24 | 20.0 to 42.1 |

`MAX_CHANGE` = 16 sits in the gap: every local edit and clean-up matched, every night left as
is. By eye: the red shirt stays red and the rest loses FLUX.2's warm cast; the blue screens stay
blue; the cool cast of a FLUX.2 4B canvas edit goes. A night made by hand (every level x0.45)
is left as is. A gamma of 0.85 comes back within 8 levels, 0.6 on average (16 bins).

Frame 140 again, through `runner.color.run`:

| result | line | median dE, blurred | mean RGB | chroma | time |
| --- | --- | --- | --- | --- | --- |
| MiniMax-H3-Image, 16 MP | `R -22.9 G -6.1 B -2.9` | 5.52 to 2.13 | 15 / 53 / 75 | 17.1 | 0.14 s |
| Qwen-Image 2.1 Turbo | `R -11.5 G +1.1 B +9.3` | 5.78 to 1.90 | 14 / 53 / 75 | 17.1 | 0.05 s |

`core.generate` was run with a stand-in pipeline returning the Qwen result: no options saved the
same bytes as before; `color=1`, `align=1,color=1` and `color=1,align=1` gave their lines in
`STEPS` order before `Saved:`; `color.run` forced to raise gave `color: failed, left as is` after
the align line, `Saved:` and True; `color=0`, `x`, `3` stopped at `ERROR:` before any model load.
`image-adjust.sh` ran `color`, `align` and `align,color` locally and through a server (port 8200),
its color output byte for byte the CLI's; a bad `<fixes>` prints the usage. The real render end to
end was left out: the installed FLUX.2 klein 4B needs more memory than the laptop had free.

`align` leaves the H3 result as it is (`the input has another shape`): MiniMax-H3-Image renders
16 MP at 5472x3072 (1.781) from a 1344x768 frame (1.75), past align's 1% ratio check. `color`
does not look at the shape.

## Revised: the first input always, the value a limit

The reference was named by number (`align=1`, `color=1`). It is always the first input now, the
picture being edited: the others are style or subject references, never one to align on or take
colors from, and `image-adjust` has one reference anyway. The option's value became the one limit
a caller might move: `align=<percent>`, the zoom past which it was asked for (3 by default, was
`MAX_ZOOM = 0.03`, now in percent), and `color=<levels>`, the mean move past which it was (16).
The key alone asks for the default, so the defaults live in the runner alone: `parse_options`
reads an item without `=` as a key with an empty value (it used to drop it; no option is written
without one). `engine/adjust.parse(options)` returns `[(step, limit or None)]`, a limit that is not
a positive number an `ERROR:`; `step()` passes the limit to `run(image, reference, emit, limit)`,
which takes its default when it is None. The other limits of align (0.75 px, a fifth of gain) stay
constants: they guard the fit, nobody tunes them.

`image-adjust.sh` passes its `<fixes>` as they are (`align`, `color`, `align=<percent>`,
`color=<levels>`, comma separated), each word checked, the number by the runner. turbopixel sent
`align=1`, which would now read as a 1% limit and leave most of Qwen's drift (up to 2.8%) in
place: it sends `align` (and `color`) with this change.

With it, from the review: `_pixels` makes the copy small before converting it (converting copied
the whole 16 MP image: 33 to 18 ms, the step 0.14 to 0.10 s), the residual's numbers are named
(`FLOOR`, `SPREAD`), the fewest pixels to fit on has its own name (`MIN_KEPT`, 100), and core
imports `step` once. The bench gives the same decisions: 32 local edits and 2 clean-ups matched,
24 nights left as is.

## Tests

`test/run.sh` runs the `adjust` engine, once registered (`install.sh adjust cpu`), in two cases
against `test/old/knight-small.png` (368x512): `image-adjust` on `knight-drift.png` (the knight
zoomed 0.6% and moved 3 px right and 2 up, then warmer and lifted, the recipe in
`test/old/README.md`) and
`image-adjust-night` on `knight-night.png` (the same move, then a night), both with
`align,color`. Their image is checked byte for byte against the machine's reference, their time
shown and not judged, as for the mask engines. Each also has to print what each fix did, the
drift `aligned` and `matched`, the night `aligned` and `left as is`: a fix that stops working
still saves an image, which `--record` would keep, and the missing line fails the case instead.
Checked on the A1000: both `NEW`, stable over two repeats, recorded, then `SAME`; with color's
limit forced to 1 level the drift case fails and the night still passes, with align's forced to
0.1% both fail. The reference was put back as it was: each machine records them with `--record`
once this is deployed.

From the review after: the fixtures went to half size (made from `knight-small.png`, 413 KB for
both against 1.3 MB, align still has its 4 tiles a side); `run.sh` sets each adjust case's input
and lines in one place and runs one `image-adjust.sh` call, the night case listed in the mode
loop (`ENGINES:${mode%-night}`) rather than inside an awk; `adjust._limit` reads a limit (nan
for what is not a number, so one `not limit > 0` refuses both); `color.run` checks that enough
pixels are left before every fit, the last one too. The bench decisions and the suite's
(new, recorded, same, each fix forced off) came out as before.
