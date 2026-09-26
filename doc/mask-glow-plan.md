# A glow on the default mask (turboCLI)

## Context

`region` exists for one reason, and `_mask.py` says it itself: "a dark object over a matching
background falls below the threshold and survives as an outline; the box replaces the whole
footprint wholesale". So erasing or replacing something with the default mask leaves the mark of
its edges, and the cure so far has been to give up on the shape of the change and mask its bounding
box instead.

A **glow** gives the default mask the same power without the box: the mask reaches past the change
and fades out, so the leftover ring ends up inside the masked area and the seam blends into the
background.

This plan is turboCLI only, tested from bash. The turbopixel slider comes after, as its own change,
and only if the results here are good. If a glow covers removals as well as `region` does, dropping
`region` is a third change again.

## Measured first, because it decides the code

A glow is a dilation with a soft edge. The obvious `MaxFilter` cannot pay for the reach we need,
and a blur costs the same whatever its radius (installed venv, `L` images):

| | 1024x1024 | 2048x2048 |
| --- | --- | --- |
| `MaxFilter(5)` / `MaxFilter(33)` / `MaxFilter(65)` | 84 ms / 2149 ms / 7358 ms | 8447 ms at 33 |
| `GaussianBlur(3 .. 32)` | 17 ms, flat | 68 ms |

So the glow is a blur, taken back to solid with a gain. Measured on one straight edge, `0` being
the edge of the mask as it is today and negative being outside it:

| glow | gain 1 | gain 2 | gain 3 |
| --- | --- | --- | --- |
| 8 | touches -20, solid +20 | touches -20, **solid 0** | touches -20, solid -4 |
| 16 | touches -41, solid +41 | touches -41, **solid 0** | touches -41, solid -7 |
| 32 | touches -81, solid +81 | touches -81, **solid 0** | touches -81, solid -15 |

The gain restores a straight edge, and a straight edge is the one shape it restores exactly: a
narrow blob averages below half under its own blur and would come back grey. So a glow is not one
blur but **every radius down from the one asked for, kept by their maximum, the mask included** -
the small radii hold thin features solid, the wide ones supply the reach, and a maximum can only
add. Without the ladder, one radius of 32 held 58% of a helmet rim where 16 held 91%, so turning
the glow up made the result worse; with it, both hold 91%.

**`GLOW_GAIN = 6`**, which is what makes a glow a thickening rather than a halo. The gain says
where the blur is taken back to solid. At 2 the solid edge lands exactly on the mask's own outline
- a blurred edge reads half at the edge, so twice it is the edge again - and the glow only hangs a
skirt off a mask that never grows. At 6 the solid edge lands one radius out instead, measured the
same at 8, 16 and 32, so **glow=N grows the mask by N and fades over about half as much again**. A
calibrated constant where 2 is the derived one, and the comment on it says so.

The gain and the radius trade against each other, so the choice is which number reads better: gain
2 at glow 16 and gain 6 at glow 8 give the same answer (leak max 22 against 25, 48% of the frame
masked against 52%). "Eight pixels thicker" is the one a person can hold in their head, so the
slider opens on 8.

**One last blur over the grown shape**, at half the radius, because the gain saturates what it
lifts: without it the growth arrives solid and only its outermost pixels fade, which reads as a
thicker mask wearing a halo rather than one graded selection. It moves coverage out of the plateau
and into the ramp without changing the total. On the qwen erase at glow 16: solid 52% to 43%,
graded band 24% to 35%, leak max 11 to 9.

## 1. `runner/engine/_mask.py`

A constant beside the others, and a block at the end of `_diff_mask`, which already owns the
margin (`dilate`) and the seam (`feather`). `ImageChops` joins the imports and numpy stays out of
it, so the whole thing is PIL primitives:

```python
GROW,   FEATHER_REGION = 60, 18

# GLOW_GAIN takes a glow's blur back to solid, and where that lands is what a glow means: at 6 the
# grown edge falls one radius out, so glow=N carries the mask N further out.
GLOW_GAIN = 6
```

```python
    if glow > 0:
        grown  = mask
        radius = glow

        while radius >= 2:
            blur = mask.filter(ImageFilter.GaussianBlur(radius))

            grown  = ImageChops.lighter(grown, blur.point(lambda v: min(255, v * GLOW_GAIN)))
            radius //= 2

        mask = ImageChops.lighter(mask, grown.filter(ImageFilter.GaussianBlur(glow / 2)))
```

Three moves: grow it, soften the growth, never lose what was there. `grown` is seeded with the
mask itself so the closing blur straddles the boundary of the union rather than of the skirt
alone, and `glow / 2` is calibrated like the gain, wide enough to dissolve the step and narrow
enough to leave the growth.

`build_mask(reference, edit, mode, thr, glow=0)` hands it to the diff branch and leaves the region
branch alone - its box is already the extreme version of the same idea. Its docstring gains one
line saying the glow is how far the diff mask reaches past the change.

## 2. `runner/engine/mask.py`

```python
    glow = int(opts.get("glow", 0))
    ...
    mask = build_mask(ref, edit, "region" if sub == "region" else "mask", cut, glow)

    emit("mask[%s cut=%d glow=%d]: masked %.0f%%"
         % (sub, cut, glow, np.asarray(mask).mean() / 255 * 100))
```

Nothing reads that line but a human and `Saved:`, so widening it is free.

## 3. The option, in the four places turboCLI lists one

`glow=N` - how much thicker the mask gets, graded; `mask` engine, `mode=default` only, 0 = none.

- `runner/engine/mask.py` header comment, the `options:` paragraph
- `bash/turbo/image-to-mask.sh` usage block, plus one example line
- `bash/turbo/README.md` (~:208 and the examples under it)
- `implementation.md` - the `--options` sentence (~:208) and the `mask` row of the engine table
  (~:307)

## 4. Testing, from bash

`image-to-mask.sh` runs the **installed** tree (`getSky` then `cd "$bin"`), so the two edited files
are copied into `Sky-runtime-bin/gg.omega/turbo/runner/engine/` before each round - the repo stays
the source, the install is the runner.

```sh
export SKY_PATH_BIN=/c/dev/workspace/msvc/Sky-runtime-bin

cd /c/dev/workspace/msvc/turboCLI

for opt in mode=default mode=default,glow=8 mode=default,glow=16 mode=default,glow=32 mode=region
do
    sh bash/turbo/image-to-mask.sh mask cpu edit.png,original.png "mask_$opt.png" "$opt"

    sh bash/turbo/image-apply-mask.sh composite \
        "edit.png,mask_$opt.png,original.png" "out_$opt.png"
done
```

`composite` takes the edit where the mask says changed and the original everywhere else, which is
exactly the erase as turbopixel would land it - so what gets compared is the finished picture, not
the matte.

**The cases.** A real removal first: a pair where something was taken out over a background that
matches it, which is what `region` was written for. I will look for one in
`Sky-runtime-bin/skz/storage/turbopixel/output` and ask for a pair if nothing there fits. Beside it
the synthetic worst case, built so the ideal answer is known exactly: a dark shape on a background
of nearly the same value, gone in the edit.

**What decides it**, beyond looking at them: the edit *is* the ideal erase, so any place `out.png`
differs from `edit.png` is the original leaking through. Measure that difference in a band around
the object's silhouette - `mode=default` shows a ring there, and the glow is good when the band
falls to what `region` leaves. Report per option: max and mean difference in the band, the masked
percentage the engine emits, and the engine's wall time.

Then: if the numbers and the pictures are good, the turbopixel slider is the next change. If a glow
cannot clear a case `region` clears, say so rather than shipping a knob for it.

## What the test said

Four cases, all in `C:/Users/barnaud/Desktop/image` under `glow_`: a stormtrooper erased by
`comfy-qwen-image-edit-2511-lightning` (a local edit), the same erase by `comfy-flux2-4b` (which
re-rendered the whole scene), a hollow frame, and a faint ghost studded with rivets. Leak is
`|composite - edit|` where the edit is the ideal, so 0 means the erase landed whole.

| | default | glow 8 | glow 16 | glow 32 | region |
| --- | --- | --- | --- | --- | --- |
| qwen, max / over 16 | 96 / 4.3% | 20 / 0.1% | 9 / 0.0% | 5 / 0.0% | 0 / 0.0% |
| qwen, masked | 27.1% | 52.4% | 60.8% | 70.4% | 92.3% |
| hollow, max / over 16 | 60 / 3.3% | 12 / 0.0% | 12 / 0.0% | 18 / 0.0% | 0 / 0.0% |
| hollow, masked | 2.0% | 5.5% | 8.3% | 12.0% | 20.6% |
| sparse ghost left, over 8 | 91.6% | 61.4% | 48.5% | 29.6% | 0.0% |
| sparse, masked | 1.9% | 7.9% | 11.0% | 14.2% | 27.8% |

**The glow does not retire `region`.** The sparse rows are why: a change the threshold only catches
in scattered specks gives a glow nothing to grow from, so it makes discs around the specks and
leaves the body between them, where `region`'s box covers the lot. Lowering the cutoff under the
ghost does fix it (`cutoff=8,glow=16` leaves nothing at all, masking 23.7% against `region`'s
27.8%), but that is two knobs and the knowledge to reach for them. `region` stays.

So the three tools divide up as: `default` + `glow` when the change has an edge somewhere, which is
removals and replacements; `cutoff` when the change is faint; `region` when it is faint *and*
scattered, or when the footprint should simply be taken wholesale.
