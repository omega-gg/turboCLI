# A soft border on a cropped region mask (turboCLI)

## Context

A mask made for *part* of a picture is made from a crop. The engine is handed a rectangle and knows
nothing of where it came from, so it fills that rectangle to the edge. `region` especially:
`_region_mask` grows every changed blob's box by `GROW = 60` and fills it solid, which on a small
crop covers the whole of it. Padded back into the full picture, the mask drops from 255 to 0 in one
pixel at the crop's border, and the composite shows a rectangle, an abrupt change of colour against
whatever is underneath.

Inside the picture the same mask falls off gently, `GaussianBlur(FEATHER_REGION = 18)` at the end
of `_region_mask`, because the box has room to blur into. The crop's border is the one edge with
nowhere to fall off, and it is not an edge of the picture at all, just where we cut.

So: a way to tell the engine which borders of its input are a cut rather than a real edge, and how
far each should fall off. turboCLI only. The caller that knows the box, turbopixel, sends it.

## The option

`fade=<left>:<top>:<right>:<bottom>`, in pixels, `mode=region` only.

**Off unless asked for.** Absent, or empty, and the engine behaves exactly as it does today, byte
for byte. Nothing infers a crop on its own: only a caller that knows its input was cut out of
something larger can say so.

**Per side, and the caller owns the distance.** Each border takes its own number, `0` leaving it
hard:

| where the box was cut from | fade | what happens |
| --- | --- | --- |
| the middle of the layer | `24:24:24:24` | all four borders fade |
| against the left edge | `0:24:24:24` | the left keeps its hard border, the other three fade |
| the top left corner | `0:0:24:24` | only the two facing into the picture fade |
| a band across the width | `0:24:0:24` | left and right are the file's own edges, so they stay |
| the whole file | *(nothing sent)* | not a crop at all, so no fade |

The rule for the caller is one line: a side fades when the picture continues past it.

**Colons, because commas cannot survive.** `runner/engine/_options.py` splits the option string on
"," before it reads "=", so a value can hold no comma. Four numbers separated by colons pass
through whole, and the order is the one CSS uses for a border, clockwise from the left.

## From bash

The option rides the same string as `mode` and `cutoff`, so nothing about the call changes:

```sh
# today, and still what happens when no fade is sent
image-to-mask mask cpu edit.png,plate.png mask.png mode=region

# cut out of the middle of a bigger picture: every border is ours, so all four fade over 24 px
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,fade=24:24:24:24

# cut out of the top left corner: its left and top sit on the picture's own edge and keep their
# hard border, only the two facing into the picture fade
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,fade=0:0:24:24

# a wider blend on the right alone, with a cutoff as well, one string, any order
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,cutoff=12,fade=0:0:60:0
```

The whole round trip for a part of an image, which is what turbopixel does for a framed run:

```sh
# 1. cut the box out of the layer and out of the plate behind it        (the app renders these)
# 2. mask the crop, telling the engine which of its borders are the cut
sh bash/turbo/image-to-mask.sh mask cpu crop.png,crop_plate.png crop_mask.png \
    mode=region,fade=24:24:24:24

# 3. pad that mask back into the size of the layer, on black            (the app renders this)
# 4. apply it
sh bash/turbo/image-apply-mask.sh composite layer.png,mask.png,plate.png out.png
```

Steps 1 and 3 are the app's crop and pad, `pGetMaskImages` and `pGetMaskFile`.

## The code

`runner/engine/_mask.py` gains one helper, `_fade_edges(mask, sides)`, which multiplies the mask by
a ramp built per border. The minimum of the two axes is what a pixel follows, so a corner between
two faded borders falls off once rather than twice, and no ramp takes more than a quarter of its
own side, so a small crop keeps a mask:

```python
    left, right  = min(sides[0], w // 4), min(sides[2], w // 4)
    top,  bottom = min(sides[1], h // 4), min(sides[3], h // 4)

    def ramp(n):
        return np.arange(n, dtype=np.float32) / n

    x, y = np.ones(w, np.float32), np.ones(h, np.float32)

    if left   > 0: x[:left]       = ramp(left)
    if right  > 0: x[w - right:]  = ramp(right)[::-1]
    if top    > 0: y[:top]        = ramp(top)
    if bottom > 0: y[h - bottom:] = ramp(bottom)[::-1]

    out = np.asarray(mask, np.float32) * np.minimum.outer(y, x)
```

`_region_mask` spends it after its own feather, `build_mask(..., fade=())` hands it through the
region branch the way `glow` goes through the diff one, and `runner/engine/mask.py` parses the
value and refuses anything that is not four counts:

```python
    fade = opts.get("fade", "")
    fade = [int(n) for n in fade.split(":")] if fade else []
    ...
    if fade and (len(fade) != 4 or min(fade) < 0):
        raise ValueError("fade takes four pixel counts, left:top:right:bottom")
```

## How wide, and who decides

The engine held the distance at first, and the walk to a good one is worth recording: `18`, the
blur the boxes end on, still read as a cut; `GROW` (60) and `feather * 4` (72) ate a quarter of a
small crop, measuring 112 px of ramp on a 479 px box. `24` was the one that looked right.

That number now lives in the caller, `maskFade` in turbopixel, since only the caller knows what the
crop is for. The engine keeps no default: what it is not told, it does not do.

## Verification

1. Through bash, on a crop of an erase pair: `mode=region` gives no ramp at all, `fade=0:0:24:0`
   gives 24 px on the right and nothing elsewhere, `fade=24:24:24:24` gives 24 on each, and
   `fade=0:0:60:0` gives 59. Measured off the written mask with PIL, reading a row inward from the
   border.
2. Refusals: `fade=0:0:24` reports *fade takes four pixel counts, left:top:right:bottom*, and
   `fade=` behaves as no option at all rather than crashing on `int("")`.
3. A crop smaller than four times a fade keeps a mask rather than fading to nothing, which is what
   the clamp is for.
4. Nothing else moves: `mode=region` with no `fade`, `mode=default`, and `default` with a `glow`
   are byte for byte what they were, compared against `git show HEAD:` on the same pair.
