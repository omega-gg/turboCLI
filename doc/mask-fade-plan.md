# A soft border on a cropped region mask (turboCLI)

## Context

A mask made for *part* of a picture is made from a crop. The engine is handed a rectangle and
knows nothing of where it came from, so it fills that rectangle to the edge. `region` especially:
`_region_mask` grows every changed blob's box by `GROW = 60` and fills it solid, which on a small
crop covers the whole of it. Padded back into the full picture, the mask drops from 255 to 0 in
one pixel at the crop's border, and the composite shows a rectangle, an abrupt change of colour
against whatever is underneath.

Inside the picture the same mask falls off gently, `GaussianBlur(FEATHER_REGION = 18)` at the end
of `_region_mask`, because the box has room to blur into. The crop's border is the one edge with
nowhere to fall off, and it is not an edge of the picture at all, just where we cut.

So: a way to tell the engine which borders of its input are a cut rather than a real edge, and to
let the mask fade out on those. turboCLI only. The caller that knows the box, turbopixel, sends it
later as its own change.

## The option

`fade=<sides>`, any of `ltrb`, `mode=region` only.

**Off unless asked for.** The option is empty by default, and an engine that is sent no `fade`
behaves exactly as it does today, byte for byte. Nothing infers a crop on its own: only a caller
that knows its input was cut out of something larger can say so, and until turbopixel sends it,
every run in the wild is unchanged.

**Per side, never all-or-nothing.** The value is the list of sides to fade, so each border is
decided on its own:

| where the box was cut from | fade | what happens |
| --- | --- | --- |
| the middle of the layer | `ltrb` | all four borders fade |
| against the left edge | `trb` | the left keeps its hard border, the other three fade |
| the top left corner | `rb` | only the two facing into the picture fade |
| a band across the whole width | `tb` | left and right are the file's own edges, so they stay |
| the whole file | *(nothing sent)* | not a crop at all, so no fade |

The rule for the caller is one line: a side fades when the picture continues past it.

The value names the borders of the input that are a crop. A caller that framed the middle of a
layer sends `fade=ltrb`; one that framed a corner, where two sides sit on the picture's own edge,
sends the two that do not. A side left out keeps its hard border, which is what an uncropped run
gives it. Empty, or absent, and nothing changes at all.

Sides rather than a bare flag because the option string is split on commas before `=` is read
(`runner/engine/_options.py`), so a value can hold no list of numbers. A subset of `ltrb` is one
word and says exactly what is needed.

## From bash

The option rides the same string as `mode` and `cutoff`, so nothing about the call changes:

```sh
# today, and still what happens when no fade is sent
image-to-mask mask cpu edit.png,plate.png mask.png mode=region

# the input was cut out of the middle of a bigger picture: every border is ours, so all four fade
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,fade=ltrb

# cut out of the top left corner: its left and top sit on the picture's own edge and keep their
# hard border, only the two facing into the picture fade
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,fade=rb

# with a cutoff as well, one string, any order
image-to-mask mask cpu crop.png,plate.png mask.png mode=region,cutoff=12,fade=ltrb
```

The whole round trip for a part of an image, which is what turbopixel does for a framed run:

```sh
# 1. cut the box out of the layer and out of the plate behind it        (the app renders these)
# 2. mask the crop, telling the engine which of its borders are the cut
sh bash/turbo/image-to-mask.sh mask cpu crop.png,crop_plate.png crop_mask.png mode=region,fade=ltrb

# 3. pad that mask back into the size of the layer, on black            (the app renders this)
# 4. apply it
sh bash/turbo/image-apply-mask.sh composite layer.png,mask.png,plate.png out.png
```

Steps 1 and 3 are the app's crop and pad, `pGetMaskImages` and `pGetMaskFile`; for the test below
they are two lines of PIL.

## The code

Three small edits in `runner/engine/_mask.py`, one in `runner/engine/mask.py`.

**1. The helper**, beside `build_mask`. `numpy` and `Image` are already imported at the top:

```python
def fade_edges(mask, sides, size):
    """Ramp an `L` mask to 0 over `size` pixels on the named borders, any of "ltrb". A mask made of
    a crop stops where the crop does, which is no edge of the picture: padded back into the whole
    of it, that border lands as a seam. The ramp never takes more than a quarter of a side, so a
    small crop keeps a mask."""
    w, h = mask.size

    n = max(1, min(size, w // 4, h // 4))

    x, y = np.ones(w, np.float32), np.ones(h, np.float32)

    if "l" in sides: x[:n]     = np.arange(n) / n
    if "r" in sides: x[w - n:] = np.arange(n - 1, -1, -1) / n
    if "t" in sides: y[:n]     = np.arange(n) / n
    if "b" in sides: y[h - n:] = np.arange(n - 1, -1, -1) / n

    # The nearest faded border is what a pixel follows, so a corner between two of them falls off
    # once rather than twice.
    out = np.asarray(mask, np.float32) * np.minimum.outer(y, x)

    return Image.fromarray(out.astype(np.uint8), "L")
```

**2. `_region_mask` takes it**, beside the `feather` it already owns, and spends it at the end:

```python
def _region_mask(generated, ref, thr, grow, feather, min_area, fade):
    ...
    if feather > 0:
        out = out.filter(ImageFilter.GaussianBlur(feather))

    # The border of a crop is ours, not the picture's, and a box that fills the crop would land
    # on it as a step. The inside falls off over `feather`, so the border does too.
    if fade:
        out = fade_edges(out, fade, feather)

    return out
```

**3. `build_mask(reference, edit, mode, thr, glow=0, fade="")`** hands it to the region branch and
leaves the diff branch alone, the same shape the `glow` argument already has.

**4. `runner/engine/mask.py`**: `fade = opts.get("fade", "")` beside the other options, passed
through to `build_mask`. The `emit` line stays as it is, since turbopixel already logs the option
string it sent.

## Docs

The four places turboCLI lists an option, as the `glow` change did:

- `runner/engine/mask.py` header comment, the `options:` paragraph
- `bash/turbo/image-to-mask.sh` usage block, plus one example
- `bash/turbo/README.md`, the options paragraph and the examples under it
- `implementation.md`, the `--options` sentence and the `mask` row of the engine table

## Verification

From bash, against the installed tree (`image-to-mask.sh` runs the install, so the two edited
files are copied into `Sky-runtime-bin/gg.omega/turbo/runner/engine/` first):

1. Take one of the erase pairs already in `C:/Users/barnaud/Desktop/image`, crop a box out of the
   middle of both, and mask the crop twice: `mode=region` and `mode=region,fade=ltrb`.
2. Pad both masks back into the full picture on black, the way turbopixel's `pGetMaskFile` does,
   apply each with `image-apply-mask composite`, and put the two side by side: the first shows the
   rectangle, the second blends into what is under it.
3. Read the ramp off the padded mask with PIL: the rows just inside a faded border climb 0 to 255
   over `n` pixels, and a border left out of `fade` stays hard. The arithmetic was checked earlier
   on a 64x40 crop with `sides = "trb"`: `n` clamped to 10 and the columns read
   `0 25 51 76 102 127 153 178 204 229 255`, with the left side untouched at 255.
4. A crop smaller than four times the fade keeps a mask rather than fading to nothing, which is
   what the clamp is for: mask a 48x48 crop with `fade=ltrb` and check the middle is still solid.
5. Nothing else moves: `mode=region` with no `fade`, `mode=default`, and `default` with a `glow`
   are byte for byte what they are today, compared against `git show HEAD:` on the same pair.

## Out of scope

- turbopixel sending `fade` from `pGetMaskOptions()`, naming the sides where the file continues
  past the box. That is where the box is known, and it is its own change.
- `mode=default`. Its mask hugs the change rather than filling the crop, so it rarely touches the
  border; when a `glow` carries it there, the same option can be opened up to it later.
- The plan is copied to `doc/mask-fade-plan.md`, beside `doc/mask-glow-plan.md`, and listed in
  `implementation.md`'s doc records.
