# A composite goes in as a reference trimmed to what it shows

## Context

Dragging a layer that draws its composite onto the reference list copies the composite into
`output` and references the copy (`insertReference`, `run/turbopixel.sky`). A composite is the
layer cut by its mask: mostly transparent, the subject somewhere inside. Handed to an image
generator as is, that transparent margin is information it does not need.

The copy is cut down to the part that shows plus a 32 px transparent margin, and named like a
generated image, `output/image_<date>.png`, whether the composite is one of ours (default
composite folder) or a file the layer points at from elsewhere. The trimming lives in turboCLI, so
anything can call it from the command line, and turbopixel only calls it.

## turboCLI: a `trim` mode for image-apply-mask

`image-apply-mask` already post-processes images through the torch-free `mask-apply` engine,
and forwards its mode as `options=mode=<mode>`. `trim` joins `composite` and `putalpha`: no new
script, no new engine, no registration. Two options:

- `pad=N`: the most transparency it keeps around what shows, on each side (32 by default)
- `speck=N`: separate blobs of fewer than N pixels are left out when the box is measured (0 by
  default, off)

```
image-apply-mask trim cutout.png trimmed.png
image-apply-mask trim cutout.png trimmed.png pad=16,speck=1024
```

### Why specks are off by default

A mask can scatter specks all over its image: grains of a textured ground a matte engine half
kept, faint and opaque alike, up to the edges. The box of everything that shows then reaches the
edges too, and the trim cuts nothing. Leaving small separate blobs out fixes that, but a small
separate blob can also be what matters: sparks, snow, particles around a character. So the exact
box is the default, and a caller that knows its masks are noisy asks for `speck`.

`speck=1024` means: a spot of visible pixels that touches nothing else and covers fewer than 1024
pixels (a 32x32 square) does not decide where the crop goes. It is not erased, a spot inside the
box the rest decides stays. A thin part joined to the subject, a saber or an antenna, belongs to
the subject's blob and always counts.

### `runner/engine/_apply.py`

```python
# The most transparency a trim keeps around what shows, on each side, when not told otherwise.
TRIM_PAD = 32

# The alpha under which a pixel counts as not showing: a feathered mask leaves a faint haze far
# past its subject.
TRIM_ALPHA = 8

# A trim that leaves specks out looks for blobs on a mask this many times smaller, which keeps
# it fast.
TRIM_SCALE = 4


def trim(image, pad=TRIM_PAD, speck=0):
    image = image.convert("RGBA")

    shown = image.getchannel("A").point(lambda a: 255 if a > TRIM_ALPHA else 0)

    box = _speck_box(shown, speck) if speck > 0 else shown.getbbox()

    if box is None:
        return image

    left, top, right, bottom = box

    return image.crop((max(0, left - pad), max(0, top - pad),
                       min(image.width, right + pad), min(image.height, bottom + pad)))


def _speck_box(shown, speck):
    from ._mask import _blob_boxes                         # numpy, only when specks are asked

    # A cell shows when any pixel of it does, so a thin part stays joined to its subject.
    small = shown.reduce(TRIM_SCALE).point(lambda a: 255 if a else 0)

    boxes = _blob_boxes(small, max(1, speck // (TRIM_SCALE * TRIM_SCALE)))
    ...  # the union of the boxes, scaled back up
```

`_blob_boxes` is the flood fill `_mask.py` already had for the region mask. On the reduced mask
it stays fast: 15 to 110 ms on a 1120x896 composite. An image without alpha converts to fully
opaque, so its box is the whole image, and an image with nothing showing comes back whole.

### `runner/engine/mask_apply.py`

`trim` takes one image and no mask, so it is handled before the two-image check, and reads its
options from the ones the engine already parses:

```python
    if mode == "trim":
        out = trim(Image.open(imgs[0]), int(options.get("pad", TRIM_PAD)),
                   int(options.get("speck", 0)))
```

### `bash/turbo/image-apply-mask.sh`

It gains `[options]` before `[server]`, where `image-to-mask` has it, and forwards it after the
mode (`options=mode=<mode>,<options>`, both on the server and the local path). The syntax check
accepts `trim` and up to five arguments. The server moves from the fourth argument to the fifth:
turbopixel never passes one, and a caller that does passes an empty options first.

`bash/turbo/README.md` mirrors the usage block, and `implementation.md` says `trim`, `pad` and
`speck` in the engine paragraph and table.

## turbopixel: the reference takes the trimmed copy

**`run/turbopixel.sky`**, `insertReference`: a composite starts the trim, and the reference goes
in at once, pointing at the file the trim is writing.

```qml
        if (controllerFile.folderPath(source) == core.getPathStorage("turbopixel/composite")
            ||
            pIsShared(-1, "composite", source))
        {
            var path = pGetPathOutput("image");

            var id = gui.async(bash_turbo_image_apply_mask, "trim", source, path,
                               "speck=1024").id;

            pTrims[id] = { "path": path, "source": source };

            referencesLoading = referencesLoading.concat(path);

            source = path;
        }
```

- `pIsShared(-1, ...)` answers whether any layer draws that file, so a composite from elsewhere
  is trimmed as well as one of ours.
- turbopixel asks for `speck=1024`: its composites come from matte engines that scatter specks.
- `referencesLoading` holds the paths a trim is still writing. `PanelReferences` shows those rows
  with an empty thumbnail and `"Loading..."`, and loads the thumbnail once the path leaves it.
- `onBashFinished` takes the answer out of `pTrims`, clears the loading path and shows the popup.
  A failed trim copies the untrimmed composite to the path, as before the trim existed, so the
  reference never points at nothing.

A plain layer, one without a composite, is referenced where it is.

## Verification

Measured on a 1120x896 composite whose mask scattered ground specks up to the edges:

| call | result |
| --- | --- |
| `trim` | 1120x875: exact, only the empty top cut down to 32 px |
| `trim speck=1024` | 672x616: the figure and the ground under it, the specks left out |
| `trim pad=0` | 1120x843 |
| a 2 px antenna on a body, specks around, `speck=1024` | the antenna kept, the specks left out |
| an RGB image, a fully transparent one | whole |
| `putalpha` | unchanged |

In turbopixel, a masked layer dragged onto the references lands at once with `"Loading..."`, then
shows the trimmed `output/image_<date>.png`. A plain layer is referenced in place.
