# Image-to-image keeps each image's ratio, sized per engine (turboCLI)

## Context

Image-to-image prepares every input image the same way, in `runner/core.py:626-633`:

```python
scale = min(img.width / width, img.height / height, 1.0)
image_width = max(1, round(width * scale))
image_height = max(1, round(height * scale))
img = ImageOps.fit(img, (image_width, image_height), Image.Resampling.LANCZOS)
```

The target is always the output's shape, and `ImageOps.fit` center crops to it. turbopixel renders
the canvas at the output size, so the first image loses nothing, but every reference after it is
cut to the canvas's shape: a portrait reference on a landscape canvas loses its top and bottom.
`implementation.md` even calls it a fit "preserving aspect", which it is not.

The crop does nothing for the pipelines, which size every image at its own ratio already:

| engines | pipeline | each image |
| --- | --- | --- |
| `qwen-image-edit-2511` (+ lightning, angles), `comfy-qwen-image-edit-2511` (+ lightning) | `QwenImageEditPlusPipeline` | resized to ~1 MP at its ratio, up or down, plus a 384² area copy for the VL encoder |
| `flux2-4b`, `comfy-flux2-4b` | `Flux2KleinPipeline` | kept up to 1 MP, larger shrunk to 1 MP, sides floored to 16 |

So how far we shrink first only decides two things, and they differ per engine:

- **Qwen** costs the same whatever we send (always ~1 MP). Shrinking first only loses detail:
  on a 512² frame a reference cut to 0.26 MP is scaled back up to 1 MP.
- **Flux2** pays per reference pixel, its tokens growing with the area it is given. Capping at the
  output's area keeps a run as fast as today.

The behaviour belongs to the engine, with a default for one that says nothing.

## The declaration

One optional contract symbol, read like `INFERENCE` is (`getattr` with a fallback):

| engine declares | each input image |
| --- | --- |
| nothing (default) | shrunk to the output's area when larger |
| `IMAGE_AREA = None` | sent as it is, the pipeline sizes it |
| `IMAGE_AREA = <pixels>` | shrunk to that area when larger |

The ratio always stays, nothing is ever cropped or enlarged. The default is today's cost with the
crop gone, so an engine that says nothing behaves as before, minus the cut.

## Changes

**`runner/core.py`**, the image loop (lines ~617-635). `ImageOps` leaves the import:

```python
        from PIL import Image

        image_list = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

        # NOTE: The engine caps the pixels an image keeps before its pipeline, None for no cap,
        #       and the output's area when it says nothing. The ratio always stays.
        area = getattr(mod, "IMAGE_AREA", width * height)

        prompt_images = []

        for ip in image_list:
            emit("loading input: %s" % ip)

            img = Image.open(ip).convert("RGB")

            if area and img.width * img.height > area:
                scale = (area / (img.width * img.height)) ** 0.5

                img = img.resize((max(1, round(img.width * scale)),
                                  max(1, round(img.height * scale))), Image.Resampling.LANCZOS)

            prompt_images.append(img)
```

**`runner/engine/qwen_image_edit_2511.py`** and **`runner/engine/comfy_qwen_image_edit_2511.py`**,
after `INFERENCE`:

```python
# The pipeline sizes every image to ~1 MP at its own ratio, so they go as they are.
IMAGE_AREA = None
```

**`runner/engine/_inherit.py`**: `"IMAGE_AREA"` joins `_INHERITED`, after `"INFERENCE"`, so the
lightning and angles presets (`BASE = base.ID`) get it.

Flux2 (`flux2-4b`, `comfy-flux2-4b`) declares nothing and takes the default. If the grids show its
references too soft on a small frame, `IMAGE_AREA = None` there is one line, and the timings say
what it costs.

**`implementation.md`**:
- an `IMAGE_AREA` row in the contract table, after `INFERENCE`: the largest area an input image
  keeps before the pipeline, `None` for as it is, the output's area by default; the ratio stays
- the image-to-image step (line ~362) says what it now does: each image shrunk to the engine's
  `IMAGE_AREA`, keeping its ratio, never cropped
- `doc/image-area-plan.md` in the doc records list

**`doc/image-area-plan.md`**: this plan.

## Why nothing else moves

- turbopixel's canvas is rendered at the output size, so under the default it is never shrunk
  (its area is the output's) and under `None` it goes as it is: it lines up with the output
  exactly as today, which the mask and composite steps rely on.
- For an image with the output's shape the default gives `min(w, W) x min(h, H)`, the same size
  the old fit gave. Only images of another shape change, and they now keep all of their content.
- The number, order and roles of the images are the same, so multi-reference runs keep working.
- Nothing after the loop reads the images' size: `prompt_images` goes straight to the pipeline.
- Text-to-image and the compute engines (mask, matte, apply) never reach this loop.

## Verification

All at the turboCLI level, through its own scripts, before turbopixel is involved. Every output
and comparison goes to a local test folder, prefixed `area_` so nothing there is overwritten, and
the grids are shown for review.

1. **Inputs**, made once: a landscape canvas (1216x832), a portrait reference (832x1216) and a
   square one (1024x1024), each with marks at its edges that a crop would cut.
2. **What each engine is handed**: the loop run on its own, before and after, printing the size
   of every prepared image for Qwen (`None`), Flux2 (default) and an engine with no declaration.
   Expected: the canvas the same size before and after; the references at their own ratio after,
   whole for Qwen, at the output's area for Flux2. A grid shows the prepared images side by side,
   `area_prepared.png`, where the cut ones are plain to see.
3. **Generations** through `bash/turbo/image-to-image.sh`, seed 7, canvas first, a prompt that
   uses both references: before (current code) and after, on `qwen-image-edit-2511-lightning` and
   `flux2-4b`, at a small frame (512x352) and at ~1 MP (1216x832).
4. **Grids**, one per engine, `area_<engine>.png`: inputs on top, then before / after per output
   size, with the time per step of each run under it.
5. Expected: the references' edges come through in the after runs, the canvas lines up with the
   output in both, Qwen and Flux2 times unchanged. A Flux2 reference that looks too soft on the
   small frame is the case for `IMAGE_AREA = None` there, and that run is added to its grid.
6. `python -m runner.cli` on a text-to-image and a mask run still works (the loop is not on their
   path), and the ≤99 column check passes on every touched file.
