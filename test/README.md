# [turboCLI](../README.md) test fixtures

Reference images (736x1024, flux2-4b generations) for re-running the `image-mask` verifications.

| file | what it is |
|---|---|
| [knight.png](knight.png) | a knight standing in a temple courtyard (the subject scene) |
| [courtyard.png](courtyard.png) | the same courtyard with the knight removed (the clean background plate) |
| [chest.png](chest.png) | the courtyard with a treasure chest added (an edit of `courtyard.png`) |

Masks are generated then applied. `image-mask <mode> <reference> <input> <mask>` (mask / region)
and `image-mask-background <model> <renderer> <input> <matte> [plate]` generate a mask;
`image-mask-apply <mode> <input> <mask> <output> [reference]` applies either
(composite / putalpha). `SKY_PATH_BIN` must point at the install. Run from anywhere:

```sh
# background matte, then cut the subject onto transparency (default model birefnet)
image-mask-background birefnet cuda knight.png matte.png
image-mask-apply putalpha knight.png matte.png subject_only.png

# lucida fine-tune, keeping the cast shadow recovered from a clean plate (the empty courtyard)
image-mask-background lucida cuda knight.png matte_shadow.png courtyard.png
image-mask-apply putalpha knight.png matte_shadow.png knight_cutout.png

# region: remove the knight (reference = knight scene, input = empty courtyard edit)
image-mask region knight.png courtyard.png remove_mask.png
image-mask-apply composite courtyard.png remove_mask.png knight_removed.png knight.png

# mask: keep an added object (reference = empty courtyard, input = chest edit)
image-mask mask courtyard.png chest.png chest_mask.png
image-mask-apply composite chest.png chest_mask.png chest_kept.png courtyard.png
```

`image-mask-background` needs the remove-background tool installed
([../bash/remove-background/build.sh](../bash/remove-background/build.sh)); `image-mask` and
`image-mask-apply` need only the turbo venv.

## Composing a scene across edits

`image-to-image` redraws the whole frame, so each edit drifts the background. Add objects one at a
time and mask each stage against the original plate -- that undoes the drift while keeping what you
added -- then feed the masked result back as the next edit's base so drift never accumulates. The
knights scene (add a knight, then a second doing an accolade) does exactly this:

```sh
# stage 1: add a knight (keep the background), then mask it onto the clean temple
image-to-image flux2-4b cuda "a knight ...; keep the original background identical" \
    courtyard.png kn1.png 736 1024 7 4
image-mask mask courtyard.png kn1.png kn1_mask.png 40
image-mask-apply composite kn1.png kn1_mask.png kn1_clean.png courtyard.png

# stage 2: add the second knight from the CLEAN base, then mask both onto the temple
image-to-image flux2-4b cuda "two knights ... accolade ...; keep the original background identical" \
    kn1_clean.png kn2.png 736 1024 7 4
image-mask mask courtyard.png kn2.png kn2_mask.png 40
image-mask-apply composite kn2.png kn2_mask.png kn2_clean.png courtyard.png

# reuse that same mask to cut both knights onto transparency
image-mask-apply putalpha kn2.png kn2_mask.png kn2_cutout.png
```

Raise the `image-mask` threshold (40 here) when flux2's whole-frame drift speckles the mask; the
last two lines show one generated mask feeding both apply modes (composite + putalpha).
