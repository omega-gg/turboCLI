# [turboCLI](../README.md) test fixtures

Reference images (736x1024, flux2-4b generations) for re-running the mask-engine verifications.

| file | what it is |
|---|---|
| [knight.png](knight.png) | a knight standing in a temple courtyard (the subject scene) |
| [courtyard.png](courtyard.png) | the same courtyard with the knight removed (the clean background plate) |
| [chest.png](chest.png) | the courtyard with a treasure chest added (an edit of `courtyard.png`) |

Masks are generated then applied. `image-to-mask <engine> <renderer> <input images> <mask>`
generates a mask/matte (engine = mask | mask-birefnet | mask-lucida | mask-inspyrenet);
`image-apply-mask <mode> <input images> <output>` applies it (mode = composite | putalpha). Inputs
are a comma-separated, ordered list -- the input always first. `SKY_PATH_BIN` must point at the
install; run from anywhere:

```sh
# background matte, then cut the subject onto transparency (birefnet)
image-to-mask mask-birefnet cuda knight.png matte.png
image-apply-mask putalpha knight.png,matte.png subject_only.png

# lucida, keeping the cast shadow recovered from a clean plate (the empty courtyard)
image-to-mask mask-lucida cuda knight.png,courtyard.png matte_shadow.png
image-apply-mask putalpha knight.png,matte_shadow.png knight_cutout.png

# region: remove the knight (input = empty courtyard edit, reference = knight scene)
image-to-mask mask cpu courtyard.png,knight.png remove_mask.png mode=region
image-apply-mask composite courtyard.png,remove_mask.png,knight.png knight_removed.png

# mask: keep an added object (input = chest edit, reference = empty courtyard)
image-to-mask mask cpu chest.png,courtyard.png chest_mask.png
image-apply-mask composite chest.png,chest_mask.png,courtyard.png chest_kept.png
```

The matte engines (mask-birefnet / mask-lucida / mask-inspyrenet) need their model installed
(`install mask-birefnet`); mask / mask-apply need only the turbo venv (install registers them,
`install mask`).

## Composing a scene across edits

`image-to-image` redraws the whole frame, so each edit drifts the background. Add objects one at a
time and mask each stage against the original plate -- that undoes the drift while keeping what you
added -- then feed the masked result back as the next edit's base so drift never accumulates. The
knights scene (add a knight, then a second doing an accolade) does exactly this:

```sh
# stage 1: add a knight (keep the background), then mask it onto the clean temple
image-to-image flux2-4b cuda "a knight ...; keep the original background identical" \
    courtyard.png kn1.png 736 1024 7 4
image-to-mask mask cpu kn1.png,courtyard.png kn1_mask.png tolerance=215
image-apply-mask composite kn1.png,kn1_mask.png,courtyard.png kn1_clean.png

# stage 2: add the second knight from the CLEAN base, then mask both onto the temple
image-to-image flux2-4b cuda "two knights ... accolade ...; keep the original background identical" \
    kn1_clean.png kn2.png 736 1024 7 4
image-to-mask mask cpu kn2.png,courtyard.png kn2_mask.png tolerance=215
image-apply-mask composite kn2.png,kn2_mask.png,courtyard.png kn2_clean.png

# reuse that same mask to cut both knights onto transparency
image-apply-mask putalpha kn2.png,kn2_mask.png kn2_cutout.png
```

Lower the mask tolerance below the default 231 (`tolerance=215` here) when flux2's whole-frame drift
speckles the mask; the last two lines show one generated mask feeding both apply modes (composite +
putalpha).
