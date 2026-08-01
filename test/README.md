# [turboCLI](../README.md) test fixtures

Reference images (736x1024, flux2-4b generations) for re-running the `image-mask` verifications.

| file | what it is |
|---|---|
| [knight.png](knight.png) | a knight standing in a temple courtyard (the subject scene) |
| [courtyard.png](courtyard.png) | the same courtyard with the knight removed (the clean background plate) |
| [chest.png](chest.png) | the courtyard with a treasure chest added (an edit of `courtyard.png`) |

`image-mask` lives in [../bash/turbo/image-mask.sh](../bash/turbo/image-mask.sh); `SKY_PATH_BIN`
must point at the install. Run from anywhere:

Signature: `image-mask <mode> <renderer> <reference> <input> <output>`.

```sh
# extract: background removal, subject only (reference is unused)
image-mask extract      cuda knight.png    knight.png subject_only.png

# extract-full: also keep the cast shadow, recovered from the clean plate (empty courtyard)
image-mask extract-full cuda courtyard.png knight.png knight_cutout.png

# region: remove the knight (reference = knight scene, input = empty courtyard edit)
image-mask region       cpu  knight.png    courtyard.png knight_removed.png

# mask: keep an added object (reference = empty courtyard, input = chest edit)
image-mask mask         cpu  courtyard.png chest.png     chest_kept.png
```

`extract` / `extract-full` need the lucida tool installed
([../bash/lucida/build.sh](../bash/lucida/build.sh)); `mask` / `region` need only the turbo venv
(the renderer is ignored for them).
