# [turboCLI](../README.md) test fixtures

Reference images (736x1024, flux2-4b generations) for re-running the `image-mask` verifications.

| file | what it is |
|---|---|
| [knight.png](knight.png) | a knight standing in a temple courtyard (the subject scene) |
| [courtyard.png](courtyard.png) | the same courtyard with the knight removed (the clean background plate) |
| [chest.png](chest.png) | the courtyard with a treasure chest added (an edit of `courtyard.png`) |

`image-mask` lives in [../bash/turbo/image-mask.sh](../bash/turbo/image-mask.sh); `SKY_PATH_BIN`
must point at the install. Run from anywhere:

`image-mask <mode> <reference> <input> <output>` (mask / region) and
`image-remove-background <model> <renderer> <input> <output> [plate]`:

```sh
# background removal, subject only (default model birefnet)
image-remove-background birefnet cuda knight.png subject_only.png

# with the lucida fine-tune instead
image-remove-background lucida  cuda knight.png subject_lucida.png

# keep the cast shadow, recovered from a clean plate (the empty courtyard)
image-remove-background birefnet cuda knight.png knight_cutout.png courtyard.png

# region: remove the knight (reference = knight scene, input = empty courtyard edit)
image-mask region knight.png    courtyard.png knight_removed.png

# mask: keep an added object (reference = empty courtyard, input = chest edit)
image-mask mask   courtyard.png chest.png     chest_kept.png
```

`image-remove-background` needs the lucida tool installed
([../bash/lucida/build.sh](../bash/lucida/build.sh)); `image-mask` needs only the turbo venv.
