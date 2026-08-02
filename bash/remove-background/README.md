# [Bash](../README.md) remove-background

Standalone subject-matte generator (BiRefNet + InSPyReNet) behind
[image-mask-background](../turbo/README.md). Emits an 8-bit grayscale matte; apply it with
image-mask-apply. Installs its own venv + models under `gg.omega/remove-background`, isolated from
the turbo venv.

### [build.sh](build.sh) - Install the tool (venv + models) in the SKY_PATH_BIN folder

```
Usage: build <cpu | cuda | mps> [latest]

latest: install the newest releases + models, ignoring the pins (not reproducible)

Downloads three models: birefnet (ZhengPeng7/BiRefNet), lucida (egeorcun/lucida) and
inspyrenet (transparent-background's InSPyReNet base), each pinned by revision.

example:
    build cuda
    build cuda latest
```

### [check.sh](check.sh) - Check the install validity (venv + pinned model revisions)

```
Usage: check
```

### [run.sh](run.sh) - Produce a subject matte (8-bit grayscale)

```
Usage: run <model> <renderer> <input image> <matte output> [plate image] [shadow threshold]

model: birefnet   (ZhengPeng7/BiRefNet) -- strong on thin glows (a neon sign, a saber)
       lucida     (egeorcun/lucida fine-tune) -- glass / camouflage / text / print
       inspyrenet (transparent-background) -- InSPyReNet, also strong on thin glows

renderer: cpu, cuda or mps (cuda / mps fall back to cpu if this build lacks them)

plate: a clean background (the same scene without the subject); its cast shadow is kept

shadow threshold: darkening floor for the plate shadow (default 12); raise it when a drifted
                  plate ghosts the background. Only used with a plate.

examples:
    run birefnet cuda photo.png matte.png
    run lucida   cuda photo.png matte.png plate.png
    run lucida   cuda photo.png matte.png plate.png 40
```
