# [Bash](../README.md) remove-background

Standalone background remover (BiRefNet + InSPyReNet) behind
[image-remove-background](../turbo/README.md). Installs its own venv + models under
`gg.omega/remove-background`, isolated from the turbo venv.

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

### [run.sh](run.sh) - Cut a subject onto a transparent background

```
Usage: run <model> <renderer> <input image> <output image> [plate image]

model: birefnet   (ZhengPeng7/BiRefNet) -- strong on thin glows (a neon sign, a saber)
       lucida     (egeorcun/lucida fine-tune) -- glass / camouflage / text / print
       inspyrenet (transparent-background) -- InSPyReNet, also strong on thin glows

renderer: cpu, cuda or mps (cuda / mps fall back to cpu if this build lacks them)

plate: a clean background (the same scene without the subject); its cast shadow is kept

examples:
    run birefnet cuda photo.png cutout.png
    run lucida  cuda photo.png cutout.png plate.png
```
