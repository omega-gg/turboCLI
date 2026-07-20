TEMPLATE = subdirs

include(bash/bash.pri)

OTHER_FILES += README.md  \
               LICENSE.md \
               implementation.md \
               dummy.md \
               update.sh \
               doc/comfy-krea2-turbo-plan.md \
               doc/comfy-z-image-turbo-plan.md \
               doc/engine-inheritance-plan.md \
               doc/DUMMY_PLAN.md \
               doc/IMPLEMENTATION_PLAN.md \

OTHER_FILES += runner/__init__.py \
               runner/check.py \
               runner/cli.py \
               runner/core.py \
               runner/install.py \
               runner/server.py \
               runner/engine/__init__.py \
               runner/engine/_inherit.py \
               runner/engine/comfy_krea2_turbo.py \
               runner/engine/comfy_krea2_turbo_realism.py \
               runner/engine/comfy_qwen_image_edit_2511.py \
               runner/engine/comfy_qwen_image_edit_2511_lightning.py \
               runner/engine/comfy_z_image_turbo.py \
               runner/engine/flux2_4b.py \
               runner/engine/qwen_image_edit_2511.py \
               runner/engine/qwen_image_edit_2511_lightning.py \
               runner/engine/qwen_image_edit_2511_lightning_angles.py \
               runner/engine/z_image_turbo.py \
