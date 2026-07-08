#==================================================================================================
#
#   Copyright (C) 2026-2026 turboCLI authors. <https://omega.gg/turboCLI>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turboCLI.
#
#   - GNU Lesser General Public License Usage:
#   This file may be used under the terms of the GNU Lesser General Public License version 3 as
#   published by the Free Software Foundation and appearing in the LICENSE.md file included in the
#   packaging of this file. Please review the following information to ensure the GNU Lesser
#   General Public License requirements will be met: https://www.gnu.org/licenses/lgpl.html.
#
#   - Private License Usage:
#   turboCLI licensees holding valid private licenses may use this file in accordance with the
#   private license agreement provided with the Software or, alternatively, in accordance with the
#   terms contained in written agreement between you and turboCLI authors. For further information
#   contact us at contact@omega.gg.
#
#==================================================================================================

# qwen-image-edit-2511 (default) -- base QwenImageEditPlusPipeline, NO LoRA. Pure declaration:
# core's default load (from_pretrained / backend.load_pipe + apply_offload) and run path cover it.
# TYPE stays "qwen-image-edit" (the backend seam's identity for this pipeline) while the engine is
# selectable as "qwen-image-edit-2511". For LoRA presets over the SAME model, see
# qwen_image_edit_2511_lightning.py and qwen_image_edit_2511_lightning_angles.py.

NAME     = "qwen-image-edit-2511"
TYPE     = "qwen-image-edit"
PIPELINE = "diffusers:QwenImageEditPlusPipeline"
MODES    = ("image-to-image",)
CFG      = ("true_cfg_scale", 1.0)
INFERENCE = 40

# Install (python -m runner.install): the base model only, no LoRA.
# "revision" pins the HF commit (mutable repos -> reproducible installs); check validates it.
MODEL = {"repository": "Qwen", "model": "Qwen-Image-Edit-2511",
         "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"}
