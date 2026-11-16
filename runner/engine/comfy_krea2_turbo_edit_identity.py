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

# comfy-krea2-turbo-edit-identity: comfy-krea2-turbo restaging a person from a reference photo,
# with conradlocke's Krea 2 Identity Edit V2 LoRA (free for personal and non-commercial use) and
# the conditioning it was trained with, as the ComfyUI-Krea2Edit node pack (v2.0.0) gives it:
#
# - appearance: each reference is fitted to the target's pixel size, VAE-encoded and set in the
#   sequence [text | references | target] on the target's grid (frame 1, 2, centered), clean and
#   on the target's timestep; the output is the target's slice.
# - semantics: the prompt is encoded through Qwen3-VL with the references in the user turn, their
#   longest side capped at 768 pixels.
# - REF_BOOST: log(REF_BOOST) on the attention of the target to the subject (the last reference),
#   in the main blocks.
#
# The offloader runs ComfyUI's Krea 2 with the node pack's encode and edit forward: the references
# go to its text encoder with the prompt, and to the transformer with their boosts and placements
# (its `edit`); the denoise, the schedule and the decode are the stock pipeline's.
# GATED: the base's scaffold sits in its gated repo (see comfy_krea2_turbo), and a variant does
# not inherit GATED.

from . import comfy_krea2_turbo as base  # cheap: base imports no torch at top level

ID   = "comfy-krea2-turbo-edit-identity"
BASE = base.ID

MODES = ("image-to-image",)

GATED = True

LORA = "krea2_identity_edit_v2.safetensors"

# The node pack's identity presets.
REF_BOOST      = 2.0  # on the subject
MAX_REFERENCES = 2    # the scene, then the subject

# Where the subject sits for the model: "beside" the target, to its right, which lets a restage
# move it (the presets' choice to restage one person), or "aligned" on the target's grid, as
# trained, which holds its layout (a face-locked portrait). The scene is always aligned.
REF_POSITION = "beside"

# The base's three files and the LoRA, at the root of a plain repo (so its `filename`),
# revision-pinned. dict(base.COMFY, ...) never mutates base.COMFY.
COMFY = dict(base.COMFY, components=base.COMFY["components"] + [
    {"role": "lora", "repository": "conradlocke/krea2-identity-edit",
     "revision": "f33b26291c640e8ec50b0da420bfee3c9a660daa",
     "filename": LORA, "path": "loras/" + LORA},
])


def load(ctx, params, ref_boost=REF_BOOST, ref_position=REF_POSITION):
    """The base's assembly on the edit pipeline, the engine's LoRA first at 1.0 (ComfyUI-native
    keys, as ComfyUI's model takes them), any user --loras after it. An engine building on this
    one passes its own boost and placement."""
    ctx.loras = [(base._by_role(ctx.model)["lora"], 1.0)] + list(ctx.loras)

    return base.load(ctx, params, _pipeline_class(ref_boost, ref_position))


def _pipeline_class(ref_boost, ref_position):
    """The base's pipeline with references. Built here so diffusers stays out of engine
    discovery."""
    import torch

    class Krea2EditPipe(base._pipeline_class()):
        def __call__(self, prompt=None, image=None, height=None, width=None, **kwargs):
            # The stock call raises on a size off its grid (vae_scale_factor * patch_size), so
            # the size is snapped down to it, as the Qwen pipelines do. The references are
            # fitted to that same size.
            multiple = self.vae_scale_factor * self.patch_size

            if height and width:
                height, width = height // multiple * multiple, width // multiple * multiple

            images = list(image or [])[:MAX_REFERENCES]

            if not images:
                return super().__call__(prompt=prompt, height=height, width=width, **kwargs)

            # Encoded through encode_prompt, so the offloader's cache keys it on the images too.
            # The stock call wants a mask beside the embeds, all of them valid here.
            embeds, _ = self.encode_prompt(prompt, image=images, device=self._execution_device)

            mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)

            # The scene aligned and unboosted, the subject (the last reference) as asked.
            boosts = [1.0] * (len(images) - 1) + [ref_boost]
            positions = ["aligned"] * (len(images) - 1) + [ref_position]

            with self.transformer.edit(images, self.vae, height, width, boosts, positions):
                return super().__call__(prompt_embeds=embeds, prompt_embeds_mask=mask,
                                        height=height, width=width, **kwargs)

    return Krea2EditPipe
