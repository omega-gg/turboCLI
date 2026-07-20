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

# comfy-krea2-turbo-realism -- comfy-krea2-turbo with the Krea2-realism-V2 LoRA always applied at
# 1.5. Inherits the whole engine (TYPE / PIPELINE / TRANSFORMER / MODES / CFG / INFERENCE /
# SCAFFOLD, the meta-builders and the streaming load) from the base via BASE. Two deltas: a 4th
# COMFY component for the LoRA -- reused from the ComfyUI install's models/loras/, exactly like
# the three single files the base already streams -- and a load() that prepends it to ctx.loras at
# STRENGTH, so the base assembly applies it on-cast onto the fp8 transformer just like a user
# --loras entry (256 patches).

from . import comfy_krea2_turbo as base  # cheap: base imports no torch at top level

ID   = "comfy-krea2-turbo-realism"
BASE = base.ID

# The LoRA (LoKr; every Krea2 key naming loads via the base's _lora_keys map) and the author's
# recommended default strength.
REALISM  = "Krea2-realism-V2.safetensors"
STRENGTH = 1.5

# Extend the base's three reused single files with the LoRA (4th component), landing in the same
# ComfyUI models/loras/ that comfy-qwen-image-edit-2511-lightning uses. It sits in a plain HF repo
# -- the file is at the repo root, not under split_files/ -- so the component carries an explicit
# `filename` (the in-repo path) distinct from its `path` (the models/-relative destination), and a
# `revision` pinning the commit for reproducible installs. _install_comfy reuses the file when it
# is already present and only downloads an absent one. dict(base.COMFY, ...) is a shallow copy
# that overrides `components` while inheriting `revision` -- it never mutates base.COMFY.
COMFY = dict(base.COMFY, components=base.COMFY["components"] + [
    {"role": "lora", "repository": "RudySen/Krea2-realism-V2",
     "revision": "ad6f07e426303e7087d215362dbc057b0073dca3",
     "filename": REALISM, "path": "loras/" + REALISM},
])


def load(ctx, params):
    """Reuse the base Krea2 assembly, but always stream the realism LoRA first at STRENGTH. The
    path comes from the engine.json comfy record (the reused ComfyUI file, same resolution the
    base uses for its own three), and is prepended to ctx.loras -- any user --loras still stack
    after -- then base.load hands the merged list to load_pipe_comfy, applied on-cast onto the fp8
    transformer."""
    ctx.loras = [(base._by_role(ctx.model)["lora"], STRENGTH)] + list(ctx.loras)

    return base.load(ctx, params)
