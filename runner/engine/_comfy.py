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

# Helpers the comfy engines share: the ComfyUI files an engine's record names and the VAEs they
# build. A helper (underscore name) so engine discovery skips it, torch free at import: an engine
# imports it beside its one base at most.

import os
import json


def by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file. Reads the engine.json comfy record
    (written by install: {root, external, components, scaffold}) and resolves each component's
    root-relative `path` (e.g. models/vae/ae.safetensors) against `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def flux2_vae(scaffold, weight_file, dtype):
    """The FLUX.2 VAE from the scaffold config, with ComfyUI's single file. AutoencoderKLFlux2 has
    no from_single_file mapping in diffusers, but needs none: the ComfyUI file is already in
    diffusers key layout (251/251 keys, identical shapes, verified), so it binds as is.
    `.to(dtype)` leaves the integer bn.num_batches_tracked buffer alone."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLFlux2

    cfg = AutoencoderKLFlux2.load_config(os.path.join(scaffold, "vae"))
    vae = AutoencoderKLFlux2.from_config(cfg)

    vae.load_state_dict(safetensors_torch.load_file(weight_file), strict=True)

    return vae.to(dtype).eval()


def comfy_vae(scaffold, weight_file, backend):
    """ComfyUI's own VAE, the model comfy/sd.py builds for the file, run as sd.py runs it through
    the offloader's comfy_vae, so it decodes and encodes as ComfyUI does (the Wan VAE engines:
    comfy-qwen-image-edit-2511, comfy-qwen-image-2-1, comfy-krea2-turbo)."""
    # z_dim and the latent statistics the pipeline reads.
    with open(os.path.join(scaffold, "vae", "config.json")) as f:
        config = json.load(f)

    return backend.comfy_vae(weight_file, config)
