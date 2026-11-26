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

# comfy-flux3-6b engine: text2img + img2img on FLUX.3 Image Klein (5.6B, the 16-step distilled
# flow-map checkpoint, no CFG), REUSING a ComfyUI install file for file: BFL's int8 ConvRot
# transformer and Qwen3-VL-4B text encoder, and the FLUX.2 VAE (the same component as
# comfy-flux2-4b). Only configs come from the diffusers repo (the scaffold); the pipeline blocks
# build their own scheduler.
#
# GATED: both BFL repos need a Hugging Face token from an account that accepted the FLUX Kommunity
# License (install.sh takes it as its last argument).
#
# OFFLOADER-ONLY: int8 needs the comfy quant path, so load() bails out for other offload modes.
# The transformer and the text encoder are ComfyUI's own models, which the offloader runs behind
# the pipeline's calls (its "flux3" model); a LoRA may also name its weights the diffusers way
# (_lora_names).
#
# The pipeline is flux3-6b's Flux3Pipe, the classic face over diffusers' modular FLUX.3, encoding
# through ComfyUI's text encoder (_pipeline_class).

import os

from . import flux3_6b as stock     # cheap: no torch at top level
from . import _comfy  # cheap: no torch at top level

ID   = "comfy-flux3-6b"
TYPE = "flux3"

PIPELINE    = "diffusers:Flux3DistilledModularPipeline"
TRANSFORMER = "diffusers:Flux3Transformer2DModel"  # ComfyUI's runs in its place

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 0.0)              # distilled / CFG-free (Flux3Pipe ignores it)

# The flow-map checkpoint takes no guidance: the denoise passes it the timestep each step jumps
# to instead.
INFERENCE = 16

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

_REPO = "black-forest-labs/flux-3-image-klein-comfy"

# The "_sd_" files are the step-distilled ones, the others the 50-step base. Revisions pinned.
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": _REPO,
         "revision": "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0",
         "path": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
         "filename": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
         # The same file (sha256 847d8933...), renamed in the repo and ComfyUI's templates.
         "alias": "diffusion_models/flux3-image-klein-int8-convrot.safetensors"},
        {"role": "text_encoder", "repository": _REPO,
         "revision": "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0",
         "path": "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
         "filename": "text_encoders/qwen3vl_4b_int8_convrot.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/flux2-dev",
         "path": "vae/flux2-vae.safetensors"},
    ],
}

# The diffusers repo, for its configs. Weight-free.
SCAFFOLD = {
    "repository": "black-forest-labs",
    "model": "flux-3-image-klein",
    "revision": "337337621b8526e120f2275d250242f4a6bedd75",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer/config.json",
        "vae/config.json",
    ],
}

# BFL's name for each diffusers key (ComfyUI's naming), the inverse of diffusers'
# convert_flux3_transformer_checkpoint_to_diffusers.
_BFL_PARTS = (
    (".attn.to_q.", ".q_proj."),
    (".attn.to_k.", ".k_proj."),
    (".attn.to_v.", ".v_proj."),
    (".attn.to_out.0.", ".attn_out."),
    (".attn.norm_q.weight", ".norm.query_norm.scale"),
    (".attn.norm_k.weight", ".norm.key_norm.scale"),
    (".ff.linear_in.", ".mlp_in."),
    (".ff.linear_out.", ".mlp_out."),
    (".linear_1.", ".in_layer."),
    (".linear_2.", ".out_layer."),
)

_BFL_PREFIXES = (
    ("mode_blocks.", "content_mode_blocks."),
    ("single_transformer_blocks.", "single_blocks."),
    ("context_embedder.", "txt_in."),
    ("x_embedders.", "emb_in."),
    ("target_timestep_embedder.", "ref_time_in."),
    ("timestep_embedder.", "time_in."),
    ("norm_out.linear.", "final_layer.image.adaLN_modulation.1."),
    ("proj_out.", "final_layer.image.linear."),
)


def _bfl_key(key):
    for diffusers, bfl in _BFL_PARTS:
        key = key.replace(diffusers, bfl)

    if "_modulations." in key:
        key = key.replace(".linear.", ".lin.")

    for diffusers, bfl in _BFL_PREFIXES:
        if key.startswith(diffusers):
            return bfl + key[len(diffusers):]
    return key


def _diffusers_key(key):
    """_bfl_key's inverse, for a module (no .weight)."""
    key += "."
    for diffusers, bfl in _BFL_PREFIXES:
        if key.startswith(bfl):
            key = diffusers + key[len(bfl):]
            break

    if "_modulations." in key:
        key = key.replace(".lin.", ".linear.")

    for diffusers, bfl in _BFL_PARTS:
        key = key.replace(bfl, diffusers)
    return key[:-1]


def _lora_names(key_map):
    """A LoRA's diffusers names for the keys of ComfyUI's model, as diffusers trains them
    (to_q, single_transformer_blocks, ...), bare or under transformer."""
    names = {}
    for key, to in key_map.items():
        if key.startswith("diffusion_model."):
            name = _diffusers_key(key[len("diffusion_model."):])
            names[name] = to
            names["transformer.{}".format(name)] = to
    return names


def _pipeline_class():
    """flux3-6b's Flux3Pipe encoding through ComfyUI's text encoder, as ComfyUI's CLIPTextEncode
    does. Built here so diffusers stays out of engine discovery."""

    class ComfyFlux3Pipe(stock._pipeline_class()):
        def encode_prompt(self, prompt, device=None):
            return self.text_encoder.encode(prompt).to(device or self._execution_device)

    return ComfyFlux3Pipe


def load(ctx, params):
    """Assemble the pipeline of _pipeline_class from ComfyUI's int8 single files via the
    disk-stream offloader. OFFLOADER-ONLY (int8 needs the comfy quant path). The VAE is built
    from the scaffold."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = _comfy.by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); both big models "
                           "are int8 and need the comfy quant path" % ID)

    from diffusers import Flux3Transformer2DModel

    vae = _comfy.flux2_vae(scaffold, files["vae"], ctx.dtype)

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(),
        {"model": "flux3", "file": files["transformer"],
         "config": Flux3Transformer2DModel.load_config(os.path.join(scaffold, "transformer")),
         "lora_names": _lora_names},
        {"model": "flux3", "file": files["text_encoder"]},
        {"tokenizer": None, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)
