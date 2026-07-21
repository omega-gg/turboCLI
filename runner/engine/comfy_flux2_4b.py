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

# comfy-flux2-4b engine -- text2img + img2img on FLUX.2-klein-4B, REUSING a ComfyUI install.
#
# Same weights as the flux2-4b engine (same repo, same pinned revision), but loaded from ComfyUI's
# split single files instead of the diffusers repo, so a user who already runs ComfyUI does not
# re-download ~15GB. Two files are reused: the bf16 klein transformer and the Qwen3-4B text
# encoder -- and that text encoder is the SAME file ComfyUI already holds for Z-Image
# (text_encoders/qwen_3_4b.safetensors, verified 398/398 keys, 0 shape mismatches against
# FLUX.2-klein-4B/text_encoder), so on a box that ran either workflow it costs nothing.
#
# Unlike comfy-z-image-turbo (bare Qwen3Model -> strip the 'model.' prefix), the klein pipeline
# wants a Qwen3ForCausalLM, whose keys ARE 'model.'-prefixed -- so the ComfyUI file binds as-is.
# The only gap is the tied lm_head, which _tie_lm_head aliases back onto the embedding.
#
# Unlike comfy-krea2-turbo, the transformer single file is plain bf16 (149 tensors, no quant
# scales), so there is no "quant" spec and no comfy quant path: BOTH the native and the offloader
# disk-stream paths work. Its ComfyUI-native keys (double_blocks/single_blocks) are remapped by
# diffusers' own convert_flux2_transformer_checkpoint_to_diffusers -- no converter to hand-write.
#
# The VAE is reused too. AutoencoderKLFlux2 has no from_single_file mapping in diffusers, but it
# needs none: ComfyUI's flux2-vae.safetensors is ALREADY in diffusers key layout (251/251 keys,
# identical shapes -- verified), so _build_vae just instantiates from the scaffold config and
# load_state_dict's the file. No converter, and all three big files come from ComfyUI.
#
# install:  runner.install --comfy <ComfyUI dir> resolves/downloads the components and writes
#           engine.json + scaffold into engine/<id> (dispatched on this COMFY declaration).
# check:    runner.check treats a COMFY engine as installed when engine.json + its files + the
#           scaffold are present (no HF revision marker).
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level -- discovery stays
# cheap. The heavy imports live inside load().

import os
import json

ID   = "comfy-flux2-4b"
TYPE = "flux2"  # speak the offloader seam's vocabulary (same pipeline family as flux2-4b)

PIPELINE    = "diffusers:Flux2KleinPipeline"
TRANSFORMER = "diffusers:Flux2Transformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 0.0)              # distilled / CFG-free

INFERENCE = 4

# ComfyUI split single files reused by this engine. `path` is the file's location under ComfyUI's
# models/, `role` the diffusers component it feeds. The transformer sits at the ROOT of the BFL
# repo (not under split_files/), so it names `filename` explicitly; the text encoder comes from
# Comfy-Org's split_files/ layout, so its filename defaults. Fetched only when missing.
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "black-forest-labs/FLUX.2-klein-4B",
         "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
         "path": "diffusion_models/flux-2-klein-4b.safetensors",
         "filename": "flux-2-klein-4b.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/z_image_turbo",
         "path": "text_encoders/qwen_3_4b.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/flux2-dev",
         "path": "vae/flux2-vae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the same repo (and revision) flux2-4b pins: configs +
# tokenizer + scheduler, so the single files can be loaded/meta-loaded offline. Weight-free --
# vae/config.json only, never the VAE weights (those are reused from ComfyUI).
SCAFFOLD = {
    "repository": "black-forest-labs",
    "model": "FLUX.2-klein-4B",
    "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ],
}


def _by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file. Reads the engine.json comfy record
    (written by install: {root, external, components, scaffold}) and resolves each component's
    root-relative `path` (e.g. models/diffusion_models/flux-2-klein-4b.safetensors) against
    `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _tie_lm_head(sd):
    """ComfyUI's qwen_3_4b file carries the 398 'model.'-prefixed keys Qwen3ForCausalLM expects,
    but not the 399th: lm_head.weight, which the config ties to the embedding
    (tie_word_embeddings=True). Alias it back -- same tensor, so streaming stays mmap-backed and
    nothing is left on meta."""
    return dict(sd, **{"lm_head.weight": sd["model.embed_tokens.weight"]})


def _transformer_meta(scaffold, dtype):
    """Meta-build (no weight RAM) the Flux2 transformer from the scaffold config, for the offloader
    to stream the ComfyUI single file into."""
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel

    cfg = Flux2Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        return Flux2Transformer2DModel.from_config(cfg).to(dtype)


def _text_encoder_meta(scaffold, dtype):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3ForCausalLM

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        return Qwen3ForCausalLM(cfg).to(dtype)


def _build_vae(scaffold, weight_file, dtype):
    """Instantiate the Flux2 VAE from the scaffold config and load ComfyUI's single file.
    AutoencoderKLFlux2 has no from_single_file mapping in diffusers, but needs none: the ComfyUI
    file is already in diffusers key layout (251/251 keys, identical shapes -- verified), so it
    binds as-is. `.to(dtype)` leaves the integer bn.num_batches_tracked buffer alone."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLFlux2

    cfg = AutoencoderKLFlux2.load_config(os.path.join(scaffold, "vae"))
    vae = AutoencoderKLFlux2.from_config(cfg)

    vae.load_state_dict(safetensors_torch.load_file(weight_file), strict=True)

    return vae.to(dtype).eval()


def _build_text_encoder(scaffold, weight_file, dtype):
    """Instantiate the Qwen3 text encoder from the scaffold config and load ComfyUI's single-file
    weights (keys already match; only the tied lm_head is added)."""
    import safetensors.torch as safetensors_torch
    from transformers import AutoConfig, Qwen3ForCausalLM

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    model = Qwen3ForCausalLM(cfg)

    model.load_state_dict(_tie_lm_head(safetensors_torch.load_file(weight_file)), strict=True)

    return model.to(dtype).eval()


def load(ctx, params):
    """Build a Flux2KleinPipeline from ComfyUI's single files. The offload backend, when present,
    meta-loads the two big models straight from those files (disk-stream); otherwise both are built
    here and placed via the shared ctx helpers. The VAE, tokenizer and scheduler always come from
    the scaffold."""
    from diffusers import (Flux2KleinPipeline, Flux2Transformer2DModel,
                           FlowMatchEulerDiscreteScheduler)
    from transformers import AutoTokenizer

    scaffold = ctx.model  # engine/comfy-flux2-4b/ (scaffolding + engine.json)
    files    = _by_role(scaffold)

    # Small resident components: scheduler + tokenizer from the scaffold, the VAE reused from
    # ComfyUI (built here rather than from_single_file -- see _build_vae).
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))
    vae = _build_vae(scaffold, files["vae"], ctx.dtype)

    # Disk-stream path: hand the backend model-agnostic specs -- the two big models' meta-builders
    # + single-file paths + key remaps (transformer via the diffusers converter, TE via the tied
    # lm_head alias) -- plus the pre-built small components. All Flux2 specifics stay in this file.
    if ctx.backend is not None:
        from diffusers.loaders.single_file_utils import (
            convert_flux2_transformer_checkpoint_to_diffusers as convert_transformer)

        return ctx.backend.load_pipe_comfy(
            Flux2KleinPipeline,
            {"meta": lambda d: _transformer_meta(scaffold, d), "file": files["transformer"],
             "convert": convert_transformer},
            {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
             "convert": _tie_lm_head},
            {"scheduler": scheduler, "tokenizer": tokenizer, "vae": vae, "is_distilled": True},
            ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

    # Native path: materialise both big models, then apply LoRAs + placement via the ctx helpers.
    text_encoder = _build_text_encoder(scaffold, files["text_encoder"], ctx.dtype)

    transformer = Flux2Transformer2DModel.from_single_file(
        files["transformer"], config=scaffold, subfolder="transformer",
        torch_dtype=ctx.dtype, local_files_only=True)

    p = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                           tokenizer=tokenizer, transformer=transformer, is_distilled=True)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)
