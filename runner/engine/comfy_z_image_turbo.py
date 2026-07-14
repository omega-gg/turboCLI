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

# comfy-z-image-turbo engine -- text2img on Z-Image-Turbo, REUSING a ComfyUI install's model files.
#
# Same weights as the z-image-turbo engine, but ComfyUI stores them as three split single-file
# safetensors (a diffusion model, a Qwen3-4B text encoder, a VAE) rather than a diffusers repo.
# This engine loads those single files straight into a diffusers ZImagePipeline, so a user who
# already runs ComfyUI does not re-download ~20GB. The big weights live wherever ComfyUI keeps
# them; this engine's registry dir (engine/comfy-z-image-turbo/) holds only the tiny diffusers
# scaffolding (configs + tokenizer + scheduler) plus an engine.json manifest that points at the
# ComfyUI files.
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

ID   = "comfy-z-image-turbo"
TYPE = "z-image"  # speak the offloader seam's vocabulary (same pipeline family as z-image-turbo)

PIPELINE    = "diffusers:ZImagePipeline"
TRANSFORMER = "diffusers:ZImageTransformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image",)
CFG   = ("guidance_scale", 0.0)

INFERENCE = 8

# This engine's registry dir inside the install: engine/comfy-z-image-turbo/ (where resolve_model
# sends a COMFY engine). It carries the scaffolding + engine.json, NOT the big checkpoints (those
# stay in the ComfyUI install). No HF "revision" -> check uses the COMFY presence test.
MODEL = {"model": "comfy-z-image-turbo"}

# ComfyUI split single files reused by this engine. `path` is the file's location under ComfyUI's
# models/ (and, under split_files/, its in-repo path too), `role` the diffusers component it feeds.
# Fetched only when missing, from ComfyUI's own published repo -- same {repository, revision} shape
# as the LORAS lists.
COMFY = {
    "repository": "Comfy-Org/z_image_turbo",
    "revision": "main",
    "components": [
        {"role": "transformer",  "path": "diffusion_models/z_image_turbo_bf16.safetensors"},
        {"role": "text_encoder", "path": "text_encoders/qwen_3_4b.safetensors"},
        {"role": "vae",          "path": "vae/ae.safetensors"},
    ],
}

# The tiny diffusers scaffolding (configs + tokenizer + scheduler, a few MB, NO weights) is fetched
# from the diffusers repo so the single files can be loaded/meta-loaded offline. Revision pinned to
# match the z-image-turbo engine for reproducibility.
SCAFFOLD = {
    "repository": "Tongyi-MAI",
    "model": "Z-Image-Turbo",
    "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    # config/tokenizer/scheduler files only -- all the pipeline needs except the big weights.
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
    root-relative `path` (e.g. models/vae/ae.safetensors) against `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _build_text_encoder(scaffold, weight_file, dtype):
    """Instantiate the Qwen3 text encoder from the scaffold config and load ComfyUI's single-file
    weights. ComfyUI prefixes every tensor with 'model.'; the bare transformers Qwen3Model does
    not, so strip it (398 keys, identical shapes -- verified)."""
    import safetensors.torch as safetensors_torch
    from transformers import AutoConfig, Qwen3Model

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    model = Qwen3Model(cfg)

    state = safetensors_torch.load_file(weight_file)
    state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)

    return model.to(dtype).eval()


def _transformer_meta(scaffold, dtype):
    """Meta-build (no weight RAM) the ZImage transformer from the scaffold config, for the
    offloader to stream the ComfyUI single file into."""
    from accelerate import init_empty_weights
    from diffusers import ZImageTransformer2DModel

    cfg = ZImageTransformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        return ZImageTransformer2DModel.from_config(cfg).to(dtype)


def _strip_model(sd):
    """ComfyUI prefixes every text-encoder key with 'model.'; strip it so the bare Qwen3Model binds
    (the single-file key remap handed to the offloader stream)."""
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}


def _text_encoder_meta(scaffold, dtype):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3Model

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        return Qwen3Model(cfg).to(dtype)


def load(ctx, params):
    """Build a ZImagePipeline from ComfyUI's single files. The offload backend, when present, meta-
    loads the big models straight from those files (disk-stream); otherwise every component is
    built here and placed via the shared ctx helpers."""
    from diffusers import (ZImagePipeline, ZImageTransformer2DModel, AutoencoderKL,
                           FlowMatchEulerDiscreteScheduler)
    from transformers import AutoTokenizer

    scaffold = ctx.model  # engine/comfy-z-image-turbo/ (scaffolding + engine.json)
    files    = _by_role(scaffold)

    # Small resident components come from the scaffold; the comfy VAE is reused (from_single_file).
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))
    vae = AutoencoderKL.from_single_file(files["vae"], config=scaffold, subfolder="vae",
                                         torch_dtype=ctx.dtype, local_files_only=True)

    # Disk-stream path: hand the backend model-agnostic specs -- the two big models' meta-builders
    # + single-file paths + key remaps (transformer via the diffusers converter, TE via `model.`
    # strip) -- plus the pre-built small components. All ZImage specifics stay in this engine file.
    if ctx.backend is not None:
        from diffusers.loaders.single_file_utils import (
            convert_z_image_transformer_checkpoint_to_diffusers as convert_transformer)

        return ctx.backend.load_pipe_comfy(
            ZImagePipeline,
            {"meta": lambda d: _transformer_meta(scaffold, d), "file": files["transformer"],
             "convert": convert_transformer},
            {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
             "convert": _strip_model},
            {"scheduler": scheduler, "tokenizer": tokenizer, "vae": vae},
            ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

    # Native path: materialise each component, then apply LoRAs + placement via the shared helpers.
    text_encoder = _build_text_encoder(scaffold, files["text_encoder"], ctx.dtype)

    transformer = ZImageTransformer2DModel.from_single_file(
        files["transformer"], config=scaffold, subfolder="transformer",
        torch_dtype=ctx.dtype, local_files_only=True)

    p = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                       tokenizer=tokenizer, transformer=transformer)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)
