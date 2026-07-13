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

# comfy-qwen-image-edit-2511-lightning -- comfy-qwen-image-edit-2511 with the "lightning" 4-step
# LoRA always applied (mirrors the stock qwen-image-edit-2511-lightning, but every weight reused
# from a ComfyUI install). Same reuse as the base engine: the 39GB bf16 transformer is streamed via
# the offloader, the 8.7GB scaled-fp8 Qwen2.5-VL text encoder runs through the vendored comfy quant
# path, and the VAE is reused (WAN key convert). The only addition is one more reused ComfyUI file
# -- the lightning LoRA in models/loras/ -- applied to the transformer at full strength on load.
# Standalone by design (see the stock -lightning engine): no cross-engine imports.
#
# OFFLOADER-ONLY: the fp8 quant path needs comfy ops, so load() bails out for other offload modes.
# All model-specific assembly lives here; the offloader (backend/) stays model-agnostic -- load()
# hands it data specs (meta-builders + single-file paths + key remaps) via load_pipe_comfy.
#
# install:  runner.install --comfy <ComfyUI dir> reuses the four single files (downloading only the
#           missing ones, each from its own repo) + the scaffold into engine/<id>.
# check:    runner.check treats a COMFY engine installed when engine.json + its files + scaffold.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level -- discovery stays
# cheap. The heavy imports live inside load() and the meta-builders.

import os
import json

ID   = "comfy-qwen-image-edit-2511-lightning"
TYPE = "qwen-image-edit"  # offloader seam vocabulary (same family as qwen-image-edit-2511)

PIPELINE    = "diffusers:QwenImageEditPlusPipeline"
TRANSFORMER = "diffusers:QwenImageTransformer2DModel"  # offloader disk-stream meta-load

MODES = ("image-to-image",)
CFG   = ("true_cfg_scale", 1.0)

INFERENCE = 4  # the lightning LoRA's design step count (vs the base engine's 40)

# This engine's registry dir: engine/comfy-qwen-image-edit-2511-lightning/ (scaffold, engine.json).
# No HF "revision" -> check uses the COMFY presence test.
MODEL = {"model": "comfy-qwen-image-edit-2511-lightning"}

# The lightning LoRA reused from ComfyUI's models/loras/. It sits in a plain HF repo -- the file is
# at the repo root, not under split_files/ -- so this component carries an explicit `filename` (the
# in-repo path) distinct from its `path` (the models/-relative destination). Revision pins the same
# commit the stock qwen-image-edit-2511-lightning uses.
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# ComfyUI split single files reused by this engine: the same three as the base engine (transformer,
# fp8 text encoder, VAE -- each in its own Comfy-Org repo) plus the lightning LoRA. Each component
# carries its own `repository` (install downloads a missing one from there).
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
         "path": "diffusion_models/qwen_image_edit_2511_bf16.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
         "path": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/Qwen-Image_ComfyUI",
         "path": "vae/qwen_image_vae.safetensors"},
        {"role": "lora", "repository": "lightx2v/Qwen-Image-Edit-2511-Lightning",
         "revision": "d74eba145674fd7e31b949324e148e21e7118abd",
         "filename": LIGHTNING, "path": "loras/" + LIGHTNING},
    ],
}

# The diffusers scaffolding fetched from the Qwen repo: everything EXCEPT the big weights -- i.e.
# the tokenizer, Qwen2VLProcessor, scheduler, and the configs used to meta-build the three reused
# models (vae/config.json only, not the VAE weights -- those are reused from ComfyUI). ~10MB.
# Revision pinned to the stock engine's.
SCAFFOLD = {
    "repository": "Qwen",
    "model": "Qwen-Image-Edit-2511",
    "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "processor/*",
        "vae/config.json",
        "transformer/config.json",
        "text_encoder/config.json",
    ],
}


def _comfy(engine_dir):
    """Read engine.json -> the comfy record {root, external, components, scaffold}. Written by
    install; the source of truth for the reused ComfyUI install and the files under it."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        return json.load(f)["comfy"]


def _by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file, resolving the component's root-relative
    `path` (e.g. models/vae/qwen_image_vae.safetensors) against the record's `root`."""
    comfy = _comfy(engine_dir)
    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _transformer_meta(scaffold, dtype):
    """Meta-build (no weight RAM) the Qwen-Image transformer from the scaffold config, for the
    offloader to stream the ComfyUI bf16 single file into (keys already diffusers layout)."""
    from accelerate import init_empty_weights
    from diffusers import QwenImageTransformer2DModel

    cfg = QwenImageTransformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        return QwenImageTransformer2DModel.from_config(cfg).to(dtype)


def _text_encoder_meta(scaffold, dtype):
    """Meta-build the multimodal Qwen2.5-VL text encoder (vision tower + language) from scaffold
    config, for the offloader to load the ComfyUI scaled-fp8 file into (as QuantizedTensors)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        return Qwen2_5_VLForConditionalGeneration(cfg)


def _build_vae(scaffold, weight_file, dtype):
    """Reuse ComfyUI's VAE: build AutoencoderKLQwenImage from the scaffold config and load the
    comfy single file, converting its WAN-style keys to the diffusers layout with the stock
    convert_wan_vae_to_diffusers (Qwen-Image's VAE is WAN-derived -- verified 1:1, 194 keys)."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    cfg = AutoencoderKLQwenImage.load_config(os.path.join(scaffold, "vae"))
    vae = AutoencoderKLQwenImage.from_config(cfg)

    state = convert_wan_vae_to_diffusers(safetensors_torch.load_file(weight_file))
    vae.load_state_dict(state, strict=True)

    return vae.to(dtype).eval()


def _flat_to_nested(sd):
    """ComfyUI's file uses the flat on-disk key form (model.layers.*, visual.*); the transformers
    module is nested (model.language_model.*, model.visual.*). Rename with the same prefix rules
    transformers applies at load -- verified 1:1 (729/729 keys)."""
    nested = ("model.language_model.", "model.visual.")

    def rn(k):
        if k.startswith("visual."):
            return "model." + k
        if k.startswith("model.") and not k.startswith(nested):
            return "model.language_model." + k[len("model."):]
        return k

    return {rn(k): v for k, v in sd.items()}


def load(ctx, params):
    """Assemble a QwenImageEditPlusPipeline from ComfyUI's single files via the disk-stream
    offloader, with the lightning LoRA applied to the streamed transformer. OFFLOADER-ONLY: the fp8
    text-encoder quant path needs comfy ops, so bail out otherwise. Build the small resident
    components (VAE/tokenizer/processor/scheduler) from the scaffold here, then hand the backend
    model-agnostic specs for the two big streamed models + the lightning LoRA."""
    scaffold = ctx.model  # engine/comfy-qwen-image-edit-2511-lightning/ (scaffold + engine.json)
    files    = _by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError(
            "comfy-qwen-image-edit-2511-lightning requires an offload backend "
            "(offload=offloader); the fp8 text-encoder quant path needs comfy ops")

    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))
    processor = Qwen2VLProcessor.from_pretrained(os.path.join(scaffold, "processor"))
    vae = _build_vae(scaffold, files["vae"], ctx.dtype)  # reused from ComfyUI (WAN key convert)

    # The lightning LoRA (reused from ComfyUI's models/loras/) at full strength on the transformer,
    # before any user-supplied ctx.loras.
    lora_files = [(files["lora"], 1.0)] + list(ctx.loras)

    return ctx.backend.load_pipe_comfy(
        QwenImageEditPlusPipeline,
        {"meta": lambda d: _transformer_meta(scaffold, d), "file": files["transformer"],
         "convert": None},
        {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
         "convert": _flat_to_nested, "quant": True},
        {"scheduler": scheduler, "tokenizer": tokenizer, "processor": processor, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=lora_files or None)
