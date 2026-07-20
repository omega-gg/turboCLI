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

# comfy-krea2-turbo -- text-to-image on Krea 2 Turbo, REUSING a ComfyUI install.
#
# Krea 2 Turbo is a distilled single-stream MMDiT (~12B): few-step (8), CFG-free (guidance 0). This
# engine builds it from ComfyUI's split single files, both SCALED-FP8 and streamed through the
# vendored comfy quant path -- kept fp8, dequantized per forward as ComfyUI does (emulated on a GPU
# without fp8 tensor cores): the ~13GB Krea2 transformer and the ~9GB Qwen3-VL-4B text encoder. The
# ComfyUI VAE is reused too: it is the same AutoencoderKLQwenImage (WAN-derived) as comfy-qwen, so
# its keys convert via diffusers' convert_wan_vae_to_diffusers. Only the tokenizer, scheduler +
# configs (~10MB, incl. vae/config.json + the 12-layer text-encoder tap) come from the diffusers
# repo.
#
# Unlike comfy-qwen-image-edit-2511 (bf16 transformer, fp8 TE), BOTH big models here are fp8, so
# both specs carry "quant": True and go through load_quant_single_file. The transformer's keys are
# ComfyUI-native (first/blocks.N/txtfusion/tproj/last), not diffusers layout, so
# _transformer_convert remaps them (validated 1:1, 430/430).
#
# OFFLOADER-ONLY: the fp8 quant path needs comfy ops, so load() bails out for other offload modes.
# All model-specific assembly lives here; the offloader (backend/) stays model-agnostic -- load()
# hands it data specs (meta-builders + single-file paths + key remaps) via load_pipe_comfy.
#
# install:  runner.install --comfy <ComfyUI dir> reuses the three single files (downloading only
#           the missing ones from their Comfy-Org repos) + the scaffold into engine/<id>.
# check:    runner.check treats a COMFY engine installed when engine.json + its files + scaffold.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level -- discovery stays
# cheap. The heavy imports live inside load() and the meta-builders.

import os
import re
import json

ID   = "comfy-krea2-turbo"
TYPE = "krea2"  # offloader seam vocabulary (single-stream Krea2 MMDiT family)

PIPELINE    = "diffusers:Krea2Pipeline"
TRANSFORMER = "diffusers:Krea2Transformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image",)
CFG   = ("guidance_scale", 0.0)  # distilled / CFG-free

INFERENCE = 8

# ComfyUI split single files reused by this engine. The transformer + text encoder live in
# Comfy-Org/Krea-2 at plain paths (not split_files/), so each sets `filename`; the VAE is the same
# file comfy-qwen reuses (Comfy-Org/Qwen-Image_ComfyUI, split_files/ layout -> filename default).
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "Comfy-Org/Krea-2",
         "path": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
         "filename": "diffusion_models/krea2_turbo_fp8_scaled.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/Krea-2",
         "path": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
         "filename": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/Qwen-Image_ComfyUI",
         "path": "vae/qwen_image_vae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the Krea repo: everything EXCEPT the big weights -- i.e.
# the Qwen2Tokenizer, scheduler, and the configs used to meta-build the three reused models
# (vae/config.json only, not the VAE weights -- those are reused from ComfyUI). ~10MB. Text-only
# (no processor). Revision pinned.
SCAFFOLD = {
    "repository": "krea",
    "model": "Krea-2-Turbo",
    "revision": "1161245028ef398cd0a951101b2bbf486464f841",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "vae/config.json",
        "transformer/config.json",
        "text_encoder/config.json",
    ],
}


def _by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file. Reads the engine.json comfy record
    (written by install: {root, external, components, scaffold}) and resolves each component's
    root-relative `path` (e.g. models/vae/qwen_image_vae.safetensors) against `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _transformer_meta(scaffold, dtype):
    """Meta-build (no weight RAM) the Krea2 transformer from the scaffold config, for the offloader
    to load the ComfyUI scaled-fp8 single file into (as QuantizedTensors, after
    _transformer_convert remaps its ComfyUI-native keys)."""
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel

    cfg = Krea2Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        return Krea2Transformer2DModel.from_config(cfg).to(dtype)


def _text_encoder_meta(scaffold, dtype):
    """Meta-build the multimodal Qwen3-VL-4B text encoder (vision tower + language) from scaffold
    config, for the offloader to load the ComfyUI scaled-fp8 file into (as QuantizedTensors)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLModel

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        return Qwen3VLModel(cfg)


def _build_vae(scaffold, weight_file, dtype):
    """Reuse ComfyUI's VAE: build AutoencoderKLQwenImage from the scaffold config and load the
    comfy single file, converting its WAN-style keys to the diffusers layout with the stock
    convert_wan_vae_to_diffusers (Krea 2 reuses Qwen-Image's WAN-derived VAE -- same file)."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    cfg = AutoencoderKLQwenImage.load_config(os.path.join(scaffold, "vae"))
    vae = AutoencoderKLQwenImage.from_config(cfg)

    state = convert_wan_vae_to_diffusers(safetensors_torch.load_file(weight_file))
    vae.load_state_dict(state, strict=True)

    return vae.to(dtype).eval()


# ComfyUI's Krea2 transformer keys -> diffusers Krea2Transformer2DModel layout. Per-block renames
# (attention q/k/v/o + qk-norm, MLP, the two RMSNorms, the modulation table) + the standalone
# in/out/time/text-fusion projections. Krea2RMSNorm uses (1 + weight), so ComfyUI's zero-centered
# .scale maps straight to diffusers .weight -- no value math. Validated 1:1 (430/430, 0 unexp).
_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_FF   = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}


def _block_key(prefix, rest):
    """Rename a within-block ComfyUI tail (rest) to its diffusers name under prefix
    (transformer_blocks.N or text_fusion.<block>.N). Returns None if unmapped."""
    m = re.match(r"attn\.(wq|wk|wv|wo|gate)\.(.+)$", rest)
    if m:
        return "%s.attn.%s.%s" % (prefix, _ATTN[m.group(1)], m.group(2))
    m = re.match(r"attn\.qknorm\.(qnorm|knorm)\.scale$", rest)
    if m:
        return "%s.attn.%s.weight" % (prefix, "norm_q" if m.group(1) == "qnorm" else "norm_k")
    m = re.match(r"mlp\.(gate|up|down)\.(.+)$", rest)
    if m:
        return "%s.%s.%s" % (prefix, _FF[m.group(1)], m.group(2))
    if rest == "prenorm.scale":
        return "%s.norm1.weight" % prefix
    if rest == "postnorm.scale":
        return "%s.norm2.weight" % prefix
    if rest == "mod.lin":
        return "%s.scale_shift_table" % prefix  # (6*H,) -> reshape (6, H)
    return None


_STANDALONE = {
    "first.weight": "img_in.weight", "first.bias": "img_in.bias",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",  # already (2, H)
    "tmlp.0.weight": "time_embed.linear_1.weight", "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight", "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight", "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight", "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight", "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
}


def _module_key(key):
    """Map one ComfyUI base key (no fp8 companion suffix) to its diffusers name, or None."""
    m = re.match(r"blocks\.(\d+)\.(.+)$", key)
    if m:
        return _block_key("transformer_blocks.%s" % m.group(1), m.group(2))
    m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", key)
    if m:
        return _block_key("text_fusion.%s.%s" % (m.group(1), m.group(2)), m.group(3))
    return _STANDALONE.get(key)


def _lora_keys(scaffold):
    """LoRA key map, copied from ComfyUI's model_lora_keys_unet Krea2 branch (comfy/lora.py) on
    comfy.utils.krea2_to_diffusers -- with `to` flipped to the diffusers key, since our streamed
    model IS the diffusers layout (upstream's is comfy-native). Covers every published naming:
    diffusers (transformer. / bare), ComfyUI-native (diffusion_model.blocks...), lycoris
    underscores. Runs inside the backend (load_pipe_comfy), where the `comfy` alias is live."""
    import comfy.utils

    with open(os.path.join(scaffold, "transformer", "config.json")) as f:
        layers = json.load(f)["num_layers"]

    diffusers_keys = comfy.utils.krea2_to_diffusers({"layers": layers})

    key_map = {}
    for k in diffusers_keys:
        if k.endswith(".weight"):
            to = k  # our model key: the diffusers name itself
            key_lora = k[:-len(".weight")]
            key_map["diffusion_model.{}".format(key_lora)] = to
            key_map["transformer.{}".format(key_lora)] = to
            key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to
            key_map[key_lora] = to
            # ComfyUI-native naming (what comfy-trained Krea2 LoRAs publish, incl. its
            # diffusion_model. prefix): krea2_to_diffusers' value side, bare.
            key_map[diffusers_keys[k][:-len(".weight")]] = to
            key_map["diffusion_model.{}".format(diffusers_keys[k][:-len(".weight")])] = to
    return key_map


def _transformer_convert(sd):
    """ComfyUI Krea2 transformer state dict -> diffusers layout. Renames each base key and carries
    its fp8 companions (.weight_scale/.comfy_quant, injected upstream by convert_old_quants) to the
    renamed module; reshapes the per-block modulation vector (6*H,) to the (6, H) scale_shift_table
    (final_layer's is already 2-D)."""
    out = {}
    for k in [k for k in sd if not k.endswith((".weight_scale", ".comfy_quant"))]:
        nk = _module_key(k)
        if nk is None:
            out[k] = sd[k]  # unmapped -> reported as unexpected by the offloader
            continue
        v = sd[k]
        if nk.endswith("scale_shift_table") and v.dim() == 1:
            v = v.reshape(6, -1)
        out[nk] = v
        base, nbase = k.rsplit(".", 1)[0], nk.rsplit(".", 1)[0]
        for comp in (".weight_scale", ".comfy_quant"):
            if base + comp in sd:
                out[nbase + comp] = sd[base + comp]
    return out


def _te_convert(sd):
    """ComfyUI's Qwen3-VL file prefixes every key with `model.`; the transformers Qwen3VLModel is
    `visual.*` (vision tower) + `language_model.*`. Strip/redirect that prefix, carrying the fp8
    companion suffixes untouched -- verified 0 unexpected (the only module gaps are rotary inv_freq
    buffers, recomputed at load)."""
    def rename(k):
        if k.startswith("model.visual."):
            return k[len("model."):]
        if k.startswith("model."):
            return "language_model." + k[len("model."):]
        return k

    return {rename(k): v for k, v in sd.items()}


def load(ctx, params):
    """Assemble a Krea2Pipeline from ComfyUI's single files via the disk-stream offloader.
    OFFLOADER-ONLY: both big models are scaled-fp8 and need the comfy quant path, so bail out for
    other offload modes. Build the small resident components (VAE/tokenizer/scheduler) from the
    scaffold here, then hand the backend model-agnostic specs for the two streamed fp8 models."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = _by_role(scaffold)

    if ctx.backend is None:
        engine_id = os.path.basename(os.path.normpath(scaffold))  # engine/<id> -> <id>
        raise RuntimeError(
            "%s requires an offload backend (offload=offloader); both big models are fp8 and need "
            "the comfy quant path" % engine_id)

    from diffusers import Krea2Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2Tokenizer

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))
    vae = _build_vae(scaffold, files["vae"], ctx.dtype)  # reused from ComfyUI (WAN key convert)

    # The 12 decoder layers Krea2Pipeline taps from the text encoder + the distilled/patch config
    # -- read straight from the scaffold's model_index.json so the tap stays pinned to the weights.
    with open(os.path.join(scaffold, "model_index.json")) as f:
        index = json.load(f)

    return ctx.backend.load_pipe_comfy(
        Krea2Pipeline,
        {"meta": lambda d: _transformer_meta(scaffold, d), "file": files["transformer"],
         "convert": _transformer_convert, "quant": True,
         "lora_keys": lambda: _lora_keys(scaffold)},
        {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
         "convert": _te_convert, "quant": True},
        {"scheduler": scheduler, "tokenizer": tokenizer, "vae": vae,
         "text_encoder_select_layers": index["text_encoder_select_layers"],
         "is_distilled": index["is_distilled"], "patch_size": index["patch_size"]},
        ctx.dtype, device=ctx.device, lora_files=list(ctx.loras) or None)
