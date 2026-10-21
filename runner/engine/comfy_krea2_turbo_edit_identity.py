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
#   longest side capped at GROUNDING_PX.
# - REF_BOOST: log(REF_BOOST) on the attention of the target to the subject (the last reference),
#   in the main blocks.
#
# diffusers' Krea2 transformer takes all of that as it is: it sequences [text, hidden_states] with
# one position id per token and one timestep, and hands its mask to the main blocks alone. So the
# references are prepended to hidden_states, the boost is that mask and the output is sliced past
# them, for the call only; the denoise, the schedule and the decode are the stock pipeline's.
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
GROUNDING_PX   = 768  # the longest side the vision encoder sees
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
    keys, which the base's _lora_keys maps), any user --loras after it. An engine building on this
    one passes its own boost and placement."""
    ctx.loras = [(base._by_role(ctx.model)["lora"], 1.0)] + list(ctx.loras)

    return base.load(ctx, params, _pipeline_class(ref_boost, ref_position))


def _pipeline_class(ref_boost, ref_position):
    """Krea2Pipeline with references. Built here so diffusers stays out of engine discovery."""
    import math

    import numpy
    import torch
    import torch.nn.functional as F
    from diffusers import Krea2Pipeline
    from PIL import Image
    from transformers import Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vision = "<|vision_start|><|image_pad|><|vision_end|>"

    class Krea2EditPipe(Krea2Pipeline):
        def __call__(self, prompt=None, image=None, height=None, width=None, **kwargs):
            images = list(image or [])[:MAX_REFERENCES]

            if not images:
                return super().__call__(prompt=prompt, height=height, width=width, **kwargs)

            # Encoded through encode_prompt, so the offloader's cache keys it on the images too.
            # The stock call wants a mask beside the embeds, all of them valid here.
            embeds, _ = self.encode_prompt(prompt, image=images, device=self._execution_device)

            mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)

            tokens, ids, bias = self._references(images, height, width, embeds)

            transformer = self.transformer

            # The forward the module has now, its own or one the offloader set, put back after.
            previous = transformer.__dict__.get("forward")

            forward = transformer.forward

            def edit_forward(hidden_states, encoder_hidden_states, position_ids, **kw):
                count = encoder_hidden_states.shape[1]

                out = forward(
                    hidden_states=torch.cat([tokens.expand(len(hidden_states), -1, -1),
                                             hidden_states], dim=1),
                    encoder_hidden_states=encoder_hidden_states,
                    position_ids=torch.cat([position_ids[:count], ids, position_ids[count:]]),
                    **kw)

                if isinstance(out, tuple):
                    return (out[0][:, len(ids):],) + out[1:]

                out.sample = out.sample[:, len(ids):]

                return out

            def boost(block, args):
                # The main blocks take the mask fourth; every token here is valid, so the boost
                # is the whole mask.
                return args[:3] + (bias,) + args[4:]

            transformer.forward = edit_forward

            hooks = [b.register_forward_pre_hook(boost) for b in transformer.transformer_blocks]

            try:
                return super().__call__(prompt_embeds=embeds, prompt_embeds_mask=mask,
                                        height=height, width=width, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()

                if previous is None:
                    del transformer.forward
                else:
                    transformer.forward = previous

        def encode_prompt(self, prompt=None, image=None, device=None, num_images_per_prompt=1,
                          prompt_embeds=None, prompt_embeds_mask=None, max_sequence_length=512):
            """The stock encode, or with `image`, grounded: the stock template with the
            references in the user turn, before the prompt. The prefix is cut where the stock
            one is (its system turn and <|im_start|>user\\n), the vision tokens kept."""
            if prompt_embeds is not None or not image:
                return super().encode_prompt(prompt, device, num_images_per_prompt,
                                             prompt_embeds, prompt_embeds_mask,
                                             max_sequence_length)

            device = device or self._execution_device

            text = (self.prompt_template_encode_prefix + vision * len(image) + prompt
                    + self.prompt_template_encode_suffix)

            # Qwen3-VL's processor, which the Krea repo does not ship (its tokenizer has the
            # vision tokens); the video processor is required, never used. Built on first use:
            # a pipeline's components are read off its __init__.
            if "_processor" not in self.__dict__:
                self._processor = Qwen3VLProcessor(
                    image_processor=Qwen2VLImageProcessor(
                        patch_size=16, temporal_patch_size=2, merge_size=2,
                        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
                        size={"shortest_edge": 65536, "longest_edge": 16777216}),
                    tokenizer=self.tokenizer, video_processor=Qwen3VLVideoProcessor())

            inputs = self._processor(text=[text], images=[_grounding(i) for i in image],
                                     return_tensors="pt").to(device)

            # A tap is what enters that decoder layer: Qwen3-VL adds its DeepStack image features
            # after a layer, and transformers 5's hidden_states are taken before that add, where
            # ComfyUI (and the transformers the LoRA was trained on) take it after.
            taps, layers = {}, self.text_encoder.language_model.layers

            def tap(index):
                def hook(layer, args, kwargs):
                    taps[index] = args[0] if args else kwargs["hidden_states"]

                return hook

            hooks = [layers[i].register_forward_pre_hook(tap(i), with_kwargs=True)
                     for i in self.text_encoder_select_layers]

            try:
                self.text_encoder(**inputs)
            finally:
                for hook in hooks:
                    hook.remove()

            hidden = torch.stack([taps[i] for i in self.text_encoder_select_layers], dim=2)

            start = len(self.tokenizer(self.prompt_template_encode_prefix).input_ids)

            return hidden[:, start:], inputs.attention_mask[:, start:].bool()

        def _references(self, images, height, width, embeds):
            """The references' packed latents, their position ids and the boost mask."""
            device, dtype = embeds.device, embeds.dtype

            p = self.patch_size

            th, tw = height // (self.vae_scale_factor * p), width // (self.vae_scale_factor * p)

            mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
            std  = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1)

            tokens, ids, sizes = [], [], []

            for frame, image in enumerate(images, 1):
                pixels = _fit(image, height, width).to(device, dtype)

                latent = self.vae.encode(pixels.mul(2).sub(1)[:, :, None]).latent_dist.mode()
                latent = (latent[:, :, 0].float().cpu() - mean) / std

                c, h, w = latent.shape[1:]

                tokens.append(self._pack_latents(latent, 1, c, h, w))

                # The fitted reference centered on the target's grid, at fractional offsets, or
                # the subject right of it.
                gh, gw = h // p, w // p

                if frame == len(images) and ref_position == "beside":
                    left = tw
                else:
                    left = max(0.0, (tw - gw) / 2)

                grid = torch.zeros(gh, gw, 3)
                grid[..., 0] = frame
                grid[..., 1] = (torch.arange(gh) + max(0.0, (th - gh) / 2))[:, None]
                grid[..., 2] = (torch.arange(gw) + left)[None, :]

                ids.append(grid.reshape(-1, 3))

                sizes.append(gh * gw)

            tokens = torch.cat(tokens, dim=1).to(device, dtype)
            ids = torch.cat(ids).to(device)

            # log(ref_boost) where the target looks at the subject, over [text | refs | target]:
            # the target's rows, the last reference's columns.
            text = embeds.shape[1]

            size = text + len(ids) + th * tw

            bias = torch.zeros(1, 1, size, size, device=device, dtype=dtype)
            bias[..., text + len(ids):, text + len(ids) - sizes[-1]:text + len(ids)] = (
                math.log(ref_boost))

            return tokens, ids, bias

    def _grounding(image):
        """The reference for the vision encoder: its longest side capped at GROUNDING_PX."""
        image = image.convert("RGB")

        scale = GROUNDING_PX / max(image.size)

        if scale >= 1:
            return image

        return image.resize((round(image.width * scale), round(image.height * scale)),
                            Image.Resampling.BOX)

    def _fit(image, height, width):
        """The node pack's `fit`: a reference within 8% of the target's shape is center-cropped to
        fill it, any other fits inside on the 16 grid, pre-cropped so its scale is exact; bicubic,
        antialiased. Returns (1, 3, h, w) in [0, 1]."""
        x = torch.from_numpy(numpy.array(image.convert("RGB"))).permute(2, 0, 1)[None]
        x = x.float() / 255

        ih, iw = x.shape[-2:]

        sc = min(height / ih, width / iw)

        if ih * sc >= height * 0.92 and iw * sc >= width * 0.92:
            s = max(height / ih, width / iw)
            ch, cw = min(ih, int(round(height / s))), min(iw, int(round(width / s)))
            nh, nw = height, width
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, height // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, width // 16 * 16))
            ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))

        y0, x0 = (ih - ch) // 2, (iw - cw) // 2

        x = x[..., y0:y0 + ch, x0:x0 + cw]

        return F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True).clamp(0, 1)

    return Krea2EditPipe
