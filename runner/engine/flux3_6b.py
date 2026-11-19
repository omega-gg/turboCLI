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

# flux3-6b engine: text2img + img2img on FLUX.3 Image Klein (5.6B, the 16-step distilled
# flow-map checkpoint, no CFG) from BFL's diffusers repo, its weights as published (bf16), run by
# diffusers' own models with or without the offloader. comfy-flux3-6b is the same model from BFL's
# int8 ComfyUI files; it builds on this engine's pipe.
#
# GATED: BFL's repo needs a Hugging Face token from an account that accepted the FLUX Kommunity
# License (install.sh takes it as its last argument).
#
# diffusers ships FLUX.3 as a modular pipeline only, so Flux3Pipe below gives it the classic face
# the runner and the offloader speak: from_pretrained, an encode_prompt the offloader wraps (its
# encode bridge, node boundary and encode cache), a progress_bar the runner hooks, a step callback
# for cancel, LoRAs, device placement, and a result with .images.

ID   = "flux3-6b"
TYPE = "flux3"

PIPELINE    = "diffusers:Flux3DistilledModularPipeline"  # behind Flux3Pipe
TRANSFORMER = "diffusers:Flux3Transformer2DModel"        # offloader disk-stream meta-load

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 0.0)              # distilled / CFG-free (Flux3Pipe ignores it)

# The flow-map checkpoint takes no guidance: the denoise passes it the timestep each step jumps
# to instead.
INFERENCE = 16

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

# The diffusers layout only: the repo also carries BFL's own copies of the transformer and VAE at
# its root, and fp8 / nvfp4 variants, which this engine does not read.
MODEL = {
    "repository": "black-forest-labs",
    "model": "flux-3-image-klein",
    "revision": "337337621b8526e120f2275d250242f4a6bedd75",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer/*",
        "text_encoder/*",
        "vae/*",
    ],
}


def _pipeline_class():
    """Flux3Pipe: diffusers' modular FLUX.3 distilled pipeline behind the classic pipeline face.
    The text encoder runs in encode_prompt, the rest of the modular blocks take the embeddings.
    Built here so diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    from diffusers import AutoencoderKLFlux2, DiffusionPipeline, Flux3Transformer2DModel
    from diffusers.loaders import Flux3LoraLoaderMixin
    from diffusers.modular_pipelines.flux3 import Flux3DistilledAutoBlocks
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    class Stop(Exception):
        """A cancel or a newer request, raised out of the denoise loop."""

    class Flux3Pipe(DiffusionPipeline, Flux3LoraLoaderMixin):
        model_cpu_offload_seq = "text_encoder->transformer->vae"

        def __init__(self, transformer, text_encoder, tokenizer, vae):
            super().__init__()

            self.register_modules(transformer=transformer, text_encoder=text_encoder,
                                  tokenizer=tokenizer, vae=vae)

            self._interrupt = False

            self._callback = None
            self._step = 0

            # The prompt is encoded as given: no prompt upsampler (diffusers 0.43's first step).
            blocks = Flux3DistilledAutoBlocks()
            blocks.sub_blocks.pop("prompt_upsampler")
            blocks.sub_blocks.pop("text_encoder")

            # NOTE: Each workflow's denoise loop opens its bar on itself: it goes through ours,
            #       which the runner hooks for its step lines, and each update runs the step
            #       callback.
            for workflow in blocks.sub_blocks["denoise"].sub_blocks.values():
                workflow.sub_blocks["denoise"].progress_bar = lambda **kwargs: self._bar(**kwargs)

            self._pipe = blocks.init_pipeline()
            self._pipe.update_components(transformer=transformer, vae=vae)

            # NOTE: The modular pipeline reads its device from where its modules sit, which an
            #       offload decides; it follows ours instead.
            outer = self
            cls = type(self._pipe)
            self._pipe.__class__ = type(cls.__name__, (cls,), {
                "_execution_device": property(lambda _: outer._execution_device)})

        @classmethod
        def from_pretrained(cls, model, transformer=None, torch_dtype=None,
                            local_files_only=True, **kwargs):
            """The repo's components (a streamed `transformer` may come in place of its own)."""
            def diffusers_part(part_cls, subfolder):
                return part_cls.from_pretrained(model, subfolder=subfolder,
                                                torch_dtype=torch_dtype,
                                                local_files_only=local_files_only, **kwargs)

            if transformer is None:
                transformer = diffusers_part(Flux3Transformer2DModel, "transformer")

            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", dtype=torch_dtype,
                local_files_only=local_files_only)
            tokenizer = Qwen3VLProcessor.from_pretrained(model, subfolder="text_encoder",
                                                         local_files_only=local_files_only)

            return cls(transformer, text_encoder, tokenizer,
                       diffusers_part(AutoencoderKLFlux2, "vae"))

        def _bar(self, total=None):
            bar = self.progress_bar(total=total)
            update = bar.update

            def stepped(n=1):
                result = update(n)

                if self._callback is not None:
                    self._callback(self, self._step, None, {})

                self._step += 1

                if self._interrupt:
                    raise Stop()

                return result

            bar.update = stepped

            return bar

        def encode_prompt(self, prompt, device=None):
            return Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                self.text_encoder, self.tokenizer, prompt,
                device=device or self._execution_device)

        def __call__(self, prompt, image=None, height=None, width=None,
                     num_inference_steps=INFERENCE, generator=None, callback_on_step_end=None,
                     **_):
            self._callback = callback_on_step_end
            self._step = 0

            prompt_embeds = self.encode_prompt(prompt, device=self._execution_device)

            # References go at their own size, unlike ComfyUI's template (ReferenceMultiLatent,
            # resize 1.0, ~1 MP): the pipeline only shrinks them past 4096 x 4096 and crops them
            # to the 16-pixel grid, so a reference at the output's size lines up with it.
            try:
                images = self._pipe(prompt_embeds=prompt_embeds, image=image, height=height,
                                    width=width, num_inference_steps=num_inference_steps,
                                    generator=generator, output="images")
            except Stop:
                images = []

            return SimpleNamespace(images=images)

    return Flux3Pipe


def load(ctx, params):
    """Flux3Pipe from the repo: through the offloader when one is chosen (load_pipe, as the other
    stock engines), else placed by the native offload modes."""
    pipeline_cls = _pipeline_class()

    if ctx.backend is not None:
        from diffusers import Flux3Transformer2DModel

        return ctx.backend.load_pipe(ctx.model, ctx.dtype, pipeline_cls, Flux3Transformer2DModel,
                                     device=ctx.device, lora_files=ctx.loras or None)

    p = pipeline_cls.from_pretrained(ctx.model, torch_dtype=ctx.dtype, use_safetensors=True,
                                     low_cpu_mem_usage=True)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)
