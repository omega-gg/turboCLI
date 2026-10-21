# Plan: comfy-krea2-turbo-edit-identity, Krea 2 Turbo identity edits (turboCLI)

## Context

conradlocke's Krea 2 Identity Edit V2 LoRA restages a person from one reference photo into new
scenes, poses, outfits and lighting (huggingface.co/conradlocke/krea2-identity-edit). It is not a
plain LoRA: it was trained with "dual conditioning", which ComfyUI's stock nodes do not provide
and which the `ComfyUI-Krea2Edit` node pack (lbouaraba, Apache-2.0, v2.0.0, one `__init__.py`)
supplies. `comfy-krea2-turbo` is text-to-image only and diffusers' `Krea2Pipeline` takes no image,
so the LoRA does nothing useful there today.

The goal is an image-to-image engine, `comfy-krea2-turbo-edit-identity`: the identity LoRA plus
that conditioning, at least as fast as ComfyUI with the node pack, in simple code. turbopixel
picks it up by itself (it lists engines from `check-model ENGINES:<mode>`, which reads `MODES`).
A first attempt (2026-07-16, V1 LoRA) was built and reverted unshipped; its verified facts are
reused here.

## What the conditioning is (node pack v2.0.0, verified in its source)

Defaults of the identity presets: Turbo, 8 steps, CFG 1 (no CFG), LoRA strength 1.0,
`grounding_px` 768, `ref_boost` 2 on the subject, placement `aligned (trained)`, `fit` geometry.

1. **Appearance, in-context VAE tokens.** Each reference (1 or 2: scene first, subject last) is
   fitted to the target's pixel size (`_fit_encode_image`, `fit`: within 8% of the target's
   ratio, center crop and fill; else fit inside, sides floored to 16, pre-cropped so the scale is
   exact; bicubic + antialias), VAE-encoded, normalized like the target latents, patchified
   (p=2).
   - Sequence `[text | ref_1 | ref_2 | target]`, output sliced to the target tokens.
   - Position ids `(frame, h, w)`: text 0; target `(0, h, w)`; ref i `(i, h + off_h, w + off_w)`
     with the fractional centered offset `max(0, (th - gh) / 2)` of a fitted ref.
   - Refs are clean (never noised) and share the target's timestep modulation (one `tvec`).
2. **Semantics, image-grounded Qwen3-VL.** The stock Krea2 template with one
   `<|vision_start|><|image_pad|><|vision_end|>` per reference at the start of the user turn,
   before the prompt. The vision encoder sees the original image, its longest side capped at
   `grounding_px` (downscale only), then Qwen's smart resize (factor 32). Same 12 tapped layers;
   the prefix is dropped up to and including `<|im_start|>user\n` (34 tokens), the vision tokens
   kept.
3. **`ref_boost`.** An additive `log(boost)` on the attention logits of target queries to that
   reference's keys, in the 28 main blocks only (not text fusion); boosts `[1, ..., ref_boost]`
   (the last reference is the boosted one).

## Design

### Why it fits diffusers without rewriting the model

`Krea2Transformer2DModel.forward` (`diffusers/models/transformers/transformer_krea2.py:457-527`)
builds `[text, hidden_states]`, takes one `(S, 3)` `position_ids`, one timestep for every token,
and passes `attention_mask` to the 28 `transformer_blocks` only (text fusion gets its own). So:
references prepended to `hidden_states` with their own ids, a float bias as that block mask, and
the output sliced past the references give exactly the node pack's model, with no forward
rewrite and the offloader's streaming, fused rope and LoRA on-cast untouched.

### Files

**`runner/engine/comfy_krea2_turbo_edit_identity.py`** (new, `BASE = comfy-krea2-turbo`), self
contained: the engine's constants, its `load()` and the pipeline it runs, in one file.

```python
ID   = "comfy-krea2-turbo-edit-identity"
BASE = base.ID

MODES = ("image-to-image",)

GATED = True  # the base's scaffold repo; a variant does not inherit GATED

LORA = "krea2_identity_edit_v2.safetensors"

REF_BOOST      = 2.0  # the identity presets: 2 on the subject
GROUNDING_PX   = 768  # the longest side the vision encoder sees
MAX_REFERENCES = 2    # scene, then subject

COMFY = dict(base.COMFY, components=base.COMFY["components"] + [
    {"role": "lora", "repository": "conradlocke/krea2-identity-edit",
     "revision": "f33b26291c640e8ec50b0da420bfee3c9a660daa",
     "filename": LORA, "path": "loras/" + LORA},
])

def load(ctx, params):
    ctx.loras = [(base._by_role(ctx.model)["lora"], 1.0)] + list(ctx.loras)

    return base.load(ctx, params, _pipeline_class())

def _pipeline_class():
    # diffusers is imported here rather than at the top, so discovery stays torch free
    from diffusers import Krea2Pipeline

    class Krea2EditPipe(Krea2Pipeline):
        ...  # below

    return Krea2EditPipe
```

The LoRA (1.83 GB, rank 256, ComfyUI-native `diffusion_model.blocks.N.*` lora_A/B + alpha) maps
through the base's `_lora_keys`, once its `wo` mapping is fixed (Results).

**`runner/engine/comfy_krea2_turbo.py`**: `load(ctx, params, pipeline_cls=None)` hands
`pipeline_cls or Krea2Pipeline` to `load_pipe_comfy`. One line; the base is otherwise unchanged.

Nothing goes to the generic turboCLI part (core, offloader): the conditioning, the grounded
encode and the reference tokens are Krea2 identity edit alone. The base's `pipeline_cls` is the
one change outside the file, and it has to happen at load: the offloader wraps the pipe's
`encode_prompt` while it builds it (`_finalize_pipe`), so a class swapped after `load()` would
have its grounded encode shadowed by those wrappers.

**`Krea2EditPipe(Krea2Pipeline)`**, in the engine file (`_pipeline_class`), its helpers module
level functions there too (fit geometry, position ids, bias), torch imported inside them:

- `__call__(self, prompt, image=None, height, width, **kwargs)`: the first `MAX_REFERENCES`
  images; encodes the prompt grounded (below) and the references, installs the reference
  forward for this call, then `super().__call__(prompt_embeds=..., prompt_embeds_mask=...,
  height=..., width=..., **kwargs)`, so the denoise loop, schedule, decode, progress and cancel
  stay the stock pipeline's.
- `encode_prompt(self, prompt, image=None, device=None, ...)`: the grounded encode, returning the
  stock `(embeds, mask)` 2-tuple, so the offloader's unpadded encode and encode cache (which
  hashes PIL images) keep working. The `Qwen3VLProcessor` is built in code (the Krea scaffold has
  none; its tokenizer has the vision tokens): `Qwen2VLImageProcessor` (patch 16, temporal 2,
  merge 2, mean/std 0.5, the Qwen3-VL size bounds) plus the required `Qwen3VLVideoProcessor`. The
  text encoder is the full `Qwen3VLModel` with its vision tower (`_te_convert` keeps `visual.*`),
  called with `pixel_values`, `image_grid_thw` and `mm_token_type_ids`, so its own
  `get_rope_index` places the image tokens. The prefix is cut at the stock prefix's token
  length (through `<|im_start|>user\n`), rather than a fixed 34.
- `_references(self, images, height, width)`: the `fit` geometry, `self.vae.encode` (the
  offloader's ComfyVAE, cached), normalized with the pipeline's own latents mean/std and packed
  with `_pack_latents`; their position ids (frame i, fitted offset); and the `ref_boost` bias
  `(1, 1, L, L)`, `log(REF_BOOST)` on target rows times the subject's columns, built once per
  call.
- The reference forward: for the call only, `transformer.forward` is wrapped so it prepends the
  reference tokens to `hidden_states` and their ids to `position_ids`, passes the bias as the
  block mask, and returns the target slice; restored in a `finally`. A wrapper rather than a
  re-classed transformer, which would move `__module__` and silently drop the offloader's fused
  rope (`adapter.use_kitchen_rope`).

### Lists and docs

- `bash/turbo/image-to-image.sh` usage list, `bash/turbo/install.sh` (gated tag),
  `bash/turbo/README.md` (install and image-to-image lists).
- `engine.md`: a row in the image editing table: identity edits from one or two references,
  ~21 GB, 8 steps, gated, offloader, LoRA free for personal and non-commercial use.
- `implementation.md`: the engine row (BASE, the LoRA component, its `Krea2EditPipe` and how it
  maps the node pack).
- This plan copied to `doc/comfy-krea2-turbo-edit-identity-plan.md`, without local paths.

## Critical files

- `runner/engine/comfy_krea2_turbo_edit_identity.py` (new, everything the engine needs),
  `runner/engine/comfy_krea2_turbo.py` (`pipeline_cls`)
- `bash/turbo/image-to-image.sh`, `bash/turbo/install.sh`, `bash/turbo/README.md`, `engine.md`,
  `implementation.md`, new `doc/comfy-krea2-turbo-edit-identity-plan.md`

## Verification

Small sizes first (512x512), 1024x1024 once it holds; seed 7; one person photo, a prompt that
restages them ("the same man sitting at a cafe terrace at night, neon light").

1. **Install** through `install.sh` (the LoRA lands in ComfyUI's `models/loras`), `check` reads it
   installed, `check-model ENGINES:image-to-image` lists it, gated.
2. **Grounded encode**: the drop lands on `<|im_start|>user\n` (34 tokens), the vision tokens
   kept; the LoRA reports its patches loaded (`offloader: applied N LoRA patches`, N > 0).
3. **ComfyUI parity**: the node pack v2.0.0 cloned into the user's ComfyUI `custom_nodes`, its
   identity workflow (Krea2 Turbo fp8, the same LoRA at 1.0, grounded encode at 768,
   `ref_boost` 2, aligned, fit, euler/simple 8 steps, CFG 1) at the same seed and inputs. Ours
   matches it (same noise path as the earlier Krea2 runs: same composition, small pixel
   difference), and the per-step time is ComfyUI's or better. A side-by-side grid saved for
   review.
4. Two references (scene + subject): runs, the subject's identity holds in the scene.
5. `comfy-krea2-turbo` text-to-image unchanged (the `pipeline_cls` default); a turbopixel Image
   to Image run with the new engine.
6. Style: 99 columns, no `--` or spaced ` - ` joins in added lines, American spelling.
7. No commit until the user says so, no attribution; the Sky-pro pin bump stays the user's.

## Results (2026-10-20)

- **Placement.** `REF_POSITION = "beside"` is the default: the subject's grid sits right of the
  target's (`w + target width`). `aligned`, the node pack's trained default, holds the source's
  pose and layout, so a restage ("lying on a bed") kept the person upright; it stays the choice
  for a face-locked portrait. The scene reference is always aligned.
- **Text encoder taps.** transformers 5 records Qwen3-VL's `hidden_states` before each layer's
  DeepStack add; ComfyUI (and transformers 4, which the LoRA was trained with) after it. The
  encode taps each layer's output at the next layer's input (forward pre-hooks).
- **LoRA key map (base).** krea2_to_diffusers lists `attn.to_out`, an alias whose weight is
  `to_out.0`'s; aimed at the name, it overwrote native `wo` and the offloader dropped every
  attention-output patch as unknown, while counting it applied. `_lora_keys` aims it at
  `to_out.0`: 256 patches bind, as in ComfyUI. Every Krea2 LoRA was affected (too red, too strong).
- **Parity** (512x512, seed 7, cafe restage): same composition as ComfyUI; mean color within
  1.1 levels (was +6 red), mean pixel difference 11.1 (was 17.2). Text-to-image with the LoRA:
  within 2 levels, 8.1 (was 22.5). Speed, 8 steps interleaved on a throttling laptop: without
  the LoRA even (53, 55 s against 45, 55 s); with it 70 to 86 s against 71 to 75 s, within the
  noise. The LoRA costs both about 20 s, ComfyUI's patches rebuilt as each weight streams.
- **Left as is.** The vision tower runs bf16 (the offloader's), ComfyUI's fp32: the image
  tokens differ slightly in the deeper taps, without a visible effect.
- **General edit.** `comfy-krea2-turbo-edit-general` builds on this engine with the General Edit V2
  LoRA (same repo and revision) at its presets, `ref_boost` 4 and the reference aligned:
  `load` takes `ref_boost` and `ref_position`, this engine's by default. The model card's
  showcase edits (a removal, a season change) match on Turbo. An edit regenerates the whole
  image, so it drifts from its input (about 3 px, 1 to 2 levels darker), as much as in
  ComfyUI; chained edits add it up.
