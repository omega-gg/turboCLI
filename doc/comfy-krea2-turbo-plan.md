# Add `comfy-krea2-turbo` engine — reuse ComfyUI's Krea 2 Turbo models

## Context

Krea 2 Turbo is Krea AI's distilled single-stream MMDiT (~12 B): few-step (8), CFG-free
(`guidance_scale=0.0`). A ComfyUI user already has the weights on disk as three split single
files, so — like `comfy-z-image-turbo` / `comfy-qwen-image-edit-2511` — the engine **reuses** them
instead of re-downloading the model, and generates through turboCLI's own diffusers
`Krea2Pipeline`.

Reused files (`ComfyUI/models/`):
- `diffusion_models/krea2_turbo_fp8_scaled.safetensors` — the transformer, **scaled-fp8** (~13 GB)
- `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` — Qwen3-VL-4B text encoder, **scaled-fp8**
  (~9 GB)
- `vae/qwen_image_vae.safetensors` — the same Qwen-Image (WAN-derived) VAE `comfy-qwen` reuses

**OFFLOADER-ONLY** (confirmed with the user): both big models are fp8, and the quant path needs
comfy ops. `load()` bails out for any other offload mode.

Unlike `comfy-qwen-image-edit-2511` (bf16 transformer + fp8 TE), **both** big models here are fp8,
so both specs carry `"quant": True`.

## What it took

### 1. diffusers upgrade
The installed build (0.39.0.dev0) had no Krea 2. Bumped the `diffusers` pin in
`bash/turbo/build.sh` to a commit carrying `Krea2Pipeline` / `Krea2Transformer2DModel` (only two
commits ever touch krea2: `7104cb43` adds it, `3993de59` the LoRA trainer — there is no later
upstream fix to wait for). Re-verified: all existing engine classes still import and a z-image
regression generation passed.

### 2. Offloader: fp8-scaled transformer support (`turbo-offloader`)
`load_pipe_comfy`'s transformer arm only did the bf16 `stream_single_file` path. Added a branch on
`transformer["quant"]` that mirrors the text-encoder quant arm, and renamed
`load_quant_text_encoder` → **`load_quant_single_file`** (it was never TE-specific). The loader
now passes the file's **metadata** to `convert_old_quants`, which is what injects the
`.comfy_quant` markers for krea2's **newer** fp8 convention (`_quantization_metadata` in the
safetensors header + bare `.weight_scale`) as opposed to the classic `scaled_fp8` +
`.scale_weight` keys. This is exactly ComfyUI's own call (`comfy/sd.py`:
`convert_old_quants(sd, model_prefix="", metadata=metadata)`).

### 3. The `(1 + weight)` RMSNorm bug — the real trap
With everything loading cleanly (0 unexpected keys, fp8 dequant numerically **exact**) the engine
still produced **pure noise**. Root cause was in the offloader, not the engine:

`comfy_ize` re-classes every custom `*RMSNorm` into comfy's fused `operations.RMSNorm` (a ~3.5×
speedup over the eager `mul`/`rsqrt`) — keyed off nothing but the class *name*. Comfy's fused norm
feeds the weight **verbatim** to `F.rms_norm`. But diffusers' `Krea2RMSNorm` uses the
**`(1 + weight)`** convention (weights stored zero-centered — ComfyUI's own krea2 `RMSNorm` does
`scale + 1.0` too). The re-class silently dropped the `+1`, so every norm in the model was wrong.

Existing engines were unaffected: their custom norms (`Qwen3RMSNorm`) use the plain `weight` form.

**Fix** (`adapter.py`): the re-class is now **proven, not assumed**. `_rmsnorm_matches_fused`
probes structurally before re-classing — hand the module a real random weight and compare its own
forward against the exact `F.rms_norm` call the re-class would make (this also guards the eps /
`normalized_shape` `_prep_rmsnorm` inferred). Only an exact match is re-classed; anything else —
another weight convention, a wrong eps, a probe that raises — **keeps its own forward**, which is
what ComfyUI does anyway (comfy hands a model `operations` for its `Linear`/`Conv` leaves; it
never swaps out the model's norm), and `keep_uncastable_resident` places it. So the re-class can
only ever be a no-op speedup, and no model name appears in the logic.

Keeping krea2's norms on their own forward costs nothing: that forward calls `F.rms_norm` too, so
with fp32 weights (see `keep_declared_fp32` below) it hits the same fused kernel.

### 4. The engine (`runner/engine/comfy_krea2_turbo.py`)
Standalone (reuses `_by_role` / `_build_vae` by copy; krea2 differs on transformer, TE and
pipeline, so `BASE` inheritance would override nearly everything).

- `ID="comfy-krea2-turbo"`, `TYPE="krea2"`, `PIPELINE="diffusers:Krea2Pipeline"`,
  `TRANSFORMER="diffusers:Krea2Transformer2DModel"`, `MODES=("text-to-image",)`,
  `CFG=("guidance_scale", 0.0)`, `INFERENCE=8`.
- `COMFY`: transformer + TE from `Comfy-Org/Krea-2` — that repo stores them at **plain** paths,
  not under `split_files/`, so each component sets `filename` explicitly (install's default is
  `split_files/<path>`). VAE from `Comfy-Org/Qwen-Image_ComfyUI` (split_files layout → default).
- `SCAFFOLD`: `krea/Krea-2-Turbo` (revision pinned) — `model_index.json`, `scheduler/*`,
  `tokenizer/*`, and the three `config.json`s. Text-only, so no `processor/*`.
- `load()` builds scheduler / `Qwen2Tokenizer` / VAE, then hands `load_pipe_comfy` the two fp8
  specs plus the Krea2 config kwargs read straight from the scaffold's `model_index.json`
  (`text_encoder_select_layers` = the 12 tapped decoder layers, `is_distilled`, `patch_size`) so
  the tap stays pinned to the weights.

**`_transformer_convert`** — the ComfyUI-native keys
(`first`/`blocks.N`/`txtfusion`/`tproj`/`last`) are not the diffusers layout and diffusers ships no
krea2 single-file converter, so the engine remaps them (validated 1:1, 430/430, and byte-checked
against the diffusers-native checkpoint: non-quant tensors match to bf16 rounding, fp8 Linears to
~2% = the expected fp8 error, **no permutation**). Notable: the per-block modulation `mod.lin` is a
flat `(6*H,)` vector → `(6, H)` `scale_shift_table`, while `last.modulation.lin` is already
`(2, H)`. `Krea2RMSNorm`'s `(1 + weight)` means ComfyUI's zero-centered `.scale` maps straight to
`.weight` — no value math.

**`_te_convert`** — ComfyUI's Qwen3-VL file prefixes every key with `model.`; `Qwen3VLModel` is
`visual.*` + `language_model.*`. A prefix rename (0 unexpected; the only gaps are rotary `inv_freq`
buffers, which transformers recomputes).

## Verification

- **Cross-framework parity.** ComfyUI's own `SingleStreamDiT` vs our diffusers transformer,
  identical inputs and weights — cosine **1.0** at every stage (`img_embed`, first norm,
  `text_fusion`, `block0`, final output). Before the RMSNorm fix the final output was **0.19**
  (first norm −0.84); the fp32-norm fix took the last 0.99997 to exactly **1.0**.
- **End-to-end.** 512² and 1024², 8 steps, cfg 0 — clean images matching the ComfyUI reference
  (`image_krea2_turbo_t2i` template: UNETLoader + CLIPLoader type `krea2` + euler/simple, cfg 1),
  bit-identical across reps.

## Benchmarks (A1000 4 GB, 512², 8 steps)

| | comfy-krea2-turbo | ComfyUI |
|---|---|---|
| load | ~30 s | ~2 s |
| **per-step** | **3.50 s/it** | 3.62 s/it |

**Per-step parity, confirmed in two independent thermal states** (this laptop throttles hard, so a
single reading proves nothing): cool — ours 3.50 vs ComfyUI 3.62; mid-session throttled — ours
13.33 vs ComfyUI 13.46. Marginally ahead in both.

Getting there took two fixes, both landed in the offloader as generic mechanisms
(`install_unpadded_encode`, `keep_declared_fp32` — see offloader.md):

- **The per-step gap was NOT diffusers-vs-comfy compute** (an earlier draft of this doc said so,
  and was wrong). It was **500 padding tokens**: diffusers pads krea2's text to a fixed 512
  (`padding="max_length"`), ComfyUI pads not at all — 1536 tokens through 28 blocks vs ~1054. Our
  own diagnostic had shown it all along: `prompt_embeds (1, 512, 12, 2560)` with `mask_sum=12`.
  Fixing it: **5.17 → 3.50 s/it**.
- The attention mask was a **symptom, not a cause** (~7 ms of a ~5 s step): it existed only to
  mask padding we invented. Measured, not assumed — FLASH is unavailable on Windows for *every*
  config including ComfyUI's, so both land on cuDNN regardless.
- Streaming was never the bottleneck: pinned H2D measures **13.0 GB/s**, and ComfyUI's own
  3.62 s/it implies just 3.56 GB/s. Both stacks are compute-bound.

The load gap is inherent to the reuse: we run the key convert (430 tensors), the quant setup and a
full VAE load + WAN convert, where ComfyUI just mmaps its native layout and streams lazily.

**Measure on a cool machine.** These figures move ~3.7× with thermal state: mid-session both
stacks degraded together (ComfyUI 3.62 → 13.46, ours 3.50 → 13.33), and z-image read 13.4 against
its documented ~4.5 for the same reason. Only cool back-to-back numbers mean anything here.

## Follow-ups

- **`use_kitchen_rope` costs ~16 s of load for no measurable per-step gain here.** It *does*
  engage (krea2's `apply_rotary_emb(..., sequence_dim=1)` on a `(cos, sin)` pair matches, so we
  run comfy's fused `apply_rope1`), but `OFFLOADER_KITCHEN_ROPE=0` leaves the per-step unchanged
  while the total drops — rope is a small fraction of a 12 B DiT's compute, so only
  comfy_kitchen's init shows up. Pre-existing offloader behavior, not introduced here.
- `adapter.py::_build_freqs_cis` rebuilds the rope `(cos,sin)→freqs_cis` shim on **every**
  `apply_rotary_emb` call — 2 per block × 28 = 56/step, each allocating a ~1.6 MB fp32 tensor
  (~90 MB of churn) though it is fully loop-invariant. ComfyUI builds `freqs` once per forward
  (`ldm/krea2/model.py:267`) and reuses it across all 28 blocks. Worth ~9 ms/step — left alone
  since per-step parity is already reached; take it if the rope path is touched anyway.
- **z-image is non-deterministic run-to-run** (meandiff 15.9 between two identical-seed runs).
  Unrelated to krea2 and pre-existing — VBAR memory pressure can flip SDPA backend selection —
  but worth its own investigation.
