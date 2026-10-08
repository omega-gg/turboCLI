# Plan: FLUX.2 [klein] 9B in turboCLI (flux2-9b + comfy-flux2-9b)

## Context

turboCLI supports FLUX.2 [klein] 4B as `flux2-4b` (diffusers repo) and `comfy-flux2-4b` (reuses a
ComfyUI install's single files). We add the 9B model the same two ways, install and test
`comfy-flux2-9b` first, and compare it against flux2-4b.

What differs from 4B (checked on the Hub):
- **Gated + licensed.** Every BFL 9B weight repo except `FLUX.2-klein-9b-kv-fp8` is gated
  (`gated: auto`), and all of them, kv-fp8 included, are under the FLUX Non-Commercial License; 4B
  is Apache-2.0 and ungated. Downloads need a Hugging Face token from an account that accepted the
  license. There is no ComfyUI-level auth: ComfyUI's own templates pull the same gated BFL URLs.
  turboCLI has no token support today (a 401 surfaces as a raw traceback).
- **Weights.** The text encoder is Qwen3-8B (36 layers, `tie_word_embeddings: false`, its file
  carries `lm_head`), same FLUX.2 VAE as 4B.

User decisions: comfy-flux2-9b reuses **ComfyUI's fp8 choice** (what its Klein 9B templates
install); the token is passed as an **install argument**; the user provides the token at install
time (it never goes through Claude).

## Model sources (pinned)

| role | comfy-flux2-9b (fp8, ComfyUI's template choice) | flux2-9b (diffusers, bf16) |
|---|---|---|
| transformer | `black-forest-labs/FLUX.2-klein-9b-fp8` @ `902d9d51` → `flux-2-klein-9b-fp8.safetensors` (9.4 GB, gated) | `black-forest-labs/FLUX.2-klein-9B` @ `92196c8e` (2 shards, 18.2 GB, gated) |
| text encoder | `Comfy-Org/vae-text-encorder-for-flux-klein-9b` @ `3f62d9d8` → `split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors` (8.7 GB, ungated) | same repo, 4 shards bf16 (16.4 GB) |
| VAE | `vae/flux2-vae.safetensors`, shared with comfy-flux2-4b (already installed) | in repo |
| scaffold (configs, tokenizer, scheduler) | `black-forest-labs/FLUX.2-klein-9B` @ `92196c8e` (gated) | n/a |

comfy-flux2-9b totals ~18 GB (fits the 32 GB test box's RAM); flux2-9b ~35 GB (exceeds it, so it
disk-streams like qwen-image-edit).

## Step 1: Token as an install argument (turboCLI)

- `bash/turbo/install.sh`: add an optional 8th positional `[token]` after `[ComfyUI folder]`;
  accept `none` for the folder so a stock engine can take a token
  (`install flux2-9b cuda default -1 offloader none none <token>`). When given, `export HF_TOKEN`
  for this run only. huggingface_hub reads `HF_TOKEN` itself, so `runner/install.py`'s
  downloads (`hf_hub_download`, `snapshot_download`, `from_pretrained`) need no token plumbing,
  and the token never enters argv of the Python process nor the recorded engine settings.
  Update the usage text (token line + a 9B example).
- `runner/install.py:main`: catch huggingface_hub's `GatedRepoError` (and a 401
  `HfHubHTTPError`) around the install dispatch and print one actionable message: the repo is
  gated, accept the license at `https://huggingface.co/<repo>`, then re-run install with the token
  argument. Exit non-zero, no traceback.

## Step 2: `runner/engine/flux2_9b.py` (stock)

`BASE = "flux2-4b"` (inherits TYPE/PIPELINE/TRANSFORMER/MODES/CFG/INFERENCE via `_inherit.py`),
plus
`ID = "flux2-9b"` and `MODEL = {"repository": "black-forest-labs", "model": "FLUX.2-klein-9B",
"revision": "92196c8e11f7b6cf2b7493e037d8c5345c559216"}`. Confirm `is_distilled: true` in the 9B
`model_index.json` once the token works (CFG 0.0 / 4 steps carry over only then).

## Step 3: `runner/engine/comfy_flux2_9b.py` (fp8, offloader-only)

Modeled on `comfy_flux2_4b.py` (constants, `_by_role`, `_transformer_meta`, `_build_vae`,
scaffold handling) and on `comfy_krea2_turbo.py` for the fp8 quant path:
- `COMFY` components and `SCAFFOLD` as in the table above.
- `load()`: offloader-only (bail out for other offload modes, as comfy-krea2-turbo does); both
  specs `"quant": True` through `ctx.backend.load_pipe_comfy`.
- Text encoder: Qwen3-8B meta-built from the scaffold config; **no `_tie_lm_head`** (the file has a
  real `lm_head`, aliasing it would be wrong); keys already match `Qwen3ForCausalLM` (`model.`
  prefix), so no rename beyond carrying fp8 companions untouched.
- Transformer `convert`: diffusers' `convert_flux2_transformer_checkpoint_to_diffusers` on the base
  weights, carrying each layer's fp8 companions (`.weight_scale`, `.comfy_quant`, any
  `.input_scale`) to the renamed module, as `comfy_krea2_turbo._transformer_convert` does. For a
  fused qkv split into three, each chunk gets the source scale (per-tensor) or the matching scale
  chunk (per-row); settle which from the file header first. The offloader's `_own_file_slice`
  already makes the chunked fp8 views stream as file slices.
- Validate before any image: 0 unexpected keys, no module left on meta.

## Step 4: Lists and docs (turboCLI)

Add both engines to the usage lists in `bash/turbo/install.sh`, `text-to-image.sh`,
`image-to-image.sh` and `bash/turbo/README.md`; the engine table in `implementation.md` (with the
license/gating note); `turboCLI.pro` OTHER_FILES (both engine files + the plan doc). Copy this plan
to `doc/comfy-flux2-9b-plan.md` (precedent: `doc/comfy-flux2-4b-plan.md`), with no local paths.
No offloader change expected: it is model-agnostic.

## Step 5: Install and verify

1. Without a token: `install comfy-flux2-9b ...` prints the gated-repo message, exits non-zero,
   leaves nothing half-written. `check-model ENGINES:text-to-image` lists both new engines.
2. **The user runs the install with their token** (the token never goes through Claude), reusing
   the portable ComfyUI so the files land in its `models/` and the VAE is shared:
   `sh install.sh comfy-flux2-9b cuda default -1 offloader none <ComfyUI folder> <token>`.
3. Smoke: comfy-flux2-9b text-to-image 512², seed 42, CUDA, offloader: no unexpected keys, image
   sane (`check_img` std), then one image-to-image (an existing attic frame).
4. flux2-9b (stock, ~35 GB download): implemented and listed; its install + smoke run only if the
   user wants the download now.

## Step 6: Compare with flux2-4b

Short, interleaved (the laptop throttles; see the benchmarking rules), CUDA, offloader, seed 42,
same prompts:
- text-to-image at 512² and 1024×768: comfy-flux2-9b vs comfy-flux2-4b, 2 pairs each; per-phase
  times (load / encode / step 1 / steady / decode) and side-by-side images.
- one image-to-image ("make it crisp and sharp, keep the dark blue night lighting" on an attic
  frame) on both.
- Optional: one comfy-flux2-9b vs ComfyUI v0.39.1 pair (template graph, same fp8 files) to confirm
  parity holds on 9B.
Report a table plus the images; commits wait for the user's go-ahead, no Claude attribution.

## Critical files

- `bash/turbo/install.sh` (token argument, usage), `runner/install.py` (gated-repo message)
- `runner/engine/flux2_9b.py`, `runner/engine/comfy_flux2_9b.py` (new)
- `bash/turbo/text-to-image.sh`, `image-to-image.sh`, `bash/turbo/README.md`, `implementation.md`,
  `turboCLI.pro`, `doc/comfy-flux2-9b-plan.md`
- Reused: `comfy_flux2_4b.py` helpers, `comfy_krea2_turbo._transformer_convert` pattern,
  offloader `load_pipe_comfy` / `load_quant_single_file` / `_own_file_slice`

## Revision: two ComfyUI engines

After the fp8 vs bf16 comparison the ComfyUI side was split in two, so each one mirrors a
reference exactly:

- `comfy-flux2-9b-distilled` (`runner/engine/comfy_flux2_9b_distilled.py`) is ComfyUI's distilled
  Klein 9B template file for file: the fp8 transformer and fp8-mixed Qwen3-8B text encoder above,
  plus BFL's small decoder VAE, `black-forest-labs/FLUX.2-small-decoder` @ `a3efc24f`
  (`full_encoder_small_decoder.safetensors`, Apache-2.0, ungated). It has the same encoder and
  latents as flux2-vae, with a narrower decoder (`decoder_block_out_channels: [96, 192, 384, 384]`)
  that is lighter and faster to decode. Its `_build_vae` overrides those widths on the
  scaffold's VAE config; the file is in the original BFL layout (not diffusers', unlike
  flux2-vae), so diffusers' `convert_ldm_vae_checkpoint` remaps it after lifting the nested quant
  convs, and the latent BatchNorm is carried as is. The result equals the repo's diffusers file
  tensor for tensor. It is the lighter, faster engine, used on the 32 GB test box. It is a
  `BASE = comfy-flux2-9b` delta too (same scaffold and pipeline): its files and its `load()`.
- `comfy-flux2-9b` (`runner/engine/comfy_flux2_9b.py`) is flux2-9b's weights as ComfyUI single
  files: the bf16 transformer `flux-2-klein-9b.safetensors` at the root of the same pinned
  FLUX.2-klein-9B revision, the bf16 `qwen_3_8b` text encoder from the same Comfy-Org repo, and the
  shared flux2-vae, ~35 GB. It is a `BASE = comfy-flux2-4b` delta (files and scaffold only): the
  9B transformer has the same plain bf16 single-file layout. `_tie_lm_head` now keeps an `lm_head`
  the file already carries (Qwen3-8B is untied), instead of aliasing the embedding over it.

Both are compared on the cloud GPU (t2i and i2i, same seeds), see Results.

`install --dtype default` now copies a stock repo as published (`snapshot_download` of
`model_index.json` and the component folders, no cast); a concrete dtype loads the pipeline cast
to it and saves that copy.

## Results

CUDA, offloader, 4 steps, seed 42 (and 7 for the broad t2i set). Cloud runs on an L4 (24 GB) with
the pipe kept loaded between images; laptop runs on the 32 GB test box, which throttles.

| engine | disk | L4 t2i 1024x768 | L4 i2i | laptop t2i 1024x768 |
|---|---|---|---|---|
| comfy-flux2-4b (bf16) | 16.1 GB | 3.1 s | 5.9 s | ~49 s (cold process) |
| comfy-flux2-9b-distilled (fp8, small decoder) | 18.4 GB | 4.4 s | 8.2 s | ~61 s (cold process) |
| comfy-flux2-9b (bf16) | 34.9 GB | 1.5-2x the distilled sampling | | does not fit RAM |

Quality, 4B vs 9B-distilled, 16 t2i prompts x 2 seeds and 7 edits:

- Ties: portrait, animal, landscape, interior, product, food, macro, anime, short text (a sign,
  a poster title), spatial layout, and the simple edits (add an object, restyle, relight to
  winter, remove the subject, recolor + add a cape). Counting fails on both.
- 9B-distilled ahead: multi-person scenes and hands (4B drops a person or a hand on some seeds),
  painterly styles, and the demanding edits: keeping the original lighting on a "make it crisp"
  pass, and placing a character from a reference image at the asked spot with matched lighting.

9B-distilled vs 9B bf16: the edits are near identical (34-36 dB PSNR between them) and t2i keeps
the same composition; short text differs per prompt without a consistent winner. bf16 costs twice
the disk and 1.5-2x the sampling time for no visible gain, so it stays an option, not the default.

Conclusion: comfy-flux2-4b is good enough for quick single-subject t2i; comfy-flux2-9b-distilled
is the sweet spot for image-to-image (lighting preservation, reference placement) at ~1.25-1.4x
the 4B time and nearly the same disk.
