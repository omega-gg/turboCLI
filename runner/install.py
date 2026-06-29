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

# Model install front-end -- the install-side mirror of the run engines. One command installs
# everything an engine needs: its base model plus any LoRAs it declares, into the model dir.
#
#   python -m runner.install --engine <name> [--model <M>] --output <dir> [--dtype default]
#
# `--model` is optional for engines that fix their own model at the module level (e.g.
# qwen-image-edit-2511 and flux2-4b each declare MODEL["model"]); an engine that omits it leaves
# the choice to `--model`.
#
# So `flux2-4b` installs FLUX.2-klein-4B; each qwen engine installs the Qwen model plus exactly the
# LoRAs it uses (none / lightning / lightning + angles). Each engine module under runner/engine
# declares its install needs (MODEL repo + optional LORAS list). Already-present LoRAs are skipped,
# and removal is just `rm -rf` on the model dir (handled by the wrapper).
#
# Separate from the generation cli/server: install is ONLINE and needs no GPU, so this does NOT
# import core (no offload-backend discovery / CUDA init). The wrapper install.sh keeps the env
# (HF_HOME, hf-transfer). Run from the deployed diffusion dir so `engine` is importable.

import os
import sys
import glob
import argparse
import importlib


def _discover():
    # name -> engine module (cheap: modules hold constants only, no torch/diffusers at top).
    engines = {}

    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "engine", "*.py"))):
        f = os.path.basename(path)

        if f.startswith("_"):
            continue

        m = importlib.import_module("%s.engine.%s" % (__package__, f[:-3]))
        engines[m.NAME] = m

    return engines


def _resolve(spec):
    """'diffusers:Flux2KleinPipeline' -> the class."""
    module_name, cls_name = spec.split(":")

    return getattr(importlib.import_module(module_name), cls_name)


def main():
    parser = argparse.ArgumentParser(prog="runner.install")

    parser.add_argument("--engine", required=True)
    # --model: model name, e.g. FLUX.2-klein-4B; optional when the engine declares its own
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", required=True)            # destination dir
    # --dtype: "default" keeps the checkpoint's native dtype (no cast on save); a concrete dtype
    # (bfloat16/float16/float32) re-casts the weights on disk. Run-time dtype is independent of this
    # (core._device_dtype picks it from the renderer), so "default" is the right install choice
    # unless you specifically want a smaller/larger on-disk copy.
    parser.add_argument("--dtype", default="default")

    args = parser.parse_args()

    mod = _discover().get(args.engine)

    if mod is None:
        print("ERROR: unknown engine '%s'" % args.engine)
        sys.exit(1)

    model = getattr(mod, "MODEL", None)

    if model is None:
        print("ERROR: engine '%s' is not installable (no MODEL)" % args.engine)
        sys.exit(1)

    # The model name comes from --model, or from the engine's own MODEL["model"] when it declares
    # one (e.g. qwen-image-edit-2511, a single fixed model). One of the two must be present.
    name = args.model or model.get("model")

    if name is None:
        print("ERROR: engine '%s' needs --model (it declares no default model)" % args.engine)
        sys.exit(1)

    loras = getattr(mod, "LORAS", [])

    # Heavy imports happen here (post argv-parse), so --help stays instant and bad args fail fast.
    import gc
    import torch
    from huggingface_hub import scan_cache_dir, hf_hub_download

    repositories = [model["repository"] + "/" + name]

    PipelineCls = _resolve(mod.PIPELINE)

    print("Prefetching model: %s" % name, flush=True)

    # "default" -> None: diffusers loads each weight in its saved dtype (no cast). NOTE: this
    # diffusers coerces a non-torch.dtype value (e.g. the string "auto") to float32, so None -- not
    # "auto" -- is what keeps the stock dtype here.
    torch_dtype = None if args.dtype == "default" else getattr(torch, args.dtype)

    pipe = PipelineCls.from_pretrained(
        repositories[0],
        torch_dtype=torch_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    pipe.save_pretrained(args.output, safe_serialization=True)

    # NOTE: drop the pipe before touching the cache (avoids permission-denied on Windows).
    del pipe
    gc.collect()

    for lora in loras:
        if os.path.exists(os.path.join(args.output, lora["file"])):
            print("LoRA already present: %s" % lora["file"], flush=True)
            continue

        print("Downloading LoRA: %s" % lora["file"], flush=True)

        hf_hub_download(repo_id=lora["repository"], filename=lora["file"], local_dir=args.output)

        repositories.append(lora["repository"])

    # Trim the HF cache for everything we just pulled (the saved copies are self-contained).
    cache = scan_cache_dir()

    for repo in cache.repos:
        if repo.repo_id in repositories:
            cache.delete_revisions(*[rev.commit_hash for rev in repo.revisions]).execute()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
