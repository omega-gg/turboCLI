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
#   python -m runner.install --engine <name> [--model <M>] --folder <base-dir> [--dtype default]
#   python -m runner.install --engine <name> --folder <base-dir> --remove
#
# The base-model revision is pinned in the engine's MODEL["revision"] (not a CLI flag), and the
# model is saved into "<base-dir>/<model>".
#
# `--model` is optional for engines that fix their own model at the module level (e.g.
# qwen-image-edit-2511 and flux2-4b each declare MODEL["model"]); an engine that omits it leaves
# the choice to `--model`.
#
# So `flux2-4b` installs FLUX.2-klein-4B; each qwen engine installs the Qwen model plus exactly the
# LoRAs it uses (none / lightning / lightning + angles). Each engine module under runner/engine
# declares its install needs (MODEL repo + optional LORAS list).
#
# Installs are selective via the "<model>/.commit" manifest (base revision + the installed
# LoRAs and their revisions): a base already at the pinned revision is kept (no re-download, no
# torch import), and only missing/stale LoRAs are fetched -- so installing
# qwen-image-edit-2511-lightning over an existing qwen-image-edit-2511 just pulls the lightning
# LoRA. Removal is `rm -rf` on the model dir.
#
# Separate from the generation cli/server: install is ONLINE and needs no GPU, so this does NOT
# import core (no offload-backend discovery / CUDA init). The wrapper install.sh keeps the env
# (HF_HOME, hf-transfer). Run from the deployed diffusion dir so `engine` is importable.

import os
import sys
import json
import glob
import shutil
import argparse
import importlib

# LoRAs install into a "lora/" subfolder of the model dir (apart from the base checkpoint files).
LORA_DIR = "lora"


def _discover():
    # name -> engine module (cheap: modules hold constants only, no torch/diffusers at top).
    engines = {}

    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "engine", "*.py"))):
        f = os.path.basename(path)

        if f.startswith("_"):
            continue

        m = importlib.import_module("%s.engine.%s" % (__package__, f[:-3]))
        engines[m.ID] = m

    return engines


def _resolve(spec):
    """'diffusers:Flux2KleinPipeline' -> the class."""
    module_name, cls_name = spec.split(":")

    return getattr(importlib.import_module(module_name), cls_name)


def _read_manifest(out):
    """Parse <out>/.commit -> (base_revision, {lora_file: revision}); (None, {}) when absent.

    Line-based and backward compatible: line 1 is the base revision, each later line is
    "<lora_file> <revision>" (revision optional). A legacy revision-only marker yields no LoRAs.
    """
    marker = os.path.join(out, ".commit")

    if not os.path.isfile(marker):
        return None, {}

    base = None
    loras = {}

    with open(marker) as f:
        for line in f:
            parts = line.split()

            if not parts:
                continue

            if base is None:
                base = parts[0]
            else:
                loras[parts[0]] = parts[1] if len(parts) > 1 else None

    return base, loras


def _write_manifest(out, revision, loras):
    """Write <out>/.commit: the base revision then one "<file> <revision>" line per LoRA."""
    lines = [revision]

    for file in sorted(loras):
        rev = loras[file]

        lines.append("%s %s" % (file, rev) if rev else file)

    with open(os.path.join(out, ".commit"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _installed(out, revision, loras):
    """True when <out> holds this base revision and every required LoRA at its pinned revision."""
    base, have = _read_manifest(out)

    if base != revision:
        return False

    for lora in loras:
        if not os.path.isfile(os.path.join(out, LORA_DIR, lora["file"])):
            return False

        want = lora.get("revision")

        if want is not None and have.get(lora["file"]) != want:
            return False

    return True


def _installed_comfy(out):
    """True when a ComfyUI-reuse engine (install --comfy) is in place: the scaffold, the comfy.json
    manifest, and every reused single file it names all exist."""
    marker = os.path.join(out, "comfy.json")

    if not os.path.isfile(marker) or not os.path.isfile(os.path.join(out, "model_index.json")):
        return False

    with open(marker) as f:
        components = json.load(f).get("components", [])

    return bool(components) and all(os.path.isfile(c["path"]) for c in components)


def _models_root(comfy):
    """ComfyUI's models/ dir: <comfy>/ComfyUI/models (portable build) or <comfy>/models."""
    nested = os.path.join(comfy, "ComfyUI", "models")

    return nested if os.path.isdir(nested) else os.path.join(comfy, "models")


def _install_comfy(mod, comfy, out):
    """Install a ComfyUI-reuse engine (--comfy): keep each COMFY component already in the ComfyUI
    install, download only the missing ones there, fetch the tiny SCAFFOLD (configs/tokenizer/
    scheduler, no weights) into `out`, and write out/comfy.json naming the reused files. No base
    repo, no GPU -- so a user who already runs ComfyUI does not re-download the big weights."""
    from huggingface_hub import scan_cache_dir, hf_hub_download, snapshot_download
    from huggingface_hub.errors import CacheNotFound

    comfy_cfg = mod.COMFY
    models = _models_root(comfy)

    repositories = []
    manifest = []

    # Components: reuse each file already under ComfyUI's models/, download the rest there.
    for c in comfy_cfg["components"]:
        dest = os.path.join(models, *c["path"].split("/"))

        if os.path.isfile(dest):
            print("Component present: %s" % c["path"], flush=True)
        else:
            print("Downloading component: %s" % c["path"], flush=True)

            # In-repo path is split_files/<path>; land it in models/ then move to its subdir.
            src = hf_hub_download(repo_id=comfy_cfg["repository"],
                                  filename="split_files/" + c["path"],
                                  revision=comfy_cfg.get("revision") or None, local_dir=models)

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)

            repositories.append(comfy_cfg["repository"])

        manifest.append({"role": c["role"], "path": dest})

    shutil.rmtree(os.path.join(models, "split_files"), ignore_errors=True)

    # Scaffold: the small diffusers configs/tokenizer/scheduler (no weights) for offline load.
    scaffold = mod.SCAFFOLD
    repo = scaffold["repository"] + "/" + scaffold["model"]

    print("Fetching scaffold: %s" % repo, flush=True)

    snapshot_download(repo_id=repo, revision=scaffold.get("revision") or None,
                      allow_patterns=scaffold["allow_patterns"], local_dir=out)

    # snapshot_download leaves a .cache/ bookkeeping dir in local_dir; drop it.
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)

    repositories.append(repo)

    # comfy.json: the manifest generation + check read to find the reused single files.
    with open(os.path.join(out, "comfy.json"), "w") as f:
        json.dump({"comfy": models, "components": manifest}, f, indent=2)

    # Trim the HF cache for anything pulled (a pure reuse pulls nothing, so it may not exist).
    try:
        cache = scan_cache_dir()

        for repo_entry in cache.repos:
            if repo_entry.repo_id in repositories:
                cache.delete_revisions(
                    *[rev.commit_hash for rev in repo_entry.revisions]).execute()
    except CacheNotFound:
        pass

    print("Done.", flush=True)


def main():
    parser = argparse.ArgumentParser(prog="runner.install")

    parser.add_argument("--engine", required=True)
    # --model: model name, e.g. FLUX.2-klein-4B; optional when the engine declares its own
    parser.add_argument("--model", default=None)
    # base model dir; saved to <out>/<model>
    parser.add_argument("--folder", required=True)
    # --dtype: "default" keeps the checkpoint's native dtype (no cast on save); a concrete dtype
    # (bfloat16/float16/float32) re-casts the weights on disk. Run-time dtype is independent of
    # this (core._device_dtype picks it from the renderer), so "default" is the right install
    # choice unless you specifically want a smaller/larger on-disk copy.
    parser.add_argument("--dtype", default="default")
    # --remove: delete the engine's model directory (base model + any LoRAs in it) instead of
    # installing.
    parser.add_argument("--remove", action="store_true")
    # --comfy: a ComfyUI install dir. Only for engines that declare COMFY -- they reuse that
    # install's single files instead of downloading a base repo (see _install_comfy).
    parser.add_argument("--comfy", default=None)

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

    # The base-model revision is pinned in the engine's MODEL (mutable HF repos -> reproducible
    # installs). The wrapper passes only the base dir; the model is saved into "<folder>/<model>".
    revision = model.get("revision")

    out = os.path.join(args.folder, name)

    # --remove: drop the engine's model directory (the base model and any LoRAs it holds).
    if args.remove:
        if os.path.isdir(out):
            shutil.rmtree(out, ignore_errors=True)
            print("Removed %s" % name, flush=True)
        else:
            print("%s is not installed" % name, flush=True)

        return

    # ComfyUI-reuse engines don't download a base repo; they reuse a ComfyUI install's files.
    if hasattr(mod, "COMFY"):
        if not args.comfy:
            print("ERROR: engine '%s' requires --comfy <ComfyUI folder>" % args.engine)
            sys.exit(1)

        _install_comfy(mod, args.comfy, out)
        return

    if args.comfy:
        print("ERROR: engine '%s' does not reuse a ComfyUI install (--comfy not supported)"
              % args.engine)
        sys.exit(1)

    # ".commit" is a small manifest: the base revision plus the LoRAs already installed (and their
    # revisions). It lets us keep an up-to-date base and fetch only the LoRAs an engine is missing.
    base_existing, installed_loras = _read_manifest(out)

    base_ok = bool(revision) and os.path.isdir(out) and base_existing == revision

    # Fully installed for this engine already -> nothing to do.
    if base_ok and _installed(out, revision, loras):
        print("%s already installed at %s" % (name, revision), flush=True)
        return

    # hf_hub_download is all we need to add a LoRA; torch/diffusers only for a base (re)install.
    from huggingface_hub import scan_cache_dir, hf_hub_download

    repositories = []

    if base_ok:
        print("Base model present; installing missing LoRAs only.", flush=True)

        manifest = dict(installed_loras)
    else:
        # Clean base (re)install: drop any previous copy, then ensure the base dir exists.
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(args.folder, exist_ok=True)

        # Heavy imports happen here (post argv-parse), so --help stays instant and bad args fail
        # fast.
        import gc
        import torch

        base_repo = model["repository"] + "/" + name

        PipelineCls = _resolve(mod.PIPELINE)

        print("Prefetching model: %s" % name, flush=True)

        # "default" -> None: diffusers loads each weight in its saved dtype (no cast). NOTE:
        # diffusers coerces a non-torch.dtype value (e.g. the string "auto") to float32, so None --
        # not "auto" -- is what keeps the stock dtype here.
        torch_dtype = None if args.dtype == "default" else getattr(torch, args.dtype)

        # `or None` so a blank revision (e.g. an engine that omits the pin) means latest, not ref
        # "".
        pipe = PipelineCls.from_pretrained(
            base_repo,
            revision=revision or None,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        pipe.save_pretrained(out, safe_serialization=True)

        # NOTE: drop the pipe before touching the cache (avoids permission-denied on Windows).
        del pipe
        gc.collect()

        repositories.append(base_repo)

        # A fresh base wipes any previous LoRAs, so the manifest starts empty.
        manifest = {}

    # Fetch each LoRA the engine needs, unless its file is already present at the pinned revision.
    for lora in loras:
        file = lora["file"]
        want = lora.get("revision")

        if os.path.isfile(os.path.join(out, LORA_DIR, file)) and manifest.get(file) == want:
            print("LoRA already present: %s" % file, flush=True)
        else:
            print("Downloading LoRA: %s" % file, flush=True)

            hf_hub_download(repo_id=lora["repository"], filename=file,
                            revision=want or None, local_dir=os.path.join(out, LORA_DIR))

            repositories.append(lora["repository"])

        manifest[file] = want

    # Trim the HF cache for everything we just pulled (the saved copies are self-contained).
    cache = scan_cache_dir()

    for repo in cache.repos:
        if repo.repo_id in repositories:
            cache.delete_revisions(*[rev.commit_hash for rev in repo.revisions]).execute()

    # Record the base revision and installed LoRAs so check / future installs stay selective.
    if revision:
        _write_manifest(out, revision, manifest)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
