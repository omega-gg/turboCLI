#!/bin/sh
set -e
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

#--------------------------------------------------------------------------------------------------
# Settings
#--------------------------------------------------------------------------------------------------

name="turbo"

repository="https://github.com/omega-gg/turboCLI.git"

repository_offloader="https://github.com/omega-gg/turbo-offloader.git"

commit="7e3930cc704a5593b0150241ff96d5c61d216db1" # Also update in check.sh

commit_offloader="7e80cdfc6dfd311cf2274b4c3a4140184e696ce9"

diffusers="60ec6f724290fb7640abaf3ca9a2b89bc15e8a8b"

#--------------------------------------------------------------------------------------------------

# NOTE: Pinned versions validated against the bundled Python 3.14.2 so a build six months from now
# resolves the same stack.

torch_version="2.12.1"
torchvision_version="0.27.1"
torchaudio_version="2.11.0"
torch_cuda="cu130"

transformers_version="5.12.1"
accelerate_version="1.14.0"
peft_version="0.19.1"

huggingface_hub_version="1.21.0"
hf_xet_version="1.5.1"
hf_transfer_version="0.1.9"
safetensors_version="0.8.0"
psutil_version="7.2.2"

comfy_aimdo_version="0.4.10"

comfy_kitchen_version="0.2.16"

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

clone()
{
    mkdir -p "$1"
    cd       "$1"

    git init

    git remote add origin "$2"

    git fetch --depth 1 origin "$3"

    git checkout FETCH_HEAD

    git rev-parse HEAD > .commit

    rm -rf .git

    cd -
}

getSky()
{
    if [ -z "$SKY_PATH_BIN" ]; then

        echo "SKY_PATH_BIN is not set" >&2

        return
    fi

    case `uname` in
        MINGW*|MSYS*|CYGWIN*)
            cygpath -u "$SKY_PATH_BIN/gg.omega";;
        *)
            echo "$SKY_PATH_BIN/gg.omega";;
    esac
}

require()
{
    if [ "$latest" = 1 ]; then

        echo "$1"
    else
        echo "$1==$2"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 1 -o $# -gt 2 ] \
   || [ "$1" != "cpu" -a "$1" != "cuda" -a "$1" != "mps" -a "$1" != "clean" ] \
   || [ "$2" != "" -a "$2" != "latest" ] \
   || [ "$2" = "latest" -a "$1" = "clean" ]; then

    echo "Usage: build <cpu | cuda | mps | clean> [latest]"
    echo ""
    echo "latest: install the newest releases, ignoring the pinned versions (not reproducible)"
    echo ""
    echo "example:"
    echo "    build cuda"
    echo "    build cuda latest"

    exit 1
fi

if [ "$2" = "latest" ]; then

    latest=1

    diffusers_ref="main"

    echo "WARNING: building with 'latest' -- ignoring pinned versions, not reproducible."
else
    latest=0

    diffusers_ref="$diffusers"
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

python="${SKY_PATH_PYTHON:-$sky/python}"

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export UV_CACHE_DIR="$sky/cache/uv"

export GIT_CONFIG_PARAMETERS="'core.longpaths=true'"

#--------------------------------------------------------------------------------------------------
# Clean
#--------------------------------------------------------------------------------------------------

mkdir -p "$sky"
cd       "$sky"

# Detach the model cache (turbo/model) and the engine registry (turbo/engine) so the wipe below
# never destroys the downloaded models (~20GB) or the installed-engine records; both are reattached
# after the fresh clone. On 'clean' they stay detached at <sky>/.turbo-model|.turbo-engine so the
# next build restores them.
if [ -d "$name/model"  ]; then mv "$name/model"  ".turbo-model";  fi
if [ -d "$name/engine" ]; then mv "$name/engine" ".turbo-engine"; fi

rm -rf "$name"

if [ "$1" = "clean" ]; then

    echo "CLEANING"

    uv cache prune

    exit 0
fi

#--------------------------------------------------------------------------------------------------
# Clone
#--------------------------------------------------------------------------------------------------

clone "$name" "$repository" "$commit"

cd "$name"

clone "temp" "$repository_offloader" "$commit_offloader"

mv "temp/offloader" "backend"

rm -rf "temp"

# Reattach the detached model cache + engine registry into the fresh install (cwd is <sky>/turbo).
if [ -d "$sky/.turbo-model"  ]; then mv "$sky/.turbo-model"  "model";  fi
if [ -d "$sky/.turbo-engine" ]; then mv "$sky/.turbo-engine" "engine"; fi

#--------------------------------------------------------------------------------------------------
# Activate
#--------------------------------------------------------------------------------------------------

# NOTE: We need a relative python path to keep the venv portable.
case `uname` in
    MINGW*|MSYS*|CYGWIN*) uv venv .venv --relocatable --python "../python/python.exe";;
    *)                    uv venv .venv --relocatable --python "../python/bin/python";;
esac

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Install
#--------------------------------------------------------------------------------------------------

if [ "$1" = "cuda" ]; then

    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)" \
        "$(require torchaudio $torchaudio_version)" \
        --index-url https://download.pytorch.org/whl/$torch_cuda

elif [ "$1" = "mps" ]; then

    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)" \
        "$(require torchaudio $torchaudio_version)"
else
    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)" \
        "$(require torchaudio $torchaudio_version)" \
        --index-url https://download.pytorch.org/whl/cpu
fi

uv pip install \
    "$(require hf_transfer $hf_transfer_version)" \
    "$(require hf_xet $hf_xet_version)" \
    "$(require safetensors $safetensors_version)" \
    "$(require accelerate $accelerate_version)" \
    "$(require transformers $transformers_version)" \
    "$(require peft $peft_version)" \
    "$(require huggingface_hub $huggingface_hub_version)" \
    "$(require psutil $psutil_version)" \
    "git+https://github.com/huggingface/diffusers@$diffusers_ref"

#--------------------------------------------------------------------------------------------------
# comfy
#--------------------------------------------------------------------------------------------------

if [ "$1" = "cuda" ]; then

    uv pip install "$(require comfy-aimdo $comfy_aimdo_version)"
fi

uv pip install "$(require comfy-kitchen $comfy_kitchen_version)"
