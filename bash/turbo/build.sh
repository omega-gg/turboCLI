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

commit="341f286c3fe2f42697c380292161ce4a218de386" # Also update in check.sh

commit_offloader="681d55c9d2f398ede568a954813d653f67e54912"

diffusers="784fa62652fb2719d415830f918fc32a49ecc7a1"

#--------------------------------------------------------------------------------------------------

# NOTE: Pinned versions validated against the bundled Python 3.14.2 so a build six months from now
# resolves the same stack.

torch_version="2.12.1"
torchvision_version="0.27.1"
torchaudio_version="2.11.0"

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

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# != 1 ] || [ "$1" != "cpu" -a "$1" != "cuda" -a "$1" != "mps" -a "$1" != "clean" ]; then

    echo "Usage: build <cpu | cuda | mps | clean>"
    echo ""
    echo "example:"
    echo "    build cuda"

    exit 1
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
        "torch==$torch_version" "torchvision==$torchvision_version" \
        "torchaudio==$torchaudio_version" \
        --index-url https://download.pytorch.org/whl/cu130

elif [ "$1" = "mps" ]; then

    uv pip install \
        "torch==$torch_version" "torchvision==$torchvision_version" \
        "torchaudio==$torchaudio_version"
else
    uv pip install \
        "torch==$torch_version" "torchvision==$torchvision_version" \
        "torchaudio==$torchaudio_version" \
        --index-url https://download.pytorch.org/whl/cpu
fi

uv pip install \
    "hf_transfer==$hf_transfer_version" "hf_xet==$hf_xet_version" \
    "safetensors==$safetensors_version" "accelerate==$accelerate_version" \
    "transformers==$transformers_version" "peft==$peft_version" \
    "huggingface_hub==$huggingface_hub_version" "psutil==$psutil_version" \
    "git+https://github.com/huggingface/diffusers@$diffusers"

#--------------------------------------------------------------------------------------------------
# comfy
#--------------------------------------------------------------------------------------------------

if [ "$1" = "cuda" ]; then

    uv pip install "comfy-aimdo==$comfy_aimdo_version"
fi

uv pip install "comfy-kitchen==$comfy_kitchen_version"
