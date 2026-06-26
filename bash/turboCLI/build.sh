#!/bin/sh
set -e
#==================================================================================================
#
#   Copyright (C) 2026-2026 turboCLI authors. <https://omega.gg/Sky>
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

name="diffusion"

repository="https://github.com/omega-gg/turboCLI.git"

commit="7e0414ca049d623c38d20daff5f4f428fd242ac4"

diffusers="784fa62652fb2719d415830f918fc32a49ecc7a1"

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

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
rm -rf "$name-model"

if [ "$1" = "clean" ]; then

    echo "CLEANING"

    uv cache prune

    exit 0
fi

#--------------------------------------------------------------------------------------------------
# Clone
#--------------------------------------------------------------------------------------------------

mkdir -p "$name"
cd       "$name"

git init

git remote add origin "$repository"

git fetch --depth 1 origin "$commit"

git checkout FETCH_HEAD

rm -rf .git

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

    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

elif [ "$1" = "mps" ]; then

    uv pip install torch torchvision torchaudio
else
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

uv pip install --upgrade hf_transfer safetensors accelerate transformers peft \
                         "huggingface_hub[hf_xet]" \
                         "git+https://github.com/huggingface/diffusers@$diffusers"
