#!/bin/bash
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

dtype="default"

inference="-1"

offload="offloader"

slicing="none"

comfy=""

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

getOs()
{
    case `uname` in
    MINGW*|MSYS*|CYGWIN*) os="windows";;
    Darwin*)              os="macOS";;
    Linux*)               os="linux";;
    *)                    os="other";;
    esac

    type=`uname -m`

    if [ $type = "x86_64" ]; then

        if [ $os = "windows" ]; then

            echo win64
        else
            echo $os
        fi

    elif [ $os = "windows" ]; then

        echo win32
    else
        echo $os
    fi
}

getPath()
{
    path="$1"

    if [ "${path#/}" = "$path" ] && [ "${path#?:[\\/]}" = "$path" ]; then

        path="$PWD/$path"
    fi

    if [ "$os" = "windows" ]; then

        # NOTE: Python does not handle backslash.
        cygpath -w "$path" | sed 's|\\|/|g'
    else
        echo "$path"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 2 -o $# -gt 7 ] \
   || \
   [ "$2" != "cpu" -a "$2" != "cuda" -a "$2" != "mps" ] \
   || \
   [ $# -ge 3 -a "$3" != "default" \
              -a "$3" != "bfloat16" -a "$3" != "float16" -a "$3" != "float32" ] \
   || \
   [ $# -ge 6 -a "$6" != "none" -a "$6" != "slice" ]; then

    echo "Usage: install <engine> <renderer> [dtype = $dtype] [inference = $inference]"
    echo "               [offload = $offload] [slicing = $slicing]"
    echo "               [ComfyUI folder]"
    echo ""
    echo "engine: flux2-4b"
    echo "        z-image-turbo"
    echo "        comfy-flux2-4b"
    echo "        comfy-z-image-turbo"
    echo "        comfy-krea2-turbo"
    echo "        comfy-krea2-turbo-realism"
    echo "        comfy-qwen-image-edit-2511"
    echo "        comfy-qwen-image-edit-2511-lightning"
    echo "        qwen-image-edit-2511"
    echo "        qwen-image-edit-2511-lightning"
    echo "        qwen-image-edit-2511-lightning-angles"
    echo "        mask                 (no download -- registers a compute engine)"
    echo "        mask-apply           (no download)"
    echo "        mask-birefnet        (BiRefNet matte model)"
    echo "        mask-lucida          (Lucida matte model)"
    echo "        mask-inspyrenet      (InSPyReNet matte model)"
    echo ""
    echo "renderer: cpu, cuda, mps"
    echo ""
    echo "dtype: default, bfloat16, float16, float32"
    echo "       (bfloat16 is recommended for CUDA, float16 for Apple MPS)"
    echo "       (the weights are cast on a fresh install alone, remove first to recast)"
    echo ""
    echo "offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)"
    echo ""
    echo "slicing: none, slice"
    echo ""
    echo "ComfyUI folder: optional. Reuse an existing ComfyUI install's model files;"
    echo "                if omitted, components download into turbo/model/ComfyUI/models/."
    echo ""
    echo "NOTE: The renderer and the options after it are recorded with the install, so a host"
    echo "      reads them back with 'check-model SETTINGS:<engine>'. Installing again over an"
    echo "      installed engine re-assigns them."
    echo ""
    echo "examples:"
    echo "    install flux2-4b cuda"
    echo "    install comfy-z-image-turbo cuda bfloat16 -1 offloader none"
    echo "    install comfy-z-image-turbo cuda default -1 offloader none C:/dev/ComfyUI_portable"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

engine="$1"

renderer="$2"

if [ $# -ge 3 ]; then dtype="$3"; fi

if [ $# -ge 4 ]; then inference="$4"; fi

if [ $# -ge 5 ]; then offload="$5"; fi

if [ $# -ge 6 ]; then slicing="$6"; fi

if [ $# -ge 7 ]; then comfy="$7"; fi

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

# NOTE: Enforce bfloat16 on a float32 architecture.
if [ $dtype = "float32" ]; then

    dtype="bfloat16"
fi

if [ -n "$comfy" ]; then comfy=$(getPath "$comfy"); fi

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export HF_HOME="$sky/cache/huggingface"

export HF_HUB_ENABLE_HF_TRANSFER=1

# NOTE: This should improve download speeds.
export HF_XET_HIGH_PERFORMANCE=1

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------------------------

echo "Install in progress... The progress output might freeze"

if [ -n "$comfy" ]; then

    # Reuse a ComfyUI install's model files (comfy-* engines).
    python -m runner.install \
           --engine    "$engine" \
           --renderer  "$renderer" \
           --dtype     "$dtype" \
           --inference "$inference" \
           --offload   "$offload" \
           --slicing   "$slicing" \
           --comfy     "$comfy"
else
    python -m runner.install \
           --engine    "$engine" \
           --renderer  "$renderer" \
           --dtype     "$dtype" \
           --inference "$inference" \
           --offload   "$offload" \
           --slicing   "$slicing"
fi
