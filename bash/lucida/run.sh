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

# Run the Lucida (BiRefNet) background remover: cut the subject out of <input> onto a transparent
# background (RGBA PNG, same size/placement). With a [plate] (the same scene without the subject)
# the cast shadow is also kept. Used standalone or by image-mask's extract / extract-full modes.

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

if [ $# -lt 3 -o $# -gt 4 ] \
   || \
   [ "$1" != "cpu" -a "$1" != "cuda" -a "$1" != "mps" ]; then

    echo "Usage: run <renderer> <input image> <output image> [plate image]"
    echo ""
    echo "renderer: cpu, cuda or mps (cuda / mps fall back to cpu if this build lacks them)"
    echo ""
    echo "plate: a clean background (the same scene without the subject); its cast shadow is kept"
    echo ""
    echo "examples:"
    echo "    run cuda photo.png cutout.png"
    echo "    run cuda photo.png cutout.png plate.png"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_LUCIDA:-$sky/lucida}"

python="${SKY_PATH_PYTHON:-$sky/python}"

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

renderer="$1"

input=$(getPath "$2")

output=$(getPath "$3")

if [ $# -ge 4 ]; then plate=$(getPath "$4"); fi

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "$renderer" = "cuda" ]; then

    # Use CUDA's stream ordered allocator to avoid the WDDM RAM spill on Windows.
    export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

elif [ "$renderer" = "mps" ]; then

    # NOTE macOS: Fallback on CPU if needed.
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    # NOTE macOS: Disable the memory cap to avoid allocation failures on large models.
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

if [ -n "$plate" ]; then

    python extract.py --input "$input" --output "$output" --device "$renderer" --plate "$plate"
else
    python extract.py --input "$input" --output "$output" --device "$renderer"
fi
