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

dtype="default"

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

if [ $# -lt 1 -o $# -gt 3 ] \
   || \
   [ $# -ge 2 -a "$2" != "default" \
              -a "$2" != "bfloat16" -a "$2" != "float16" -a "$2" != "float32" ]; then

    echo "Usage: install <engine> [dtype = $dtype] [ComfyUI folder]"
    echo ""
    echo "engine: flux2-4b"
    echo "        z-image-turbo"
    echo "        comfy-z-image-turbo"
    echo "        comfy-krea2-turbo"
    echo "        comfy-krea2-turbo-realism"
    echo "        qwen-image-edit-2511"
    echo "        qwen-image-edit-2511-lightning"
    echo "        qwen-image-edit-2511-lightning-angles"
    echo ""
    echo "dtype: default, bfloat16, float16, float32"
    echo "       (bfloat16 is recommended for CUDA, float16 for Apple MPS)"
    echo ""
    echo "ComfyUI folder: optional. Reuse an existing ComfyUI install's model files;"
    echo "                if omitted, components download into turbo/model/ComfyUI/models/."
    echo ""
    echo "examples:"
    echo "    install flux2-4b"
    echo "    install comfy-z-image-turbo"
    echo "    install comfy-z-image-turbo default C:/dev/test/ComfyUI_windows_portable"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

engine="$1"

if [ $# -ge 2 ]; then dtype="$2"; fi

if [ $# -ge 3 ]; then comfy="$3"; fi

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
           --engine "$engine" \
           --dtype  "$dtype" \
           --comfy  "$comfy"
else
    python -m runner.install \
           --engine "$engine" \
           --dtype  "$dtype"
fi
