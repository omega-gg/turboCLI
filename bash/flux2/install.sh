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

engine="flux2-4b"

model="FLUX.2-klein-4B"

dtype="bfloat16"

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

if [ $# -gt 3 ] \
   || \
   [ $# -ge 2 -a "$2" != "bfloat16" -a "$2" != "float16" -a "$2" != "float32" ] \
   || \
   [ $# -ge 3 -a "$3" != "fast" ]; then

    echo "Usage: install [engine = $engine] [dtype = $dtype] [fast]"
    echo ""
    echo "engine: flux2-4b"
    echo ""
    echo "dtype: bfloat16, float16, float32"
    echo ""
    echo "example:"
    echo "    install flux2-4b"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_FLUX2:-$sky/diffusion}"

bin_model="${SKY_PATH_FLUX2_MODEL:-$sky/flux2-model}"

python="${SKY_PATH_PYTHON:-$sky/python}"

if [ $# -ge 1 ]; then engine="$1"; fi

if [ $# -ge 2 ]; then dtype="$2"; fi

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

model_path=$(getPath "$bin_model/$model")

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export HF_HOME="$sky/cache/huggingface"

export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$3" = "fast" ]; then

    # NOTE: This should improve download speeds.
    export HF_XET_HIGH_PERFORMANCE=1
fi

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Clean
#--------------------------------------------------------------------------------------------------

mkdir -p "$bin_model"

rm -rf "$model_path"

#--------------------------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------------------------

echo "Install in progress... The progress output might freeze"

python -m runner.install \
       --engine "$engine" \
       --output "$model_path" \
       --dtype "$dtype"
