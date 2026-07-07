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

width="512"

height="512"

renderer="cpu"

seed="-1"

inference="8"

offload="offloader"

slicing="none"

loras="none"

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

if [ $# -lt 2 -o $# -gt 11 ] \
   || \
   [ $# -ge 5 -a "$5" != "cpu" -a "$5" != "cuda" -a "$5" != "mps" ] \
   || \
   [ $# -ge 9 -a "$9" != "none" -a "$9" != "slice" ]; then

    echo "Usage: run <prompt> <output image> [width = $width] [height = $height]"
    echo "           [renderer = $renderer] [seed = $seed] [inference = $inference]"
    echo "           [offload = $offload] [slicing = $slicing]"
    echo "           [loras = $loras]"
    echo "           [server]"
    echo ""
    echo "renderer: cpu, cuda, mps"
    echo ""
    echo "offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)"
    echo ""
    echo "slicing: none, slice"
    echo ""
    echo "loras: none, comma separated <path>@[weight]"
    echo ""
    echo "server: host:port (or port for 127.0.0.1) of a rendering server"
    echo ""
    echo "examples:"
    echo "    run \"knight in armor\" output.png"
    echo "    run \"knight in armor\" output.png 512 512 cuda -1 8 offloader none none 8080"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_Z_IMAGE:-$sky/turbo}"

bin_model="${SKY_PATH_Z_IMAGE_MODEL:-$sky/z-image}"

python="${SKY_PATH_PYTHON:-$sky/python}"

if [ $# -ge 3 ]; then width="$3"; fi

if [ $# -ge 4 ]; then height="$4"; fi

if [ $# -ge 5 ]; then renderer="$5"; fi

if [ $# -ge 6 ]; then seed="$6"; fi

if [ $# -ge 7 ]; then inference="$7"; fi

if [ $# -ge 8 ]; then offload="$8"; fi

if [ $# -ge 9 ]; then slicing="$9"; fi

if [ $# -ge 10 ]; then loras="${10}"; fi

if [ $# -ge 11 ]; then server="${11}"; fi

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

path=$(getPath "$2")

folder=$(getPath "$bin_model")

#--------------------------------------------------------------------------------------------------
# LoRAs
#--------------------------------------------------------------------------------------------------

if [ "$loras" = "none" ]; then

    loras=""
else
    separator=","

    list="$loras"

    loras=""

    temp=$IFS

    IFS="$separator"

    for entry in $list; do

        case "$entry" in
            *@*) lora=$(getPath "${entry%@*}")"@${entry##*@}";;
            *)   lora=$(getPath "$entry");;
        esac

        loras="$loras$lora$separator"
    done

    IFS=$temp

    loras="${loras%$separator}"
fi

#--------------------------------------------------------------------------------------------------
# Server
#--------------------------------------------------------------------------------------------------

if [ -n "$server" ]; then

    case "$server" in
        *:*) host="${server%:*}"; port="${server##*:}";;
        *)   host="127.0.0.1";    port="$server";;
    esac

    base="http://$host:$port"

    echo "Using server at $base"

    stream=$(mktemp)

    curl -sS -N --max-time "3600" \
                --data-urlencode "engine=z-image-turbo" \
                --data-urlencode "mode=generate" \
                --data-urlencode "folder=$folder" \
                --data-urlencode "prompt=$1" \
                --data-urlencode "output=$path" \
                --data-urlencode "width=$width" \
                --data-urlencode "height=$height" \
                --data-urlencode "seed=$seed" \
                --data-urlencode "inference=$inference" \
                --data-urlencode "renderer=$renderer" \
                --data-urlencode "offload=$offload" \
                --data-urlencode "slicing=$slicing" \
                --data-urlencode "loras=$loras" \
                "$base/generate" | tee "$stream"

    if grep -q '^Saved: ' "$stream"; then

        rm -f "$stream"

        exit 0
    fi

    echo "Server request failed"

    rm -f "$stream"

    exit 1
fi

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

    # Use CUDA's stream ordered allocator so large VAE decodes fit and avoid the WDDM RAM spill.
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

# NOTE: The single-shot generation now lives in the shared engine (runner/cli.py -> runner/core.py),
#       the same code the server runs, instead of an inlined heredoc. The prompt is passed via argv
#       (--prompt="$1"), so arbitrary text reaches Python untouched; the equals form keeps a prompt
#       that starts with '-' from being read as a flag.
python -m runner.cli \
       --engine "z-image-turbo" \
       --mode "generate" \
       --folder "$folder" \
       --prompt="$1" \
       --output "$path" \
       --width "$width" \
       --height "$height" \
       --seed "$seed" \
       --inference "$inference" \
       --renderer "$renderer" \
       --offload "$offload" \
       --slicing "$slicing" \
       --loras "$loras"
