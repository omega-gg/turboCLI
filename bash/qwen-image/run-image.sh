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

model="Qwen-Image-Edit-2511"

width="512"

height="512"

renderer="cpu"

seed="-1"

inference="4"

cuda_offload="sequential_cpu"

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

if [ $# -lt 3 -o $# -gt 12 ] \
   || \
   [ $# -ge 6 -a "$6" != "cpu" -a "$6" != "cuda" -a "$6" != "mps" ] \
   || \
   [ $# -ge 10 -a "${10}" != "none" -a "${10}" != "slice" ]; then

    echo "Usage: run-image <prompt> <input images> <output image>"
    echo "                 [width = $width] [height = $height]"
    echo "                 [renderer = $renderer] [seed = $seed] [inference = $inference]"
    echo "                 [cuda_offload = $cuda_offload] [slicing = $slicing]"
    echo "                 [loras = $loras]"
    echo "                 [server]"
    echo ""
    echo "input images: separated by a comma, 4 maximum"
    echo ""
    echo "renderer: cpu, cuda, mps"
    echo ""
    echo "cuda_offload: none, model_cpu, sequential_cpu, custom (turboCLI/backend folder)"
    echo ""
    echo "slicing: none, slice"
    echo ""
    echo "loras: none, comma separated <path>@[weight]"
    echo ""
    echo "server: host:port (or port for 127.0.0.1) of a rendering server"
    echo ""
    echo "examples:"
    echo "    run \"knight in armor\" shield.png,helmet.png output.png"
    echo "    run \"knight in armor\" shield.png,helmet.png output.png 512 512 cuda -1 4 sequential_cpu none none 8080"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_QWEN_IMAGE:-$sky/turboCLI}"

bin_model="${SKY_PATH_QWEN_IMAGE_MODEL:-$sky/qwen-image-model}"

python="${SKY_PATH_PYTHON:-$sky/python}"

if [ $# -ge 4 ]; then width="$4"; fi

if [ $# -ge 5 ]; then height="$5"; fi

if [ $# -ge 6 ]; then renderer="$6"; fi

if [ $# -ge 7 ]; then seed="$7"; fi

if [ $# -ge 8 ]; then inference="$8"; fi

if [ $# -ge 9 ]; then cuda_offload="$9"; fi

if [ $# -ge 10 ]; then slicing="${10}"; fi

if [ $# -ge 11 ]; then loras="${11}"; fi

if [ $# -ge 12 ]; then server="${12}"; fi

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

path=$(getPath "$3")

model=$(getPath "$bin_model/$model")

#--------------------------------------------------------------------------------------------------
# Images
#--------------------------------------------------------------------------------------------------

separator=","

temp=$IFS

IFS="$separator"

for p in $2; do

    image=$(getPath "$p")

    images="$images$image$separator"
done

IFS=$temp

images="${images%$separator}"

#--------------------------------------------------------------------------------------------------
# LoRAs
#--------------------------------------------------------------------------------------------------

if [ "$loras" = "none" ]; then

    loras=""
else
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
                --data-urlencode "engine=qwen-image-edit-2511-lightning-angles" \
                --data-urlencode "mode=edit" \
                --data-urlencode "model=$model" \
                --data-urlencode "prompt=$1" \
                --data-urlencode "images=$images" \
                --data-urlencode "loras=$loras" \
                --data-urlencode "output=$path" \
                --data-urlencode "width=$width" \
                --data-urlencode "height=$height" \
                --data-urlencode "seed=$seed" \
                --data-urlencode "inference=$inference" \
                --data-urlencode "renderer=$renderer" \
                --data-urlencode "cuda_offload=$cuda_offload" \
                --data-urlencode "slicing=$slicing" \
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
       --engine "qwen-image-edit-2511-lightning-angles" \
       --mode "edit" \
       --model "$model" \
       --prompt="$1" \
       --images "$images" \
       --loras "$loras" \
       --output "$path" \
       --width "$width" \
       --height "$height" \
       --seed "$seed" \
       --inference "$inference" \
       --renderer "$renderer" \
       --cuda_offload "$cuda_offload" \
       --slicing "$slicing"
