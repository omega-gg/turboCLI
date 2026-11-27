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

prompt="none"

duration="5"

width="512"

height="512"

seed="-1"

inference="-1"

offload="offloader"

slicing="none"

loras="none"

options=""

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

if [ $# -lt 4 -o $# -gt 15 ] \
   || \
   [ "$2" != "cpu" -a "$2" != "cuda" -a "$2" != "mps" ] \
   || \
   [ $# -ge 12 -a "${12}" != "none" -a "${12}" != "slice" ]; then

    echo "Usage: image-to-video <engine> <renderer> <input references> <output video>"
    echo "                      [prompt = $prompt] [duration = $duration]"
    echo "                      [width = $width] [height = $height]"
    echo "                      [seed = $seed] [inference = $inference]"
    echo "                      [offload = $offload] [slicing = $slicing]"
    echo "                      [loras = $loras]"
    echo "                      [options] [server]"
    echo ""
    echo "engine: minimax-h3"
    echo "        comfy-minimax-h3"
    echo "        comfy-minimax-h3-quality"
    echo "        comfy-minimax-h3-upscale"
    echo "        comfy-minimax-h3-quality-upscale"
    echo ""
    echo "renderer: cpu, cuda, mps"
    echo ""
    echo "input references: separated by a comma, each tagged by its role:"
    echo "                  first=<image> last=<image>   the frames the video starts and ends on"
    echo "                  image=<image>                a picture reference, <Picture 1>..."
    echo "                  audio=<audio>                a voice or sound reference, <Audio 1>..."
    echo "                  guide=<audio>@<seconds>      a recording pinned at that time"
    echo "                  latent=<take>.video.latent   a take's latents (latent=save), alone:"
    echo "                                               an upscale engine renders the take again"
    echo "                                               at 2x, the rest of it the take's"
    echo "                  an untagged image is a frame, the first then the last. With a"
    echo "                  picture or audio reference the engine runs its reference workflow"
    echo ""
    echo "prompt: what to film, none for an empty one"
    echo ""
    echo "duration: in seconds, each engine snapping it to its frame grid"
    echo ""
    echo "offload: none, offloader, model_cpu, sequential_cpu, custom (turboCLI/backend folder)"
    echo ""
    echo "slicing: none, slice"
    echo ""
    echo "loras: none, comma separated <path>@[weight]"
    echo ""
    echo "options (comfy-minimax-h3 engines), comma separated:"
    echo "         attention=cudnn puts cuDNN attention ahead of flash: about twice as fast"
    echo "         on a B200"
    echo "         scheduler=simple|beta in place of the engine's: simple, or beta with"
    echo "         references on comfy-minimax-h3-quality"
    echo "         latent=save also writes the video's latents next to it, for a later"
    echo "         pass: <output>.video.latent and .audio.latent, as ComfyUI's SaveLatent"
    echo "         denoise=0-1 the upscale engines' refine strength, 0.45 by default:"
    echo "         lower keeps more of the take's fine detail, 0 the upscaler's alone"
    echo ""
    echo "server: host:port (or port for 127.0.0.1) of a rendering server"
    echo ""
    echo "examples:"
    echo "    image-to-video comfy-minimax-h3 cuda frame.png output.mp4 \"the camera pushes in\""
    echo "    image-to-video comfy-minimax-h3 cuda first.png,last.png output.mp4 \"the camera pushes in\" 10 1344 768"
    echo "    image-to-video comfy-minimax-h3-quality cuda first=fedora.png,audio=voice.wav output.mp4 \"he speaks in the voice of <Audio 1>\" 5 1376 576"
    echo "    image-to-video comfy-minimax-h3-quality-upscale cuda latent=output.video.latent output-2x.mp4"

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

if [ $# -ge 5 ]; then prompt="$5"; fi

if [ $# -ge 6 ]; then duration="$6"; fi

if [ $# -ge 7 ]; then width="$7"; fi

if [ $# -ge 8 ]; then height="$8"; fi

if [ $# -ge 9 ]; then seed="$9"; fi

if [ $# -ge 10 ]; then inference="${10}"; fi

if [ $# -ge 11 ]; then offload="${11}"; fi

if [ $# -ge 12 ]; then slicing="${12}"; fi

if [ $# -ge 13 ]; then loras="${13}"; fi

if [ $# -ge 14 ]; then options="${14}"; fi

if [ $# -ge 15 ]; then server="${15}"; fi

if [ "$prompt" = "none" ]; then prompt=""; fi

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

path=$(getPath "$4")

#--------------------------------------------------------------------------------------------------
# Images
#--------------------------------------------------------------------------------------------------

separator=","

temp=$IFS

IFS="$separator"

for p in $3; do

    # NOTE: Each path converted, its role= and @seconds kept.
    case "$p" in
        first=*|last=*|image=*|audio=*|guide=*|latent=*) role="${p%%=*}="; p="${p#*=}";;
        *)                                               role="";;
    esac

    case "$role$p" in
        guide=*@*) image="$role$(getPath "${p%@*}")@${p##*@}";;
        *)         image="$role$(getPath "$p")";;
    esac

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
                --data-urlencode "engine=$engine" \
                --data-urlencode "mode=image-to-video" \
                --data-urlencode "prompt=$prompt" \
                --data-urlencode "images=$images" \
                --data-urlencode "output=$path" \
                --data-urlencode "width=$width" \
                --data-urlencode "height=$height" \
                --data-urlencode "duration=$duration" \
                --data-urlencode "seed=$seed" \
                --data-urlencode "inference=$inference" \
                --data-urlencode "renderer=$renderer" \
                --data-urlencode "offload=$offload" \
                --data-urlencode "slicing=$slicing" \
                --data-urlencode "loras=$loras" \
                --data-urlencode "options=$options" \
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

# NOTE: The prompt is passed via argv (--prompt="$prompt"), so arbitrary text reaches Python
#       untouched; the equals form keeps a prompt that starts with '-' from being read as a flag.
python -m runner.cli \
       --engine "$engine" \
       --mode "image-to-video" \
       --prompt="$prompt" \
       --images "$images" \
       --output "$path" \
       --width "$width" \
       --height "$height" \
       --duration "$duration" \
       --seed "$seed" \
       --inference "$inference" \
       --renderer "$renderer" \
       --offload "$offload" \
       --slicing "$slicing" \
       --loras "$loras" \
       --options "$options"
