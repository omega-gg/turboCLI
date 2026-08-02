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

# Apply a precomputed mask (from image-mask or image-mask-background) onto an image. Torch-free
# (PIL, no GPU), runs in the turbo venv via runner.apply. Two modes: composite pastes the input's
# masked region onto a reference (the original scene, or a new backdrop); putalpha writes the mask
# as the input's alpha channel (an RGBA cutout, transparent elsewhere).

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

# composite needs a reference (5 args); putalpha does not (4 args).
valid=""

if [ "$1" = "composite" ] && [ $# -eq 5 ]; then valid="yes"; fi
if [ "$1" = "putalpha"  ] && [ $# -eq 4 ]; then valid="yes"; fi

if [ -z "$valid" ]; then

    echo "Usage: image-mask-apply <mode> <input image> <mask image> <output image> [reference]"
    echo ""
    echo "Apply a precomputed mask (from image-mask or image-mask-background). Torch-free (PIL)."
    echo ""
    echo "mode: composite  paste the input's masked region onto a reference (needs a reference)"
    echo "      putalpha   write the mask as the input's alpha channel (an RGBA cutout)"
    echo ""
    echo "input: the source image the mask was computed for"
    echo ""
    echo "mask: an 8-bit grayscale mask / matte (255 = kept)"
    echo ""
    echo "reference: base canvas shown where the mask is black -- REQUIRED for composite (5 args),"
    echo "           omit for putalpha (4 args). The original scene to restore, or a new backdrop."
    echo ""
    echo "examples:"
    echo "    image-mask-apply composite edited.png mask.png output.png original.png"
    echo "    image-mask-apply putalpha  photo.png  matte.png cutout.png"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

mode="$1"

input=$(getPath "$2")

mask=$(getPath "$3")

output=$(getPath "$4")

if [ "$mode" = "composite" ]; then reference=$(getPath "$5"); fi

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

if [ "$mode" = "composite" ]; then

    python -m runner.apply --mode composite \
           --input "$input" --mask "$mask" --output "$output" --reference "$reference"
else
    python -m runner.apply --mode putalpha \
           --input "$input" --mask "$mask" --output "$output"
fi
