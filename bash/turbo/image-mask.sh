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

# Change threshold: pixels differing from the reference beyond this are kept in the mask (255).
# Higher = tighter (keeps less), lower = keeps more. Raise it when the generator drifts the whole
# frame (e.g. a flux2 img2img edit). Override per-call with the optional [threshold] arg.
threshold="24"

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

if [ $# -lt 4 -o $# -gt 5 ] \
   || \
   [ "$1" != "mask" -a "$1" != "region" ]; then

    echo "Usage: image-mask <mode> <reference image> <input image> <mask output> [threshold]"
    echo ""
    echo "Generate a soft mask of where an edit differs from its reference -- no compositing. Pure"
    echo "image processing (PIL + numpy, no GPU). Apply it with image-mask-apply."
    echo ""
    echo "mode: mask   soft pixel diff, best for adding an object / recoloring"
    echo "      region grown bounding boxes, best for removal / replace (ghost-free)"
    echo ""
    echo "reference: the base canvas (the original scene)"
    echo ""
    echo "input: the edited / generated image"
    echo ""
    echo "threshold: pixels differing from the reference beyond this are kept in the mask. Default"
    echo "           $threshold; higher = tighter, lower keeps more. Raise when the frame drifts."
    echo ""
    echo "The mask is an 8-bit grayscale PNG at the input resolution."
    echo ""
    echo "To apply: image-mask-apply composite (restore the reference outside the change), or"
    echo "putalpha (cut it out). For a model-based subject matte, see image-mask-background."
    echo ""
    echo "examples:"
    echo "    image-mask mask   original.png edited.png mask.png"
    echo "    image-mask mask   original.png edited.png mask.png 40"
    echo "    image-mask region original.png edited.png mask.png"

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

reference=$(getPath "$2")

input=$(getPath "$3")

output=$(getPath "$4")

if [ $# -eq 5 ]; then threshold="$5"; fi          # optional override of the Settings default

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

python -m runner.mask \
       --reference "$reference" \
       --input     "$input" \
       --output    "$output" \
       --mode      "$mode" \
       --threshold "$threshold"
