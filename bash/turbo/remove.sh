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

if [ $# -ne 1 ]; then

    echo "Usage: remove <engine>"
    echo ""
    echo "Remove an engine and garbage-collect its model / LoRAs / comfy components once no other"
    echo "installed engine references them (a real ComfyUI install's files are never deleted)."
    echo ""
    echo "engine: an installed id (see check-model)"
    echo ""
    echo "example:"
    echo "    remove comfy-z-image-turbo"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

engine="$1"

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

cd "$bin"

#--------------------------------------------------------------------------------------------------
# Remove
#--------------------------------------------------------------------------------------------------

python -m runner.install --engine "$engine" --remove
