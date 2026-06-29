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

check()
{
    if [ ! -d "$1" ]; then

        return
    fi

    if [ -n "$models" ]; then

        models="$models,$1"
    else
        models="$1"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# != 0 -a $# != 1 ]; then

    echo "Usage: check [model]"
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_Z_IMAGE_MODEL:-$sky/z-image}"

#--------------------------------------------------------------------------------------------------
# Check
#--------------------------------------------------------------------------------------------------

cd "$bin"

models=""

if [ $# = 1 ]; then

    check "$1"

    if [ -z "$models" ]; then

        echo "$1 is not installed"

        exit 1
    else
        echo "$1 is installed"

        exit 0
    fi
fi

check "Z-Image-Turbo"

if [ -z "$models" ]; then

    echo "No model is installed"

    exit 1
fi

echo "$models"

exit 0
