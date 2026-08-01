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

name="remove-background"

# Pinned model revisions -- Also update in build.sh.
birefnet_revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
lucida_revision="6ee11122534c8de59402a589d2293c198cfbf848"
inspyrenet_revision="1.2.12"

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

revision()
{
    # $1 = model dir under bin/model, $2 = expected revision (the build.sh .revision marker).
    [ -f "$bin/model/$1/.revision" ] && [ "`cat "$bin/model/$1/.revision"`" = "$2" ]
}

#--------------------------------------------------------------------------------------------------
# Check
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_REMOVE_BACKGROUND:-$sky/$name}"

# NOTE: the .revision marker is written only after a successful download (build.sh runs with
#       set -e), so a matching revision already implies the model files are present -- no need to
#       also stat the weights.
if { [ -f "$bin/.venv/Scripts/activate" ] || [ -f "$bin/.venv/bin/activate" ]; } \
   && [ -f "$bin/extract.py" ] \
   && revision birefnet    "$birefnet_revision" \
   && revision lucida     "$lucida_revision" \
   && revision inspyrenet "$inspyrenet_revision"; then

    echo "remove-background is installed"

    exit 0
fi

echo "remove-background is not installed"

exit 1
