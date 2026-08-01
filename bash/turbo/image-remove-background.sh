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

# Cut a subject out of an image onto a transparent background, via the standalone lucida tool
# (BiRefNet). This is the turbo-namespace front-end; it delegates to bash/lucida/run.sh, which owns
# the tool's venv + models under gg.omega/lucida.

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 4 -o $# -gt 5 ] \
   || \
   [ "$1" != "general" -a "$1" != "lucida" -a "$1" != "inspyrenet" ] \
   || \
   [ "$2" != "cpu" -a "$2" != "cuda" -a "$2" != "mps" ]; then

    echo "Usage: image-remove-background <model> <renderer> <input image> <output image> [plate]"
    echo ""
    echo "Cut the subject out of the input onto a transparent background (RGBA PNG, same size and"
    echo "placement). Delegates to the lucida tool (own venv; see bash/lucida)."
    echo ""
    echo "model: general    (ZhengPeng7/BiRefNet) -- strong on thin glows (a neon sign, a saber)"
    echo "       inspyrenet (transparent-background) -- InSPyReNet, also strong on thin glows"
    echo "       lucida     (egeorcun fine-tune) -- glass / camouflage / text / print"
    echo ""
    echo "renderer: cpu, cuda or mps (cuda / mps fall back to cpu if the lucida build lacks them)"
    echo ""
    echo "plate: a clean background (the same scene without the subject); its cast shadow is kept"
    echo ""
    echo "examples:"
    echo "    image-remove-background general cuda photo.png cutout.png"
    echo "    image-remove-background lucida  cuda photo.png cutout.png plate.png"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

# NOTE: the arguments match run.sh's <model> <renderer> <input> <output> [plate], so pass them
#       straight through; run.sh resolves the paths and runs in the lucida venv (own subshell).
run="$(cd "$(dirname "$0")" && pwd)/../lucida/run.sh"

sh "$run" "$@"
