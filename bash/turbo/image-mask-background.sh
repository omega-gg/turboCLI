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

# Generate a subject matte (8-bit grayscale) from an image, via the standalone remove-background
# tool. This is the turbo-namespace front-end; it delegates to bash/remove-background/run.sh, which
# owns the tool's venv + models under gg.omega/remove-background. Apply the matte with
# image-mask-apply (putalpha to cut out, composite to drop onto a new background).

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 4 -o $# -gt 6 ] \
   || \
   [ "$1" != "birefnet" -a "$1" != "lucida" -a "$1" != "inspyrenet" ] \
   || \
   [ "$2" != "cpu" -a "$2" != "cuda" -a "$2" != "mps" ]; then

    echo "Usage: image-mask-background <model> <renderer> <input> <mask output> [plate]"
    echo "                             [shadow threshold]"
    echo ""
    echo "Generate a subject matte (8-bit grayscale PNG, same size and placement) from the input."
    echo "Delegates to the remove-background tool (see bash/remove-background)."
    echo "Apply the matte with image-mask-apply (putalpha to cut out, composite onto a new bg)."
    echo ""
    echo "model: birefnet   (ZhengPeng7/BiRefNet) -- strong on thin glows (a neon sign, a saber)"
    echo "       lucida     (egeorcun fine-tune) -- glass / camouflage / text / print"
    echo "       inspyrenet (transparent-background) -- InSPyReNet, also strong on thin glows"
    echo ""
    echo "renderer: cpu, cuda or mps (cuda / mps fall back to cpu if the build lacks them)"
    echo ""
    echo "plate: a clean background (the same scene without the subject); its cast shadow is kept"
    echo ""
    echo "shadow threshold: darkening floor for the plate shadow; raise it when a drifted plate"
    echo "                  ghosts the background. Only used with a plate."
    echo ""
    echo "examples:"
    echo "    image-mask-background birefnet cuda photo.png matte.png"
    echo "    image-mask-background lucida   cuda photo.png matte.png plate.png"
    echo "    image-mask-background lucida   cuda photo.png matte.png plate.png 40"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

# NOTE: the arguments match run.sh's <model> <renderer> <input> <matte> [plate] [shadow threshold],
#       so pass them straight through; run.sh resolves the paths and runs in the tool's own venv.
run="$(cd "$(dirname "$0")" && pwd)/../remove-background/run.sh"

sh "$run" "$@"
