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

# Parse the flat --options string ("key=value,...") into a dict. Engine-specific and torch-free;
# a helper (underscore name) so engine discovery skips it. Each engine reads the keys it needs and
# casts the string value itself (e.g. threshold=40, op=composite).


def parse_options(spec):
    """"threshold=40,op=composite" -> {"threshold": "40", "op": "composite"}. Blank -> {}. Items
    with no "=" are ignored. Values stay strings; the caller casts."""
    out = {}

    for item in spec.split(","):
        if "=" not in item:
            continue

        key, value = item.split("=", 1)

        out[key.strip()] = value.strip()

    return out
