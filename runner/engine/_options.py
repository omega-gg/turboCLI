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
# casts the string value itself (e.g. cutoff=40, mode=region). A key alone (align) asks for its
# default.


def parse_options(spec):
    """"cutoff=40,mode=region,align" -> {"cutoff": "40", "mode": "region", "align": ""}. Blank
    -> {}. A key with no "=" has an empty value. Values stay strings; the caller casts."""
    out = {}

    for item in spec.split(","):
        key, _, value = item.partition("=")

        if key.strip():
            out[key.strip()] = value.strip()

    return out


# NOTE: image-to-video's input references: the roles a path takes, "first=a.png" and so on. An
#       untagged path is a frame, the first then the last. "latent=" is a take's saved
#       latents, for an upscale engine (runner/video.py save_latents).
ROLES = ("first", "last", "image", "audio", "guide", "latent")


def parse_references(spec):
    """"first=a.png,audio=v.wav,guide=l.wav@3.5" -> {"frames": ["a.png", None], "references":
    [("audio", "v.wav")], "guide": ("l.wav", 3.5), "latent": None}: the first and last frames
    (None where absent), the picture and audio references in order, the audio guide with its
    time in seconds and the take's video latent file (None without them)."""
    out = {"frames": [None, None], "references": [], "guide": None, "latent": None}
    untagged = 0

    for item in (i.strip() for i in spec.split(",")):
        if not item:
            continue

        role, _, path = item.partition("=")

        if role not in ROLES:  # a path, maybe C:/... : a frame
            role, path = ("first", "last")[min(untagged, 1)], item
            untagged += 1

        if role in ("first", "last"):
            if untagged > 2 or out["frames"][role == "last"] is not None:
                raise ValueError("one first and one last frame at most")
            out["frames"][role == "last"] = path
        elif role == "guide":
            path, at, seconds = path.rpartition("@")
            try:
                out["guide"] = (path, float(seconds))
            except ValueError:
                at = ""
            if not at:
                raise ValueError("a guide is <audio>@<seconds>: %s" % item)
        elif role == "latent":
            if out["latent"] is not None:
                raise ValueError("one latent at most")
            out["latent"] = path
        else:
            out["references"].append((role, path))

    return out
