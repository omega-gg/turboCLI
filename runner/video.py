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

# A video engine's result written to a file: its frames and its soundtrack, as ComfyUI's
# CreateVideo and SaveVideo write them (VideoFromComponents.save_to, v0.45.4,
# comfy_api/latest/_input_impl/video_types.py:283-415): an mp4, H.264 in yuv420p tagged sRGB on
# BT.709 primaries, the audio in AAC at its own rate. No metadata tags (SaveVideo writes the
# prompt there), so a file depends on its pictures and sound alone. Beside it, on request, the
# latents it was decoded from (save_latents). Torch free (PyAV, numpy, safetensors).

import json
import math
import os

from fractions import Fraction

import numpy as np

# NOTE: SaveVideo's quality when its encoding is left to auto (video_encoder_options).
CRF = 18


def save_latents(path, latents, settings):
    """Write a video's latents (float32 arrays: the video's, the audio's) next to the video at
    `path`, <stem>.video.latent and <stem>.audio.latent, as SaveLatent (nodes.py:510-559) writes
    each latent LTXVSeparateAVLatent splits off (nodes_lt.py:972-1000), for LoadLatent. The
    metadata, where SaveLatent writes its prompt, holds the request's `settings` as JSON under
    "settings", for a later pass on them. Returns the paths."""
    from safetensors.numpy import save_file

    stem = os.path.splitext(path)[0]
    paths = []

    for kind, latent in zip(("video", "audio"), latents):
        paths.append("%s.%s.latent" % (stem, kind))
        save_file({"latent_tensor": np.ascontiguousarray(latent),
                   "latent_format_version_0": np.zeros(0, np.float32)}, paths[-1],
                  metadata={"settings": json.dumps(settings)})

    return paths


def save(path, frames, fps, audio=None, rate=None, crf=CRF):
    """Write `frames` (uint8, [T, H, W, 3]) at `fps`, with `audio` (float32 [channels, samples]
    at `rate`) when given, to the mp4 at `path`."""
    import av

    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

    def color(target):
        # set_video_color_properties for sRGB (video_types.py:313-317): BT.709, limited range.
        target.color_primaries = ColorPrimaries.BT709
        target.color_trc = ColorTrc.IEC61966_2_1
        target.colorspace = 1  # BT709_NCL
        target.color_range = ColorRange.MPEG

    frame_rate = Fraction(round(fps * 1000), 1000)

    with av.open(path, mode="w", format="mp4",
                 options={"movflags": "use_metadata_tags+faststart"}) as output:
        stream = output.add_stream("h264", rate=frame_rate)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}

        color(stream.codec_context)

        if audio is not None:
            # NOTE: The soundtrack is cut to the video's length.
            waveform = audio[:, :math.ceil((rate / frame_rate) * frames.shape[0])]

            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")

            audio_stream = output.add_stream("aac", rate=rate, layout=layout)

        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace=1)  # BT709_NCL

            color(frame)

            output.mux(stream.encode(frame))

        output.mux(stream.encode(None))

        if audio is not None:
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform, dtype=np.float32),
                                               format="fltp", layout=layout)
            frame.sample_rate = rate
            frame.pts = 0

            output.mux(audio_stream.encode(frame))
            output.mux(audio_stream.encode(None))
