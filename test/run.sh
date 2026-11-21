#!/bin/bash
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

set -o pipefail

#--------------------------------------------------------------------------------------------------
# Settings
#--------------------------------------------------------------------------------------------------

seed="42"

fox="a red fox sitting in a snowy forest, golden hour"

edit="turn the scene into a snowy winter night lit by torches"

film="the camera slowly pushes in on a snowy forest as the wind sways the trees"

duration="5"

tolerance="10"

repeat="1"

port="8199"

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

usage()
{
    echo "Usage: run.sh [options] [engine ...]"
    echo ""
    echo "Runs each engine, stock and comfy, at a fixed seed: its image must stay byte-identical"
    echo "to this machine's reference (test/reference/), its sampling no slower than it beyond"
    echo "the tolerance. No engine: every installed one. One that is not installed is skipped."
    echo ""
    echo "--renderer <cuda|mps|cpu>  default: cuda when nvidia-smi answers, mps on macOS, else cpu"
    echo "--size <N|WxH>             output size, default 1024 on cuda, 512 on mps and cpu"
    echo "--record                   write this machine's reference instead of checking it"
    echo "--repeat <N>               run each case N times, keep the fastest (default $repeat)"
    echo "--cool                     before each run, wait for the GPU to cool to 50 C"
    echo "                           (nvidia-smi), or pause 60 s on mps and cpu"
    echo "--warmup                   before each case, one run of each side that is not kept, so"
    echo "                           no measured run reads the model files cold"
    echo "--tolerance <P>            percent slower still passing (default $tolerance)"
    echo "--comfy <dir>              also run the comfy engines through the ComfyUI in <dir>"
    echo "--comfy-only <dir>         run the comfy engines through the ComfyUI in <dir> alone"
    echo "--mode <case>              only that case: a mode, or image-to-video-ref and kin"
    echo "--video                    the video cases instead, a suite of their own (MiniMax H3:"
    echo "                           a GPU that holds it, modal.sh run --video); a video --mode"
    echo "                           implies it"
    echo "--list                     print the engines a run would test, and run nothing"
    echo ""
    echo "examples:"
    echo "    run.sh --record"
    echo "    run.sh flux2-4b comfy-flux2-4b"
    echo "    run.sh --repeat 2 --cool --comfy ~/ComfyUI comfy-z-image-turbo"
    echo "    run.sh --video --size 512 --comfy ~/ComfyUI"

    exit 1
}

renderer=""
size=""
record=false
listing=false
cool=false
warmup=false
comfy=""
sides="turbo"
engines=""
only=""
video=false

while [ $# -gt 0 ]; do

    case "$1" in
        --renderer)   renderer="$2"; shift;;
        --size)       size="$2"; shift;;
        --record)     record=true;;
        --repeat)     repeat="$2"; shift;;
        --cool)       cool=true;;
        --warmup)     warmup=true;;
        --tolerance)  tolerance="$2"; shift;;
        --comfy)      comfy="$2"; sides="turbo comfy"; shift;;
        --comfy-only) comfy="$2"; sides="comfy"; shift;;
        --mode)       only="$2"; shift;;
        --video)      video=true;;
        --list)       listing=true;;
        -*)           usage;;
        *)            engines="$engines $1";;
    esac

    shift
done

case "$only" in *-video*) video=true;; esac

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

test="$(cd "$(dirname "$0")" && pwd)"

turbo="$test/../bash/turbo"

images="$test/old"

if [ "$sides" != "comfy" ] && [ -z "$SKY_PATH_BIN" ]; then

    echo "SKY_PATH_BIN is not set" >&2

    exit 1
fi

if [ -z "$renderer" ]; then

    if nvidia-smi > /dev/null 2>&1; then

        renderer="cuda"

    elif [ "$(uname)" = "Darwin" ]; then

        renderer="mps"
    else
        renderer="cpu"
    fi
fi

if [ -z "$size" ]; then

    if [ "$renderer" = "cuda" ]; then size="1024"; else size="512"; fi
fi

width="${size%x*}"

height="${size#*x}"

# An upscaler's image-to-image case: old/knight-small.png (368x512) to the size's height, x2 at
# least (x2 at 1024, x4 at 2048, x8 at 4096), the same image as the other modes' at each size.
scale=$(( height / 512 > 2 ? height / 512 : 2 ))

upscaled="$(( 368 * scale ))x$(( 512 * scale ))"

if [ "$renderer" = "cuda" ]; then

    device="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"

elif [ "$(uname)" = "Darwin" ]; then

    device="$(sysctl -n machdep.cpu.brand_string)"
else
    device="$(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2)"
fi

# The machine's name: its renderer, device and size, which an image is only identical within.
machine="$renderer-$(echo $device | sed 's/[^A-Za-z0-9.]\{1,\}/-/g')-$size"

reference="$test/reference/$machine.txt"

results="$test/results/$machine"

mkdir -p "$results" "$test/reference"

if [ -n "$comfy" ]; then

    comfy="$(cd "$comfy" && pwd)"

    if [ -n "$COMFY_PYTHON" ]; then

        python="$COMFY_PYTHON"

    elif [ -f "$comfy/../python_embeded/python.exe" ]; then

        python="$comfy/../python_embeded/python.exe"

    elif [ -f "$comfy/venv/bin/python" ]; then

        python="$comfy/venv/bin/python"
    else
        python="python3"
    fi
fi

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

# Each line of the input stamped with the time it arrives, to the millisecond (a progress bar's
# updates, split on their carriage returns, too).
stamp()
{
    perl -MTime::HiRes=time -e '$| = 1; while (sysread(STDIN, my $b, 65536)) {
        printf "%.3f %s\n", time, $_ for grep /\S/, split /[\r\n]+/, $b }'
}

now()
{
    perl -MTime::HiRes=time -e 'printf "%.3f", time'
}

md5()
{
    if command -v md5sum > /dev/null; then

        md5sum "$1" | cut -c 1-32
    else
        md5 -q "$1"
    fi
}

wait_cool()
{
    if [ $cool = false ]; then return; fi

    if [ "$renderer" = "cuda" ]; then

        for i in $(seq 120); do

            if [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader \
                    | head -n 1 | tr -d '\r')" -le 50 ]; then return; fi

            sleep 5
        done
    else
        sleep 60
    fi
}

# Every installed engine, stock and comfy, in each of its modes, "engine mode" per line.
# image-adjust-night is a second case of image-adjust's engines: a night, whose colors stay;
# image-to-video-ref one of image-to-video's: the knight as a picture reference.
turbo_cases()
{
    for mode in text-to-image image-to-image text-to-video image-to-video image-to-video-ref \
                image-to-mask image-adjust image-adjust-night; do

        bash "$turbo/check-model.sh" ENGINES:$(base $mode) | tr -d '\r' \
            | awk -v m=$mode '$2 == "installed" { print $1, m }'
    done
}

# The mode a case runs: image-adjust-night runs image-adjust, image-to-video-ref image-to-video.
base()
{
    local mode="${1%-night}"

    echo "${mode%-ref}"
}

# The (engine, mode) pairs comfy.py has a ComfyUI graph for.
comfy_cases()
{
    "$python" -s "$test/comfy.py" cases | tr -d '\r'
}

# Whether the ComfyUI in $comfy has every file of an engine.
comfy_has()
{
    for path in $("$python" -s "$test/comfy.py" files "$1" | tr -d '\r'); do

        if [ ! -f "$comfy/models/$path" ]; then return 1; fi
    done
}

# One run: <engine> <mode> <image> <log>. Prints "wall load encode sampling decode" in seconds,
# "-" for what the run does not show. The wall time ends at "Saved:", where the client takes the
# image, not at the process's exit, as ComfyUI's ends when it returns the image.
run_turbo()
{
    local engine="$1" mode="$2" output="$3" log="$4" input prompt="$edit" target=$size start end \
          expect=()

    # NOTE: A fix that stops working still saves an image, which --record would keep: each
    #       adjust case also needs the line of each fix saying it did what it should.
    case "$mode" in
        image-to-image)
            input="$images/knight.png"

            # NOTE: An upscaler enlarges the small knight, with no prompt, rather than edit it.
            case "$engine" in
                *-upscale) input="$images/knight-small.png" prompt=none target=$upscaled;;
            esac;;
        image-to-video)
            input="$images/knight.png";;
        image-to-video-ref)
            input="image=$images/knight.png";;
        image-adjust)
            input="$images/knight-drift.png"
            expect=("align: .*, aligned$" "color: .*, matched$");;
        image-adjust-night)
            input="$images/knight-night.png"
            expect=("align: .*, aligned$" "color: .*, left as is$");;
    esac

    start="$(now)"

    case "$mode" in
        text-to-image)
            bash "$turbo/text-to-image.sh" "$engine" $renderer "$output" "$fox" \
                 $width $height $seed -1 ${offload:-offloader};;
        image-to-image)
            bash "$turbo/image-to-image.sh" "$engine" $renderer "$input" "$output" "$prompt" \
                 ${target%x*} ${target#*x} $seed -1 ${offload:-offloader};;
        text-to-video)
            bash "$turbo/text-to-video.sh" "$engine" $renderer "$output" "$film" $duration \
                 $width $height $seed -1 ${offload:-offloader};;
        image-to-video*)
            bash "$turbo/image-to-video.sh" "$engine" $renderer "$input" "$output" "$film" \
                 $duration $width $height $seed -1 ${offload:-offloader};;
        image-to-mask)
            if [ "$engine" = "mask" ]; then
                input="$images/chest.png,$images/courtyard.png"
            else
                input="$images/knight.png"
            fi
            bash "$turbo/image-to-mask.sh" "$engine" $renderer "$input" "$output";;
        image-adjust*)
            bash "$turbo/image-adjust.sh" "$input" "$images/knight-small.png" "$output" \
                 align,color;;
    esac 2>&1 | stamp > "$log" || return 1

    end="$(now)"

    for line in "${expect[@]}"; do

        grep -q "$line" "$log" || return 1
    done

    awk -v start=$start -v end=$end '
        $2 == "model" && $3 == "ready" { ready = $1 }
        $2 == "generating"             { generating = $1 }
        $2 ~ /\|step$/ {
            split($3, step, "/")
            if (step[1] == 0) first = $1
            if (step[1] == step[2]) last = $1
        }
        $2 == "Saved:" { saved = $1 }
        function span(a, b) { return (a && b) ? sprintf("%.2f", b - a) : "-" }
        END {
            print span(start, saved ? saved : end), span(start, ready), span(generating, first),
                  span(first, last), span(last, saved)
        }' "$log"
}

run_comfy()
{
    local engine="$1" mode="$2" output="$3" log="$4" prompt="$fox" input="-" target=$size flag="" \
          server start end status

    case "$mode" in
        text-to-video)  prompt="$film";;
        image-to-video*) prompt="$film" input="$images/knight.png";;
    esac

    if [ "$mode" = "image-to-image" ]; then

        case "$engine" in
            *-upscale) prompt="" input="$images/knight-small.png" target=$upscaled;;
            *)         prompt="$edit" input="$images/knight.png";;
        esac
    fi

    if [ "$renderer" = "cpu" ]; then flag="--cpu"; fi

    # Without the pinned copy of the weights when turbo runs without it (the same videos).
    if [ "$OFFLOADER_DISABLE_PINNED_MEMORY" = 1 ]; then flag="$flag --disable-pinned-memory"; fi

    # A fresh server per run, as a turbo run starts fresh, timed from its launch as turbo's is.
    start="$(now)"

    (cd "$comfy" && exec "$python" -s main.py --port $port --disable-auto-launch $flag \
                       $("$python" -s "$test/comfy.py" args "$engine" | tr -d '\r')) \
        > >(stamp > "$log") 2>&1 &
    server=$!

    until curl -s -o /dev/null "http://127.0.0.1:$port/system_stats"; do

        if ! kill -0 $server 2> /dev/null; then return 1; fi

        sleep 0.1
    done

    "$python" -s "$test/comfy.py" run "$engine" "$mode" "$prompt" $seed $target "$input" \
        "$output" "http://127.0.0.1:$port" $duration >> "$log.run" 2>&1
    status=$?

    end="$(now)"

    kill $server 2> /dev/null
    wait $server 2> /dev/null

    while curl -s -o /dev/null "http://127.0.0.1:$port/system_stats"; do sleep 1; done

    if [ $status != 0 ]; then return 1; fi

    # Its sampler's progress bar, from 0/N to N/N.
    awk -v start=$start -v end=$end '
        {
            for (i = 2; i < NF; i++) {
                if ($i ~ /^[0-9]+\/[0-9]+$/ && $(i + 1) ~ /^\[/) {
                    split($i, step, "/")
                    if (step[1] == 0 && !first) { first = $1; steps = step[2] }
                    if (first && !last && step[1] == steps) last = $1
                }
            }
        }
        function span(a, b) { return (a && b) ? sprintf("%.2f", b - a) : "-" }
        END { print span(start, end), "-", "-", span(first, last), "-" }' "$log"
}

# The reference's "md5 wall load encode sampling decode" for <side> <engine> <mode>.
referenced()
{
    if [ -f "$reference" ]; then

        awk -v s=$1 -v e=$2 -v m=$3 '
            $1 == s && $2 == e && $3 == m { print $4, $5, $6, $7, $8, $9 }' "$reference"
    fi
}

# <side> <engine> <mode> <md5> <times...> into the reference, in place of its former line.
remember()
{
    local lines

    if [ -f "$reference" ]; then

        lines="$(awk -v s=$1 -v e=$2 -v m=$3 '!/^#/ && !($1 == s && $2 == e && $3 == m)' \
                 "$reference")"
    fi

    {
        echo "# $machine, recorded $(date +%F): turboCLI $(cat "$commit" 2> /dev/null || echo -)"
        echo "# side engine mode md5 wall load encode sampling decode (seconds)"
        { if [ -n "$lines" ]; then echo "$lines"; fi; echo "$*"; } | sort
    } > "$reference"
}

# "+4%" for <time> against <reference time>.
delta()
{
    awk -v a=$1 -v b=$2 'BEGIN { printf "%+d%%", (a - b) * 100 / b }'
}

slower()
{
    awk -v a=$1 -v b=$2 -v t=$tolerance 'BEGIN { exit !(a > b * (1 + t / 100)) }'
}

report()
{
    echo "$*" | tee -a "$summary"
}

fail()
{
    report "$*"

    failed=true
}

# The cases, "engine mode side..." per line: turbo's, or ComfyUI's with --comfy-only. With
# --comfy, a turbo case also runs on ComfyUI when it has a graph there.
cases()
{
    if [ "$sides" = "comfy" ]; then

        comfy_cases | sed 's/$/ comfy/'
        return
    fi

    local graphs="$(if [ "$sides" = "turbo comfy" ]; then comfy_cases; fi)"

    turbo_cases | while read engine mode; do

        if echo "$graphs" | grep -qx "$engine $mode"; then

            echo "$engine $mode turbo comfy"
        else
            echo "$engine $mode turbo"
        fi
    done
}

# The engine's runs of one side, the fastest first: its image must not change between them, the
# fastest is its result, checked against the reference or recorded.
judge()
{
    local side=$1 engine=$2 mode=$3 best md5 ref result

    best=($(grep "^$side " "$runs" | sort -n -k 2 | head -n 1))

    if [ ${#best[@]} = 0 ]; then return; fi

    if [ $(grep "^$side " "$runs" | cut -d ' ' -f 3 | sort -u | wc -l) != 1 ]; then

        fail "$side $engine $mode: UNSTABLE, its image changes between runs"
    fi

    # best: side, time, md5, wall, load, encode, sampling, decode, image
    md5=${best[2]}

    if [ $record = true ]; then

        remember $side $engine $mode $md5 ${best[@]:3:5}
        result="RECORDED"
    else
        # ref: md5, wall, load, encode, sampling, decode
        ref=($(referenced $side $engine $mode))

        if [ ${#ref[@]} = 0 ]; then

            result="NEW"
        else
            if [ "${ref[0]}" = "$md5" ]; then

                result="SAME"
            else
                result="DIFFERS"
                failed=true
            fi

            # Sampling decides. A run without steps (masks) is mostly start-up: its wall time is
            # shown, not judged.
            if [ "${ref[4]}" = "-" ]; then

                result="$result $(delta ${best[1]} ${ref[1]})"

            elif slower ${best[1]} ${ref[4]}; then

                result="$result $(delta ${best[1]} ${ref[4]}) SLOWER"
                failed=true
            else
                result="$result $(delta ${best[1]} ${ref[4]})"
            fi
        fi
    fi

    report "$(printf "%-5s %-38s %-18s %8ss  load %6s encode %6s decode %6s  %s" $side $engine \
              $mode ${best[1]} ${best[4]} ${best[5]} ${best[7]} "$result")"
}

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

commit="${SKY_PATH_TURBOCLI:-$SKY_PATH_BIN/gg.omega/turbo}/.commit"

summary="$results/summary.txt"

runs="$results/runs"

list="$results/cases"

failed=false

: > "$summary"

cases > "$list"

# The video cases are a suite of their own (--video), the others the default one.
awk -v video=$video '($2 ~ /-video/) == (video == "true")' "$list" > "$list.asked"

mv "$list.asked" "$list"

# Only the engines asked for, when some are.
if [ -n "$engines" ]; then

    for engine in $engines; do

        if ! grep -q "^$engine " "$list"; then

            echo "$engine: skipped (not installed, no ComfyUI graph, or video without --video)"
        fi
    done

    grep -E "^($(echo $engines | tr ' ' '|')) " "$list" > "$list.asked"

    mv "$list.asked" "$list"
fi

# Only the case asked for, when one is.
if [ -n "$only" ]; then

    awk -v m="$only" '$2 == m' "$list" > "$list.asked"

    mv "$list.asked" "$list"
fi

# --list: the engines a run would test, and no run.
if [ $listing = true ]; then

    cut -d ' ' -f 1 "$list" | sort -u

    exit 0
fi

while read engine mode pairs; do

    # A case with a ComfyUI side runs it only when that ComfyUI has the engine's files.
    if [[ " $pairs " == *" comfy "* ]] && ! comfy_has $engine; then

        pairs="${pairs% comfy}"

        if [ "$pairs" = "comfy" ]; then

            report "$engine $mode: skipped (its files are not in that ComfyUI)"
            continue
        fi
    fi

    # The offload the engine was installed with, for its turbo runs.
    offload="$(bash "$turbo/check-model.sh" SETTINGS:$engine 2> /dev/null | tr -d '\r' \
               | awk '$1 == "offload:" { print $2 }')"

    : > "$runs"

    ext=png

    case "$mode" in *-video*) ext=mp4;; esac

    # --warmup: one run of each side first, not kept. The model files come off the disk or the
    # Volume for it, so the side that goes first does not take that read in a measured run.
    if [ $warmup = true ]; then

        for side in $pairs; do

            printf "warming up %s %s %s..." $side $engine $mode

            if run_$side $engine $mode "$results/$side-$engine-$mode-warmup.$ext" \
                      "$results/$side-$engine-$mode-warmup.log" < /dev/null > /dev/null; then
                echo " done"
            else
                echo " failed (see $results/$side-$engine-$mode-warmup.log)"
            fi
        done
    fi

    # Each run's line: side, the time that decides (sampling, or wall time for a run without
    # steps), md5, wall, load, encode, sampling, decode, image. Sides alternate, so heat lands on
    # both alike, and each repeat swaps which goes first, so neither always reads the files cold.
    order="$pairs"

    for r in $(seq $repeat); do

        for side in $order; do

            wait_cool

            output="$results/$side-$engine-$mode-$r.$ext"
            log="$results/$side-$engine-$mode-$r.log"

            printf "running %s %s %s (%s/%s)..." $side $engine $mode $r $repeat

            if times=($(run_$side $engine $mode "$output" "$log" < /dev/null)); then

                echo " ${times[0]}s from launch to image"

                echo "$side ${times[3]/-/${times[0]}} $(md5 "$output") ${times[*]} $output" \
                    >> "$runs"
            else
                echo ""
                fail "$side $engine $mode: FAILED, see $log"
            fi
        done

        order="$(echo $order | awk '{ for (i = NF; i > 0; i--) printf "%s ", $i }')"
    done

    for side in $pairs; do

        judge $side $engine $mode
    done

    # Turbo against ComfyUI: the speed, and how close the images are.
    ours=($(grep "^turbo " "$runs" | sort -n -k 2 | head -n 1))
    theirs=($(grep "^comfy " "$runs" | sort -n -k 2 | head -n 1))

    if [ ${#ours[@]} != 0 ] && [ ${#theirs[@]} != 0 ]; then

        result="$(delta ${ours[1]} ${theirs[1]}) against ComfyUI,"
        result="$result $("$python" -s "$test/comfy.py" psnr ${ours[8]} ${theirs[8]})"

        if slower ${ours[1]} ${theirs[1]}; then

            fail "      $engine $mode: $result SLOWER"
        else
            report "      $engine $mode: $result"
        fi
    fi

done < "$list"

if [ $failed = true ]; then exit 1; fi
