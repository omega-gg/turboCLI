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

build="bash/turbo/build.sh"

check="bash/turbo/check.sh"

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

getValue()
{
    sed -n "s|^$2=\"\([^\"]*\)\".*|\1|p" "$1"
}

setValue()
{
    # NOTE: The pattern stops at the closing quote so a trailing comment survives.
    sed -i "s|^$2=\"[^\"]*\"|$2=\"$3\"|" "$1"
}

pushCommit()
{
    # NOTE: Only the files we rewrote are staged, so unrelated work in progress is never swept in.
    git add "$@"

    git commit -m "$message"

    git push origin "$branch"

    echo "pushed \"$message\""
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -ne 0 ]; then

    echo "Usage: update"
    echo ""
    echo "Pin the latest turbo-offloader commit in build.sh, then record turboCLI's own commit in"
    echo "build.sh and check.sh. Each step commits and pushes only when the hash actually changed,"
    echo "so running it twice in a row is a no-op."

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

cd "$(dirname "$0")"

branch=$(git rev-parse --abbrev-ref HEAD)

#--------------------------------------------------------------------------------------------------
# turbo-offloader
#--------------------------------------------------------------------------------------------------

repository=$(getValue "$build" repository_offloader)

commit=$(git ls-remote "$repository" HEAD | cut -f1)

current=$(getValue "$build" commit_offloader)

if [ "$commit" = "$current" ]; then

    echo "turbo-offloader is up to date ($current)"
else
    setValue "$build" commit_offloader "$commit"

    message="turbo-offloader $commit"

    pushCommit "$build"
fi

#--------------------------------------------------------------------------------------------------
# turboCLI
#--------------------------------------------------------------------------------------------------

# NOTE: The recorded hash is always the PREVIOUS commit, a commit cannot carry its own hash.
commit=$(git rev-parse HEAD)

current=$(getValue "$build" commit)

if [ "$(git log -1 --format=%s)" = "turboCLI $current" ]; then

    echo "turboCLI is up to date ($current)"
else
    setValue "$build" commit "$commit"

    setValue "$check" commit "$commit"

    message="turboCLI $commit"

    pushCommit "$build" "$check"
fi
