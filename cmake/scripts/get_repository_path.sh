#!/bin/sh

if [ -d .git ]; then
    git remote -v | grep 'origin.*fetch' | awk '{ print $2 }'
else
    echo https://github.com/hipacc/hipacc/releases
fi

exit 0
