#!/bin/sh

if [ -d .git ]; then
    git remote -v | grep 'origin.*fetch' | awk '{ print $2 }'
else
    echo http://sourceforge.net/projects/hipacc/files/latest/download
fi

exit 0

