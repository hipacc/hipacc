#!/bin/sh

if [ -d .git ]; then
    git log -1 --pretty=format:%H
else
    echo v0.6.0
fi

exit 0

