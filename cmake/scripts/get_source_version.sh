#!/bin/sh

if [ -d .git ]; then
    git log -1 --pretty=format:%H
else
    echo v0.8.0
fi

exit 0

