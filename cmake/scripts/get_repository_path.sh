#!/bin/sh

git remote -v | grep 'fetch' | awk '{ print $2 }'

exit 0

