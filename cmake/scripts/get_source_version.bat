@echo off
if exist .git (
    git log -1 --pretty=format:%H
) else (
    echo v0.8.0
)