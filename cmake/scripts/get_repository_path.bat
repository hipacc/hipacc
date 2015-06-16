@echo off
if exist .git (
    git remote -v | findstr "origin.*fetch"
) else (
    echo https://github.com/hipacc/hipacc/releases
)