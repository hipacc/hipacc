#!/bin/bash
#
# Run Hipacc Debian package creation within Docker image 'ubuntu-minimal'
#

set -e

WORKSPACE="/workspace"

if [ ! -d "${WORKSPACE}/hipacc/build" ]; then
  echo "Error: Hipacc build directory not found."
  exit 1
fi

mkdir -p "${WORKSPACE}/hipacc/build"

cd "${WORKSPACE}/hipacc/build"
cpack -G DEB

