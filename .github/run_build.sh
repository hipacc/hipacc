#!/bin/bash
#
# Build Hipacc within Docker image 'ubuntu-minimal'
#

set -e

WORKSPACE="/workspace"

if [ ! -d "${WORKSPACE}/hipacc" ]; then
  echo "Error: Hipacc source directory not found."
  exit 1
fi

mkdir -p "${WORKSPACE}/target"
mkdir -p "${WORKSPACE}/hipacc/build"

export PATH=/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH

cd "${WORKSPACE}/hipacc/build"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/target"
make install -j$(nproc --all)

