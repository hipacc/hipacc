#!/bin/bash
#
# Build Hipacc within Docker image 'ubuntu-minimal'
#

set -e

BRANCH=${1:-master}
WORKSPACE="/workspace"

mkdir -p "${WORKSPACE}"

# get sources if not existing
if [ ! -d "${WORKSPACE}/hipacc" ]; then
  git clone --recursive https://github.com/hipacc/hipacc -b ${BRANCH} "${WORKSPACE}/hipacc"
fi

mkdir -p "${WORKSPACE}/target"
mkdir -p "${WORKSPACE}/hipacc/build"

export PATH=/llvm-8.0.1-minimal/bin:$PATH

cd "${WORKSPACE}/hipacc/build"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/target"
make install -j$(nproc --all)

