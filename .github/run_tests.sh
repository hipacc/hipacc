#!/bin/bash
#
# Run Hipacc sample tests within Docker image 'ubuntu-minimal'
#

set -e

WORKSPACE="/workspace"

if [ ! -d "${WORKSPACE}/target/samples" ]; then
  echo "Error: Hipacc samples directory not found."
  exit 1
fi

# verify that at least CPU is running for OpenCL
"${WORKSPACE}/target/bin/cl_bandwidth_test" -d CPU

mkdir -p "${WORKSPACE}/target/samples/build"
cd "${WORKSPACE}/target/samples/build"

cmake .. -DCMAKE_BUILD_TYPE=Release $@
make -j$(nproc --all)

ctest -j$(nproc --all) --output-on-failure

