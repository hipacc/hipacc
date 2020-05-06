#!/bin/bash
#
# Run Hipacc build, test and package within Docker image 'ubuntu-minimal'
#

set -e

BRANCH=${1:-master}
WORKSPACE="/workspace"

mkdir -p "${WORKSPACE}"

# Get sources if not existing
if [ ! -d "${WORKSPACE}/hipacc" ]; then
  git clone --recursive https://github.com/hipacc/hipacc -b ${BRANCH} "${WORKSPACE}/hipacc"
fi

# Start build
${WORKSPACE}/hipacc/.github/run_build.sh

# Start tests in background
${WORKSPACE}/hipacc/.github/run_tests.sh &
PID_TEST=$!

# Start package creation in background
${WORKSPACE}/hipacc/.github/run_package.sh &
PID_PACKAGE=$!

# Wait for background processes
wait ${PID_TEST} ${PID_PACKAGE}

