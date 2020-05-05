#!/bin/bash
#
# Run Hipacc build, test and package within Docker image 'ubuntu-minimal'
#

set -e

# Start build
$(dirname $0)/run_build.sh

# Start tests in background
$(dirname $0)/run_test.sh &
PID_TEST=$!

# Start package creation in background
$(dirname $0)/run_package.sh &
PID_PACKAGE=$!

# Wait for background processes
wait ${PID_TEST} ${PID_PACKAGE}

