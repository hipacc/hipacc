#
# Run Hipacc sample tests within Docker image 'windows-minimal'
# (need to build in $Env:TEMP instead of Docker bound mount $Workspace)
#

$ErrorActionPreference = "Stop"

$UseNinja=$True
$Workspace="C:/workspace"
$Cores=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if ( -not ( Test-Path "$Env:TEMP/target/samples" -PathType Container ) ) {
  echo "Error: Hipacc samples directory not found."
  exit 1
}

# verify that at least CPU is running for OpenCL
& "${WORKSPACE}/target/bin/cl_bandwidth_test.exe" -d CPU
if (-not $?) { exit 1 }

New-Item -ItemType Directory -Force -Path "$Env:TEMP/target/samples/build" | Out-Null
cd "$Env:TEMP/target/samples/build"

if ( $UseNinja ) {
  # Compile with Ninja (slightly faster)
  cmd.exe /C 'C:\BuildTools\Common7\Tools\VsDevCmd.bat' -arch=x64 '&' `
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release $args '&' `
    ninja all -j $Cores
  if (-not $?) { exit 1 }
} else {
  # Compile with MSBuild from Visual Studio 2017
  cmake .. -G "Visual Studio 15" -Ax64 -DCMAKE_BUILD_TYPE=Release $args
  cmake --build . --config Release --target ALL_BUILD -j $Cores
  if (-not $?) { exit 1 }
}

ctest -C Release -j $Cores --output-on-failure
if (-not $?) { exit 1 }

Copy-Item -Force -Path "$Env:TEMP/target/samples/build" -Destination "$Workspace/target/samples" -Recurse
