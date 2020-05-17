#
# Run Hipacc Windows installer creation within Docker image 'windows-minimal'
# (need to build in $Env:TEMP instead of Docker bound mount $Workspace)
#

$ErrorActionPreference = "Stop"

$Workspace="C:/workspace"

if ( -not ( Test-Path "$Env:TEMP/build" -PathType Container ) ) {
  echo "Error: Hipacc build directory not found."
  exit 1
}

cd "$Env:TEMP/build"
cpack -G NSIS
if (-not $?) { exit 1 }

Copy-Item -Force -Path "$Env:TEMP/build/Hipacc-*" -Destination "$Workspace/hipacc/build" -Recurse
