#
# Run Hipacc Windows installer creation within Docker image 'windows-minimal'
#

$Workspace="C:/workspace"

if ( -not ( Test-Path "$Env:Temp/build" -PathType Container ) ) {
  echo "Error: Hipacc build directory not found."
  exit 1
}

cd "$Env:Temp/build"
cpack -G NSIS

Copy-Item -Force -Path "$Env:Temp/build/Hipacc-*" -Destination "$Workspace/hipacc/build" -Recurse
