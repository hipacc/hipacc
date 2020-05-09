#
# Run Hipacc Debian package creation within Docker image 'windows-minimal'
#

$Workspace="C:/workspace"

if ( -not ( Test-Path "C:/TEMP/build" -PathType Container ) ) {
  echo "Error: Hipacc build directory not found."
  exit 1
}

cd "C:/TEMP/build"
cpack -G NSIS

Copy-Item -Force -Path "C:/TEMP/build/Hipacc-*" -Destination "$Workspace/hipacc/build" -Recurse