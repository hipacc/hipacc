#
# Build Hipacc within Docker image 'windows-minimal'
#

$Workspace="C:/workspace"
$Cores=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if ( -not ( Test-Path "$Workspace/hipacc" -PathType Container ) ) {
  echo "Error: Hipacc source directory not found."
  exit 1
}

New-Item -ItemType Directory -Force -Path "C:/TEMP/target" | Out-Null
New-Item -ItemType Directory -Force -Path "C:/TEMP/build" | Out-Null

$Env:Path="C:/LLVM_8.0.1-minimal/bin;$Env:Path"

cd "C:/TEMP/build"
cmake $Workspace/hipacc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="C:/TEMP/target"
cmake --build . --config Release --target INSTALL -j $Cores

Copy-Item -Force -Path "C:/TEMP/build" -Destination "$Workspace/hipacc" -Recurse
Copy-Item -Force -Path "C:/TEMP/target" -Destination "$Workspace" -Recurse