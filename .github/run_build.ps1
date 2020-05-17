#
# Build Hipacc within Docker image 'windows-minimal'
# (need to build in $Env:TEMP instead of Docker bound mount $Workspace)
#

$ErrorActionPreference = "Stop"

$UseNinja=$True
$Workspace="C:/workspace"
$Cores=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if ( -not ( Test-Path "$Workspace/hipacc" -PathType Container ) ) {
  echo "Error: Hipacc source directory not found."
  exit 1
}

New-Item -ItemType Directory -Force -Path "$Env:TEMP/target" | Out-Null
New-Item -ItemType Directory -Force -Path "$Env:TEMP/build" | Out-Null

$Env:Path="C:/LLVM_8.0.1-minimal/bin;$Env:Path"

cd "$Env:TEMP/build"
if ( $UseNinja ) {
  # Compile with Ninja (slightly faster)
  cmd.exe /C 'C:\BuildTools\Common7\Tools\VsDevCmd.bat' -arch=x64 '&&' `
    cmake $Workspace/hipacc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$Env:TEMP/target" '&&' `
    ninja install -j $Cores
  if (-not $?) { exit 1 }
} else {
  # Compile with MSBuild from Visual Studio 2017
  cmake $Workspace/hipacc -G "Visual Studio 15" -Ax64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$Env:TEMP/target"
  cmake --build . --config Release --target INSTALL -j $Cores
  if (-not $?) { exit 1 }
}

Copy-Item -Force -Path "$Env:TEMP/build" -Destination "$Workspace/hipacc" -Recurse
Copy-Item -Force -Path "$Env:TEMP/target" -Destination "$Workspace" -Recurse
