#
# Build Hipacc within Docker image 'windows-minimal'
#

$Workspace="C:/workspace"
$Cores=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if ( -not ( Test-Path "$Workspace/hipacc" -PathType Container ) ) {
  echo "Error: Hipacc source directory not found."
  exit 1
}

New-Item -ItemType Directory -Force -Path "$Env:Temp/target" | Out-Null
New-Item -ItemType Directory -Force -Path "$Env:Temp/build" | Out-Null

$Env:Path="C:/LLVM_8.0.1-minimal/bin;$Env:Path"

cd "$Env:Temp/build"

# Compile with MSBuild from Visual Studio 2017
cmake $Workspace/hipacc -G "Visual Studio 15" -Ax64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="$Env:Temp/target" `
  -DOpenCL_INCLUDE_DIR="C:/Khronos-OpenCL_v2020.03.13/include" `
  -DOpenCL_LIBRARY="C:/Khronos-OpenCL_v2020.03.13/lib/OpenCL.lib"
cmake --build . --config Release --target INSTALL -j $Cores

# Compile with Ninja (much faster)
#cmd.exe /C 'C:\BuildTools\Common7\Tools\VsDevCmd.bat' -arch=x64 '&' `
#  cmake $Workspace/hipacc -G Ninja `
#    -DCMAKE_BUILD_TYPE=Release `
#    -DCMAKE_INSTALL_PREFIX="$Env:Temp/target" `
#    -DOpenCL_INCLUDE_DIR="C:/Khronos-OpenCL_v2020.03.13/include" `
#    -DOpenCL_LIBRARY="C:/Khronos-OpenCL_v2020.03.13/lib/OpenCL.lib" '&' `
#  ninja install -j $Cores

Copy-Item -Force -Path "$Env:Temp/build" -Destination "$Workspace/hipacc" -Recurse
Copy-Item -Force -Path "$Env:Temp/target" -Destination "$Workspace" -Recurse
