#
# Run Hipacc sample tests within Docker image 'windows-minimal'
#

$Workspace="C:/workspace"
$Cores=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if ( -not ( Test-Path "C:/TEMP/target/samples" -PathType Container ) ) {
  echo "Error: Hipacc samples directory not found."
  exit 1
}

New-Item -ItemType Directory -Force -Path "C:/TEMP/target/samples/build" | Out-Null

cd "C:/TEMP/target/samples/build"
# Compile with MSBuild from Visual Studio 2017
cmake .. -G "Visual Studio 15" -Ax64 `
  -DCMAKE_BUILD_TYPE=Release $args `
  -DOpenCL_INCLUDE_DIR="C:/Khronos-OpenCL_v2020.03.13/include" `
  -DOpenCL_LIBRARY="C:/Khronos-OpenCL_v2020.03.13/lib/OpenCL.lib"
cmake --build . --config Release --target ALL_BUILD -j $Cores

# Compile with Ninja (much faster)
#cmd.exe /C 'C:\BuildTools\Common7\Tools\VsDevCmd.bat' -arch=x64 '&' `
#  cmake .. -G Ninja `
#    -DCMAKE_BUILD_TYPE=Release $args `
#    -DOpenCL_INCLUDE_DIR="C:/Khronos-OpenCL_v2020.03.13/include" `
#    -DOpenCL_LIBRARY="C:/Khronos-OpenCL_v2020.03.13/lib/OpenCL.lib" '&' `
#  ninja all -j $Cores

ctest -C Release -j $Cores

Copy-Item -Force -Path "C:/TEMP/target/samples/build" -Destination "$Workspace/target/samples" -Recurse
