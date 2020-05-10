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
cmake .. -DCMAKE_BUILD_TYPE=Release $args
cmake --build . --config Release --target ALL_BUILD -j $Cores
ctest -C Release -j $Cores

Copy-Item -Force -Path "C:/TEMP/target/samples/build" -Destination "$Workspace/target/samples" -Recurse