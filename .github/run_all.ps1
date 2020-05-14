#
# Run Hipacc build, test and package within Docker image 'windows-minimal'
#

$Branch="master"
$Workspace="C:/workspace"

if ( $args.count -gt 0 ) {
  $Branch=$args[0]
}

New-Item -ItemType Directory -Force -Path "$Workspace" | Out-Null

# Get sources if not existing
if ( -not ( Test-Path "$Workspace/hipacc" -PathType Container ) ) {
  git clone --recursive https://github.com/hipacc/hipacc -b $Branch "$Workspace/hipacc"
}

# Start build
& "$Workspace/hipacc/.github/run_build.ps1"

# Start tests
& "$Workspace/hipacc/.github/run_tests.ps1"

# Start package creation
& "$Workspace/hipacc/.github/run_package.ps1"

