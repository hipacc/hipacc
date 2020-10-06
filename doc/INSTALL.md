# Installation
Binary packages can be found [here](../../../releases).
### On Ubuntu
Download the Debian package and run `sudo dpkg -i hipacc_<version>_amd64.deb`.
### On Windows
Download and run the installer `Hipacc-<version>-win64.exe`.


# Run Samples
### On Ubuntu
Open a terminal and run:
```bash
cp -r /usr/local/hipacc/samples .
mkdir samples/build; cd samples/build
cmake ..

# Compile and run Box Blur for CPU
make Box_Blur_Cpu
./1_Local_Operators/Box_Blur/Box_Blur_Cpu

# Compile and run Box Blur for CUDA
make Box_Blur_Cuda
./1_Local_Operators/Box_Blur/Box_Blur_Cuda
```
### On Windows
###### With Visual Studio 2019
Open `%HIPACC_PATH%\samples\CMakeLists.txt`, select the desired startup item, and run build.

###### With MSVC C++ Build Tools
To run the samples without Visual Studio, install the latest **Windows SDK** and
**MSVC 141** or **MSVC 142** provided by
[Build Tools for Visual Studio 2019](https://aka.ms/buildtools).
Open PowerShell and run:
```PowerShell
cp -r "$Env:HIPACC_PATH\samples" .
mkdir .\samples\build; cd .\samples\build
cmake ..

# Compile and run Box Blur for CPU
cmake --build . --target Box_Blur_Cpu --config Release
.\1_Local_Operators\Box_Blur\Release\Box_Blur_Cpu.exe

# Compile and run Box Blur for CUDA
cmake --build . --target Box_Blur_Cuda --config Release
.\1_Local_Operators\Box_Blur\Release\Box_Blur_Cuda.exe
```


# Build from Source
To build Hipacc from source, either the development packages of **Clang/LLVM**
and **libc++** are required, or the corresponding components from Clang/LLVM
have to be manually installed.

### Get the Sources
The tag to checkout `<tag>` can be found in
[dependencies.sh](dependencies.sh). Steps can also be followed from [Getting Started: Building and Running Clang](https://clang.llvm.org/get_started.html). In the following, all sources are cloned to
directory `<SRC>`:

```bash
cd <SRC>

# Get Hipacc
git clone --recursive https://github.com/hipacc/hipacc.git

# Get LLVM Project (already includes clang, libcxx, compiler-rt etc.)
git clone --branch <tag> https://github.com/llvm/llvm-project.git
```

### Compile the Sources
The build system of Hipacc uses **CMake** (3.4.3 or later), **Git** (2.7 or later),
and requires installation to an absolute path. To configure Hipacc, call `cmake`
in the root directory. A working installation of Clang/LLVM is 
required. The `llvm-config` tool will be used to determine the configuration for
Hipacc and must be present in the environment.

The following variables can be set to configure the Hipacc installation:

Variable               | Meaning
-----------------------|----------------------------------------------------
`CMAKE_BUILD_TYPE`     | Build type (`Debug` or `Release`)
`CMAKE_INSTALL_PREFIX` | Installation prefix (this must be an absolute path)

For OpenCL, the installation location can be specified:

Variable             | Meaning
---------------------|-----------------------------------------------------
`OpenCL_INCLUDE_DIR` | OpenCL include path (e.g. `/usr/local/cuda/include`)
`OpenCL_LIBRARY`     | OpenCL library path (e.g. `/usr/lib64/libOpenCL.so`)

The following options can be enabled or disabled:

Variable           | Meaning
-------------------|----------------------------------------------------------------------
`USE_JIT_ESTIMATE` | Use JIT compilation to get resource estimates (unavailable for macOS)

In the following, all binaries will be installed to `<DST>`.

#### On GNU/Linux and macOS
Note that **Xcode** and the command line tools are required to build Hipacc on
macOS (use `xcode-select --install` to install command line tools).
When configuring LLVM on macOS, the default sysroot (`DEFAULT_SYSROOT`) path
must be set to `` `xcrun --sdk macosx --show-sdk-path` ``.

```bash
# Compile Clang/LLVM on GNU/Linux
mkdir <SRC>/llvm-project/build && cd <SRC>/llvm-project/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST> -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt;lld;polly"
make install

# Compile Clang/LLVM on macOS
mkdir <SRC>/llvm-project/build && cd <SRC>/llvm-project/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST> -DDEFAULT_SYSROOT=`xcrun --sdk macosx --show-sdk-path` -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt;lld;polly"
make install

# Make sure previously built 'llvm-config' is present in PATH
export PATH=<DST>/bin:$PATH

# Compile Hipacc
mkdir <SRC>/hipacc/build && cd <SRC>/hipacc/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST>
make install
```

#### On Windows
Compiling on Windows requires **Visual Studio 2017**. Furthermore, download
**Python** and install it to `<PYTHON_DIR>`.
```PowerShell
# Compile Clang/LLVM
mkdir <SRC>\llvm-project\build; cd <SRC>\llvm-project\build
cmake.exe .. -G "Visual Studio 15 2017 Win64" -Thost=x64 -DCMAKE_INSTALL_PREFIX="<DST>" -DPYTHON_EXECUTABLE="<PYTHON_DIR>\python.exe" -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt;lld;polly"
cmake.exe --build . --target INSTALL

# Install LLVM-vs2017 platform toolset
#  1. Get Toolset-llvm-vs2017-x64.* and Toolset-llvm-vs2017-xp-x64.* from
#     https://github.com/arves100/llvm-vs2017-integration/tree/v8.0
#  2. Copy Toolset-llvm-vs2017-x64.* to [VS2017]/Common7/IDE/VC/VCTargets/Platforms/x64/PlatformToolsets/LLVM-vs2017/Toolset.*
#     Copy Toolset-llvm-vs2017-xp-x64.* to [VS2017]/Common7/IDE/VC/VCTargets/Platforms/x64/PlatformToolsets/LLVM-vs2017_xp/Toolset.*
#  3. Adjust Clang version (probably 8.0.1) in IncludePath and LibraryPath in both *.props files
#  4. Set the path to the Clang installation in the registry:
reg.exe add HKLM\SOFTWARE\WOW6432Node\LLVM\LLVM /t REG_SZ /d "<DST>"

# Compile libc++
# Set comparison "_MSC_VER < 1912" in <SRC>\libcxx\include\__config to version "1900"
mkdir <SRC>\libcxx\build; cd <SRC>\libcxx\build
cmake.exe .. -G "Visual Studio 15 2017 Win64" -T "LLVM-vs2017" -DCMAKE_INSTALL_PREFIX="<DST>" -DLIBCXX_ENABLE_SHARED=YES -DLIBCXX_ENABLE_STATIC=NO -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO
cmake.exe --build . --target INSTALL

# Compile Hipacc
mkdir <SRC>\hipacc\build; cd <SRC>\hipacc\build
cmake.exe .. -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="<DST>"
cmake.exe --build . --target INSTALL
```
