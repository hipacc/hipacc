# Installation
Binary packages can be found [here](../../releases).
### On Ubuntu
Download the Debian package and run `sudo dpkg -i hipacc_<version>_amd64.deb`.
### On Windows
Download and run the installer `Hipacc-<version>-win64.exe`.


# Run Samples
### On Ubuntu
Open a terminal and run:
```bash
cp -r /usr/local/hipacc/samples .
cd samples/1_Local_Operators/Box_Blur
# Compile and run
make
# Compile and run for CUDA
make cuda
```
### On Windows
###### With Visual Studio 2015 and 2017
Navigate to sample directory `%HIPACC_PATH%\samples\1_Local_Operators\Box_Blur`,
open the project `sample.vcxproj`, and run build.

###### With Visual C++ Build Tools 2015
To run the samples without Visual Studio, install the latest **Windows SDK** and
**VC++ 2015 toolset** provided by
[Build Tools for Visual Studio 2017](https://aka.ms/buildtools).
Open PowerShell and run:
```PowerShell
cp -r "$Env:HIPACC_PATH\samples" .
cd .\samples\1_Local_Operators\Box_Blur
# Compile and run
.\make.bat
# Compile and run for CUDA
.\make.bat cuda
```


# Build from Source
To build Hipacc from source, either the development packages of **Clang/LLVM**
and **libc++** are required, or the corresponding components from Clang/LLVM
have to be manually installed.

### Get the Sources
The revision to checkout `<rev>` can be found in
[dependencies.sh](dependencies.sh). In the following, all sources are cloned to
directory `<SRC>`:

```bash
cd <SRC>

# Get Hipacc
git clone https://github.com/hipacc/hipacc.git

# Get libc++
git clone --branch <rev> http://llvm.org/git/libcxx.git

# Get LLVM
git clone --branch <rev> http://llvm.org/git/llvm.git

# Get Clang
cd <SRC>/llvm/tools
git clone --branch <rev> http://llvm.org/git/clang.git

# Apply patches if present
cd <SRC>/llvm/tools/clang
git apply <SRC>/hipacc/patches/clang-<rev>.patch
```

### Compile the Sources
The build system of Hipacc uses **CMake** (3.1 or later) and requires
installation to an absolute path. To configure Hipacc, call `cmake` in the
root directory. A working installation of Clang/LLVM (and Polly) is required.
The `llvm-config` tool will be used to determine the configuration for Hipacc
and must be present in the environment.

The following variables can be set to configure the Hipacc installation:

Variable               | Meaning
-----------------------|----------------------------------------------------
`CMAKE_BUILD_TYPE`     | Build type (`Debug` or `Release`)
`CMAKE_INSTALL_PREFIX` | Installation prefix (this must be an absolute path)

For OpenCL, the installation location can be specified:

Variable         | Meaning
-----------------|-----------------------------------------------------
`OpenCL_INC_DIR` | OpenCL include path (e.g. `/usr/local/cuda/include`)
`OpenCL_LIB_DIR` | OpenCL library path (e.g. `/usr/lib64`)

The following options can be enabled or disabled:

Variable           | Meaning
-------------------|----------------------------------------------------------------------
`USE_POLLY`        | Use Polly for kernel analysis (e.g. `-DUSE_POLLY=ON`)
`USE_JIT_ESTIMATE` | Use JIT compilation to get resource estimates (unavailable for macOS)

In the following, all binaries will be installed to `<DST>`.

#### On Linux and macOS
Note that **Xcode** and the command line tools are required to build Hipacc on
macOS (use `xcode-select --install` to install command line tools).

```bash
# Compile Clang/LLVM
mkdir <SRC>/llvm/build && cd <SRC>/llvm/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST>
make install

# Compile libc++
mkdir <SRC>/libcxx/build && cd <SRC>/libcxx/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST>
make install

# Make sure previously built 'llvm-config' is present in PATH
export PATH=<DST>/bin:$PATH

# Compile Hipacc
mkdir <SRC>/hipacc/build && cd <SRC>/hipacc/build
cmake .. -DCMAKE_INSTALL_PREFIX=<DST>
make install
```

#### On Windows
Compiling on Windows requires **Visual Studio 2015**. Furthermore, download
**Python** and install it to `<PYTHON_DIR>`.
```PowerShell
# Compile Clang/LLVM
mkdir <SRC>\llvm\build; cd <SRC>\llvm\build
cmake.exe .. -G "Visual Studio 14 2015 Win64" -Thost=x64 -DCMAKE_INSTALL_PREFIX="<DST>" -DPYTHON_EXECUTABLE="<PYTHON_DIR>\python.exe"
cmake.exe --build . --target INSTALL

# Install LLVM-vs2014 platform toolset
<DST>\tools\msbuild\install.bat
reg.exe add HKLM\SOFTWARE\WOW6432Node\LLVM\LLVM /t REG_SZ /d "<DST>"

# Compile libc++
mkdir <SRC>\libcxx\build; cd <SRC>\libcxx\build
cmake.exe .. -G "Visual Studio 14 2015 Win64" -T "LLVM-vs2014" -DCMAKE_INSTALL_PREFIX="<DST>" -DLIBCXX_ENABLE_SHARED=YES -DLIBCXX_ENABLE_STATIC=NO -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO
cmake.exe --build . --target INSTALL

# Compile Hipacc
mkdir <SRC>\hipacc\build; cd <SRC>\hipacc\build
cmake.exe .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX="<DST>"
cmake.exe --build . --target INSTALL
```
