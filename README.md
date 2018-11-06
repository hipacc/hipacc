# Hipacc
A domain-specific language and compiler for image processing

Hipacc allows to design image processing kernels and algorithms in a domain-specific language (DSL).
From this high-level description, low-level target code for GPU accelerators is generated using source-to-source translation.
As back ends, the framework supports C/C++, CUDA, OpenCL, and Renderscript.
There is also a fork of Hipacc that targets [FPGAs](https://github.com/hipacc/hipacc-fpga).

# Install
See [Hipacc documentation](http://hipacc-lang.org/install.html) and [Install notes](INSTALL.md) for detailed information.

# use Kernel Fusion
* clone this branch and get the submodule samples
```bash
git clone -b kernel_fusion https://github.com/hipacc/hipacc.git
cd hipacc
git submodule init && git submodule update
```
* build Hipacc from source
```bash
mkdir build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/release
make && make install
```
* run Harris corner (without kernel fusion by default)
```bash
cd release/samples/3_Preprocessing/Harris_Corner/
make cuda
```
* enable fusion flag
```bash
cd "inside Harris_Corner folder"/../../common/config  
change '-fuse off' to '-fuse on' in cuda.conf 
```
