# Hipacc
A domain-specific language and compiler for image processing

Hipacc allows to design image processing kernels and algorithms in a domain-specific language (DSL).
From this high-level description, low-level target code for GPU accelerators is generated using source-to-source translation.
As back ends, the framework supports C/C++, CUDA, OpenCL, and Renderscript.
There is also a fork of Hipacc that targets [FPGAs](https://github.com/hipacc/hipacc-fpga).

## Install
See [Hipacc documentation](http://hipacc-lang.org/install.html) and [Install notes](doc/INSTALL.md) for detailed information.

## Integration

See [Integration](./doc/INTEGRATION.md) for how to add Hipacc to cmake targets.
