# Integration

The Hipacc build system is based on [cmake](https://cmake.org). Therefore, Hipacc provides cmake scripts to seamlessly integrate it into your own project.

CMake can locate the Hipacc package like this:

```cmake
find_package(Hipacc HINTS "$ENV{HIPACC_PATH}/cmake")
```

Optionally, one can request a specific version of Hipacc. See the [cmake documentation](https://cmake.org/cmake/help/latest/command/find_package.html) for more details.

The Hipacc package provides the cmake function `add_hipacc_sources` with which one can add a Hipacc DSL file to the build chain of a specific target. CMake will then automatically run Hipacc in order to generate the target device specific source code which is then compiled to the actual binary.

This is an example for a CUDA target:

```cmake
project(CudaExample LANGUAGES CXX CUDA)

add_executable(CudaExample)

add_hipacc_sources(TARGET CudaExample PRIVATE
                   TARGET_ARCH CUDA
                   HIPACC_MIN_VERSION "0.9.0"
                   SOURCES      "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp"
                   INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/"
                   OPTIONS -target Kepler-30
                           -use-config 128x1
                           -reduce-config 16x16
                           -use-textures off
                           -use-local off
                           -vectorize off
                           -pixels-per-thread 1)
```

The `add_hipacc_sources` function supports the following parameters:

* `TARGET`: Target to which the sources have to be added (executable, static or shared library). In case of a CUDA target, the project of the `TARGET` must have CUDA enabled with `project(CudaExample LANGUAGES CXX CUDA)`.
* `PUBLIC | PRIVATE | INTERFACE`: Scope with which the sources as well as the Hipacc runtime library are added to the `TARGET`.
* `TARGET_ARCH`: The target's device architecture. Currently, `CPU`, `OPENCL-ACC`, `OPENCL-CPU`, `OPENCL-GPU` and `CUDA` are supported.
* `HIPACC_EXACT_VERSION`: Exactly required version of Hipacc.
* `HIPACC_MIN_VERSION`: Minimum required version of Hipacc.
* `SOURCES`: List of DSL source files to add to the `TARGET`.
* `INCLUDE_DIRS`: List of include directories, Hipacc have to consider.
* `OPTIONS`: List of options to be passed to Hipacc.
* `OUTPUT_DIR_VAR`: Variable which is to be set by `add_hipacc_sources` to the output directory of the generated sources.

In addition to the device-specific Hipacc options passed to `add_hipacc_sources`, one can define Hipacc options globally by setting the following variables:

* `HIPACC_OPTIONS`: Hipacc options passed to targets of all kinds of devices.
* `HIPACC_OPTIONS_CPU`: Hipacc options passed to `CPU` targets.
* `HIPACC_OPTIONS_CUDA`: Hipacc options passed to `CUDA` targets.
* `HIPACC_OPTIONS_OPENCL_ACC`: Hipacc options passed to `OPENCL-ACC` targets.
* `HIPACC_OPTIONS_OPENCL_CPU`: Hipacc options passed to `OPENCL-CPU` targets.
* `HIPACC_OPTIONS_OPENCL_GPU`: Hipacc options passed to `OPENCL-GPU` targets.
* `HIPACC_INCLUDE_DIRS`: List of include directories for all Hipacc sources.
* `HIPACC_RT_INCLUDE_DIRS`: List of runtime include directories. Those are not required for Hipacc but for the generated sources.
