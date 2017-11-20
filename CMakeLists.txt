cmake_minimum_required(VERSION 3.1)

project(hipacc)

set(HIPACC_MAJOR_VERSION 0)
set(HIPACC_MINOR_VERSION 8)
set(HIPACC_PATCH_VERSION 2)
set(HIPACC_VERSION ${HIPACC_MAJOR_VERSION}.${HIPACC_MINOR_VERSION}.${HIPACC_PATCH_VERSION})

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions("-Wall -Wunused")

# provide only Debug and Release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "build config types" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type: Debug or Release" FORCE)
endif()

if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX has to be an absolute path!")
endif()

option(USE_POLLY "Use Polly for analysis" OFF)
include(CMakeDependentOption)
cmake_dependent_option(USE_JIT_ESTIMATE "Compile kernels JIT to estimate resource usage" ON "NOT APPLE" OFF)


# get repository and revision
execute_process(COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/scripts/get_repository_path.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY)
string(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)

execute_process(COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/scripts/get_source_version.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE HIPACC_GIT_VERSION)
string(STRIP ${HIPACC_GIT_VERSION} HIPACC_GIT_VERSION)


# add path for custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_library(RT_LIBRARIES rt)
if(NOT RT_LIBRARIES)
    set(RT_LIBRARIES "")
endif()
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED)
find_program(llvm-config NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR})
find_program(clang       NAMES clang       PATHS ${LLVM_TOOLS_BINARY_DIR})
find_package(CUDA)
find_package(NVML)
find_package(OpenCL)
find_package(RenderScript)
find_package(OpenCV)
find_package(Threads)
if(THREADS_HAVE_PTHREAD_ARG)
    set(THREADS_ARG "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    set(THREADS_LINK "${CMAKE_THREAD_LIBS_INIT}")
endif()

if(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "7.0")
    message(WARNING "At least CUDA version 7.0 required, but found CUDA version ${CUDA_VERSION}.")
    set(CUDA_FOUND FALSE)
endif()

if(CUDA_FOUND)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    if(APPLE)
        set(NVCC_LINK "-Xlinker -lnvrtc -Xlinker -framework,CUDA")
        set(NVCC_COMP "-ccbin ${clang}")
    else()
        set(NVCC_LINK "${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY} ${NVML_LIBRARIES}")
        find_library(CUDA_NVRTC_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        if(CUDA_NVRTC_LIBRARY)
            set(NVCC_LINK "${NVCC_LINK} ${CUDA_NVRTC_LIBRARY}")
            set(NVRTC_FOUND TRUE)
        endif()
        set(NVCC_COMP "-I${NVML_INCLUDE_DIRS}")
    endif()
else()
    set(USE_JIT_ESTIMATE OFF)
endif()

if(OpenCL_FOUND)
    set(CL_COMPILER "${CMAKE_INSTALL_PREFIX}/bin/cl_compile")
endif()

message(STATUS "Configuration summary:")
message(STATUS "===")
message(STATUS "CUDA support: ${CUDA_FOUND}")
message(STATUS "OpenCL support: ${OpenCL_FOUND}")
message(STATUS "Renderscript support: ${Renderscript_FOUND}")
message(STATUS "OpenCV support: ${OpenCV_FOUND}")
message(STATUS "Polly support: ${USE_POLLY}")
message(STATUS "JIT estimates: ${USE_JIT_ESTIMATE}")
message(STATUS "===")


# set include directory, add src directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
add_subdirectory(lib)
add_subdirectory(compiler)
add_subdirectory(tools)


# configure header files to pass some of the CMake settings to the source code
configure_file(include/hipacc/Config/config.h.cmake ${CMAKE_BINARY_DIR}/include/hipacc/Config/config.h)
configure_file(runtime/hipacc_cu.hpp.cmake          ${CMAKE_BINARY_DIR}/runtime/hipacc_cu.hpp)
configure_file(tests/Android.mk.cmake               ${CMAKE_BINARY_DIR}/tests/Android.mk)
configure_file(tests/Application.mk.cmake           ${CMAKE_BINARY_DIR}/tests/Application.mk)
configure_file(tests/Makefile.cmake                 ${CMAKE_BINARY_DIR}/tests/Makefile)
configure_file(tests/Makefile_test.cmake            ${CMAKE_BINARY_DIR}/tests/Makefile_test)
configure_file(tests/Makefile_test_opencv.cmake     ${CMAKE_BINARY_DIR}/tests/Makefile_test_opencv)
configure_file(tests/CMakeLists.txt.cmake           ${CMAKE_BINARY_DIR}/tests/CMakeLists.txt @ONLY)


# install dsl and runtime header files
file(GLOB DSL_HEADERS ${CMAKE_SOURCE_DIR}/dsl/*.hpp)
file(GLOB RUNTIME_HEADERS ${CMAKE_SOURCE_DIR}/runtime/*.hpp ${CMAKE_BINARY_DIR}/runtime/*.hpp)
install(FILES ${RUNTIME_HEADERS} DESTINATION include)
install(FILES ${DSL_HEADERS} DESTINATION include/dsl)

# install tests
install(DIRECTORY tests
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "*.cmake" EXCLUDE)
install(DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "*Makefile_test*" EXCLUDE)

# install Makefiles for test cases
file(GLOB TEST_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/*)
foreach(DIR IN LISTS TEST_DIRS)
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/${DIR})
        if(${DIR} MATCHES "opencv_*")
            install(FILES ${CMAKE_BINARY_DIR}/tests/Makefile_test_opencv
                DESTINATION tests/${DIR}
                RENAME Makefile)
        else()
            install(FILES ${CMAKE_BINARY_DIR}/tests/Makefile_test
                DESTINATION tests/${DIR}
                RENAME Makefile)
        endif()
    endif()
endforeach()
