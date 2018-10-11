cmake_minimum_required(VERSION 3.1)

project(hipacc)

set(HIPACC_MAJOR_VERSION 0)
set(HIPACC_MINOR_VERSION 8)
set(HIPACC_PATCH_VERSION 3)
set(HIPACC_VERSION ${HIPACC_MAJOR_VERSION}.${HIPACC_MINOR_VERSION}.${HIPACC_PATCH_VERSION})

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions("/W2")
else()
    add_definitions("-Wall -Wunused")
endif()

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

# get git repository and revision
if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
    execute_process(COMMAND git remote get-url origin     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} TIMEOUT 5 RESULT_VARIABLE git_result OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY)
    execute_process(COMMAND git log -1 --pretty=format:%H WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} TIMEOUT 5 RESULT_VARIABLE git_result OUTPUT_VARIABLE HIPACC_GIT_VERSION)
    string(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)
    string(STRIP ${HIPACC_GIT_VERSION}    HIPACC_GIT_VERSION)
else()
    set(HIPACC_GIT_REPOSITORY "https://github.com/hipacc/hipacc/releases")
    set(HIPACC_GIT_VERSION    "v${HIPACC_VERSION}")
endif()

# add path for custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED)
find_program(llvm-config NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR})
find_program(clang       NAMES clang       PATHS ${LLVM_TOOLS_BINARY_DIR})
find_package(CUDA)
find_package(NVML)
find_package(OpenCL)

if(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "7.0")
    message(WARNING "At least CUDA version 7.0 required, but found CUDA version ${CUDA_VERSION}.")
    set(CUDA_FOUND FALSE)
endif()

if(CUDA_FOUND)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    set(CU_COMPILER ${NVCC})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    find_library(CUDA_NVRTC_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib       ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                                ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    if(CUDA_NVRTC_LIBRARY)
        set(NVRTC_FOUND TRUE)
    endif()
else()
    set(USE_JIT_ESTIMATE OFF)
endif()

if(OpenCL_FOUND)
    set(CL_COMPILER "${CMAKE_INSTALL_PREFIX}/bin/cl_compile")
endif()

set(RUNTIME_INCLUDES "${CMAKE_INSTALL_PREFIX}/include")

message(STATUS "Configuration summary:")
message(STATUS "===")
message(STATUS "CUDA support: ${CUDA_FOUND}")
message(STATUS "OpenCL support: ${OpenCL_FOUND}")
message(STATUS "Polly support: ${USE_POLLY}")
message(STATUS "JIT estimates: ${USE_JIT_ESTIMATE}")
message(STATUS "===")


# from LLVM CMake to enable / disable RTTI
if(NOT DEFINED LLVM_COMPILER_IS_GCC_COMPATIBLE)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(LLVM_COMPILER_IS_GCC_COMPATIBLE ON)
    elseif(MSVC)
        set(LLVM_COMPILER_IS_GCC_COMPATIBLE OFF)
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(LLVM_COMPILER_IS_GCC_COMPATIBLE ON)
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
        set(LLVM_COMPILER_IS_GCC_COMPATIBLE ON)
    endif()
endif()

if(NOT LLVM_ENABLE_RTTI)
    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
        list(APPEND CMAKE_CXX_FLAGS "-fno-rtti")
    elseif(MSVC)
        list(APPEND CMAKE_CXX_FLAGS "/GR-")
    endif()
elseif(MSVC)
    list(APPEND CMAKE_CXX_FLAGS "/GR")
endif()
# from LLVM CMake to enable / disable RTTI


# set include directory, add src directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/submodules ${CMAKE_BINARY_DIR}/include)
add_subdirectory(lib)
add_subdirectory(compiler)
add_subdirectory(tools)


# configure header files to pass some of the CMake settings to the source code
configure_file(include/hipacc/Config/config.h.cmake   ${CMAKE_BINARY_DIR}/include/hipacc/Config/config.h)
configure_file(runtime/hipacc_cu.hpp.cmake            ${CMAKE_BINARY_DIR}/runtime/hipacc_cu.hpp)
configure_file(runtime/hipacc_cu_standalone.hpp.cmake ${CMAKE_BINARY_DIR}/runtime/hipacc_cu_standalone.hpp)

# configure build tools
configure_file(buildtools/unix/Makefile.in ${CMAKE_BINARY_DIR}/buildtools/unix/Makefile)


# install dsl and runtime header files
file(GLOB DSL_HEADERS ${CMAKE_SOURCE_DIR}/dsl/*.hpp)
file(GLOB RUNTIME_HEADERS ${CMAKE_SOURCE_DIR}/runtime/*.hpp
                          ${CMAKE_SOURCE_DIR}/runtime/*.tpp
                          ${CMAKE_BINARY_DIR}/runtime/*.hpp)
install(FILES ${RUNTIME_HEADERS} DESTINATION include COMPONENT headers_runtime)
install(FILES ${DSL_HEADERS} DESTINATION include/dsl COMPONENT headers_dsl)


# install samples
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples DESTINATION . COMPONENT samples)

# deploy build tool for samples
file(GLOB SAMPLE_DIRS RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/samples/[0-9]*/*)
foreach(SAMPLE_DIR IN LISTS SAMPLE_DIRS)
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/${SAMPLE_DIR})
        if(UNIX)
            install(FILES ${CMAKE_BINARY_DIR}/buildtools/unix/Makefile DESTINATION ${SAMPLE_DIR} COMPONENT samples)
        elseif(WIN32)
            file(GLOB NMAKE_FILES ${CMAKE_SOURCE_DIR}/buildtools/nmake/*)
            file(GLOB VS2015_FILES ${CMAKE_SOURCE_DIR}/buildtools/vs2015/*)
            install(FILES ${NMAKE_FILES} ${VS2015_FILES} DESTINATION ${SAMPLE_DIR} COMPONENT samples)
        endif()
    endif()
endforeach()


# copy system's clang headers to current build dir
execute_process(COMMAND ${clang} -print-file-name=include OUTPUT_VARIABLE CLANG_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CLANG_HEADERS_SRC ${CLANG_INCLUDE_DIRS})
set(CLANG_HEADERS_DST ${CMAKE_BINARY_DIR}/include/clang)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CLANG_HEADERS_DST}
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CLANG_HEADERS_SRC} ${CLANG_HEADERS_DST})

# add clang headers to package
install(DIRECTORY ${CLANG_HEADERS_DST} DESTINATION include COMPONENT headers_clang)

# copy system's libcxx to current build dir
set(LIBCXX_SRC ${LLVM_INCLUDE_DIRS}/c++/v1)
set(LIBCXX_DST ${CMAKE_BINARY_DIR}/include/c++/v1)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCXX_DST} COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBCXX_SRC} ${LIBCXX_DST})

# add libcxx to package
install(DIRECTORY ${LIBCXX_DST} DESTINATION include/c++ COMPONENT libcxx)

# create release packages for Ubuntu/Windows/macOS
execute_process(COMMAND ${llvm-config} --shared-mode OUTPUT_VARIABLE LLVM_SHARED_MODE OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_BUILD_TYPE MATCHES Release AND LLVM_SHARED_MODE MATCHES static)
    include(PackageHipacc)
endif()
