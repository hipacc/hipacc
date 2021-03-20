# ./CMakeLists.txt

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# add path for custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

include(VersionHipacc)
get_hipacc_version()
message(STATUS "This is Hipacc ${HIPACC_VERSION}")
message(STATUS "This is HIPACC_GIT_REPOSITORY ${HIPACC_GIT_REPOSITORY}")
message(STATUS "This is HIPACC_GIT_VERSION ${HIPACC_GIT_VERSION}")

project(hipacc
        VERSION "${HIPACC_VERSION_MAJOR}.${HIPACC_VERSION_MINOR}.${HIPACC_VERSION_PATCH}"
        DESCRIPTION "A Domain-Specific Language and Compiler for Image Processing"
        HOMEPAGE_URL ${HIPACC_GIT_REPOSITORY}
        LANGUAGES CXX)

# global setup flags that don't want to be localized
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(-Wall -Wunused)
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_program(llvm-config NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR})
find_program(clang       NAMES clang       PATHS ${LLVM_TOOLS_BINARY_DIR})
find_package(CUDA 10.0)
find_package(OpenCL)

message(STATUS "Configuration summary:")
message(STATUS "===")
message(STATUS "CUDA support: ${CUDA_FOUND}")
message(STATUS "OpenCL support: ${OpenCL_FOUND}")
message(STATUS "===")

# TODO: still required?
# platform specific fixes linker flags
if (MSVC)
    # Correct syntax for LLVM libraries
    string(STRIP "${LLVM_MODULE_LIBS}" LLVM_MODULE_LIBS)
    string(REPLACE "-lLLVM" "LLVM" LLVM_MODULE_LIBS "${LLVM_MODULE_LIBS}")
    string(REPLACE " " ";" LLVM_MODULE_LIBS "${LLVM_MODULE_LIBS}")

    # Erase incompatible LLVM linker flags
    set(LLVM_LFLAGS "")

    # Additional linker settings
    set(CUDA_LINK "")
    set(CUDA_COMP "")
    set(TIME_LINK "")
endif()

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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    elseif(MSVC)
        string(REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
    endif()
elseif(MSVC)
    string(REPLACE "/GR-" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
endif()

# from LLVM CMake to set visibility
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# enable static code analysis by cppcheck
include(CppcheckTargets OPTIONAL RESULT_VARIABLE CppcheckTargets_FOUND)
if(CppcheckTargets_FOUND)
  LIST(APPEND CPPCHECK_CUSTOM_ARGUMENTS --std=c++${CMAKE_CXX_STANDARD} -DCLANG_VERSION_MAJOR=10 -D__clang__ -DCUDA_VERSION=10000)
else()
    # stub macro, so we do not need to check CppcheckTargets_FOUND everywhere add_cppcheck() is used
    macro(add_cppcheck)
    endmacro()
endif()

# enable code coverage measurement
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    option(HIPACC_ENABLE_CODECOVERAGE "Enable Code Coverage Measurement" OFF)
    if(HIPACC_ENABLE_CODECOVERAGE)
        include(CodeCoverage OPTIONAL RESULT_VARIABLE CodeCoverage_FOUND)
        if(CodeCoverage_FOUND)
            append_coverage_compiler_flags()
        endif()
    endif()
endif()

add_subdirectory(compiler)
add_subdirectory(lib)
add_subdirectory(tools)


include(CMakePackageConfigHelpers)
configure_package_config_file (
    cmake/hipacc-config.cmake.in
    "${CMAKE_BINARY_DIR}/export/hipacc-config.cmake"
    INSTALL_DESTINATION cmake)
write_basic_package_version_file (
    "${CMAKE_BINARY_DIR}/export/hipacc-config-version.cmake"
    COMPATIBILITY SameMajorVersion
)

install(FILES cmake/hipacc-functions.cmake DESTINATION cmake COMPONENT cmake)
install(FILES "${CMAKE_BINARY_DIR}/export/hipacc-config.cmake" DESTINATION cmake COMPONENT cmake)
install(FILES "${CMAKE_BINARY_DIR}/export/hipacc-config-version.cmake" DESTINATION cmake COMPONENT cmake)

# install dsl and runtime header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/runtime/ DESTINATION include COMPONENT headers_runtime)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/dsl/ DESTINATION include/dsl COMPONENT headers_dsl)

# copy system's clang headers to current build dir
execute_process(COMMAND ${clang} -print-file-name=include
                OUTPUT_VARIABLE CLANG_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
install(DIRECTORY ${CLANG_INCLUDE_DIRS}/ DESTINATION include/clang COMPONENT headers_clang)

# copy system's libcxx to current build dir
install(DIRECTORY ${LLVM_INCLUDE_DIRS}/c++/v1/ DESTINATION include/c++/v1 COMPONENT libcxx)

# create doxygen target for dsl and runtime
include(DoxygenTargets OPTIONAL RESULT_VARIABLE DoxygenTargets_FOUND)
if(DoxygenTargets_FOUND)
    add_doxygen(Doxyfile
                DOC_TARGET Doxygen
                EXTRA_INPUT ${CMAKE_SOURCE_DIR}/runtime/ ${CMAKE_SOURCE_DIR}/dsl/
                OUTPUT_DIRECTORY doxygen
                INSTALL_DESTINATION doxygen
                INSTALL_COMPONENT documentation)
endif()

# add *d to output binaries
set(CMAKE_DEBUG_POSTFIX d)

# create release packages for Ubuntu/Windows/macOS
execute_process(COMMAND ${llvm-config} --shared-mode
                OUTPUT_VARIABLE LLVM_SHARED_MODE OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_BUILD_TYPE MATCHES Release AND LLVM_SHARED_MODE MATCHES static)
    include(PackageHipacc)
endif()

# add samples if available
if(EXISTS ${CMAKE_SOURCE_DIR}/samples-public/CMakeLists.txt)
    add_subdirectory(samples-public)
endif()

