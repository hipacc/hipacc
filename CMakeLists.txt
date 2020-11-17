cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)

# add path for custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(Git)
if(Git_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE HIPACC_VERSION)
    string(STRIP "${HIPACC_VERSION}" HIPACC_VERSION)
endif()

if(NOT HIPACC_VERSION)
    message(FATAL_ERROR "Could not determine version of Hipacc!")
endif()

string(REGEX REPLACE "^v(.*)" "\\1" HIPACC_VERSION "${HIPACC_VERSION}")

string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" HIPACC_VERSION_MAJOR "${HIPACC_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" HIPACC_VERSION_MINOR "${HIPACC_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" HIPACC_VERSION_PATCH "${HIPACC_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" HIPACC_VERSION_TWEAK "${HIPACC_VERSION}")

# get git repository and revision
if(EXISTS ${CMAKE_SOURCE_DIR}/.git)

    execute_process(COMMAND git remote get-url origin
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    TIMEOUT 5
                    RESULT_VARIABLE git_result
                    OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY
                    ERROR_QUIET)

    if(HIPACC_GIT_REPOSITORY)
        string(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)
        string(REGEX REPLACE "://.+@" "://" HIPACC_GIT_REPOSITORY ${HIPACC_GIT_REPOSITORY})
    endif()

endif()

if(NOT HIPACC_GIT_REPOSITORY)
    set(HIPACC_GIT_REPOSITORY "https://github.com/hipacc/hipacc/releases")
endif()

set(HIPACC_GIT_VERSION "${HIPACC_VERSION}")

project(hipacc
        VERSION "${HIPACC_VERSION_MAJOR}.${HIPACC_VERSION_MINOR}.${HIPACC_VERSION_PATCH}"
        HOMEPAGE_URL ${HIPACC_GIT_REPOSITORY})

message("This is Hipacc ${HIPACC_VERSION}")

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options("/W3")
    # move warnings from LLVM 8.0.1 to warning level 4
    add_compile_options("/w44141" "/w44146" "/w44244" "/w44267" "/w44291")
else()
    add_compile_options("-Wall" "-Wunused")
endif()

# provide only Debug and Release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "build config types" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type: Debug or Release" FORCE)
endif()

# if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
#     message(FATAL_ERROR "CMAKE_INSTALL_PREFIX has to be an absolute path!")
# endif()

# add *d to output binaries
set(CMAKE_DEBUG_POSTFIX d)

include(CMakeDependentOption)
cmake_dependent_option(USE_JIT_ESTIMATE "Compile kernels JIT to estimate resource usage" ON "NOT APPLE" OFF)


find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_program(llvm-config NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR})
find_program(clang       NAMES clang       PATHS ${LLVM_TOOLS_BINARY_DIR})
find_package(CUDA)
find_package(OpenCL)

if(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "10.0")
    message(WARNING "At least CUDA version 10.0 required, but found CUDA version ${CUDA_VERSION}.")
    set(CUDA_FOUND FALSE)
endif()

if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
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
message(STATUS "JIT estimates: ${USE_JIT_ESTIMATE}")
message(STATUS "===")


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


# debug and release builds
string(TOUPPER "${CMAKE_BUILD_TYPE}" build)


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

# from LLVM CMake to enable / disable RTTI
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

function(append_if condition value)
  if (${condition})
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endif()
endfunction()

if(NOT WIN32 AND NOT CYGWIN)
    # MinGW warns if -fvisibility-inlines-hidden is used.
    check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    append_if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG "-fvisibility-inlines-hidden" CMAKE_CXX_FLAGS)
endif()
# from LLVM CMake to set visibility

include(InstallRequiredSystemLibraries)
include(CMakePackageConfigHelpers)

install(FILES cmake/hipacc-functions.cmake DESTINATION cmake)

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

# enable static code analysis by cppcheck
include(CppcheckTargets OPTIONAL RESULT_VARIABLE CppcheckTargets_FOUND)
if(CppcheckTargets_FOUND)
    LIST(APPEND CPPCHECK_CUSTOM_ARGUMENTS --std=c++11 -DCLANG_VERSION_MAJOR=10 -D__clang__ -DCUDA_VERSION=10000)
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

# set include directory, add src directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
add_subdirectory(lib)
add_subdirectory(compiler)
add_subdirectory(tools)


# configure header files to pass some of the CMake settings to the source code
configure_file(include/hipacc/Config/config.h.cmake   ${CMAKE_BINARY_DIR}/include/hipacc/Config/config.h)

# install dsl and runtime header files
file(GLOB DSL_HEADERS ${CMAKE_SOURCE_DIR}/dsl/*.hpp)
file(GLOB RUNTIME_HEADERS ${CMAKE_SOURCE_DIR}/runtime/*.hpp
                          ${CMAKE_SOURCE_DIR}/runtime/*.tpp
                          ${CMAKE_BINARY_DIR}/runtime/*.hpp)
install(FILES ${RUNTIME_HEADERS} DESTINATION include COMPONENT headers_runtime)
install(FILES ${DSL_HEADERS} DESTINATION include/dsl COMPONENT headers_dsl)

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

if(WIN32)
    # patch filesystem header from libcxx 8.0.1 (this is solved in later releases)
    file(READ "${CMAKE_BINARY_DIR}/include/c++/v1/filesystem" LIBCXX_FILESYSTEM)
    STRING(REPLACE "_LIBCPP_FUNC_VIS\n  void __create_what" "void __create_what"
        LIBCXX_FILESYSTEM "${LIBCXX_FILESYSTEM}")
    file(WRITE "${CMAKE_BINARY_DIR}/include/c++/v1/filesystem" "${LIBCXX_FILESYSTEM}")
endif()

# add libcxx to package
install(DIRECTORY ${LIBCXX_DST} DESTINATION include/c++ COMPONENT libcxx)


# create release packages for Ubuntu/Windows/macOS
execute_process(COMMAND ${llvm-config} --shared-mode OUTPUT_VARIABLE LLVM_SHARED_MODE OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_BUILD_TYPE MATCHES Release AND LLVM_SHARED_MODE MATCHES static)
    include(PackageHipacc)
endif()


# add samples if available
if(EXISTS ${CMAKE_SOURCE_DIR}/samples-public/CMakeLists.txt)
    add_subdirectory(samples-public)
endif()

