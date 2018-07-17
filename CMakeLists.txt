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

if (WIN32)
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


# set repository and revision commands
if (WIN32)
    set(CMD_GET_REPO git remote get-url origin)
    set(CMD_GET_VERSION git log -1 --pretty=format:%H)
else()
    set(CMD_GET_REPO sh ${CMAKE_SOURCE_DIR}/cmake/scripts/get_repository_path.sh)
    set(CMD_GET_VERSION sh ${CMAKE_SOURCE_DIR}/cmake/scripts/get_source_version.sh)
endif()


# get repository and revision
execute_process(COMMAND ${CMD_GET_REPO}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY)
string(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)

execute_process(COMMAND ${CMD_GET_VERSION}
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
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
add_subdirectory(lib)
add_subdirectory(compiler)
add_subdirectory(tools)


# configure header files to pass some of the CMake settings to the source code
configure_file(include/hipacc/Config/config.h.cmake   ${CMAKE_BINARY_DIR}/include/hipacc/Config/config.h)
configure_file(runtime/hipacc_cu_standalone.hpp.cmake ${CMAKE_BINARY_DIR}/runtime/hipacc_cu_standalone.hpp)


# install dsl and runtime header files
file(GLOB DSL_HEADERS ${CMAKE_SOURCE_DIR}/dsl/*.hpp)
file(GLOB RUNTIME_HEADERS ${CMAKE_SOURCE_DIR}/runtime/*.hpp
                          ${CMAKE_SOURCE_DIR}/runtime/*.tpp
                          ${CMAKE_BINARY_DIR}/runtime/*.hpp)
install(FILES ${RUNTIME_HEADERS} DESTINATION include
        COMPONENT headers)
install(FILES ${DSL_HEADERS} DESTINATION include/dsl
        COMPONENT headers)


# configure and install tests
if (WIN32)

configure_file(tests/vs2015/example_blur/example_blur/example_blur.vcxproj.cmake
               ${CMAKE_BINARY_DIR}/tests/vs2015/example_blur/example_blur/example_blur.vcxproj)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/vs2015
    DESTINATION tests
    COMPONENT tests
    PATTERN "*.cmake" EXCLUDE)
install(DIRECTORY ${CMAKE_BINARY_DIR}/tests/vs2015
    DESTINATION tests
    COMPONENT tests)

else()

configure_file(tests/Android.mk.cmake                 ${CMAKE_BINARY_DIR}/tests/Android.mk)
configure_file(tests/Application.mk.cmake             ${CMAKE_BINARY_DIR}/tests/Application.mk)
configure_file(tests/Makefile.cmake                   ${CMAKE_BINARY_DIR}/tests/Makefile)
configure_file(tests/Makefile_test.cmake              ${CMAKE_BINARY_DIR}/tests/Makefile_test)
configure_file(tests/Makefile_test_opencv.cmake       ${CMAKE_BINARY_DIR}/tests/Makefile_test_opencv)
configure_file(tests/CMakeLists.txt.cmake             ${CMAKE_BINARY_DIR}/tests/CMakeLists.txt @ONLY)

install(DIRECTORY tests
    DESTINATION .
    COMPONENT tests
    PATTERN "*.cmake" EXCLUDE)
install(DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    DESTINATION .
    COMPONENT tests
    PATTERN "*Makefile_test*" EXCLUDE)

# install Makefiles for test cases
file(GLOB TEST_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/*)
foreach(DIR IN LISTS TEST_DIRS)
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/${DIR})
        if(${DIR} MATCHES "opencv_*")
            install(FILES ${CMAKE_BINARY_DIR}/tests/Makefile_test_opencv
                DESTINATION tests/${DIR}
                COMPONENT tests
                RENAME Makefile)
        else()
            install(FILES ${CMAKE_BINARY_DIR}/tests/Makefile_test
                DESTINATION tests/${DIR}
                COMPONENT tests
                RENAME Makefile)
        endif()
    endif()
endforeach()

endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
    # Setup packaging variables
    set (CPACK_PACKAGE_NAME Hipacc)
    set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Hipacc Framework")
    set (CPACK_PACKAGE_VERSION_MAJOR ${HIPACC_MAJOR_VERSION})
    set (CPACK_PACKAGE_VERSION_MINOR ${HIPACC_MINOR_VERSION})
    set (CPACK_PACKAGE_VERSION_PATCH ${HIPACC_PATCH_VERSION})
    set (CPACK_PACKAGE_VERSION ${HIPACC_VERSION})
    set (CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)

    # Setup Debian packaging options
    if (UNIX AND NOT APPLE)
        set (CPACK_GENERATOR DEB)
        #set (CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv (>= 2.4.0)")
        set (CPACK_DEBIAN_PACKAGE_MAINTAINER "Richard Membarth, Oliver Reiche, M. Akif Özkan, Bo Qiao")
        set (CPACK_DEBIAN_PACKAGE_SECTION "devel")

        if(DEFINED PLATFORM)
            if (PLATFORM MATCHES 32)
                set (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
            elseif (PLATFORM MATCHES 64)
                set (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
            endif (PLATFORM MATCHES 32)
        else (DEFINED PLATFORM)
            find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
            if(DPKG_PROGRAM)
                execute_process(
                    COMMAND ${DPKG_PROGRAM} --print-architecture
                    OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            endif(DPKG_PROGRAM)
        endif(DEFINED PLATFORM)

        string(TOLOWER "${CPACK_PACKAGE_NAME}" CPACK_PACKAGE_NAME_LOWERCASE)

        set(CPACK_PACKAGE_FILE_NAME
            "${CPACK_PACKAGE_NAME_LOWERCASE}_${HIPACC_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

        if(NOT HIPACC_PACKAGING_PREFIX)
            set(HIPACC_PACKAGING_PREFIX "/usr/local")
        endif()
        set(CPACK_PACKAGING_INSTALL_PREFIX "${HIPACC_PACKAGING_PREFIX}/${CPACK_PACKAGE_NAME_LOWERCASE}-${HIPACC_VERSION}")
        set(HIPACC_PACKAGE_SYMLINK "${HIPACC_PACKAGING_PREFIX}/${CPACK_PACKAGE_NAME_LOWERCASE}")

        # postinst script for creating symlink and profile file
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/postinst "\
ln -s ${CPACK_PACKAGING_INSTALL_PREFIX} ${HIPACC_PACKAGE_SYMLINK}
echo \"export PATH=${HIPACC_PACKAGE_SYMLINK}/bin:\\\$PATH\" > /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh
echo \"export CPLUS_INCLUDE_PATH=${HIPACC_PACKAGE_SYMLINK}/include:\\\$CPLUS_INCLUDE_PATH\" >> /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh
echo \"export LIBRARY_PATH=${HIPACC_PACKAGE_SYMLINK}/lib:\\\$LIBRARY_PATH\" >> /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh")

        # postrm script for cleaning symlink and profile file
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/postrm "\
rm -f ${HIPACC_PACKAGE_SYMLINK}
rm -f /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh")

        # add extra scripts
        set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
            ${CMAKE_CURRENT_BINARY_DIR}/postinst;
            ${CMAKE_CURRENT_BINARY_DIR}/postrm)

        # enable component install for Debian
        set(CPACK_DEB_COMPONENT_INSTALL ON)
    endif (UNIX AND NOT APPLE)

    # Setup Windows Nullsoft Scriptable Install System
    if (WIN32)
        set (CPACK_GENERATOR NSIS)
        set(CPACK_PACKAGE_INSTALL_DIRECTORY
            ${CPACK_PACKAGE_NAME}-${HIPACC_VERSION})
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/templates)

        # setup icons and banner
        set(CPACK_NSIS_MUI_ICON ${CMAKE_SOURCE_DIR}/res/hipacc.ico)
        set(CPACK_NSIS_MUI_UNIICON ${CMAKE_SOURCE_DIR}/res/hipacc.ico)
        set(CPACK_NSIS_MUI_HEADERIMAGE ${CMAKE_SOURCE_DIR}/res/hipacc_banner.bmp)
        string(REPLACE "/" "\\\\"
               CPACK_NSIS_MUI_HEADERIMAGE "${CPACK_NSIS_MUI_HEADERIMAGE}")

        # set HIPACC_PATH and append to PATH environment variables
        set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "\
StrCpy \$ADD_TO_PATH_ALL_USERS \\\"1\\\"
Push \\\"HIPACC_PATH\\\"
Push \\\"\$INSTDIR\\\"
Call AddToEnvVar
Push \\\"\$INSTDIR/bin\\\"
Call AddToPath")

        # unset HIPACC_PATH and remove from PATH environment variables
        set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "\
StrCpy \$ADD_TO_PATH_ALL_USERS \\\"1\\\"
Push \\\"HIPACC_PATH\\\"
Push \\\"\$INSTDIR\\\"
Call un.RemoveFromEnvVar
Push \\\"\$INSTDIR/bin\\\"
Call un.RemoveFromPath")
    endif (WIN32)

    # Copy system's libcxx to current build dir
    set(LIBCXX_SRC ${LLVM_INCLUDE_DIRS}/c++)
    set(LIBCXX_DST ${CMAKE_BINARY_DIR}/include/c++)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCXX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBCXX_SRC} ${LIBCXX_DST})

    # Add libcxx to package
    install(DIRECTORY ${LIBCXX_DST}
            DESTINATION include
            COMPONENT libcxx)

    # Only package compiler, headers, and runtime, without tools and tests
    set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
    set(CPACK_COMPONENTS_ALL compiler headers runtime libcxx)

    include (CPack)
endif()

