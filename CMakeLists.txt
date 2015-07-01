CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

#SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# provide only Debug and Release configurations
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "build config types" FORCE)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug CACHE STRING "build type: Debug or Release" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

IF(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    MESSAGE(FATAL_ERROR "CMAKE_INSTALL_PREFIX has to be an absolute path!")
ENDIF(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})


PROJECT(HIPACC)
SET(HIPACC_MAJOR_VERSION 0)
SET(HIPACC_MINOR_VERSION 8)
SET(HIPACC_PATCH_VERSION 0)
SET(HIPACC_VERSION ${HIPACC_MAJOR_VERSION}.${HIPACC_MINOR_VERSION}.${HIPACC_PATCH_VERSION})
SET(HIPACC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(HIPACC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

OPTION(USE_POLLY "Use Polly for analysis" OFF)
INCLUDE(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(USE_JIT_ESTIMATE "Compile kernels JIT to estimate resource usage" ON "NOT APPLE" OFF)


# Select the correct script file extension (for repository path and revision) depending on the currently used OS
IF(WIN32)
    SET(SHELL_SCRIPT_EXTENSION "bat")
ELSE(WIN32)
    SET(SHELL_SCRIPT_EXTENSION "sh")
ENDIF(WIN32)

# get repository and revision
EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/get_repository_path.${SHELL_SCRIPT_EXTENSION}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY)
STRING(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)

EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/get_source_version.${SHELL_SCRIPT_EXTENSION}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE HIPACC_GIT_VERSION)
STRING(STRIP ${HIPACC_GIT_VERSION} HIPACC_GIT_VERSION)


# add path for custom modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
FIND_LIBRARY(RT_LIBRARIES rt)
IF(NOT RT_LIBRARIES)
    SET(RT_LIBRARIES "")
ENDIF()
FIND_PACKAGE(Clang REQUIRED)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenCL)
FIND_PACKAGE(RenderScript)
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Threads)
IF(THREADS_HAVE_PTHREAD_ARG)
    SET(THREADS_ARG "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    SET(THREADS_LINK "${CMAKE_THREAD_LIBS_INIT}")
endif()


IF(NOT CUDA_FOUND AND NOT OPENCL_FOUND AND NOT RENDERSCRIPT_FOUND)
    MESSAGE(WARNING "Neither CUDA, OpenCL nor RenderScript found. At least one of them is required to use HIPACC properly.")
ENDIF(NOT CUDA_FOUND AND NOT OPENCL_FOUND AND NOT RENDERSCRIPT_FOUND)

IF(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "5.0")
    MESSAGE(WARNING "At least CUDA version 5.0 required, but found CUDA version ${CUDA_VERSION}.")
    SET(CUDA_FOUND FALSE)
ENDIF(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "5.0")

IF(CUDA_FOUND)
    SET(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    IF(CUDA_VERSION VERSION_LESS "6.5")
        SET(USE_JIT_ESTIMATE OFF)
        SET(NVCC_CPP_STD "")
    ELSE()
        INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
        SET(NVCC_CPP_STD "-std=c++11")
    ENDIF()
    MARK_AS_ADVANCED(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_HOST_COMPILER
        CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION CUDA_TOOLKIT_ROOT_DIR
        CUDA_VERBOSE_BUILD)
ELSE(CUDA_FOUND)
    SET(USE_JIT_ESTIMATE OFF)
ENDIF(CUDA_FOUND)

IF(OPENCL_FOUND)
    SET(CL_COMPILER "${CMAKE_INSTALL_PREFIX}/bin/cl_compile")
ENDIF(OPENCL_FOUND)

MESSAGE(STATUS "Configuration summary:")
MESSAGE(STATUS "===")
MESSAGE(STATUS "CUDA support: ${CUDA_FOUND}")
MESSAGE(STATUS "OpenCL support: ${OPENCL_FOUND}")
MESSAGE(STATUS "Renderscript support: ${RENDERSCRIPT_FOUND}")
MESSAGE(STATUS "OpenCV support: ${OPENCV_FOUND}")
MESSAGE(STATUS "Polly support: ${USE_POLLY}")
MESSAGE(STATUS "JIT estimates: ${USE_JIT_ESTIMATE}")
MESSAGE(STATUS "===")


# platform specific fixes linker flags
IF(MSVC)
    # Correct syntax for LLVM libraries
    STRING(STRIP "${LLVM_MODULE_LIBS}" LLVM_MODULE_LIBS)
    STRING(REPLACE "-lLLVM" "LLVM" LLVM_MODULE_LIBS "${LLVM_MODULE_LIBS}")
    STRING(REPLACE " " ";" LLVM_MODULE_LIBS "${LLVM_MODULE_LIBS}")

    # Erase incompatible LLVM linker flags
    SET(LLVM_LFLAGS "")

    # Additional linker settings
    SET(CUDA_LINK "")
    SET(CUDA_COMP "")
    SET(TIME_LINK "")
ELSEIF(APPLE)
    SET(CUDA_LINK "-Xlinker -framework,CUDA")
    SET(NVCC_COMP "${NVCC_CPP_STD} -ccbin ${CLANG_EXECUTABLE} -Xcompiler -arch -Xcompiler x86_64 -Xcompiler -stdlib=libstdc++")
ELSE()
    SET(CUDA_LINK "-lcuda -lcudart")
    SET(NVCC_COMP "${NVCC_CPP_STD}")

    OPTION(USE_NCURSES "Use NCurses library instead of Curses library" OFF)
    IF(USE_NCURSES)
        STRING(REPLACE " -lcurses" " -lncurses" LLVM_LFLAGS "${LLVM_LFLAGS}")
    ENDIF()
ENDIF()


# debug and release builds
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" build)

# Set compiler-specific flags
IF(MSVC)
    # Disable generation of runtime type information
    STRING(REPLACE "/GR " "/GR- " LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

    # Remove invalid compiler flags
    STRING(REPLACE " -w14062" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

    # Add required include directories
    IF(NOT LLVM_SOURCE_DIR)
        MESSAGE(FATAL_ERROR "LLVM_SOURCE_DIR has to be an absolute path!")
    ENDIF()
    IF(NOT LLVM_BUILD_DIR)
        MESSAGE(FATAL_ERROR "LLVM_BUILD_DIR has to be an absolute path!")
    ENDIF()
    SET(CLANG_INCLUDE_DIR "tools/clang/include")
    SET(LLVM_CXXFLAGS "/I${LLVM_BUILD_DIR}/${CLANG_INCLUDE_DIR} ${LLVM_CXXFLAGS}")
    SET(LLVM_CXXFLAGS "/I${LLVM_SOURCE_DIR}/${CLANG_INCLUDE_DIR} ${LLVM_CXXFLAGS}")


    IF(build STREQUAL "RELEASE")
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    ELSE()
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    ENDIF()
ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti")

    # Add required include directories
    IF(NOT LLVM_SOURCE_DIR)
        MESSAGE(FATAL_ERROR "LLVM_SOURCE_DIR has to be an absolute path!")
    ENDIF()
    IF(NOT LLVM_BUILD_DIR)
        MESSAGE(FATAL_ERROR "LLVM_BUILD_DIR has to be an absolute path!")
    ENDIF()
    SET(CLANG_INCLUDE_DIR "tools/clang/include")
    SET(LLVM_CXXFLAGS "-I${LLVM_BUILD_DIR}/${CLANG_INCLUDE_DIR} ${LLVM_CXXFLAGS}")
    SET(LLVM_CXXFLAGS "-I${LLVM_SOURCE_DIR}/${CLANG_INCLUDE_DIR} ${LLVM_CXXFLAGS}")


    IF(build STREQUAL "RELEASE")
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    ELSE()
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wunused")
    ENDIF()
ENDIF()

# Set other platform specific settings
IF(MSVC)
    IF(${MSVC_VERSION} MATCHES 1700)
        SET(VISUAL_STUDIO_COMMON_TOOLS_PATH "$ENV{VS110COMNTOOLS}")
    ELSEIF(${MSVC_VERSION} MATCHES 1800)
        SET(VISUAL_STUDIO_COMMON_TOOLS_PATH "$ENV{VS120COMNTOOLS}")
    ELSE()
        MESSAGE(FATAL_ERROR "The currently selected version of Visual Studio is not supported!")
    ENDIF()

    STRING(REPLACE "\\" "/" VISUAL_STUDIO_COMMON_TOOLS_PATH "${VISUAL_STUDIO_COMMON_TOOLS_PATH}")
    STRING(REPLACE "Common7/Tools/" "" HOST_COMPILER_INSTALL_PREFIX "${VISUAL_STUDIO_COMMON_TOOLS_PATH}")
ELSE(MSVC)
    SET(HOST_COMPILER_INSTALL_PREFIX "")
ENDIF(MSVC)


# set include directory, add src directories
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include)
ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(compiler)
ADD_SUBDIRECTORY(tools)



# configure header files to pass some of the CMake settings to the source code
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/include/hipacc/Config/config.h.cmake"
    "${HIPACC_BINARY_DIR}/include/hipacc/Config/config.h")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/runtime/hipacc_cu.hpp.cmake"
    "${HIPACC_BINARY_DIR}/runtime/hipacc_cu.hpp")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/Android.mk.cmake"
    "${HIPACC_BINARY_DIR}/tests/Android.mk")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/Application.mk.cmake"
    "${HIPACC_BINARY_DIR}/tests/Application.mk")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/Makefile.cmake"
    "${HIPACC_BINARY_DIR}/tests/Makefile")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/Makefile_test.cmake"
    "${HIPACC_BINARY_DIR}/tests/Makefile_test")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/Makefile_test_opencv.cmake"
    "${HIPACC_BINARY_DIR}/tests/Makefile_test_opencv")
CONFIGURE_FILE(
    "${HIPACC_SOURCE_DIR}/tests/CMakeLists.txt.cmake"
    "${HIPACC_BINARY_DIR}/tests/CMakeLists.txt"
    @ONLY)


# install dsl and runtime header files
FILE(GLOB dsl_headers "${CMAKE_CURRENT_SOURCE_DIR}/dsl/*.hpp")
FILE(GLOB runtime_headers "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.hpp" "${HIPACC_BINARY_DIR}/runtime/*.hpp")
INSTALL(FILES ${runtime_headers} DESTINATION include)
INSTALL(FILES ${dsl_headers} DESTINATION include/dsl)
# install tests
INSTALL(DIRECTORY tests
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "*.cmake" EXCLUDE)
INSTALL(DIRECTORY ${HIPACC_BINARY_DIR}/tests/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "*Makefile_test*" EXCLUDE)

# install Makefiles for test cases
FILE(GLOB TEST_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/*)
FOREACH(DIR IN LISTS TEST_DIRS)
    IF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/${DIR})
        IF(${DIR} MATCHES "opencv_*")
            INSTALL(FILES ${HIPACC_BINARY_DIR}/tests/Makefile_test_opencv
                DESTINATION tests/${DIR}
                RENAME Makefile)
        ELSE(${DIR} MATCHES "opencv_*")
            INSTALL(FILES ${HIPACC_BINARY_DIR}/tests/Makefile_test
                DESTINATION tests/${DIR}
                RENAME Makefile)
        ENDIF(${DIR} MATCHES "opencv_*")
    ENDIF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/${DIR})
ENDFOREACH(DIR IN LISTS TEST_DIRS)

