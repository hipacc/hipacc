# Module for locating Renderscript.
#
# Customizable variables:
#   ANDROID_SOURCE_DIR
#     Specifies Android source directory.
#
#   HOST_TYPE
#     Host type of the build system (eg. linux-x86).
#     Must be a directory in ${ANDROID_SOURCE_DIR}/out/host.
#
#   TARGET_NAME
#     Name of the target product Android was build for.
#     Must be a directory in ${ANDROID_SOURCE_DIR}/out/target/product.
#
#   NDK_TOOLCHAIN_DIR
#     Path to android NDK standalone toolchain.
#
#   RS_TARGET_API
#     Defines Android API level to use.
#
#   EMBEDDED_OPENCL_INC_PATH
#     Path to OpenCL include files for Android
#
#   EMBEDDED_OPENCL_LIB_PATH
#     Path to OpenCL library directory for Android - system/vendor/lib/egl for Mali.
#
#   EMBEDDED_OPENCL_LIB
#     Name to link against for Android - GLES_mali for Mali.
#
# Read-only variables:
#   RENDERSCRIPT_FOUND
#     Indicates whether Renderscript has been found.
#
#   RS_COMPILER
#     Specifies the Renderscript compiler executable.
#
#   RS_FLAGS
#     Specifies the Renderscript compiler flags.
#
#   RS_INCLUDE_DIRS
#     Specifies the Renderscript include directories.
#
#   RS_SOURCES
#     Renderscript source files.
#
#   NDK_CXX_COMPILER
#     Specifies the NDK C++ compiler executable.
#
#   NDK_CXX_FLAGS
#     Specifies the NDK C++ compiler flags.
#
#   NDK_INCLUDE_DIRS
#     Specifies the NDK include directories.
#
#   NDK_LINK_LIBRARIES
#     Specifies the NDK link libraries.
#
#   NDK_INCLUDE_DIRS_STR
#     Specifies the NDK include directories as string.
#
#   NDK_LINK_LIBRARIES_STR
#     Specifies the NDK link libraries as string.
#
#   EMBEDDED_OPENCL_CFLAGS
#     OpenCL C compiler flags
#
#   EMBEDDED_OPENCL_LFLAGS
#     OpenCL linker flags
#

INCLUDE (FindPackageHandleStandardArgs)

SET(ANDROID_SOURCE_DIR          $ENV{ANDROID_SOURCE_DIR}        CACHE PATH      "Android source directory.")
SET(HOST_TYPE                   $ENV{HOST_TYPE}                 CACHE STRING    "Name of the local compile host type.")
SET(TARGET_NAME                 $ENV{TARGET_NAME}               CACHE STRING    "Name of target platform.")
SET(NDK_TOOLCHAIN_DIR           $ENV{NDK_TOOLCHAIN_DIR}         CACHE PATH      "Android NDK directory.")
SET(RS_TARGET_API               $ENV{RS_TARGET_API}             CACHE STRING    "Android API level.")

SET(EMBEDDED_OPENCL_INC_PATH    $ENV{EMBEDDED_OPENCL_INC_PATH}  CACHE PATH      "Embedded OpenCL header files directory.")
SET(EMBEDDED_OPENCL_LIB_PATH    $ENV{EMBEDDED_OPENCL_LIB_PATH}  CACHE STRING    "Embedded OpenCL library directory within the target system.")
SET(EMBEDDED_OPENCL_LIB         $ENV{EMBEDDED_OPENCL_LIB}       CACHE STRING    "Embedded OpenCL library name.")

SET(RS_INCLUDE_DIR RS_INCLUDE_DIR-NOTFOUND)
SET(RS_EXECUTABLE RS_EXECUTABLE-NOTFOUND)
SET(NDK_CXX_EXECUTABLE NDK_CXX_EXECUTABLE-NOTFOUND)
SET(NDK_LIBRARY_DIR NDK_LIBRARY_DIR-NOTFOUND)

IF (RS_TARGET_API STREQUAL "")
  SET(RS_TARGET_API "0")
ENDIF ()

IF (${RS_TARGET_API} GREATER 18)

# For Android 4.4 we just need a target API version and the corresponding NDK
# version installed, which includes the static Renderscript C++ library.
# (Embedded OpenCL can be omitted, because it is not available anyway.)
FIND_PROGRAM(NDK_BUILD_EXECUTABLE
  NAMES ndk-build
  DOC "NDK build executable")

FIND_LIBRARY(RS_STATIC_LIBRARY
  NAME libRScpp_static.a
  HINTS "${NDK_BUILD_EXECUTABLE}/../platforms/android-${RS_TARGET_API}/arch-arm/usr/lib/rs/"
  DOC "Renderscript static library")

FIND_PACKAGE_HANDLE_STANDARD_ARGS (Renderscript REQUIRED_VARS
  NDK_BUILD_EXECUTABLE RS_STATIC_LIBRARY)

ELSE (${RS_TARGET_API} GREATER 18)

FIND_PATH(RS_INCLUDE_DIR
  NAMES frameworks/rs/scriptc/rs_core.rsh
  HINTS ${ANDROID_SOURCE_DIR}
  DOC "Renderscript include directory")

FIND_PROGRAM(RS_EXECUTABLE
  NAME llvm-rs-cc
  HINTS ${ANDROID_SOURCE_DIR}/out/host/${HOST_TYPE}/bin
  DOC "Renderscript compiler executable")

SET (RS_FLAGS -allow-rs-prefix -reflect-c++ -target-api ${RS_TARGET_API} -o ./)

SET (RS_INCLUDE_DIRS -I${ANDROID_SOURCE_DIR}/frameworks/rs/scriptc
                     -I${ANDROID_SOURCE_DIR}/external/clang/lib/Headers)

SET (RS_SOURCES "")

FIND_PROGRAM(NDK_CXX_EXECUTABLE
  NAME arm-linux-androideabi-g++
  HINTS ${NDK_TOOLCHAIN_DIR}/bin
  DOC "NDK compiler executable")

FIND_PATH(NDK_LIBRARY_DIR
  NAME libRScpp.so
  HINTS ${ANDROID_SOURCE_DIR}/out/target/product/${TARGET_NAME}/system/lib
  DOC "NDK target library directory")

SET (NDK_CXX_FLAGS "-fno-rtti")

SET (NDK_INCLUDE_DIRS -I${ANDROID_SOURCE_DIR}/frameworks/rs/cpp
                      -I${ANDROID_SOURCE_DIR}/frameworks/rs
                      -I${ANDROID_SOURCE_DIR}/frameworks/native/include
                      -I${ANDROID_SOURCE_DIR}/system/core/include
                      -I${ANDROID_SOURCE_DIR}/out/target/product/${TARGET_NAME}/obj/SHARED_LIBRARIES/libRS_intermediates
                      -I${CMAKE_CURRENT_BINARY_DIR})

SET (NDK_LINK_LIBRARIES -l${NDK_LIBRARY_DIR}/libcutils.so
                        -l${NDK_LIBRARY_DIR}/libRScpp.so)

SET (RS_COMPILER ${RS_EXECUTABLE})
SET (NDK_CXX_COMPILER ${NDK_CXX_EXECUTABLE})

MARK_AS_ADVANCED (RS_INCLUDE_DIR RS_EXECUTABLE NDK_CXX_EXECUTABLE NDK_LIBRARY_DIR)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (Renderscript REQUIRED_VARS
    ANDROID_SOURCE_DIR TARGET_NAME HOST_TYPE NDK_TOOLCHAIN_DIR RS_TARGET_API
    RS_INCLUDE_DIR RS_EXECUTABLE NDK_CXX_EXECUTABLE NDK_LIBRARY_DIR)


# Begin embedded OpenCL
FIND_LIBRARY(EMBEDDED_OPENCL_LIBRARY_DIR ${EMBEDDED_OPENCL_LIB}
    HINTS ${ANDROID_SOURCE_DIR}/out/target/product/${TARGET_NAME}/system/${EMBEDDED_OPENCL_LIB_PATH})
GET_FILENAME_COMPONENT(EMBEDDED_OPENCL_LIBRARY_DIR ${EMBEDDED_OPENCL_LIBRARY_DIR} PATH)

FIND_PATH(EMBEDDED_OPENCL_INCLUDE_DIR CL/cl.h
    HINTS ${EMBEDDED_OPENCL_INC_PATH})

SET(EMBEDDED_OPENCL_CFLAGS "-I${EMBEDDED_OPENCL_INCLUDE_DIR}")
SET(EMBEDDED_OPENCL_LFLAGS "-l${EMBEDDED_OPENCL_LIBRARY_DIR}/${EMBEDDED_OPENCL_LIB}")

SET(NDK_INCLUDE_DIRS_STR "")
FOREACH(S ${NDK_INCLUDE_DIRS})
    SET(NDK_INCLUDE_DIRS_STR "${NDK_INCLUDE_DIRS_STR} ${S}")
ENDFOREACH(S)
SET(NDK_LINK_LIBRARIES_STR "")
FOREACH(S ${NDK_LINK_LIBRARIES})
    SET(NDK_LINK_LIBRARIES_STR "${NDK_LINK_LIBRARIES_STR} ${S}")
ENDFOREACH(S)

IF(EMBEDDED_OPENCL_INCLUDE_DIR AND EMBEDDED_OPENCL_LIBRARY_DIR)
    MESSAGE(STATUS "Embedded OpenCL includes found at: ${EMBEDDED_OPENCL_INCLUDE_DIR}")
    MESSAGE(STATUS "Embedded OpenCL library found at: ${EMBEDDED_OPENCL_LIBRARY_DIR}")
    SET(EMBEDDED_OPENCL_FOUND true)
ELSE(EMBEDDED_OPENCL_INCLUDE_DIR AND EMBEDDED_OPENCL_LIBRARY_DIR)
    MESSAGE(STATUS "Could NOT find embedded OpenCL. Set EMBEDDED_OPENCL_INC_PATH, EMBEDDED_OPENCL_LIB_PATH and EMBEDDED_OPENCL_LIB to point to the OpenCL includes, the library path, and the name of the OpenCL library.")
    SET(EMBEDDED_OPENCL_FOUND false)
ENDIF(EMBEDDED_OPENCL_INCLUDE_DIR AND EMBEDDED_OPENCL_LIBRARY_DIR)

MARK_AS_ADVANCED(NDK_INCLUDE_DIRS_STR NDK_LINK_LIBRARIES_STR
    EMBEDDED_OPENCL_LIBRARY_DIR EMBEDDED_OPENCL_INCLUDE_DIR)
# End embedded OpenCL


MACRO (RS_WRAP_SCRIPTS DEST)
  FOREACH (SCRIPT ${ARGN})
    LIST (APPEND RS_SOURCES ${SCRIPT})
    STRING (REGEX REPLACE "\^.*/([a-zA-Z0-9_.-]*).(rs|fs)"
                         "${CMAKE_CURRENT_BINARY_DIR}/ScriptC_\\1.cpp"
                         SCRIPT ${SCRIPT})
    LIST (APPEND ${DEST} ${SCRIPT})
  ENDFOREACH ()
ENDMACRO ()

MACRO (RS_DEFINITIONS)
  LIST (APPEND NDK_DEFINITIONS ${ARGN})
ENDMACRO ()

MACRO (RS_INCLUDE_DIRECTORIES)
  FOREACH (INC ${ARGN})
    LIST (APPEND NDK_INCLUDE_DIRS -I${INC})
    LIST (APPEND RS_INCLUDE_DIRS -I${INC})
  ENDFOREACH ()
ENDMACRO ()

MACRO (RS_ADD_EXECUTABLE NAME)
  IF (RS_SOURCES STREQUAL "")
    ADD_CUSTOM_TARGET (${NAME} ALL
      COMMAND ${NDK_CXX_COMPILER} ${NDK_CXX_FLAGS} ${NDK_DEFINITIONS}
              -DRS_TARGET_API=${RS_TARGET_API} ${NDK_INCLUDE_DIRS}
              ${NDK_LINK_LIBRARIES} ${ARGN} -o ${NAME}
              ${NDK_LINK_LIBRARIES_${NAME}})
  ELSE ()
    ADD_CUSTOM_TARGET (${NAME} ALL
      COMMAND ${RS_COMPILER} ${RS_FLAGS} ${RS_INCLUDE_DIRS} ${RS_SOURCES}
      COMMAND ${NDK_CXX_COMPILER} ${NDK_CXX_FLAGS} ${NDK_DEFINITIONS}
              -DRS_TARGET_API=${RS_TARGET_API} ${NDK_INCLUDE_DIRS}
              ${NDK_LINK_LIBRARIES} ${ARGN} -o ${NAME}
              ${NDK_LINK_LIBRARIES_${NAME}})
  ENDIF ()
ENDMACRO ()

MACRO (RS_LINK_LIBRARIES NAME)
  FOREACH (LIB ${ARGN})
    LIST (APPEND NDK_LINK_LIBRARIES_${NAME} -l${LIB})
  ENDFOREACH ()
ENDMACRO ()

ENDIF (${RS_TARGET_API} GREATER 18)

