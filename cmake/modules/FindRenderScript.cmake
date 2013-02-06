# Module for locating RenderScript.
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
# Read-only variables:
#   RENDERSCRIPT_FOUND
#     Indicates whether RenderScript has been found.
#
#   RS_COMPILER
#     Specifies the RenderScript compiler executable.
#
#   RS_FLAGS
#     Specifies the RenderScript compiler flags.
#
#   RS_INCLUDE_DIRS
#     Specifies the RenderScript include directories.
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

INCLUDE (FindPackageHandleStandardArgs)

FIND_PATH (RS_INCLUDE_DIR
  NAMES frameworks/rs/scriptc/rs_core.rsh
  HINTS ${ANDROID_SOURCE_DIR}
  DOC "RenderScript include directory")

FIND_PROGRAM(RS_EXECUTABLE
  NAME llvm-rs-cc
  HINTS ${ANDROID_SOURCE_DIR}/out/host/${HOST_TYPE}/bin
  DOC "RenderScript compiler executable")

SET (RS_FLAGS -allow-rs-prefix -reflect-c++ -target-api 16 -o .)

SET (RS_INCLUDE_DIRS -I${ANDROID_SOURCE_DIR}/frameworks/rs/scriptc
                     -I${ANDROID_SOURCE_DIR}/external/clang/lib/Headers)

FIND_PROGRAM(NDK_CXX_EXECUTABLE
  NAME arm-linux-androideabi-g++
  HINTS ${NDK_TOOLCHAIN_DIR}/bin
  DOC "NDK compiler executable")

FIND_PATH(NDK_LIBRARY_DIR
  NAME libRScpp.so
  HINTS ${ANDROID_SOURCE_DIR}/out/target/product/${TARGET_NAME}/obj/lib
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

MARK_AS_ADVANCED (RS_INCLUDE_DIR RS_EXECUTABLE NDK_CXX_EXECUTABLE)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (RenderScript REQUIRED_VARS
    ANDROID_SOURCE_DIR TARGET_NAME HOST_TYPE NDK_TOOLCHAIN_DIR
    RS_INCLUDE_DIR RS_EXECUTABLE NDK_CXX_EXECUTABLE NDK_LIBRARY_DIR)

MACRO (RS_WRAP_SCRIPTS DEST)
  FOREACH (SCRIPT ${ARGN})
    STRING (REGEX REPLACE "\^.*/([a-zA-Z0-9_.-]*).rs"
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
  ADD_CUSTOM_TARGET (${NAME} ALL
    COMMAND ${RS_COMPILER} ${RS_FLAGS} ${RS_INCLUDE_DIRS} ${PROJECT_RS}
    COMMAND ${NDK_CXX_COMPILER} ${NDK_CXX_FLAGS} ${NDK_DEFINITIONS}
            ${NDK_INCLUDE_DIRS} ${NDK_LINK_LIBRARIES} ${ARGN} -o ${NAME}
            ${NDK_LINK_LIBRARIES_${NAME}})
ENDMACRO ()

MACRO (RS_LINK_LIBRARIES NAME)
  FOREACH (LIB ${ARGN})
    LIST (APPEND NDK_LINK_LIBRARIES_${NAME} -l${LIB})
  ENDFOREACH ()
ENDMACRO ()

