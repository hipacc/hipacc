# Module for locating Renderscript.
#
# Customizable variables:
#  NDK_DIR              - NDK source directory (e.g. NDK_DIR=/opt/android/android-ndk-r10e)
#  ANDROID_API          - Defines Android API level to use
#
# Once done this will define
#  RENDERSCRIPT_FOUND   - True if Renderscript is found

SET(NDK_DIR         $ENV{NDK_DIR}   CACHE PATH      "NDK source directory.")
SET(ANDROID_API     "19"            CACHE STRING    "Android API level.")

FIND_PROGRAM(NDK_BUILD_EXECUTABLE
  NAMES ndk-build
  HINTS "${NDK_DIR}"
  DOC "NDK build executable")

FIND_LIBRARY(RS_STATIC_LIBRARY
  NAME libRScpp_static.a
  HINTS "${NDK_BUILD_EXECUTABLE}/../platforms/android-${ANDROID_API}/arch-arm/usr/lib/rs"
  DOC "Renderscript static library")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Renderscript REQUIRED_VARS NDK_BUILD_EXECUTABLE RS_STATIC_LIBRARY)

MARK_AS_ADVANCED(NDK_BUILD_EXECUTABLE RS_STATIC_LIBRARY)

