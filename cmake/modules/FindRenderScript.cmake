# Module for locating Renderscript.
#
# Customizable variables:
#  NDK_DIR                              - NDK source directory (e.g. NDK_DIR=/opt/android/android-ndk-r14)
#
# Once done this will define
#  Renderscript_ndk_build_EXECUTABLE    - ndk-build executable
#  Renderscript_FOUND                   - True if Renderscript is found

set(NDK_DIR         $ENV{NDK_DIR}   CACHE PATH      "NDK source directory.")

find_program(Renderscript_ndk_build_EXECUTABLE ndk-build HINTS "${NDK_DIR}" DOC "NDK build executable")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Renderscript REQUIRED_VARS Renderscript_ndk_build_EXECUTABLE)
