# Find the MinGW includes
#
# To set manually the paths, define these environment variables:
# MINGW_ROOT_PATH       - The root path where to search for an MinGW installation (e.g. MINGW_ROOT_PATH=/opt/MinGW-4.0)
#
# Once done this will define
#  MINGW_INCLUDE_ROOT       - where to find OpenCL include files
#  MINGW_INCLUDE_ROOT_CPP   - where to find OpenCL libs
#  MINGW_FOUND              - True if OpenCL found.

FIND_PACKAGE(PackageHandleStandardArgs)

SET(MINGW_ROOT_PATH $ENV{MINGW_ROOT_PATH} CACHE PATH "MinGW root folder.")

IF (IS_ABSOLUTE ${MINGW_ROOT_PATH})
    FIND_PATH(MINGW_INCLUDE_ROOT include/_mingw.h PATHS ${MINGW_ROOT_PATH} ${MINGW_ROOT_PATH}/* NO_DEFAULT_PATH)
    FIND_PATH(MINGW_INCLUDE_ROOT_CPP include/stddef.h HINTS ${MINGW_INCLUDE_ROOT}/lib/gcc/mingw32/*)
ELSE(IS_ABSOLUTE ${MINGW_ROOT_PATH})
    MESSAGE(FATAL_ERROR "MINGW_ROOT_PATH has to be an absolute path!")
ENDIF(IS_ABSOLUTE ${MINGW_ROOT_PATH})

IF(MINGW_INCLUDE_ROOT)
    SET(MINGW_INCLUDE_ROOT ${MINGW_INCLUDE_ROOT}/include)
    MESSAGE(STATUS "MinGW include path found at: ${MINGW_INCLUDE_ROOT}")
ENDIF(MINGW_INCLUDE_ROOT)

IF(MINGW_INCLUDE_ROOT_CPP)
    SET(MINGW_INCLUDE_ROOT_CPP ${MINGW_INCLUDE_ROOT_CPP}/include)
    MESSAGE(STATUS "MinGW-GCC include path found at: ${MINGW_INCLUDE_ROOT_CPP}")
ENDIF(MINGW_INCLUDE_ROOT_CPP)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(MINGW DEFAULT_MSG MINGW_INCLUDE_ROOT MINGW_INCLUDE_ROOT_CPP)

MARK_AS_ADVANCED(
    MINGW_INCLUDE_ROOT
    MINGW_INCLUDE_ROOT_CPP
)
