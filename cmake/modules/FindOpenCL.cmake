# Find the OpenCL includes and library
#
# To set manually the paths, define these environment variables:
# OPENCL_INC_PATH       - Include path (e.g. OPENCL_INC_PATH=/opt/cuda/include)
# OPENCL_LIB_PATH       - Library path (e.g. OPENCL_LIB_PATH=/usr/lib64/nvidia)
#
# Once done this will define
#  OPENCL_INCLUDE_DIR   - where to find OpenCL include files
#  OPENCL_LIBRARY_DIR   - where to find OpenCL libs
#  OPENCL_CFLAGS        - OpenCL C compiler flags
#  OPENCL_LFLAGS        - OpenCL linker flags
#  OPENCL_FOUND         - True if OpenCL is found

FIND_PACKAGE(PackageHandleStandardArgs)

SET(OPENCL_INC_PATH $ENV{OPENCL_INC_PATH} CACHE PATH "OpenCL header files directory.")
SET(OPENCL_LIB_PATH $ENV{OPENCL_LIB_PATH} CACHE PATH "OpenCL library files directory.")

IF(APPLE)
    FIND_LIBRARY(OPENCL_LIBRARY_DIR OpenCL)
    FIND_PATH(OPENCL_INCLUDE_DIR OpenCL/cl.h)
    SET(OPENCL_CFLAGS "")
    SET(OPENCL_LFLAGS "-framework OpenCL")
ELSE(APPLE)
    # Unix style platforms
    FIND_LIBRARY(OPENCL_LIBRARY_DIR OpenCL
        HINTS ${OPENCL_LIB_PATH}
    )
    GET_FILENAME_COMPONENT(OPENCL_LIBRARY_DIR ${OPENCL_LIBRARY_DIR} PATH)

    FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h
        HINTS ${OPENCL_INC_PATH})
    SET(OPENCL_CFLAGS "-I${OPENCL_INCLUDE_DIR}")
    SET(OPENCL_LFLAGS "-L${OPENCL_LIBRARY_DIR} -lOpenCL")
ENDIF(APPLE)

MARK_AS_ADVANCED(
    OPENCL_INCLUDE_DIR
    OPENCL_LIBRARY_DIR
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OPENCL_INCLUDE_DIR OPENCL_LIBRARY_DIR)

IF(OPENCL_FOUND)
    MESSAGE(STATUS "Using OpenCL library at ${OPENCL_LIBRARY_DIR}")
ELSE(OPENCL_FOUND)
    IF(NOT OPENCL_INCLUDE_DIR)
        MESSAGE(STATUS "Could NOT find OpenCL: Set OPENCL_INC_PATH to point to the OpenCL includes.")
    ENDIF()
    IF(NOT OPENCL_LIBRARY_DIR)
        MESSAGE(STATUS "Could NOT find OpenCL: Set OPENCL_LIB_PATH to point to the OpenCL libraries.")
    ENDIF()
ENDIF(OPENCL_FOUND)

