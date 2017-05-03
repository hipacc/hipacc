# Find the OpenCL includes and library
#
# To set manually the paths, define these environment variables:
#  OpenCL_INC_DIR       - Include path (e.g. OpenCL_INC_DIR=/usr/local/cuda/include)
#  OpenCL_LIB_DIR       - Library path (e.g. OpenCL_LIB_DIR=/usr/lib64/nvidia)
#
# Once done this will define
#  OpenCL_INCLUDE_DIRS  - where to find OpenCL include files
#  OpenCL_LIBRARIES     - where to find OpenCL libs
#  OpenCL_FOUND         - True if OpenCL is found

set(OpenCL_INC_DIR $ENV{OpenCL_INC_DIR} CACHE PATH "OpenCL header files directory.")
set(OpenCL_LIB_DIR $ENV{OpenCL_LIB_DIR} CACHE PATH "OpenCL library files directory.")

if(APPLE)
    find_path(OpenCL_INCLUDE_DIR OpenCL/cl.h)
    find_library(OpenCL_LIBRARY OpenCL)
    # hack: CMake converts framework path to the following, but we don't use CMake for this 
    set(OpenCL_LIBRARY "-framework OpenCL")
else()
    find_path(OpenCL_INCLUDE_DIR CL/cl.h HINTS ${OpenCL_INC_DIR})
    find_library(OpenCL_LIBRARY OpenCL HINTS ${OpenCL_LIB_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL DEFAULT_MSG OpenCL_INCLUDE_DIR OpenCL_LIBRARY)

set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})

mark_as_advanced(OpenCL_INCLUDE_DIR OpenCL_LIBRARY)
