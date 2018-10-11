# Find the NVML includes and library
#
# Once done this will define
#  NVML_INCLUDE_DIRS    - where to find NVML include files
#  NVML_LIBRARIES       - where to find NVML libs
#  NVML_FOUND           - True if NVML is found

find_path(NVML_INCLUDE_DIR nvml.h PATHS ${CUDA_INCLUDE_DIRS} /usr/include/nvidia/gdk /usr/include)
find_library(NVML_LIBRARY nvidia-ml)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG NVML_INCLUDE_DIR NVML_LIBRARY)

set(NVML_INCLUDE_DIRS ${NVML_INCLUDE_DIR})
set(NVML_LIBRARIES ${NVML_LIBRARY})
if(NOT NVML_LIBRARIES)
    set(NVML_LIBRARIES "")
endif()

mark_as_advanced(NVML_INCLUDE_DIR NVML_LIBRARY)
