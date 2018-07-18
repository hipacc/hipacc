# Find the OpenCV includes and library
#
# To set manually the paths, define these environment variables:
#  OpenCV_DIR           - OpenCV installation path (e.g. OpenCV_DIR=/opt/local)
#
# Once done this will define
#  OpenCV_INCLUDE_DIRS  - the OpenCV include directories
#  OpenCV_LIBRARIES     - the OpenCV libraries
#  OpenCV_DEFINITIONS   - the OpenCV definitions
#  OpenCV_FOUND         - True if OpenCV is found

set(OpenCV_DIR $ENV{OpenCV_DIR} CACHE PATH "OpenCV installation path.")

find_path(OpenCV_INCLUDE_DIR opencv/cv.h                   HINTS ${OpenCV_DIR}/include)
find_library(OpenCV_CORE_LIBRARY opencv_core               HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
find_library(OpenCV_IMGPROC_LIBRARY opencv_imgproc         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
find_library(OpenCV_HIGHGUI_LIBRARY opencv_highgui         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
find_library(OpenCV_VIDEOIO_LIBRARY opencv_videoio         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
find_library(OpenCV_CUDAFILTERS_LIBRARY opencv_cudafilters HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
set(OpenCV_DEFINITIONS "")
if(NOT OpenCV_CORE_LIBRARY)
    set(OpenCV_CORE_LIBRARY "")
else()
    set(OpenCV_DEFINITIONS "-D OPENCV")
endif()
if(OpenCV_CUDAFILTERS_LIBRARY)
    set(OpenCV_DEFINITIONS "${OpenCV_DEFINITIONS} -D OPENCV_CUDA_FOUND")
else()
    set(OpenCV_CUDAFILTERS_LIBRARY "")
endif()
if(NOT OpenCV_VIDEOIO_LIBRARY)
    set(OpenCV_VIDEOIO_LIBRARY "")
endif()
if(NOT OpenCV_HIGHGUI_LIBRARY)
    set(OpenCV_HIGHGUI_LIBRARY "")
endif()
if(NOT OpenCV_IMGPROC_LIBRARY)
    set(OpenCV_IMGPROC_LIBRARY "")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCV DEFAULT_MSG OpenCV_INCLUDE_DIR OpenCV_CORE_LIBRARY)

set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
set(OpenCV_LIBRARIES "${OpenCV_CORE_LIBRARY} ${OpenCV_IMGPROC_LIBRARY} ${OpenCV_HIGHGUI_LIBRARY} ${OpenCV_VIDEOIO_LIBRARY} ${OpenCV_CUDAFILTERS_LIBRARY}")

mark_as_advanced(OpenCV_INCLUDE_DIR OpenCV_CORE_LIBRARY OpenCV_IMGPROC_LIBRARY OpenCV_HIGHGUI_LIBRARY OpenCV_VIDEOIO_LIBRARY OpenCV_CUDAFILTERS_LIBRARY)
