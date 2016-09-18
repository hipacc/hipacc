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

SET(OpenCV_DIR $ENV{OpenCV_DIR} CACHE PATH "OpenCV installation path.")

FIND_PATH(OpenCV_INCLUDE_DIR opencv/cv.h                   HINTS ${OpenCV_DIR}/include)
FIND_LIBRARY(OpenCV_CORE_LIBRARY opencv_core               HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
FIND_LIBRARY(OpenCV_IMGPROC_LIBRARY opencv_imgproc         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
FIND_LIBRARY(OpenCV_HIGHGUI_LIBRARY opencv_highgui         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
FIND_LIBRARY(OpenCV_VIDEOIO_LIBRARY opencv_videoio         HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
FIND_LIBRARY(OpenCV_CUDAFILTERS_LIBRARY opencv_cudafilters HINTS ${OpenCV_DIR}/lib ${OpenCV_DIR}/lib64)
SET(OpenCV_DEFINITIONS "-D OPENCV")
IF(OpenCV_CUDAFILTERS_LIBRARY)
    SET(OpenCV_DEFINITIONS "${OpenCV_DEFINITIONS} -D OPENCV_CUDA_FOUND")
ELSE()
    SET(OpenCV_CUDAFILTERS_LIBRARY "")
ENDIF()
IF(NOT OpenCV_VIDEOIO_LIBRARY)
    SET(OpenCV_VIDEOIO_LIBRARY "")
ENDIF()
IF(NOT OpenCV_HIGHGUI_LIBRARY)
    SET(OpenCV_HIGHGUI_LIBRARY "")
ENDIF()
IF(NOT OpenCV_IMGPROC_LIBRARY)
    SET(OpenCV_IMGPROC_LIBRARY "")
ENDIF()
IF(NOT OpenCV_CORE_LIBRARY)
    SET(OpenCV_CORE_LIBRARY "")
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCV DEFAULT_MSG OpenCV_INCLUDE_DIR OpenCV_CORE_LIBRARY)

SET(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
SET(OpenCV_LIBRARIES "${OpenCV_CORE_LIBRARY} ${OpenCV_IMGPROC_LIBRARY} ${OpenCV_HIGHGUI_LIBRARY} ${OpenCV_VIDEOIO_LIBRARY} ${OpenCV_CUDAFILTERS_LIBRARY}")

MARK_AS_ADVANCED(OpenCV_INCLUDE_DIR OpenCV_CORE_LIBRARY OpenCV_IMGPROC_LIBRARY OpenCV_HIGHGUI_LIBRARY OpenCV_VIDEOIO_LIBRARY OpenCV_CUDAFILTERS_LIBRARY)
