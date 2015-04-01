# Find the OpenCV includes and library
#
# To set manually the paths, define these environment variables:
#  OPENCV_DIR           - OpenCV installation path (e.g. OPENCV_DIR=/opt/local)
#
# Once done this will define
#  OPENCV_INCLUDE_DIRS  - the OpenCV include directories
#  OPENCV_LIBRARY_DIRS  - the OpenCV library directories
#  OPENCV_FOUND         - True if OpenCV is found

SET(OPENCV_DIR $ENV{OPENCV_DIR} CACHE PATH "OpenCV installation path.")

FIND_PATH(OPENCV_INCLUDE_DIR opencv/cv.h    HINTS ${OPENCV_DIR}/include)
FIND_LIBRARY(OPENCV_LIBRARY_DIR opencv_core HINTS ${OPENCV_DIR}/lib ${OPENCV_DIR}/lib64)
GET_FILENAME_COMPONENT(OPENCV_LIBRARY_DIR ${OPENCV_LIBRARY_DIR} PATH)

SET(OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
SET(OPENCV_LIBRARY_DIRS ${OPENCV_LIBRARY_DIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCV DEFAULT_MSG OPENCV_INCLUDE_DIR OPENCV_LIBRARY_DIR)

MARK_AS_ADVANCED(OPENCV_INCLUDE_DIR OPENCV_LIBRARY_DIR)

