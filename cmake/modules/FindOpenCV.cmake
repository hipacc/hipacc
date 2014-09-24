# Find the OpenCV includes and library
#
# To set manually the paths, define these environment variables:
# OPENCV_INC_PATH       - Include path (e.g. OPENCV_INC_PATH=/opt/local/include)
# OPENCV_LIB_PATH       - Library path (e.g. OPENCV_LIB_PATH=/usr/local/lib)
#
# Once done this will define
#  OPENCV_INCLUDE_DIR   - where to find OpenCV include files
#  OPENCV_LIBRARY_DIR   - where to find OpenCV libs
#  OPENCV_FOUND         - True if OpenCV is found

FIND_PACKAGE(PackageHandleStandardArgs)

SET(OPENCV_INC_PATH $ENV{OPENCV_INC_PATH} CACHE PATH "OpenCV header files directory.")
SET(OPENCV_LIB_PATH $ENV{OPENCV_LIB_PATH} CACHE PATH "OpenCV library files directory.")

FIND_PATH(OPENCV_INCLUDE_DIR opencv/cv.h    HINTS ${OPENCV_INC_PATH})
FIND_LIBRARY(OPENCV_LIBRARY_DIR opencv_core HINTS ${OPENCV_LIB_PATH})
GET_FILENAME_COMPONENT(OPENCV_LIBRARY_DIR ${OPENCV_LIBRARY_DIR} PATH)

MARK_AS_ADVANCED(
    OPENCV_INCLUDE_DIR
    OPENCV_LIBRARY_DIR
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCV DEFAULT_MSG OPENCV_INCLUDE_DIR OPENCV_LIBRARY_DIR)

IF(OPENCV_FOUND)
    MESSAGE(STATUS "Using OpenCV library at ${OPENCV_LIBRARY_DIR}")
ELSE(OPENCV_FOUND)
    IF(NOT OPENCV_INCLUDE_DIR)
        MESSAGE(STATUS "Could NOT find OpenCV: Set OPENCV_INC_PATH to point to the OpenCV includes.")
    ENDIF()
    IF(NOT OPENCV_LIBRARY_DIR)
        MESSAGE(STATUS "Could NOT find OpenCV: Set OPENCV_LIB_PATH to point to the OpenCV libraries.")
    ENDIF()
ENDIF(OPENCV_FOUND)

