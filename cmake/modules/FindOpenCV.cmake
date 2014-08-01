# Find the OpenCV includes and library
#
# To set manually the paths, define these environment variables:
# OPENCV_INC_PATH       - Include path (e.g. OPENCV_INC_PATH=/opt/local/include)
# OPENCV_LIB_PATH       - Library path (e.g. OPENCV_LIB_PATH=/usr/local/lib)
#
# Once done this will define
#  OPENCV_INCLUDE_DIR   - where to find OpenCV include files
#  OPENCV_LIBRARY_DIR   - where to find OpenCV libs
#  OPENCV_FOUND         - True if OpenCV found.

FIND_PACKAGE(PackageHandleStandardArgs)

SET(OPENCV_INC_PATH $ENV{OPENCV_INC_PATH} CACHE PATH "OpenCV header files directory.")
SET(OPENCV_LIB_PATH $ENV{OPENCV_LIB_PATH} CACHE PATH "OpenCV library files directory.")

FIND_PATH(OPENCV_INCLUDE_DIR opencv/cv.h    HINTS ${OPENCV_INC_PATH})
FIND_LIBRARY(OPENCV_LIBRARY_DIR opencv_core HINTS ${OPENCV_LIB_PATH})
GET_FILENAME_COMPONENT(OPENCV_LIBRARY_DIR ${OPENCV_LIBRARY_DIR} PATH)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCV DEFAULT_MSG OPENCV_INCLUDE_DIR OPENCV_LIBRARY_DIR)

IF(OPENCV_INCLUDE_DIR AND OPENCV_LIBRARY_DIR)
    MESSAGE(STATUS "OpenCV includes found at: ${OPENCV_INCLUDE_DIR}")
    MESSAGE(STATUS "OpenCV library found at: ${OPENCV_LIBRARY_DIR}")
    SET(OPENCV_FOUND TRUE)
ELSE(OPENCV_INCLUDE_DIR AND OPENCV_LIBRARY_DIR)
    MESSAGE(STATUS "Could NOT find OpenCV. Set OPENCV_INC_PATH and OPENCV_LIB_PATH to point to the OpenCV includes and libraries.")
    SET(OPENCV_FOUND FALSE)
ENDIF(OPENCV_INCLUDE_DIR AND OPENCV_LIBRARY_DIR)

MARK_AS_ADVANCED(
    OPENCV_FOUND
    OPENCV_INCLUDE_DIR
    OPENCV_LIBRARY_DIR
)

