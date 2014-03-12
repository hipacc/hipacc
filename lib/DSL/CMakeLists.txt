SET(CURRENT_HEADER_DIR "${HIPACC_RELATIVE_INCLUDE_DIR}/DSL")

SET(ClassRepresentation_SOURCES ClassRepresentation.cpp)
SET(ClassRepresentation_HEADERS "${CURRENT_HEADER_DIR}/ClassRepresentation.h"
                                "${CURRENT_HEADER_DIR}/CompilerKnownClasses.h")

ADD_LIBRARY(hipaccClassRepresentation ${ClassRepresentation_SOURCES} ${ClassRepresentation_HEADERS})

