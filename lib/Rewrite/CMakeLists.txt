SET(CURRENT_HEADER_DIR "${HIPACC_RELATIVE_INCLUDE_DIR}/Rewrite")

SET(Rewrite_SOURCES Rewrite.cpp)
SET(Rewrite_HEADERS "${CURRENT_HEADER_DIR}/Rewrite.h")

SET(CreateHostStrings_SOURCES CreateHostStrings.cpp)
SET(CreateHostStrings_HEADERS "${CURRENT_HEADER_DIR}/CreateHostStrings.h")

ADD_LIBRARY(hipaccRewrite ${Rewrite_SOURCES} ${Rewrite_HEADERS})
ADD_LIBRARY(hipaccCreateHostStrings ${CreateHostStrings_SOURCES} ${CreateHostStrings_HEADERS})

