SET(CURRENT_HEADER_DIR "${HIPACC_RELATIVE_INCLUDE_DIR}/Device")

SET(Builtins_SOURCES Builtins.cpp)
SET(Builtins_HEADERS "${CURRENT_HEADER_DIR}/Builtins.def"
                     "${CURRENT_HEADER_DIR}/Builtins.h"
                     "${CURRENT_HEADER_DIR}/TargetDescription.h"
                     "${CURRENT_HEADER_DIR}/TargetDevices.h")

ADD_LIBRARY(hipaccBuiltins ${Builtins_SOURCES} ${Builtins_HEADERS})

