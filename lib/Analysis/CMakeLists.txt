SET(CURRENT_HEADER_DIR "${HIPACC_RELATIVE_INCLUDE_DIR}/Analysis")

SET(KernelStatistics_SOURCES KernelStatistics.cpp)
SET(KernelStatistics_HEADERS "${CURRENT_HEADER_DIR}/KernelStatistics.h")

SET(Polly_SOURCES Polly.cpp)
SET(Polly_HEADERS "${CURRENT_HEADER_DIR}/Polly.h")

ADD_LIBRARY(hipaccKernelStatistics ${KernelStatistics_SOURCES} ${KernelStatistics_HEADERS})
IF(USE_POLLY)
    ADD_LIBRARY(hipaccPolly ${Polly_SOURCES} ${Polly_HEADERS})
ENDIF(USE_POLLY)

