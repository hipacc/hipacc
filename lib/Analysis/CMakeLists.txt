SET(KernelStatistics_SOURCES KernelStatistics.cpp)
SET(Polly_SOURCES Polly.cpp)

ADD_LIBRARY(hipaccKernelStatistics ${KernelStatistics_SOURCES})
IF(USE_POLLY)
    ADD_LIBRARY(hipaccPolly ${Polly_SOURCES})
ENDIF(USE_POLLY)

