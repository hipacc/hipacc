SET(KernelStatistics_SOURCES KernelStatistics.cpp)
SET(Polly_SOURCES Polly.cpp)

ADD_LIBRARY(KernelStatistics ${KernelStatistics_SOURCES})
IF(USE_POLLY)
    ADD_LIBRARY(Polly ${Polly_SOURCES})
ENDIF(USE_POLLY)

