set(KernelStatistics_SOURCES KernelStatistics.cpp)
set(Polly_SOURCES Polly.cpp)

add_library(hipaccKernelStatistics ${KernelStatistics_SOURCES})
if(Polly_FOUND)
    add_library(hipaccPolly ${Polly_SOURCES})
endif()
