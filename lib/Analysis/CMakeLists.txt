set(KernelStatistics_SOURCES KernelStatistics.cpp)
set(HostDataDeps_SOURCES HostDataDeps.cpp)
set(Polly_SOURCES Polly.cpp)

add_library(hipaccKernelStatistics ${KernelStatistics_SOURCES})
add_library(hipaccHostDataDeps ${HostDataDeps_SOURCES})
if(USE_POLLY)
    add_library(hipaccPolly ${Polly_SOURCES})
endif()
