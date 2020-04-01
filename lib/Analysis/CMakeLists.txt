set(KernelStatistics_SOURCES KernelStatistics.cpp)
set(Polly_SOURCES Polly.cpp)

add_library(hipaccKernelStatistics ${KernelStatistics_SOURCES})
add_cppcheck(hipaccKernelStatistics POSSIBLE_ERROR FAIL_ON_WARNINGS)

if(Polly_FOUND)
    add_library(hipaccPolly ${Polly_SOURCES})
    add_cppcheck(hipaccPolly POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif()
