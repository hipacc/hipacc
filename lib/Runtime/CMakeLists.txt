set(Runtime_SOURCES Base.cpp CPU.cpp)

if(OpenCL_FOUND)
    list(APPEND Runtime_SOURCES OpenCL.cpp)
endif()

if(CUDA_FOUND)
    list(APPEND Runtime_SOURCES CUDA.cpp)
endif()

include_directories(${CMAKE_SOURCE_DIR}/runtime
                    ${CMAKE_BINARY_DIR}/runtime)

add_library(hipaccRuntime ${Runtime_SOURCES})

# this library is not installed because the static runtime lib has been skipped in favour of the header-only runtime\
# this library is only build to detect compilation issues during the build of Hipacc
#install(TARGETS hipaccRuntime EXPORT ${TARGETS_EXPORT_NAME} ARCHIVE DESTINATION lib COMPONENT runtime)

add_cppcheck(hipaccRuntime POSSIBLE_ERROR FAIL_ON_WARNINGS)
