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
install(TARGETS hipaccRuntime ARCHIVE DESTINATION lib COMPONENT runtime)
