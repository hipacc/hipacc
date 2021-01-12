# ./lib/Runtime/CMakeLists.txt

# this library is not installed because the static runtime lib has been skipped
# in favour of the header-only runtime. this library is only build to detect
# compilation issues during the build of Hipacc

add_library(hipaccRuntime Base.cpp CPU.cpp)
add_cppcheck(hipaccRuntime POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccRuntime
  PRIVATE $<$<BOOL:${CUDA_FOUND}>:${CUDA_INCLUDE_DIRS}>
  PUBLIC  ${CMAKE_SOURCE_DIR}/runtime
          ${CMAKE_BINARY_DIR}/runtime)
target_include_directories(hipaccRuntime SYSTEM
  PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccRuntime
  PRIVATE ${LLVM_DEFINITIONS})

if(OpenCL_FOUND)
  target_sources(hipaccRuntime
    PRIVATE OpenCL.cpp)
  target_include_directories(hipaccRuntime
    PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

if(CUDA_FOUND)
  target_sources(hipaccRuntime
    PRIVATE CUDA.cpp)
endif()

target_compile_options(hipaccRuntime
  PRIVATE	-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)

#install(TARGETS hipaccRuntime EXPORT ${TARGETS_EXPORT_NAME} ARCHIVE DESTINATION lib COMPONENT runtime)

