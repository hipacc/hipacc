# ./lib/DSL/CMakeLists.txt

add_library(hipaccClassRepresentation ClassRepresentation.cpp)
add_cppcheck(hipaccClassRepresentation POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccClassRepresentation
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include
          $<$<BOOL:${CUDA_FOUND}>:${CUDA_INCLUDE_DIRS}>)
target_include_directories(hipaccClassRepresentation SYSTEM
  PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccClassRepresentation
  PRIVATE ${LLVM_DEFINITIONS}
          "$<$<AND:$<NOT:$<BOOL:${APPLE}>>,$<BOOL:${CUDA_FOUND}>>:USE_JIT_ESTIMATE>")
target_link_libraries(hipaccClassRepresentation
  PUBLIC  hipaccKernelStatistics
          hipaccCompilerOptions
  PRIVATE clangAST)

