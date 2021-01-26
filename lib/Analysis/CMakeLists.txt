# ./lib/Analysis/CMakeLists.txt

add_library(hipaccKernelStatistics KernelStatistics.cpp)
add_cppcheck(hipaccKernelStatistics POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccKernelStatistics
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccKernelStatistics
  PRIVATE hipaccCompilerOptions
          clangAST
          clangAnalysis)

add_library(hipaccHostDataDeps HostDataDeps.cpp)
add_cppcheck(hipaccHostDataDeps POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccHostDataDeps
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccHostDataDeps
  PUBLIC  hipaccASTNode
          hipaccClassRepresentation
          hipaccCompilerOptions
          hipaccBuiltins
          hipaccKernelStatistics
  PRIVATE clangAST
          clangAnalysis
          clangBasic)
