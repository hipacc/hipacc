# ./lib/Analysis/CMakeLists.txt

add_library(hipaccKernelStatistics KernelStatistics.cpp)
add_cppcheck(hipaccKernelStatistics POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccKernelStatistics
  PUBLIC  ${CMAKE_SOURCE_DIR}/include/hipacc/Analysis
  PRIVATE ${CMAKE_SOURCE_DIR}/include/hipacc/Device
          ${CMAKE_SOURCE_DIR}/include/hipacc/DSL
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccKernelStatistics
  PRIVATE hipaccCompilerOptions
  PRIVATE clangAST
          clangAnalysis)

add_library(hipaccHostDataDeps HostDataDeps.cpp)
add_cppcheck(hipaccHostDataDeps POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccHostDataDeps
  PUBLIC  ${CMAKE_SOURCE_DIR}/include/hipacc/Analysis
  PRIVATE ${CMAKE_SOURCE_DIR}/include/hipacc/Device
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
