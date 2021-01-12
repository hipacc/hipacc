# ./lib/AST/CMakeLists.txt

add_library(hipaccASTNode ASTNode.cpp)
add_cppcheck(hipaccASTNode POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccASTNode
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccASTNode
  PRIVATE hipaccCompilerOptions
          clangAST)

add_library(hipaccASTFuse ASTFuse.cpp)
add_cppcheck(hipaccASTFuse POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccASTFuse
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccASTFuse
  PUBLIC  hipaccASTNode
          hipaccASTTranslate
          hipaccCompilerOptions
          hipaccHostDataDeps
          hipaccKernelStatistics
          hipaccBuiltins
          hipaccClassRepresentation
          hipaccSIMDTypes
  PRIVATE clangAST
          clangFrontend
          clangSema)

add_library(hipaccASTTranslate ASTTranslate.cpp)
target_sources(hipaccASTTranslate
  PRIVATE ASTClone.cpp
          ASTTranslate.cpp
          BorderHandling.cpp
          Convolution.cpp
          Interpolate.cpp
          MemoryAccess.cpp)
add_cppcheck(hipaccASTTranslate POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccASTTranslate
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccASTTranslate
  PUBLIC  hipaccASTNode
          hipaccCompilerOptions
          hipaccKernelStatistics
          hipaccBuiltins
          hipaccClassRepresentation
          hipaccSIMDTypes
  PRIVATE clangAST
          clangFrontend
          clangSema)

