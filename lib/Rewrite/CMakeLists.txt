# ./lib/Rewrite/CMakeLists.txt

add_library(hipaccRewrite Rewrite.cpp)
add_cppcheck(hipaccRewrite POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccRewrite
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_include_directories(hipaccRewrite SYSTEM
  PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccRewrite
  PRIVATE ${LLVM_DEFINITIONS}
          "$<$<AND:$<NOT:$<BOOL:${APPLE}>>,$<BOOL:${CUDA_FOUND}>>:USE_JIT_ESTIMATE>")
target_link_libraries(hipaccRewrite
  PRIVATE hipaccASTTranslate
          hipaccASTNode
          hipaccASTFuse
          hipaccHostDataDeps
          hipaccBackend
          hipaccCompilerOptions
          hipaccCreateHostStrings
  PRIVATE clangFrontend
          clangAST
          clangRewrite)

add_library(hipaccCreateHostStrings CreateHostStrings.cpp)
add_cppcheck(hipaccCreateHostStrings POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccCreateHostStrings
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_BINARY_DIR}/include)
target_compile_features(hipaccCreateHostStrings
  PRIVATE cxx_std_14)
target_link_libraries(hipaccCreateHostStrings
  PUBLIC  hipaccClassRepresentation
          hipaccASTFuse)

