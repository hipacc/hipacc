# ./lib/Vectorization/CMakeLists.txt

add_library(hipaccSIMDTypes SIMDTypes.cpp)
add_cppcheck(hipaccSIMDTypes POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccSIMDTypes
  PUBLIC  ${CMAKE_SOURCE_DIR}/include/hipacc/Vectorization
  PRIVATE ${CMAKE_BINARY_DIR}/include)
target_compile_features(hipaccSIMDTypes
  PRIVATE cxx_std_14)
target_link_libraries(hipaccSIMDTypes
  PUBLIC  hipaccBuiltins
  PRIVATE hipaccASTNode
          hipaccCompilerOptions
          clangAST)

