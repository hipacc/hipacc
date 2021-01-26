# ./lib/Device/CMakeLists.txt

add_library(hipaccBuiltins Builtins.cpp)
add_cppcheck(hipaccBuiltins POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccBuiltins
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_link_libraries(hipaccBuiltins
  PRIVATE hipaccCompilerOptions
          clangAST
          clangBasic)
