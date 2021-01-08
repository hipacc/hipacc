# ./lib/Device/CMakeLists.txt

add_library(hipaccBuiltins Builtins.cpp)
add_cppcheck(hipaccBuiltins POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccBuiltins SYSTEM
  PUBLIC  ${CMAKE_SOURCE_DIR}/include/hipacc/Device
  PRIVATE ${CMAKE_BINARY_DIR}/include)
target_compile_features(hipaccBuiltins
  PRIVATE cxx_std_14)
target_link_libraries(hipaccBuiltins
  PRIVATE hipaccCompilerOptions
          clangAST
          clangBasic)
