# ./lib/Config/CMakeLists.txt

add_library(hipaccCompilerOptions CompilerOptions.cpp)
add_cppcheck(hipaccCompilerOptions POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccCompilerOptions
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_include_directories(hipaccCompilerOptions SYSTEM
  PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccCompilerOptions
  PUBLIC  ${LLVM_DEFINITIONS})
