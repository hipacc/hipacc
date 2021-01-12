# ./lib/Config/CMakeLists.txt

add_library(hipaccCompilerOptions CompilerOptions.cpp)
add_cppcheck(hipaccCompilerOptions POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipaccCompilerOptions
  PUBLIC  ${CMAKE_SOURCE_DIR}/include/hipacc/Config
          ${CMAKE_SOURCE_DIR}/include/hipacc/Device
          ${CMAKE_BINARY_DIR}/include)
target_include_directories(hipaccCompilerOptions SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccCompilerOptions 
  PUBLIC  ${LLVM_DEFINITIONS})
