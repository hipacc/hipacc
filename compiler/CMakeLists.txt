# ./compiler/CMakeLists.txt

add_executable(hipacc hipacc.cpp)
add_cppcheck(hipacc POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipacc
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_include_directories(hipacc SYSTEM
  PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipacc
  PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(hipacc
  PRIVATE hipaccRewrite
          hipaccBackend
          hipaccCompilerOptions
  PRIVATE clangFrontend
          clangDriver
          clangLex)

# use llvm cmake macro
if(LLVM_LINK_LLVM_DYLIB)
  set(HIPACC_LLVM_LINK_SHARED "USE_SHARED")
endif()
llvm_config(hipacc ${HIPACC_LLVM_LINK_SHARED} support)

install(TARGETS hipacc RUNTIME DESTINATION bin COMPONENT compiler)

