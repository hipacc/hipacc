# ./compiler/CMakeLists.txt

add_executable(hipacc hipacc.cpp)
add_cppcheck(hipacc POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_include_directories(hipacc SYSTEM
  PRIVATE ${CMAKE_SOURCE_DIR}/include/Device
          ${CMAKE_BINARY_DIR}/include
          ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipacc
  PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(hipacc
  PRIVATE hipaccRewrite
          hipaccBackend
          hipaccCompilerOptions
  PRIVATE clangFrontend
          clangDriver
          clangLex)
target_compile_features(hipacc
  PRIVATE cxx_std_14)

# use llvm cmake macro
if(LLVM_LINK_LLVM_DYLIB)
  set(HIPACC_LLVM_LINK_SHARED "USE_SHARED")
endif()
llvm_config(hipacc ${HIPACC_LLVM_LINK_SHARED} support)

install(TARGETS hipacc RUNTIME DESTINATION bin COMPONENT compiler)

