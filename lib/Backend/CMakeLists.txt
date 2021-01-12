# ./lib/Backend/CMakeLists.txt

add_library(hipaccBackend BackendConfigurationManager.cpp)
add_cppcheck(hipaccBackend POSSIBLE_ERROR FAIL_ON_WARNINGS)
target_sources(hipaccBackend
  PRIVATE VectorizationAST.cpp
          Vectorizer.cpp
          CPU_x86.cpp
          CUDA.cpp
          OpenCL_ACC.cpp
          OpenCL_CPU.cpp
          OpenCL_GPU.cpp
          ClangASTHelper.cpp
          InstructionSets.cpp)
target_include_directories(hipaccBackend
  PRIVATE ${CMAKE_SOURCE_DIR}/include
          ${CMAKE_BINARY_DIR}/include)
target_include_directories(hipaccBackend SYSTEM
  PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(hipaccBackend
  PRIVATE ${LLVM_DEFINITIONS}
          HIPACC_VERSION="\\"${HIPACC_VERSION}\\""
          GIT_REPOSITORY="\\"${HIPACC_GIT_REPOSITORY}\\""
          GIT_VERSION="\\"${HIPACC_GIT_VERSION}\\""
          CLANG_LIB_INCLUDE_DIR="${CLANG_LIB_INCLUDE_DIR}"
          "$<$<AND:$<NOT:$<BOOL:${APPLE}>>,$<BOOL:${CUDA_FOUND}>>:USE_JIT_ESTIMATE>"
          "$<$<BOOL:${MSVC}>:HOST_COMPILER_INSTALL_PREFIX=${HOST_COMPILER_INSTALL_PREFIX}>")

# Activate Runtime Type Information for the Backend library
if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
  string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif(MSVC)
  string(REPLACE "/GR-" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  target_compile_options(hipaccBackend PRIVATE "/GR")
endif()

target_link_libraries(hipaccBackend
  PUBLIC  hipaccCompilerOptions
  PRIVATE hipaccASTTranslate
          hipaccASTNode
          hipaccClassRepresentation
          clangAST)
