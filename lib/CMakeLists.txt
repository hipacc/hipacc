include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")
endif()

add_subdirectory(Analysis)
add_subdirectory(AST)
add_subdirectory(Backend)
add_subdirectory(Device)
add_subdirectory(DSL)
add_subdirectory(Rewrite)
add_subdirectory(Vectorization)
