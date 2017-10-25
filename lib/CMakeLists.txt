include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(Analysis)
add_subdirectory(AST)
add_subdirectory(Device)
add_subdirectory(DSL)
add_subdirectory(Rewrite)
add_subdirectory(Vectorization)
