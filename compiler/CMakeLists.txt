include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIBRARIES analysis core instcombine instrumentation ipo option scalaropts support target transformutils vectorize)
set(CLANG_LIBRARIES clangFrontendTool clangFrontend clangDriver clangSerialization clangCodeGen clangParse clangSema clangRewriteFrontend clangRewrite clangAnalysis clangEdit clangAST clangLex clangBasic)

set(HIPACC_LIBS
    hipaccRewrite
    hipaccBackend
    hipaccCreateHostStrings
    hipaccClassRepresentation
    hipaccASTTranslate
    hipaccSIMDTypes)

list(APPEND HIPACC_LIBS
    hipaccKernelStatistics
    hipaccHostDataDeps
    hipaccBuiltins
    hipaccASTFuse
    hipaccASTNode)

set(hipacc_SOURCES hipacc.cpp)
add_executable(hipacc ${hipacc_SOURCES})
add_cppcheck(hipacc POSSIBLE_ERROR FAIL_ON_WARNINGS)

if(LLVM_LINK_LLVM_DYLIB)
    set(HIPACC_LLVM_LINK_SHARED "USE_SHARED")
endif()
target_link_libraries(hipacc PRIVATE ${HIPACC_LIBS})
target_link_libraries(hipacc PRIVATE ${CLANG_LIBRARIES})
llvm_config(hipacc ${HIPACC_LLVM_LINK_SHARED} ${LLVM_LIBRARIES})
add_dependencies(hipacc ${HIPACC_LIBS})


set_target_properties(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

install(TARGETS hipacc RUNTIME DESTINATION bin COMPONENT compiler)
