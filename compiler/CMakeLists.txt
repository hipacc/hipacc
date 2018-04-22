include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBRARIES analysis core instcombine instrumentation ipo option scalaropts support target transformutils vectorize)
set(CLANG_LIBRARIES clangFrontendTool clangFrontend clangDriver clangSerialization clangCodeGen clangParse clangSema clangRewriteFrontend clangRewrite clangAnalysis clangEdit clangAST clangLex clangBasic)

set(HIPACC_LIBS
    hipaccRewrite
    hipaccCreateHostStrings
    hipaccClassRepresentation
    hipaccASTTranslate
    hipaccSIMDTypes)

if(USE_POLLY)
    list(APPEND HIPACC_LIBS hipaccPolly)
    find_library(Polly_LIBRARY NAMES Polly HINTS ${LLVM_LIBRARY_DIRS})
    mark_as_advanced(Polly_LIBRARY)
    list(APPEND LLVM_LIBRARIES ${Polly_LIBRARY})
endif()

list(APPEND HIPACC_LIBS
    hipaccHostDataDeps
    hipaccKernelStatistics
    hipaccBuiltins
    hipaccASTNode)

set(hipacc_SOURCES hipacc.cpp)
add_executable(hipacc ${hipacc_SOURCES})

target_link_libraries(hipacc ${HIPACC_LIBS} ${CLANG_LIBRARIES} ${LLVM_LIBRARIES})
add_dependencies(hipacc ${HIPACC_LIBS})


set_target_properties(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

set_target_properties(hipacc
    PROPERTIES
    LINK_FLAGS "${LLVM_COMPILE_FLAGS}")

set_target_properties(hipacc
    PROPERTIES
    VERSION ${HIPACC_VERSION})

install(TARGETS hipacc RUNTIME DESTINATION bin)
