SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti ${LLVM_CXXFLAGS}")
SET(CLANG_USED_LIBS
    clangFrontendTool
    clangFrontend
    clangDriver
    clangSerialization
    clangCodeGen
    clangParse
    clangSema
    clangRewriteFrontend
    clangRewrite
    clangAnalysis
    clangEdit
    clangAST
    clangLex
    clangBasic)

SET(HIPACC_USED_LIBS
    hipaccRewrite
    hipaccCreateHostStrings
    hipaccClassRepresentation
    hipaccASTTranslate
    hipaccSIMDTypes)

IF(USE_POLLY)
    LIST(APPEND HIPACC_USED_LIBS hipaccPolly)
    LIST(APPEND LLVM_MODULE_LIBS Polly)
ENDIF(USE_POLLY)

LIST(APPEND HIPACC_USED_LIBS
    hipaccKernelStatistics
    hipaccBuiltins
    hipaccASTNode)

SET(hipacc_SOURCES hipacc.cpp)
ADD_EXECUTABLE(hipacc ${hipacc_SOURCES})

LINK_DIRECTORIES(${LLVM_LIBRARY_DIR})


TARGET_LINK_LIBRARIES(hipacc ${HIPACC_USED_LIBS} ${CLANG_USED_LIBS} ${LLVM_MODULE_LIBS} ${LLVM_LFLAGS} ${LLVM_SYSFLAGS})
ADD_DEPENDENCIES(hipacc ${HIPACC_USED_LIBS})

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    VERSION ${HIPACC_VERSION})

INSTALL(TARGETS hipacc RUNTIME DESTINATION bin)

