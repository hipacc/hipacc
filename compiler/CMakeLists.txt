SET(CLANG_USED_LIBS
    clangFrontendTool
    clangFrontend
    clangDriver
    clangSerialization
    clangCodeGen
    clangParse
    clangSema
    clangStaticAnalyzerFrontend
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    clangAnalysis
    clangARCMigrate
    clangRewrite
    clangEdit
    clangAST
    clangLex
    clangBasic
    )

SET(HIPACC_USED_LIBS
    Rewrite
    CreateHostStrings
    ClassRepresentation
    ASTTranslate
    SIMDTypes)

IF(USE_POLLY)
    LIST(APPEND HIPACC_USED_LIBS Polly)
    ADD_LIBRARY(LLVMPolly SHARED IMPORTED)
    SET_TARGET_PROPERTIES(LLVMPolly
        PROPERTIES
        IMPORTED_LOCATION ${LLVM_LIBRARY_DIR}/LLVMPolly${CMAKE_SHARED_LIBRARY_SUFFIX})
ENDIF(USE_POLLY)

LIST(APPEND HIPACC_USED_LIBS
    KernelStatistics
    Builtins
    ASTNode)

IF(USE_POLLY)
    LIST(APPEND HIPACC_USED_LIBS LLVMPolly)
ENDIF(USE_POLLY)

SET(hipacc_SOURCES hipacc.cpp)
ADD_EXECUTABLE(hipacc ${hipacc_SOURCES})

LINK_DIRECTORIES(${LLVM_LIBRARY_DIR})


TARGET_LINK_LIBRARIES(hipacc ${HIPACC_USED_LIBS} ${CLANG_USED_LIBS} ${LLVM_MODULE_LIBS} ${LLVM_LFLAGS})
ADD_DEPENDENCIES(hipacc ${HIPACC_USED_LIBS})

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    VERSION ${HIPACC_VERSION})

INSTALL(TARGETS hipacc RUNTIME DESTINATION bin)

