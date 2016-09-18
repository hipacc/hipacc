INCLUDE_DIRECTORIES(${LLVM_INCLUDE_DIRS})
ADD_DEFINITIONS(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBRARIES analysis core instcombine instrumentation ipo option scalaropts support target transformutils vectorize)
clang_map_components_to_libnames(Clang_LIBRARIES)

SET(HIPACC_LIBS
    hipaccRewrite
    hipaccCreateHostStrings
    hipaccClassRepresentation
    hipaccASTTranslate
    hipaccSIMDTypes)

IF(USE_POLLY)
    LIST(APPEND HIPACC_LIBS hipaccPolly)
    FIND_LIBRARY(Polly_LIBRARY NAMES Polly HINTS ${LLVM_LIBRARY_DIRS})
    MARK_AS_ADVANCED(Polly_LIBRARY)
    LIST(APPEND LLVM_LIBRARIES ${Polly_LIBRARY})
ENDIF(USE_POLLY)

LIST(APPEND HIPACC_LIBS
    hipaccKernelStatistics
    hipaccBuiltins
    hipaccASTNode)

SET(hipacc_SOURCES hipacc.cpp)
ADD_EXECUTABLE(hipacc ${hipacc_SOURCES})

TARGET_LINK_LIBRARIES(hipacc ${HIPACC_LIBS} ${Clang_LIBRARIES} ${LLVM_LIBRARIES})
ADD_DEPENDENCIES(hipacc ${HIPACC_LIBS})


SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    VERSION ${HIPACC_VERSION})

INSTALL(TARGETS hipacc RUNTIME DESTINATION bin)
