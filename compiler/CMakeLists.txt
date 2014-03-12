INCLUDE_DIRECTORIES(${LLVM_INCLUDE_DIRS})
ADD_DEFINITIONS(${LLVM_DEFINITIONS})

IF (MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")
ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti ${LLVM_CXXFLAGS}")
ENDIF()

llvm_map_components_to_libnames(LLVM_LIBS analysis core instcombine instrumentation ipa ipo option scalaropts support target transformutils vectorize)
clang_map_components_to_libnames(CLANG_LIBS)

SET(HIPACC_LIBS
    hipaccRewrite
    hipaccBackend
    hipaccCreateHostStrings
    hipaccClassRepresentation
    hipaccASTTranslate
    hipaccSIMDTypes)

IF(USE_POLLY)
    LIST(APPEND HIPACC_LIBS hipaccPolly)
    FIND_LIBRARY(POLLY_LIB NAMES Polly HINTS ${LLVM_LIBRARY_DIRS})
    MARK_AS_ADVANCED(POLLY_LIB)
    LIST(APPEND LLVM_LIBS ${POLLY_LIB})
ENDIF(USE_POLLY)

LIST(APPEND HIPACC_LIBS
    hipaccKernelStatistics
    hipaccBuiltins
    hipaccASTNode)

LINK_DIRECTORIES(${LLVM_LIBRARY_DIR})

SET(hipacc_SOURCES hipacc.cpp)
SET(hipacc_HEADERS hipacc.h "../include/hipacc/Config/CompilerOptions.h"
                            "${HIPACC_BINARY_DIR}/include/hipacc/Config/config.h")
ADD_EXECUTABLE(hipacc ${hipacc_SOURCES} ${hipacc_HEADERS})

TARGET_LINK_LIBRARIES(hipacc ${HIPACC_LIBS} ${CLANG_LIBS} ${LLVM_LIBS})
ADD_DEPENDENCIES(hipacc ${HIPACC_LIBS})


SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")

SET_TARGET_PROPERTIES(hipacc
    PROPERTIES
    VERSION ${HIPACC_VERSION})

INSTALL(TARGETS hipacc RUNTIME DESTINATION bin)

