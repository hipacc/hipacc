# Find the native Clang binary and libraries
#
#  CLANG_EXECUTABLE       - clang binary
#  LLVM_CONFIG_EXECUTABLE - llvm-config binary
#  CLANG_FOUND            - True if clang is found

FIND_PACKAGE(LLVM REQUIRED CONFIG)
FIND_PACKAGE(PackageHandleStandardArgs)

FIND_PROGRAM(LLVM_CONFIG_EXECUTABLE CACHE NAMES llvm-config DOC "llvm-config executable")
FIND_PROGRAM(CLANG_EXECUTABLE CACHE NAMES clang DOC "clang executable")

SET(CLANG_LIBS
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
    clangBasic
)

FUNCTION(clang_map_components_to_libnames out_libs)
    FOREACH(l ${CLANG_LIBS})
        FIND_LIBRARY(LIB_${l} NAMES ${l} HINTS ${LLVM_LIBRARY_DIRS} )
        MARK_AS_ADVANCED(LIB_${l})
        LIST(APPEND clang_libs ${LIB_${l}})
    ENDFOREACH(l)

    SET(${out_libs} ${clang_libs} PARENT_SCOPE)
ENDFUNCTION()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLANG DEFAULT_MSG CLANG_EXECUTABLE LLVM_CONFIG_EXECUTABLE)

