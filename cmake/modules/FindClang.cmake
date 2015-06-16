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

# Overwrite the LLVM library dir with the correct value
IF(MSVC)
    # This step is required, because the variable LLVM_LIBRARY_DIRS contains a Visual Studio macro, which cannot be resolved by the OS
    EXECUTE_PROCESS(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
        OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
ENDIF(MSVC)

FUNCTION(clang_map_components_to_libnames out_libs)
    FOREACH(l ${CLANG_LIBS})
        FIND_LIBRARY(LIB_${l} NAMES ${l} HINTS ${LLVM_LIBRARY_DIRS} )
        MARK_AS_ADVANCED(LIB_${l})
        LIST(APPEND clang_libs ${LIB_${l}})
    ENDFOREACH(l)

    SET(${out_libs} ${clang_libs} PARENT_SCOPE)
ENDFUNCTION()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Clang DEFAULT_MSG CLANG_EXECUTABLE LLVM_CONFIG_EXECUTABLE)

