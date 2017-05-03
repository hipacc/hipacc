# Find the native Clang binary and libraries
#
#  Clang_EXECUTABLE             - clang binary
#  Clang_llvm_config_EXECUTABLE - llvm-config binary
#  Clang_FOUND                  - True if clang is found

find_package(LLVM REQUIRED CONFIG)
find_package(PackageHandleStandardArgs)

find_program(Clang_llvm_config_EXECUTABLE CACHE NAMES llvm-config DOC "llvm-config executable")
find_program(Clang_EXECUTABLE CACHE NAMES clang DOC "clang executable")

set(CLANG_LIBS
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
if (MSVC)
    # This step is required, because the variable LLVM_LIBRARY_DIRS contains a Visual Studio macro, which cannot be resolved by the OS
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
        OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif(MSVC)

function(clang_map_components_to_libnames out_libs)
    foreach(l ${CLANG_LIBS})
        find_library(LIB_${l} NAMES ${l} HINTS ${LLVM_LIBRARY_DIRS} )
        mark_as_advanced(LIB_${l})
        list(APPEND clang_libs ${LIB_${l}})
    endforeach()

    set(${out_libs} ${clang_libs} PARENT_SCOPE)
endfunction()

find_package_handle_standard_args(Clang DEFAULT_MSG Clang_EXECUTABLE Clang_llvm_config_EXECUTABLE)
