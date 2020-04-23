set(CURRENT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/hipacc/Backend")
string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(Backend_Generators_SOURCES CPU_x86.cpp CUDA.cpp Filterscript.cpp OpenCL_ACC.cpp OpenCL_CPU.cpp OpenCL_GPU.cpp Renderscript.cpp)
set(Backend_Generators_HEADERS "${CURRENT_HEADER_DIR}/CPU_x86.h"
                               "${CURRENT_HEADER_DIR}/CUDA.h"
                               "${CURRENT_HEADER_DIR}/Filterscript.h"
                               "${CURRENT_HEADER_DIR}/OpenCL_ACC.h"
                               "${CURRENT_HEADER_DIR}/OpenCL_CPU.h"
                               "${CURRENT_HEADER_DIR}/OpenCL_GPU.h"
                               "${CURRENT_HEADER_DIR}/Renderscript.h")

set(Backend_Generators_Base_HEADERS "${CURRENT_HEADER_DIR}/AcceleratorDeviceBase.h"
                                    "${CURRENT_HEADER_DIR}/AndroidBase.h"
                                    "${CURRENT_HEADER_DIR}/CodeGeneratorBaseImplT.h")

set(Backend_Common_SOURCES ClangASTHelper.cpp InstructionSets.cpp)

set(Backend_Common_HEADERS "${CURRENT_HEADER_DIR}/BackendExceptions.h"
                           "${CURRENT_HEADER_DIR}/ClangASTHelper.h"
                           "${CURRENT_HEADER_DIR}/CommonDefines.h"
                           "${CURRENT_HEADER_DIR}/ICodeGenerator.h"
                           "${CURRENT_HEADER_DIR}/InstructionSets.h"
                           "${CURRENT_HEADER_DIR}/OptionParsers.h"
                           "${CURRENT_HEADER_DIR}/VectorizationAST.h"
                           "${CURRENT_HEADER_DIR}/Vectorizer.h")

set(Backend_SOURCES BackendConfigurationManager.cpp VectorizationAST.cpp Vectorizer.cpp ${Backend_Common_SOURCES} ${Backend_Generators_SOURCES})
set(Backend_HEADERS "${CURRENT_HEADER_DIR}/BackendConfigurationManager.h"
                    ${Backend_Common_HEADERS}
                    ${Backend_Generators_HEADERS}
                    ${Backend_Generators_Base_HEADERS})

add_library(hipaccBackend ${Backend_SOURCES} ${Backend_HEADERS})

source_group("Source Files\\Common" FILES ${Backend_Common_SOURCES})
source_group("Source Files\\Generators" FILES ${Backend_Generators_SOURCES})
source_group("Header Files\\Common" FILES ${Backend_Common_HEADERS})
source_group("Header Files\\Generators" FILES ${Backend_Generators_HEADERS})
source_group("Header Files\\Generators\\Base" FILES ${Backend_Generators_Base_HEADERS})

# Activate Runtime Type Information for the Backend library
if (MSVC)
    set_target_properties(hipaccBackend PROPERTIES COMPILE_OPTIONS "/GR")
endif(MSVC)

