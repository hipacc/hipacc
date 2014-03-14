SET(CURRENT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/hipacc/Backend")

SET(Backend_Generators_SOURCES CPU_x86.cpp CUDA.cpp Filterscript.cpp OpenCL_ACC.cpp OpenCL_CPU.cpp OpenCL_GPU.cpp Renderscript.cpp)
SET(Backend_Generators_HEADERS "${CURRENT_HEADER_DIR}/CPU_x86.h"
	                           "${CURRENT_HEADER_DIR}/CUDA.h"
	                           "${CURRENT_HEADER_DIR}/Filterscript.h"
	                           "${CURRENT_HEADER_DIR}/OpenCL_ACC.h"
	                           "${CURRENT_HEADER_DIR}/OpenCL_CPU.h"
	                           "${CURRENT_HEADER_DIR}/OpenCL_GPU.h"
                               "${CURRENT_HEADER_DIR}/Renderscript.h")

SET(Backend_Generators_Base_HEADERS "${CURRENT_HEADER_DIR}/AcceleratorDeviceBase.h"
                                    "${CURRENT_HEADER_DIR}/AndroidBase.h"
                                    "${CURRENT_HEADER_DIR}/CodeGeneratorBaseImplT.h")

SET(Backend_Common_HEADERS "${CURRENT_HEADER_DIR}/BackendExceptions.h"
	                       "${CURRENT_HEADER_DIR}/CommonDefines.h"
	                       "${CURRENT_HEADER_DIR}/ICodeGenerator.h")

SET(Backend_SOURCES BackendConfigurationManager.cpp ${Backend_Generators_SOURCES})
SET(Backend_HEADERS "${CURRENT_HEADER_DIR}/BackendConfigurationManager.h"
	                ${Backend_Common_HEADERS}
                    ${Backend_Generators_HEADERS}
                    ${Backend_Generators_Base_HEADERS})

ADD_LIBRARY(Backend ${Backend_SOURCES} ${Backend_HEADERS})

SOURCE_GROUP("Source Files\\Generators" FILES ${Backend_Generators_SOURCES})
SOURCE_GROUP("Header Files\\Common" FILES ${Backend_Common_HEADERS})
SOURCE_GROUP("Header Files\\Generators" FILES ${Backend_Generators_HEADERS})
SOURCE_GROUP("Header Files\\Generators\\Base" FILES ${Backend_Generators_Base_HEADERS})
