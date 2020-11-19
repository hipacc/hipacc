cmake_minimum_required(VERSION 3.14)

option(HIPACC_SKIP_CODEGEN "Skip automatic code generation by Hipacc" OFF)

set(HIPACC_PATH "${CMAKE_CURRENT_LIST_DIR}/../")

find_program(HIPACC_EXE hipacc
             HINTS "${HIPACC_PATH}/bin" "$ENV{HIPACC_PATH}/bin"
                   "/usr/local/hipacc/bin")

if(HIPACC_EXE-NOTFOUND)
    message(FATAL_ERROR "Hipacc not found")
endif()

get_filename_component(HIPACC_PATH "${HIPACC_EXE}" DIRECTORY)
get_filename_component(HIPACC_PATH "${HIPACC_PATH}" DIRECTORY)
get_filename_component(CCBIN_PATH "${CMAKE_CXX_COMPILER}" DIRECTORY)

message(STATUS "Found Hipacc: ${HIPACC_PATH}")

find_package(CUDA 10)
find_package(OpenCL)

find_program(NVCC_EXE nvcc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin/)
find_program(CL_COMPILER_EXE cl_compile HINTS ${HIPACC_PATH}/bin/)

if(NOT NVCC_EXE)
    message(WARNING "Hipacc: Could not find CUDA compiler!")
endif()

if(NOT CL_COMPILER_EXE)
    message(WARNING "Hipacc: Could not find the OpenCL compiler!")
endif()

set(HIPACC_OPTIONS_CPU "")
set(HIPACC_OPTIONS_CUDA "-nvcc-path" "${NVCC_EXE}" "-ccbin-path" "${CCBIN_PATH}")
set(HIPACC_OPTIONS_OPENCL "")
set(HIPACC_OPTIONS_OPENCL_ACC ${HIPACC_OPTIONS_OPENCL})
set(HIPACC_OPTIONS_OPENCL_CPU ${HIPACC_OPTIONS_OPENCL})
set(HIPACC_OPTIONS_OPENCL_GPU ${HIPACC_OPTIONS_OPENCL} "-cl-compiler-path" "${CL_COMPILER_EXE}")
set(HIPACC_RT_INCLUDE_DIRS "${HIPACC_PATH}/include/")

function(add_hipacc_sources)

    set(options PUBLIC PRIVATE INTERFACE)
    set(oneValueArgs TARGET TARGET_ARCH HIPACC_MIN_VERSION HIPACC_EXACT_VERSION OUTPUT_DIR_VAR)
    set(multiValueArgs SOURCES OPTIONS INCLUDE_DIRS)
    
    cmake_parse_arguments(ARG
                          "${options}"
                          "${oneValueArgs}"
                          "${multiValueArgs}"
                          ${ARGN})
                          
    list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS})
    list(APPEND _HIPACC_OPTIONS ${ARG_OPTIONS})
    string(REPLACE " " ";" _HIPACC_OPTIONS "${_HIPACC_OPTIONS}")

    ##################################
    # define C++ standard based on target property

    get_target_property(TARGET_CXX_STANDARD ${ARG_TARGET} CXX_STANDARD)
    list(APPEND _HIPACC_OPTIONS "-std=c++${TARGET_CXX_STANDARD}")
    list(APPEND _HIPACC_OPTIONS "-nostdinc++")
    list(APPEND _HIPACC_OPTIONS "-fexceptions")

    ##################################
    # check hipacc version requirement

    if(ARG_HIPACC_EXACT_VERSION)
        if(NOT ARG_HIPACC_EXACT_VERSION VERSION_EQUAL HIPACC_PACKAGE_VERSION)
            message(SEND_ERROR "Found Hipacc version ${HIPACC_PACKAGE_VERSION} does not match required version ${ARG_HIPACC_EXACT_VERSION}!")
        endif()
    elseif(ARG_HIPACC_MIN_VERSION)
        if(ARG_HIPACC_MIN_VERSION VERSION_GREATER HIPACC_PACKAGE_VERSION)
            message(SEND_ERROR "Found Hipacc version ${HIPACC_PACKAGE_VERSION} does not match required minimum version ${ARG_HIPACC_MIN_VERSION}!")
        endif()
    endif()
    
    #######################
    # check scope arguments

    if(${ARG_PRIVATE})
        set(_SCOPE PRIVATE)
    endif()

    if(${ARG_PUBLIC})
        if(_SCOPE)
            message(SEND_ERROR "add_hipacc_sources: Multiple scope options are not allowed!")
        endif()

        set(_SCOPE PUBLIC)
    endif()

    if(${ARG_INTERFACE})
        if(_SCOPE)
            message(SEND_ERROR "add_hipacc_sources: Multiple scope options are not allowed!")
        endif()
        
        set(_SCOPE INTERFACE)
    endif()

    if(NOT _SCOPE)
        message(WARNING "add_hipacc_sources: Not scope option specified")
    endif()   
    
    ##############################################
    # prepare target architecture specific options

    if(NOT ARG_TARGET_ARCH)
        message(FATAL_ERROR "add_hipacc_sources: No valid target architecture specified!")
    endif()

    # prepare CPU target device
    if("${ARG_TARGET_ARCH}" STREQUAL "CPU")
        list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS_CPU} "-emit-cpu")
        set(_OUTPUT_EXT ".cpp")

    # prepare CUDA target device
    elseif("${ARG_TARGET_ARCH}" STREQUAL "CUDA")

        if(NOT CUDA_FOUND)
            message(FATAL_ERROR "Required CUDA not found!")
        endif()

        list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS_CUDA} "-emit-cuda")
        set(_OUTPUT_EXT ".cu")

        set_target_properties(${ARG_TARGET} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

        get_filename_component(CUDA_LIB_DIR "${CUDA_LIBRARIES}" DIRECTORY)
        target_link_directories(${ARG_TARGET} ${_SCOPE} ${CUDA_LIB_DIR})
        target_link_libraries(${ARG_TARGET} ${_SCOPE} cuda nvrtc)
    
    # prepare OpenCL target devices (accelerators, cpu, gpu)
    elseif("${ARG_TARGET_ARCH}" MATCHES  "^OPENCL-(ACC|CPU|GPU)$")

        if(NOT OpenCL_FOUND)
            message(FATAL_ERROR "Required OpenCL not found!")
        endif()

        if("${ARG_TARGET_ARCH}" STREQUAL  "OPENCL-ACC")
            list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS_OPENCL_ACC} "-emit-opencl-acc")
        elseif("${ARG_TARGET_ARCH}" STREQUAL  "OPENCL-CPU")
            list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS_OPENCL_CPU} "-emit-opencl-cpu")
        else()
            list(APPEND _HIPACC_OPTIONS ${HIPACC_OPTIONS_OPENCL_GPU} "-emit-opencl-gpu")
        endif()

        set(_OUTPUT_EXT ".cxx")

        target_link_libraries(${ARG_TARGET} ${_SCOPE} OpenCL::OpenCL)

    else()
        message(FATAL_ERROR "add_hipacc_sources: No valid target architecture specified!")
    endif()
    
    if(NOT _OUTPUT_EXT)
        message(FATAL_ERROR "add_hipacc_sources: Invalid file extension")
    endif()
    
    #############################
    # prepare include directories

    list(APPEND _HIPACC_INCLUDE_DIRS ${HIPACC_INCLUDE_DIRS})
    list(APPEND _HIPACC_INCLUDE_DIRS ${ARG_INCLUDE_DIRS})
    list(TRANSFORM _HIPACC_INCLUDE_DIRS PREPEND "-I;")
    list(APPEND _HIPACC_OPTIONS ${_HIPACC_INCLUDE_DIRS})
    
    ###########################
    # add all sources to target

    foreach(_SRC ${ARG_SOURCES})

        get_filename_component(_SRC ${_SRC} ABSOLUTE)      

        if(NOT EXISTS ${_SRC})
            message(FATAL_ERROR "add_hipacc_sources: Hipacc DSL file ${_SRC} does not exist!")
        endif()
        
        # prepare output file name for generated source

        get_filename_component(_SRC_NAME_WLE ${_SRC} NAME_WLE) 
        get_filename_component(_SRC_DIR ${_SRC} DIRECTORY ) 
        
        set(_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipacc/${ARG_TARGET}/${_SRC_NAME_WLE})
        set(_OUTPUT_FILE_NAME ${_SRC_NAME_WLE}_${ARG_TARGET_ARCH}${_OUTPUT_EXT})
        set(_OUTPUT_FILE_PATH ${_OUTPUT_DIR}/${_OUTPUT_FILE_NAME})
        file(RELATIVE_PATH _OUTPUT_FILE_PATH_REL "${CMAKE_CURRENT_BINARY_DIR}" "${_OUTPUT_DIR}")

        file(MAKE_DIRECTORY "${_OUTPUT_DIR}") 

        if(ARG_OUTPUT_DIR_VAR)
            set(${ARG_OUTPUT_DIR_VAR} "${_OUTPUT_DIR}" PARENT_SCOPE)
        endif()

        set(_HIPACC_CMD_DEPENDENCIES ${_SRC})

        # compile hipacc command

        if(HIPACC_SKIP_CODEGEN)
            set(_HIPACC_CMD echo WARNING: Automatic code generation of ${_SRC} by Hipacc is skipped! Disable the CMake option HIPACC_SKIP_CODEGEN to run the code generation.)
        else()
            set(_HIPACC_CMD "${HIPACC_EXE}" ${_HIPACC_OPTIONS} -o "${_OUTPUT_FILE_PATH}" "${_SRC}")
        endif()

        message(VERBOSE "Running Hipacc Command: ${_HIPACC_CMD}")
        
        # create custom command to run hipacc

        add_custom_command(
            OUTPUT  "${_OUTPUT_FILE_PATH_REL}/${_OUTPUT_FILE_NAME}"
            COMMAND ${_HIPACC_CMD}
            DEPENDS ${_HIPACC_CMD_DEPENDENCIES}
            WORKING_DIRECTORY "${_OUTPUT_DIR}"
        )
       
        # add generated source to target

        target_sources(${ARG_TARGET} PRIVATE "${_OUTPUT_FILE_PATH}")
        target_include_directories(${ARG_TARGET} ${_SCOPE} "${_OUTPUT_DIR}" "${_SRC_DIR}" ${HIPACC_RT_INCLUDE_DIRS})
        
    endforeach()

    ######################################
    # add hipacc runtime library to target

    target_include_directories(${ARG_TARGET} ${_SCOPE} ${HIPACC_RT_INCLUDE_DIRS} ${ARG_INCLUDE_DIRS})

endfunction()
