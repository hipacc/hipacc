cmake_minimum_required(VERSION 3.14)

set(HIPACC_PATH "${CMAKE_CURRENT_LIST_DIR}/../")

find_program(HIPACC_EXE hipacc
             HINTS "${HIPACC_PATH}/bin" "$ENV{HIPACC_PATH}/bin"
            )

if(HIPACC_EXE-NOTFOUND)
    message(FATAL_ERROR "Hipacc not found")
endif()

get_filename_component(HIPACC_PATH "${HIPACC_EXE}" DIRECTORY)
get_filename_component(HIPACC_PATH "${HIPACC_PATH}" DIRECTORY)

message(STATUS "Found Hipacc: ${HIPACC_PATH}")

find_package(CUDA 7)
find_package(OpenCL)

set(HIPACC_OPTIONS "-std=c++11")
set(HIPACC_OPTIONS "-nostdinc++")
set(HIPACC_OPTIONS_CPU "")
set(HIPACC_OPTIONS_CUDA "")
set(HIPACC_OPTIONS_OPENCL_ACC "")
set(HIPACC_OPTIONS_OPENCL_CPU "")
set(HIPACC_OPTIONS_OPENCL_GPU "")
set(HIPACC_INCLUDE_DIRS "${HIPACC_PATH}/include/dsl" "${HIPACC_PATH}/include/c++/v1" "${HIPACC_PATH}/include/clang")
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

        if(NOT EXISTS ${_SRC})
            message(FATAL_ERROR "add_hipacc_sources: Hipacc DSL file ${_SRC} does not exist!")
        endif()
        
        # prepare output file name for generated source

        get_filename_component(_SRC_NAME_WLE ${_SRC} NAME_WLE) 
        
        set(_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipacc/${ARG_TARGET}/${_SRC_NAME_WLE})
        set(_OUTPUT_FILE_NAME ${_SRC_NAME_WLE}_${ARG_TARGET_ARCH}${_OUTPUT_EXT})
        set(_OUTPUT_FILE_PATH ${_OUTPUT_DIR}/${_OUTPUT_FILE_NAME})
        file(RELATIVE_PATH _OUTPUT_FILE_PATH_REL "${CMAKE_CURRENT_BINARY_DIR}" "${_OUTPUT_DIR}")

        file(MAKE_DIRECTORY "${_OUTPUT_DIR}") 

        if(ARG_OUTPUT_DIR_VAR)
            set(${ARG_OUTPUT_DIR_VAR} "${_OUTPUT_DIR}" PARENT_SCOPE)
        endif()

        # compile hipacc command

        set(_HIPACC_CMD "${HIPACC_EXE}" ${_HIPACC_OPTIONS} "${_SRC}" -o "${_OUTPUT_FILE_PATH}")

        message(VERBOSE "Running Hipacc Command: ${_HIPACC_CMD}")
        
        # create custom command to run hipacc

        add_custom_command(
            OUTPUT  "${_OUTPUT_FILE_PATH_REL}/${_OUTPUT_FILE_NAME}"
            COMMAND ${_HIPACC_CMD}
            DEPENDS ${_SRC}
            WORKING_DIRECTORY "${_OUTPUT_DIR}"
        )
       
        # add generated source to target

        target_sources(${ARG_TARGET} ${_SCOPE} "${_OUTPUT_FILE_PATH}")
        target_include_directories(${ARG_TARGET} ${_SCOPE} "${_OUTPUT_DIR}")
        
    endforeach()

    ######################################
    # add hipacc runtime library to target

    target_include_directories(${ARG_TARGET} ${_SCOPE} ${HIPACC_RT_INCLUDE_DIRS} ${ARG_INCLUDE_DIRS})

endfunction()
