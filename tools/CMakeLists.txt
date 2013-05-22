IF(OPENCL_FOUND)
    INCLUDE_DIRECTORIES(
        ${OPENCL_INCLUDE_DIR}
        ${HIPACC_SOURCE_DIR}/runtime)

    SET(ocl_compile_SOURCES ocl_compile.cc)
    ADD_EXECUTABLE(ocl_compile ${ocl_compile_SOURCES})
    TARGET_LINK_LIBRARIES(ocl_compile ${OPENCL_LFLAGS} ${TIME_LINK})

    INSTALL(TARGETS ocl_compile RUNTIME DESTINATION bin)

    SET(ocl_bandwidth_test_SOURCES ocl_bandwidth_test.cc)
    ADD_EXECUTABLE(ocl_bandwidth_test ${ocl_bandwidth_test_SOURCES})
    TARGET_LINK_LIBRARIES(ocl_bandwidth_test ${OPENCL_LFLAGS} ${TIME_LINK})

    INSTALL(TARGETS ocl_bandwidth_test RUNTIME DESTINATION bin)
ENDIF(OPENCL_FOUND)

