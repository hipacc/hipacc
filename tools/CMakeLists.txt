IF(OPENCL_FOUND)
    INCLUDE_DIRECTORIES(
        ${OPENCL_INCLUDE_DIR}
        ${HIPACC_SOURCE_DIR}/runtime)

    SET(cl_compile_SOURCES cl_compile.cc)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    ADD_EXECUTABLE(cl_compile ${cl_compile_SOURCES})
    TARGET_LINK_LIBRARIES(cl_compile ${OPENCL_LFLAGS} ${TIME_LINK})

    INSTALL(TARGETS cl_compile RUNTIME DESTINATION bin)

    SET(cl_bandwidth_test_SOURCES cl_bandwidth_test.cc)
    ADD_EXECUTABLE(cl_bandwidth_test ${cl_bandwidth_test_SOURCES})
    TARGET_LINK_LIBRARIES(cl_bandwidth_test ${OPENCL_LFLAGS} ${TIME_LINK})

    INSTALL(TARGETS cl_bandwidth_test RUNTIME DESTINATION bin)
ENDIF(OPENCL_FOUND)

