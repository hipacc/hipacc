IF(OpenCL_FOUND)
    INCLUDE_DIRECTORIES(
        ${OpenCL_INCLUDE_DIRS}
        ${HIPACC_SOURCE_DIR}/runtime)

    SET(cl_compile_SOURCES cl_compile.cc)
    ADD_EXECUTABLE(cl_compile ${cl_compile_SOURCES})
    TARGET_LINK_LIBRARIES(cl_compile ${OpenCL_LIBRARIES} ${RT_LIBRARIES})

    INSTALL(TARGETS cl_compile RUNTIME DESTINATION bin)

    SET(cl_bandwidth_test_SOURCES cl_bandwidth_test.cc)
    ADD_EXECUTABLE(cl_bandwidth_test ${cl_bandwidth_test_SOURCES})
    TARGET_LINK_LIBRARIES(cl_bandwidth_test ${OpenCL_LIBRARIES} ${RT_LIBRARIES})

    INSTALL(TARGETS cl_bandwidth_test RUNTIME DESTINATION bin)
ENDIF(OpenCL_FOUND)
