if(OpenCL_FOUND)
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/runtime)

    set(cl_compile_SOURCES cl_compile.cc)
    add_executable(cl_compile ${cl_compile_SOURCES})
    target_compile_options(cl_compile PRIVATE
        -DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
    target_link_libraries(cl_compile PRIVATE ${OpenCL_LIBRARIES})

    install(TARGETS cl_compile RUNTIME DESTINATION bin COMPONENT tools)

    set(cl_bandwidth_test_SOURCES cl_bandwidth_test.cc)
    add_executable(cl_bandwidth_test ${cl_bandwidth_test_SOURCES})
    target_compile_options(cl_bandwidth_test PRIVATE
        -DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
    target_link_libraries(cl_bandwidth_test PRIVATE ${OpenCL_LIBRARIES})

    install(TARGETS cl_bandwidth_test RUNTIME DESTINATION bin COMPONENT tools)
endif()
