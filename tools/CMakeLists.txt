if(OpenCL_FOUND)
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/runtime)

    set(cl_compile_SOURCES cl_compile.cc)
    add_executable(cl_compile ${cl_compile_SOURCES})
    target_link_libraries(cl_compile ${OpenCL_LIBRARIES} ${RT_LIBRARIES})

    install(TARGETS cl_compile RUNTIME DESTINATION bin)

    set(cl_bandwidth_test_SOURCES cl_bandwidth_test.cc)
    add_executable(cl_bandwidth_test ${cl_bandwidth_test_SOURCES})
    target_link_libraries(cl_bandwidth_test ${OpenCL_LIBRARIES} ${RT_LIBRARIES})

    install(TARGETS cl_bandwidth_test RUNTIME DESTINATION bin)
endif()
