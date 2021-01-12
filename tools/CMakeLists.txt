# ./tools/CMakeLists.txt

if(OpenCL_FOUND)
  add_executable(cl_compile cl_compile.cc)
  add_cppcheck(cl_compile POSSIBLE_ERROR FAIL_ON_WARNINGS)
  target_include_directories(cl_compile
    PRIVATE ${OpenCL_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/runtime)
  target_compile_options(cl_compile
    PRIVATE -DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  target_link_libraries(cl_compile
    PRIVATE ${OpenCL_LIBRARIES})
  install(TARGETS cl_compile RUNTIME DESTINATION bin COMPONENT tools)

  add_executable(cl_bandwidth_test cl_bandwidth_test.cc)
  add_cppcheck(cl_bandwidth_test POSSIBLE_ERROR FAIL_ON_WARNINGS)
  target_include_directories(cl_bandwidth_test
    PRIVATE ${OpenCL_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/runtime)
  target_compile_options(cl_bandwidth_test
    PRIVATE -DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  target_link_libraries(cl_bandwidth_test
    PRIVATE ${OpenCL_LIBRARIES})
  install(TARGETS cl_bandwidth_test RUNTIME DESTINATION bin COMPONENT tools)
endif()
