cmake_minimum_required(VERSION 3.14)

configure_file(CMakeLists.txt.in ${CMAKE_BINARY_DIR}/samples/CMakeLists.txt @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/samples/CMakeLists.txt DESTINATION samples COMPONENT samples)

# install common files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common DESTINATION samples COMPONENT samples)

# install readme file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/readme.txt DESTINATION samples COMPONENT samples)

file(GLOB SAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/[0-9]*)
foreach(SAMPLE_DIR IN LISTS SAMPLE_DIRS)
    # install samples
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SAMPLE_DIR} DESTINATION samples COMPONENT samples)
endforeach()
