cmake_minimum_required(VERSION 3.14)

configure_file(CMakeLists.txt.in ${CMAKE_BINARY_DIR}/samples/CMakeLists.txt @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/samples/CMakeLists.txt DESTINATION samples COMPONENT samples)

# install common files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common DESTINATION samples COMPONENT samples)

# install readme file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/readme.txt DESTINATION samples COMPONENT samples)

# install samples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/0_Point_Operators DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/1_Local_Operators DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/2_Global_Operators DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3_Preprocessing DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/4_Postprocessing DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/5_Other DESTINATION samples COMPONENT samples)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/6_Test DESTINATION samples COMPONENT samples)
