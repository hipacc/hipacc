set(SIMDTypes_SOURCES SIMDTypes.cpp)

add_library(hipaccSIMDTypes ${SIMDTypes_SOURCES})
add_cppcheck(hipaccSIMDTypes POSSIBLE_ERROR FAIL_ON_WARNINGS)
