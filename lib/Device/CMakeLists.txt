set(Builtins_SOURCES Builtins.cpp)

add_library(hipaccBuiltins ${Builtins_SOURCES})
add_cppcheck(hipaccBuiltins POSSIBLE_ERROR FAIL_ON_WARNINGS)
