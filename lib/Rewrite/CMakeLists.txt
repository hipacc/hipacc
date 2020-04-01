set(Rewrite_SOURCES Rewrite.cpp)
set(CreateHostStrings_SOURCES CreateHostStrings.cpp)

add_library(hipaccRewrite ${Rewrite_SOURCES})
add_library(hipaccCreateHostStrings ${CreateHostStrings_SOURCES})

add_cppcheck(hipaccRewrite POSSIBLE_ERROR FAIL_ON_WARNINGS)
add_cppcheck(hipaccCreateHostStrings POSSIBLE_ERROR FAIL_ON_WARNINGS)
