set(Rewrite_SOURCES Rewrite.cpp)
set(CreateHostStrings_SOURCES CreateHostStrings.cpp)

add_library(hipaccRewrite ${Rewrite_SOURCES})
add_library(hipaccCreateHostStrings ${CreateHostStrings_SOURCES})
