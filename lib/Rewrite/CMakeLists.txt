SET(Rewrite_SOURCES Rewrite.cpp)
SET(CreateHostStrings_SOURCES CreateHostStrings.cpp)

ADD_LIBRARY(hipaccRewrite ${Rewrite_SOURCES})
ADD_LIBRARY(hipaccCreateHostStrings ${CreateHostStrings_SOURCES})

