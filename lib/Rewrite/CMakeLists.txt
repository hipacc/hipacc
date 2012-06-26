SET(Rewrite_SOURCES Rewrite.cpp)
SET(CreateHostStrings_SOURCES CreateHostStrings.cpp)

ADD_LIBRARY(Rewrite ${Rewrite_SOURCES})
ADD_LIBRARY(CreateHostStrings ${CreateHostStrings_SOURCES})

