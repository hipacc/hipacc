SET(ASTNode_SOURCES ASTNode.cpp)
SET(ASTTranslate_SOURCES ASTClone.cpp ASTTranslate.cpp BorderHandling.cpp
    Convolution.cpp Interpolate.cpp MemoryAccess.cpp)

ADD_LIBRARY(ASTNode ${ASTNode_SOURCES})
ADD_LIBRARY(ASTTranslate ${ASTTranslate_SOURCES})

