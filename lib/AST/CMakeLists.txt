SET(ASTNode_SOURCES ASTNode.cpp)
SET(ASTTranslate_SOURCES ASTClone.cpp ASTTranslate.cpp BorderHandling.cpp
    Convolution.cpp Interpolate.cpp MemoryAccess.cpp)

ADD_LIBRARY(hipaccASTNode ${ASTNode_SOURCES})
ADD_LIBRARY(hipaccASTTranslate ${ASTTranslate_SOURCES})

