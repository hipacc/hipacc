SET(ASTNode_SOURCES ASTNode.cpp)
SET(ASTTranslate_SOURCES ASTTranslate.cpp BorderHandling.cpp Interpolate.cpp MemoryAccess.cpp)

ADD_LIBRARY(ASTNode ${ASTNode_SOURCES})
ADD_LIBRARY(ASTTranslate ${ASTTranslate_SOURCES})

