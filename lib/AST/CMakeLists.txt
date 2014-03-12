SET(CURRENT_HEADER_DIR "${HIPACC_RELATIVE_INCLUDE_DIR}/AST")

SET(ASTNode_SOURCES ASTNode.cpp)
SET(ASTNode_HEADERS "${CURRENT_HEADER_DIR}/ASTNode.h")

SET(ASTTranslate_SOURCES ASTClone.cpp ASTTranslate.cpp BorderHandling.cpp
                         Convolution.cpp Interpolate.cpp MemoryAccess.cpp)
SET(ASTTranslate_HEADERS "${CURRENT_HEADER_DIR}/ASTTranslate.h")

ADD_LIBRARY(hipaccASTNode ${ASTNode_SOURCES} ${ASTNode_HEADERS})
ADD_LIBRARY(hipaccASTTranslate ${ASTTranslate_SOURCES} ${ASTTranslate_HEADERS})

