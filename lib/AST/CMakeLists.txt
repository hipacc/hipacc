set(ASTFuse_SOURCES ASTFuse.cpp)
set(ASTNode_SOURCES ASTNode.cpp)
set(ASTTranslate_SOURCES ASTClone.cpp ASTTranslate.cpp BorderHandling.cpp Convolution.cpp Interpolate.cpp MemoryAccess.cpp)

add_library(hipaccASTFuse ${ASTFuse_SOURCES})
add_library(hipaccASTNode ${ASTNode_SOURCES})
add_library(hipaccASTTranslate ${ASTTranslate_SOURCES})
