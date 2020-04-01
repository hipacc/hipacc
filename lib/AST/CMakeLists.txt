set(ASTNode_SOURCES ASTNode.cpp)
set(ASTTranslate_SOURCES ASTClone.cpp ASTTranslate.cpp BorderHandling.cpp Convolution.cpp Interpolate.cpp MemoryAccess.cpp)

add_library(hipaccASTNode ${ASTNode_SOURCES})
add_library(hipaccASTTranslate ${ASTTranslate_SOURCES})

add_cppcheck(hipaccASTNode POSSIBLE_ERROR FAIL_ON_WARNINGS)
add_cppcheck(hipaccASTTranslate POSSIBLE_ERROR FAIL_ON_WARNINGS)
