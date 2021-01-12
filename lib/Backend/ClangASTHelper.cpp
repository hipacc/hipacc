//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//===--- ClangASTHelper.cpp - Implements helper class for easy clang AST handling. ---===//
//
// This file implements a helper class which contains a few methods for easy clang AST handling.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/BackendExceptions.h"
#include "hipacc/Backend/ClangASTHelper.h"

using namespace clang::hipacc::Backend;
using namespace clang;


unsigned int ClangASTHelper::CountNumberOfReferences(Stmt *pStatement, const std::string &crstrReferenceName)
{
  if (pStatement == nullptr)
  {
    return 0;
  }
  else if (isa<::clang::DeclRefExpr>(pStatement))
  {
    DeclRefExpr *pCurrentDeclRef = dyn_cast<DeclRefExpr>(pStatement);

    if (pCurrentDeclRef->getNameInfo().getAsString() == crstrReferenceName)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }
  else
  {
    unsigned int uiChildRefCount = 0;

    for (auto itChild = pStatement->child_begin(); itChild != pStatement->child_end(); itChild++)
    {
      uiChildRefCount += CountNumberOfReferences(*itChild, crstrReferenceName);
    }

    return uiChildRefCount;
  }
}


ArraySubscriptExpr* ClangASTHelper::CreateArraySubscriptExpression(Expr *pArrayRef, Expr *pIndexExpression, const QualType &crReturnType, bool bIsLValue)
{
  ExprValueKind  eValueKind = bIsLValue ? VK_LValue : VK_RValue;

  return new (GetASTContext()) ArraySubscriptExpr(pArrayRef, pIndexExpression, crReturnType, eValueKind, OK_Ordinary, SourceLocation());
}

BinaryOperator* ClangASTHelper::CreateBinaryOperator(Expr *pLhs, Expr *pRhs, BinaryOperatorKind eOperatorKind, const QualType &crReturnType)
{
  return ASTNode::createBinaryOperator(GetASTContext(), pLhs, pRhs, eOperatorKind, crReturnType);
}

BinaryOperator* ClangASTHelper::CreateBinaryOperatorComma(Expr *pLhs, Expr *pRhs)
{
  return CreateBinaryOperator(pLhs, pRhs, BO_Comma, pRhs->getType());
}

BinaryOperator* ClangASTHelper::CreateBinaryOperatorLessThan(Expr *pLhs, Expr *pRhs)
{
  return CreateBinaryOperator(pLhs, pRhs, BO_LT, GetASTContext().BoolTy);
}

CXXBoolLiteralExpr* ClangASTHelper::CreateBoolLiteral(bool bValue)
{
  return new (GetASTContext()) CXXBoolLiteralExpr(bValue, GetASTContext().BoolTy, SourceLocation());
}

BreakStmt* ClangASTHelper::CreateBreakStatement()
{
  return new (GetASTContext()) BreakStmt(SourceLocation());
}

CompoundStmt* ClangASTHelper::CreateCompoundStatement(Stmt *pStatement)
{
  StatementVectorType vecStatements;

  vecStatements.push_back(pStatement);

  return CreateCompoundStatement(vecStatements);
}

CompoundStmt* ClangASTHelper::CreateCompoundStatement(const StatementVectorType &crvecStatements)
{
  return ASTNode::createCompoundStmt(GetASTContext(), crvecStatements);
}

ConditionalOperator* ClangASTHelper::CreateConditionalOperator(Expr *pCondition, Expr *pThenExpr, Expr *pElseExpr, const QualType &crReturnType)
{
  return new (GetASTContext()) ConditionalOperator(pCondition, SourceLocation(), pThenExpr, SourceLocation(), pElseExpr, crReturnType, VK_RValue, OK_Ordinary);
}

ContinueStmt* ClangASTHelper::CreateContinueStatement()
{
  return new (GetASTContext()) ContinueStmt(SourceLocation());
}

DeclRefExpr* ClangASTHelper::CreateDeclarationReferenceExpression(ValueDecl *pValueDecl)
{
  return ASTNode::createDeclRefExpr(GetASTContext(), pValueDecl);
}

DeclStmt* ClangASTHelper::CreateDeclarationStatement(DeclRefExpr *pDeclRef)
{
  return CreateDeclarationStatement(pDeclRef->getDecl());
}

DeclStmt* ClangASTHelper::CreateDeclarationStatement(ValueDecl *pValueDecl)
{
  return ASTNode::createDeclStmt(GetASTContext(), pValueDecl);
}

CallExpr* ClangASTHelper::CreateFunctionCall(FunctionDecl *pFunctionDecl, const ExpressionVectorType &crvecArguments)
{
  return ASTNode::createFunctionCall(GetASTContext(), pFunctionDecl, crvecArguments);
}

FunctionDecl* ClangASTHelper::CreateFunctionDeclaration(std::string strFunctionName, const QualType &crReturnType, const StringVectorType &crvecArgumentNames, const QualTypeVectorType &crvecArgumentTypes)
{
  return ASTNode::createFunctionDecl( GetASTContext(), GetASTContext().getTranslationUnitDecl(), strFunctionName, crReturnType,
                                      ArrayRef<::clang::QualType>(crvecArgumentTypes), ArrayRef<std::string>(crvecArgumentNames) );
}

IfStmt* ClangASTHelper::CreateIfStatement(Expr *pCondition, Stmt *pThenBranch, Stmt *pElseBranch)
{
  return ASTNode::createIfStmt(GetASTContext(), pCondition, pThenBranch, pElseBranch, nullptr);
}

IfStmt* ClangASTHelper::CreateIfStatement(const ExpressionVectorType &crvecConditions, const StatementVectorType &crvecBranchBodies, Stmt *pDefaultBranch)
{
  const size_t cszBranchCount = crvecConditions.size();

  if (cszBranchCount != crvecBranchBodies.size())
  {
    throw RuntimeErrorException("The number of branch conditions must be equal to the number of branch bodies!");
  }

  switch (cszBranchCount)
  {
  case 0:   return CreateIfStatement( CreateBoolLiteral(true), pDefaultBranch, nullptr );
  case 1:   return CreateIfStatement( crvecConditions.front(), crvecBranchBodies.front(), pDefaultBranch );

  default:

    // Create branch statements
    VectorType<IfStmt*> vecIfStatements;
    for (size_t szIdx = static_cast<size_t>(0); szIdx < cszBranchCount; ++szIdx)
    {
      vecIfStatements.push_back( CreateIfStatement( crvecConditions[szIdx], crvecBranchBodies[szIdx], nullptr ) );
    }

    // Link branch statements to a multi-branch statement
    for (size_t szIdx = static_cast<size_t>(1); szIdx < cszBranchCount; ++szIdx)
    {
      vecIfStatements[szIdx - 1]->setElse( vecIfStatements[szIdx] );
    }

    // Link default branch
    vecIfStatements.back()->setElse( pDefaultBranch );

    // Return the root of the cascade
    return vecIfStatements.front();
  }
}

ImplicitCastExpr* ClangASTHelper::CreateImplicitCastExpression(Expr *pOperandExpression, const QualType &crReturnType, CastKind eCastKind, bool bIsLValue)
{
  ExprValueKind  eValueKind = bIsLValue ? VK_LValue : VK_RValue;

  return ASTNode::createImplicitCastExpr(GetASTContext(), crReturnType, eCastKind, pOperandExpression, nullptr, eValueKind);
}

InitListExpr* ClangASTHelper::CreateInitListExpression(const ExpressionVectorType &crvecExpressions)
{
  return new (GetASTContext()) InitListExpr( GetASTContext(), SourceLocation(), ArrayRef<Expr*>(crvecExpressions), SourceLocation() );
}

DoStmt* ClangASTHelper::CreateLoopDoWhile(::clang::Expr *pCondition, ::clang::Stmt *pBody)
{
  return new (GetASTContext()) DoStmt(pBody, pCondition, SourceLocation(), SourceLocation(), SourceLocation());
}

ForStmt* ClangASTHelper::CreateLoopFor(Expr *pCondition, Stmt *pBody, Stmt *pInitializer, Expr *pIncrement)
{
  return ASTNode::createForStmt(GetASTContext(), pInitializer, pCondition, pIncrement, pBody, nullptr);
}

WhileStmt* ClangASTHelper::CreateLoopWhile(Expr *pCondition, Stmt *pBody)
{
  return ASTNode::createWhileStmt(GetASTContext(), nullptr, pCondition, pBody);
}

Stmt* ClangASTHelper::CreateOpenMPDirectiveParallelFor(ForStmt* pLoop, int nChunkSize)
{
  ASTContext& Context = GetASTContext();
  std::vector<OMPClause*> vecClauses;

	if (nChunkSize > 1) {
		Expr* pChunkSize = ASTNode::createIntegerLiteral(Context, nChunkSize);

		// we need a valid SourceLocation
		SourceManager &SM = Context.getSourceManager();
		FileID mainFileID = SM.getMainFileID();
		SourceLocation validLoc = SM.getLocForStartOfFile(mainFileID);

		// create OMP schedule clause with chunk size
		OMPScheduleClause* pClause = new (Context) OMPScheduleClause(
				validLoc, validLoc, validLoc, validLoc, validLoc,
				OMPC_SCHEDULE_static, pChunkSize, nullptr,
				OMPC_SCHEDULE_MODIFIER_unknown, validLoc,
				OMPC_SCHEDULE_MODIFIER_unknown, validLoc);

		vecClauses.push_back(pClause);
    }

    // create captured encapsulating the loop statement internally
    std::vector<CapturedStmt::Capture> vecCaptures;
    std::vector<Expr*> vecCaptureInits;
    auto CD = CapturedDecl::Create(Context, 0, 0);
    auto RD = RecordDecl::CreateDeserialized(Context, 0);
    Stmt* pCaptured = CapturedStmt::Create(Context, pLoop, CR_OpenMP, vecCaptures, vecCaptureInits, CD, RD);

    return OMPParallelForDirective::Create(Context, SourceLocation(),
      SourceLocation(), 0, vecClauses, pCaptured, OMPLoopDirective::HelperExprs(),
      false);
}

ParenExpr* ClangASTHelper::CreateParenthesisExpression(Expr *pSubExpression)
{
  return ASTNode::createParenExpr(GetASTContext(), pSubExpression);
}

UnaryOperator* ClangASTHelper::CreatePostIncrementOperator(DeclRefExpr *pDeclRef)
{
  return CreateUnaryOperator(pDeclRef, UO_PostInc, pDeclRef->getType());
}

CXXReinterpretCastExpr* ClangASTHelper::CreateReinterpretCast(Expr *pOperandExpression, const QualType &crReturnType, CastKind eCastKind, bool bIsLValue)
{
  ExprValueKind  eValueKind = bIsLValue ? VK_LValue : VK_RValue;

  CXXCastPath CastPath;

  return CXXReinterpretCastExpr::Create(GetASTContext(), crReturnType, eValueKind, eCastKind, pOperandExpression, &CastPath, GetASTContext().getTrivialTypeSourceInfo(crReturnType), SourceLocation(), SourceLocation(), SourceRange());
}

ReturnStmt* ClangASTHelper::CreateReturnStatement(Expr *pReturnValue)
{
  return ReturnStmt::Create(GetASTContext(), SourceLocation(), pReturnValue, nullptr);
}

CXXStaticCastExpr* ClangASTHelper::CreateStaticCast(Expr *pOperandExpression, const QualType &crReturnType, CastKind eCastKind, bool bIsLValue)
{
  ExprValueKind  eValueKind = bIsLValue ? VK_LValue : VK_RValue;

  CXXCastPath CastPath;

  return CXXStaticCastExpr::Create(GetASTContext(), crReturnType, eValueKind, eCastKind, pOperandExpression, &CastPath, GetASTContext().getTrivialTypeSourceInfo(crReturnType), SourceLocation(), SourceLocation(), SourceRange());
}

StringLiteral* ClangASTHelper::CreateStringLiteral(std::string strValue)
{
  return StringLiteral::Create( GetASTContext(), llvm::StringRef(strValue), StringLiteral::Ascii, false, GetASTContext().getPointerType( GetASTContext().CharTy ), SourceLocation() );
}

UnaryOperator* ClangASTHelper::CreateUnaryOperator(Expr *pSubExpression, UnaryOperatorKind eOperatorKind, const QualType &crResultType)
{
  return ASTNode::createUnaryOperator(GetASTContext(), pSubExpression, eOperatorKind, crResultType);
}

VarDecl* ClangASTHelper::CreateVariableDeclaration(DeclContext *pDeclContext, const std::string &crstrVariableName, const QualType &crVariableType, Expr *pInitExpression)
{
  VarDecl *pVarDecl = ASTNode::createVarDecl(GetASTContext(), pDeclContext, crstrVariableName, crVariableType, pInitExpression);
  pDeclContext->addDecl(pVarDecl);

  return pVarDecl;
}

VarDecl* ClangASTHelper::CreateVariableDeclaration(FunctionDecl *pParentFunction, const std::string &crstrVariableName, const QualType &crVariableType, Expr *pInitExpression)
{
  return CreateVariableDeclaration(FunctionDecl::castToDeclContext(pParentFunction), crstrVariableName, crVariableType, pInitExpression);
}


QualType ClangASTHelper::GetConstantArrayType(const QualType &crElementType, const size_t cszDimension)
{
  return GetASTContext().getConstantArrayType( crElementType, llvm::APInt(32, static_cast<uint64_t>(cszDimension), false), nullptr, ArrayType::Normal, 0 );
}


bool ClangASTHelper::AreSignaturesEqual(const FunctionDeclarationVectorType &crvecFunctionDecls)
{
  if (crvecFunctionDecls.size() > 1)
  {
    FunctionDecl *pRefDecl = crvecFunctionDecls.front();

    for (auto itOtherDecl = crvecFunctionDecls.begin() + 1; itOtherDecl != crvecFunctionDecls.end(); ++itOtherDecl)
    {
      bool          bMismatch  = false;
      FunctionDecl *pOtherDecl = *itOtherDecl;

      if (pRefDecl->getReturnType() != pOtherDecl->getReturnType())
      {
        bMismatch = true;
      }
      else if (pRefDecl->getNumParams() != pOtherDecl->getNumParams())
      {
        bMismatch = true;
      }
      else
      {
        for (unsigned int uiParamIdx = pRefDecl->getNumParams(); uiParamIdx < pRefDecl->getNumParams(); ++uiParamIdx)
        {
          ParmVarDecl *pRefParam   = pRefDecl->getParamDecl(uiParamIdx);
          ParmVarDecl *pOtherParam = pOtherDecl->getParamDecl(uiParamIdx);

          if (pRefParam->getType() != pOtherParam->getType())
          {
            bMismatch = true;
            break;
          }
        }
      }

      if (bMismatch)
      {
        return false;
      }
    }
  }

  return true;
}

DeclRefExpr* ClangASTHelper::FindDeclaration(FunctionDecl *pFunction, const std::string &crstrDeclName)
{
  DeclContext *pDeclContext = FunctionDecl::castToDeclContext(pFunction);

  for (auto itDecl = pDeclContext->decls_begin(); itDecl != pDeclContext->decls_end(); itDecl++)
  {
    Decl *pDecl = *itDecl;

    if ((pDecl == nullptr) || (!isa<ValueDecl>(pDecl)))
    {
      continue;
    }

    ValueDecl* pValueDecl = dyn_cast<ValueDecl>(pDecl);

    if (pValueDecl->getNameAsString() == crstrDeclName)
    {
      return CreateDeclarationReferenceExpression(pValueDecl);
    }
  }

  return nullptr;
}

std::string ClangASTHelper::GetFullyQualifiedFunctionName(FunctionDecl *pFunctionDecl)
{
  std::string strFunctionName = pFunctionDecl->getNameAsString();

  // Unroll the namespaces
  DeclContext  *pNameSpaceDeclContext = pFunctionDecl->getEnclosingNamespaceContext();
  while (pNameSpaceDeclContext != nullptr)
  {
    // Check if we have a namespace declaration (there namespace aliases, too)
    if (! isa<NamespaceDecl>(pNameSpaceDeclContext))
    {
      break;
    }

    NamespaceDecl *pNameSpace = NamespaceDecl::castFromDeclContext(pNameSpaceDeclContext);

    strFunctionName = pNameSpace->getNameAsString() + std::string("::") + strFunctionName;

    // Get the parent namespace
    DeclContext *pParentNameSpaceDeclContext = pNameSpaceDeclContext->getParent();
    if (pParentNameSpaceDeclContext == pNameSpaceDeclContext)
    {
      // The global namespace is encapsulating itself (weird) => break as soon as a namespace is its own parent
      break;
    }

    pNameSpaceDeclContext = pParentNameSpaceDeclContext;
  }

  // Remove global namespace specifier if present
  if ((strFunctionName.size() > 2) && (strFunctionName.substr(0, 2) == "::"))
  {
    strFunctionName = strFunctionName.substr(2, std::string::npos);
  }

  return strFunctionName;
}

ClangASTHelper::FunctionDeclarationVectorType ClangASTHelper::GetKnownFunctionDeclarations()
{
  TranslationUnitDecl *pTranslationUnit = GetASTContext().getTranslationUnitDecl();

  return GetFunctionDeclarationsFromContext( pTranslationUnit->getEnclosingNamespaceContext() );
}

ClangASTHelper::FunctionDeclarationVectorType ClangASTHelper::GetFunctionDeclarationsFromContext(DeclContext *pDeclContextRoot)
{
  FunctionDeclarationVectorType vecFunctionDecls;

  for (auto itDecl = pDeclContextRoot->decls_begin(); itDecl != pDeclContextRoot->decls_end(); itDecl++)
  {
    Decl *pDecl = *itDecl;
    if (pDecl == nullptr)
    {
      continue;
    }

    if (isa<FunctionDecl>(pDecl))
    {
      vecFunctionDecls.push_back(dyn_cast<FunctionDecl>(pDecl));
    }
    else if (isa<CXXRecordDecl>(pDecl))
    {
      // We don't want to find class member functions at this point => Don't traverse through CXXRecordDecl objects
      continue;
    }
    else if (isa<DeclContext>(pDecl))
    {
      FunctionDeclarationVectorType vecNamespaceFunctionDecls = GetFunctionDeclarationsFromContext( dyn_cast<DeclContext>(pDecl) );

      vecFunctionDecls.insert( vecFunctionDecls.end(), vecNamespaceFunctionDecls.begin(), vecNamespaceFunctionDecls.end() );
    }
  }

  return vecFunctionDecls;
}

bool ClangASTHelper::IsSingleBranchStatement(Stmt *pStatement)
{
  if (pStatement == nullptr)                                      // Empty statements have no children
  {
    return true;
  }
  else if (pStatement->child_begin() == pStatement->child_end())  // No children
  {
    return true;
  }
  else                                                            // Statement has children
  {
    // Check if more than one child is present
    auto itChild = pStatement->child_begin();
    itChild++;

    if (itChild == pStatement->child_end())   // Only one child => Check if the child is a single branch
    {
      return IsSingleBranchStatement(*(pStatement->child_begin()));
    }
    else                                      // More than one child => Statement is not a single branch statement
    {
      return false;
    }
  }
}

void ClangASTHelper::ReplaceDeclarationReferences(Stmt* pStatement, const std::string &crstrDeclRefName, ValueDecl *pNewDecl)
{
  if (pStatement == nullptr)
  {
    return;
  }
  else if (isa<::clang::DeclRefExpr>(pStatement))
  {
    DeclRefExpr *pDeclRef = dyn_cast<DeclRefExpr>(pStatement);

    if (pDeclRef->getNameInfo().getName().getAsString() == crstrDeclRefName)
    {
      pDeclRef->setDecl(pNewDecl);
    }
  }
  else
  {
    for (auto itChild = pStatement->child_begin(); itChild != pStatement->child_end(); itChild++)
    {
      ReplaceDeclarationReferences(*itChild, crstrDeclRefName, pNewDecl);
    }
  }
}

bool ClangASTHelper::IsPointerToConstType(const QualType& crPointer)
{
  const Type* pType = crPointer.getTypePtrOrNull();

  if (pType != nullptr)
  {
    return pType->getPointeeType().isConstQualified();
  }
  else
  {
    return false;
  }
}


// vim: set ts=2 sw=2 sts=2 et ai:

