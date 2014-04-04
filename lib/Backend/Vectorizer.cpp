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

//===--- Vectorizer.cpp - Implements a vectorizing component for clang's syntax trees. -===//
//
// This file implements a vectorizing component for clang's syntax trees
//
//===-----------------------------------------------------------------------------------===//

#include "hipacc/Backend/Vectorizer.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <sstream>
using namespace clang::hipacc::Backend::Vectorization;
using namespace std;


void  Vectorizer::VASTBuilder::VariableNameTranslator::AddRenameEntry(string strOriginalName, string strNewName)
{
  if (_lstRenameStack.empty())
  {
    throw InternalErrorException("Rename stack is empty");
  }
  else if (strOriginalName == strNewName)
  {
    throw InternalErrorException("The original and new name cannot be identical!");
  }

  RenameMapType &rCurrentMap = _lstRenameStack.front();
  if (rCurrentMap.find(strOriginalName) != rCurrentMap.end())
  {
    throw InternalErrorException(string("The variable name \"") + strOriginalName + string("\" is already known at this layer!"));
  }

  rCurrentMap[strOriginalName] = strNewName;
}

string Vectorizer::VASTBuilder::VariableNameTranslator::TranslateName(string strOriginalName) const
{
  for (auto itMap = _lstRenameStack.begin(); itMap != _lstRenameStack.end(); itMap++)
  {
    auto itEntry = itMap->find(strOriginalName);

    if (itEntry != itMap->end())
    {
      return itEntry->second;
    }
  }

  return strOriginalName;
}



AST::Expressions::BinaryOperatorPtr Vectorizer::VASTBuilder::_BuildBinaryOperatorExpression(::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, ::clang::BinaryOperatorKind eOpKind)
{
  AST::Expressions::BinaryOperatorPtr spReturnOperator(nullptr);

  if (eOpKind == ::clang::BO_Assign)
  {
    spReturnOperator = AST::Expressions::AssignmentOperator::Create();
  }
  else if (::clang::BinaryOperator::isComparisonOp(eOpKind) || ::clang::BinaryOperator::isLogicalOp(eOpKind))
  {
    typedef AST::Expressions::RelationalOperator::RelationalOperatorType  OperatorType;
    OperatorType eOperatorType = OperatorType::Equal;

    switch (eOpKind)
    {
    case BO_EQ:     eOperatorType = OperatorType::Equal;          break;
    case BO_GT:     eOperatorType = OperatorType::Greater;        break;
    case BO_GE:     eOperatorType = OperatorType::GreaterEqual;   break;
    case BO_LT:     eOperatorType = OperatorType::Less;           break;
    case BO_LE:     eOperatorType = OperatorType::LessEqual;      break;
    case BO_LAnd:   eOperatorType = OperatorType::LogicalAnd;     break;
    case BO_LOr:    eOperatorType = OperatorType::LogicalOr;      break;
    case BO_NE:     eOperatorType = OperatorType::NotEqual;       break;
    default:        throw RuntimeErrorException("Invalid relational operator type!");
    }

    spReturnOperator = AST::Expressions::RelationalOperator::Create( eOperatorType );
  }
  else
  {
    typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType  OperatorType;
    OperatorType eOperatorType = OperatorType::Add;

    switch (eOpKind)
    {
    case BO_Add:    eOperatorType = OperatorType::Add;          break;
    case BO_And:    eOperatorType = OperatorType::BitwiseAnd;   break;
    case BO_Or:     eOperatorType = OperatorType::BitwiseOr;    break;
    case BO_Xor:    eOperatorType = OperatorType::BitwiseXOr;   break;
    case BO_Div:    eOperatorType = OperatorType::Divide;       break;
    case BO_Rem:    eOperatorType = OperatorType::Modulo;       break;
    case BO_Mul:    eOperatorType = OperatorType::Multiply;     break;
    case BO_Shl:    eOperatorType = OperatorType::ShiftLeft;    break;
    case BO_Shr:    eOperatorType = OperatorType::ShiftRight;   break;
    case BO_Sub:    eOperatorType = OperatorType::Subtract;     break;
    default:        throw RuntimeErrorException("Invalid arithmetic operator type!");
    }

    spReturnOperator = AST::Expressions::ArithmeticOperator::Create( eOperatorType );
  }


  spReturnOperator->SetLHS( _BuildExpression(pExprLHS) );
  spReturnOperator->SetRHS( _BuildExpression(pExprRHS) );

  return spReturnOperator;
}

void Vectorizer::VASTBuilder::_BuildBranchingStatement(::clang::IfStmt *pIfStmt, AST::ScopePtr spEnclosingScope)
{
  AST::ControlFlow::BranchingStatementPtr spBranchingStmt = AST::ControlFlow::BranchingStatement::Create();
  spEnclosingScope->AddChild(spBranchingStmt);

  // Unroll the "if-else"-cascade in the clang AST
  ::clang::Stmt *pCurrentStatement = pIfStmt;
  while (isa<::clang::IfStmt>(pCurrentStatement))
  {
    pCurrentStatement = _BuildConditionalBranch(dyn_cast<::clang::IfStmt>(pCurrentStatement), spBranchingStmt);
    if (pCurrentStatement == nullptr)
    {
      break;
    }
  }

  // Build default branch
  AST::ScopePtr spDefaultBranch = spBranchingStmt->GetDefaultBranch();
  if (pCurrentStatement != nullptr)
  {
    if (isa<::clang::CompoundStmt>(pCurrentStatement))
    {
      _ConvertScope(spDefaultBranch, dyn_cast<::clang::CompoundStmt>(pCurrentStatement));
    }
    else
    {
      AST::BaseClasses::NodePtr spChild = _BuildStatement(pCurrentStatement, spDefaultBranch);
      if (spChild)
      {
        spDefaultBranch->AddChild(spChild);
      }
    }
  }
}

::clang::Stmt* Vectorizer::VASTBuilder::_BuildConditionalBranch(::clang::IfStmt *pIfStmt, AST::ControlFlow::BranchingStatementPtr spBranchingStatement)
{
  AST::ControlFlow::ConditionalBranchPtr spBranch = AST::ControlFlow::ConditionalBranch::Create( _BuildExpression(pIfStmt->getCond()) );
  spBranchingStatement->AddConditionalBranch(spBranch);

  AST::ScopePtr spBranchBody  = spBranch->GetBody();
  ::clang::Stmt *pIfBody      = pIfStmt->getThen();
  if (pIfBody != nullptr)
  {
    if (isa<::clang::CompoundStmt>(pIfBody))
    {
      _ConvertScope(spBranchBody, dyn_cast<::clang::CompoundStmt>(pIfBody));
    }
    else
    {
      AST::BaseClasses::NodePtr spChild = _BuildStatement(pIfBody, spBranchBody);
      if (spChild)
      {
        spBranchBody->AddChild(spChild);
      }
    }
  }

  return pIfStmt->getElse();
}

AST::Expressions::ConstantPtr Vectorizer::VASTBuilder::_BuildConstantExpression(::clang::Expr *pExpression)
{
  if (isa<::clang::IntegerLiteral>(pExpression))
  {
    ::clang::IntegerLiteral *pIntLiteral  = dyn_cast<::clang::IntegerLiteral>(pExpression);
    llvm::APInt             llvmIntValue  = pIntLiteral->getValue();

    bool          bSigned     = pIntLiteral->getType()->isSignedIntegerType();
    unsigned int  uiBitWidth  = llvmIntValue.getBitWidth();

    uint64_t ui64Value = *llvmIntValue.getRawData();

    if (uiBitWidth <= 8)
    {
      if (bSigned)  return AST::Expressions::Constant::Create( static_cast<int8_t >(ui64Value) );
      else          return AST::Expressions::Constant::Create( static_cast<uint8_t>(ui64Value) );
    }
    else if (uiBitWidth <= 16)
    {
      if (bSigned)  return AST::Expressions::Constant::Create( static_cast<int16_t >(ui64Value) );
      else          return AST::Expressions::Constant::Create( static_cast<uint16_t>(ui64Value) );
    }
    else if (uiBitWidth <= 32)
    {
      if (bSigned)  return AST::Expressions::Constant::Create( static_cast<int32_t >(ui64Value) );
      else          return AST::Expressions::Constant::Create( static_cast<uint32_t>(ui64Value) );
    }
    else
    {
      if (bSigned)  return AST::Expressions::Constant::Create( static_cast<int64_t >(ui64Value) );
      else          return AST::Expressions::Constant::Create( static_cast<uint64_t>(ui64Value) );
    }
  }
  else if (isa<::clang::FloatingLiteral>(pExpression))
  {
    llvm::APFloat llvmFloatValue = dyn_cast<::clang::FloatingLiteral>(pExpression)->getValue();

    if ( (llvm::APFloat::semanticsPrecision(llvmFloatValue.getSemantics()) == llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEhalf)) ||
         (llvm::APFloat::semanticsPrecision(llvmFloatValue.getSemantics()) == llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEsingle)) )
    {
      return AST::Expressions::Constant::Create( llvmFloatValue.convertToFloat() );
    }
    else
    {
      return AST::Expressions::Constant::Create( llvmFloatValue.convertToDouble() );
    }
  }
  else if (isa<::clang::CXXBoolLiteralExpr>(pExpression))
  {
    return AST::Expressions::Constant::Create( dyn_cast<::clang::CXXBoolLiteralExpr>(pExpression)->getValue() );
  }
  else
  {
    throw InternalErrorException("Unknown literal expression!");
  }
}

AST::Expressions::ConversionPtr Vectorizer::VASTBuilder::_BuildConversionExpression(::clang::CastExpr *pCastExpr)
{
  return AST::Expressions::Conversion::Create( _ConvertTypeInfo(pCastExpr->getType()), _BuildExpression(pCastExpr->getSubExpr()), (! isa<::clang::ImplicitCastExpr>(pCastExpr)) );
}

AST::BaseClasses::ExpressionPtr Vectorizer::VASTBuilder::_BuildExpression(::clang::Expr *pExpression)
{
  AST::BaseClasses::ExpressionPtr spReturnExpression(nullptr);

  if (isa<::clang::IntegerLiteral>(pExpression) || isa<::clang::FloatingLiteral>(pExpression) || isa<::clang::CXXBoolLiteralExpr>(pExpression))
  {
    spReturnExpression = _BuildConstantExpression(pExpression);
  }
  else if (isa<::clang::DeclRefExpr>(pExpression))
  {
    spReturnExpression = _BuildIdentifier( dyn_cast< ::clang::DeclRefExpr >(pExpression)->getNameInfo().getAsString() );
  }
  else if (isa<::clang::CompoundAssignOperator>(pExpression))
  {
    ::clang::CompoundAssignOperator *pCompoundAssignment  = dyn_cast<::clang::CompoundAssignOperator>(pExpression);
    ::clang::Expr                   *pExprLHS             = pCompoundAssignment->getLHS();
    ::clang::Expr                   *pExprRHS             = pCompoundAssignment->getRHS();
    ::clang::BinaryOperatorKind     eOpKind               = pCompoundAssignment->getOpcode();

    switch (eOpKind)
    {
    case BO_AddAssign:  eOpKind = BO_Add;   break;
    case BO_AndAssign:  eOpKind = BO_And;   break;
    case BO_DivAssign:  eOpKind = BO_Div;   break;
    case BO_MulAssign:  eOpKind = BO_Mul;   break;
    case BO_OrAssign:   eOpKind = BO_Or;    break;
    case BO_RemAssign:  eOpKind = BO_Rem;   break;
    case BO_ShlAssign:  eOpKind = BO_Shl;   break;
    case BO_ShrAssign:  eOpKind = BO_Shr;   break;
    case BO_SubAssign:  eOpKind = BO_Sub;   break;
    case BO_XorAssign:  eOpKind = BO_Xor;   break;
    }

    spReturnExpression = AST::Expressions::AssignmentOperator::Create( _BuildExpression(pExprLHS), _BuildBinaryOperatorExpression(pExprLHS, pExprRHS, eOpKind) );
  }
  else if (isa<::clang::BinaryOperator>(pExpression))
  {
    ::clang::BinaryOperator *pBinOp = dyn_cast<::clang::BinaryOperator>(pExpression);

    spReturnExpression = _BuildBinaryOperatorExpression( pBinOp->getLHS(), pBinOp->getRHS(), pBinOp->getOpcode() );
  }
  else if (isa<::clang::CastExpr>(pExpression))
  {
    spReturnExpression = _BuildConversionExpression(dyn_cast<::clang::CastExpr>(pExpression));
  }
  else if (isa<::clang::ParenExpr>(pExpression))
  {
    spReturnExpression = AST::Expressions::Parenthesis::Create( _BuildExpression(dyn_cast<::clang::ParenExpr>(pExpression)->getSubExpr()) );
  }
  else if (isa<::clang::ArraySubscriptExpr>(pExpression))
  {
    ::clang::ArraySubscriptExpr *pArraySubscript = dyn_cast<::clang::ArraySubscriptExpr>(pExpression);

    spReturnExpression = AST::Expressions::MemoryAccess::Create( _BuildExpression(pArraySubscript->getLHS()), _BuildExpression(pArraySubscript->getRHS()) );
  }
  else if (isa<::clang::UnaryOperator>(pExpression))
  {
    ::clang::UnaryOperator      *pUnaryOp = dyn_cast<::clang::UnaryOperator>(pExpression);
    ::clang::Expr               *pSubExpr = pUnaryOp->getSubExpr();
    ::clang::UnaryOperatorKind  eOpCode   = pUnaryOp->getOpcode();

    if (eOpCode == ::clang::UO_Deref)
    {
      spReturnExpression = AST::Expressions::MemoryAccess::Create( _BuildExpression(pSubExpr), AST::Expressions::Constant::Create< int32_t >( 0 ) );
    }
    else
    {
      spReturnExpression = _BuildUnaryOperatorExpression(pSubExpr, eOpCode);
    }
  }
  else if (isa<::clang::CallExpr>(pExpression))
  {
    ::clang::CallExpr *pCallExpr  = dyn_cast<::clang::CallExpr>(pExpression);

    // Build the function call node
    AST::Expressions::FunctionCallPtr spFunctionCall(nullptr);
    {
      // Get the function name
      ::clang::FunctionDecl *pCalleeDecl = pCallExpr->getDirectCallee();
      if (pCalleeDecl == nullptr)
      {
        throw InternalErrors::NullPointerException("pCalleeDecl");
      }

      std::string strFunctionName = ClangASTHelper::GetFullyQualifiedFunctionName( pCalleeDecl );

      // Convert the return type
      AST::BaseClasses::TypeInfo  ReturnType = _ConvertTypeInfo( pCallExpr->getCallReturnType() );
      if (ReturnType.IsSingleValue())
      {
        ReturnType.SetConst(true);
      }

      spFunctionCall = AST::Expressions::FunctionCall::Create( strFunctionName, ReturnType );
      spReturnExpression = spFunctionCall;
    }


    // Build the call parameter expressions
    for (unsigned int i = 0; i < pCallExpr->getNumArgs(); ++i)
    {
      ::clang::Expr *pArg = pCallExpr->getArg(i);
      if (pArg == nullptr)
      {
        throw InternalErrors::NullPointerException("pArg");
      }

      spFunctionCall->AddCallParameter( _BuildExpression(pArg) );
    }
  }
  else
  {
    throw ASTExceptions::UnknownExpressionClass( pExpression->getStmtClassName() );
  }

  return spReturnExpression;
}

AST::Expressions::IdentifierPtr Vectorizer::VASTBuilder::_BuildIdentifier(string strIdentifierName)
{
  return AST::Expressions::Identifier::Create( _VarTranslator.TranslateName(strIdentifierName) );
}

void Vectorizer::VASTBuilder::_BuildLoop(::clang::Stmt *pLoopStatement, AST::ScopePtr spEnclosingScope)
{
  AST::ControlFlow::LoopPtr spLoop = AST::ControlFlow::Loop::Create();

  ::clang::Stmt   *pLoopBody  = nullptr;
  ::clang::Expr   *pCondition = nullptr;

  if (isa<::clang::ForStmt>(pLoopStatement))
  {
    spLoop->SetLoopType(AST::ControlFlow::Loop::LoopType::TopControlled);

    ::clang::ForStmt *pForLoop = dyn_cast<::clang::ForStmt>(pLoopStatement);

    // If we have an init statement, create a container scope around the loop and add the init statement
    if (pForLoop->getInit())
    {
      AST::ScopePtr spLoopHolderScope = AST::Scope::Create();
      spEnclosingScope->AddChild(spLoopHolderScope);

      AST::BaseClasses::NodePtr spInitStatement = _BuildStatement(pForLoop->getInit(), spLoopHolderScope);
      if (spInitStatement)
      {
        spLoopHolderScope->AddChild(spInitStatement);
      }

      spLoopHolderScope->AddChild(spLoop);
    }
    else
    {
      spEnclosingScope->AddChild(spLoop);
    }

    // Build increment expression if it is present
    if (pForLoop->getInc())
    {
      spLoop->SetIncrement( _BuildExpression(pForLoop->getInc()) );
    }

    pCondition  = pForLoop->getCond();
    pLoopBody   = pForLoop->getBody();
  }
  else
  {
    spEnclosingScope->AddChild(spLoop);

    if (isa<::clang::DoStmt>(pLoopStatement))
    {
      spLoop->SetLoopType(AST::ControlFlow::Loop::LoopType::BottomControlled);

      ::clang::DoStmt *pDoWhileLoop = dyn_cast<::clang::DoStmt>(pLoopStatement);
      pCondition  = pDoWhileLoop->getCond();
      pLoopBody   = pDoWhileLoop->getBody();
    }
    else if (isa<::clang::WhileStmt>(pLoopStatement))
    {
      spLoop->SetLoopType(AST::ControlFlow::Loop::LoopType::TopControlled);

      ::clang::WhileStmt *pWhileLoop = dyn_cast<::clang::WhileStmt>(pLoopStatement);
      pCondition  = pWhileLoop->getCond();
      pLoopBody   = pWhileLoop->getBody();
    }
    else
    {
      throw ASTExceptions::UnknownStatementClass(pLoopStatement->getStmtClassName());
    }
  }


  // Build condition expression
  spLoop->SetCondition( _BuildExpression(pCondition) );

  // Build loop body if it is present
  if (pLoopBody != nullptr)
  {
    AST::ScopePtr spLoopBody = spLoop->GetBody();

    if (isa<::clang::CompoundStmt>(pLoopBody))
    {
      _ConvertScope(spLoopBody, dyn_cast<::clang::CompoundStmt>(pLoopBody));
    }
    else
    {
      AST::BaseClasses::NodePtr spChild = _BuildStatement(pLoopBody, spLoopBody);
      if (spChild)
      {
        spLoopBody->AddChild(spChild);
      }
    }
  }
}

AST::BaseClasses::NodePtr Vectorizer::VASTBuilder::_BuildStatement(::clang::Stmt *pStatement, AST::ScopePtr spEnclosingScope)
{
  AST::BaseClasses::NodePtr spStatement(nullptr);

  if (isa<::clang::CompoundStmt>(pStatement))
  {
    ::clang::CompoundStmt *pCurrentCompound = dyn_cast<::clang::CompoundStmt>(pStatement);

    AST::ScopePtr spChildScope = AST::Scope::Create();
    spEnclosingScope->AddChild( spChildScope );

    _ConvertScope(spChildScope, pCurrentCompound);

    spStatement = nullptr;
  }
  else if (isa<::clang::DeclStmt>(pStatement))
  {
    ::clang::DeclStmt     *pDeclStatement = dyn_cast<::clang::DeclStmt>(pStatement);
    ::clang::DeclGroupRef  DeclGroup      = pDeclStatement->getDeclGroup();

    for (auto itDecl = DeclGroup.begin(); itDecl != DeclGroup.end(); itDecl++)
    {
      ::clang::Decl *pDecl = *itDecl;
      if (pDecl == nullptr)
      {
        continue;
      }
      else if (! isa<::clang::VarDecl>(pDecl))
      {
        continue;
      }

      ::clang::VarDecl  *pVarDecl = dyn_cast<::clang::VarDecl>(pDecl);
      spEnclosingScope->AddVariableDeclaration( _BuildVariableInfo(pVarDecl, spEnclosingScope) );

      ::clang::Expr     *pInitExpr = pVarDecl->getInit();
      if (pInitExpr == nullptr)
      {
        continue;
      }

      spEnclosingScope->AddChild( AST::Expressions::AssignmentOperator::Create( _BuildIdentifier(pVarDecl->getNameAsString()), _BuildExpression(pInitExpr) ) );
    }

    spStatement = nullptr;
  }
  else if (isa<::clang::Expr>(pStatement))
  {
    AST::BaseClasses::ExpressionPtr spExpression = _BuildExpression(dyn_cast<::clang::Expr>(pStatement));
    if (!spExpression)
    {
      throw InternalErrors::NullPointerException("spStatement");
    }

    spStatement = spExpression;
  }
  else if ( isa<::clang::DoStmt>(pStatement) || isa<::clang::ForStmt>(pStatement) || isa<::clang::WhileStmt>(pStatement) )
  {
    _BuildLoop(pStatement, spEnclosingScope);
  }
  else if (isa<::clang::IfStmt>(pStatement))
  {
    _BuildBranchingStatement(dyn_cast<::clang::IfStmt>(pStatement), spEnclosingScope);
  }
  else if (isa<::clang::BreakStmt>(pStatement))
  {
    spStatement = AST::ControlFlow::LoopControlStatement::Create( AST::ControlFlow::LoopControlStatement::LoopControlType::Break );
  }
  else if (isa<::clang::ContinueStmt>(pStatement))
  {
    spStatement = AST::ControlFlow::LoopControlStatement::Create( AST::ControlFlow::LoopControlStatement::LoopControlType::Continue );
  }
  else if (isa<::clang::ReturnStmt>(pStatement))
  {
    ::clang::ReturnStmt *pRetStmt = dyn_cast<::clang::ReturnStmt>(pStatement);

    if (pRetStmt->getRetValue() != nullptr)
    {
      throw InternalErrorException("Currently only return statements without a return value allowed!");
    }

    spStatement = AST::ControlFlow::ReturnStatement::Create();
  }
  else
  {
    throw ASTExceptions::UnknownStatementClass(pStatement->getStmtClassName());
  }

  return spStatement;
}

AST::Expressions::UnaryOperatorPtr Vectorizer::VASTBuilder::_BuildUnaryOperatorExpression(::clang::Expr *pSubExpr, ::clang::UnaryOperatorKind eOpKind)
{
  typedef AST::Expressions::UnaryOperator::UnaryOperatorType OperatorType;
  OperatorType eOperatorType = OperatorType::AddressOf;

  switch (eOpKind)
  {
  case UO_AddrOf:   eOperatorType = OperatorType::AddressOf;      break;
  case UO_Not:      eOperatorType = OperatorType::BitwiseNot;     break;
  case UO_LNot:     eOperatorType = OperatorType::LogicalNot;     break;
  case UO_Minus:    eOperatorType = OperatorType::Minus;          break;
  case UO_Plus:     eOperatorType = OperatorType::Plus;           break;
  case UO_PostDec:  eOperatorType = OperatorType::PostDecrement;  break;
  case UO_PostInc:  eOperatorType = OperatorType::PostIncrement;  break;
  case UO_PreDec:   eOperatorType = OperatorType::PreDecrement;   break;
  case UO_PreInc:   eOperatorType = OperatorType::PreIncrement;   break;
  default:          throw RuntimeErrorException("Invalid unary operator type!");
  }

  return AST::Expressions::UnaryOperator::Create( eOperatorType, _BuildExpression(pSubExpr) );
}

AST::BaseClasses::VariableInfoPtr Vectorizer::VASTBuilder::_BuildVariableInfo(::clang::VarDecl *pVarDecl, AST::IVariableContainerPtr spVariableContainer)
{
  string strVariableName = pVarDecl->getNameAsString();

  if ( spVariableContainer->IsVariableUsed(strVariableName) )
  {
    string strNewName = GetNextFreeVariableName( spVariableContainer, strVariableName );

    _VarTranslator.AddRenameEntry(strVariableName, strNewName);

    strVariableName = strNewName;
  }

  return AST::BaseClasses::VariableInfo::Create( strVariableName, _ConvertTypeInfo(pVarDecl->getType()), false );
}

void Vectorizer::VASTBuilder::_ConvertScope(AST::ScopePtr spScope, ::clang::CompoundStmt *pCompoundStatement)
{
  _VarTranslator.AddLayer();

  for (auto itChild = pCompoundStatement->child_begin(); itChild != pCompoundStatement->child_end(); itChild++)
  {
    ::clang::Stmt *pChildStatement = *itChild;
    if (pChildStatement == nullptr)
    {
      continue;
    }

    AST::BaseClasses::NodePtr spChild = _BuildStatement(pChildStatement, spScope);
    if (spChild)
    {
      spScope->AddChild(spChild);
    }
  }

  _VarTranslator.PopLayer();
}

void Vectorizer::VASTBuilder::_ConvertTypeInfo(AST::BaseClasses::TypeInfo &rTypeInfo, ::clang::QualType qtSourceType)
{
  while (qtSourceType->isArrayType())
  {
    const ::clang::ArrayType *pArrayType = qtSourceType->getAsArrayTypeUnsafe();

    if (pArrayType->isConstantArrayType())
    {
      const ::clang::ConstantArrayType *pConstArrayType = dyn_cast<::clang::ConstantArrayType>(pArrayType);

      rTypeInfo.GetArrayDimensions().push_back( static_cast< size_t >( *(pConstArrayType->getSize().getRawData()) ) );
    }
    else
    {
      throw RuntimeErrorException("Only constant size array types allowed!");
    }

    qtSourceType = pArrayType->getElementType();
  }

  if (qtSourceType->isPointerType())
  {
    rTypeInfo.SetPointer(true);
    qtSourceType = qtSourceType->getPointeeType();

    if (qtSourceType->isPointerType())
    {
      throw RuntimeErrorException("Only one level of indirection is allowed for pointer types!");
    }
  }
  else
  {
    rTypeInfo.SetPointer(false);
  }

  rTypeInfo.SetConst(qtSourceType.isConstQualified());

  if (qtSourceType->isScalarType())
  {
    qtSourceType = qtSourceType->getCanonicalTypeInternal();

    if (qtSourceType->isBuiltinType())
    {
      typedef ::clang::BuiltinType                    ClangTypes;
      typedef AST::BaseClasses::TypeInfo::KnownTypes  KnownTypes;

      const ::clang::BuiltinType *pBuiltInType = qtSourceType->getAs<::clang::BuiltinType>();

      KnownTypes eType;

      switch (pBuiltInType->getKind())
      {
      case ClangTypes::Bool:                              eType = KnownTypes::Bool;     break;
      case ClangTypes::Char_S: case ClangTypes::SChar:    eType = KnownTypes::Int8;     break;
      case ClangTypes::Char_U: case ClangTypes::UChar:    eType = KnownTypes::UInt8;    break;
      case ClangTypes::Short:                             eType = KnownTypes::Int16;    break;
      case ClangTypes::UShort:                            eType = KnownTypes::UInt16;   break;
      case ClangTypes::Int:                               eType = KnownTypes::Int32;    break;
      case ClangTypes::UInt:                              eType = KnownTypes::UInt32;   break;
      case ClangTypes::Long:                              eType = KnownTypes::Int64;    break;
      case ClangTypes::ULong:                             eType = KnownTypes::UInt64;   break;
      case ClangTypes::Float:                             eType = KnownTypes::Float;    break;
      case ClangTypes::Double:                            eType = KnownTypes::Double;   break;
      default:                                            throw RuntimeErrorException("Unsupported built-in type detected!");
      }

      rTypeInfo.SetType(eType);
    }
    else
    {
      throw RuntimeErrorException("Expected a built-in type!");
    }
  }
  else
  {
    throw RuntimeErrorException("Only scalar types, pointers to scalar types, or arrays of scalar or pointers to scalar types allowed!");
  }
}


AST::FunctionDeclarationPtr Vectorizer::VASTBuilder::BuildFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration)
{
  AST::FunctionDeclarationPtr spFunctionDecl = AST::FunctionDeclaration::Create( pFunctionDeclaration->getName() );

  for (size_t i = 0; i < pFunctionDeclaration->getNumParams(); ++i)
  {
    AST::BaseClasses::VariableInfoPtr spVariable = _BuildVariableInfo( pFunctionDeclaration->getParamDecl(i), spFunctionDecl );

    spFunctionDecl->AddParameter(spVariable);
  }

  ::clang::Stmt* pBody = pFunctionDeclaration->getBody();

  if ((pBody == nullptr) || (!isa<::clang::CompoundStmt>(pBody)))
  {
    throw RuntimeErrorException("Invalid function body");
  }

  _ConvertScope(spFunctionDecl->GetBody(), dyn_cast<::clang::CompoundStmt>(pBody));

  return spFunctionDecl;
}


string Vectorizer::VASTBuilder::GetNextFreeVariableName(AST::IVariableContainerPtr spVariableContainer, string strRootName)
{
  if (!spVariableContainer)
  {
    throw InternalErrors::NullPointerException("spVariableContainer");
  }
  else if (!spVariableContainer->IsVariableUsed(strRootName))
  {
    return strRootName;
  }

  for (int iVarSuffix = 0; true; ++iVarSuffix)
  {
    stringstream VarNameStream;

    VarNameStream << strRootName << "_" << iVarSuffix;

    string strCurrentName = VarNameStream.str();

    if (!spVariableContainer->IsVariableUsed(strCurrentName))
    {
      return strCurrentName;
    }
  }
}


// Implementation of class Vectorizer::VASTExporterBase
Vectorizer::VASTExporterBase::VASTExporterBase(::clang::ASTContext &rAstContext) : _ASTHelper(rAstContext), _pDeclContext(nullptr)
{
  // Parse known function declarations
  FunctionDeclVectorType vecFunctionDecls = _GetASTHelper().GetKnownFunctionDeclarations();

  for (auto itFunctionDecl : vecFunctionDecls)
  {
    _AddKnownFunctionDeclaration( itFunctionDecl );
  }
}

void Vectorizer::VASTExporterBase::_AddKnownFunctionDeclaration(::clang::FunctionDecl *pFunctionDecl)
{
  string strFunctionName = ClangASTHelper::GetFullyQualifiedFunctionName( pFunctionDecl );

  _mapKnownFunctions[ strFunctionName ][ pFunctionDecl->getNumParams() ].push_back( pFunctionDecl );
}

::clang::Expr* Vectorizer::VASTExporterBase::_BuildConstant(AST::Expressions::ConstantPtr spConstant)
{
  if (! spConstant)
  {
    throw InternalErrors::NullPointerException("spConstant");
  }

  typedef AST::BaseClasses::TypeInfo::KnownTypes  KnownTypes;

  switch (spConstant->GetValueType())
  {
  case KnownTypes::Bool:      return _GetASTHelper().CreateLiteral( spConstant->GetValue< bool     >() );
  case KnownTypes::Int8:      return _GetASTHelper().CreateLiteral( spConstant->GetValue< int8_t   >() );
  case KnownTypes::UInt8:     return _GetASTHelper().CreateLiteral( spConstant->GetValue< uint8_t  >() );
  case KnownTypes::Int16:     return _GetASTHelper().CreateLiteral( spConstant->GetValue< int16_t  >() );
  case KnownTypes::UInt16:    return _GetASTHelper().CreateLiteral( spConstant->GetValue< uint16_t >() );
  case KnownTypes::Int32:     return _GetASTHelper().CreateLiteral( spConstant->GetValue< int32_t  >() );
  case KnownTypes::UInt32:    return _GetASTHelper().CreateLiteral( spConstant->GetValue< uint32_t >() );
  case KnownTypes::Int64:     return _GetASTHelper().CreateLiteral( spConstant->GetValue< int64_t  >() );
  case KnownTypes::UInt64:    return _GetASTHelper().CreateLiteral( spConstant->GetValue< uint64_t >() );
  case KnownTypes::Float:     return _GetASTHelper().CreateLiteral( spConstant->GetValue< float    >() );
  case KnownTypes::Double:    return _GetASTHelper().CreateLiteral( spConstant->GetValue< double   >() );
  case KnownTypes::Unknown:   throw RuntimeErrorException("VAST constant type is unknown!");
  default:                    throw InternalErrorException("Unsupported VAST constant type detected!");
  }
}

::clang::FunctionDecl* Vectorizer::VASTExporterBase::_BuildFunctionDeclaration(AST::FunctionDeclarationPtr spFunction)
{
  if (! spFunction)
  {
    throw InternalErrors::NullPointerException("spFunction");
  }

  string strFunctionName = spFunction->GetName();

  // Convert the function parameters
  ClangASTHelper::StringVectorType    vecArgumentNames;
  ClangASTHelper::QualTypeVectorType  vecArgumentTypes;

  for (IndexType iParamIdx = static_cast<IndexType>(0); iParamIdx < spFunction->GetParameterCount(); ++iParamIdx)
  {
    AST::Expressions::IdentifierPtr spParameter = spFunction->GetParameter(iParamIdx);
    vecArgumentNames.push_back( spParameter->GetName() );

    vecArgumentTypes.push_back( _GetVariableType( spParameter->LookupVariableInfo() ) );
  }

  // Build the function declaration and add it to the map (required for recursive calls)
  ::clang::FunctionDecl *pFunctionDecl = _GetASTHelper().CreateFunctionDeclaration( strFunctionName, _GetASTContext().VoidTy, vecArgumentNames, vecArgumentTypes );
  _AddKnownFunctionDeclaration( pFunctionDecl );

  _pDeclContext = ::clang::FunctionDecl::castToDeclContext(pFunctionDecl);


  // Add all parameters to the map of known value declarations
  for (unsigned int uiParamIdx = 0; uiParamIdx < pFunctionDecl->getNumParams(); ++uiParamIdx)
  {
    ::clang::ParmVarDecl *pParam = pFunctionDecl->getParamDecl(uiParamIdx);

    _mapKnownDeclarations[ pParam->getNameAsString() ] = pParam;
  }

  return pFunctionDecl;
}

::clang::Stmt* Vectorizer::VASTExporterBase::_BuildLoop(AST::ControlFlow::Loop::LoopType eLoopType, ::clang::Expr *pCondition, ::clang::Stmt *pBody, ::clang::Expr *pIncrement)
{
  typedef AST::ControlFlow::Loop::LoopType  LoopType;

  if (eLoopType == LoopType::TopControlled)
  {
    if (pIncrement == nullptr)
    {
      return _GetASTHelper().CreateLoopWhile( pCondition, pBody );
    }
    else
    {
      return _GetASTHelper().CreateLoopFor( pCondition, pBody, nullptr, pIncrement );
    }
  }
  else if (eLoopType == LoopType::BottomControlled)
  {
    if (pIncrement != nullptr)
    {
      pIncrement = _CreateParenthesis( pIncrement );
      pCondition = _CreateParenthesis( pCondition );
      pCondition = _GetASTHelper().CreateBinaryOperatorComma( pIncrement, pCondition );
    }

    return _GetASTHelper().CreateLoopDoWhile( pCondition, pBody );
  }
  else
  {
    throw InternalErrorException("Unsupported VAST loop type detected!");
  }
}

::clang::Stmt* Vectorizer::VASTExporterBase::_BuildLoopControlStatement(AST::ControlFlow::LoopControlStatementPtr spLoopControl)
{
  typedef AST::ControlFlow::LoopControlStatement::LoopControlType   LoopControlType;

  if (spLoopControl->IsVectorized())
  {
    throw InternalErrorException("Cannot handle vectorized loop control statements!");
  }

  switch ( spLoopControl->GetControlType() )
  {
  case LoopControlType::Break:      return _GetASTHelper().CreateBreakStatement();
  case LoopControlType::Continue:   return _GetASTHelper().CreateContinueStatement();
  default:                          throw InternalErrorException("Unknown VAST loop control statement detected!");
  }
}

::clang::ValueDecl* Vectorizer::VASTExporterBase::_BuildValueDeclaration(AST::Expressions::IdentifierPtr spIdentifier, ::clang::Expr *pInitExpression)
{
  if (_pDeclContext == nullptr)
  {
    throw InternalErrorException( "The declaration context is not set! Run \"\" first." );
  }

  string strVariableName = spIdentifier->GetName();

  ::clang::ValueDecl *pVarDecl = _GetASTHelper().CreateVariableDeclaration( _pDeclContext, strVariableName, _GetVariableType( spIdentifier->LookupVariableInfo() ), pInitExpression );

  _mapKnownDeclarations[ strVariableName ] = pVarDecl;

  return pVarDecl;
}


::clang::BinaryOperatorKind Vectorizer::VASTExporterBase::_ConvertArithmeticOperatorType(AST::Expressions::ArithmeticOperator::ArithmeticOperatorType eOpType)
{
  typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType  OperatorType;

  switch (eOpType)
  {
  case OperatorType::Add:         return ::clang::BO_Add;
  case OperatorType::BitwiseAnd:  return ::clang::BO_And;
  case OperatorType::BitwiseOr:   return ::clang::BO_Or;
  case OperatorType::BitwiseXOr:  return ::clang::BO_Xor;
  case OperatorType::Divide:      return ::clang::BO_Div;
  case OperatorType::Modulo:      return ::clang::BO_Rem;
  case OperatorType::Multiply:    return ::clang::BO_Mul;
  case OperatorType::ShiftLeft:   return ::clang::BO_Shl;
  case OperatorType::ShiftRight:  return ::clang::BO_Shr;
  case OperatorType::Subtract:    return ::clang::BO_Sub;
  default:                        throw InternalErrorException("Unknown VAST arithmetic operator type detected!");
  }
}

::clang::BinaryOperatorKind Vectorizer::VASTExporterBase::_ConvertRelationalOperatorType(AST::Expressions::RelationalOperator::RelationalOperatorType eOpType)
{
  typedef AST::Expressions::RelationalOperator::RelationalOperatorType  OperatorType;

  switch (eOpType)
  {
  case OperatorType::Equal:         return ::clang::BO_EQ;
  case OperatorType::Greater:       return ::clang::BO_GT;
  case OperatorType::GreaterEqual:  return ::clang::BO_GE;
  case OperatorType::Less:          return ::clang::BO_LT;
  case OperatorType::LessEqual:     return ::clang::BO_LE;
  case OperatorType::LogicalAnd:    return ::clang::BO_LAnd;
  case OperatorType::LogicalOr:     return ::clang::BO_LOr;
  case OperatorType::NotEqual:      return ::clang::BO_NE;
  default:                          throw InternalErrorException("Unknown VAST relational operator type detected!");
  }
}

::clang::UnaryOperatorKind Vectorizer::VASTExporterBase::_ConvertUnaryOperatorType(AST::Expressions::UnaryOperator::UnaryOperatorType eOpType)
{
  typedef AST::Expressions::UnaryOperator::UnaryOperatorType  OperatorType;

  switch (eOpType)
  {
  case OperatorType::AddressOf:       return UO_AddrOf;
  case OperatorType::BitwiseNot:      return UO_Not;
  case OperatorType::LogicalNot:      return UO_LNot;
  case OperatorType::Minus:           return UO_Minus;
  case OperatorType::Plus:            return UO_Plus;
  case OperatorType::PostDecrement:   return UO_PostDec;
  case OperatorType::PostIncrement:   return UO_PostInc;
  case OperatorType::PreDecrement:    return UO_PreDec;
  case OperatorType::PreIncrement:    return UO_PreInc;
  default:                            throw InternalErrorException("Unknown VAST unary operator type detected!");
  }
}

::clang::QualType Vectorizer::VASTExporterBase::_ConvertTypeInfo(const AST::BaseClasses::TypeInfo &crTypeInfo)
{
  typedef AST::BaseClasses::TypeInfo::KnownTypes    KnownTypes;

  ::clang::QualType qtReturnType;
  switch (crTypeInfo.GetType())
  {
  case KnownTypes::Bool:      qtReturnType = _GetASTContext().BoolTy;           break;
  case KnownTypes::Int8:      qtReturnType = _GetASTContext().SignedCharTy;     break;
  case KnownTypes::UInt8:     qtReturnType = _GetASTContext().UnsignedCharTy;   break;
  case KnownTypes::Int16:     qtReturnType = _GetASTContext().ShortTy;          break;
  case KnownTypes::UInt16:    qtReturnType = _GetASTContext().UnsignedShortTy;  break;
  case KnownTypes::Int32:     qtReturnType = _GetASTContext().IntTy;            break;
  case KnownTypes::UInt32:    qtReturnType = _GetASTContext().UnsignedIntTy;    break;
  case KnownTypes::Int64:     qtReturnType = _GetASTContext().LongTy;           break;
  case KnownTypes::UInt64:    qtReturnType = _GetASTContext().UnsignedLongTy;   break;
  case KnownTypes::Float:     qtReturnType = _GetASTContext().FloatTy;          break;
  case KnownTypes::Double:    qtReturnType = _GetASTContext().DoubleTy;         break;
  case KnownTypes::Unknown:   throw RuntimeErrorException("VAST element type is unknown!");
  default:                    throw InternalErrorException("Unsupported VAST element type detected!");
  }

  if (crTypeInfo.GetConst())
  {
    qtReturnType.addConst();
  }
  else
  {
    qtReturnType.removeLocalConst();
  }

  if (crTypeInfo.GetPointer())
  {
    qtReturnType = _GetASTHelper().GetPointerType( qtReturnType );
  }

  for (auto itDim = crTypeInfo.GetArrayDimensions().end(); itDim != crTypeInfo.GetArrayDimensions().begin(); itDim--)
  {
    qtReturnType = _GetASTHelper().GetConstantArrayType( qtReturnType, *(itDim-1) );
  }

  return qtReturnType;
}


::clang::CastExpr* Vectorizer::VASTExporterBase::_CreateCast(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr)
{
  if (crTargetType.IsArray())
  {
    throw RuntimeErrorException("Conversions into array types are not supported!");
  }
  else if (crTargetType.GetPointer())
  {
    return _CreateCastPointer(crSourceType, crTargetType, pSubExpr);
  }
  else
  {
    return _CreateCastSingleValue(crSourceType, crTargetType, pSubExpr);
  }
}

::clang::CastExpr* Vectorizer::VASTExporterBase::_CreateCastPointer(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr)
{
  if (! crTargetType.GetPointer())
  {
    throw InternalErrorException("Expected a pointer type as target type!");
  }

  ::clang::CastKind eCastKind = ::clang::CK_ReinterpretMemberPointer;

  if (crSourceType.IsArray())
  {
    if (crSourceType.GetArrayDimensions().size() > 1)
    {
      throw RuntimeErrorException("Cannot convert multi-dimensional arrays into pointers!");
    }
    else if (crSourceType.GetPointer())
    {
      throw RuntimeErrorException("Cannot convert pointer arrays into pointers!");
    }

    eCastKind = ::clang::CK_ArrayToPointerDecay;
  }
  else if (crSourceType.IsSingleValue())
  {
    throw RuntimeErrorException("Cannot convert a single value into a pointer!");
  }

  return _GetASTHelper().CreateReinterpretCast( pSubExpr, _ConvertTypeInfo(crTargetType), eCastKind, (! crTargetType.GetConst()) );
}

::clang::CastExpr* Vectorizer::VASTExporterBase::_CreateCastSingleValue(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr)
{
  typedef AST::BaseClasses::TypeInfo::KnownTypes  KnownTypes;

  if (! crTargetType.IsSingleValue())
  {
    throw InternalErrorException("Expected a single value type as target type!");
  }
  else if (! crSourceType.IsSingleValue())
  {
    throw RuntimeErrorException("Cannot dereference a type by a conversion!");
  }


  AST::BaseClasses::TypeInfo TargetType( crTargetType );
  TargetType.SetConst(false);

  KnownTypes eTargetType = TargetType.GetType();
  KnownTypes eSourceType = crSourceType.GetType();

  ::clang::CastKind eCastKind = ::clang::CK_IntegralCast;

  if ( (eTargetType == KnownTypes::Unknown) || (eSourceType == KnownTypes::Unknown) )
  {
    throw InternalErrorException("Cannot convert in between unknown types!");
  }
  else if ( (eSourceType == KnownTypes::Float) || (eSourceType == KnownTypes::Double) )
  {
    switch (eTargetType)
    {
    case KnownTypes::Bool:                            eCastKind = ::clang::CK_FloatingToBoolean;  break;
    case KnownTypes::Float: case KnownTypes::Double:  eCastKind = ::clang::CK_FloatingCast;       break;
    case KnownTypes::Int8:  case KnownTypes::UInt8:
    case KnownTypes::Int16: case KnownTypes::UInt16:
    case KnownTypes::Int32: case KnownTypes::UInt32:
    case KnownTypes::Int64: case KnownTypes::UInt64:  eCastKind = ::clang::CK_FloatingToIntegral; break;
    default:                                          throw InternalErrorException("Unknown conversion target type detected!");
    }
  }
  else if ( (eSourceType == KnownTypes::Int8)  || (eSourceType == KnownTypes::UInt8)  ||
            (eSourceType == KnownTypes::Int16) || (eSourceType == KnownTypes::UInt16) ||
            (eSourceType == KnownTypes::Int32) || (eSourceType == KnownTypes::UInt32) ||
            (eSourceType == KnownTypes::Int64) || (eSourceType == KnownTypes::UInt64) ||
            (eSourceType == KnownTypes::Bool) )
  {
    switch (eTargetType)
    {
    case KnownTypes::Bool:                            eCastKind = ::clang::CK_IntegralToBoolean;  break;
    case KnownTypes::Float: case KnownTypes::Double:  eCastKind = ::clang::CK_IntegralToFloating; break;
    case KnownTypes::Int8:  case KnownTypes::UInt8:
    case KnownTypes::Int16: case KnownTypes::UInt16:
    case KnownTypes::Int32: case KnownTypes::UInt32:
    case KnownTypes::Int64: case KnownTypes::UInt64:  eCastKind = ::clang::CK_IntegralCast;       break;
    default:                                          throw InternalErrorException("Unknown conversion target type detected!");
    }
  }
  else
  {
    throw InternalErrorException("Unknown conversion source type detected!");
  }

  return _GetASTHelper().CreateStaticCast( pSubExpr, _ConvertTypeInfo(TargetType), eCastKind );
}

::clang::DeclRefExpr* Vectorizer::VASTExporterBase::_CreateDeclarationReference(std::string strValueName)
{
  if (! _HasValueDeclaration(strValueName))
  {
    throw InternalErrorException(string("The identifier \"") + strValueName + string("\" has not been declared yet!"));
  }

  return _GetASTHelper().CreateDeclarationReferenceExpression( _mapKnownDeclarations[strValueName] );
}

::clang::ParenExpr* Vectorizer::VASTExporterBase::_CreateParenthesis(::clang::Expr *pSubExpr)
{
  return _GetASTHelper().CreateParenthesisExpression( pSubExpr );
}


::clang::FunctionDecl* Vectorizer::VASTExporterBase::_GetFirstMatchingFunctionDeclaration(string strFunctionName, const QualTypeVectorType &crvecArgTypes)
{
  const unsigned int cuiArgumentCount = static_cast< unsigned int >( crvecArgTypes.size() );

  // Find the first exactly matching function
  FunctionDeclVectorType vecFunctionDecls = _GetMatchingFunctionDeclarations( strFunctionName, cuiArgumentCount );

  for (auto itFuncDecl : vecFunctionDecls)
  {
    bool bFound = true;

    for (unsigned int uiArgIdx = 0; uiArgIdx < cuiArgumentCount; ++uiArgIdx)
    {
      ::clang::QualType qtParamType = itFuncDecl->getParamDecl( uiArgIdx )->getType();

      if ( qtParamType->getCanonicalTypeUnqualified() != crvecArgTypes[uiArgIdx]->getCanonicalTypeUnqualified() )
      {
        bFound = false;
        break;
      }
    }

    if (bFound)
    {
      return itFuncDecl;
    }
  }

  // Cannot find matching function declaration
  return nullptr;
}

Vectorizer::VASTExporterBase::FunctionDeclVectorType Vectorizer::VASTExporterBase::_GetMatchingFunctionDeclarations(string strFunctionName, unsigned int uiParamCount)
{
  FunctionDeclVectorType vecFunctionDecls;

  auto itFunctionParamCountMap = _mapKnownFunctions.find( strFunctionName );
  if (itFunctionParamCountMap == _mapKnownFunctions.end())
  {
    return vecFunctionDecls;
  }

  FunctionDeclParamCountMapType &rParamCountMap = itFunctionParamCountMap->second;
  
  auto itFunctionDeclVec = rParamCountMap.find(uiParamCount);
  if (itFunctionDeclVec == rParamCountMap.end())
  {
    return vecFunctionDecls;
  }

  FunctionDeclVectorType &rvecKnownFunctionDecls = itFunctionDeclVec->second;

  vecFunctionDecls.insert( vecFunctionDecls.begin(), rvecKnownFunctionDecls.begin(), rvecKnownFunctionDecls.end() );

  return std::move( vecFunctionDecls );
}

::clang::QualType Vectorizer::VASTExporterBase::_GetVariableType(AST::BaseClasses::VariableInfoPtr spVariableInfo)
{
  if ( spVariableInfo->GetVectorize() )
  {
    return _GetVectorizedType( spVariableInfo->GetTypeInfo() );
  }
  else
  {
    return _ConvertTypeInfo( spVariableInfo->GetTypeInfo() );
  }
}

bool Vectorizer::VASTExporterBase::_HasValueDeclaration(string strDeclName)
{
  return (_mapKnownDeclarations.find(strDeclName) != _mapKnownDeclarations.end());
}

void Vectorizer::VASTExporterBase::_Reset()
{
  _mapKnownDeclarations.clear();
  _pDeclContext = nullptr;
}





// Implementation of class Vectorizer::VASTExportArray
::clang::Expr* Vectorizer::VASTExportArray::VectorIndex::CreateIndexExpression(::clang::hipacc::Backend::ClangASTHelper &rASTHelper) const
{
  switch (_ceIndexType)
  {
  case VectorIndexType::Constant:     return rASTHelper.CreateIntegerLiteral( static_cast<int32_t>(_ciVectorIndex) );
  case VectorIndexType::Identifier:   return rASTHelper.CreateDeclarationReferenceExpression( const_cast<::clang::ValueDecl*>( _cpIndexExprDecl ) );
  default:                            throw InternalErrorException("Unknown vector index type!");
  }
}

Vectorizer::VASTExportArray::VASTExportArray(IndexType VectorWidth, ::clang::ASTContext &rAstContext) : BaseType(rAstContext), _VectorWidth(VectorWidth)
{
  if (VectorWidth <= static_cast<IndexType>(0))
  {
    throw InternalErrorException("The vector width must be positive");
  }
}


::clang::CompoundStmt* Vectorizer::VASTExportArray::_BuildCompoundStatement(AST::ScopePtr spScope)
{
  ClangASTHelper::StatementVectorType vecChildren;

  // Declare all array variables
  {
    AST::Scope::VariableDeclarationVectorType vecVarDecls = spScope->GetVariableDeclarations();

    for (auto itVarDecl : vecVarDecls)
    {
      AST::BaseClasses::VariableInfoPtr spVarInfo = itVarDecl->LookupVariableInfo();
      AST::BaseClasses::TypeInfo        &rVarType = spVarInfo->GetTypeInfo();

      if (rVarType.IsArray())
      {
        // Remove const flag, otherwise the assignments will not compile
        if (! rVarType.GetPointer())
        {
          rVarType.SetConst(false);
        }

        ::clang::ValueDecl *pValueDecl = _BuildValueDeclaration(itVarDecl);
        
        vecChildren.push_back( _GetASTHelper().CreateDeclarationStatement(pValueDecl) );
      }
    }
  }

  // Build child statements
  for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < spScope->GetChildCount(); ++iChildIdx)
  {
    AST::BaseClasses::NodePtr spChild = spScope->GetChild(iChildIdx);
    ::clang::Stmt *pChildStmt = nullptr;

    if (spChild->IsType<AST::Scope>())
    {
      pChildStmt = _BuildCompoundStatement( spChild->CastToType<AST::Scope>() );
    }
    else if (spChild->IsType<AST::Expressions::AssignmentOperator>())
    {
      AST::Expressions::AssignmentOperatorPtr spAssignment = spChild->CastToType<AST::Expressions::AssignmentOperator>();
      AST::BaseClasses::ExpressionPtr spLHS = spAssignment->GetLHS();

      if (spLHS->IsType<AST::Expressions::Identifier>())
      {
        AST::Expressions::IdentifierPtr spIdentifier = spLHS->CastToType<AST::Expressions::Identifier>();

        if (! _HasValueDeclaration(spIdentifier->GetName()) )
        {
          // Create variable declaration on first use
          AST::BaseClasses::VariableInfoPtr spVariableInfo = spIdentifier->LookupVariableInfo();

          AST::BaseClasses::ExpressionPtr spRHS = spAssignment->GetRHS();
          ::clang::Expr *pInitExpression = nullptr;

          // Build init expression
          if (spVariableInfo->GetVectorize() && spVariableInfo->GetTypeInfo().IsSingleValue())
          {
            ClangASTHelper::ExpressionVectorType vecSubExpr;

            for (IndexType iVecIdx = static_cast<IndexType>(0); iVecIdx < _VectorWidth; ++iVecIdx)
            {
              vecSubExpr.push_back( _BuildExpression(spRHS, iVecIdx) );
            }

            pInitExpression = _GetASTHelper().CreateInitListExpression(vecSubExpr);
          }
          else
          {
            pInitExpression = _BuildExpression( spRHS, VectorIndex() );
          }

          ::clang::ValueDecl *pVarDecl = _BuildValueDeclaration(spIdentifier, pInitExpression);
          pChildStmt = _GetASTHelper().CreateDeclarationStatement(pVarDecl);
        }
        else
        {
          pChildStmt = _BuildExpressionStatement(spAssignment);
        }
      }
      else
      {
        pChildStmt = _BuildExpressionStatement(spAssignment);
      }
    }
    else if (spChild->IsType<AST::BaseClasses::Expression>())
    {
      pChildStmt = _BuildExpressionStatement( spChild->CastToType<AST::BaseClasses::Expression>() );
    }
    else if (spChild->IsType<AST::BaseClasses::ControlFlowStatement>())
    {
      AST::BaseClasses::ControlFlowStatementPtr spControlFlow = spChild->CastToType<AST::BaseClasses::ControlFlowStatement>();

      if (spControlFlow->IsType<AST::ControlFlow::Loop>())
      {
        pChildStmt = _BuildLoop( spControlFlow->CastToType<AST::ControlFlow::Loop>() );
      }
      else if (spControlFlow->IsType<AST::ControlFlow::LoopControlStatement>())
      {
        pChildStmt = _BuildLoopControlStatement( spControlFlow->CastToType<AST::ControlFlow::LoopControlStatement>() );
      }
      else if (spControlFlow->IsType<AST::ControlFlow::BranchingStatement>())
      {
        pChildStmt = _BuildIfStatement( spControlFlow->CastToType<AST::ControlFlow::BranchingStatement>() );
      }
      else if (spControlFlow->IsType<AST::ControlFlow::ReturnStatement>())
      {
        pChildStmt = _GetASTHelper().CreateReturnStatement();
      }
      else
      {
        throw InternalErrorException("Unsupported VAST control flow statement detected!");
      }
    }
    else
    {
      throw InternalErrorException("Unsupported VAST node detected!");
    }

    vecChildren.push_back( pChildStmt );
  }

  return _GetASTHelper().CreateCompoundStatement(vecChildren);
}

::clang::Expr* Vectorizer::VASTExportArray::_BuildExpression(AST::BaseClasses::ExpressionPtr spExpression, const VectorIndex &crVectorIndex)
{
  ::clang::Expr *pReturnExpr = nullptr;

  if (spExpression->IsType<AST::Expressions::Value>())
  {
    AST::Expressions::ValuePtr spValue = spExpression->CastToType<AST::Expressions::Value>();

    if (spValue->IsType<AST::Expressions::Constant>())
    {
      pReturnExpr = _BuildConstant(spValue->CastToType<AST::Expressions::Constant>());
    }
    else if (spValue->IsType<AST::Expressions::Identifier>())
    {
      AST::Expressions::IdentifierPtr   spIdentifier      = spValue->CastToType<AST::Expressions::Identifier>();
      AST::BaseClasses::VariableInfoPtr spIdentifierInfo  = spIdentifier->LookupVariableInfo();
      AST::BaseClasses::TypeInfo        IdentifierType    = spIdentifierInfo->GetTypeInfo();

      string strIdentifierName = spIdentifier->GetName();

      if (! _HasValueDeclaration(strIdentifierName))
      {
        throw InternalErrorException( string("The referenced identifier \"") + strIdentifierName + string("\" has not been declared!") );
      }

      ::clang::DeclRefExpr *pDeclRef = _CreateDeclarationReference( strIdentifierName );

      if (spIdentifierInfo->GetVectorize() && IdentifierType.IsSingleValue())
      {
        ::clang::Expr *pIndex = crVectorIndex.CreateIndexExpression( _GetASTHelper() );

        pReturnExpr = _GetASTHelper().CreateArraySubscriptExpression( pDeclRef, pIndex, _ConvertTypeInfo(IdentifierType), (! IdentifierType.GetConst()) );
      }
      else
      {
        pReturnExpr = pDeclRef;
      }
    }
    else if (spValue->IsType<AST::Expressions::MemoryAccess>())
    {
      AST::Expressions::MemoryAccessPtr spMemoryAccess    = spValue->CastToType<AST::Expressions::MemoryAccess>();
      AST::BaseClasses::ExpressionPtr   spIndexExpression = spMemoryAccess->GetIndexExpression();
      AST::BaseClasses::TypeInfo        ReturnType        = spMemoryAccess->GetResultType();

      ::clang::Expr *pArrayRef  = _BuildExpression( spMemoryAccess->GetMemoryReference(), crVectorIndex );
      ::clang::Expr *pIndexExpr = _BuildExpression( spIndexExpression, crVectorIndex );

      if ( ReturnType.IsSingleValue() && (! spIndexExpression->IsVectorized()) )
      {
        ::clang::Expr *pIndexOffset = crVectorIndex.CreateIndexExpression( _GetASTHelper() );

        pIndexExpr = _GetASTHelper().CreateBinaryOperator( pIndexExpr, pIndexOffset, ::clang::BO_Add, _ConvertTypeInfo(spIndexExpression->GetResultType()) );
      }

      pReturnExpr = _GetASTHelper().CreateArraySubscriptExpression(pArrayRef, pIndexExpr, _ConvertTypeInfo(ReturnType), (!ReturnType.GetConst()));
    }
    else
    {
      throw InternalErrorException("Unknown VAST value node detected!");
    }
  }
  else if (spExpression->IsType<AST::Expressions::UnaryExpression>())
  {
    AST::Expressions::UnaryExpressionPtr  spUnaryExpression = spExpression->CastToType<AST::Expressions::UnaryExpression>();
    AST::BaseClasses::ExpressionPtr       spSubExpression   = spUnaryExpression->GetSubExpression();
    AST::BaseClasses::TypeInfo            ResultType        = spUnaryExpression->GetResultType();
    ::clang::Expr*                        pSubExpr          = _BuildExpression( spSubExpression, crVectorIndex );

    if (spUnaryExpression->IsType<AST::Expressions::Conversion>())
    {
      pReturnExpr = _CreateCast( spSubExpression->GetResultType(), ResultType, pSubExpr );
    }
    else if (spUnaryExpression->IsType<AST::Expressions::Parenthesis>())
    {
      pReturnExpr = _CreateParenthesis( pSubExpr );
    }
    else if (spUnaryExpression->IsType<AST::Expressions::UnaryOperator>())
    {
      ::clang::UnaryOperatorKind eOpCode = _ConvertUnaryOperatorType( spUnaryExpression->CastToType<AST::Expressions::UnaryOperator>()->GetOperatorType() );

      pReturnExpr = _GetASTHelper().CreateUnaryOperator( pSubExpr, eOpCode, _ConvertTypeInfo(ResultType) );
    }
    else
    {
      throw InternalErrorException("Unknown VAST unary expression node detected!");
    }
  }
  else if (spExpression->IsType<AST::Expressions::BinaryOperator>())
  {
    AST::Expressions::BinaryOperatorPtr spBinaryOperator = spExpression->CastToType<AST::Expressions::BinaryOperator>();

    ::clang::Expr *pExprLHS = _BuildExpression( spBinaryOperator->GetLHS(), crVectorIndex );
    ::clang::Expr *pExprRHS = _BuildExpression( spBinaryOperator->GetRHS(), crVectorIndex );

    ::clang::BinaryOperatorKind eOpCode;
    bool bIsMaskedAssignment = false;

    if (spBinaryOperator->IsType<AST::Expressions::ArithmeticOperator>())
    {
      eOpCode = _ConvertArithmeticOperatorType( spBinaryOperator->CastToType<AST::Expressions::ArithmeticOperator>()->GetOperatorType() );
    }
    else if (spBinaryOperator->IsType<AST::Expressions::AssignmentOperator>())
    {
      eOpCode             = ::clang::BO_Assign;
      bIsMaskedAssignment = spBinaryOperator->CastToType<AST::Expressions::AssignmentOperator>()->IsMasked();
    }
    else if (spBinaryOperator->IsType<AST::Expressions::RelationalOperator>())
    {
      eOpCode = _ConvertRelationalOperatorType( spBinaryOperator->CastToType<AST::Expressions::RelationalOperator>()->GetOperatorType() );
    }
    else
    {
      throw InternalErrorException("Unknown VAST binary operator node detected!");
    }

    if (bIsMaskedAssignment)
    {
      AST::Expressions::AssignmentOperatorPtr spAssignment = spBinaryOperator->CastToType<AST::Expressions::AssignmentOperator>();

      ::clang::Expr *pExprMaskCond  = _CreateParenthesis( _BuildExpression(spAssignment->GetMask(), crVectorIndex) );
      ::clang::Expr *pExprLHSParen  = _CreateParenthesis( pExprLHS );

      pExprRHS = _CreateParenthesis( pExprRHS );
      pExprRHS = _GetASTHelper().CreateConditionalOperator( pExprMaskCond, pExprRHS, pExprLHSParen, pExprRHS->getType() );
    }

    pReturnExpr = _GetASTHelper().CreateBinaryOperator( pExprLHS, pExprRHS, eOpCode, _ConvertTypeInfo(spBinaryOperator->GetResultType()) );
  }
  else if (spExpression->IsType<AST::Expressions::FunctionCall>())
  {
    pReturnExpr = _BuildFunctionCall(spExpression->CastToType<AST::Expressions::FunctionCall>(), crVectorIndex);
  }
  else if (spExpression->IsType<AST::VectorSupport::VectorExpression>())
  {
    AST::VectorSupport::VectorExpressionPtr spVectorExpression = spExpression->CastToType<AST::VectorSupport::VectorExpression>();

    if (spVectorExpression->IsType<AST::VectorSupport::BroadCast>())
    {
      AST::VectorSupport::BroadCastPtr  spBroadCast     = spVectorExpression->CastToType<AST::VectorSupport::BroadCast>();
      AST::BaseClasses::ExpressionPtr   spSubExpression = spBroadCast->GetSubExpression();

      if (spSubExpression->IsVectorized())
      {
        throw RuntimeErrorException("Cannot broad cast vectorized expressions!");
      }
      else if (! spSubExpression->GetResultType().IsSingleValue())
      {
        throw RuntimeErrorException("Broad casts are only allowed for single values!");
      }

      pReturnExpr = _BuildExpression(spSubExpression, VectorIndex());
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::CheckActiveElements>())
    {
      typedef AST::VectorSupport::CheckActiveElements::CheckType  CheckType;

      AST::VectorSupport::CheckActiveElementsPtr  spCheckElements = spVectorExpression->CastToType<AST::VectorSupport::CheckActiveElements>();
      AST::BaseClasses::ExpressionPtr             spSubExpression = spCheckElements->GetSubExpression();

      if (! spSubExpression->IsVectorized())
      {
        throw RuntimeErrorException("Cannot check active vector elements in a scalar expression!");
      }

      ::clang::BinaryOperatorKind eCombineOpCode;
      switch (spCheckElements->GetCheckType())
      {
      case CheckType::All:    eCombineOpCode = ::clang::BO_LAnd;  break;
      case CheckType::Any:    eCombineOpCode = ::clang::BO_LOr;   break;
      case CheckType::None:   eCombineOpCode = ::clang::BO_LOr;   break;
      default:                throw RuntimeErrorException("Unknown active vector element check type detected!");
      }

      ClangASTHelper::ExpressionVectorType vecSubExpressions;
      for (IndexType iVecIdx = static_cast<IndexType>(0); iVecIdx < _VectorWidth; ++iVecIdx)
      {
        ::clang::Expr *pCurrentSubExpr = _BuildExpression( spSubExpression, iVecIdx );
        vecSubExpressions.push_back( _CreateParenthesis( pCurrentSubExpr ) );
      }

      ::clang::QualType qtBoolType = _GetASTContext().BoolTy;

      for (IndexType iVecIdx = static_cast<IndexType>(1); iVecIdx < _VectorWidth; ++iVecIdx)
      {
        vecSubExpressions[0] = _GetASTHelper().CreateBinaryOperator(vecSubExpressions[0], vecSubExpressions[iVecIdx], eCombineOpCode, qtBoolType);
      }

      pReturnExpr = vecSubExpressions[0];

      if (spCheckElements->GetCheckType() == CheckType::None)
      {
        pReturnExpr = _CreateParenthesis( pReturnExpr );
        pReturnExpr = _GetASTHelper().CreateUnaryOperator(pReturnExpr, ::clang::UO_LNot, qtBoolType);
      }
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::VectorIndex>())
    {
      pReturnExpr = crVectorIndex.CreateIndexExpression( _GetASTHelper() );
    }
    else
    {
      throw InternalErrorException("Unknown VAST vector expression node detected!");
    }
  }
  else
  {
    throw InternalErrorException("Unknown VAST expression node detected!");
  }

  return pReturnExpr;
}

::clang::Stmt* Vectorizer::VASTExportArray::_BuildExpressionStatement(AST::BaseClasses::ExpressionPtr spExpression)
{
  if (spExpression->IsVectorized())
  {
    if (_pVectorIndexExpr)
    {
      // Create a vector loop
      ::clang::Stmt *pExprStmt = _BuildExpression(spExpression, VectorIndex(_pVectorIndexExpr));
      pExprStmt                = _GetASTHelper().CreateCompoundStatement(pExprStmt);

      ::clang::Expr *pCondition = _GetASTHelper().CreateBinaryOperatorLessThan( _GetASTHelper().CreateDeclarationReferenceExpression(_pVectorIndexExpr),
                                                                                _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(_VectorWidth)) );

      ::clang::Expr *pIncrement = _GetASTHelper().CreateUnaryOperator( _GetASTHelper().CreateDeclarationReferenceExpression(_pVectorIndexExpr), ::clang::UO_PreInc, _pVectorIndexExpr->getType() );

      return _GetASTHelper().CreateLoopFor( pCondition, pExprStmt, _GetASTHelper().CreateDeclarationStatement(_pVectorIndexExpr), pIncrement );
    }
    else
    {
      // Unroll the vector-loops
      ClangASTHelper::StatementVectorType vecStatements;

      for (IndexType iVecIdx = static_cast<IndexType>(0); iVecIdx < _VectorWidth; ++iVecIdx)
      {
        vecStatements.push_back( _BuildExpression(spExpression, iVecIdx) );
      }

      return _GetASTHelper().CreateCompoundStatement(vecStatements);
    }
  }
  else
  {
    return _BuildExpression(spExpression, VectorIndex());
  }
}

::clang::Expr* Vectorizer::VASTExportArray::_BuildFunctionCall(AST::Expressions::FunctionCallPtr spFunctionCall, const VectorIndex &crVectorIndex)
{
  string          strFunctionName = spFunctionCall->GetName();
  const IndexType ciArgumentCount = spFunctionCall->GetCallParameterCount();

  // Build the argument expressions
  ClangASTHelper::ExpressionVectorType  vecArguments;
  ClangASTHelper::QualTypeVectorType    vecArgumentTypes;

  for (IndexType iArgIdx = static_cast<IndexType>(0); iArgIdx < ciArgumentCount; ++iArgIdx)
  {
    AST::BaseClasses::ExpressionPtr spCurrentArgument     = spFunctionCall->GetCallParameter(iArgIdx);
    ::clang::Expr                   *pCurrentArgumentExpr = _BuildExpression(spCurrentArgument, crVectorIndex);

    vecArguments.push_back( pCurrentArgumentExpr );
    vecArgumentTypes.push_back( _ConvertTypeInfo(spCurrentArgument->GetResultType()) );
  }


  // Find the first exactly matching function
  ::clang::FunctionDecl *pCalleeDecl = _GetFirstMatchingFunctionDeclaration( strFunctionName, vecArgumentTypes );
  if (pCalleeDecl == nullptr)
  {
    throw RuntimeErrorException(string("Could not find matching FunctionDecl object for function call \"") + strFunctionName + string("\"!"));
  }

  return _GetASTHelper().CreateFunctionCall( pCalleeDecl, vecArguments );
}

::clang::IfStmt* Vectorizer::VASTExportArray::_BuildIfStatement(AST::ControlFlow::BranchingStatementPtr spBranchingStatement)
{
  if (spBranchingStatement->IsVectorized())
  {
    throw RuntimeErrorException("Cannot export branching statements with vectorized conditions => Rebuild the branching statements before calling the export!");
  }


  ClangASTHelper::ExpressionVectorType  vecConditions;
  ClangASTHelper::StatementVectorType   vecBranchBodies;

  for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < spBranchingStatement->GetConditionalBranchesCount(); ++iBranchIdx)
  {
    AST::ControlFlow::ConditionalBranchPtr spCurrentBranch = spBranchingStatement->GetConditionalBranch(iBranchIdx);

    vecConditions.push_back( _BuildExpression( spCurrentBranch->GetCondition(), VectorIndex() ) );
    vecBranchBodies.push_back( _BuildCompoundStatement( spCurrentBranch->GetBody() ) );
  }


  AST::ScopePtr spDefaultBranch = spBranchingStatement->GetDefaultBranch();
  ::clang::Stmt *pDefaultBranch = ( spDefaultBranch->GetChildCount() > 0 ) ? _BuildCompoundStatement( spDefaultBranch ) : nullptr;

  return _GetASTHelper().CreateIfStatement( vecConditions, vecBranchBodies, pDefaultBranch );
}

::clang::Stmt* Vectorizer::VASTExportArray::_BuildLoop(AST::ControlFlow::LoopPtr spLoop)
{
  if (spLoop->IsVectorized())
  {
    throw RuntimeErrorException("Cannot export loops with vectorized conditions => Rebuild the loops before calling the export!");
  }

  ::clang::CompoundStmt *pLoopBody      = _BuildCompoundStatement( spLoop->GetBody() );
  ::clang::Expr         *pConditionExpr = _BuildExpression( spLoop->GetCondition(), VectorIndex() );
  ::clang::Expr         *pIncrementExpr = nullptr;

  if (spLoop->GetIncrement())
  {
    AST::BaseClasses::ExpressionPtr spIncrement   = spLoop->GetIncrement();

    if (spIncrement->IsVectorized())
    {
      ClangASTHelper::ExpressionVectorType vecIncExpressions;

      for (IndexType iVecIdx = static_cast<IndexType>(0); iVecIdx < _VectorWidth; ++iVecIdx)
      {
        vecIncExpressions.push_back( _CreateParenthesis( _BuildExpression(spIncrement, iVecIdx) ) );
      }

      for (IndexType iVecIdx = static_cast<IndexType>(1); iVecIdx < _VectorWidth; ++iVecIdx)
      {
        vecIncExpressions[0] = _GetASTHelper().CreateBinaryOperatorComma( vecIncExpressions[0], vecIncExpressions[iVecIdx] );
      }

      pIncrementExpr = vecIncExpressions[0];
    }
    else
    {
      pIncrementExpr = _BuildExpression( spIncrement, VectorIndex() );
    }
  }

  return BaseType::_BuildLoop( spLoop->GetLoopType(), pConditionExpr, pIncrementExpr );
}


::clang::QualType Vectorizer::VASTExportArray::_GetVectorizedType(AST::BaseClasses::TypeInfo &crOriginalTypeInfo)
{
  AST::BaseClasses::TypeInfo ReturnType = crOriginalTypeInfo;

  if (! crOriginalTypeInfo.GetPointer())
  {
    ReturnType.GetArrayDimensions().push_back( _VectorWidth );
  }

  return _ConvertTypeInfo( ReturnType );
}

::clang::FunctionDecl* Vectorizer::VASTExportArray::ExportVASTFunction(AST::FunctionDeclarationPtr spVASTFunction, bool bUnrollVectorLoops)
{
  if (! spVASTFunction)
  {
    throw InternalErrors::NullPointerException("spVASTFunction");
  }

  // Create the function declaration statement
  ::clang::FunctionDecl *pFunctionDecl = _BuildFunctionDeclaration( spVASTFunction );

  // If requested, create the vector index expression
  if (bUnrollVectorLoops)
  {
    _pVectorIndexExpr = nullptr;
  }
  else
  {
    _pVectorIndexExpr = _GetASTHelper().CreateVariableDeclaration( pFunctionDecl, "_vec_idx_", _GetASTContext().IntTy, _GetASTHelper().CreateIntegerLiteral(0) );
  }

  // Build the function body
  pFunctionDecl->setBody( _BuildCompoundStatement( spVASTFunction->GetBody() ) );


  // Reset exporter state
  _Reset();
  _pVectorIndexExpr = nullptr;

  return pFunctionDecl;
}




void Vectorizer::Transformations::CheckInternalDeclaration::Execute(AST::ScopePtr spScope)
{
  if (spScope->HasVariableDeclaration(_strDeclName))
  {
    _bFound = true;
  }
}

void Vectorizer::Transformations::FindBranchingInternalAssignments::Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt)
{
  list< AST::BaseClasses::ExpressionPtr > lstConditions;

  // Find all assignments in every conditional branch => each branch depends on its condition as well as on the conditions of the preceding branches
  for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < spBranchingStmt->GetConditionalBranchesCount(); ++iBranchIdx)
  {
    AST::ControlFlow::ConditionalBranchPtr spBranch = spBranchingStmt->GetConditionalBranch(iBranchIdx);
    if (! spBranch)
    {
      throw InternalErrors::NullPointerException("spBranch");
    }

    lstConditions.push_back(spBranch->GetCondition());

    Transformations::FindAssignments AssignmentFinder;

    Transformations::Run(spBranch->GetBody(), AssignmentFinder);

    for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
    {
      // Check if the assignment is done to a branch internal variable (these variables do not depend on the conditions)
      CheckInternalDeclaration DeclChecker( _GetAssigneeInfo(itAssignment)->GetName() );
      Transformations::Run(spBranch, DeclChecker);

      // Only add assignments to external variables to the result map
      if (! DeclChecker.Found())
      {
        for each (auto itCondition in lstConditions)
        {
          mapConditionalAssignments[itAssignment].push_back(itCondition);
        }
      }
    }
  }

  // Find the assignments in the default branch => this branch depends on the conditions of ALL conditional branches
  {
    Transformations::FindAssignments AssignmentFinder;

    Transformations::Run(spBranchingStmt->GetDefaultBranch(), AssignmentFinder);

    for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
    {
      // Check if the assignment is done to a branch internal variable (these variables do not depend on the conditions)
      CheckInternalDeclaration DeclChecker( _GetAssigneeInfo(itAssignment)->GetName() );
      Transformations::Run(spBranchingStmt->GetDefaultBranch(), DeclChecker);

      // Only add assignments to external variables to the result map
      if (! DeclChecker.Found())
      {
        for each (auto itCondition in lstConditions)
        {
          mapConditionalAssignments[itAssignment].push_back(itCondition);
        }
      }
    }
  }
}

void Vectorizer::Transformations::FindLoopInternalAssignments::Execute(AST::ControlFlow::LoopPtr spLoop)
{
  // Find all assignments inside the loop body
  Transformations::FindAssignments AssignmentFinder;

  Transformations::Run(spLoop->GetBody(), AssignmentFinder);

  for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
  {
    // Check if the assignment is done to a loop internal variable (these variables do not depend on the loop condition)
    CheckInternalDeclaration DeclChecker( _GetAssigneeInfo(itAssignment)->GetName() );
    Transformations::Run(spLoop->GetBody(), DeclChecker);

    // Only add assignments to external variables to the result map
    if (! DeclChecker.Found())
    {
      mapConditionalAssignments[itAssignment].push_back(spLoop->GetCondition());
    }
  }
}

void Vectorizer::Transformations::FlattenMemoryAccesses::Execute(AST::Expressions::MemoryAccessPtr spMemoryAccess)
{
  if (spMemoryAccess->IsVectorized())
  {
    AST::BaseClasses::ExpressionPtr spIndexExpr = spMemoryAccess->GetIndexExpression();

    bool bIsSingleValue = spIndexExpr->IsType<AST::Expressions::Identifier>() || spIndexExpr->IsType<AST::Expressions::Constant>();

    if (! bIsSingleValue)
    {
      AST::ScopePosition  ScopePos        = spMemoryAccess->GetScopePosition();
      AST::ScopePtr       spCurrentScope  = ScopePos.GetScope();

      // Create a name for a new tempory index variable
      string strIndexVariableName = VASTBuilder::GetNextFreeVariableName(spCurrentScope, VASTBuilder::GetTemporaryNamePrefix() + string("_index"));

      // Create the new index variable declaration
      spCurrentScope->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create(strIndexVariableName, spIndexExpr->GetResultType(), spIndexExpr->IsVectorized()) );

      // Create the assignment expression for the new index variable
      spCurrentScope->InsertChild( ScopePos.GetChildIndex(), AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create(strIndexVariableName), spIndexExpr ) );

      // Set the new index variable as index expression for the memory access
      spMemoryAccess->SetIndexExpression( AST::Expressions::Identifier::Create( strIndexVariableName ) );
    }
  }
}

Vectorizer::IndexType Vectorizer::Transformations::FlattenScopes::ProcessChild(AST::ScopePtr spParentScope, IndexType iChildIndex, AST::ScopePtr spChildScope)
{
  bool      bRemoved    = false;
  IndexType ChildCount  = spChildScope->GetChildCount();

  if (ChildCount == static_cast<IndexType>(0))
  {
    spParentScope->RemoveChild(iChildIndex);
    bRemoved = true;
    --iChildIndex;
  }
  else if (ChildCount == static_cast<IndexType>(1))
  {
    spParentScope->SetChild(iChildIndex, spChildScope->GetChild(0));
    bRemoved = true;
  }

  if (bRemoved)
  {
    spParentScope->ImportVariableDeclarations( spChildScope );
  }

  return iChildIndex;
}

void Vectorizer::Transformations::InsertRequiredBroadcasts::Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp)
{
  if (! spCurrentBinOp->IsVectorized())
  {
    return; // Broadcasts only required for vectorized expressions
  }

  AST::BaseClasses::ExpressionPtr spLHS = spCurrentBinOp->GetLHS();
  AST::BaseClasses::ExpressionPtr spRHS = spCurrentBinOp->GetRHS();

  if ((! spLHS) || (! spRHS))
  {
    return; // Nothing to do for incomplete binary operator expressions
  }

  if (spCurrentBinOp->IsType<AST::Expressions::AssignmentOperator>())
  {
    AST::BaseClasses::TypeInfo AssigneeType = spLHS->GetResultType();

    if (! spRHS->IsVectorized())
    {
      spCurrentBinOp->SetRHS( AST::VectorSupport::BroadCast::Create(spRHS) );
    }
  }
  else if (spCurrentBinOp->IsType<AST::Expressions::ArithmeticOperator>() || spCurrentBinOp->IsType<AST::Expressions::RelationalOperator>())
  {
    if (! spLHS->IsVectorized())
    {
      spCurrentBinOp->SetLHS( AST::VectorSupport::BroadCast::Create(spLHS) );
    }

    if (! spRHS->IsVectorized())
    {
      spCurrentBinOp->SetRHS( AST::VectorSupport::BroadCast::Create(spRHS) );
    }
  }
  else
  {
    throw InternalErrorException("Unknown VAST binary operator type detected!");
  }
}

void Vectorizer::Transformations::InsertRequiredConversions::Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp)
{
  AST::BaseClasses::ExpressionPtr spLHS = spCurrentBinOp->GetLHS();
  AST::BaseClasses::ExpressionPtr spRHS = spCurrentBinOp->GetRHS();

  if ((! spLHS) || (! spRHS))
  {
    return; // Nothing to do for incomplete binary operator expressions
  }

  if (spCurrentBinOp->IsType<AST::Expressions::ArithmeticOperator>())
  {
    AST::BaseClasses::TypeInfo ResultType = spCurrentBinOp->CastToType< AST::Expressions::ArithmeticOperator >()->GetResultType();

    if (! spLHS->GetResultType().IsEqual(ResultType, true))
    {
      spCurrentBinOp->SetLHS( AST::Expressions::Conversion::Create(ResultType, spLHS, false) );
    }

    if (! spRHS->GetResultType().IsEqual(ResultType, true))
    {
      spCurrentBinOp->SetRHS( AST::Expressions::Conversion::Create(ResultType, spRHS, false) );
    }
  }
  else if (spCurrentBinOp->IsType<AST::Expressions::AssignmentOperator>())
  {
    AST::BaseClasses::TypeInfo AssigneeType = spLHS->GetResultType();

    if (! spRHS->GetResultType().IsEqual(AssigneeType, true))
    {
      spCurrentBinOp->SetRHS( AST::Expressions::Conversion::Create(AssigneeType, spRHS, false) );
    }
  }
  else if (spCurrentBinOp->IsType<AST::Expressions::RelationalOperator>())
  {
    AST::BaseClasses::TypeInfo ComparisonType = spCurrentBinOp->CastToType< AST::Expressions::RelationalOperator >()->GetComparisonType();

    if (! spLHS->GetResultType().IsEqual(ComparisonType, true))
    {
      spCurrentBinOp->SetLHS( AST::Expressions::Conversion::Create(ComparisonType, spLHS, false) );
    }

    if (! spRHS->GetResultType().IsEqual(ComparisonType, true))
    {
      spCurrentBinOp->SetRHS( AST::Expressions::Conversion::Create(ComparisonType, spRHS, false) );
    }
  }
  else
  {
    throw InternalErrorException("Unknown VAST binary operator type detected!");
  }
}

Vectorizer::IndexType Vectorizer::Transformations::RemoveImplicitConversions::ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion)
{
  AST::BaseClasses::ExpressionPtr spSubExpression = spConversion->GetSubExpression();
  AST::BaseClasses::TypeInfo      ConvertType = spConversion->GetResultType();

  if (! spConversion->GetExplicit())
  {
    spParentExpression->SetSubExpression(iChildIndex, spConversion->GetSubExpression());
  }

  return iChildIndex;
}

Vectorizer::IndexType Vectorizer::Transformations::RemoveUnnecessaryConversions::ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion)
{
  AST::BaseClasses::ExpressionPtr spSubExpression = spConversion->GetSubExpression();
  AST::BaseClasses::TypeInfo      ConvertType     = spConversion->GetResultType();

  if (spSubExpression)
  {
    AST::BaseClasses::TypeInfo  ChildType = spConversion->GetSubExpression()->GetResultType();
    bool bRemoveConversion = false;

    if (spSubExpression->IsType<AST::Expressions::Constant>())
    {
      AST::Expressions::ConstantPtr spConstant = spSubExpression->CastToType<AST::Expressions::Constant>();
      spConstant->ChangeType(ConvertType.GetType());
      bRemoveConversion = true;
    }
    else
    {
      bRemoveConversion = ConvertType.IsEqual(ChildType, true);
    }

    if (bRemoveConversion)
    {
      if (spParentExpression->IsType<AST::Expressions::Conversion>())
      {
        AST::Expressions::ConversionPtr spParentConversion = spParentExpression->CastToType<AST::Expressions::Conversion>();

        spParentConversion->SetExplicit( spParentConversion->GetExplicit() || spConversion->GetExplicit() );
      }
      else
      {
        AST::BaseClasses::ExpressionPtr spSubExpr = spConversion->GetSubExpression();
        if (spSubExpr && spSubExpr->IsType<AST::Expressions::Conversion>())
        {
          AST::Expressions::ConversionPtr spChildConversion = spSubExpr->CastToType<AST::Expressions::Conversion>();

          spChildConversion->SetExplicit( spChildConversion->GetExplicit() || spConversion->GetExplicit() );
        }
      }

      spParentExpression->SetSubExpression(iChildIndex, spConversion->GetSubExpression());
    }
  }

  return iChildIndex;
}

void Vectorizer::Transformations::SeparateBranchingStatements::Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt)
{
  if (spBranchingStmt->IsVectorized())
  {
    IndexType iFirstVecBranch = static_cast< IndexType >( 0 );
    for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < spBranchingStmt->GetConditionalBranchesCount(); ++iBranchIdx)
    {
      if (spBranchingStmt->GetConditionalBranch(iBranchIdx)->IsVectorized())
      {
        iFirstVecBranch = iBranchIdx;
        break;
      }
    }

    if (iFirstVecBranch == static_cast<IndexType>(0))
    {
      return;
    }


    AST::ControlFlow::BranchingStatementPtr spVecBranchStatement = AST::ControlFlow::BranchingStatement::Create();

    while (iFirstVecBranch < spBranchingStmt->GetConditionalBranchesCount())
    {
      spVecBranchStatement->AddConditionalBranch( spBranchingStmt->GetConditionalBranch(iFirstVecBranch) );

      spBranchingStmt->RemoveConditionalBranch(iFirstVecBranch);
    }

    AST::ScopePtr spDefaultBranch     = spBranchingStmt->GetDefaultBranch();
    AST::ScopePtr spDefaultBranchNew  = spVecBranchStatement->GetDefaultBranch();

    while (spDefaultBranch->GetChildCount() > static_cast< IndexType >(0))
    {
      spDefaultBranchNew->AddChild( spDefaultBranch->GetChild(0) );

      spDefaultBranch->RemoveChild(0);
    }

    spDefaultBranchNew->ImportVariableDeclarations( spDefaultBranch );

    spDefaultBranch->AddChild(spVecBranchStatement);
  }
}


void Vectorizer::_CreateLocalMaskComputation(AST::ScopePtr spParentScope, AST::BaseClasses::ExpressionPtr spCondition, string strLocalMaskName, string strGlobalMaskName, bool bExclusiveBranches)
{
  typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType  ArithmeticOperatorType;

  if (! spCondition->IsVectorized())
  {
    spCondition = AST::VectorSupport::BroadCast::Create( spCondition );
  }

  // Initialize with the global mask
  spParentScope->AddChild( AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create(strLocalMaskName),
                                                                         AST::Expressions::Identifier::Create(strGlobalMaskName) ) );

  // Assign the condition to the local mask and mask this with the global mask
  spParentScope->AddChild( AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create(strLocalMaskName), spCondition,
                                                                         AST::Expressions::Identifier::Create(strGlobalMaskName) ) );

  // Update global mask
  ArithmeticOperatorType eOpType = bExclusiveBranches ? ArithmeticOperatorType::BitwiseXOr : ArithmeticOperatorType::BitwiseAnd;
  AST::Expressions::ArithmeticOperatorPtr spGlobalUpdate = AST::Expressions::ArithmeticOperator::Create( eOpType, AST::Expressions::Identifier::Create(strGlobalMaskName),
                                                                                                         AST::Expressions::Identifier::Create(strLocalMaskName) );

  spParentScope->AddChild( AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create(strGlobalMaskName), spGlobalUpdate ) );
}

void Vectorizer::_CreateVectorizedConditionalBranch(AST::ScopePtr spParentScope, AST::ScopePtr spBranchScope, string strMaskName)
{
  typedef AST::VectorSupport::CheckActiveElements::CheckType  VectorCheckType;

  if (! spBranchScope->IsEmpty())
  {
    // Create new branching statement
    AST::ControlFlow::BranchingStatementPtr spBranchingStatement = AST::ControlFlow::BranchingStatement::Create();
    spParentScope->AddChild(spBranchingStatement);

    // Create new conditional branch
    AST::VectorSupport::CheckActiveElementsPtr  spCheckMask = AST::VectorSupport::CheckActiveElements::Create( VectorCheckType::Any, AST::Expressions::Identifier::Create(strMaskName) );
    AST::ControlFlow::ConditionalBranchPtr      spNewBranch = AST::ControlFlow::ConditionalBranch::Create( spCheckMask );
    spBranchingStatement->AddConditionalBranch(spNewBranch);

    spNewBranch->GetBody()->ImportScope(spBranchScope);
  }
}

void Vectorizer::_FlattenSubExpression(const string &crstrTempVarNameRoot, AST::BaseClasses::ExpressionPtr spSubExpression)
{
  if (! spSubExpression->IsSubExpression())
  {
    return; // Nothing to do, expression is not a sub-expression
  }

  AST::BaseClasses::ExpressionPtr spParentExpr  = spSubExpression->GetParent()->CastToType<AST::BaseClasses::Expression>();
  AST::ScopePosition              SubExprPos    = spSubExpression->GetScopePosition();

  string strTempVarName = VASTBuilder::GetNextFreeVariableName( SubExprPos.GetScope(), crstrTempVarNameRoot );

  AST::BaseClasses::TypeInfo VarInfo = spSubExpression->GetResultType();
  VarInfo.SetConst(true);
  SubExprPos.GetScope()->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create(strTempVarName, VarInfo, spSubExpression->IsVectorized()) );

  spParentExpr->SetSubExpression( spSubExpression->GetParentIndex(), AST::Expressions::Identifier::Create(strTempVarName) );

  SubExprPos.GetScope()->InsertChild( SubExprPos.GetChildIndex(), AST::Expressions::AssignmentOperator::Create(AST::Expressions::Identifier::Create(strTempVarName), spSubExpression) );
}

AST::BaseClasses::VariableInfoPtr Vectorizer::_GetAssigneeInfo(AST::Expressions::AssignmentOperatorPtr spAssignment)
{
  if (! spAssignment)
  {
    throw InternalErrors::NullPointerException("spAssignment");
  }

  // Step through to the leaf node of the left hand side of the assignment expression
  AST::BaseClasses::ExpressionPtr spValueExpression = spAssignment;
  while (spValueExpression->GetSubExpressionCount() != static_cast<IndexType>(0))
  {
    spValueExpression = spValueExpression->GetSubExpression(0);
    if (! spValueExpression)
    {
      throw InternalErrors::NullPointerException("spValueExpression");
    }
  }

  // Check if the assignee is an identifier
  if (spValueExpression->IsType<AST::Expressions::Identifier>())
  {
    // Fetch the variable info object which belongs to this identifier
    AST::Expressions::IdentifierPtr   spIdentifier    = spValueExpression->CastToType<AST::Expressions::Identifier>();
    AST::BaseClasses::VariableInfoPtr spVariableInfo  = spIdentifier->LookupVariableInfo();

    if (! spVariableInfo)
    {
      throw InternalErrorException(string("Could not find variable info for identifier: ") + spIdentifier->GetName());
    }

    return spVariableInfo;
  }
  else
  {
    throw InternalErrorException("Expected an identifier expression!");
  }
}



AST::FunctionDeclarationPtr Vectorizer::ConvertClangFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration)
{
//VASTBuilder().Import(pFunctionDeclaration);

  return VASTBuilder().BuildFunctionDecl(pFunctionDeclaration);
}

::clang::FunctionDecl* Vectorizer::ConvertVASTFunctionDecl(AST::FunctionDeclarationPtr spVASTFunction, const size_t cszVectorWidth, ::clang::ASTContext &rASTContext, bool bUnrollVectorLoops)
{
  VASTExportArray Exporter(cszVectorWidth, rASTContext);

  return Exporter.ExportVASTFunction(spVASTFunction, bUnrollVectorLoops);
}



void Vectorizer::DumpVASTNodeToXML(AST::BaseClasses::NodePtr spVastNode, string strXmlFilename)
{
  if (! spVastNode)
  {
    throw InternalErrors::NullPointerException("spVastNode");
  }

  std::ofstream XmlStream(strXmlFilename);

  XmlStream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  XmlStream << spVastNode->DumpToXML(0);

  XmlStream.flush();
  XmlStream.close();
}


void Vectorizer::RebuildControlFlow(AST::FunctionDeclarationPtr spFunction)
{
  typedef map< AST::BaseClasses::NodePtr, list< string > >      ControlMaskMapType;

  const AST::BaseClasses::TypeInfo  MaskTypeInfo(AST::BaseClasses::TypeInfo::KnownTypes::Bool, true, false);
  ControlMaskMapType                mapControlMasks;


  // Separate mixed branching statements into scalar and vectorized branching statements
  SeparateBranchingStatements(spFunction);


  // Find all return statements and create a function control mask if required
  Transformations::FindNodes< AST::ControlFlow::ReturnStatement >   ReturnStmtFinder(Transformations::DirectionType::TopDown);
  {
    Transformations::Run(spFunction, ReturnStmtFinder);

    for each (auto itReturnStmt in ReturnStmtFinder.lstFoundNodes)
    {
      if (itReturnStmt->IsVectorized())
      {
        // Found a return statement in a vectorized context => Create a function control mask
        string strFunctionMaskName = VASTBuilder::GetNextFreeVariableName(spFunction, "_mask_function");

        spFunction->GetBody()->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create( strFunctionMaskName, MaskTypeInfo, true ) );
        mapControlMasks[spFunction].push_front(strFunctionMaskName);


        // Declare ALL vectorized assignments as masked assignments
        Transformations::FindAssignments  AssignmentFinder;
        Transformations::Run( spFunction, AssignmentFinder );

        for (auto itAssignment : AssignmentFinder.lstFoundNodes)
        {
          if (itAssignment->IsVectorized())
          {
            itAssignment->SetMask( AST::Expressions::Identifier::Create( strFunctionMaskName ) );
          }
        }

        // Initialize the function control mask with as fully active
        spFunction->GetBody()->InsertChild( 0, AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create( strFunctionMaskName ),
                                                                                             AST::VectorSupport::BroadCast::Create( AST::Expressions::Constant::Create( true ) ) ) );

        break;
      }
    }
  }


  // Find all vectorized loops and branching statements in hierarchical order
  list< AST::BaseClasses::ControlFlowStatementPtr >  lstControlFlowStatements;
  {
    Transformations::FindNodes< AST::BaseClasses::ControlFlowStatement >  ControlFlowFinder( Transformations::DirectionType::TopDown );

    Transformations::Run(spFunction, ControlFlowFinder);

    for each (auto itControlFlow in ControlFlowFinder.lstFoundNodes)
    {
      bool bAddNode = false;

      if (itControlFlow->IsType<AST::ControlFlow::Loop>())
      {
        bAddNode = itControlFlow->CastToType<AST::ControlFlow::Loop>()->IsVectorized();
      }
      else if (itControlFlow->IsType<AST::ControlFlow::BranchingStatement>())
      {
        bAddNode = itControlFlow->CastToType<AST::ControlFlow::BranchingStatement>()->IsVectorized();
      }

      if (bAddNode)
      {
        lstControlFlowStatements.push_back(itControlFlow);
      }
    }
  }

  // Rebuild the control flow statements
  {
    typedef AST::VectorSupport::CheckActiveElements::CheckType    VectorCheckType;

    for each (auto itControlFlow in lstControlFlowStatements)
    {
      // Fetch the parent mask
      AST::BaseClasses::ExpressionPtr spParentMask = nullptr;
      {
        AST::BaseClasses::NodePtr spCurrentNode = itControlFlow;
        while (true)
        {
          spCurrentNode = spCurrentNode->GetParent();
          if (! spCurrentNode)
          {
            break;
          }

          auto itCurrentMaskList = mapControlMasks.find( spCurrentNode );
          if (itCurrentMaskList != mapControlMasks.end())
          {
            // Found the parent mask => Create a corresponding identifier to the first mask in this stack
            spParentMask = AST::Expressions::Identifier::Create( itCurrentMaskList->second.front() );
            break;
          }
        }

        if (! spParentMask)
        {
          // There is no parent mask => Create a fully active control mask
          spParentMask = AST::VectorSupport::BroadCast::Create(AST::Expressions::Constant::Create(true));
        }
      }


      // Definitions for the latter assignment masking
      string                                          strCurrentMaskName;
      list< AST::Expressions::AssignmentOperatorPtr > lstInternalAssignments;


      // Rebuild the corresponding control flow statement
      if (itControlFlow->IsType<AST::ControlFlow::Loop>())
      {
        AST::ControlFlow::LoopPtr spLoop      = itControlFlow->CastToType<AST::ControlFlow::Loop>();
        AST::ScopePtr             spLoopBody = spLoop->GetBody();

        // Find all internal vectorized assignments for latter masking
        {
          Transformations::FindAssignments  AssignmentFinder;
          Transformations::Run( spLoopBody, AssignmentFinder );

          for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
          {
            if (itAssignment->IsVectorized())
            {
              lstInternalAssignments.push_back( itAssignment );
            }
          }
        }

        // Create the control masks
        string strGlobalMaskName, strLocalMaskName;
        {
          AST::ScopePosition  LoopScopePos      = spLoop->GetScopePosition();
          AST::ScopePtr       spLoopParentScope = LoopScopePos.GetScope();

          strGlobalMaskName = VASTBuilder::GetNextFreeVariableName(spLoopParentScope, "_mask_loop_global");

          spLoopParentScope->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create(strGlobalMaskName, MaskTypeInfo, true) );

          // Create global mask assignment
          spLoopParentScope->InsertChild(LoopScopePos.GetChildIndex(), AST::Expressions::AssignmentOperator::Create(AST::Expressions::Identifier::Create(strGlobalMaskName), spParentMask));

          // Insert the global control mask into the map
          mapControlMasks[spLoop].push_front(strGlobalMaskName);


          // Check if a local control mask is required
          bool bUseLocalMask = false;
          {
            Transformations::FindNodes< AST::ControlFlow::LoopControlStatement >  LoopControlFinder( Transformations::DirectionType::TopDown );
            Transformations::Run(spLoopBody, LoopControlFinder);

            // The local control mask is only required if "continue" statements are present for this loop
            for each (auto itLoopControl in LoopControlFinder.lstFoundNodes)
            {
              if ((itLoopControl->GetControlledLoop() == spLoop) && (itLoopControl->GetControlType() == AST::ControlFlow::LoopControlStatement::LoopControlType::Continue))
              {
                bUseLocalMask = true;
                break;
              }
            }
          }

          // Create the local control mask if required
          if (bUseLocalMask)
          {
            strLocalMaskName = VASTBuilder::GetNextFreeVariableName(spLoopParentScope, "_mask_loop_local");

            spLoopBody->AddVariableDeclaration(AST::BaseClasses::VariableInfo::Create(strLocalMaskName, MaskTypeInfo, true));

            mapControlMasks[spLoop].push_front(strLocalMaskName);

            strCurrentMaskName = strLocalMaskName;
          }
          else
          {
            strLocalMaskName    = "";
            strCurrentMaskName  = strGlobalMaskName;
          }
        }

        // Rebuild the loop
        if (spLoop->GetLoopType() == AST::ControlFlow::Loop::LoopType::TopControlled)
        {
          if (strLocalMaskName.empty())
          {
            // Only global control mask is used
            AST::Expressions::AssignmentOperatorPtr spGlobalMaskUpdate = AST::Expressions::AssignmentOperator::Create( AST::Expressions::Identifier::Create(strGlobalMaskName),
                                                                                                                       spLoop->GetCondition(),
                                                                                                                       AST::Expressions::Identifier::Create(strGlobalMaskName) );

            spLoopBody->InsertChild(0, spGlobalMaskUpdate);
          }
          else
          {
            // Local control is used
            AST::ScopePtr spTempScope = AST::Scope::Create();
            _CreateLocalMaskComputation(spTempScope, spLoop->GetCondition(), strLocalMaskName, strGlobalMaskName, false);

            spLoopBody->ImportVariableDeclarations(spTempScope);

            for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < spTempScope->GetChildCount(); ++iChildIdx)
            {
              spLoopBody->InsertChild(iChildIdx, spTempScope->GetChild(iChildIdx));
            }
          }
        }
        else
        {
          throw InternalErrorException("Only top controlled VAST loops can be vectorized => please rewrite the kernel code!");
        }

        spLoop->SetCondition( AST::VectorSupport::CheckActiveElements::Create(VectorCheckType::Any, AST::Expressions::Identifier::Create(strGlobalMaskName)) );
      }
      else if (itControlFlow->IsType<AST::ControlFlow::BranchingStatement>())
      {
        AST::ControlFlow::BranchingStatementPtr spBranchingStatement  = itControlFlow->CastToType<AST::ControlFlow::BranchingStatement>();

        // Find all internal vectorized assignments for latter masking
        {
          Transformations::FindAssignments  AssignmentFinder;
          Transformations::Run(spBranchingStatement, AssignmentFinder);

          for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
          {
            if (itAssignment->IsVectorized())
            {
              lstInternalAssignments.push_back( itAssignment );
            }
          }
        }

        // Create the control masks
        string strGlobalMaskName, strLocalMaskName;
        AST::ScopePtr spBranchingScope = AST::Scope::Create();
        {
          AST::ScopePosition BranchingScopePos = spBranchingStatement->GetScopePosition();
          BranchingScopePos.GetScope()->SetChild( BranchingScopePos.GetChildIndex(), spBranchingScope );

          strGlobalMaskName = VASTBuilder::GetNextFreeVariableName(spBranchingScope, "_mask_branch_global");
          strLocalMaskName  = VASTBuilder::GetNextFreeVariableName(spBranchingScope, "_mask_branch_local");

          spBranchingScope->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create(strGlobalMaskName, MaskTypeInfo, true) );
          spBranchingScope->AddVariableDeclaration( AST::BaseClasses::VariableInfo::Create(strLocalMaskName,  MaskTypeInfo, true) );

          // Create global mask assignment
          spBranchingScope->AddChild( AST::Expressions::AssignmentOperator::Create(AST::Expressions::Identifier::Create(strGlobalMaskName), spParentMask) );

          // Insert the control masks into the map
          mapControlMasks[ spBranchingScope ].push_front( strGlobalMaskName );
          mapControlMasks[ spBranchingScope ].push_front( strLocalMaskName );

          strCurrentMaskName = strLocalMaskName;
        }


        // Convert all conditional branches
        for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < spBranchingStatement->GetConditionalBranchesCount(); ++iBranchIdx)
        {
          AST::ControlFlow::ConditionalBranchPtr spBranch = spBranchingStatement->GetConditionalBranch(iBranchIdx);

          // Add local mask assignment
          _CreateLocalMaskComputation( spBranchingScope, spBranch->GetCondition(), strLocalMaskName, strGlobalMaskName, true );

          // Add a single branch branching statement
          _CreateVectorizedConditionalBranch(spBranchingScope, spBranch->GetBody(), strLocalMaskName);
        }

        // Convert default branch
        _CreateVectorizedConditionalBranch(spBranchingScope, spBranchingStatement->GetDefaultBranch(), strGlobalMaskName);
      }


      // Mask all internal vectorized assignments
      for each (auto itAssignment in lstInternalAssignments)
      {
        itAssignment->SetMask( AST::Expressions::Identifier::Create( strCurrentMaskName ) );
      }
    }
  }


  // Convert loop control statements
  {
    Transformations::FindNodes< AST::ControlFlow::LoopControlStatement >  LoopControlFinder;
    Transformations::Run(spFunction, LoopControlFinder);

    for each (auto itLoopControl in LoopControlFinder.lstFoundNodes)
    {
      AST::ControlFlow::LoopPtr spControlledLoop = itLoopControl->GetControlledLoop();

      // Find the affected control masks of this loop control statement
      list< string >  lstAffectedControlMasks;
      {
        AST::BaseClasses::NodePtr spCurrentNode = itLoopControl;
        while (true)
        {
          spCurrentNode = spCurrentNode->GetParent();

          const bool cbIsControlledLoop = (spCurrentNode == spControlledLoop);

          auto itCurrentMaskList = mapControlMasks.find( spCurrentNode );
          if (itCurrentMaskList != mapControlMasks.end())
          {
            if (cbIsControlledLoop && (itLoopControl->GetControlType() == AST::ControlFlow::LoopControlStatement::LoopControlType::Continue))
            {
              // The continue statement affects only the local mask of the controlled loop
              lstAffectedControlMasks.push_back( itCurrentMaskList->second.front() );
            }
            else
            {
              lstAffectedControlMasks.insert( lstAffectedControlMasks.end(), itCurrentMaskList->second.begin(), itCurrentMaskList->second.end() );
            }
          }

          if (cbIsControlledLoop)
          {
            break;
          }
        }
      }

      if (lstAffectedControlMasks.empty())
      {
        // Scalar loop => nothing to do
        continue;
      }
      else
      {
        auto itLoopMaskList = mapControlMasks.find(spControlledLoop);

        if (itLoopMaskList == mapControlMasks.end())
        {
          throw InternalErrorException("Found a vectorized loop control statement which controls a scalar loop => Please rewrite the kernel code!");
        }
        else if (itLoopMaskList->second.front() == lstAffectedControlMasks.front())
        {
          // Unconditional loop control statement inside a vectorized loop => statement affects ALL vector elements and thus does not need to be converted
          continue;
        }
        else
        {
          // Conditional loop control statement (affects only some vector elements) => Update all relevant control masks and remove the statement
          AST::ScopePosition  LoopControlPos  = itLoopControl->GetScopePosition();
          IndexType           iCurrentIdx     = LoopControlPos.GetChildIndex();

          // Remove the loop control statements
          LoopControlPos.GetScope()->RemoveChild(iCurrentIdx);

          // Add the mask updates
          string strConditionMask = lstAffectedControlMasks.front();
          for each (auto itMaskName in lstAffectedControlMasks)
          {
            typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType  ArithmeticOperatorType;

            AST::Expressions::ArithmeticOperatorPtr spUnsetOp = AST::Expressions::ArithmeticOperator::Create( ArithmeticOperatorType::BitwiseXOr,
                                                                                                              AST::Expressions::Identifier::Create(itMaskName),
                                                                                                              AST::Expressions::Identifier::Create(strConditionMask) );

            LoopControlPos.GetScope()->InsertChild( iCurrentIdx++, AST::Expressions::AssignmentOperator::Create(AST::Expressions::Identifier::Create(itMaskName), spUnsetOp) );
          }
        }
      }
    }
  }


  // Convert return statements
  if (mapControlMasks.find(spFunction) != mapControlMasks.end())
  {
    for each (auto itReturnStmt in ReturnStmtFinder.lstFoundNodes)
    {
      // Find the affected control masks of this loop control statement
      list< string >  lstAffectedControlMasks;
      {
        AST::BaseClasses::NodePtr spCurrentNode = itReturnStmt;
        while (true)
        {
          spCurrentNode = spCurrentNode->GetParent();
          if (! spCurrentNode)
          {
            break;
          }

          auto itCurrentMaskList = mapControlMasks.find(spCurrentNode);
          if (itCurrentMaskList != mapControlMasks.end())
          {
            lstAffectedControlMasks.insert(lstAffectedControlMasks.end(), itCurrentMaskList->second.begin(), itCurrentMaskList->second.end());
          }
        }
      }

      if (lstAffectedControlMasks.empty())
      {
        // Unconditional return statement => nothing to do
        continue;
      }
      else if (lstAffectedControlMasks.front() != mapControlMasks[spFunction].front())
      {
        // Found a return statement inside a conditional context => Eliminate the currently active control mask from the WHOLE control mask stack
        AST::ScopePosition  ReturnPos   = itReturnStmt->GetScopePosition();
        IndexType           iCurrentIdx = ReturnPos.GetChildIndex();

        // Remove the return statement
        ReturnPos.GetScope()->RemoveChild(iCurrentIdx);

        // Add the mask updates
        string strConditionMask = lstAffectedControlMasks.front();
        for each (auto itMaskName in lstAffectedControlMasks)
        {
          typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType  ArithmeticOperatorType;

          AST::Expressions::ArithmeticOperatorPtr spUnsetOp = AST::Expressions::ArithmeticOperator::Create( ArithmeticOperatorType::BitwiseXOr,
                                                                                                            AST::Expressions::Identifier::Create(itMaskName),
                                                                                                            AST::Expressions::Identifier::Create(strConditionMask) );

          ReturnPos.GetScope()->InsertChild( iCurrentIdx++, AST::Expressions::AssignmentOperator::Create(AST::Expressions::Identifier::Create(itMaskName), spUnsetOp) );
        }
      }
    }
  }
}

void Vectorizer::RebuildDataFlow(AST::FunctionDeclarationPtr spFunction, bool bEnsureMonoTypeVectorExpressions)
{
  Transformations::Run( spFunction, Transformations::RemoveImplicitConversions() );

  Transformations::Run( spFunction, Transformations::InsertRequiredConversions() );

  RemoveUnnecessaryConversions( spFunction );

  Transformations::Run(spFunction, Transformations::InsertRequiredBroadcasts());

  if (bEnsureMonoTypeVectorExpressions)
  {
    // Remove all vector conversion expressions from all expressions except direct assignments => All vector expressions will have only a single vector type now
    {
      Transformations::FindNodes< AST::Expressions::Conversion >  ConversionFinder( Transformations::DirectionType::BottomUp );
      Transformations::Run( spFunction, ConversionFinder );

      for each (auto itConversion in ConversionFinder.lstFoundNodes)
      {
        if ( itConversion->IsVectorized() && (! itConversion->GetParent()->IsType<AST::Expressions::AssignmentOperator>()) )
        {
          // Extract conversion into an own assignment to a temporary variable
          _FlattenSubExpression( VASTBuilder::GetTemporaryNamePrefix() + string("_conv"), itConversion );

          // If the sub-expression of the conversion is not a leaf expression (i.e. identifier, constant etc.) extract it too
          AST::BaseClasses::ExpressionPtr spSubExpression = itConversion->GetSubExpression();
          if ( spSubExpression && (! spSubExpression->IsLeafNode()) )
          {
            _FlattenSubExpression( VASTBuilder::GetTemporaryNamePrefix() + string("_conv_sub"), spSubExpression );
          }
        }
      }
    }

    // Re-arrange broadcast expressions => avoid unnecessary computations
    {
      Transformations::FindNodes< AST::VectorSupport::BroadCast >  BroadCastFinder(Transformations::DirectionType::BottomUp);
      Transformations::Run( spFunction, BroadCastFinder );

      for each (auto itBroadCast in BroadCastFinder.lstFoundNodes)
      {
        AST::BaseClasses::ExpressionPtr spSubExpression = itBroadCast->GetSubExpression();
        if ( spSubExpression && (! spSubExpression->IsLeafNode()) )
        {
          _FlattenSubExpression(VASTBuilder::GetTemporaryNamePrefix() + string("_broadcast"), spSubExpression);
        }
      }
    }
  }
}


void Vectorizer::VectorizeFunction(AST::FunctionDeclarationPtr spFunction)
{
  typedef map< AST::BaseClasses::VariableInfoPtr, std::list< AST::BaseClasses::ExpressionPtr > >   VariableDependencyMapType;

  if (! spFunction)
  {
    throw InternalErrors::NullPointerException("spFunction");
  }

  VariableDependencyMapType mapVariableDependencies;

  // Find all asignment expression (they express direct variable dependencies)
  {
    Transformations::FindAssignments AssignmentFinder;

    Transformations::Run(spFunction, AssignmentFinder);

    for each (auto itAssignment in AssignmentFinder.lstFoundNodes)
    {
      mapVariableDependencies[ _GetAssigneeInfo(itAssignment) ].push_back( itAssignment->GetRHS() );
    }
  }

  // Find all loop internal assignments (the vectorization of these assignments also depends on the loop condition)
  {
    Transformations::FindLoopInternalAssignments LoopAssignmentFinder;

    Transformations::Run(spFunction, LoopAssignmentFinder);

    for each (auto itCondAssignment in LoopAssignmentFinder.mapConditionalAssignments)
    {
      AST::BaseClasses::VariableInfoPtr spVariableInfo = _GetAssigneeInfo(itCondAssignment.first);

      for each (auto itCondition in itCondAssignment.second)
      {
        mapVariableDependencies[spVariableInfo].push_back(itCondition);
      }
    }
  }

  // Find all assignments inside all conditional branching statement (the vectorization of these assignments also depends on the branching condition cascade)
  {
    Transformations::FindBranchingInternalAssignments BranchAssignmentFinder;

    Transformations::Run(spFunction, BranchAssignmentFinder);

    for each (auto itCondAssignment in BranchAssignmentFinder.mapConditionalAssignments)
    {
      AST::BaseClasses::VariableInfoPtr spVariableInfo = _GetAssigneeInfo(itCondAssignment.first);

      for each (auto itCondition in itCondAssignment.second)
      {
        mapVariableDependencies[spVariableInfo].push_back(itCondition);
      }
    }
  }


  // Continue to mark dependent variables as vectorized until nothing is changing anymore
  bool bChanged = true;
  while (bChanged)
  {
    bChanged = false;

    for each (auto itEntry in mapVariableDependencies)
    {
      if (! itEntry.first->GetVectorize())
      {
        for each (auto itExpression in itEntry.second)
        {
          if (itExpression->IsVectorized())
          {
            itEntry.first->SetVectorize(true);
            bChanged = true;
            break;
          }
        }
      }
    }
  }

}



// vim: set ts=2 sw=2 sts=2 et ai:

