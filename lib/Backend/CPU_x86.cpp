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

//===--- CPU_x86.cpp - Implements the C++ code generator for x86-based CPUs. ---------===//
//
// This file implements the C++ code generator for CPUs which are based on the x86-microarchitecture.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Backend/CPU_x86.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <utility>

using namespace clang::hipacc::Backend::Vectorization;
using namespace clang::hipacc::Backend;
using namespace clang::hipacc;
using namespace clang;


//#define DUMP_INSTRUCTION_SETS 1   // Uncomment this for a complete dump of the available instruction set contents
//#define DUMP_VAST_CONTENTS    1   // Uncomment this for an incremental dump of the vectorized kernel sub-function AST during the vectorization process


// Implementation of class CPU_x86::DumpInstructionSet
ArraySubscriptExpr* CPU_x86::DumpInstructionSet::_CreateArraySubscript(DeclRefExpr *pArrayRef, int32_t iIndex)
{
  return _ASTHelper.CreateArraySubscriptExpression( pArrayRef, _ASTHelper.CreateLiteral(iIndex), pArrayRef->getType()->getAsArrayTypeUnsafe()->getElementType() );
}

StringLiteral* CPU_x86::DumpInstructionSet::_CreateElementTypeString(VectorElementTypes eElementType)
{
  return _ASTHelper.CreateStringLiteral( TypeInfo::GetTypeString(eElementType) );
}

QualType CPU_x86::DumpInstructionSet::_GetClangType(VectorElementTypes eElementType)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:  return _ASTHelper.GetASTContext().DoubleTy;
  case VectorElementTypes::Float:   return _ASTHelper.GetASTContext().FloatTy;
  case VectorElementTypes::Int8:    return _ASTHelper.GetASTContext().CharTy;
  case VectorElementTypes::UInt8:   return _ASTHelper.GetASTContext().UnsignedCharTy;
  case VectorElementTypes::Int16:   return _ASTHelper.GetASTContext().ShortTy;
  case VectorElementTypes::UInt16:  return _ASTHelper.GetASTContext().UnsignedShortTy;
  case VectorElementTypes::Int32:   return _ASTHelper.GetASTContext().IntTy;
  case VectorElementTypes::UInt32:  return _ASTHelper.GetASTContext().UnsignedIntTy;
  case VectorElementTypes::Int64:   return _ASTHelper.GetASTContext().LongLongTy;
  case VectorElementTypes::UInt64:  return _ASTHelper.GetASTContext().UnsignedLongLongTy;
  default:                          throw InternalErrorException( "CPU_x86::DumpInstructionSet::_GetClangType() -> Unsupported vector element type detected!" );
  }
}

FunctionDecl* CPU_x86::DumpInstructionSet::_DumpInstructionSet(Vectorization::InstructionSetBasePtr spInstructionSet, std::string strFunctionName)
{
  #define DUMP_INSTR(__container, __instr)  try{ __container.push_back(__instr); } catch (std::exception &e) { __container.push_back( _ASTHelper.CreateStringLiteral(e.what()) ); }

  typedef std::map<VectorElementTypes, DeclRefExpr*> VectorDeclRefMapType;
  
  std::list<VectorElementTypes> lstSupportedElementTypes;
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Double)) lstSupportedElementTypes.push_back(VectorElementTypes::Double);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Float))  lstSupportedElementTypes.push_back(VectorElementTypes::Float);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Int8))   lstSupportedElementTypes.push_back(VectorElementTypes::Int8);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::UInt8))  lstSupportedElementTypes.push_back(VectorElementTypes::UInt8);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Int16))  lstSupportedElementTypes.push_back(VectorElementTypes::Int16);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::UInt16)) lstSupportedElementTypes.push_back(VectorElementTypes::UInt16);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Int32))  lstSupportedElementTypes.push_back(VectorElementTypes::Int32);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::UInt32)) lstSupportedElementTypes.push_back(VectorElementTypes::UInt32);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::Int64))  lstSupportedElementTypes.push_back(VectorElementTypes::Int64);
  if (spInstructionSet->IsElementTypeSupported(VectorElementTypes::UInt64)) lstSupportedElementTypes.push_back(VectorElementTypes::UInt64);


  FunctionDecl *pFunctionDecl = _ASTHelper.CreateFunctionDeclaration(strFunctionName, _ASTHelper.GetASTContext().VoidTy, ClangASTHelper::StringVectorType(), ClangASTHelper::QualTypeVectorType());

  ClangASTHelper::StatementVectorType vecBody;

  // Create variable declarations
  VectorDeclRefMapType   mapVectorArrayDecls;
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("Vector declarations") );

    for (auto itElementType : lstSupportedElementTypes)
    {
      const size_t cszArraySize = std::max( spInstructionSet->GetVectorWidthBytes() / spInstructionSet->GetVectorElementCount(itElementType), static_cast<size_t>(2) );
      QualType qtDeclType       = _ASTHelper.GetConstantArrayType( spInstructionSet->GetVectorType(itElementType), cszArraySize );
      VarDecl *pDecl            = _ASTHelper.CreateVariableDeclaration( pFunctionDecl, std::string("mm") + TypeInfo::GetTypeString(itElementType), qtDeclType, nullptr );

      mapVectorArrayDecls[itElementType] = _ASTHelper.CreateDeclarationReferenceExpression( pDecl );

      vecBody.push_back( _ASTHelper.CreateDeclarationStatement(pDecl) );
    }

    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump arithmetic operators
  if (_uiDumpFlags & DF_Arithmetic)
  {
    typedef Vectorization::AST::Expressions::ArithmeticOperator   ArithmeticOperator;
    typedef ArithmeticOperator::ArithmeticOperatorType            ArithmeticOperatorType;

    ArithmeticOperatorType aeArithOps[] = { ArithmeticOperatorType::Add,        ArithmeticOperatorType::BitwiseAnd, ArithmeticOperatorType::BitwiseOr,  ArithmeticOperatorType::BitwiseXOr,
                                            ArithmeticOperatorType::Divide,     ArithmeticOperatorType::Modulo,     ArithmeticOperatorType::Multiply,   ArithmeticOperatorType::ShiftLeft,
                                            ArithmeticOperatorType::ShiftRight, ArithmeticOperatorType::Subtract };


    vecBody.push_back( _ASTHelper.CreateStringLiteral("ArithmeticOperator") );

    ClangASTHelper::StatementVectorType vecArithmeticOperators;

    for (auto eCurrentOp : aeArithOps)
    {
      vecArithmeticOperators.push_back( _ASTHelper.CreateStringLiteral( ArithmeticOperator::GetOperatorTypeString(eCurrentOp) ) );

      ClangASTHelper::StatementVectorType vecCurrentOp;

      for (auto itElementType : lstSupportedElementTypes)
      {
        auto itArrayDecl = mapVectorArrayDecls[itElementType];

        vecCurrentOp.push_back( _CreateElementTypeString(itElementType) );
        DUMP_INSTR( vecCurrentOp, spInstructionSet->ArithmeticOperator( itElementType, eCurrentOp, _CreateArraySubscript(itArrayDecl, 0), _CreateArraySubscript(itArrayDecl, 1) ) );
      }

      vecArithmeticOperators.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentOp) );
      vecArithmeticOperators.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecArithmeticOperators) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump blend functions
  if (_uiDumpFlags & DF_Blend)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("BlendVectors") );

    ClangASTHelper::StatementVectorType vecBlendVectors;

    // Create mask declarations
    vecBlendVectors.push_back( _ASTHelper.CreateStringLiteral("MaskDeclarations") );
    VectorDeclRefMapType mapMaskDecls;
    for (auto itElementType : lstSupportedElementTypes)
    {
      VarDecl *pMaskDecl = _ASTHelper.CreateVariableDeclaration( pFunctionDecl, std::string("mmMask") + TypeInfo::GetTypeString(itElementType), spInstructionSet->GetVectorType(itElementType), nullptr );

      mapMaskDecls[ itElementType ] = _ASTHelper.CreateDeclarationReferenceExpression( pMaskDecl );

      vecBlendVectors.push_back( _ASTHelper.CreateDeclarationStatement(pMaskDecl) );
    }
    vecBlendVectors.push_back( _ASTHelper.CreateStringLiteral("") );

    for (auto itElementType : lstSupportedElementTypes)
    {
      vecBlendVectors.push_back( _CreateElementTypeString(itElementType) );

      auto itArrayDecl = mapVectorArrayDecls[itElementType];

      DUMP_INSTR( vecBlendVectors, spInstructionSet->BlendVectors( itElementType, mapMaskDecls[ itElementType ], _CreateArraySubscript(itArrayDecl, 0), _CreateArraySubscript(itArrayDecl, 1) ) );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecBlendVectors) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump broadcasts
  if (_uiDumpFlags & DF_BroadCast)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("BroadCasts") );

    ClangASTHelper::StatementVectorType vecBroadCasts;

    for (auto itElementType : lstSupportedElementTypes)
    {
      vecBroadCasts.push_back( _CreateElementTypeString(itElementType) );

      DUMP_INSTR( vecBroadCasts, spInstructionSet->BroadCast( itElementType, _ASTHelper.CreateLiteral(1) ) );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecBroadCasts) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump check active elements
  if (_uiDumpFlags & DF_CheckActive)
  {
    typedef Vectorization::AST::VectorSupport::CheckActiveElements    CheckActiveElements;
    typedef CheckActiveElements::CheckType                            CheckType;

    CheckType aeCheckTypes[] = { CheckType::All, CheckType::Any, CheckType::None };

    vecBody.push_back( _ASTHelper.CreateStringLiteral("CheckActiveElements") );

    ClangASTHelper::StatementVectorType vecCheckActiveElements;

    for (auto eCheckType : aeCheckTypes)
    {
      vecCheckActiveElements.push_back( _ASTHelper.CreateStringLiteral( CheckActiveElements::GetCheckTypeString(eCheckType) ) );

      ClangASTHelper::StatementVectorType vecCurrentCheck;

      for (auto itElementType : lstSupportedElementTypes)
      {
        vecCurrentCheck.push_back( _CreateElementTypeString(itElementType) );

        auto itArrayDecl = mapVectorArrayDecls[itElementType];

        DUMP_INSTR( vecCurrentCheck, spInstructionSet->CheckActiveElements( itElementType, eCheckType, _CreateArraySubscript(itArrayDecl, 0) ) );
      }

      vecCheckActiveElements.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentCheck) );
      vecCheckActiveElements.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecCheckActiveElements) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump conversion
  if (_uiDumpFlags & DF_Convert)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("ConvertVector") );

    ClangASTHelper::StatementVectorType vecConvertVector;

    for (int i = 0; i <= 1; ++i)
    {
      const bool cbMaskConversion = (i == 0);

      vecConvertVector.push_back( _ASTHelper.CreateStringLiteral( cbMaskConversion ? "Mask conversion" : "Vector conversion" ) );

      ClangASTHelper::StatementVectorType vecCurrentConvertFrom;

      for (auto itElementTypeFrom : lstSupportedElementTypes)
      {
        vecCurrentConvertFrom.push_back( _ASTHelper.CreateStringLiteral( std::string("Convert ") + TypeInfo::GetTypeString(itElementTypeFrom) + std::string(" to ...") ) );

        ClangASTHelper::StatementVectorType vecCurrentConvertTo;

        auto itArrayDecl = mapVectorArrayDecls[itElementTypeFrom];

        for (auto itElementTypeTo : lstSupportedElementTypes)
        {
          vecCurrentConvertTo.push_back( _CreateElementTypeString(itElementTypeTo) );

          const size_t cszSizeFrom  = TypeInfo::GetTypeSize( itElementTypeFrom );
          const size_t cszSizeTo    = TypeInfo::GetTypeSize( itElementTypeTo );

          if (cszSizeFrom == cszSizeTo)
          {
            Expr *pVectorRef = _CreateArraySubscript(itArrayDecl, 0);

            if (cbMaskConversion)
            {
              DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertMaskSameSize( itElementTypeFrom, itElementTypeTo, pVectorRef ) );
            }
            else
            {
              DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertVectorSameSize( itElementTypeFrom, itElementTypeTo, pVectorRef ) );
            }
          }
          else if (cszSizeFrom < cszSizeTo)
          {
            Expr *pVectorRef = _CreateArraySubscript(itArrayDecl, 0);

            for (uint32_t uiGroupIndex = 0; uiGroupIndex < static_cast<uint32_t>(cszSizeTo / cszSizeFrom); ++uiGroupIndex)
            {
              if (cbMaskConversion)
              {
                DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertMaskUp( itElementTypeFrom, itElementTypeTo, pVectorRef, uiGroupIndex ) );
              }
              else
              {
                DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertVectorUp( itElementTypeFrom, itElementTypeTo, pVectorRef, uiGroupIndex ) );
              }
            }
          }
          else
          {
            ClangASTHelper::ExpressionVectorType vecElements;
            for (int32_t iElem = 0; iElem < static_cast<int32_t>(cszSizeFrom / cszSizeTo); ++iElem)
            {
              vecElements.push_back( _CreateArraySubscript(itArrayDecl, iElem) );
            }

            if (cbMaskConversion)
            {
              DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertMaskDown( itElementTypeFrom, itElementTypeTo, vecElements ) );
            }
            else
            {
              DUMP_INSTR( vecCurrentConvertTo, spInstructionSet->ConvertVectorDown( itElementTypeFrom, itElementTypeTo, vecElements ) );
            }
          }
        }

        vecCurrentConvertFrom.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentConvertTo) );
        vecCurrentConvertFrom.push_back( _ASTHelper.CreateStringLiteral("") );
      }

      vecConvertVector.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentConvertFrom) );
      vecConvertVector.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecConvertVector) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump create vector
  if (_uiDumpFlags & DF_CreateVector)
  {
    vecBody.push_back(_ASTHelper.CreateStringLiteral("CreateVector"));

    ClangASTHelper::StatementVectorType vecCreateVector;

    const char *apcDumpName[] = { "Normal order", "Reversed order", "Create ones", "Create negative ones", "Create zeros" };

    for (int i = 0; i < 5; ++i)
    {
      vecCreateVector.push_back(_ASTHelper.CreateStringLiteral(apcDumpName[i]));

      ClangASTHelper::StatementVectorType vecCurrentCreate;

      for (auto itElementType : lstSupportedElementTypes)
      {
        vecCurrentCreate.push_back(_CreateElementTypeString(itElementType));

        if (i <= 1)
        {
          ClangASTHelper::ExpressionVectorType vecCreateArgs;
          for (int32_t iArgIdx = 0; iArgIdx < static_cast<int32_t>(spInstructionSet->GetVectorElementCount(itElementType)); ++iArgIdx)
          {
            vecCreateArgs.push_back(_ASTHelper.CreateIntegerLiteral(iArgIdx));
          }

          DUMP_INSTR(vecCurrentCreate, spInstructionSet->CreateVector(itElementType, vecCreateArgs, (i == 1)));
        }
        else if (i == 2)
        {
          DUMP_INSTR(vecCurrentCreate, spInstructionSet->CreateOnesVector(itElementType, false));
        }
        else if (i == 3)
        {
          DUMP_INSTR(vecCurrentCreate, spInstructionSet->CreateOnesVector(itElementType, true));
        }
        else if (i == 4)
        {
          DUMP_INSTR(vecCurrentCreate, spInstructionSet->CreateZeroVector(itElementType));
        }
      }

      vecCreateVector.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentCreate) );
      vecCreateVector.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecCreateVector) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump extract element
  if (_uiDumpFlags & DF_Extract)
  {
    vecBody.push_back(_ASTHelper.CreateStringLiteral("ExtractElement"));

    ClangASTHelper::StatementVectorType vecExtractElement;

    for (auto itElementType : lstSupportedElementTypes)
    {
      vecExtractElement.push_back( _CreateElementTypeString(itElementType) );

      Expr *pVectorRef = _CreateArraySubscript( mapVectorArrayDecls[ itElementType ], 0 );

      for (uint32_t uiIndex = 0; uiIndex < spInstructionSet->GetVectorElementCount(itElementType); ++uiIndex)
      {
        DUMP_INSTR( vecExtractElement, spInstructionSet->ExtractElement( itElementType, pVectorRef, uiIndex ) );
      }
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecExtractElement) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump insert element
  if (_uiDumpFlags & DF_Insert)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("InsertElement") );

    ClangASTHelper::StatementVectorType vecInsertElement;

    for (auto itElementType : lstSupportedElementTypes)
    {
      vecInsertElement.push_back( _CreateElementTypeString(itElementType) );

      Expr *pVectorRef    = _CreateArraySubscript( mapVectorArrayDecls[itElementType], 0 );
      Expr *pInsertValue  = _ASTHelper.CreateIntegerLiteral( 2 );

      for (uint32_t uiIndex = 0; uiIndex < spInstructionSet->GetVectorElementCount(itElementType); ++uiIndex)
      {
        DUMP_INSTR( vecInsertElement, spInstructionSet->InsertElement( itElementType, pVectorRef, pInsertValue, uiIndex ) );
      }
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecInsertElement) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump memory transfers
  if (_uiDumpFlags & DF_MemoryTransfers)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("MemoryTransfers") );

    ClangASTHelper::StatementVectorType vecMemoryTransfers;

    VectorDeclRefMapType  mapPointerDecls;

    // Create pointer declarations
    vecMemoryTransfers.push_back( _ASTHelper.CreateStringLiteral("Pointer declarations") );
    for (auto itElementType : lstSupportedElementTypes)
    {
      QualType qtPointerType = _ASTHelper.GetPointerType( _GetClangType(itElementType) );

      VarDecl *pPointerDecl = _ASTHelper.CreateVariableDeclaration( pFunctionDecl, std::string("p") + TypeInfo::GetTypeString(itElementType), qtPointerType, nullptr );

      mapPointerDecls[ itElementType ] = _ASTHelper.CreateDeclarationReferenceExpression( pPointerDecl );

      vecMemoryTransfers.push_back( _ASTHelper.CreateDeclarationStatement(pPointerDecl) );
    }
    vecMemoryTransfers.push_back( _ASTHelper.CreateStringLiteral("") );


    const char *apcDumpName[] = { "LoadVector", "StoreVector", "StoreVectorMasked" };

    for (int i = 0; i <= 2; ++i)
    {
      vecMemoryTransfers.push_back( _ASTHelper.CreateStringLiteral(apcDumpName[i]) );

      ClangASTHelper::StatementVectorType vecCurrentTransfer;
      
      for (auto itElementType : lstSupportedElementTypes)
      {
        vecCurrentTransfer.push_back(_CreateElementTypeString(itElementType));

        Expr *pVectorRef  = _CreateArraySubscript( mapVectorArrayDecls[itElementType], 0 );
        Expr *pPointerRef = mapPointerDecls[itElementType];

        if (i == 0)
        {
          DUMP_INSTR( vecCurrentTransfer, spInstructionSet->LoadVector(itElementType, pPointerRef) );
        }
        else if (i == 1)
        {
          DUMP_INSTR( vecCurrentTransfer, spInstructionSet->StoreVector(itElementType, pPointerRef, pVectorRef) );
        }
        else
        {
          Expr *pMaskRef = _CreateArraySubscript(mapVectorArrayDecls[itElementType], 1);

          DUMP_INSTR( vecCurrentTransfer, spInstructionSet->StoreVectorMasked(itElementType, pPointerRef, pVectorRef, pMaskRef) );
        }
      }

      vecMemoryTransfers.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentTransfer) );
      vecMemoryTransfers.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecMemoryTransfers) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump relational operators
  if (_uiDumpFlags & DF_Relational)
  {
    typedef Vectorization::AST::Expressions::RelationalOperator   RelationalOperator;
    typedef RelationalOperator::RelationalOperatorType            RelationalOperatorType;

    RelationalOperatorType aeRelationalOps[] =  { RelationalOperatorType::Equal,     RelationalOperatorType::Greater,   RelationalOperatorType::GreaterEqual,
                                                  RelationalOperatorType::Less,      RelationalOperatorType::LessEqual, RelationalOperatorType::LogicalAnd,
                                                  RelationalOperatorType::LogicalOr, RelationalOperatorType::NotEqual };


    vecBody.push_back( _ASTHelper.CreateStringLiteral("RelationalOperator") );

    ClangASTHelper::StatementVectorType vecRelationalOperators;

    for (auto eCurrentOp : aeRelationalOps)
    {
      vecRelationalOperators.push_back( _ASTHelper.CreateStringLiteral( RelationalOperator::GetOperatorTypeString(eCurrentOp) ) );

      ClangASTHelper::StatementVectorType vecCurrentOp;

      for (auto itElementType : lstSupportedElementTypes)
      {
        auto itArrayDecl = mapVectorArrayDecls[itElementType];

        vecCurrentOp.push_back( _CreateElementTypeString(itElementType) );
        DUMP_INSTR( vecCurrentOp, spInstructionSet->RelationalOperator( itElementType, eCurrentOp, _CreateArraySubscript(itArrayDecl, 0), _CreateArraySubscript(itArrayDecl, 1) ) );
      }

      vecRelationalOperators.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentOp) );
      vecRelationalOperators.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecRelationalOperators) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump shift elements
  if (_uiDumpFlags & DF_ShiftElements)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("ShiftElements") );

    ClangASTHelper::StatementVectorType vecShiftElements;

    const char *apcDumpName[] = { "Shift left by zero", "Shift left by constant", "Shift right by zero", "Shift right by constant" };

    for (int i = 0; i <= 3; ++i)
    {
      vecShiftElements.push_back( _ASTHelper.CreateStringLiteral(apcDumpName[i]) );

      ClangASTHelper::StatementVectorType vecCurrentShift;
      
      for (auto itElementType : lstSupportedElementTypes)
      {
        vecCurrentShift.push_back( _CreateElementTypeString(itElementType) );

        Expr *pVectorRef  = _CreateArraySubscript( mapVectorArrayDecls[itElementType], 0 );

        DUMP_INSTR( vecCurrentShift, spInstructionSet->ShiftElements( itElementType, pVectorRef, i < 2, (i & 1) ? (i + 1) : 0 ) );
      }

      vecShiftElements.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentShift) );
      vecShiftElements.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecShiftElements) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump unary operators
  if (_uiDumpFlags & DF_Unary)
  {
    typedef Vectorization::AST::Expressions::UnaryOperator    UnaryOperator;
    typedef UnaryOperator::UnaryOperatorType                  UnaryOperatorType;

    UnaryOperatorType aeUnaryOps[] =  { UnaryOperatorType::AddressOf,     UnaryOperatorType::BitwiseNot,    UnaryOperatorType::LogicalNot,
                                        UnaryOperatorType::Minus,         UnaryOperatorType::Plus,          UnaryOperatorType::PostDecrement,
                                        UnaryOperatorType::PostIncrement, UnaryOperatorType::PreDecrement,  UnaryOperatorType::PreIncrement };


    vecBody.push_back( _ASTHelper.CreateStringLiteral("UnaryOperator") );

    ClangASTHelper::StatementVectorType vecUnaryOperators;

    for (auto eCurrentOp : aeUnaryOps)
    {
      vecUnaryOperators.push_back( _ASTHelper.CreateStringLiteral( UnaryOperator::GetOperatorTypeString(eCurrentOp) ) );

      ClangASTHelper::StatementVectorType vecCurrentOp;

      for (auto itElementType : lstSupportedElementTypes)
      {
        auto itArrayDecl = mapVectorArrayDecls[itElementType];

        vecCurrentOp.push_back( _CreateElementTypeString(itElementType) );
        DUMP_INSTR( vecCurrentOp, spInstructionSet->UnaryOperator( itElementType, eCurrentOp, _CreateArraySubscript(itArrayDecl, 0) ) );
      }

      vecUnaryOperators.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentOp) );
      vecUnaryOperators.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecUnaryOperators) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump vectorized memory transfers
  if (_uiDumpFlags & DF_VecMemTransfers)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("VectorizedMemoryTransfers") );

    ClangASTHelper::StatementVectorType vecVecMemTransfers;

    VectorDeclRefMapType  mapPointerDecls;

    // Create pointer declarations
    vecVecMemTransfers.push_back( _ASTHelper.CreateStringLiteral("Pointer declarations") );
    for (auto itElementType : lstSupportedElementTypes)
    {
      QualType qtPointerType = _ASTHelper.GetPointerType( _GetClangType(itElementType) );

      VarDecl *pPointerDecl = _ASTHelper.CreateVariableDeclaration( pFunctionDecl, std::string("p") + TypeInfo::GetTypeString(itElementType), qtPointerType, nullptr );

      mapPointerDecls[ itElementType ] = _ASTHelper.CreateDeclarationReferenceExpression( pPointerDecl );

      vecVecMemTransfers.push_back( _ASTHelper.CreateDeclarationStatement(pPointerDecl) );
    }
    vecVecMemTransfers.push_back( _ASTHelper.CreateStringLiteral("") );


    const char *apcDumpName[] = { "LoadVectorGathered" };

    VectorElementTypes aeIndexElementTypes[] = { VectorElementTypes::Int32, VectorElementTypes::Int64 };

    for (int i = 0; i <= 0; ++i)
    {
      vecVecMemTransfers.push_back( _ASTHelper.CreateStringLiteral(apcDumpName[i]) );

      ClangASTHelper::StatementVectorType vecCurrentTransfer;
      
      for (auto itElementType : lstSupportedElementTypes)
      {
        for (auto itIndexType : aeIndexElementTypes)
        {
          auto itIndexTypeEntry = mapVectorArrayDecls.find( itIndexType );
          if ( itIndexTypeEntry == mapVectorArrayDecls.end() )
          {
            continue;
          }

          vecCurrentTransfer.push_back( _ASTHelper.CreateStringLiteral( TypeInfo::GetTypeString(itElementType) + std::string("[ ") + TypeInfo::GetTypeString(itIndexType) + std::string(" ]") ) );

          ClangASTHelper::ExpressionVectorType vecIndexVectors;

          for (size_t szElementCount = static_cast<size_t>(0); szElementCount < spInstructionSet->GetVectorElementCount(itElementType); szElementCount += spInstructionSet->GetVectorElementCount(itIndexType))
          {
            vecIndexVectors.push_back( _CreateArraySubscript( itIndexTypeEntry->second, static_cast<int32_t>(szElementCount / spInstructionSet->GetVectorElementCount(itIndexType)) ) );
          }

          Expr *pPointerRef   = mapPointerDecls[itElementType];

          if (i == 0)
          {
            const uint32_t cuiMaxIndex = std::max( static_cast<uint32_t>(spInstructionSet->GetVectorElementCount(itIndexType) / spInstructionSet->GetVectorElementCount(itElementType)), static_cast<uint32_t>(1) );

            for (uint32_t uiGroupIndex = 0; uiGroupIndex < cuiMaxIndex; ++uiGroupIndex)
            {
              DUMP_INSTR( vecCurrentTransfer, spInstructionSet->LoadVectorGathered(itElementType, itIndexType, pPointerRef, vecIndexVectors, uiGroupIndex) );
            }
          }
        }
      }

      vecVecMemTransfers.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentTransfer) );
      vecVecMemTransfers.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecVecMemTransfers) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump special built-in functions
  if (_uiDumpFlags & DF_BuiltinFunctions)
  {
    typedef Vectorization::BuiltinFunctionsEnum   FunctionType;

    FunctionType aeFunctions[] = { FunctionType::Abs, FunctionType::Ceil, FunctionType::Floor, FunctionType::Max, FunctionType::Min, FunctionType::Sqrt };

    vecBody.push_back( _ASTHelper.CreateStringLiteral("BuiltinFunction") );

    ClangASTHelper::StatementVectorType vecBuiltinFunctions;

    for (auto eCurFunc : aeFunctions)
    {
      vecBuiltinFunctions.push_back( _ASTHelper.CreateStringLiteral( GetBuiltinFunctionTypeString(eCurFunc) ) );

      ClangASTHelper::StatementVectorType vecCurrentFunc;

      for (auto itElementType : lstSupportedElementTypes)
      {
        auto itArrayDecl = mapVectorArrayDecls[itElementType];

        for (uint32_t uiArgCount = 0; uiArgCount <= 2; ++uiArgCount)
        {
          if (spInstructionSet->IsBuiltinFunctionSupported(itElementType, eCurFunc, uiArgCount))
          {
            vecCurrentFunc.push_back( _CreateElementTypeString(itElementType) );

            ClangASTHelper::ExpressionVectorType vecArguments;
            for (uint32_t uiArg = 0; uiArg < uiArgCount; ++uiArg)
            {
              vecArguments.push_back( _CreateArraySubscript(itArrayDecl, uiArg) );
            }

            DUMP_INSTR( vecCurrentFunc, spInstructionSet->BuiltinFunction( itElementType, eCurFunc, vecArguments ) );
          }
        }
      }

      vecBuiltinFunctions.push_back( _ASTHelper.CreateCompoundStatement(vecCurrentFunc) );
      vecBuiltinFunctions.push_back( _ASTHelper.CreateStringLiteral("") );
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecBuiltinFunctions) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }

  // Dump check of single mask element
  if (_uiDumpFlags & DF_CheckMaskElement)
  {
    vecBody.push_back( _ASTHelper.CreateStringLiteral("CheckSingleMaskElement") );

    ClangASTHelper::StatementVectorType vecCheckMaskElement;

    for (auto itElementType : lstSupportedElementTypes)
    {
      vecCheckMaskElement.push_back( _CreateElementTypeString(itElementType) );

      Expr *pVectorRef = _CreateArraySubscript( mapVectorArrayDecls[ itElementType ], 0 );

      for (uint32_t uiIndex = 0; uiIndex < spInstructionSet->GetVectorElementCount(itElementType); ++uiIndex)
      {
        DUMP_INSTR( vecCheckMaskElement, spInstructionSet->CheckSingleMaskElement( itElementType, pVectorRef, uiIndex ) );
      }
    }

    vecBody.push_back( _ASTHelper.CreateCompoundStatement(vecCheckMaskElement) );
    vecBody.push_back( _ASTHelper.CreateStringLiteral("") );
  }


  pFunctionDecl->setBody( _ASTHelper.CreateCompoundStatement(vecBody) );
  return pFunctionDecl;

  #undef DUMP_INSTR
}

CPU_x86::DumpInstructionSet::DumpInstructionSet(ASTContext &rASTContext, std::string strDumpfile, InstructionSetEnum eIntrSet) : _ASTHelper(rASTContext), _uiDumpFlags(0)
{
  typedef std::pair<std::string, Vectorization::InstructionSetBasePtr> InstructionSetInfoPair;
  
  std::list<InstructionSetInfoPair> lstInstructionSets;

  switch (eIntrSet)
  {
  case InstructionSetEnum::AVX_2:     lstInstructionSets.push_front( InstructionSetInfoPair("DumpAVX_2",    Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetAVX2  >(rASTContext)) );
  case InstructionSetEnum::AVX:       lstInstructionSets.push_front( InstructionSetInfoPair("DumpAVX",      Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetAVX   >(rASTContext)) );
  case InstructionSetEnum::SSE_4_2:   lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSE_4_2",  Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSE4_2>(rASTContext)) );
  case InstructionSetEnum::SSE_4_1:   lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSE_4_1",  Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSE4_1>(rASTContext)) );
  case InstructionSetEnum::SSSE_3:    lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSSE_3",   Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSSE3 >(rASTContext)) );
  case InstructionSetEnum::SSE_3:     lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSE_3",    Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSE3  >(rASTContext)) );
  case InstructionSetEnum::SSE_2:     lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSE_2",    Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSE2  >(rASTContext)) );
  case InstructionSetEnum::SSE:       lstInstructionSets.push_front( InstructionSetInfoPair("DumpSSE",      Vectorization::InstructionSetBase::Create<Vectorization::InstructionSetSSE   >(rASTContext)) );
  default:                            break;    // Useless default branch avoiding GCC compiler warnings
  }

  // Select the requested instruction set parts
  _uiDumpFlags |= DF_Arithmetic;
  _uiDumpFlags |= DF_Blend;
  _uiDumpFlags |= DF_BroadCast;
  _uiDumpFlags |= DF_CheckActive;
  _uiDumpFlags |= DF_Convert;
  _uiDumpFlags |= DF_CreateVector;
  _uiDumpFlags |= DF_Extract;
  _uiDumpFlags |= DF_Insert;
  _uiDumpFlags |= DF_MemoryTransfers;
  _uiDumpFlags |= DF_Relational;
  _uiDumpFlags |= DF_ShiftElements;
  _uiDumpFlags |= DF_Unary;
  _uiDumpFlags |= DF_VecMemTransfers;
  _uiDumpFlags |= DF_BuiltinFunctions;
  _uiDumpFlags |= DF_CheckMaskElement;


  ClangASTHelper::FunctionDeclarationVectorType vecFunctionDecls;

  for (auto itInstrSet : lstInstructionSets)
  {
    vecFunctionDecls.push_back( _DumpInstructionSet(itInstrSet.second, itInstrSet.first) );
  }


  if (! vecFunctionDecls.empty())
  {
    std::string strErrorInfo;
    std::error_code errorCode;
    llvm::raw_fd_ostream outputStream(llvm::StringRef(strDumpfile.c_str()), errorCode, llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write); //strErrorInfo);

    for (auto itFuncDecl : vecFunctionDecls)
    {
      itFuncDecl->print( outputStream );
      outputStream << "\n\n";
    }

    outputStream.flush();
    outputStream.close();
  }
}



// Implementation of class CPU_x86::HipaccHelper
int CPU_x86::HipaccHelper::_FindKernelParamIndex(const std::string &crstrParamName)
{
  for (unsigned int i = 0; i < _pKernelFunction->getNumParams(); ++i)
  {
    if (_pKernelFunction->getParamDecl(i)->getNameAsString() == crstrParamName)
    {
      return static_cast<int>( i );
    }
  }

  return -1;
}

MemoryAccess CPU_x86::HipaccHelper::GetImageAccess(const std::string &crstrParamName)
{
  int iParamIndex = _FindKernelParamIndex(crstrParamName);

  if (iParamIndex < 0)        // Parameter not found
  {
    return UNDEFINED;
  }
  else if (iParamIndex == 0)  // First parameter is always the output image
  {
    return WRITE_ONLY;
  }
  else                        // Parameter found
  {
    ::FieldDecl*  pFieldDescriptor = _pKernel->getDeviceArgFields()[iParamIndex];

    if (_pKernel->getImgFromMapping(pFieldDescriptor) == nullptr)   // Parameter is not an image
    {
      return UNDEFINED;
    }
    else
    {
      return _pKernel->getKernelClass()->getMemAccess(pFieldDescriptor);
    }
  }
}

HipaccAccessor* CPU_x86::HipaccHelper::GetImageFromMapping(const std::string &crstrParamName)
{
  int iParamIndex = _FindKernelParamIndex(crstrParamName);

  if (iParamIndex < 0)        // Parameter not found
  {
    return nullptr;
  }
  else if (iParamIndex == 0)  // First parameter is always the output image
  {
    return _pKernel->getIterationSpace();
  }
  else                        // Parameter found
  {
    return _pKernel->getImgFromMapping(_pKernel->getDeviceArgFields()[iParamIndex] );
  }
}

HipaccMask* CPU_x86::HipaccHelper::GetMaskFromMapping(const std::string &crstrParamName)
{
  int iParamIndex = _FindKernelParamIndex(crstrParamName);

  if (iParamIndex < 0)        // Parameter not found
  {
    return nullptr;
  }
  else if (iParamIndex == 0)  // First parameter is always the output image not a mask
  {
    return nullptr;
  }
  else                        // Parameter found
  {
    return _pKernel->getMaskFromMapping(_pKernel->getDeviceArgFields()[iParamIndex]);
  }
}

::clang::DeclRefExpr* CPU_x86::HipaccHelper::GetImageParameterDecl(const std::string &crstrImageName, ImageParamType eParamType)
{
  HipaccAccessor* pAccessor = GetImageFromMapping(crstrImageName);
  if (pAccessor == nullptr)
  {
    return nullptr;
  }

  switch (eParamType)
  {
  case ImageParamType::Buffer:
    {
      int iParamIndex = _FindKernelParamIndex(crstrImageName);
      ::clang::ParmVarDecl *pParamDecl = GetKernelFunction()->getParamDecl(static_cast<unsigned int>(iParamIndex));
      return ClangASTHelper(_GetASTContext()).CreateDeclarationReferenceExpression(pParamDecl);
    }
  case ImageParamType::Width:   return pAccessor->getWidthDecl();
  case ImageParamType::Height:  return pAccessor->getHeightDecl();
  case ImageParamType::Stride:  return pAccessor->getStrideDecl();
  default:                      throw RuntimeErrorException("Unknown image parameter type!");
  }
}

::clang::Expr* CPU_x86::HipaccHelper::GetIterationSpaceLimitX()
{
  ::clang::Expr *pUpperX = _pKernel->getIterationSpace()->getWidthDecl();

  if (::clang::DeclRefExpr *pOffsetX = _pKernel->getIterationSpace()->getOffsetXDecl())
  {
    pUpperX = ASTNode::createBinaryOperator(_GetASTContext(), pUpperX, pOffsetX, BO_Add, _GetASTContext().IntTy);
  }

  return pUpperX;
}

::clang::Expr* CPU_x86::HipaccHelper::GetIterationSpaceLimitY()
{
  ::clang::Expr *pUpperY = _pKernel->getIterationSpace()->getHeightDecl();

  if (::clang::DeclRefExpr *pOffsetY = _pKernel->getIterationSpace()->getOffsetYDecl())
  {
    pUpperY = ASTNode::createBinaryOperator(_GetASTContext(), pUpperY, pOffsetY, BO_Add, _GetASTContext().IntTy);
  }

  return pUpperY;
}



// Implementation of class CPU_x86::VASTExportInstructionSet
CPU_x86::VASTExportInstructionSet::VASTExportInstructionSet(size_t VectorWidth, ASTContext &rAstContext, InstructionSetBasePtr spInstructionSet) :  BaseType(rAstContext),
                                                                                                                                                    _cVectorWidth(VectorWidth),
                                                                                                                                                    _spInstructionSet(spInstructionSet)
{
  if (! _spInstructionSet)
  {
    throw InternalErrors::NullPointerException( "spInstructionSet" );
  }
}

CompoundStmt* CPU_x86::VASTExportInstructionSet::_BuildCompoundStatement(AST::ScopePtr spScope)
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
  for (AST::IndexType iChildIdx = static_cast<AST::IndexType>(0); iChildIdx < spScope->GetChildCount(); ++iChildIdx)
  {
    AST::BaseClasses::NodePtr spNode = spScope->GetChild( iChildIdx );

    Stmt *pChildStmt = nullptr;

    if (spNode->IsType<AST::Scope>())
    {
      pChildStmt = _BuildCompoundStatement( spNode->CastToType<AST::Scope>() );
    }
    else if (spNode->IsType<AST::BaseClasses::ControlFlowStatement>())
    {
      AST::BaseClasses::ControlFlowStatementPtr spControlFlow = spNode->CastToType<AST::BaseClasses::ControlFlowStatement>();
      if (spControlFlow->IsVectorized())
      {
        throw RuntimeErrorException("Cannot handle vectorized control flow statements => Rebuild the control flow before calling the export!");
      }

      if (spControlFlow->IsType<AST::ControlFlow::BranchingStatement>())
      {
        AST::ControlFlow::BranchingStatementPtr spBranchingStatement = spControlFlow->CastToType<AST::ControlFlow::BranchingStatement>();

        ClangASTHelper::ExpressionVectorType  vecConditions;
        ClangASTHelper::StatementVectorType   vecBranchBodies;

        for (AST::IndexType iBranchIdx = static_cast<AST::IndexType>(0); iBranchIdx < spBranchingStatement->GetConditionalBranchesCount(); ++iBranchIdx)
        {
          AST::ControlFlow::ConditionalBranchPtr spCurrentBranch = spBranchingStatement->GetConditionalBranch(iBranchIdx);

          vecConditions.push_back( _BuildScalarExpression( spCurrentBranch->GetCondition() ) );
          vecBranchBodies.push_back( _BuildCompoundStatement( spCurrentBranch->GetBody() ) );
        }


        AST::ScopePtr spDefaultBranch = spBranchingStatement->GetDefaultBranch();
        ::clang::Stmt *pDefaultBranch = ( spDefaultBranch->GetChildCount() > 0 ) ? _BuildCompoundStatement( spDefaultBranch ) : nullptr;

        pChildStmt = _GetASTHelper().CreateIfStatement( vecConditions, vecBranchBodies, pDefaultBranch );
      }
      else if (spControlFlow->IsType<AST::ControlFlow::Loop>())
      {
        AST::ControlFlow::LoopPtr spLoop = spControlFlow->CastToType<AST::ControlFlow::Loop>();

        ::clang::CompoundStmt *pLoopBody      = _BuildCompoundStatement( spLoop->GetBody() );
        ::clang::Expr         *pConditionExpr = _BuildScalarExpression( spLoop->GetCondition() );
        ::clang::Expr         *pIncrementExpr = nullptr;

        if (spLoop->GetIncrement())
        {
          AST::BaseClasses::ExpressionPtr spIncrement = spLoop->GetIncrement();

          if (spIncrement->IsVectorized())
          {
            VectorElementTypes eElementType = _GetExpressionElementType( spIncrement );

            ClangASTHelper::ExpressionVectorType vecIncExpressions;

            // Create one increment expression for each vector group
            for (size_t szIdx = static_cast<size_t>(0); szIdx < _GetVectorArraySize( eElementType ); ++szIdx)
            {
              vecIncExpressions.push_back( _CreateParenthesis( _BuildVectorExpression(spIncrement, _CreateVectorIndex(eElementType, szIdx)) ) );
            }

            // Collapse increment expressions into a comma separated list
            for (size_t szIdx = static_cast<size_t>(1); szIdx < vecIncExpressions.size(); ++szIdx)
            {
              vecIncExpressions[0] = _GetASTHelper().CreateBinaryOperatorComma( vecIncExpressions[0], vecIncExpressions[szIdx] );
            }

            pIncrementExpr = vecIncExpressions.front();
          }
          else
          {
            pIncrementExpr = _BuildScalarExpression( spIncrement );
          }
        }

        pChildStmt = _BuildLoop( spLoop->GetLoopType(), pConditionExpr, pLoopBody, pIncrementExpr );
      }
      else if (spControlFlow->IsType<AST::ControlFlow::LoopControlStatement>())
      {
        pChildStmt = _BuildLoopControlStatement( spControlFlow->CastToType<AST::ControlFlow::LoopControlStatement>() );
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
    else if (spNode->IsType<AST::BaseClasses::Expression>())
    {
      AST::BaseClasses::ExpressionPtr spExpression  = spNode->CastToType<AST::BaseClasses::Expression>();
      bool                            bHandled      = false;

      // Handle all assignments to newly declared variables
      if (spExpression->IsType<AST::Expressions::AssignmentOperator>())
      {
        AST::Expressions::AssignmentOperatorPtr spAssignment = spExpression->CastToType<AST::Expressions::AssignmentOperator>();

        if ( spAssignment->GetLHS()->IsType<AST::Expressions::Identifier>() )
        {
          AST::Expressions::IdentifierPtr spAssignee = spAssignment->GetLHS()->CastToType<AST::Expressions::Identifier>();

          if ( ! _HasValueDeclaration(spAssignee->GetName()) )
          {
            // The variable is not declared yet => Create a declaration statement with an init expression
            AST::BaseClasses::ExpressionPtr spRHS             = spAssignment->GetRHS();
            Expr                            *pInitExpression  = nullptr;

            if ( spRHS->IsVectorized() )
            {
              VectorElementTypes eElementType = _GetExpressionElementType( spRHS );

              ClangASTHelper::ExpressionVectorType vecInitExpressions;

              for (size_t szGroupIndex = 0; szGroupIndex < _GetVectorArraySize( eElementType ); ++szGroupIndex)
              {
                vecInitExpressions.push_back( _BuildVectorExpression( spRHS, _CreateVectorIndex(eElementType, szGroupIndex) ) );
              }

              // If the init expression is masked, initialize the invalid vector elements with zero
              if ( spAssignment->IsMasked() )
              {
                bool bAddMasking = true;

                // If the initializer is equal to zero, no masking is required
                if (spRHS->IsType<AST::VectorSupport::BroadCast>())
                {
                  AST::BaseClasses::ExpressionPtr spInitValue = spRHS->CastToType<AST::VectorSupport::BroadCast>()->GetSubExpression();

                  if ( spInitValue->IsType<AST::Expressions::Constant>() && (spInitValue->CastToType<AST::Expressions::Constant>()->GetValue<double>() == 0.) )
                  {
                    bAddMasking = false;
                  }
                }

                if (bAddMasking)
                {
                  Expr *pMask = _BuildVectorExpression( spAssignment->GetMask(), _CreateVectorIndex(_GetMaskElementType(), 0) );

                  for (size_t szIdx = 0; szIdx < vecInitExpressions.size(); ++szIdx)
                  {
                    Expr *pMaskGroup = _ConvertMaskUp( eElementType, pMask, _CreateVectorIndex(eElementType, szIdx) );

                    vecInitExpressions[szIdx] = _spInstructionSet->BlendVectors( eElementType, pMaskGroup, vecInitExpressions[szIdx], _spInstructionSet->CreateZeroVector(eElementType) );
                  }
                }
              }

              // Build the actual initializer expression
              if ( vecInitExpressions.size() == 1 )
              {
                pInitExpression = vecInitExpressions.front();
              }
              else
              {
                pInitExpression = _GetASTHelper().CreateInitListExpression( vecInitExpressions );
              }
            }
            else
            {
              pInitExpression = _BuildScalarExpression( spRHS );
            }

            pChildStmt = _GetASTHelper().CreateDeclarationStatement( _BuildValueDeclaration(spAssignee, pInitExpression) );

            bHandled = true;
          }
        }
      }

      if (! bHandled)
      {
        pChildStmt = _BuildExpressionStatement( spExpression );
      }
    }
    else
    {
      throw InternalErrorException("Unsupported VAST node detected!");
    }

    vecChildren.push_back(pChildStmt);
  }

  return _GetASTHelper().CreateCompoundStatement(vecChildren);
}

Stmt* CPU_x86::VASTExportInstructionSet::_BuildExpressionStatement(AST::BaseClasses::ExpressionPtr spExpression)
{
  if (spExpression->IsVectorized())
  {
    VectorElementTypes eElementType = _GetExpressionElementType( spExpression );

    ClangASTHelper::StatementVectorType vecChildStatements;

    if ( _NeedsUnwrap( spExpression ) )
    {
      for (size_t szElementIndex = 0; szElementIndex < _cVectorWidth; ++szElementIndex)
      {
        vecChildStatements.push_back( _BuildUnrolledVectorExpression( spExpression, static_cast<uint32_t>(szElementIndex) ) );
      }
    }
    else
    {
      for (size_t szGroupIndex = 0; szGroupIndex < _GetVectorArraySize( eElementType ); ++szGroupIndex)
      {
        vecChildStatements.push_back( _BuildVectorExpression( spExpression, _CreateVectorIndex(eElementType, szGroupIndex) ) );
      }
    }

    if (vecChildStatements.size() == 1)
    {
      return vecChildStatements.front();
    }
    else
    {
      return _GetASTHelper().CreateCompoundStatement( vecChildStatements );
    }
  }
  else
  {
    return _BuildScalarExpression( spExpression );
  }
}

Expr* CPU_x86::VASTExportInstructionSet::_BuildScalarExpression(AST::BaseClasses::ExpressionPtr spExpression)
{
  if (spExpression->IsVectorized())
  {
    throw InternalErrorException("Expected a scalar expression!");
  }

  Expr *pReturnExpr = nullptr;

  if (spExpression->IsType<AST::Expressions::BinaryOperator>())
  {
    AST::Expressions::BinaryOperatorPtr spBinaryOp = spExpression->CastToType<AST::Expressions::BinaryOperator>();

    Expr *pExprLHS = _BuildScalarExpression( spBinaryOp->GetLHS() );
    Expr *pExprRHS = _BuildScalarExpression( spBinaryOp->GetRHS() );

    BinaryOperatorKind eOpCode = BO_Add;

    if (spBinaryOp->IsType<AST::Expressions::ArithmeticOperator>())
    {
      eOpCode = _ConvertArithmeticOperatorType( spBinaryOp->CastToType<AST::Expressions::ArithmeticOperator>()->GetOperatorType() );
    }
    else if (spBinaryOp->IsType<AST::Expressions::AssignmentOperator>())
    {
      eOpCode = BO_Assign;
    }
    else if (spBinaryOp->IsType<AST::Expressions::RelationalOperator>())
    {
      eOpCode = _ConvertRelationalOperatorType( spBinaryOp->CastToType<AST::Expressions::RelationalOperator>()->GetOperatorType() );
    }
    else
    {
      throw InternalErrorException("Unknown VAST binary operator node detected!");
    }

    pReturnExpr = _GetASTHelper().CreateBinaryOperator( pExprLHS, pExprRHS, eOpCode, _ConvertTypeInfo( spBinaryOp->GetResultType() ) );
  }
  else if (spExpression->IsType<AST::Expressions::FunctionCall>())
  {
    AST::Expressions::FunctionCallPtr spFunctionCall = spExpression->CastToType<AST::Expressions::FunctionCall>();

    ClangASTHelper::ExpressionVectorType vecArguments;

    for (AST::IndexType iParamIndex = static_cast<AST::IndexType>(0); iParamIndex < spFunctionCall->GetCallParameterCount(); ++iParamIndex)
    {
      vecArguments.push_back( _BuildScalarExpression( spFunctionCall->GetCallParameter(iParamIndex) ) );
    }

    pReturnExpr = _BuildScalarFunctionCall( spFunctionCall->GetName(), vecArguments );
  }
  else if (spExpression->IsType<AST::Expressions::UnaryExpression>())
  {
    AST::Expressions::UnaryExpressionPtr  spUnaryExpression = spExpression->CastToType<AST::Expressions::UnaryExpression>();
    AST::BaseClasses::TypeInfo            ResultType        = spUnaryExpression->GetResultType();
    AST::BaseClasses::ExpressionPtr       spSubExpression   = spUnaryExpression->GetSubExpression();
    Expr                                  *pSubExpr         = _BuildScalarExpression( spSubExpression );

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
  else if (spExpression->IsType<AST::Expressions::Value>())
  {
    AST::Expressions::ValuePtr spValue = spExpression->CastToType<AST::Expressions::Value>();

    if (spValue->IsType<AST::Expressions::Constant>())
    {
      pReturnExpr = _BuildConstant( spValue->CastToType<AST::Expressions::Constant>() );
    }
    else if (spValue->IsType<AST::Expressions::Identifier>())
    {
      pReturnExpr = _CreateDeclarationReference( spValue->CastToType<AST::Expressions::Identifier>()->GetName() );
    }
    else if (spValue->IsType<AST::Expressions::MemoryAccess>())
    {
      AST::Expressions::MemoryAccessPtr spMemoryAccess  = spValue->CastToType<AST::Expressions::MemoryAccess>();
      AST::BaseClasses::TypeInfo        ReturnType      = spMemoryAccess->GetResultType();

      Expr *pMemoryRef = _BuildScalarExpression( spMemoryAccess->GetMemoryReference() );
      Expr *pIndexExpr = _BuildScalarExpression( spMemoryAccess->GetIndexExpression() );

      pReturnExpr = _GetASTHelper().CreateArraySubscriptExpression( pMemoryRef, pIndexExpr, _ConvertTypeInfo(ReturnType), (! ReturnType.GetConst()) );
    }
    else
    {
      throw InternalErrorException("Unknown VAST value node detected!");
    }
  }
  else if (spExpression->IsType<AST::VectorSupport::VectorExpression>())
  {
    AST::VectorSupport::VectorExpressionPtr spVectorExpression = spExpression->CastToType<AST::VectorSupport::VectorExpression>();

    if (spVectorExpression->IsType<AST::VectorSupport::CheckActiveElements>())
    {
      AST::VectorSupport::CheckActiveElementsPtr spCheckElements = spVectorExpression->CastToType<AST::VectorSupport::CheckActiveElements>();
      AST::BaseClasses::ExpressionPtr            spMaskReference = spCheckElements->GetSubExpression();
      VectorElementTypes                         eElementType    = _GetExpressionElementType( spMaskReference );

      Expr  *pMaskRef = _BuildVectorExpression( spMaskReference, _CreateVectorIndex(eElementType, 0) );
      pReturnExpr     = _spInstructionSet->CheckActiveElements( eElementType, spCheckElements->GetCheckType(), pMaskRef );
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

CallExpr* CPU_x86::VASTExportInstructionSet::_BuildScalarFunctionCall(std::string strFunctionName, const ClangASTHelper::ExpressionVectorType &crVecArguments)
{
  ClangASTHelper::QualTypeVectorType vecArgumentTypes;

  for (auto itArgument : crVecArguments)
  {
    vecArgumentTypes.push_back( itArgument->getType() );
  }

  // Find the first exactly matching function
  ::clang::FunctionDecl *pCalleeDecl = _GetFirstMatchingFunctionDeclaration( strFunctionName, vecArgumentTypes );
  if (pCalleeDecl == nullptr)
  {
    throw RuntimeErrorException(std::string("Could not find matching FunctionDecl object for function call \"") + strFunctionName + std::string("\"!"));
  }

  return _GetASTHelper().CreateFunctionCall( pCalleeDecl, crVecArguments );
}

Expr* CPU_x86::VASTExportInstructionSet::_BuildVectorConversion(VectorElementTypes eTargetElementType, AST::BaseClasses::ExpressionPtr spSubExpression, const VectorIndex &crVectorIndex)
{
  const VectorElementTypes  ceSourceElementType   = spSubExpression->GetResultType().GetType();

  if (ceSourceElementType == VectorElementTypes::Bool)   // Mask conversions
  {
    if (eTargetElementType == VectorElementTypes::Bool)
    {
      // Nothing to do, just return the sub-expression
      return _BuildVectorExpression( spSubExpression, crVectorIndex );
    }
    else
    {
      // This is a conversion of a mask into a numeric type => Blend zeros and ones
      const size_t cszMaskElementCount    = _spInstructionSet->GetVectorElementCount( _GetMaskElementType() );
      const size_t cszTargetElementCount  = _spInstructionSet->GetVectorElementCount( eTargetElementType );

      // Convert the mask expression into the target type
      Expr *pConvertedMask = nullptr;
      if (cszMaskElementCount == cszTargetElementCount)
      {
        pConvertedMask = _spInstructionSet->ConvertMaskSameSize( _GetMaskElementType(), eTargetElementType, _BuildVectorExpression(spSubExpression, crVectorIndex) );
      }
      else if (cszMaskElementCount > cszTargetElementCount)
      {
        const VectorIndex cMaskIndex            = _CreateVectorIndex( _GetMaskElementType(), static_cast<size_t>(crVectorIndex.GetElementIndex()) / cszMaskElementCount );
        const uint32_t    cuiConvertGroupIndex  = ( crVectorIndex.GetElementIndex() - cMaskIndex.GetElementIndex() ) / crVectorIndex.GetElementCount();

        pConvertedMask = _spInstructionSet->ConvertMaskUp( _GetMaskElementType(), eTargetElementType, _BuildVectorExpression(spSubExpression, cMaskIndex), cuiConvertGroupIndex );
      }
      else
      {
        throw InternalErrorException("The mask element type is expected to be the smallest element type used inside a function!");
      }

      // Create the final output
      return _spInstructionSet->BlendVectors( eTargetElementType, pConvertedMask, _spInstructionSet->CreateOnesVector(eTargetElementType), _spInstructionSet->CreateZeroVector(eTargetElementType) );
    }
  }
  else if (eTargetElementType == VectorElementTypes::Bool)  // Numeric to boolean decay => Check vector elements for equality to zero
  {
    const size_t cszSourceElementCount  = _spInstructionSet->GetVectorElementCount( ceSourceElementType );
    const size_t cszTargetElementCount  = _spInstructionSet->GetVectorElementCount( _GetMaskElementType() );
    const size_t cszGroupOffset         = static_cast<size_t>(crVectorIndex.GetElementIndex()) / cszSourceElementCount;

    ClangASTHelper::ExpressionVectorType vecComparisons;

    for (size_t szGroup = static_cast<size_t>(0); szGroup < (cszTargetElementCount / cszSourceElementCount); ++szGroup)
    {
      const VectorIndex cSourceIndex = _CreateVectorIndex( ceSourceElementType, cszGroupOffset + szGroup );

      Expr *pSubExpr = _CreateParenthesis( _BuildVectorExpression( spSubExpression, cSourceIndex ) );

      vecComparisons.push_back( _spInstructionSet->RelationalOperator( ceSourceElementType, AST::Expressions::RelationalOperator::RelationalOperatorType::NotEqual,
                                                                       pSubExpr, _spInstructionSet->CreateZeroVector( ceSourceElementType ) ) );
    }

    return _ConvertMaskDown( ceSourceElementType, vecComparisons );
  }
  else    // Numeric conversion
  {
    const size_t cszSourceElementCount = _spInstructionSet->GetVectorElementCount( ceSourceElementType );
    const size_t cszTargetElementCount = _spInstructionSet->GetVectorElementCount( eTargetElementType );

    if (cszSourceElementCount == cszTargetElementCount)       // Same size conversion
    {
      return _spInstructionSet->ConvertVectorSameSize( ceSourceElementType, eTargetElementType, _BuildVectorExpression(spSubExpression, crVectorIndex) );
    }
    else if (cszSourceElementCount > cszTargetElementCount)   // Upward conversion   => Select a specific group
    {
      const VectorIndex cSourceIndex          = _CreateVectorIndex( ceSourceElementType, static_cast<size_t>(crVectorIndex.GetElementIndex()) / cszSourceElementCount );
      const uint32_t    cuiConvertGroupIndex  = ( crVectorIndex.GetElementIndex() - cSourceIndex.GetElementIndex() ) / crVectorIndex.GetElementCount();

      return _spInstructionSet->ConvertVectorUp( ceSourceElementType, eTargetElementType, _BuildVectorExpression(spSubExpression, cSourceIndex), cuiConvertGroupIndex );
    }
    else                                                      // Downward conversion => Merge multiple vectors
    {
      const size_t cszGroupOffset = static_cast<size_t>(crVectorIndex.GetElementIndex()) / cszSourceElementCount;

      // Build all required sub-expressions
      ClangASTHelper::ExpressionVectorType vecSubExpressions;

      for (size_t szGroup = static_cast<size_t>(0); szGroup < (cszTargetElementCount / cszSourceElementCount); ++szGroup)
      {
        const VectorIndex cSourceIndex = _CreateVectorIndex( ceSourceElementType, cszGroupOffset + szGroup );

        vecSubExpressions.push_back( _BuildVectorExpression( spSubExpression, cSourceIndex ) );
      }


      // Generate a conversion helper function if required (just for cleaning up the source code a bit)
      const std::string cstrConversionHelperName  = std::string("_ConvertVectorDown")  + AST::BaseClasses::TypeInfo::GetTypeString( ceSourceElementType ) +
                                              std::string("To")                  + AST::BaseClasses::TypeInfo::GetTypeString( eTargetElementType );
      {
        ClangASTHelper::QualTypeVectorType vecArgumentTypes;
        for (auto itSubExpr : vecSubExpressions)
        {
          vecArgumentTypes.push_back( itSubExpr->getType() );
        }

        bool bFunctionUnknown = _GetFirstMatchingFunctionDeclaration( cstrConversionHelperName, vecArgumentTypes ) == nullptr;

        // Create the argument names
        ClangASTHelper::StringVectorType vecArgumentNames;
        for (size_t szParamIdx = static_cast<size_t>(0); szParamIdx < vecArgumentTypes.size(); ++szParamIdx)
        {
          std::stringstream ssParamName;
          ssParamName << "value" << szParamIdx;
          vecArgumentNames.push_back( ssParamName.str() );
        }

        // Create the function declaration
        FunctionDecl* pConversionHelper = _GetASTHelper().CreateFunctionDeclaration( cstrConversionHelperName, _spInstructionSet->GetVectorType(eTargetElementType), vecArgumentNames, vecArgumentTypes );

        if (bFunctionUnknown)
        {
          // Create the function body (i.e. the actual downward conversion)
          {
            ClangASTHelper::ExpressionVectorType vecHelperFuncParams;
            for (unsigned int uiParamIdx = 0; uiParamIdx < pConversionHelper->getNumParams(); ++uiParamIdx)
            {
              vecHelperFuncParams.push_back( _GetASTHelper().CreateDeclarationReferenceExpression( pConversionHelper->getParamDecl( uiParamIdx ) ) );
            }

            Stmt *pBody = _GetASTHelper().CreateReturnStatement( _spInstructionSet->ConvertVectorDown( ceSourceElementType, eTargetElementType, vecHelperFuncParams ) );

            pConversionHelper->setBody( _GetASTHelper().CreateCompoundStatement( pBody ) );
          }


          // Add the generated helper function to the known functions
          _AddKnownFunctionDeclaration( pConversionHelper );
        }

        _vecHelperFunctions.push_back( pConversionHelper );
      }

      return _BuildScalarFunctionCall( cstrConversionHelperName, vecSubExpressions );
    }
  }
}

Expr* CPU_x86::VASTExportInstructionSet::_BuildUnrolledVectorExpression(AST::BaseClasses::ExpressionPtr spExpression, const uint32_t cuiElementIndex)
{
  if (! spExpression->IsVectorized())
  {
    throw InternalErrorException("Expected a vectorized expression for the unrolling!");
  }


  Expr *pReturnExpr = nullptr;

  if      (spExpression->IsType<AST::Expressions::BinaryOperator>())
  {
    AST::Expressions::BinaryOperatorPtr spBinaryOp = spExpression->CastToType<AST::Expressions::BinaryOperator>();

    Expr *pExprLHS = _BuildUnrolledVectorExpression( spBinaryOp->GetLHS(), cuiElementIndex );
    Expr *pExprRHS = _BuildUnrolledVectorExpression( spBinaryOp->GetRHS(), cuiElementIndex );

    BinaryOperatorKind eOpCode = BO_Add;

    if      (spBinaryOp->IsType< AST::Expressions::ArithmeticOperator>())
    {
      eOpCode = _ConvertArithmeticOperatorType( spBinaryOp->CastToType<AST::Expressions::ArithmeticOperator>()->GetOperatorType() );
    }
    else if (spBinaryOp->IsType<AST::Expressions::AssignmentOperator>())
    {
      AST::Expressions::AssignmentOperatorPtr spAssignment = spBinaryOp->CastToType<AST::Expressions::AssignmentOperator>();
      if (spAssignment->IsMasked())
      {
        Expr *pMaskElement  = _BuildUnrolledVectorExpression( spAssignment->GetMask(), cuiElementIndex );
        pMaskElement        = _GetASTHelper().CreateBinaryOperator( pMaskElement, _GetASTHelper().CreateLiteral(0), BO_NE, _GetASTContext().BoolTy );

        pExprRHS            = _GetASTHelper().CreateConditionalOperator( _CreateParenthesis( pMaskElement ), _CreateParenthesis( pExprRHS ),
                                                                         _CreateParenthesis( pExprLHS ), pExprLHS->getType() );
      }

      if (spBinaryOp->GetLHS()->IsType<AST::Expressions::Identifier>())
      {
        AST::Expressions::IdentifierPtr spAssignee = spBinaryOp->GetLHS()->CastToType<AST::Expressions::Identifier>();

        if (spAssignee->GetResultType().IsSingleValue())
        {
          const VectorElementTypes  ceElementType   = _GetExpressionElementType( spAssignee );
          const size_t              cszElementCount = _spInstructionSet->GetVectorElementCount( ceElementType );

          pExprLHS = _BuildVectorExpression(spAssignee, _CreateVectorIndex(ceElementType, static_cast<size_t>(cuiElementIndex) / cszElementCount));
          pExprRHS = _spInstructionSet->InsertElement( ceElementType, pExprLHS, pExprRHS, cuiElementIndex % cszElementCount );
        }
      }

      eOpCode = BO_Assign;
    }
    else if (spBinaryOp->IsType<AST::Expressions::RelationalOperator>())
    {
      eOpCode = _ConvertRelationalOperatorType( spBinaryOp->CastToType<AST::Expressions::RelationalOperator>()->GetOperatorType() );
    }
    else
    {
      throw InternalErrorException("Unknown VAST binary operator node detected!");
    }

    pReturnExpr = _GetASTHelper().CreateBinaryOperator( pExprLHS, pExprRHS, eOpCode, _ConvertTypeInfo( spBinaryOp->GetResultType() ) );
  }
  else if (spExpression->IsType<AST::Expressions::FunctionCall>())
  {
    AST::Expressions::FunctionCallPtr spFunctionCall = spExpression->CastToType<AST::Expressions::FunctionCall>();

    ClangASTHelper::ExpressionVectorType vecArguments;

    for (AST::IndexType iParamIndex = static_cast<AST::IndexType>(0); iParamIndex < spFunctionCall->GetCallParameterCount(); ++iParamIndex)
    {
      AST::BaseClasses::ExpressionPtr spCallParam = spFunctionCall->GetCallParameter( iParamIndex );

      if (spCallParam->IsVectorized())
      {
        vecArguments.push_back( _BuildUnrolledVectorExpression( spCallParam, cuiElementIndex ) );
      }
      else
      {
        vecArguments.push_back( _BuildScalarExpression( spCallParam ) );
      }
    }

    pReturnExpr = _BuildScalarFunctionCall( spFunctionCall->GetName(), vecArguments );
  }
  else if (spExpression->IsType<AST::Expressions::UnaryExpression>())
  {
    AST::Expressions::UnaryExpressionPtr  spUnaryExpression = spExpression->CastToType<AST::Expressions::UnaryExpression>();
    AST::BaseClasses::TypeInfo            ResultType        = spUnaryExpression->GetResultType();
    AST::BaseClasses::ExpressionPtr       spSubExpression   = spUnaryExpression->GetSubExpression();
    Expr                                  *pSubExpr         = _BuildUnrolledVectorExpression( spSubExpression, cuiElementIndex );

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
  else if (spExpression->IsType<AST::Expressions::Value>())
  {
    AST::Expressions::ValuePtr spValue = spExpression->CastToType<AST::Expressions::Value>();

    if (spValue->IsType<AST::Expressions::Constant>())
    {
      pReturnExpr = _BuildConstant( spValue->CastToType<AST::Expressions::Constant>() );
    }
    else if (spValue->IsType<AST::Expressions::Identifier>())
    {
      AST::Expressions::IdentifierPtr spIdentifier    = spValue->CastToType<AST::Expressions::Identifier>();
      const VectorElementTypes        ceElementType   = _GetExpressionElementType( spIdentifier );
      const size_t                    cszElementCount = _spInstructionSet->GetVectorElementCount( ceElementType );

      pReturnExpr = _BuildVectorExpression( spIdentifier, _CreateVectorIndex(ceElementType, static_cast<size_t>(cuiElementIndex) / cszElementCount) );

      if (spIdentifier->GetResultType().IsSingleValue())
      {
        pReturnExpr = _spInstructionSet->ExtractElement( ceElementType, pReturnExpr, cuiElementIndex % static_cast<uint32_t>(cszElementCount) );
      }
    }
    else if (spValue->IsType<AST::Expressions::MemoryAccess>())
    {
      AST::Expressions::MemoryAccessPtr spMemoryAccess    = spValue->CastToType<AST::Expressions::MemoryAccess>();
      AST::BaseClasses::TypeInfo        ReturnType        = spMemoryAccess->GetResultType();
      AST::BaseClasses::ExpressionPtr   spMemoryReference = spMemoryAccess->GetMemoryReference();
      AST::BaseClasses::ExpressionPtr   spIndexExpression = spMemoryAccess->GetIndexExpression();

      Expr *pMemoryRef = nullptr;
      if (spMemoryReference->IsVectorized())
      {
        pMemoryRef = _BuildUnrolledVectorExpression( spMemoryReference, cuiElementIndex );
      }
      else
      {
        pMemoryRef = _BuildScalarExpression( spMemoryReference );
      }

      Expr *pIndexExpr = nullptr;
      if (spIndexExpression->IsVectorized())
      {
        pIndexExpr = _BuildUnrolledVectorExpression( spIndexExpression, cuiElementIndex );
      }
      else
      {
        pIndexExpr = _BuildScalarExpression( spIndexExpression );
      }

      if (cuiElementIndex != 0)
      {
        pIndexExpr = _GetASTHelper().CreateBinaryOperator( _CreateParenthesis(pIndexExpr), _GetASTHelper().CreateLiteral( static_cast<int32_t>(cuiElementIndex) ),
                                                           BO_Add, pIndexExpr->getType() );
      }

      pReturnExpr = _GetASTHelper().CreateArraySubscriptExpression( pMemoryRef, pIndexExpr, _ConvertTypeInfo(ReturnType), (! ReturnType.GetConst()) );
    }
    else
    {
      throw InternalErrorException("Unknown VAST value node detected!");
    }
  }
  else if (spExpression->IsType<AST::VectorSupport::VectorExpression>())
  {
    AST::VectorSupport::VectorExpressionPtr spVectorExpression = spExpression->CastToType<AST::VectorSupport::VectorExpression>();

    if (spVectorExpression->IsType<AST::VectorSupport::BroadCast>())
    {
      pReturnExpr = _BuildScalarExpression( spVectorExpression->CastToType<AST::VectorSupport::BroadCast>()->GetSubExpression() );
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::CheckActiveElements>())
    {
      throw InternalErrorException("The check active elements node should be handled by scalar expression builder!");
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::VectorIndex>())
    {
      pReturnExpr = _GetASTHelper().CreateIntegerLiteral( static_cast<int32_t>(cuiElementIndex) );
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

Expr* CPU_x86::VASTExportInstructionSet::_BuildVectorExpression(AST::BaseClasses::ExpressionPtr spExpression, const VectorIndex &crVectorIndex)
{
  if (! spExpression->IsVectorized())
  {
    throw InternalErrorException("Expected a vectorized expression!");
  }


  Expr *pReturnExpr = nullptr;

  if (spExpression->IsType<AST::Expressions::BinaryOperator>())
  {
    AST::Expressions::BinaryOperatorPtr spBinaryOp    = spExpression->CastToType<AST::Expressions::BinaryOperator>();
    AST::BaseClasses::ExpressionPtr     spLHS         = spBinaryOp->GetLHS();
    AST::BaseClasses::ExpressionPtr     spRHS         = spBinaryOp->GetRHS();
    VectorElementTypes                  eElementType = _GetExpressionElementType(spLHS);

    if (eElementType != _GetExpressionElementType(spRHS))
    {
      throw RuntimeErrorException("Expected identical element types for both sides of a vectorized binary operator!");
    }


    if (spBinaryOp->IsType<AST::Expressions::AssignmentOperator>())
    {
      AST::Expressions::AssignmentOperatorPtr spAssignment  = spBinaryOp->CastToType<AST::Expressions::AssignmentOperator>();
      AST::BaseClasses::ExpressionPtr         spAssignee    = spLHS;

      if (! spAssignee->IsType<AST::Expressions::Value>())
      {
        throw InternalErrorException("Expected a VAST value node as assignee!");
      }

      // Check for a masked assignment
      Expr *pAssignmentMask = nullptr;
      if (spAssignment->IsMasked())
      {
        AST::Expressions::IdentifierPtr spMask = spAssignment->GetMask();
        if (! spMask->IsVectorized())
        {
          throw RuntimeErrorException("Vector assignment masks cannot be scalar values!");
        }

        pAssignmentMask = _BuildVectorExpression( spMask, _CreateVectorIndex( _GetMaskElementType(), 0 ) );
      }

      Expr *pExprRHS      = _BuildVectorExpression( spRHS, crVectorIndex );
      bool bHandled       = false;

      if (spAssignee->IsType<AST::Expressions::MemoryAccess>())   // Vector stores
      {
        AST::Expressions::MemoryAccessPtr spMemoryAccess = spAssignee->CastToType<AST::Expressions::MemoryAccess>();

        if (! spMemoryAccess->GetMemoryReference()->GetResultType().IsArray())
        {
          if (spMemoryAccess->GetIndexExpression()->IsVectorized())
          {
            // Scatter write => must be unrolled
            AST::BaseClasses::ExpressionPtr spMemoryReference = spMemoryAccess->GetMemoryReference();
            AST::BaseClasses::ExpressionPtr spIndexExpression = spMemoryAccess->GetIndexExpression();

            // Build the pointer reference expression
            Expr *pPointerRef = nullptr;
            if (spMemoryReference->IsVectorized())
            {
              pPointerRef = _BuildUnrolledVectorExpression( spMemoryReference, crVectorIndex.GetElementIndex() );
            }
            else
            {
              pPointerRef = _BuildScalarExpression( spMemoryReference );
            }

            // Build a store expression for each vector element
            ClangASTHelper::ExpressionVectorType vecElementStores;
            for (uint32_t uiElemIdx = 0; uiElemIdx < crVectorIndex.GetElementCount(); ++uiElemIdx)
            {
              Expr *pAssignmentVal  = _spInstructionSet->ExtractElement( eElementType, pExprRHS, uiElemIdx );
              Expr *pIndexExpr      = _BuildUnrolledVectorExpression( spIndexExpression, uiElemIdx + crVectorIndex.GetElementIndex() );
              Expr *pMemAccess      = _GetASTHelper().CreateArraySubscriptExpression(pPointerRef, pIndexExpr, pPointerRef->getType()->getPointeeType(), true );

              if (pAssignmentMask)
              {
                Expr *pMaskCheck  = _spInstructionSet->CheckSingleMaskElement( _GetMaskElementType(), pAssignmentMask, uiElemIdx + crVectorIndex.GetElementIndex() );
                pAssignmentVal    = _GetASTHelper().CreateConditionalOperator( _CreateParenthesis(pMaskCheck), pAssignmentVal, pMemAccess, pMemAccess->getType() );
              }

              vecElementStores.push_back( _GetASTHelper().CreateBinaryOperator(pMemAccess, pAssignmentVal, BO_Assign, pMemAccess->getType() ) );
              vecElementStores.back() = _CreateParenthesis( vecElementStores.back() );
            }

            // Wrap all single-value stores into a comma operator chain
            for (size_t szIdx = static_cast<size_t>(1); szIdx < vecElementStores.size(); ++szIdx)
            {
              vecElementStores[0] = _GetASTHelper().CreateBinaryOperatorComma( vecElementStores[0], vecElementStores[szIdx] );
            }

            pReturnExpr = _CreateParenthesis( vecElementStores.front() );
          }
          else
          {
            // Normal vector store
            Expr *pPointerRef = _TranslateMemoryAccessToPointerRef( spMemoryAccess, crVectorIndex );

            if (pAssignmentMask)
            {
              pAssignmentMask = _ConvertMaskUp( eElementType, pAssignmentMask, crVectorIndex );
              pReturnExpr     = _spInstructionSet->StoreVectorMasked( eElementType, pPointerRef, pExprRHS, pAssignmentMask );
            }
            else
            {
              pReturnExpr = _spInstructionSet->StoreVector( eElementType, pPointerRef, pExprRHS );
            }
          }

          bHandled = true;
        }
      }

      if (! bHandled)
      {
        Expr *pAssigneeExpr = _BuildVectorExpression( spAssignee, crVectorIndex );
        if (pAssignmentMask)
        {
          pAssignmentMask = _ConvertMaskUp( eElementType, pAssignmentMask, crVectorIndex );
          pExprRHS        = _spInstructionSet->BlendVectors( eElementType, pAssignmentMask, pExprRHS, pAssigneeExpr );
        }

        pReturnExpr = _GetASTHelper().CreateBinaryOperator( pAssigneeExpr, pExprRHS, BO_Assign, pAssigneeExpr->getType() );
      }
    }
    else
    {
      if (spBinaryOp->IsType<AST::Expressions::ArithmeticOperator>())
      {
        ArithmeticOperatorType eOpType = spBinaryOp->CastToType<AST::Expressions::ArithmeticOperator>()->GetOperatorType();

        Expr *pExprLHS = _BuildVectorExpression( spLHS, crVectorIndex );
        Expr *pExprRHS = nullptr;
        bool bIsScalar = false;

        // FIXME: This introduces some kind of cross dependencies. We know that
        // current instruction sets (SSE, AVX) require scalar arguments for
        // shift operations. Therefore, remove broadcast expression introduced
        // previously.
        if (spRHS->IsType<AST::VectorSupport::BroadCast>() &&
            (eOpType == ArithmeticOperatorType::ShiftLeft ||
             eOpType == ArithmeticOperatorType::ShiftRight)) {
          pExprRHS = _BuildScalarExpression(spRHS->CastToType<AST::VectorSupport::BroadCast>()->GetSubExpression());
          bIsScalar = true;
        } else {
          pExprRHS = _BuildVectorExpression( spRHS, crVectorIndex );
        }

        pReturnExpr = _spInstructionSet->ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS, bIsScalar);
      }
      else if (spBinaryOp->IsType<AST::Expressions::RelationalOperator>())
      {
        // The return type of the comparison is the ComparisonType => A mask conversion is required here
        AST::Expressions::RelationalOperator::RelationalOperatorType eOperatorType = spBinaryOp->CastToType<AST::Expressions::RelationalOperator>()->GetOperatorType();

        ClangASTHelper::ExpressionVectorType vecRelationalExprs;

        for (size_t szGroupIndex = 0; szGroupIndex < _GetVectorArraySize( eElementType ); ++szGroupIndex)
        {
          Expr *pExprLHS = _BuildVectorExpression( spLHS, _CreateVectorIndex( eElementType, szGroupIndex) );
          Expr *pExprRHS = _BuildVectorExpression( spRHS, _CreateVectorIndex( eElementType, szGroupIndex) );

          vecRelationalExprs.push_back( _spInstructionSet->RelationalOperator( eElementType, eOperatorType, pExprLHS, pExprRHS ) );
        }

        pReturnExpr = _ConvertMaskDown( eElementType, vecRelationalExprs );
      }
      else
      {
        throw InternalErrorException("Unknown VAST binary operator node detected!");
      }
    }
  }
  else if (spExpression->IsType<AST::Expressions::FunctionCall>())
  {
    AST::Expressions::FunctionCallPtr spFunctionCall = spExpression->CastToType<AST::Expressions::FunctionCall>();

    if (! _SupportsVectorFunctionCall(spFunctionCall))
    {
      throw InternalErrorException("Unsupported vector function call detected => The parent expression should have been unrolled!");
    }


    const VectorElementTypes ceFunctionElementType = _GetExpressionElementType( spFunctionCall );
    ClangASTHelper::ExpressionVectorType vecArguments;

    for (AST::IndexType iParamIdx = static_cast<AST::IndexType>(0); iParamIdx < spFunctionCall->GetCallParameterCount(); ++iParamIdx)
    {
      AST::BaseClasses::ExpressionPtr spCallParam = spFunctionCall->GetCallParameter( iParamIdx );

      if (spCallParam->IsVectorized())
      {
        vecArguments.push_back( _BuildVectorExpression( spCallParam, crVectorIndex ) );
      }
      else
      {
        vecArguments.push_back( _spInstructionSet->BroadCast( ceFunctionElementType, _BuildScalarExpression(spCallParam) ) );
      }
    }

    pReturnExpr = _spInstructionSet->BuiltinFunction( _GetExpressionElementType( spFunctionCall ), _GetBuiltinVectorFunctionType( spFunctionCall->GetName() ), vecArguments );
  }
  else if (spExpression->IsType<AST::Expressions::UnaryExpression>())
  {
    AST::Expressions::UnaryExpressionPtr  spUnaryExpression = spExpression->CastToType<AST::Expressions::UnaryExpression>();
    AST::BaseClasses::ExpressionPtr       spSubExpression   = spUnaryExpression->GetSubExpression();
    AST::BaseClasses::TypeInfo            ResultType        = spExpression->GetResultType();
    VectorElementTypes                    eSubElementType   = _GetExpressionElementType( spSubExpression );

    if (spUnaryExpression->IsType<AST::Expressions::Conversion>())
    {
      if (ResultType.IsArray())
      {
        throw RuntimeErrorException("Conversions into array types are not supported!");
      }
      else if (ResultType.GetPointer())
      {
        throw RuntimeErrorException("Explicit pointer conversions are not supported!");
      }
      else
      {
        AST::BaseClasses::TypeInfo SubExprType = spSubExpression->GetResultType();
        if (! SubExprType.IsSingleValue())
        {
          throw RuntimeErrorException("Cannot dereference a type by a conversion!");
        }

        pReturnExpr = _BuildVectorConversion( ResultType.GetType(), spSubExpression, crVectorIndex );
      }
    }
    else if (spUnaryExpression->IsType<AST::Expressions::Parenthesis>())
    {
      pReturnExpr = _CreateParenthesis( _BuildVectorExpression( spSubExpression, crVectorIndex ) );
    }
    else if (spUnaryExpression->IsType<AST::Expressions::UnaryOperator>())
    {
      typedef AST::Expressions::UnaryOperator::UnaryOperatorType  OperatorType;

      OperatorType eOpType = spUnaryExpression->CastToType<AST::Expressions::UnaryOperator>()->GetOperatorType();

      if (spSubExpression->GetResultType().IsSingleValue())
      {
        if (eOpType == OperatorType::LogicalNot)
        {
          // This operator creates a mask => Apply it to all vector elements and convert the resulting masks
          ClangASTHelper::ExpressionVectorType vecSubExpressions;

          for (size_t szGroupIndex = 0; szGroupIndex < _GetVectorArraySize( eSubElementType ); ++szGroupIndex)
          {
            Expr *pSubExpr = _BuildVectorExpression( spSubExpression, _CreateVectorIndex(eSubElementType, szGroupIndex) );

            vecSubExpressions.push_back( _spInstructionSet->UnaryOperator( eSubElementType, eOpType, pSubExpr ) );
          }

          pReturnExpr = _ConvertMaskDown( eSubElementType, vecSubExpressions );
        }
        else
        {
          pReturnExpr = _spInstructionSet->UnaryOperator( eSubElementType, eOpType, _BuildVectorExpression(spSubExpression, crVectorIndex) );
        }
      }
      else
      {
        pReturnExpr = _GetASTHelper().CreateUnaryOperator( _BuildVectorExpression(spSubExpression, crVectorIndex), _ConvertUnaryOperatorType(eOpType), _ConvertTypeInfo(ResultType) );
      }
    }
    else
    {
      throw InternalErrorException("Unknown VAST unary expression node detected!");
    }
  }
  else if (spExpression->IsType<AST::Expressions::Value>())
  {
    AST::Expressions::ValuePtr spValue = spExpression->CastToType<AST::Expressions::Value>();

    if (spValue->IsType<AST::Expressions::Constant>())
    {
      throw InternalErrorException("A constant cannot be translated into a vector expression!");
    }
    else if (spValue->IsType<AST::Expressions::Identifier>())
    {
      AST::Expressions::IdentifierPtr spIdentifier = spValue->CastToType<AST::Expressions::Identifier>();

      Expr *pIdentifierRef = _CreateDeclarationReference( spIdentifier->GetName() );

      if (! spIdentifier->GetResultType().GetPointer())
      {
        VectorElementTypes eElementType = _GetExpressionElementType( spIdentifier );

        if (_GetVectorArraySize(eElementType) > static_cast<size_t>(1))
        {
          IntegerLiteral  *pIndexExpr   = _GetASTHelper().CreateIntegerLiteral( static_cast<int32_t>(crVectorIndex.GetGroupIndex()) );
          QualType        qtReturnType  = pIdentifierRef->getType()->getAsArrayTypeUnsafe()->getElementType();

          pIdentifierRef = _GetASTHelper().CreateArraySubscriptExpression( pIdentifierRef, pIndexExpr, qtReturnType );
        }
      }

      pReturnExpr = pIdentifierRef;
    }
    else if (spValue->IsType<AST::Expressions::MemoryAccess>())
    {
      AST::Expressions::MemoryAccessPtr spMemoryAccess    = spValue->CastToType<AST::Expressions::MemoryAccess>();
      AST::BaseClasses::ExpressionPtr   spMemoryReference = spMemoryAccess->GetMemoryReference();
      AST::BaseClasses::ExpressionPtr   spIndexExpression = spMemoryAccess->GetIndexExpression();

      VectorElementTypes  eElementType  = _GetExpressionElementType( spMemoryAccess );


      if (spMemoryReference->GetResultType().IsArray())  // Array access
      {
        if (spIndexExpression->IsVectorized())
        {
          throw RuntimeErrorException("Gather loads are not supported for array accesses!");
        }

        Expr *pMemoryRef  = spMemoryReference->IsVectorized() ? _BuildVectorExpression( spMemoryReference, crVectorIndex ) : _BuildScalarExpression( spMemoryReference );
        pReturnExpr       = _GetASTHelper().CreateArraySubscriptExpression( pMemoryRef, _BuildScalarExpression(spIndexExpression),
                                                                            pMemoryRef->getType()->getAsArrayTypeUnsafe()->getElementType() );
      }
      else      // Pointer access
      {
        if (spIndexExpression->IsVectorized())
        {
          VectorElementTypes  eIndexElementType = _GetExpressionElementType( spIndexExpression );

          const uint32_t cuiElementCount      = crVectorIndex.GetElementCount();
          const uint32_t cuiIndexElementCount = static_cast<uint32_t>(_spInstructionSet->GetVectorElementCount(eIndexElementType));

          ClangASTHelper::ExpressionVectorType  vecIndexExpressions;
          uint32_t                              uiGroupIndex = 0;

          if (cuiElementCount == cuiIndexElementCount)
          {
            vecIndexExpressions.push_back( _BuildVectorExpression( spIndexExpression, _CreateVectorIndex(eIndexElementType, crVectorIndex.GetGroupIndex()) ) );
          }
          else if (cuiElementCount > cuiIndexElementCount)
          {
            for (uint32_t uiElementOffset = 0; uiElementOffset < cuiElementCount; uiElementOffset += cuiIndexElementCount)
            {
              vecIndexExpressions.push_back( _BuildVectorExpression( spIndexExpression, _CreateVectorIndex(eIndexElementType, (crVectorIndex.GetElementIndex() + uiElementOffset) / cuiIndexElementCount) ) );
            }
          }
          else
          {
            uiGroupIndex %= (cuiIndexElementCount / cuiElementCount);

            vecIndexExpressions.push_back( _BuildVectorExpression( spIndexExpression, _CreateVectorIndex(eIndexElementType, crVectorIndex.GetElementIndex() / cuiIndexElementCount) ) );
          }

          Expr *pMemoryRef  = spMemoryReference->IsVectorized() ? _BuildVectorExpression( spMemoryReference, crVectorIndex ) : _BuildScalarExpression( spMemoryReference );
          pReturnExpr       = _spInstructionSet->LoadVectorGathered( eElementType, eIndexElementType, pMemoryRef, vecIndexExpressions, uiGroupIndex );
        }
        else
        {
          Expr *pPointerRef = _TranslateMemoryAccessToPointerRef( spMemoryAccess, crVectorIndex );
          pReturnExpr       = _spInstructionSet->LoadVector( eElementType, pPointerRef );
        }
      }
    }
    else
    {
      throw InternalErrorException("Unknown VAST value node detected!");
    }
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

      const VectorElementTypes ceElementType = _GetExpressionElementType( spBroadCast );

      if ( spSubExpression->IsType<AST::Expressions::Constant>() && (spSubExpression->CastToType<AST::Expressions::Constant>()->GetValue<double>() == 0.) )
      {
        pReturnExpr = _spInstructionSet->CreateZeroVector( ceElementType );
      }
      else
      {
        Expr *pBroadCastValue = _BuildScalarExpression(spSubExpression);

        if (spSubExpression->GetResultType().GetType() == VectorElementTypes::Bool)
        {
          pBroadCastValue = _GetASTHelper().CreateConditionalOperator( pBroadCastValue, _GetASTHelper().CreateLiteral(-1), _GetASTHelper().CreateLiteral(0), _GetASTContext().IntTy );
        }

        pReturnExpr = _spInstructionSet->BroadCast( ceElementType, pBroadCastValue );
      }
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::CheckActiveElements>())
    {
      throw InternalErrorException("The check active elements node should be handled by scalar expression builder!");
    }
    else if (spVectorExpression->IsType<AST::VectorSupport::VectorIndex>())
    {
      ClangASTHelper::ExpressionVectorType vecIndices;

      for (uint32_t uiIndex = 0; uiIndex < crVectorIndex.GetElementCount(); ++uiIndex)
      {
        int32_t iCurrentIndex = static_cast<int32_t>(uiIndex + crVectorIndex.GetElementIndex());
        vecIndices.push_back( _GetASTHelper().CreateIntegerLiteral( iCurrentIndex ) );
      }

      pReturnExpr = _spInstructionSet->CreateVector( _GetExpressionElementType(spVectorExpression), vecIndices );
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

Expr* CPU_x86::VASTExportInstructionSet::_ConvertMaskDown(VectorElementTypes eSourceElementType, const ClangASTHelper::ExpressionVectorType &crvecSubExpressions)
{
  switch ( crvecSubExpressions.size() )
  {
  case 0:   throw InternalErrorException("The source expression vector for a mask conversion cannot be empty!");
  case 1:   return _spInstructionSet->ConvertMaskSameSize( eSourceElementType, _GetMaskElementType(), crvecSubExpressions.front() );
  default:
    {
      // Generate a conversion helper function if required (just for cleaning up the source code a bit)
      const std::string cstrConversionHelperName = std::string("_PackMask") + AST::BaseClasses::TypeInfo::GetTypeString(eSourceElementType);
      {
        ClangASTHelper::QualTypeVectorType vecArgumentTypes;
        for (auto itSubExpr : crvecSubExpressions)
        {
          vecArgumentTypes.push_back( itSubExpr->getType() );
        }

        bool bFunctionUnknown = _GetFirstMatchingFunctionDeclaration( cstrConversionHelperName, vecArgumentTypes ) == nullptr;

        // Create the argument names
        ClangASTHelper::StringVectorType vecArgumentNames;
        for (size_t szParamIdx = static_cast<size_t>(0); szParamIdx < vecArgumentTypes.size(); ++szParamIdx)
        {
          std::stringstream ssParamName;
          ssParamName << "mask" << szParamIdx;
          vecArgumentNames.push_back( ssParamName.str() );
        }

        // Create the function declaration
        FunctionDecl *pConversionHelper = _GetASTHelper().CreateFunctionDeclaration( cstrConversionHelperName, _spInstructionSet->GetVectorType( _GetMaskElementType() ), vecArgumentNames, vecArgumentTypes );

        if (bFunctionUnknown)
        {
          // Create the function body (i.e. the actual downward conversion)
          {
            ClangASTHelper::ExpressionVectorType vecHelperFuncParams;
            for (unsigned int uiParamIdx = 0; uiParamIdx < pConversionHelper->getNumParams(); ++uiParamIdx)
            {
              vecHelperFuncParams.push_back( _GetASTHelper().CreateDeclarationReferenceExpression( pConversionHelper->getParamDecl( uiParamIdx ) ) );
            }

            Stmt *pBody = _GetASTHelper().CreateReturnStatement( _spInstructionSet->ConvertMaskDown( eSourceElementType, _GetMaskElementType(), vecHelperFuncParams ) );

            pConversionHelper->setBody( _GetASTHelper().CreateCompoundStatement( pBody ) );
          }

          // Add the generated helper function to the known functions
          _AddKnownFunctionDeclaration( pConversionHelper );
        }

        _vecHelperFunctions.push_back( pConversionHelper );
      }

      return _BuildScalarFunctionCall( cstrConversionHelperName, crvecSubExpressions );
    }
  }
}

Expr* CPU_x86::VASTExportInstructionSet::_ConvertMaskUp(Vectorization::VectorElementTypes eTargetElementType, Expr *pMaskExpr, const VectorIndex &crVectorIndex)
{
  const size_t cszMaskElementCount    = _spInstructionSet->GetVectorElementCount( _GetMaskElementType() );
  const size_t cszTargetElementCount  = _spInstructionSet->GetVectorElementCount( eTargetElementType );

  if (cszMaskElementCount == cszTargetElementCount)
  {
    return _spInstructionSet->ConvertMaskSameSize( _GetMaskElementType(), eTargetElementType, pMaskExpr );
  }
  else if (cszMaskElementCount > cszTargetElementCount)
  {
    return _spInstructionSet->ConvertMaskUp( _GetMaskElementType(), eTargetElementType, pMaskExpr, crVectorIndex.GetGroupIndex() );
  }
  else
  {
    throw InternalErrorException("The mask element type is expected to be the smallest element type used inside a function!");
  }
}

CPU_x86::VASTExportInstructionSet::VectorIndex CPU_x86::VASTExportInstructionSet::_CreateVectorIndex(VectorElementTypes eElementType, size_t szGroupIndex)
{
  uint32_t uiElementCount = static_cast<uint32_t>(_spInstructionSet->GetVectorElementCount(eElementType));

  return VectorIndex( VectorIndex::IndexTypeEnum::VectorStart, static_cast<uint32_t>(szGroupIndex) * uiElementCount, uiElementCount );
}

BuiltinFunctionsEnum CPU_x86::VASTExportInstructionSet::_GetBuiltinVectorFunctionType(std::string strFunctionName)
{
  if      (strFunctionName == "abs")                return BuiltinFunctionsEnum::Abs;
  else if (strFunctionName == "ceil")               return BuiltinFunctionsEnum::Ceil;
  else if (strFunctionName == "floor")              return BuiltinFunctionsEnum::Floor;
  else if (strFunctionName == "max")                return BuiltinFunctionsEnum::Max;
  else if (strFunctionName == "min")                return BuiltinFunctionsEnum::Min;
  else if (strFunctionName == "sqrt")               return BuiltinFunctionsEnum::Sqrt;

  else if (strFunctionName == "std::abs")           return BuiltinFunctionsEnum::Abs;
  else if (strFunctionName == "std::ceil")          return BuiltinFunctionsEnum::Ceil;
  else if (strFunctionName == "std::floor")         return BuiltinFunctionsEnum::Floor;
  else if (strFunctionName == "std::max")           return BuiltinFunctionsEnum::Max;
  else if (strFunctionName == "std::min")           return BuiltinFunctionsEnum::Min;
  else if (strFunctionName == "std::sqrt")          return BuiltinFunctionsEnum::Sqrt;

  else if (strFunctionName == "hipacc::math::max")  return BuiltinFunctionsEnum::Max;
  else if (strFunctionName == "hipacc::math::min")  return BuiltinFunctionsEnum::Min;

  else                                              return BuiltinFunctionsEnum::UnknownFunction;
}

VectorElementTypes CPU_x86::VASTExportInstructionSet::_GetExpressionElementType(AST::BaseClasses::ExpressionPtr spExpression)
{
  VectorElementTypes eElementType = spExpression->GetResultType().GetType();

  if ( (eElementType == VectorElementTypes::Bool) && spExpression->IsVectorized() )
  {
    eElementType = _GetMaskElementType();
  }

  return eElementType;
}

VectorElementTypes CPU_x86::VASTExportInstructionSet::_GetMaskElementType()
{
  const VectorElementTypes caeMaskElementType[] = { VectorElementTypes::UInt8, VectorElementTypes::UInt16, VectorElementTypes::UInt32 };

  // Return the smallest mask type which does not waste vector elements
  for (auto itMaskElemType : caeMaskElementType)
  {
    if (_cVectorWidth >= _spInstructionSet->GetVectorElementCount(itMaskElemType))
    {
      return itMaskElemType;
    }
  }

  // If no mask type has been determined earlier, return the largest possible type
  return VectorElementTypes::UInt64;
}

size_t CPU_x86::VASTExportInstructionSet::_GetVectorArraySize(Vectorization::VectorElementTypes eElementType)
{
  return std::max( static_cast<size_t>(1), _cVectorWidth / _spInstructionSet->GetVectorElementCount(eElementType) );
}

QualType CPU_x86::VASTExportInstructionSet::_GetVectorizedType(AST::BaseClasses::TypeInfo &crOriginalTypeInfo)
{
  if (crOriginalTypeInfo.GetPointer())
  {
    return _ConvertTypeInfo( crOriginalTypeInfo );
  }

  AST::BaseClasses::TypeInfo::KnownTypes eElementType = crOriginalTypeInfo.GetType();
  if (eElementType == AST::BaseClasses::TypeInfo::KnownTypes::Bool)
  {
    // Change the element type to the current mask type
    eElementType = _GetMaskElementType();
  }


  QualType qtReturnType = _spInstructionSet->GetVectorType( eElementType );

  // If this vector type is translated into a vector array, add its dimension
  {
    const size_t cszArraySize = _GetVectorArraySize( eElementType );

    if ( cszArraySize > static_cast<size_t>(1) )
    {
      qtReturnType = _GetASTHelper().GetConstantArrayType( qtReturnType, cszArraySize );
    }
  }


  // Add the user-defined array dimensions
  for (auto itDim = crOriginalTypeInfo.GetArrayDimensions().end(); itDim != crOriginalTypeInfo.GetArrayDimensions().begin(); itDim--)
  {
    qtReturnType = _GetASTHelper().GetConstantArrayType( qtReturnType, *(itDim-1) );
  }

  return qtReturnType;
}

bool CPU_x86::VASTExportInstructionSet::_NeedsUnwrap(AST::BaseClasses::ExpressionPtr spExpression)
{
  for (AST::IndexType iSubExprIdx = static_cast<AST::IndexType>(0); iSubExprIdx < spExpression->GetSubExpressionCount(); ++iSubExprIdx)
  {
    AST::BaseClasses::ExpressionPtr spSubExpr = spExpression->GetSubExpression( iSubExprIdx );

    if (spSubExpr->IsVectorized() && spSubExpr->IsType<AST::Expressions::FunctionCall>())
    {
      if ( ! _SupportsVectorFunctionCall(spSubExpr->CastToType<AST::Expressions::FunctionCall>()) )
      {
        return true;
      }
    }
    else if ( _NeedsUnwrap(spSubExpr) )
    {
      return true;
    }
  }

  return false;
}

bool CPU_x86::VASTExportInstructionSet::_SupportsVectorFunctionCall(AST::Expressions::FunctionCallPtr spFunctionCall)
{
  BuiltinFunctionsEnum eFunctionType = _GetBuiltinVectorFunctionType( spFunctionCall->GetName() );

  if (! spFunctionCall->GetReturnType().IsSingleValue())
  {
    return false;
  }

  const VectorElementTypes  ceElementType = _GetExpressionElementType( spFunctionCall );
  const AST::IndexType      ciParamCount  = spFunctionCall->GetCallParameterCount();

  for (AST::IndexType iParamIdx = static_cast<AST::IndexType>(0); iParamIdx < ciParamCount; ++iParamIdx)
  {
    AST::BaseClasses::ExpressionPtr spCallParam = spFunctionCall->GetCallParameter( iParamIdx );

    if (! spCallParam->GetResultType().IsSingleValue())
    {
      return false;
    }
    else if (_GetExpressionElementType(spCallParam) != ceElementType)
    {
      return false;
    }
  }

  return _spInstructionSet->IsBuiltinFunctionSupported( ceElementType, eFunctionType, static_cast<uint32_t>(ciParamCount) );
}

Expr* CPU_x86::VASTExportInstructionSet::_TranslateMemoryAccessToPointerRef(AST::Expressions::MemoryAccessPtr spMemoryAccess, const VectorIndex &crVectorIndex)
{
  if (! spMemoryAccess->IsVectorized())
  {
    throw InternalErrorException("Expected a vectorized memory access!");
  }

  AST::BaseClasses::ExpressionPtr spMemoryReference = spMemoryAccess->GetMemoryReference();
  AST::BaseClasses::ExpressionPtr spIndexExpression = spMemoryAccess->GetIndexExpression();

  if (spMemoryReference->GetResultType().IsArray())
  {
    throw RuntimeErrorException("Cannot translate array accesses to pointer references!");
  }
  else if (spIndexExpression->IsVectorized())
  {
    throw RuntimeErrorException("Cannot handle vectorized index expressions!");
  }


  // Check if we have a zero offset, which can be ignored
  bool bBuildIndex = true;
  if ( spIndexExpression->IsType< AST::Expressions::Constant >() )
  {
    AST::Expressions::ConstantPtr spIndexConstant = spIndexExpression->CastToType<AST::Expressions::Constant>();

    bBuildIndex = (spIndexConstant->GetValue<int32_t>() != 0);
  }

  Expr *pIndexExpr = bBuildIndex ? _BuildScalarExpression( spIndexExpression ) : nullptr;

  // Check if we have a vector offset
  int32_t iElementIndex = static_cast<int32_t>(crVectorIndex.GetElementIndex());
  if (iElementIndex != 0)
  {
    Expr *pOffsetExpr = _GetASTHelper().CreateIntegerLiteral( iElementIndex );

    if (pIndexExpr)
    {
      pIndexExpr = _GetASTHelper().CreateBinaryOperator( _CreateParenthesis(pIndexExpr), pOffsetExpr, BO_Add, pIndexExpr->getType() );
    }
    else
    {
      pIndexExpr = pOffsetExpr;
    }
  }

  Expr *pMemoryRef = _BuildVectorExpression( spMemoryReference, crVectorIndex );

  // Check if we actually need an index
  if (pIndexExpr)
  {
    return _GetASTHelper().CreateBinaryOperator( pMemoryRef, _CreateParenthesis(pIndexExpr), BO_Add, pMemoryRef->getType() );
  }
  else
  {
    return pMemoryRef;
  }
}


FunctionDecl* CPU_x86::VASTExportInstructionSet::ExportVASTFunction(AST::FunctionDeclarationPtr spVASTFunction)
{
  if (! spVASTFunction)
  {
    throw InternalErrors::NullPointerException("spVASTFunction");
  }


  // Clear helper functions from previous call
  _vecHelperFunctions.clear();


  // Create the function declaration statement
  ::clang::FunctionDecl *pFunctionDecl = _BuildFunctionDeclaration( spVASTFunction );


  // Build the function body
  pFunctionDecl->setBody( _BuildCompoundStatement( spVASTFunction->GetBody() ) );


  // Reset exporter state
  _Reset();

  return pFunctionDecl;
}



// Implementation of class CPU_x86::CodeGenerator::Descriptor
CPU_x86::CodeGenerator::Descriptor::Descriptor()
{
  SetTargetLang(::clang::hipacc::Language::C99);
  SetName("CPU-x86");
  SetEmissionKey("cpu");
  SetDescription("Emit C++ code for x86-CPUs");
}


// Implementation of class CPU_x86::CodeGenerator::KernelSubFunctionBuilder
void CPU_x86::CodeGenerator::KernelSubFunctionBuilder::AddCallParameter(::clang::DeclRefExpr *pCallParam, bool bForceConstDecl)
{
  ::clang::ValueDecl  *pValueDecl = pCallParam->getDecl();
  ::clang::QualType   qtParamType = pValueDecl->getType();

  if (bForceConstDecl)
  {
    qtParamType.addConst();
  }

  _vecArgumentTypes.push_back(qtParamType);
  _vecArgumentNames.push_back(pValueDecl->getNameAsString());
  _vecCallParams.push_back(pCallParam);
}

CPU_x86::CodeGenerator::KernelSubFunctionBuilder::DeclCallPairType  CPU_x86::CodeGenerator::KernelSubFunctionBuilder::CreateFuntionDeclarationAndCall(std::string strFunctionName, const ::clang::QualType &crResultType)
{
  DeclCallPairType pairDeclAndCall;

  pairDeclAndCall.first  = _ASTHelper.CreateFunctionDeclaration( strFunctionName, crResultType, _vecArgumentNames, _vecArgumentTypes );
  pairDeclAndCall.second = _ASTHelper.CreateFunctionCall( pairDeclAndCall.first, _vecCallParams );

  return pairDeclAndCall;
}

void CPU_x86::CodeGenerator::KernelSubFunctionBuilder::ImportUsedParameters(::clang::FunctionDecl *pRootFunctionDecl, ::clang::Stmt *pSubFunctionBody)
{
  for (unsigned int i = 0; i < pRootFunctionDecl->getNumParams(); ++i)
  {
    ParmVarDecl *pParamVarDecl = pRootFunctionDecl->getParamDecl(i);

    if (IsVariableUsed(pParamVarDecl->getNameAsString(), pSubFunctionBody))
    {
      AddCallParameter( _ASTHelper.CreateDeclarationReferenceExpression(pParamVarDecl) );
    }
  }
}

bool CPU_x86::CodeGenerator::KernelSubFunctionBuilder::IsVariableUsed(const std::string &crstrVariableName, ::clang::Stmt *pStatement)
{
  return (ClangASTHelper::CountNumberOfReferences(pStatement, crstrVariableName) > 0);
}


// Implementation of class CPU_x86::CodeGenerator::ImageAccessTranslator
CPU_x86::CodeGenerator::ImageAccessTranslator::ImageAccessTranslator(HipaccHelper &rHipaccHelper) : _rHipaccHelper(rHipaccHelper), _ASTHelper(_rHipaccHelper.GetKernelFunction()->getASTContext())
{
  ::clang::FunctionDecl *pKernelFunction = rHipaccHelper.GetKernelFunction();

  _pDRGidX = _ASTHelper.FindDeclaration(pKernelFunction, _rHipaccHelper.GlobalIdX());
  _pDRGidY = _ASTHelper.FindDeclaration(pKernelFunction, _rHipaccHelper.GlobalIdY());
}

std::list<::clang::ArraySubscriptExpr*> CPU_x86::CodeGenerator::ImageAccessTranslator::_FindImageAccesses(const std::string &crstrImageName, ::clang::Stmt *pStatement)
{
  std::list<::clang::ArraySubscriptExpr*> lstImageAccesses;

  if (pStatement == nullptr)
  {
    return lstImageAccesses;
  }
  else if (isa<::clang::ArraySubscriptExpr>(pStatement))
  {
    // Found an array subscript expression => Check if the structure corresponds to an image access
    ::clang::ArraySubscriptExpr *pRootArraySubscript = dyn_cast<::clang::ArraySubscriptExpr>(pStatement);

    // Look through implicit cast expressions
    ::clang::Expr *pLhs = pRootArraySubscript->getLHS();
    while (isa<::clang::ImplicitCastExpr>(pLhs))
    {
      pLhs = dyn_cast<::clang::ImplicitCastExpr>(pLhs)->getSubExpr();
    }

    if (isa<::clang::ArraySubscriptExpr>(pLhs))
    {
      // At least 2-dimensional array found => Look for the declaration reference to the image
      ::clang::ArraySubscriptExpr *pChildArraySubscript = dyn_cast<::clang::ArraySubscriptExpr>(pLhs);

      // Look through implicit cast expressions
      pLhs = pChildArraySubscript->getLHS();
      while (isa<::clang::ImplicitCastExpr>(pLhs))
      {
        pLhs = dyn_cast<::clang::ImplicitCastExpr>(pLhs)->getSubExpr();
      }

      if (isa<::clang::DeclRefExpr>(pLhs))
      {
        // Found a 2-dimensional array access => check if the array if the specified image
        ::clang::DeclRefExpr* pArrayDeclRef = dyn_cast<::clang::DeclRefExpr>(pLhs);

        if (pArrayDeclRef->getNameInfo().getAsString() == crstrImageName)
        {
          lstImageAccesses.push_back(pRootArraySubscript);
        }
      }
    }
  }

  // Parse all children everytime (in case an image access is used as an index expression for another image access)
  for (auto itChild = pStatement->child_begin(); itChild != pStatement->child_end(); itChild++)
  {
    std::list<::clang::ArraySubscriptExpr*> lstImageAccessesInternal = _FindImageAccesses(crstrImageName, *itChild);

    if (!lstImageAccessesInternal.empty())
    {
      lstImageAccesses.insert(lstImageAccesses.end(), lstImageAccessesInternal.begin(), lstImageAccessesInternal.end());
    }
  }

  return lstImageAccesses;
}

void CPU_x86::CodeGenerator::ImageAccessTranslator::_LinearizeImageAccess(const std::string &crstrImageName, ::clang::ArraySubscriptExpr *pImageAccessRoot)
{
  // Find the horizontal and vertical index expression of the 2-dimensional image access
  ::clang::Expr *pIndexExprX = pImageAccessRoot->getRHS();
  ::clang::Expr *pIndexExprY = pImageAccessRoot->getLHS();
  {
    while (! isa<::clang::ArraySubscriptExpr>(pIndexExprY))
    {
      pIndexExprY = dyn_cast<::clang::Expr>(*pIndexExprY->child_begin());
    }

    pIndexExprY = dyn_cast<::clang::ArraySubscriptExpr>(pIndexExprY)->getRHS();
  }

  // The image pointer have been re-routed to the current pixel => strip the reference to the global pixel ID
  pIndexExprX = _SubtractReference(pIndexExprX, _pDRGidX);
  pIndexExprY = _SubtractReference(pIndexExprY, _pDRGidY);


  // Build the final 1-dimensional index expression
  ::clang::Expr *pFinalIndexExpression = nullptr;
  if (pIndexExprY == nullptr)     // No vertical index expression required
  {
    if (pIndexExprX == nullptr)     // Neither horizontal nor vertical index required => access to the current pixel
    {
      pFinalIndexExpression = _ASTHelper.CreateIntegerLiteral(0);
    }
    else                            // Only horizontal index required => Access inside the current row
    {
      pFinalIndexExpression = pIndexExprX;
    }
  }
  else                            // Vertical index required
  {
    // Account for the row offset necessary for linear memory indexing
    ::clang::DeclRefExpr *pDRImageStride = _rHipaccHelper.GetImageParameterDecl(crstrImageName, HipaccHelper::ImageParamType::Stride);
    _rHipaccHelper.MarkParamUsed(pDRImageStride->getNameInfo().getAsString());

    pIndexExprY = _ASTHelper.CreateBinaryOperator(pIndexExprY, pDRImageStride, BO_Mul, pIndexExprY->getType());


    if (pIndexExprX == nullptr)     // Only vertical index required => Access inside the current column
    {
      pFinalIndexExpression = pIndexExprY;
    }
    else                            // Both horizontal and vertical index required => Add up both index expressions
    {
      pFinalIndexExpression = _ASTHelper.CreateBinaryOperator(pIndexExprY, pIndexExprX, BO_Add, pIndexExprY->getType());
    }
  }

  // Create the final 1-dimensional array subscript expression
  pImageAccessRoot->setLHS( _rHipaccHelper.GetImageParameterDecl(crstrImageName, HipaccHelper::ImageParamType::Buffer) );
  pImageAccessRoot->setRHS( pFinalIndexExpression );
}

::clang::Expr* CPU_x86::CodeGenerator::ImageAccessTranslator::_SubtractReference(::clang::Expr *pExpression, ::clang::DeclRefExpr *pDRSubtrahend)
{
  std::string        strStripVarName   = pDRSubtrahend->getNameInfo().getAsString();
  unsigned int  uiReferenceCount  = _ASTHelper.CountNumberOfReferences(pExpression, strStripVarName);

  ::clang::Expr *pReturnExpr = nullptr;

  if (uiReferenceCount == 1)    // One reference to stripping variable found => Try to remove it completely
  {
    if (_ASTHelper.IsSingleBranchStatement(pExpression))        // The expression refers only to the stripping variable => Entire expression is obsolete
    {
      return nullptr;
    }
    else if (_TryRemoveReference(pExpression, strStripVarName)) // Try to remove the stripping variable from the expression
    {
      pReturnExpr = pExpression;
    }
    else                                                        // The stripping variable could not be removed => subtract it to ensure the correct result
    {
      pReturnExpr = _ASTHelper.CreateBinaryOperator(pExpression, pDRSubtrahend, ::clang::BO_Sub, pExpression->getType());
    }
  }
  else                          // Either none or more than one reference found => subtract the stripping variable to ensure the correct result
  {
    pReturnExpr = _ASTHelper.CreateBinaryOperator(pExpression, pDRSubtrahend, ::clang::BO_Sub, pExpression->getType());
  }

  return _ASTHelper.CreateParenthesisExpression(pReturnExpr);
}

bool CPU_x86::CodeGenerator::ImageAccessTranslator::_TryRemoveReference(::clang::Expr *pExpression, std::string strStripVarName)
{
  // Try to find the bottom-most operator referencing the stripping variable
  ::clang::Expr           *pCurrentExpression = pExpression;
  ::clang::BinaryOperator *pBottomOperator    = nullptr;

  while (true)
  {
    if (isa<::clang::CastExpr>(pCurrentExpression))             // Current node is a cast expression => Step to the subexpression
    {
      pCurrentExpression = dyn_cast<::clang::CastExpr>(pCurrentExpression)->getSubExpr();
    }
    else if (isa<::clang::ParenExpr>(pCurrentExpression))       // Current node is a parenthesis expression => Step to the subexpression
    {
      pCurrentExpression = dyn_cast<::clang::ParenExpr>(pCurrentExpression)->getSubExpr();
    }
    else if (isa<::clang::DeclRefExpr>(pCurrentExpression))     // Current node is a declaration reference => Check for the stripping variable
    {
      if (dyn_cast<::clang::DeclRefExpr>(pCurrentExpression)->getNameInfo().getAsString() == strStripVarName)
      {
        // Found the reference to the stripping variable => Break here and remove it
        break;
      }
      else  // Found a wrong declaration reference (something went wrong) => Stripping failed
      {
        return false;
      }
    }
    else if (isa<::clang::BinaryOperator>(pCurrentExpression))  // Found a binary operator => Step into its correct branch
    {
      ::clang::BinaryOperator *pCurrentOperator = dyn_cast<::clang::BinaryOperator>(pCurrentExpression);

      if (pCurrentOperator->getOpcode() == ::clang::BO_Add)       // Found an addition => Both branches are supported
      {
        if (_ASTHelper.CountNumberOfReferences(pCurrentOperator->getLHS(), strStripVarName) == 1)
        {
          // Found the reference in the left-hand-branch => Step into it
          pCurrentExpression = pCurrentOperator->getLHS();
        }
        else if (_ASTHelper.CountNumberOfReferences(pCurrentOperator->getRHS(), strStripVarName) == 1)
        {
          // Found the reference in the right-hand-branch => Step into it
          pCurrentExpression = pCurrentOperator->getRHS();
        }
        else  // Something went wrong => Stripping failed
        {
          return false;
        }
      }
      else if (pCurrentOperator->getOpcode() == ::clang::BO_Sub)  // Found a substraction => Only the left-hand-branch is supported for the stripping
      {
        if (_ASTHelper.CountNumberOfReferences(pCurrentOperator->getLHS(), strStripVarName) == 1)
        {
          // Found the reference in the left-hand-branch => Step into it
          pCurrentExpression = pCurrentOperator->getLHS();
        }
        else  // Reference is not in the left-hand-branch => Stripping failed
        {
          return false;
        }
      }
      else  // The type of the binary operator is not supported => Stripping failed
      {
        return false;
      }

      // Set the current operator as the bottom-most operator
      pBottomOperator = pCurrentOperator;
    }
    else                                                        // Found an unsupported expression => Stripping failed
    {
      return false;
    }
  }


  if (pBottomOperator != nullptr)   // Found the bottom-most binary operator referencing the stripping variable
  {
    // Replace the operator branch containing the stripping variable with zero (note: this operator could be reduced, but then its parent would need to be changed)
    ::clang::Expr *pZeroLiteral = _ASTHelper.CreateIntegerLiteral(0);

    if (_ASTHelper.CountNumberOfReferences(pBottomOperator->getLHS(), strStripVarName) == 1)
    {
      pBottomOperator->setLHS(pZeroLiteral);
    }
    else if (_ASTHelper.CountNumberOfReferences(pBottomOperator->getRHS(), strStripVarName) == 1)
    {
      pBottomOperator->setRHS(pZeroLiteral);
    }

    return true;
  }
  else                              // Could not find the bottom-most reference => Stripping failed
  {
    return false;
  }
}

CPU_x86::CodeGenerator::ImageAccessTranslator::ImageLinePosDeclPairType CPU_x86::CodeGenerator::ImageAccessTranslator::CreateImageLineAndPosDecl(std::string strImageName)
{
  ImageLinePosDeclPairType LinePosDeclPair;

  ::clang::DeclRefExpr  *pImageDeclRef    = _rHipaccHelper.GetImageParameterDecl(strImageName, HipaccHelper::ImageParamType::Buffer);
  ::clang::FunctionDecl *pKernelFunction  = _rHipaccHelper.GetKernelFunction();

  // Fetch the required image access and pointer types
  QualType qtArrayAccessType, qtImagePointerType;
  {
    qtArrayAccessType = pImageDeclRef->getType()->getPointeeType();
    QualType qtElementType = qtArrayAccessType->getAsArrayTypeUnsafe()->getElementType();

    if (_rHipaccHelper.GetImageAccess(strImageName) == READ_ONLY)
    {
      qtElementType.addConst();
    }

    qtImagePointerType = _ASTHelper.GetPointerType(qtElementType);
  }


  // Create the declaration of the currently processed image line, e.g. "float *Output_CurrentLine = Output[gid_y];"
  {
    HipaccAccessor* pImage = _rHipaccHelper.GetImageFromMapping(strImageName);
    ::clang::DeclRefExpr *pStride = pImage->getStrideDecl();
    _rHipaccHelper.MarkParamUsed(pStride->getNameInfo().getAsString());
    ::clang::BinaryOperator *pOffset          = _ASTHelper.CreateBinaryOperator(pStride, _pDRGidY, BO_Mul, _pDRGidY->getType());
    ::clang::BinaryOperator *pPointerAddition = _ASTHelper.CreateBinaryOperator(pImageDeclRef, pOffset, BO_Add, qtImagePointerType);
    LinePosDeclPair.first                     = _ASTHelper.CreateVariableDeclaration(pKernelFunction, strImageName + std::string("_CurrentLine"), qtImagePointerType, pPointerAddition);
  }

  ::clang::DeclRefExpr *pImageLineDeclRef = _ASTHelper.CreateDeclarationReferenceExpression(LinePosDeclPair.first);


  // Create the declaration of the currently processed image pixel, e.g. "float *Output_CurrentPos = Output_CurrentLine + gid_x;"
  ::clang::BinaryOperator *pPointerAddition = _ASTHelper.CreateBinaryOperator(pImageLineDeclRef, _pDRGidX, BO_Add, qtImagePointerType);
  LinePosDeclPair.second                    = _ASTHelper.CreateVariableDeclaration(pKernelFunction, strImageName + std::string("_CurrentPos"), qtImagePointerType, pPointerAddition);

  return LinePosDeclPair;
}

void CPU_x86::CodeGenerator::ImageAccessTranslator::TranslateImageAccesses(::clang::Stmt *pStatement)
{
  ::clang::FunctionDecl *pKernelFunction = _rHipaccHelper.GetKernelFunction();

  // Parse through all kernel images
  for (unsigned int i = 0; i < pKernelFunction->getNumParams(); ++i)
  {
    ::clang::ParmVarDecl  *pParamDecl   = pKernelFunction->getParamDecl(i);
    std::string                strParamName  = pParamDecl->getNameAsString();

    // Skip all kernel function parameters, which are unused or do not refer to an HIPAcc image
    if ((!_rHipaccHelper.IsParamUsed(strParamName)) || (_rHipaccHelper.GetImageFromMapping(strParamName) == nullptr))
    {
      continue;
    }

    // Find all access to the current image
    std::list<::clang::ArraySubscriptExpr*>  lstImageAccesses = _FindImageAccesses(strParamName, pStatement);

    // Linearize all found image accesses
    for (auto itImageAccess : lstImageAccesses)
    {
      _LinearizeImageAccess(strParamName, itImageAccess);
    }
  }
}

void CPU_x86::CodeGenerator::ImageAccessTranslator::TranslateImageDeclarations(::clang::FunctionDecl *pFunctionDecl, ImageDeclarationTypes eDeclType)
{
  // Parse through all kernel images
  for (unsigned int i = 0; i < pFunctionDecl->getNumParams(); ++i)
  {
    ::clang::ParmVarDecl *pParamDecl = pFunctionDecl->getParamDecl(i);
    std::string          strParamName = pParamDecl->getNameAsString();

    // Skip all kernel function parameters, which are unused or do not refer to an HIPAcc image
    if ((!_rHipaccHelper.IsParamUsed(strParamName)) || (_rHipaccHelper.GetImageFromMapping(strParamName) == nullptr))
    {
      continue;
    }

    ::clang::DeclRefExpr  *pImageDeclRef = _rHipaccHelper.GetImageParameterDecl(strParamName, HipaccHelper::ImageParamType::Buffer);

    // Get the actual pixel type
    QualType qtElementType;
    {
      QualType qtArrayAccessType = pImageDeclRef->getType()->getPointeeType();
      qtElementType = qtArrayAccessType->getAsArrayTypeUnsafe()->getElementType();

      if (_rHipaccHelper.GetImageAccess(strParamName) == READ_ONLY)
      {
        qtElementType.addConst();
      }
    }

    // Build the desired new declaration type
    QualType qtFinalImageType;
    switch (eDeclType)
    {
    case ImageDeclarationTypes::NativePointer:
      {
        qtFinalImageType = _ASTHelper.GetPointerType(qtElementType);
      }
      break;
    case ImageDeclarationTypes::ConstantArray:
      {
        HipaccImage *pImage = _rHipaccHelper.GetImageFromMapping(strParamName)->getImage();

        QualType qtArrayType = _ASTHelper.GetConstantArrayType( qtElementType, static_cast<size_t>(pImage->getSizeX()) );
        qtFinalImageType     = _ASTHelper.GetConstantArrayType( qtArrayType,   static_cast<size_t>(pImage->getSizeY()) );
      }
      break;
    default:    throw InternalErrorException("Unknown image declaration type");
    }

    // Replace the delcaration type of the image
    pParamDecl->setType(qtFinalImageType);
  }
}



// Implementation of class CPU_x86::CodeGenerator
CPU_x86::CodeGenerator::CodeGenerator(::clang::hipacc::CompilerOptions *pCompilerOptions) : BaseType(pCompilerOptions, Descriptor())
{
  _InitSwitch<KnownSwitches::EmitPadding      >(CompilerSwitchTypeEnum::EmitPadding);
  _InitSwitch<KnownSwitches::InstructionSet   >(CompilerSwitchTypeEnum::InstructionSet);
  _InitSwitch<KnownSwitches::Parallelize      >(CompilerSwitchTypeEnum::Parallelize);
  _InitSwitch<KnownSwitches::RowsPerThread    >(CompilerSwitchTypeEnum::RowsPerThread);
  _InitSwitch<KnownSwitches::UnrollVectorLoops>(CompilerSwitchTypeEnum::UnrollVectorLoops);
  _InitSwitch<KnownSwitches::VectorizeKernel  >(CompilerSwitchTypeEnum::VectorizeKernel);
  _InitSwitch<KnownSwitches::VectorWidth      >(CompilerSwitchTypeEnum::VectorWidth);

  _eInstructionSet    = InstructionSetEnum::Array;
  _bUnrollVectorLoops = true;
  _bVectorizeKernel   = false;
  _bParallelize       = false;
  _szVectorWidth      = static_cast<size_t>(0);
}

size_t CPU_x86::CodeGenerator::_HandleSwitch(CompilerSwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex)
{
  std::string  strCurrentSwitch  = rvecArguments[szCurrentIndex];
  size_t  szReturnIndex     = szCurrentIndex;

  switch (eSwitch)
  {
  case CompilerSwitchTypeEnum::EmitPadding:
    {
      GetCompilerOptions().setPadding(_ParseOption<KnownSwitches::EmitPadding>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::InstructionSet:
    _eInstructionSet = _ParseOption<KnownSwitches::InstructionSet>(rvecArguments, szCurrentIndex);
    ++szReturnIndex;
    break;
  case CompilerSwitchTypeEnum::Parallelize:
    {
      ::clang::hipacc::CompilerOption eOption = _ParseOption<KnownSwitches::Parallelize>(rvecArguments, szCurrentIndex);

      _bParallelize = (eOption == USER_ON);
      GetCompilerOptions().setOpenMP(_bParallelize);
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::RowsPerThread:
    {
      GetCompilerOptions().setPixelsPerThread(_ParseOption<KnownSwitches::RowsPerThread>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::UnrollVectorLoops:
    {
      ::clang::hipacc::CompilerOption eOption = _ParseOption<KnownSwitches::UnrollVectorLoops>(rvecArguments, szCurrentIndex);

      _bUnrollVectorLoops = (eOption == USER_ON);

      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::VectorizeKernel:
    {
      ::clang::hipacc::CompilerOption eOption = _ParseOption< KnownSwitches::VectorizeKernel >(rvecArguments, szCurrentIndex);
      if (eOption == USER_ON) {
        GetCompilerOptions().setVectorizeKernels(USER_ON);
        _bVectorizeKernel = true;
      }
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::VectorWidth:
    {
      int iRequestedVecWidth = _ParseOption<KnownSwitches::VectorWidth>(rvecArguments, szCurrentIndex);
      if (iRequestedVecWidth <= 0)
      {
        throw RuntimeErrors::InvalidOptionException(KnownSwitches::VectorWidth::Key(), rvecArguments[szCurrentIndex + 1]);
      }

      _szVectorWidth = static_cast<size_t>( iRequestedVecWidth );
      ++szReturnIndex;
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strCurrentSwitch, GetName());
  }

  return szReturnIndex;
}


::clang::ForStmt* CPU_x86::CodeGenerator::_CreateIterationSpaceLoop(ClangASTHelper &rAstHelper, ::clang::DeclRefExpr *pLoopCounter, ::clang::Expr *pUpperLimit, ::clang::Stmt *pLoopBody, const size_t szIncrement)
{
  ::clang::Stmt* pFinalLoopBody = pLoopBody;
  if (! isa<::clang::CompoundStmt>(pFinalLoopBody))
  {
    pFinalLoopBody = rAstHelper.CreateCompoundStatement(pLoopBody);
  }

  ::clang::DeclStmt   *pInitStatement = rAstHelper.CreateDeclarationStatement(pLoopCounter);
  ::clang::Expr       *pCondition     = rAstHelper.CreateBinaryOperatorLessThan(pLoopCounter, pUpperLimit);
  ::clang::Expr       *pIncrement;

  if (szIncrement == 1) {
    pIncrement = rAstHelper.CreatePostIncrementOperator(pLoopCounter);
  } else {
    pIncrement = rAstHelper.CreateBinaryOperator(pLoopCounter, rAstHelper.CreateIntegerLiteral(static_cast<int>(szIncrement)), BO_AddAssign, pLoopCounter->getType());
  }

  return ASTNode::createForStmt(rAstHelper.GetASTContext(), pInitStatement, pCondition, pIncrement, pFinalLoopBody);
}


::clang::IfStmt* CPU_x86::CodeGenerator::_CreateBoundaryBranching(ClangASTHelper &rASTHelper, HipaccHelper &rHipaccHelper, ::clang::DeclRefExpr *pGidX, ::clang::DeclRefExpr *pGidY, int iSizeX, int iSizeY, ::clang::Stmt *pThenStmt, ::clang::Stmt *pElseStmt) {
  const ::clang::QualType cqtCmpType = rASTHelper.CreateBoolLiteral(true)->getType();
  const ::clang::QualType cqtIntType = rASTHelper.CreateIntegerLiteral(0)->getType();

  return rASTHelper.CreateIfStatement(
      rASTHelper.CreateBinaryOperator(
        rASTHelper.CreateBinaryOperator(
          rASTHelper.CreateBinaryOperator(pGidX, rASTHelper.CreateIntegerLiteral(iSizeX), clang::BinaryOperatorKind::BO_GE, cqtCmpType),
          rASTHelper.CreateBinaryOperator(pGidX, rASTHelper.CreateBinaryOperator(rHipaccHelper.GetIterationSpaceLimitX(), rASTHelper.CreateIntegerLiteral(iSizeX), clang::BinaryOperatorKind::BO_Sub, cqtIntType), clang::BinaryOperatorKind::BO_LT, cqtCmpType),
          clang::BinaryOperatorKind::BO_LAnd, cqtCmpType),
        rASTHelper.CreateBinaryOperator(
          rASTHelper.CreateBinaryOperator(pGidY, rASTHelper.CreateIntegerLiteral(iSizeY), clang::BinaryOperatorKind::BO_GE, cqtCmpType),
          rASTHelper.CreateBinaryOperator(pGidY, rASTHelper.CreateBinaryOperator(rHipaccHelper.GetIterationSpaceLimitY(), rASTHelper.CreateIntegerLiteral(iSizeY), clang::BinaryOperatorKind::BO_Sub, cqtIntType), clang::BinaryOperatorKind::BO_LT, cqtCmpType),
          clang::BinaryOperatorKind::BO_LAnd, cqtCmpType),
        clang::BinaryOperatorKind::BO_LAnd, cqtCmpType),
      pThenStmt, pElseStmt);
}


std::string CPU_x86::CodeGenerator::_GetImageDeclarationString(std::string strName, HipaccMemory *pHipaccMemoryObject, bool bConstPointer)
{
  std::stringstream FormatStream;

  if (bConstPointer)
  {
    FormatStream << "const ";
  }

  FormatStream << pHipaccMemoryObject->getTypeStr() << "* " << strName;

  return FormatStream.str();
}


InstructionSetBasePtr CPU_x86::CodeGenerator::_CreateInstructionSet(::clang::ASTContext &rAstContext)
{
  switch ( _eInstructionSet )
  {
  case InstructionSetEnum::SSE:       return InstructionSetBase::Create<InstructionSetSSE   >(rAstContext);
  case InstructionSetEnum::SSE_2:     return InstructionSetBase::Create<InstructionSetSSE2  >(rAstContext);
  case InstructionSetEnum::SSE_3:     return InstructionSetBase::Create<InstructionSetSSE3  >(rAstContext);
  case InstructionSetEnum::SSSE_3:    return InstructionSetBase::Create<InstructionSetSSSE3 >(rAstContext);
  case InstructionSetEnum::SSE_4_1:   return InstructionSetBase::Create<InstructionSetSSE4_1>(rAstContext);
  case InstructionSetEnum::SSE_4_2:   return InstructionSetBase::Create<InstructionSetSSE4_2>(rAstContext);
  case InstructionSetEnum::AVX:       return InstructionSetBase::Create<InstructionSetAVX   >(rAstContext);
  case InstructionSetEnum::AVX_2:     return InstructionSetBase::Create<InstructionSetAVX2  >(rAstContext);
  default:                            throw InternalErrorException("Unexpected instruction set selected!");
  }
}

std::string CPU_x86::CodeGenerator::_FormatFunctionHeader(FunctionDecl *pFunctionDecl, HipaccHelper &rHipaccHelper, bool bCheckUsage, bool bPrintActualImageType)
{
  std::vector<std::string> vecParamStrings;

  // Translate function parameters to declaration strings
  for (unsigned int i = 0; i < pFunctionDecl->getNumParams(); ++i)
  {
    ::clang::ParmVarDecl  *pParamDecl = pFunctionDecl->getParamDecl(i);
    std::string strName(pParamDecl->getNameAsString());

    if ( bCheckUsage && (!rHipaccHelper.IsParamUsed(strName)) )
    {
      continue;
    }

    // Translate argument, dependent on its type
    if (HipaccMask *pMask = rHipaccHelper.GetMaskFromMapping(strName))                // check if we have a Mask or Domain
    {
      if (!pMask->isConstant())
      {
        vecParamStrings.push_back( _GetImageDeclarationString(pMask->getName(), pMask, true) );
      }
    }
    else if (HipaccAccessor *pAccessor = rHipaccHelper.GetImageFromMapping(strName))  // check if we have an Accessor
    {
      if (bPrintActualImageType)
      {
        pParamDecl->getType().getAsStringInternal(strName, GetPrintingPolicy());
        vecParamStrings.push_back( strName );
      }
      else
      {
        vecParamStrings.push_back( _GetImageDeclarationString(strName, pAccessor->getImage(), rHipaccHelper.GetImageAccess(strName) == READ_ONLY) );
      }
    }
    else                                                                              // normal arguments
    {
      std::string strParamBuffer;
      llvm::raw_string_ostream ParamStream(strParamBuffer);

      pParamDecl->getType().getAsStringInternal(strName, GetPrintingPolicy());
      ParamStream << strName;

      // default arguments ...
      if (Expr *Init = pParamDecl->getInit())
      {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init);

        if (!CCE || CCE->getConstructor()->isCopyConstructor())
        {
          ParamStream << " = ";
        }

        Init->printPretty(ParamStream, 0, GetPrintingPolicy(), 0);
      }

      vecParamStrings.push_back( ParamStream.str() );
    }
  }


  std::stringstream OutputStream;

  // Write function name and qualifiers
  OutputStream << pFunctionDecl->getReturnType().getAsString(GetPrintingPolicy()) << " " << pFunctionDecl->getNameAsString() << "(";

  // Write all parameters with comma delimiters
  if (! vecParamStrings.empty())
  {
    OutputStream << vecParamStrings[0];

    for (size_t i = static_cast<size_t>(1); i < vecParamStrings.size(); ++i)
    {
      OutputStream << ", " << vecParamStrings[i];
    }
  }

  OutputStream << ") ";

  return OutputStream.str();
}

std::string CPU_x86::CodeGenerator::_GetInstructionSetIncludeFile()
{
  switch (_eInstructionSet)
  {
  case InstructionSetEnum::SSE:     return "xmmintrin.h";
  case InstructionSetEnum::SSE_2:   return "emmintrin.h";
  case InstructionSetEnum::SSE_3:   return "pmmintrin.h";
  case InstructionSetEnum::SSSE_3:  return "tmmintrin.h";
  case InstructionSetEnum::SSE_4_1: return "smmintrin.h";
  case InstructionSetEnum::SSE_4_2: return "nmmintrin.h";
  case InstructionSetEnum::AVX:     return "immintrin.h";
  case InstructionSetEnum::AVX_2:   return "immintrin.h";
  default:  return "";
  }
}

size_t CPU_x86::CodeGenerator::_GetVectorWidth(Vectorization::AST::FunctionDeclarationPtr spVecFunction)
{
  if (_eInstructionSet == InstructionSetEnum::Array)
  {
    if (_szVectorWidth == static_cast<size_t>(0))
    {
      if (GetCompilerOptions().printVerbose()) {
        llvm::errs() << "\nNOTE: No vector width for array export selected => Set default value \"4\"\n\n";
      }
      _szVectorWidth = static_cast<size_t>(4);
    }
  }
  else
  {
    size_t cszMaxVecWidth = static_cast<size_t>(0);
    switch (_eInstructionSet)
    {
    case InstructionSetEnum::SSE:
    case InstructionSetEnum::SSE_2:
    case InstructionSetEnum::SSE_3:
    case InstructionSetEnum::SSSE_3:
    case InstructionSetEnum::SSE_4_1:
    case InstructionSetEnum::SSE_4_2:   cszMaxVecWidth = static_cast<size_t>(16); break;
    case InstructionSetEnum::AVX:
    case InstructionSetEnum::AVX_2:     cszMaxVecWidth = static_cast<size_t>(32); break;
    default:                            throw InternalErrorException("Unexpected instruction set ID!");
    }

    if (_szVectorWidth > cszMaxVecWidth)
    {
      llvm::errs() << "\nWARNING: Selected vector width exceeds the maximum width for the instruction set => Clipping vector width to \"" << cszMaxVecWidth << "\"\n\n";
      _szVectorWidth = cszMaxVecWidth;
    }
    else
    {
      // Compute the minimum width dependend on the size of the image element types
      size_t szMinVecWidth = static_cast<size_t>(1);
      {
        size_t szMinTypeSize = cszMaxVecWidth;

        std::vector<std::string> vecVariableNames = spVecFunction->GetKnownVariableNames();

        for (auto itVarName : vecVariableNames)
        {
          Vectorization::AST::BaseClasses::VariableInfoPtr  spParamInfo   = spVecFunction->GetVariableInfo( itVarName );
          AST::BaseClasses::TypeInfo::KnownTypes            eElementType  = spParamInfo->GetTypeInfo().GetType();

          if ( spParamInfo->GetVectorize() && (eElementType != AST::BaseClasses::TypeInfo::KnownTypes::Bool) )
          {
            size_t szCurrentTypeSize  = Vectorization::AST::BaseClasses::TypeInfo::GetTypeSize( eElementType );
            szMinTypeSize             = std::min( szMinTypeSize, szCurrentTypeSize );
          }
        }

        szMinVecWidth = cszMaxVecWidth / szMinTypeSize;
      }

      if (_szVectorWidth == static_cast<size_t>(0))
      {
        if (GetCompilerOptions().printVerbose()) {
          llvm::errs() << "\nNOTE: No vector width for the instruction set selected => Set kernel-based minimum value \"" << szMinVecWidth << "\"\n\n";
        }
        _szVectorWidth = szMinVecWidth;
      }
      else if (_szVectorWidth < szMinVecWidth)
      {
        llvm::errs() << "\nWARNING: Selected vector width is below the minimum width for the instruction set => Set kernel-based minimum value \"" << szMinVecWidth << "\"\n\n";
        _szVectorWidth = szMinVecWidth;
      }
      else
      {
        for (size_t szCurWidth = szMinVecWidth; szCurWidth <= cszMaxVecWidth; szCurWidth <<= 1)
        {
          if (_szVectorWidth == szCurWidth)
          {
            break;
          }
          else if (_szVectorWidth < szCurWidth)
          {
            llvm::errs() << "\nWARNING: The selected vector width for the instruction set must be a power of 2 => Promote width \"" << _szVectorWidth << "\" to \"" << szCurWidth << "\"\n\n";
            _szVectorWidth = szCurWidth;
            break;
          }
        }
      }
    }
  }

  return _szVectorWidth;
}

::clang::FunctionDecl* CPU_x86::CodeGenerator::_VectorizeKernelSubFunction(const std::string cstrKernelName, FunctionDecl *pSubFunction, HipaccHelper &rHipaccHelper, llvm::raw_ostream &rOutputStream)
{
  #ifdef DUMP_VAST_CONTENTS
  #define DUMP_VAST(__RootNode, __Filename)   Vectorizer::DumpVASTNodeToXML( __RootNode, __Filename )
  #else
  #define DUMP_VAST(__RootNode, __Filename)
  #endif


  try
  {
    Vectorization::Vectorizer Vectorizer;

    Vectorization::AST::FunctionDeclarationPtr spVecFunction = Vectorizer.ConvertClangFunctionDecl(pSubFunction);

    spVecFunction->SetName( cstrKernelName );

    DUMP_VAST( spVecFunction, "Dump_1.xml" );


    Vectorizer.RemoveUnnecessaryConversions(spVecFunction);
    DUMP_VAST( spVecFunction, "Dump_2.xml" );


    Vectorizer.FlattenScopeTrees(spVecFunction);
    DUMP_VAST( spVecFunction, "Dump_3.xml" );


    // Vectorize the kernel sub-function
    {
      // Mark all HIPAcc images as vectorized variables
      for (unsigned int uiParam = 0; uiParam < pSubFunction->getNumParams(); ++uiParam)
      {
        std::string strParamName = pSubFunction->getParamDecl(uiParam)->getNameAsString();

        if (rHipaccHelper.GetImageFromMapping(strParamName) != nullptr)
        {
          Vectorization::AST::BaseClasses::VariableInfoPtr spVariableInfo = spVecFunction->GetVariableInfo(strParamName);
          if (! spVariableInfo)
          {
            throw InternalErrorException(std::string("Missing vectorization parameter: ") + strParamName);
          }

          spVariableInfo->SetVectorize(true);
        }
      }

      // Mark the horizontal global ID as vectorized variable if it used by the kernel sub-function
      Vectorization::AST::BaseClasses::VariableInfoPtr spGidXInfo = spVecFunction->GetVariableInfo(rHipaccHelper.GlobalIdX());
      if (spGidXInfo)
      {
        spGidXInfo->SetVectorize(true);
      }
    }

    Vectorizer.VectorizeFunction(spVecFunction);
    DUMP_VAST( spVecFunction, "Dump_4.xml" );


    Vectorizer.RebuildControlFlow(spVecFunction);
    DUMP_VAST( spVecFunction, "Dump_5.xml" );


    Vectorizer.FlattenMemoryAccesses(spVecFunction);
    DUMP_VAST( spVecFunction, "Dump_6.xml" );


    // Convert vectorized function parameters
    {
      Vectorization::AST::ScopePtr spVecFunctionBody = spVecFunction->GetBody();

      for (Vectorization::AST::IndexType iParamIdx = static_cast<Vectorization::AST::IndexType>(0); iParamIdx < spVecFunction->GetParameterCount(); ++iParamIdx)
      {
        Vectorization::AST::Expressions::IdentifierPtr    spParam     = spVecFunction->GetParameter( iParamIdx );
        Vectorization::AST::BaseClasses::VariableInfoPtr  spParamInfo = spParam->LookupVariableInfo();

        // Each vectorized single value parameter will be converted into a scalar parameter and a new vectorized variable will be created
        if (spParamInfo->GetVectorize() && spParamInfo->GetTypeInfo().IsSingleValue())
        {
          std::string strParamName = spParam->GetName();

          // Replace parameter
          Vectorization::AST::BaseClasses::VariableInfoPtr spNewParamInfo = Vectorization::AST::BaseClasses::VariableInfo::Create( spParamInfo->GetName() + std::string("_base"), spParamInfo->GetTypeInfo(), false );
          Vectorization::AST::Expressions::IdentifierPtr   spNewParam     = Vectorization::AST::Expressions::Identifier::Create( spNewParamInfo->GetName() );

          spVecFunction->SetParameter(iParamIdx, spNewParamInfo);


          // Create the assignment expression for the new variable
          Vectorization::AST::Expressions::AssignmentOperatorPtr  spAssignment    = Vectorization::AST::Expressions::AssignmentOperator::Create( spParam );
          Vectorization::AST::VectorSupport::BroadCastPtr         spBaseBroadCast = Vectorization::AST::VectorSupport::BroadCast::Create( spNewParam );

          if (strParamName == HipaccHelper::GlobalIdX())
          {
            typedef Vectorization::AST::Expressions::ArithmeticOperator::ArithmeticOperatorType   OperatorType;

            // The horizontal global id must be incremental vector
            Vectorization::AST::VectorSupport::VectorIndexPtr       spVectorIndex = Vectorization::AST::VectorSupport::VectorIndex::Create( spParamInfo->GetTypeInfo().GetType() );
            Vectorization::AST::Expressions::ArithmeticOperatorPtr  spAddOperator = Vectorization::AST::Expressions::ArithmeticOperator::Create( OperatorType::Add, spBaseBroadCast, spVectorIndex );

            spAssignment->SetRHS(spAddOperator);
          }
          else
          {
            // Every other variable will be initialized with the base value
            spAssignment->SetRHS(spBaseBroadCast);
          }

          // Add the declaration and the assignment statement
          spVecFunctionBody->AddVariableDeclaration(spParamInfo);
          spVecFunctionBody->InsertChild(0, spAssignment);
        }
      }
    }
    DUMP_VAST( spVecFunction, "Dump_7.xml" );

    Vectorizer.RebuildDataFlow(spVecFunction, _eInstructionSet != InstructionSetEnum::Array);
    DUMP_VAST( spVecFunction, "Dump_8.xml" );


    ::clang::ASTContext &rAstContext    = pSubFunction->getASTContext();
    const size_t        cszVectorWidth  = _GetVectorWidth(spVecFunction);

    if (_eInstructionSet == InstructionSetEnum::Array)
    {
      return Vectorizer.ConvertVASTFunctionDecl( spVecFunction, cszVectorWidth, rAstContext, _bUnrollVectorLoops );
    }
    else
    {
      #ifdef DUMP_INSTRUCTION_SETS
      DumpInstructionSet( rAstContext, "Dump_IS.cpp", _eInstructionSet );
      #endif

      VASTExportInstructionSet Exporter( cszVectorWidth, rAstContext, _CreateInstructionSet(rAstContext) );

      ::clang::FunctionDecl *pExportedKernelFunction = Exporter.ExportVASTFunction( spVecFunction );


      // Print all generated helper functions
      for (auto itHelperFunction : Exporter.GetGeneratedHelperFunctions())
      {
        rOutputStream << "\n";
        if (!itHelperFunction->hasBody()) {
          // Forward declaration
          itHelperFunction->print( rOutputStream, GetPrintingPolicy() );
          rOutputStream << ";";
        } else {
          // Function definition/implementation
          rOutputStream << "inline ";
          itHelperFunction->print( rOutputStream, GetPrintingPolicy() );
        }
      }

      return pExportedKernelFunction;
    }
  }
  catch (std::exception &e)
  {
    llvm::errs() << "\n\nERROR: " << e.what() << "\n\n";
    exit(EXIT_FAILURE);
  }

  #undef DUMP_VAST
}


FunctionDecl *CPU_x86::CodeGenerator::_CreateKernelFunctionWithoutBH(FunctionDecl *pKernelFunction, HipaccKernel *pKernel)
{
  clang::ASTContext &astContext = pKernelFunction->getASTContext();

  // Copy existing kernel
  HipaccKernelClass *pKernelClass = pKernel->getKernelClass();
  FunctionDecl *pKernelFunctionNoBH = ASTNode::createFunctionDecl(astContext,
      astContext.getTranslationUnitDecl(), pKernel->getKernelName(),
      astContext.VoidTy, pKernel->getArgTypes(), pKernel->getDeviceArgNames());
  hipacc::Builtin::Context cBuiltins(astContext);

  // Disable boundary handling for all images
  for (auto img : pKernelClass->getImgFields()) {
    HipaccAccessor *pAcc = pKernel->getImgFromMapping(img);
    HipaccBoundaryCondition *pBC = pAcc->getBC();
    pBC->setBoundaryMode(Boundary::UNDEFINED);
  }

  // Copy of Hipacc kernel to avoid reset of used variables
  HipaccKernel *pKernelNoBH = new HipaccKernel(*pKernel);

  // Rerun AST translation with disabled boundary handling
  ASTTranslate *Hipacc = new ASTTranslate(astContext, pKernelFunctionNoBH, pKernelNoBH, pKernelClass, cBuiltins, GetCompilerOptions());
  pKernelFunctionNoBH->setBody(Hipacc->Hipacc(pKernelClass->getKernelFunction()->getBody()));

  // Cleanup
  delete pKernelNoBH;

  return pKernelFunctionNoBH;
}


void CPU_x86::CodeGenerator::Configure(CommonDefines::ArgumentVectorType & rvecArguments)
{
  GetCompilerOptions().setTargetDevice(Device::CPU);
  CodeGeneratorBaseImplT::Configure(rvecArguments);
}


CommonDefines::ArgumentVectorType CPU_x86::CodeGenerator::GetAdditionalClangArguments() const
{
  CommonDefines::ArgumentVectorType vecArguments;

  // Add required macro definition which toggle the correct include files
  switch (_eInstructionSet)   // The case-fall-through is intenden here
  {
  case InstructionSetEnum::AVX_2:     vecArguments.push_back("-D __AVX2__");
  case InstructionSetEnum::AVX:       vecArguments.push_back("-D __AVX__");
  case InstructionSetEnum::SSE_4_2:   vecArguments.push_back("-D __SSE4_2__");
  case InstructionSetEnum::SSE_4_1:   vecArguments.push_back("-D __SSE4_1__");
  case InstructionSetEnum::SSSE_3:    vecArguments.push_back("-D __SSSE3__");
  case InstructionSetEnum::SSE_3:     vecArguments.push_back("-D __SSE3__");
  case InstructionSetEnum::SSE_2:     vecArguments.push_back("-D __SSE2__");
  case InstructionSetEnum::SSE:
    {
      vecArguments.push_back("-D __SSE__");
      vecArguments.push_back("-D __MMX__");

      // Add the common intrinsics header
      vecArguments.push_back("-includeimmintrin.h");  // FIXME: Due to a bug in the clang command arguments parser the space between option and switch is missing

      // Enable 64bit extensions of the SSE instruction sets
      vecArguments.push_back("-D __x86_64__");

      break;
    }
  default:                            break;    // Useless default branch avoiding GCC compiler warnings
  }

  return vecArguments;
}

bool CPU_x86::CodeGenerator::PrintKernelFunction(FunctionDecl *pKernelFunction, HipaccKernel *pKernel, llvm::raw_ostream &rOutputStream)
{
  HipaccHelper          hipaccHelper(pKernelFunction, pKernel);
  ImageAccessTranslator ImgAccessTranslator(hipaccHelper);

  // Print the instruction set include directive
  if (_bVectorizeKernel)
  {
    std::string strIncludeFile = _GetInstructionSetIncludeFile();

    if (! strIncludeFile.empty())
    {
      rOutputStream << "#include <" << strIncludeFile << ">\n";
    }
  }

  rOutputStream << "\n\n";

  // Add the iteration space loops
  {
    ClangASTHelper  ASTHelper(pKernelFunction->getASTContext());

    DeclRefExpr *gid_x_ref = ASTHelper.FindDeclaration(pKernelFunction, HipaccHelper::GlobalIdX());
    DeclRefExpr *gid_y_ref = ASTHelper.FindDeclaration(pKernelFunction, HipaccHelper::GlobalIdY());

    ::clang::CallExpr *pSubFuncCallScalar     = nullptr;
    ::clang::CallExpr *pSubFuncCallVectorized = nullptr;

    ::clang::CallExpr *pSubFuncCallScalarNoBH     = nullptr;
    ::clang::CallExpr *pSubFuncCallVectorizedNoBH = nullptr;

    // If vectorization is enabled, split the kernel function into the "iteration space"-part and the "pixel-wise processing"-part
    if (_bVectorizeKernel)
    {
      ImgAccessTranslator.TranslateImageAccesses(pKernelFunction->getBody());

      // Push loop body to own function
      ::clang::Stmt *pKernelBody = pKernelFunction->getBody();

      KernelSubFunctionBuilder SubFuncBuilder(ASTHelper.GetASTContext());

      SubFuncBuilder.ImportUsedParameters(pKernelFunction, pKernelBody);

      if ( SubFuncBuilder.IsVariableUsed(HipaccHelper::GlobalIdY(), pKernelBody) )
      {
        SubFuncBuilder.AddCallParameter(gid_y_ref, true);
      }

      if ( SubFuncBuilder.IsVariableUsed(HipaccHelper::GlobalIdX(), pKernelBody) )
      {
        SubFuncBuilder.AddCallParameter(gid_x_ref, true);
      }


      KernelSubFunctionBuilder::DeclCallPairType  DeclCallPair = SubFuncBuilder.CreateFuntionDeclarationAndCall(pKernelFunction->getNameAsString() + std::string("_Scalar"), pKernelFunction->getReturnType());
      ImgAccessTranslator.TranslateImageDeclarations(DeclCallPair.first, ImageAccessTranslator::ImageDeclarationTypes::NativePointer);
      DeclCallPair.first->setBody(pKernelBody);


      // Create function call reference for the scalar sub-function
      pSubFuncCallScalar = DeclCallPair.second;


      // Print the new kernel body sub-function
      rOutputStream << "inline " << _FormatFunctionHeader(DeclCallPair.first, hipaccHelper, false, true);
      DeclCallPair.first->getBody()->printPretty(rOutputStream, 0, GetPrintingPolicy(), 0);
      rOutputStream << "\n\n";


      // Vectorize the kernel sub-function and print it
      ::clang::FunctionDecl *pVecSubFunction = _VectorizeKernelSubFunction(pKernelFunction->getNameAsString() + std::string("_Vectorized"), DeclCallPair.first, hipaccHelper, rOutputStream);
      pSubFuncCallVectorized = ASTHelper.CreateFunctionCall(pVecSubFunction, SubFuncBuilder.GetCallParameters());

      rOutputStream << "inline " << _FormatFunctionHeader(pVecSubFunction, hipaccHelper, false, true);
      pVecSubFunction->getBody()->printPretty(rOutputStream, 0, GetPrintingPolicy(), 0);
      rOutputStream << "\n\n";


      // Create kernel variants without boundary handling for local operators
      if (pKernel->getMaxSizeX() > 0 || pKernel->getMaxSizeY() > 0) {
        // Replicate existing kernel with disabled boundary handling enforced
        FunctionDecl *pKernelFunctionNoBH = _CreateKernelFunctionWithoutBH(pKernelFunction, pKernel);
        ImgAccessTranslator.TranslateImageAccesses(pKernelFunctionNoBH->getBody());


        KernelSubFunctionBuilder::DeclCallPairType  DeclCallPairNoBH = SubFuncBuilder.CreateFuntionDeclarationAndCall(pKernelFunctionNoBH->getNameAsString() + std::string("_Scalar_NoBH"), pKernelFunctionNoBH->getReturnType());
        ImgAccessTranslator.TranslateImageDeclarations(DeclCallPairNoBH.first, ImageAccessTranslator::ImageDeclarationTypes::NativePointer);
        DeclCallPairNoBH.first->setBody(pKernelFunctionNoBH->getBody());


        // Create function call reference for the scalar sub-function
        pSubFuncCallScalarNoBH = DeclCallPairNoBH.second;


        // Print the new kernel body sub-function
        rOutputStream << "inline " << _FormatFunctionHeader(DeclCallPairNoBH.first, hipaccHelper, false, true);
        DeclCallPairNoBH.first->getBody()->printPretty(rOutputStream, 0, GetPrintingPolicy(), 0);
        rOutputStream << "\n\n";


        // Vectorize the kernel sub-function and print it
        ::clang::FunctionDecl *pVecSubFunctionNoBH = _VectorizeKernelSubFunction(pKernelFunctionNoBH->getNameAsString() + std::string("_Vectorized_NoBH"), DeclCallPairNoBH.first, hipaccHelper, rOutputStream);
        pSubFuncCallVectorizedNoBH = ASTHelper.CreateFunctionCall(pVecSubFunctionNoBH, SubFuncBuilder.GetCallParameters());

        rOutputStream << "inline " << _FormatFunctionHeader(pVecSubFunctionNoBH, hipaccHelper, false, true);
        pVecSubFunctionNoBH->getBody()->printPretty(rOutputStream, 0, GetPrintingPolicy(), 0);
        rOutputStream << "\n\n";
      }
    }


    // Create the iteration space
    ClangASTHelper::StatementVectorType vecKernelFunctionBody;
    ClangASTHelper::StatementVectorType vecOuterLoopBody;

    {
      ClangASTHelper::StatementVectorType vecInnerLoopBody;

      if (_bVectorizeKernel)
      {
        // If vectorization is enabled, redirect all image pointers for the internal kernel function to the currently processed pixel
        for (unsigned int i = 0; i < pKernelFunction->getNumParams(); ++i)
        {
          ::clang::ParmVarDecl  *pParamDecl   = pKernelFunction->getParamDecl(i);
          std::string                strParamName  = pParamDecl->getNameAsString();

          // Skip all kernel function parameters, which are unused or to not refer to an HIPAcc image
          if ((! hipaccHelper.IsParamUsed(strParamName)) || (hipaccHelper.GetImageFromMapping(strParamName) == nullptr))
          {
            continue;
          }

          // Create declaration for "current line" and "current pixel" image pointers and add them to the iteration space loops
          ImageAccessTranslator::ImageLinePosDeclPairType LinePosDeclPair = ImgAccessTranslator.CreateImageLineAndPosDecl(strParamName);

          vecOuterLoopBody.push_back( ASTHelper.CreateDeclarationStatement(LinePosDeclPair.first) );
          vecInnerLoopBody.push_back( ASTHelper.CreateDeclarationStatement(LinePosDeclPair.second) );


          // Replace all references to the HIPAcc image by the "current pixel" pointer
          ASTHelper.ReplaceDeclarationReferences( pKernelFunction->getBody(), strParamName, LinePosDeclPair.second );
          ASTHelper.ReplaceDeclarationReferences( pSubFuncCallScalar,         strParamName, LinePosDeclPair.second );
          ASTHelper.ReplaceDeclarationReferences( pSubFuncCallVectorized,     strParamName, LinePosDeclPair.second );
          ASTHelper.ReplaceDeclarationReferences((Stmt*)pSubFuncCallScalar, strParamName, LinePosDeclPair.second);
          ASTHelper.ReplaceDeclarationReferences((Stmt*)pSubFuncCallVectorized, strParamName, LinePosDeclPair.second);
          if (pSubFuncCallScalarNoBH) {
            ASTHelper.ReplaceDeclarationReferences((Stmt*)pSubFuncCallScalarNoBH, strParamName, LinePosDeclPair.second);
          }
          if (pSubFuncCallVectorizedNoBH) {
            ASTHelper.ReplaceDeclarationReferences((Stmt*)pSubFuncCallVectorizedNoBH, strParamName, LinePosDeclPair.second);
          }
        }

        // Compute the iteration space range, which must be handled by the scalar sub-function
        ::clang::DeclRefExpr  *pScalarRangeRef = nullptr;
        {
          ::clang::VarDecl *pGidXDecl = dyn_cast<::clang::VarDecl>(gid_x_ref->getDecl());

          ::clang::DeclRefExpr  *pIterationSpaceWidth = hipaccHelper.GetImageParameterDecl(pKernelFunction->getParamDecl(0)->getNameAsString(), HipaccHelper::ImageParamType::Width);
          ::clang::QualType     qtReturnType          = pIterationSpaceWidth->getType();

          ::clang::IntegerLiteral *pVectorWidth = ASTHelper.CreateIntegerLiteral(static_cast<int>(_szVectorWidth));
          ::clang::Expr           *pInitExpr    = ASTHelper.CreateBinaryOperator(pIterationSpaceWidth, pVectorWidth, ::clang::BO_Rem, qtReturnType);

          pInitExpr = ASTHelper.CreateParenthesisExpression(pInitExpr);
          pInitExpr = ASTHelper.CreateBinaryOperator(pGidXDecl->getInit(), pInitExpr, ::clang::BO_Add, qtReturnType);

          qtReturnType.addConst();

          ::clang::ValueDecl *pScalarRangeDecl = ASTHelper.CreateVariableDeclaration(pKernelFunction, "is_scalar_range", qtReturnType, pInitExpr);
          vecKernelFunctionBody.push_back(ASTHelper.CreateDeclarationStatement(pScalarRangeDecl));

          pScalarRangeRef = ASTHelper.CreateDeclarationReferenceExpression(pScalarRangeDecl);
        }

        // Create the horizontal iteration space loops and push them to the vertical loop body
        {
          ::clang::ValueDecl    *pNewGidXDecl = ASTHelper.CreateVariableDeclaration(pKernelFunction, HipaccHelper::GlobalIdX(), gid_x_ref->getType(), pScalarRangeRef);
          ::clang::DeclRefExpr  *pNewGidXRef = ASTHelper.CreateDeclarationReferenceExpression(pNewGidXDecl);

          // Add the loop for the scalar function call
          if (!pSubFuncCallScalarNoBH) {
            vecInnerLoopBody.push_back(pSubFuncCallScalar);
          } else {
            vecInnerLoopBody.push_back(_CreateBoundaryBranching(ASTHelper, hipaccHelper, gid_x_ref, gid_y_ref, pKernel->getMaxSizeX(), pKernel->getMaxSizeY(), pSubFuncCallScalarNoBH, pSubFuncCallScalar));
          }
          vecOuterLoopBody.push_back( _CreateIterationSpaceLoop(ASTHelper, gid_x_ref, pScalarRangeRef, ASTHelper.CreateCompoundStatement(vecInnerLoopBody)) );

          // Add the loop for the vectorized function call
          vecInnerLoopBody.pop_back();
          if (!pSubFuncCallVectorizedNoBH) {
            vecInnerLoopBody.push_back(pSubFuncCallVectorized);
          } else {
            vecInnerLoopBody.push_back(_CreateBoundaryBranching(ASTHelper, hipaccHelper, gid_x_ref, gid_y_ref, static_cast<int>((pKernel->getMaxSizeX() + _szVectorWidth-1) / _szVectorWidth * _szVectorWidth), pKernel->getMaxSizeY(), pSubFuncCallVectorizedNoBH, pSubFuncCallVectorized));
          }
          vecOuterLoopBody.push_back( _CreateIterationSpaceLoop(ASTHelper, pNewGidXRef, hipaccHelper.GetIterationSpaceLimitX(), ASTHelper.CreateCompoundStatement(vecInnerLoopBody), _szVectorWidth) );
        }
      }
      else
      {
        // Create the horizontal iteration space loop and push it to the vertical loop body
        ForStmt *pInnerLoop = _CreateIterationSpaceLoop(ASTHelper, gid_x_ref, hipaccHelper.GetIterationSpaceLimitX(), pKernelFunction->getBody());
        vecOuterLoopBody.push_back(pInnerLoop);
      }
    }

    // Create the vertical iteration space loop
    ForStmt *pOuterLoop = _CreateIterationSpaceLoop(ASTHelper, gid_y_ref, hipaccHelper.GetIterationSpaceLimitY(), ASTHelper.CreateCompoundStatement(vecOuterLoopBody));

    // Add an OpenMP parallel for directive with outer iteration space loop as associated statement if parallelization is activated
    if (_bParallelize)
    {
      vecKernelFunctionBody.push_back( ASTHelper.CreateOpenMPDirectiveParallelFor(pOuterLoop, GetCompilerOptions().getPixelsPerThread()) );
    }
    else
    {
      vecKernelFunctionBody.push_back(pOuterLoop);
    }

    pKernelFunction->setBody( ASTHelper.CreateCompoundStatement(vecKernelFunctionBody) );
  }


  rOutputStream << _FormatFunctionHeader(pKernelFunction, hipaccHelper, true, false);

  // print kernel body
  pKernelFunction->getBody()->printPretty(rOutputStream, 0, GetPrintingPolicy(), 0);

  return true;
}


#ifdef DUMP_INSTRUCTION_SETS
#undef DUMP_INSTRUCTION_SETS
#endif

#ifdef DUMP_VAST_CONTENTS
#undef DUMP_VAST_CONTENTS
#endif


// vim: set ts=2 sw=2 sts=2 et ai:

