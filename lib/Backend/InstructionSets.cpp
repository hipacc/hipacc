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

//===--- InstructionSets.cpp - Definition of known vector instruction sets. ----------===//
//
// This file contains definitions of known vector instruction sets.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/BackendExceptions.h"
#include "hipacc/Backend/InstructionSets.h"
#include <sstream>

using namespace clang::hipacc::Backend::Vectorization;
using namespace clang::hipacc::Backend;
using namespace clang;
using namespace std;


// Implementation of class InstructionSetExceptions
string InstructionSetExceptions::IndexOutOfRange::_ConvertLimit(uint32_t uiUpperLimit)
{
  stringstream streamOutput;

  streamOutput << uiUpperLimit;

  return streamOutput.str();
}

InstructionSetExceptions::IndexOutOfRange::IndexOutOfRange(string strMethodType, VectorElementTypes eElementType, uint32_t uiUpperLimit) :
      BaseType( string("The index for a \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + string("\" element ") +
                strMethodType + string(" must be in the range of [0; ") + _ConvertLimit(uiUpperLimit) + string("] !"))
{
}

string InstructionSetExceptions::UnsupportedBuiltinFunctionType::_ConvertParamCount(uint32_t uiParamCount)
{
  stringstream streamOutput;

  streamOutput << uiParamCount;

  return streamOutput.str();
}

InstructionSetExceptions::UnsupportedBuiltinFunctionType::UnsupportedBuiltinFunctionType( VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType,
                                                                                          uint32_t uiParamCount, string strInstructionSetName ) :
      BaseType( string("The built-in function \"") + GetBuiltinFunctionTypeString(eFunctionType) + string("(") + _ConvertParamCount(uiParamCount) +
                string(")\" is not supported for element type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + string("\" in the instruction set \"") +
                strInstructionSetName + string("\" !") )
{
}

InstructionSetExceptions::UnsupportedConversion::UnsupportedConversion(VectorElementTypes eSourceType, VectorElementTypes eTargetType, string strInstructionSetName) :
      BaseType( string("A conversion from type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eSourceType) + string("\" to type \"") +
                AST::BaseClasses::TypeInfo::GetTypeString(eTargetType) + string("\" is not supported in the instruction set \"") + 
                strInstructionSetName + string("\" !") )
{
}



// Implementation of class InstructionSetBase
InstructionSetBase::InstructionSetBase(ASTContext &rAstContext, string strFunctionNamePrefix) : _ASTHelper(rAstContext), _strIntrinsicPrefix(strFunctionNamePrefix)
{
  const size_t cszPrefixLength = strFunctionNamePrefix.size();
  ClangASTHelper::FunctionDeclarationVectorType vecFunctionDecls = _ASTHelper.GetKnownFunctionDeclarations();

  for each (auto itFuncDecl in vecFunctionDecls)
  {
    string strFuncName = ClangASTHelper::GetFullyQualifiedFunctionName(itFuncDecl);

    bool bAddFuncDecl = true;
    if (! strFunctionNamePrefix.empty())
    {
      if (strFuncName.size() < cszPrefixLength)
      {
        bAddFuncDecl = false;
      }
      else if (strFuncName.substr(0, cszPrefixLength) != strFunctionNamePrefix)
      {
        bAddFuncDecl = false;
      }
    }

    if (bAddFuncDecl)
    {
      _mapKnownFuncDecls[strFuncName].push_back(itFuncDecl);
    }
  }
}

Expr* InstructionSetBase::_ConvertDown(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, bool bMaskConversion)
{
  const size_t cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType);
  const size_t cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType);

  if (cszSourceSize <= cszTargetSize)
  {
    throw RuntimeErrorException("The data size of the source type must be larger than the one of the target type for a downward conversion!");
  }
  else if (crvecVectorRefs.size() != static_cast<size_t>(cszSourceSize / cszTargetSize))
  {
    throw RuntimeErrorException("The number of arguments for the downward conversion must be equal to the size spread between source and target type!");
  }

  return _ConvertVector(eSourceType, eTargetType, crvecVectorRefs, 0, bMaskConversion);
}

Expr* InstructionSetBase::_ConvertSameSize(VectorElementTypes eSourceType, VectorElementTypes eTargetType, Expr *pVectorRef, bool bMaskConversion)
{
  const size_t cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize( eSourceType );
  const size_t cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize( eTargetType );

  if (cszSourceSize != cszTargetSize)
  {
    throw RuntimeErrorException("The data size of the source type and the target type must be equal for a same size conversion!");
  }

  ClangASTHelper::ExpressionVectorType vecVectorRefs;

  vecVectorRefs.push_back( pVectorRef );

  return _ConvertVector( eSourceType, eTargetType, vecVectorRefs, 0, bMaskConversion );
}

Expr* InstructionSetBase::_ConvertUp(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef, uint32_t uiGroupIndex, bool bMaskConversion)
{
  const size_t cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType);
  const size_t cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType);

  if (cszSourceSize >= cszTargetSize)
  {
    throw RuntimeErrorException("The data size of the source type must be smaller than the one of the target type for an upward conversion!");
  }
  else if (uiGroupIndex >= static_cast<uint32_t>(cszTargetSize / cszSourceSize))
  {
    throw RuntimeErrorException("The group index for the upward conversion must be smaller than the size spread between source and target type!");
  }

  ClangASTHelper::ExpressionVectorType vecVectorRefs;

  vecVectorRefs.push_back( pVectorRef );

  return _ConvertVector( eSourceType, eTargetType, vecVectorRefs, uiGroupIndex, bMaskConversion );
}

void InstructionSetBase::_CreateIntrinsicDeclaration(string strFunctionName, const QualType &crReturnType, const ClangASTHelper::QualTypeVectorType &crvecArgTypes, const ClangASTHelper::StringVectorType &crvecArgNames)
{
  if (_mapKnownFuncDecls.find(strFunctionName) == _mapKnownFuncDecls.end())
  {
    _mapKnownFuncDecls[ strFunctionName ].push_back( _ASTHelper.CreateFunctionDeclaration(strFunctionName, crReturnType, crvecArgNames, crvecArgTypes) );
  }
}

void InstructionSetBase::_CreateIntrinsicDeclaration(string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, string strArgName1)
{
  ClangASTHelper::QualTypeVectorType  vecArgTypes;
  ClangASTHelper::StringVectorType    vecArgNames;

  vecArgTypes.push_back( crArgType1 );
  vecArgNames.push_back( strArgName1 );

  _CreateIntrinsicDeclaration(strFunctionName, crReturnType, vecArgTypes, vecArgNames);
}

void InstructionSetBase::_CreateIntrinsicDeclaration(string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, string strArgName1, const QualType &crArgType2, string strArgName2)
{
  ClangASTHelper::QualTypeVectorType  vecArgTypes;
  ClangASTHelper::StringVectorType    vecArgNames;

  vecArgTypes.push_back( crArgType1 );
  vecArgNames.push_back( strArgName1 );

  vecArgTypes.push_back( crArgType2 );
  vecArgNames.push_back( strArgName2 );

  _CreateIntrinsicDeclaration(strFunctionName, crReturnType, vecArgTypes, vecArgNames);
}

void InstructionSetBase::_CreateIntrinsicDeclaration(string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, string strArgName1, const QualType &crArgType2, string strArgName2, const QualType &crArgType3, string strArgName3)
{
  ClangASTHelper::QualTypeVectorType  vecArgTypes;
  ClangASTHelper::StringVectorType    vecArgNames;

  vecArgTypes.push_back( crArgType1 );
  vecArgNames.push_back( strArgName1 );

  vecArgTypes.push_back( crArgType2 );
  vecArgNames.push_back( strArgName2 );

  vecArgTypes.push_back( crArgType3 );
  vecArgNames.push_back( strArgName3 );

  _CreateIntrinsicDeclaration(strFunctionName, crReturnType, vecArgTypes, vecArgNames);
}

void InstructionSetBase::_CreateMissingIntrinsicsSSE()
{
  // Get float vector type
  QualType  qtFloatVector = _GetFunctionReturnType("_mm_setzero_ps");

  // Create missing SSE intrinsic functions
  _CreateIntrinsicDeclaration( "_mm_ceil_ps",     qtFloatVector, qtFloatVector, "a" );
  _CreateIntrinsicDeclaration( "_mm_floor_ps",    qtFloatVector, qtFloatVector, "a" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_ps",  qtFloatVector, qtFloatVector, "a", qtFloatVector, "b", _GetClangType(VectorElementTypes::UInt32), "imm" );
}

void InstructionSetBase::_CreateMissingIntrinsicsSSE2()
{
  // Get required types
  QualType  qtDoubleVector  = _GetFunctionReturnType("_mm_setzero_pd");
  QualType  qtIntegerVector = _GetFunctionReturnType("_mm_setzero_si128");
  QualType  qtInt           = _GetClangType(VectorElementTypes::Int32);

  // Create missing SSE2 intrinsic functions
  _CreateIntrinsicDeclaration( "_mm_ceil_pd",         qtDoubleVector,  qtDoubleVector,  "a" );
  _CreateIntrinsicDeclaration( "_mm_floor_pd",        qtDoubleVector,  qtDoubleVector,  "a" );
  _CreateIntrinsicDeclaration( "_mm_slli_si128",      qtIntegerVector, qtIntegerVector, "a", qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_srli_si128",      qtIntegerVector, qtIntegerVector, "a", qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shufflehi_epi16", qtIntegerVector, qtIntegerVector, "a", qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shufflelo_epi16", qtIntegerVector, qtIntegerVector, "a", qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_epi32",   qtIntegerVector, qtIntegerVector, "a", qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_pd",      qtDoubleVector,  qtDoubleVector,  "a", qtDoubleVector, "b", qtInt, "imm" );
}

void InstructionSetBase::_CreateMissingIntrinsicsSSE4_1()
{
  // Get required types
  QualType  qtFloatVector   = GetVectorType(VectorElementTypes::Float);
  QualType  qtIntegerVector = GetVectorType(VectorElementTypes::Int32);
  QualType  qtInt64         = _GetClangType(VectorElementTypes::Int64);
  QualType  qtInt           = _GetClangType(VectorElementTypes::Int32);
  QualType  qtConstInt      = qtInt;
  qtConstInt.addConst();

  // Create missing SSE2 intrinsic functions
  _CreateIntrinsicDeclaration( "_mm_extract_ps",    qtInt,   qtFloatVector,   "a", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_extract_epi8",  qtInt,   qtIntegerVector, "a", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_extract_epi32", qtInt,   qtIntegerVector, "a", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_extract_epi64", qtInt64, qtIntegerVector, "a", qtConstInt, "imm");

  _CreateIntrinsicDeclaration( "_mm_insert_ps",    qtFloatVector,   qtFloatVector,   "a", qtFloatVector, "b", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_insert_epi8",  qtIntegerVector, qtIntegerVector, "a", qtInt,         "i", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_insert_epi32", qtIntegerVector, qtIntegerVector, "a", qtInt,         "i", qtConstInt, "imm");
  _CreateIntrinsicDeclaration( "_mm_insert_epi64", qtIntegerVector, qtIntegerVector, "a", qtInt64,       "i", qtConstInt, "imm");
}

CastExpr* InstructionSetBase::_CreatePointerCast(Expr *pPointerRef, const QualType &crNewPointerType)
{
  return _GetASTHelper().CreateReinterpretCast(pPointerRef, crNewPointerType, CK_ReinterpretMemberPointer);
}

CastExpr* InstructionSetBase::_CreateValueCast(Expr *pValueRef, const QualType &crNewValueType, CastKind eCastKind)
{
  return _GetASTHelper().CreateStaticCast(pValueRef, crNewValueType, eCastKind);
}

QualType InstructionSetBase::_GetClangType(VectorElementTypes eType)
{
  ASTContext &rContext = _GetASTHelper().GetASTContext();

  switch (eType)
  {
  case VectorElementTypes::Bool:    return rContext.BoolTy;
  case VectorElementTypes::Double:  return rContext.DoubleTy;
  case VectorElementTypes::Float:   return rContext.FloatTy;
  case VectorElementTypes::Int8:    return rContext.CharTy;
  case VectorElementTypes::UInt8:   return rContext.UnsignedCharTy;
  case VectorElementTypes::Int16:   return rContext.ShortTy;
  case VectorElementTypes::UInt16:  return rContext.UnsignedShortTy;
  case VectorElementTypes::Int32:   return rContext.IntTy;
  case VectorElementTypes::UInt32:  return rContext.UnsignedIntTy;
  case VectorElementTypes::Int64:   return rContext.LongLongTy;
  case VectorElementTypes::UInt64:  return rContext.UnsignedLongLongTy;
  default:                          throw InternalErrorException("Unsupported type detected!");
  }
}

ClangASTHelper::FunctionDeclarationVectorType InstructionSetBase::_GetFunctionDecl(string strFunctionName)
{
  auto itFunctionDecl = _mapKnownFuncDecls.find(strFunctionName);

  if (itFunctionDecl != _mapKnownFuncDecls.end())
  {
    return itFunctionDecl->second;
  }
  else
  {
    throw InternalErrorException(string("Cannot find function \"") + strFunctionName + string("\" !"));
  }
}

QualType InstructionSetBase::_GetFunctionReturnType(string strFuntionName)
{
  auto vecFunctionsDecls = _GetFunctionDecl(strFuntionName);

  if (vecFunctionsDecls.size() != static_cast<size_t>(1))
  {
    throw InternalErrorException(string("The function declaration \"") + strFuntionName + string("\" is ambiguous!"));
  }

  return vecFunctionsDecls.front()->getResultType();
}

ClangASTHelper::ExpressionVectorType InstructionSetBase::_SwapExpressionOrder(const ClangASTHelper::ExpressionVectorType &crvecExpressions)
{
  ClangASTHelper::ExpressionVectorType vecSwappedExpressions;

  for (auto itExpr = crvecExpressions.end(); itExpr != crvecExpressions.begin(); itExpr--)
  {
    vecSwappedExpressions.push_back( *(itExpr - 1) );
  }

  return vecSwappedExpressions;
}



// Implementation of class InstructionSetSSE
InstructionSetSSE::InstructionSetSSE(ASTContext &rAstContext) : InstructionSetBase(rAstContext, _GetIntrinsicPrefix())
{
  _InitIntrinsicsMap();

  _CreateMissingIntrinsicsSSE();  // Only required due to Clang's incomplete intrinsic headers

  _LookupIntrinsics();
}

void InstructionSetSSE::_CheckElementType(VectorElementTypes eElementType) const
{
  if (eElementType != VectorElementTypes::Float)
  {
    throw RuntimeErrorException(string("The element type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + string("\" is not supported in instruction set \"SSE\"!"));
  }
}

Expr* InstructionSetSSE::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  if ((eSourceType == VectorElementTypes::Float) && (eTargetType == VectorElementTypes::Float))
  {
    return crvecVectorRefs.front();   // Nothing to do
  }
  else
  {
    throw InstructionSetExceptions::UnsupportedConversion(eSourceType, eTargetType, "SSE");
  }
}

void InstructionSetSSE::_InitIntrinsicsMap()
{
  _InitIntrinsic( IntrinsicsSSEEnum::AddFloat,                    "add_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::AndFloat,                    "and_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::AndNotFloat,                 "andnot_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::BroadCastFloat,              "set1_ps"     );
  _InitIntrinsic( IntrinsicsSSEEnum::CeilFloat,                   "ceil_ps"     );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareEqualFloat,           "cmpeq_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareGreaterEqualFloat,    "cmpge_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareGreaterThanFloat,     "cmpgt_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareLessEqualFloat,       "cmple_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareLessThanFloat,        "cmplt_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareNotEqualFloat,        "cmpneq_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareNotGreaterEqualFloat, "cmpnge_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareNotGreaterThanFloat,  "cmpngt_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareNotLessEqualFloat,    "cmpnle_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::CompareNotLessThanFloat,     "cmpnlt_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::DivideFloat,                 "div_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::ExtractLowestFloat,          "cvtss_f32"   );
  _InitIntrinsic( IntrinsicsSSEEnum::FloorFloat,                  "floor_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::InsertLowestFloat,           "move_ss"     );
  _InitIntrinsic( IntrinsicsSSEEnum::LoadFloat,                   "loadu_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::MaxFloat,                    "max_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::MinFloat,                    "min_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::MoveFloatHighLow,            "movehl_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::MoveFloatLowHigh,            "movelh_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::MoveMaskFloat,               "movemask_ps" );
  _InitIntrinsic( IntrinsicsSSEEnum::MultiplyFloat,               "mul_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::OrFloat,                     "or_ps"       );
  _InitIntrinsic( IntrinsicsSSEEnum::ReciprocalFloat,             "rcp_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::ReciprocalSqrtFloat,         "rsqrt_ps"    );
  _InitIntrinsic( IntrinsicsSSEEnum::SetFloat,                    "set_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::SetZeroFloat,                "setzero_ps"  );
  _InitIntrinsic( IntrinsicsSSEEnum::ShuffleFloat,                "shuffle_ps"  );
  _InitIntrinsic( IntrinsicsSSEEnum::SqrtFloat,                   "sqrt_ps"     );
  _InitIntrinsic( IntrinsicsSSEEnum::StoreFloat,                  "storeu_ps"   );
  _InitIntrinsic( IntrinsicsSSEEnum::SubtractFloat,               "sub_ps"      );
  _InitIntrinsic( IntrinsicsSSEEnum::UnpackHighFloat,             "unpackhi_ps" );
  _InitIntrinsic( IntrinsicsSSEEnum::UnpackLowFloat,              "unpacklo_ps" );
  _InitIntrinsic( IntrinsicsSSEEnum::XorFloat,                    "xor_ps"      );
}

Expr* InstructionSetSSE::_CreateFullBitMask(VectorElementTypes eElementType)
{
  _CheckElementType(eElementType);

  return RelationalOperator( eElementType, RelationalOperatorType::Equal, CreateZeroVector(eElementType), CreateZeroVector(eElementType) );
}

Expr* InstructionSetSSE::_MergeVectors(VectorElementTypes eElementType, Expr *pVectorRef1, Expr *pVectorRef2, bool bLowHalf)
{
  _CheckElementType( eElementType );

  return _CreateFunctionCall( bLowHalf ? IntrinsicsSSEEnum::MoveFloatLowHigh : IntrinsicsSSEEnum::MoveFloatHighLow, pVectorRef1, pVectorRef2 );
}

Expr* InstructionSetSSE::_UnpackVectors(VectorElementTypes eElementType, Expr *pVectorRef1, Expr *pVectorRef2, bool bLowHalf)
{
  _CheckElementType(eElementType);
  
  return _CreateFunctionCall( bLowHalf ? IntrinsicsSSEEnum::UnpackLowFloat : IntrinsicsSSEEnum::UnpackHighFloat, pVectorRef1, pVectorRef2 );
}

Expr* InstructionSetSSE::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  _CheckElementType(eElementType);

  switch (eOpType)
  {
  case ArithmeticOperatorType::Add:         return _CreateFunctionCall(IntrinsicsSSEEnum::AddFloat,       pExprLHS, pExprRHS);
  case ArithmeticOperatorType::BitwiseAnd:  return _CreateFunctionCall(IntrinsicsSSEEnum::AndFloat,       pExprLHS, pExprRHS);
  case ArithmeticOperatorType::BitwiseOr:   return _CreateFunctionCall(IntrinsicsSSEEnum::OrFloat,        pExprLHS, pExprRHS);
  case ArithmeticOperatorType::BitwiseXOr:  return _CreateFunctionCall(IntrinsicsSSEEnum::XorFloat,       pExprLHS, pExprRHS);
  case ArithmeticOperatorType::Divide:      return _CreateFunctionCall(IntrinsicsSSEEnum::DivideFloat,    pExprLHS, pExprRHS);
  case ArithmeticOperatorType::Multiply:    return _CreateFunctionCall(IntrinsicsSSEEnum::MultiplyFloat,  pExprLHS, pExprRHS);
  case ArithmeticOperatorType::Subtract:    return _CreateFunctionCall(IntrinsicsSSEEnum::SubtractFloat,  pExprLHS, pExprRHS);
  case ArithmeticOperatorType::Modulo:      throw RuntimeErrorException("Modulo operation is undefined for floating point data types!");
  case ArithmeticOperatorType::ShiftLeft:
  case ArithmeticOperatorType::ShiftRight:  throw RuntimeErrorException("Shift operations are undefined for floating point data types!");
  default:                                  throw InternalErrorException("Unsupported arithmetic operation detected!");
  }
}

Expr* InstructionSetSSE::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  _CheckElementType(eElementType);

  Expr *pSelectTrue  = _CreateFunctionCall( IntrinsicsSSEEnum::AndFloat,    pMaskRef, pVectorTrue  );
  Expr *pSelectFalse = _CreateFunctionCall( IntrinsicsSSEEnum::AndNotFloat, pMaskRef, pVectorFalse );

  return _CreateFunctionCall( IntrinsicsSSEEnum::OrFloat, pSelectTrue, pSelectFalse );
}

Expr* InstructionSetSSE::BroadCast(VectorElementTypes eElementType, Expr *pBroadCastValue)
{
  _CheckElementType(eElementType);

  return _CreateFunctionCall(IntrinsicsSSEEnum::BroadCastFloat, pBroadCastValue);
}

Expr* InstructionSetSSE::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  _CheckElementType(eElementType);

  const uint32_t cuiParamCount = static_cast< uint32_t >( crvecArguments.size() );

  if (! IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount))
  {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "SSE");
  }

  switch (eFunctionType)
  {
  case BuiltinFunctionsEnum::Abs:
    {
      // No intrinsic for this => multiply with 1 or -1
      Expr *pMultiplier = RelationalOperator( eElementType, RelationalOperatorType::Less, crvecArguments[0], CreateZeroVector(eElementType) );
      pMultiplier       = BlendVectors( eElementType, pMultiplier, CreateOnesVector(eElementType, true), CreateOnesVector(eElementType, false) );

      return ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, crvecArguments[0], pMultiplier );
    }
  case BuiltinFunctionsEnum::Ceil:    return _CreateFunctionCall( IntrinsicsSSEEnum::CeilFloat,  crvecArguments[0] );
  case BuiltinFunctionsEnum::Floor:   return _CreateFunctionCall( IntrinsicsSSEEnum::FloorFloat, crvecArguments[0] );
  case BuiltinFunctionsEnum::Max:     return _CreateFunctionCall( IntrinsicsSSEEnum::MaxFloat,   crvecArguments[0], crvecArguments[1] );
  case BuiltinFunctionsEnum::Min:     return _CreateFunctionCall( IntrinsicsSSEEnum::MinFloat,   crvecArguments[0], crvecArguments[1] );
  case BuiltinFunctionsEnum::Sqrt:    return _CreateFunctionCall( IntrinsicsSSEEnum::SqrtFloat,  crvecArguments[0] );
  default:                            throw InternalErrorException("Unknown built-in function type detected!");
  }
}

Expr* InstructionSetSSE::CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, Expr *pMaskExpr)
{
  _CheckElementType(eMaskElementType);

  int32_t             iTestValue      = (eCheckType == ActiveElementsCheckType::All) ? 0xF   : 0;
  BinaryOperatorKind  eCompareOpType  = (eCheckType == ActiveElementsCheckType::Any) ? BO_NE : BO_EQ;

  CallExpr        *pMoveMask      = _CreateFunctionCall(IntrinsicsSSEEnum::MoveMaskFloat, pMaskExpr);
  IntegerLiteral  *pTestConstant  = _GetASTHelper().CreateIntegerLiteral(iTestValue);

  return _GetASTHelper().CreateBinaryOperator( pMoveMask, pTestConstant, eCompareOpType, _GetClangType(VectorElementTypes::Bool) );
}

Expr* InstructionSetSSE::CreateOnesVector(VectorElementTypes eElementType, bool bNegative)
{
  _CheckElementType(eElementType);

  return BroadCast( eElementType, _GetASTHelper().CreateLiteral(bNegative ? -1.f : 1.f) );
}

Expr* InstructionSetSSE::CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder)
{
  _CheckElementType(eElementType);

  if (! bReversedOrder)
  {
    // SSE vector creation methods expect the arguments in reversed order
    return CreateVector( eElementType, _SwapExpressionOrder( crvecElements ), true );
  }

  if (crvecElements.size() != GetVectorElementCount(eElementType))
  {
    throw RuntimeErrorException("The number of init expressions must be equal to the vector element count!");
  }

  return _CreateFunctionCall( IntrinsicsSSEEnum::SetFloat, crvecElements );
}

Expr* InstructionSetSSE::CreateZeroVector(VectorElementTypes eElementType)
{
  _CheckElementType(eElementType);

  return _CreateFunctionCall(IntrinsicsSSEEnum::SetZeroFloat);
}

Expr* InstructionSetSSE::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  _CheckElementType(eElementType);
  _CheckExtractIndex(eElementType, uiIndex);

  Expr *pIntermediateValue = nullptr;

  if (uiIndex == 0)
  {
    // The lowest element is requested => it can be extracted directly
    pIntermediateValue = pVectorRef;
  }
  else
  {
    // Swap vector elements such that the desired value is in the lowest element
    int32_t iControlFlags = 0;

    switch (uiIndex)
    {
    case 1:   iControlFlags = 0xE1;   break;  // Swap element 0 and 1
    case 2:   iControlFlags = 0xC6;   break;  // Swap element 0 and 2
    case 3:   iControlFlags = 0x27;   break;  // Swap element 0 and 3
    default:  throw InternalErrorException("Unexpected index detected!");
    }

    IntegerLiteral *pControlFlags = _GetASTHelper().CreateIntegerLiteral(iControlFlags);

    pIntermediateValue = _CreateFunctionCall(IntrinsicsSSEEnum::ShuffleFloat, pVectorRef, CreateZeroVector(eElementType), pControlFlags);
  }

  return _CreateFunctionCall(IntrinsicsSSEEnum::ExtractLowestFloat, pIntermediateValue);
}

QualType InstructionSetSSE::GetVectorType(VectorElementTypes eElementType)
{
  _CheckElementType(eElementType);

  return _GetFunctionReturnType(IntrinsicsSSEEnum::SetZeroFloat);
}

Expr* InstructionSetSSE::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  _CheckElementType(eElementType);
  _CheckInsertIndex(eElementType, uiIndex);

  Expr *pBroadCast = BroadCast(eElementType, pElementValue);

  if (uiIndex == 0)
  {
    return _CreateFunctionCall(IntrinsicsSSEEnum::InsertLowestFloat, pVectorRef, pBroadCast);
  }
  else
  {
    int32_t iControlFlags = 0;

    switch (uiIndex)
    {
    case 1:   iControlFlags = 0xE1;   break;  // Swap element 0 and 1
    case 2:   iControlFlags = 0xC6;   break;  // Swap element 0 and 2
    case 3:   iControlFlags = 0x27;   break;  // Swap element 0 and 3
    default:  throw InternalErrorException("Unexpected index detected!");
    }

    Expr *pInsertExpr = _CreateFunctionCall( IntrinsicsSSEEnum::ShuffleFloat,      pVectorRef,  pVectorRef, _GetASTHelper().CreateIntegerLiteral(iControlFlags) );
    pInsertExpr       = _CreateFunctionCall( IntrinsicsSSEEnum::InsertLowestFloat, pInsertExpr, pBroadCast );

    if (uiIndex == 1)
    {
      pInsertExpr = _CreateFunctionCall( IntrinsicsSSEEnum::ShuffleFloat, pInsertExpr, pVectorRef, _GetASTHelper().CreateIntegerLiteral(iControlFlags) );
    }
    else
    {
      iControlFlags = (uiIndex == 2) ? 0xC4 : 0x24;
      pInsertExpr   = _CreateFunctionCall( IntrinsicsSSEEnum::ShuffleFloat, pVectorRef, pInsertExpr, _GetASTHelper().CreateIntegerLiteral(iControlFlags) );
    }

    return pInsertExpr;
  }
}

bool InstructionSetSSE::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  if (eElementType != VectorElementTypes::Float)
  {
    return false;
  }

  switch (eFunctionType)
  {
  case BuiltinFunctionsEnum::Abs:     return (uiParamCount == 1);
  case BuiltinFunctionsEnum::Ceil:    return (uiParamCount == 1);
  case BuiltinFunctionsEnum::Floor:   return (uiParamCount == 1);
  case BuiltinFunctionsEnum::Max:     return (uiParamCount == 2);
  case BuiltinFunctionsEnum::Min:     return (uiParamCount == 2);
  case BuiltinFunctionsEnum::Sqrt:    return (uiParamCount == 1);
  }

  return false;
}

bool InstructionSetSSE::IsElementTypeSupported(VectorElementTypes eElementType) const
{
  return (eElementType == VectorElementTypes::Float);
}

Expr* InstructionSetSSE::LoadVector(VectorElementTypes eElementType, Expr *pPointerRef)
{
  _CheckElementType(eElementType);

  return _CreateFunctionCall(IntrinsicsSSEEnum::LoadFloat, pPointerRef);
}

Expr* InstructionSetSSE::LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex)
{
  throw RuntimeErrorException( "Gathered vector loads are not supported in the SSE instruction set!" );
}

Expr* InstructionSetSSE::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  _CheckElementType(eElementType);

  switch (eOpType)
  {
  case RelationalOperatorType::Equal:         return _CreateFunctionCall( IntrinsicsSSEEnum::CompareEqualFloat,         pExprLHS, pExprRHS );
  case RelationalOperatorType::Greater:       return _CreateFunctionCall( IntrinsicsSSEEnum::CompareGreaterThanFloat,   pExprLHS, pExprRHS );
  case RelationalOperatorType::GreaterEqual:  return _CreateFunctionCall( IntrinsicsSSEEnum::CompareGreaterEqualFloat,  pExprLHS, pExprRHS );
  case RelationalOperatorType::Less:          return _CreateFunctionCall( IntrinsicsSSEEnum::CompareLessThanFloat,      pExprLHS, pExprRHS );
  case RelationalOperatorType::LessEqual:     return _CreateFunctionCall( IntrinsicsSSEEnum::CompareLessEqualFloat,     pExprLHS, pExprRHS );
  case RelationalOperatorType::NotEqual:      return _CreateFunctionCall( IntrinsicsSSEEnum::CompareNotEqualFloat,      pExprLHS, pExprRHS );
  case RelationalOperatorType::LogicalAnd:    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseAnd, pExprLHS, pExprRHS );
  case RelationalOperatorType::LogicalOr:     return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseOr,  pExprLHS, pExprRHS );
  default:                                    throw InternalErrorException( "Unsupported relational operation detected!" );
  }
}

Expr* InstructionSetSSE::ShiftElements(VectorElementTypes eElementType, Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount)
{
  if (uiCount == 0)
  {
    return pVectorRef;  // Nothing to do
  }
  else
  {
    throw RuntimeErrorException("Shift operations are undefined for floating point data types!");
  }
}

Expr* InstructionSetSSE::StoreVector(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue)
{
  _CheckElementType(eElementType);

  return _CreateFunctionCall(IntrinsicsSSEEnum::StoreFloat, pPointerRef, pVectorValue);
}

Expr* InstructionSetSSE::StoreVectorMasked(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue, Expr *pMaskRef)
{
  _CheckElementType(eElementType);

  return StoreVector( eElementType, pPointerRef, BlendVectors(eElementType, pMaskRef, pVectorValue, LoadVector(eElementType, pPointerRef) ) );
}

Expr* InstructionSetSSE::UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, Expr *pSubExpr)
{
  _CheckElementType(eElementType);

  switch (eOpType)
  {
  case UnaryOperatorType::AddressOf:      return _GetASTHelper().CreateUnaryOperator( pSubExpr, UO_AddrOf, _GetASTHelper().GetASTContext().getPointerType(pSubExpr->getType()) );
  case UnaryOperatorType::BitwiseNot: case UnaryOperatorType::LogicalNot:
    {
      return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseXOr, pSubExpr, _CreateFullBitMask(eElementType) );
    }
  case UnaryOperatorType::Minus:          return ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, pSubExpr, CreateOnesVector(eElementType, true) );
  case UnaryOperatorType::Plus:           return pSubExpr;
  case UnaryOperatorType::PostDecrement:  return _CreatePostfixedUnaryOp( IntrinsicsSSEEnum::SubtractFloat, eElementType, pSubExpr );
  case UnaryOperatorType::PostIncrement:  return _CreatePostfixedUnaryOp( IntrinsicsSSEEnum::AddFloat,      eElementType, pSubExpr );
  case UnaryOperatorType::PreDecrement:   return _CreatePrefixedUnaryOp( IntrinsicsSSEEnum::SubtractFloat,  eElementType, pSubExpr );
  case UnaryOperatorType::PreIncrement:   return _CreatePrefixedUnaryOp( IntrinsicsSSEEnum::AddFloat,       eElementType, pSubExpr );
  default:                                throw InternalErrorException("Unsupported unary operation detected!");
  }
}



// Implementation of class InstructionSetSSE2
InstructionSetSSE2::InstructionSetSSE2(ASTContext &rAstContext) : BaseType(rAstContext)
{
  _InitIntrinsicsMap();

  _CreateMissingIntrinsicsSSE2();  // Only required due to Clang's incomplete intrinsic headers

  _LookupIntrinsics();
}

Expr* InstructionSetSSE2::_ArithmeticOpInteger(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  switch (eOpType)
  {
  case ArithmeticOperatorType::Add:
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _CreateFunctionCall( IntrinsicsSSE2Enum::AddInt8,  pExprLHS, pExprRHS );
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::AddInt16, pExprLHS, pExprRHS );
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::AddInt32, pExprLHS, pExprRHS );
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE2Enum::AddInt64, pExprLHS, pExprRHS );
    default:                                                          throw InternalErrorException("Unsupported vector element type detected!");
    }
  case ArithmeticOperatorType::BitwiseAnd:  return _CreateFunctionCall( IntrinsicsSSE2Enum::AndInteger, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::BitwiseOr:   return _CreateFunctionCall( IntrinsicsSSE2Enum::OrInteger,  pExprLHS, pExprRHS );
  case ArithmeticOperatorType::BitwiseXOr:  return _CreateFunctionCall( IntrinsicsSSE2Enum::XorInteger, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::Divide:      return _SeparatedArithmeticOpInteger( eElementType, BO_Div, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::Multiply:
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      {
        // Convert vector to a signed / unsigned 16-bit integer intermediate type, do the multiplication and convert back
        const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2, AST::BaseClasses::TypeInfo::IsSigned(eElementType)).GetType();

        ClangASTHelper::ExpressionVectorType vecResultVectors;

        for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx)
        {
          Expr *pConvertedLHS = ConvertVectorUp( eElementType, ceIntermediateType, pExprLHS, uiGroupIdx );
          Expr *pConvertedRHS = ConvertVectorUp( eElementType, ceIntermediateType, pExprRHS, uiGroupIdx );

          vecResultVectors.push_back( ArithmeticOperator( ceIntermediateType, eOpType, pConvertedLHS, pConvertedRHS ) );
        }

        return ConvertVectorDown( ceIntermediateType, eElementType, vecResultVectors );
      }
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::MultiplyInt16, pExprLHS, pExprRHS );
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      {
        ClangASTHelper::ExpressionVectorType vecHalfVectors;

        // Multiply the low half of the vector
        vecHalfVectors.push_back( _CreateFunctionCall( IntrinsicsSSE2Enum::MultiplyUInt32, pExprLHS, pExprRHS ) );

        // Multiply the high half of the vector
        Expr *pHighLHS = _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackHighInt64, pExprLHS, CreateZeroVector(eElementType) );
        Expr *pHighRHS = _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackHighInt64, pExprRHS, CreateZeroVector(eElementType) );
        vecHalfVectors.push_back( _CreateFunctionCall( IntrinsicsSSE2Enum::MultiplyUInt32, pHighLHS, pHighRHS ) );

        // Pack vectors back into the element type
        return ConvertVectorDown( VectorElementTypes::UInt64, eElementType, vecHalfVectors );
      }
    default:    return _SeparatedArithmeticOpInteger(eElementType, BO_Mul, pExprLHS, pExprRHS);
    }
  case ArithmeticOperatorType::Modulo:      return _SeparatedArithmeticOpInteger( eElementType, BO_Rem, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::ShiftLeft:   return _SeparatedArithmeticOpInteger( eElementType, BO_Shl, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::ShiftRight:  return _SeparatedArithmeticOpInteger( eElementType, BO_Shr, pExprLHS, pExprRHS );
  case ArithmeticOperatorType::Subtract:
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _CreateFunctionCall( IntrinsicsSSE2Enum::SubtractInt8,  pExprLHS, pExprRHS );
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::SubtractInt16, pExprLHS, pExprRHS );
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::SubtractInt32, pExprLHS, pExprRHS );
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE2Enum::SubtractInt64, pExprLHS, pExprRHS );
    default:                                                          throw InternalErrorException("Unsupported vector element type detected!");
    }
  default:                                  throw InternalErrorException("Unsupported arithmetic operation detected!");
  }
}

Expr* InstructionSetSSE2::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  if (bMaskConversion)
  {
    switch (eSourceType)
    {
    case VectorElementTypes::Double:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:                                  return crvecVectorRefs.front();   // Same type => nothing to do
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE2Enum::CastDoubleToInteger, crvecVectorRefs.front() );
      case VectorElementTypes::Float:
        {
          ClangASTHelper::ExpressionVectorType vecCastedVectors;

          vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt64, crvecVectorRefs[0]) );
          vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt64, crvecVectorRefs[1]) );

          Expr *pPackedMask = ConvertMaskDown(VectorElementTypes::UInt64, VectorElementTypes::UInt32, vecCastedVectors);

          return ConvertMaskSameSize(VectorElementTypes::UInt32, eTargetType, pPackedMask);
        }
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
        {
          ClangASTHelper::ExpressionVectorType vecCastedVectors;

          for each (auto itVec in crvecVectorRefs)
          {
            vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt64, itVec) );
          }

          return ConvertMaskDown( VectorElementTypes::UInt64, eTargetType, vecCastedVectors );
        }
      }

      break;

    case VectorElementTypes::Float:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:
      {
        Expr *pConvertedMask  = ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt32, crvecVectorRefs.front());
        pConvertedMask        = ConvertMaskUp(VectorElementTypes::UInt32, VectorElementTypes::UInt64, pConvertedMask, uiGroupIndex);
        return ConvertMaskSameSize(VectorElementTypes::UInt64, eTargetType, pConvertedMask);
      }
      case VectorElementTypes::Float:                                   return crvecVectorRefs.front();   // Same type => nothing to do
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::CastFloatToInteger, crvecVectorRefs.front() );
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          ClangASTHelper::ExpressionVectorType vecCastedVectors;

          for each (auto itVec in crvecVectorRefs)
          {
            vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt32, itVec) );
          }

          return _ConvertVector( VectorElementTypes::UInt32, eTargetType, vecCastedVectors, uiGroupIndex, bMaskConversion );
        }
      }

      break;

    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

      switch (eTargetType)
      {
      case VectorElementTypes::Double: case VectorElementTypes::Float:
        {
          const size_t              cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType);
          const size_t              cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType);

          if (cszSourceSize == cszTargetSize)
          {
            // There is no binary difference between integer masks and floating point masks of same size => Just cast here
            return _CreateFunctionCall( (eTargetType == VectorElementTypes::Double) ? IntrinsicsSSE2Enum::CastIntegerToDouble : IntrinsicsSSE2Enum::CastIntegerToFloat, crvecVectorRefs.front() );
          }
          else
          {
            // Convert the mask(s) into an unsigned integer type with the same size as the target type, and then do the final conversion
            const VectorElementTypes  ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(cszTargetSize, false).GetType();

            Expr *pConvertedMask = _ConvertVector( eSourceType, ceIntermediateType, crvecVectorRefs, uiGroupIndex, bMaskConversion );
            return ConvertMaskSameSize( ceIntermediateType, eTargetType, pConvertedMask );
          }
        }
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          const size_t cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize( eSourceType );
          const size_t cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize( eTargetType );

          if (cszSourceSize == cszTargetSize)
          {
            // There is no difference between signed and unsigned masks => Nothing to do
            return crvecVectorRefs.front();
          }
          else if (cszSourceSize > cszTargetSize)
          {
            // The source type is larger => Pack the masks into a smaller type
            if ((crvecVectorRefs.size() & 1) != 0)
            {
              throw InternalErrorException("Expected a power of 2 as argument count for a downward conversion!");
            }

            ClangASTHelper::ExpressionVectorType vecPackedMasks;

            // Pack each adjacent mask pairs into a mask with a decreased intermediate type
            for (size_t szOutIdx = static_cast<size_t>(0); szOutIdx < crvecVectorRefs.size(); szOutIdx += static_cast<size_t>(2))
            {
              vecPackedMasks.push_back( _CreateFunctionCall( IntrinsicsSSE2Enum::PackInt16ToInt8, crvecVectorRefs[ szOutIdx ], crvecVectorRefs[ szOutIdx + 1 ] ) );
            }

            // Run the conversion from the decreased intermediate type into the target type
            const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType( cszSourceSize >> 1, false ).GetType();

            return _ConvertVector( ceIntermediateType, eTargetType, vecPackedMasks, uiGroupIndex, bMaskConversion );
          }
          else
          {
            // The source type is smaller => Select a group out of mask and duplicate it into a larger type
            if (cszSourceSize == 4)
            {
              // Source type is a 32-bit integer => Shuffling is most efficient
              const int32_t ciShuffleConstant = (uiGroupIndex == 0) ? 0x50 : 0xFA;

              // Target type must be a 64-bit integer => This is the end of the conversion cascade
              return _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, crvecVectorRefs.front(), _GetASTHelper().CreateIntegerLiteral(ciShuffleConstant) );
            }
            else
            {
              // Source type must be smaller than a 32-bit integer => Duplicate the mask group by un-packing the mask
              const uint32_t cuiSwapIndex = static_cast< uint32_t >( cszTargetSize / cszSourceSize ) >> 1;

              // Select the correct un-packing function by the group index
              IntrinsicsSSE2Enum eUnpackID = IntrinsicsSSE2Enum::UnpackLowInt8;
              if (uiGroupIndex >= cuiSwapIndex)
              {
                eUnpackID     = IntrinsicsSSE2Enum::UnpackHighInt8;
                uiGroupIndex -= cuiSwapIndex;   // Adjust the group index for the next step in the conversion cascade
              }

              // Un-pack the mask
              ClangASTHelper::ExpressionVectorType vecConvertedMask;
              vecConvertedMask.push_back( _CreateFunctionCall( eUnpackID, crvecVectorRefs.front(), crvecVectorRefs.front() ) );

              // Run the conversion from the increased intermediate type into the target type
              const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType( cszSourceSize << 1, false ).GetType();

              return _ConvertVector( ceIntermediateType, eTargetType, vecConvertedMask, uiGroupIndex, bMaskConversion );
            }
          }
        }
      }
    }
  }
  else
  {
    switch (eSourceType)
    {
    case VectorElementTypes::Double:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:  return crvecVectorRefs.front();   // Same type => nothing to do
      case VectorElementTypes::Float:
        {
          Expr *pValuesLow  = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertDoubleFloat, crvecVectorRefs[0] );
          Expr *pValuesHigh = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertDoubleFloat, crvecVectorRefs[1] );

          return _MergeVectors( eTargetType, pValuesLow, pValuesHigh, true );
        }
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
        {
          Expr *pValuesLow  = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertDoubleInt32, crvecVectorRefs[0] );
          Expr *pValuesHigh = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertDoubleInt32, crvecVectorRefs[1] );

          return _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackLowInt64, pValuesLow, pValuesHigh );
        }
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          // Packed conversion not possible => Extract and convert both values separately
          Expr *pValueLow   = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertSingleDoubleInt64, crvecVectorRefs.front() );

          Expr *pValueHigh  = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleDouble, crvecVectorRefs.front(), CreateZeroVector(eSourceType), _GetASTHelper().CreateIntegerLiteral( 1 ) );
          pValueHigh        = _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertSingleDoubleInt64, pValueHigh );

          return _CreateFunctionCall( IntrinsicsSSE2Enum::SetInt64, pValueHigh, pValueLow );
        }
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          ClangASTHelper::ExpressionVectorType vecConvertedVectors;

          if ((crvecVectorRefs.size() & 1) != 0)
          {
            throw InternalErrorException("Expected a power of 2 as argument count for a downward conversion!");
          }

          for (size_t szOutIdx = static_cast<size_t>(0); szOutIdx < crvecVectorRefs.size(); szOutIdx += static_cast<size_t>(2))
          {
            ClangASTHelper::ExpressionVectorType vecConvArgs;

            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx ] );
            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx + 1] );

            vecConvertedVectors.push_back( ConvertVectorDown( eSourceType, VectorElementTypes::Int32, vecConvArgs ) );
          }

          return ConvertVectorDown( VectorElementTypes::Int32, eTargetType, vecConvertedVectors );
        }
      }

      break;

    case VectorElementTypes::Float:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:
        {
          Expr *pSourceRef = crvecVectorRefs.front();
          if (uiGroupIndex != 0)
          {
            pSourceRef = _MergeVectors( eSourceType, CreateZeroVector(eSourceType), pSourceRef, false );
          }

          return _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertFloatDouble, pSourceRef );
        }
      case VectorElementTypes::Float:                                   return crvecVectorRefs.front();   // Same type => nothing to do
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertFloatInt32, crvecVectorRefs.front() );
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          Expr *pConvertedVector = ConvertVectorUp( eSourceType, VectorElementTypes::Double, crvecVectorRefs.front(), uiGroupIndex );

          return ConvertVectorSameSize( VectorElementTypes::Double, eTargetType, pConvertedVector );
        }
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          ClangASTHelper::ExpressionVectorType vecConvertedVectors;

          for each (auto itVectorRef in crvecVectorRefs)
          {
            vecConvertedVectors.push_back( ConvertVectorSameSize(eSourceType, VectorElementTypes::Int32, itVectorRef) );
          }

          return ConvertVectorDown(VectorElementTypes::Int32, eTargetType, vecConvertedVectors);
        }
      }

      break;

    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:

      switch (eTargetType)
      {
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return crvecVectorRefs.front();   // No difference between signed and unsigned vectors => nothing to do
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          // Handle signed and unsigned up-conversion
          Expr *pInterleaveVector = CreateZeroVector(eSourceType);
          if (AST::BaseClasses::TypeInfo::IsSigned(eSourceType))
          {
            // Insert sign based high-word => Create the corresponding mask by a comparison
            pInterleaveVector = RelationalOperator( eSourceType, RelationalOperatorType::Less, crvecVectorRefs.front(), pInterleaveVector );
          }

          return _CreateFunctionCall( (uiGroupIndex == 0) ? IntrinsicsSSE2Enum::UnpackLowInt8 : IntrinsicsSSE2Enum::UnpackHighInt8, crvecVectorRefs.front(), pInterleaveVector );
        }
      case VectorElementTypes::Int32:  case VectorElementTypes::UInt32:
        {
          // Convert into a signed / unsigned 16-bit integer intermediate type and then do the final conversion
          const VectorElementTypes  ceIntermediateType  = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2, AST::BaseClasses::TypeInfo::IsSigned(eSourceType)).GetType();
          const uint32_t            cuiSwapIndex        = static_cast< uint32_t >( AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) ) >> 1;

          Expr *pConvertedVector = ConvertVectorUp( eSourceType, ceIntermediateType, crvecVectorRefs.front(), uiGroupIndex / cuiSwapIndex );

          if (uiGroupIndex >= cuiSwapIndex)
          {
            uiGroupIndex -= cuiSwapIndex;
          }

          return ConvertVectorUp( ceIntermediateType, eTargetType, pConvertedVector, uiGroupIndex );
        }
      case VectorElementTypes::Int64:  case VectorElementTypes::UInt64:
      case VectorElementTypes::Double: case VectorElementTypes::Float:
        {
          // Convert into a 32-bit integer intermediate type and then do the final conversion
          VectorElementTypes  eIntermediateType;
          if ( (eTargetType == VectorElementTypes::Int64) || (eTargetType == VectorElementTypes::UInt64) )
          {
            // Unsigned integer conversions are faster, but only possible if the source type is unsigned => select proper intermediate type
            eIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(4, AST::BaseClasses::TypeInfo::IsSigned(eSourceType)).GetType();
          }
          else
          {
            // Signed integer to floating point conversions are much faster than their unsigned counterparts
            // => intermediate type is "Int32" (it can hold the whole "UInt16" value range)
            eIntermediateType = VectorElementTypes::Int32;
          }

          const uint32_t cuiSwapIndex   = static_cast< uint32_t >( AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) ) >> 2;

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;
          vecConvertedVectors.push_back( ConvertVectorUp( eSourceType, eIntermediateType, crvecVectorRefs.front(), uiGroupIndex / cuiSwapIndex ) );

          if (uiGroupIndex >= cuiSwapIndex)
          {
            uiGroupIndex %= cuiSwapIndex;
          }

          return _ConvertVector( eIntermediateType, eTargetType, vecConvertedVectors, uiGroupIndex, bMaskConversion );
        }
      }

      break;

    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:

      switch (eTargetType)
      {
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
        {
          // We need to remove the high-byte from all vector elements to avoid the saturation
          const int32_t cuiMaskConstant = 0xFF;

          Expr *pValuesLow  = ArithmeticOperator( eSourceType, ArithmeticOperatorType::BitwiseAnd, crvecVectorRefs[0], BroadCast(eSourceType, _GetASTHelper().CreateIntegerLiteral(cuiMaskConstant)) );
          Expr *pValuesHigh = ArithmeticOperator( eSourceType, ArithmeticOperatorType::BitwiseAnd, crvecVectorRefs[1], BroadCast(eSourceType, _GetASTHelper().CreateIntegerLiteral(cuiMaskConstant)) );

          return _CreateFunctionCall( IntrinsicsSSE2Enum::PackInt16ToUInt8, pValuesLow, pValuesHigh );
        }
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return crvecVectorRefs.front();   // No difference between signed and unsigned vectors => nothing to do
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
        {
          // Handle signed and unsigned up-conversion
          Expr *pInterleaveVector = CreateZeroVector(eSourceType);
          if (AST::BaseClasses::TypeInfo::IsSigned(eSourceType))
          {
            // Insert sign based high-word => Create the corresponding mask by a comparison
            pInterleaveVector = RelationalOperator( eSourceType, RelationalOperatorType::Less, crvecVectorRefs.front(), pInterleaveVector );
          }

          return _CreateFunctionCall( (uiGroupIndex == 0) ? IntrinsicsSSE2Enum::UnpackLowInt16 : IntrinsicsSSE2Enum::UnpackHighInt16, crvecVectorRefs.front(), pInterleaveVector );
        }
      case VectorElementTypes::Int64:  case VectorElementTypes::UInt64:
      case VectorElementTypes::Double: case VectorElementTypes::Float:
        {
          // Convert into a 32-bit integer intermediate type and then do the final conversion
          VectorElementTypes  eIntermediateType;
          if ( (eTargetType == VectorElementTypes::Int64) || (eTargetType == VectorElementTypes::UInt64) )
          {
            // Unsigned integer conversions are faster, but only possible if the source type is unsigned => select proper intermediate type
            eIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(4, AST::BaseClasses::TypeInfo::IsSigned(eSourceType)).GetType();
          }
          else
          {
            // Signed integer to floating point conversions are much faster than their unsigned counterparts
            // => intermediate type is "Int32" (it can hold the whole "UInt16" value range)
            eIntermediateType = VectorElementTypes::Int32;
          }

          const uint32_t cuiSwapIndex   = static_cast< uint32_t >( AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) ) >> 1;

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;
          vecConvertedVectors.push_back( ConvertVectorUp( eSourceType, eIntermediateType, crvecVectorRefs.front(), uiGroupIndex / cuiSwapIndex ) );

          if (uiGroupIndex >= cuiSwapIndex)
          {
            uiGroupIndex -= cuiSwapIndex;
          }

          return _ConvertVector( eIntermediateType, eTargetType, vecConvertedVectors, uiGroupIndex, bMaskConversion );
        }
      }

      break;

    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:
        {
          if (AST::BaseClasses::TypeInfo::IsSigned(eSourceType))
          {
            Expr *pSelectedGroup = crvecVectorRefs.front();
            if (uiGroupIndex != 0)
            {
              pSelectedGroup = _ShiftIntegerVectorBytes( pSelectedGroup, 8, false );
            }

            return _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertInt32Double, pSelectedGroup );
          }
          else
          {
            // No SSE support for this conversion => Extract elements, convert them one by one and recreate the vector
            ClangASTHelper::ExpressionVectorType vecConvertedElements;

            for (uint32_t uiIdx = 0; uiIdx < GetVectorElementCount(eTargetType); ++uiIdx)
            {
              Expr *pElement = ExtractElement( eSourceType, crvecVectorRefs.front(), uiIdx + (uiGroupIndex << 1) );

              vecConvertedElements.push_back( _CreateValueCast( pElement, _GetClangType(eTargetType), CK_IntegralToFloating ) );
            }

            return CreateVector( eTargetType, vecConvertedElements, false );
          }
        }
      case VectorElementTypes::Float:
        {
          if (AST::BaseClasses::TypeInfo::IsSigned(eSourceType))
          {
            return _CreateFunctionCall( IntrinsicsSSE2Enum::ConvertInt32Float, crvecVectorRefs.front() );
          }
          else
          {
            // No SSE support for this conversion => Extract elements, convert them one by one and recreate the vector
            ClangASTHelper::ExpressionVectorType vecConvertedElements;

            for (uint32_t uiIdx = 0; uiIdx < GetVectorElementCount(eTargetType); ++uiIdx)
            {
              vecConvertedElements.push_back( _CreateValueCast( ExtractElement(eSourceType, crvecVectorRefs.front(), uiIdx), _GetClangType(eTargetType), CK_IntegralToFloating ) );
            }

            return CreateVector( eTargetType, vecConvertedElements, false );
          }
        }
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return crvecVectorRefs.front();   // No difference between signed and unsigned vectors => nothing to do
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
        {
          // Convert into a signed / unsigned 16-bit integer intermediate type and then do the final conversion
          const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2, AST::BaseClasses::TypeInfo::IsSigned(eSourceType)).GetType();

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;

          if ((crvecVectorRefs.size() & 1) != 0)
          {
            throw InternalErrorException("Expected a power of 2 as argument count for a downward conversion!");
          }

          for (size_t szOutIdx = static_cast<size_t>(0); szOutIdx < crvecVectorRefs.size(); szOutIdx += static_cast<size_t>(2))
          {
            ClangASTHelper::ExpressionVectorType vecConvArgs;

            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx ] );
            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx + 1 ] );

            vecConvertedVectors.push_back( ConvertVectorDown(eSourceType, ceIntermediateType, vecConvArgs) );
          }

          return ConvertVectorDown( ceIntermediateType, eTargetType, vecConvertedVectors );
        }
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          // Shuffle each 4 packed low-words into the low 8 bytes
          ClangASTHelper::ExpressionVectorType vecShuffledVectors;

          for (size_t szIdx = static_cast<size_t>(0); szIdx < crvecVectorRefs.size(); ++szIdx)
          {
            const int32_t ciShuffleConstant = 0xD8;

            Expr *pShuffledVec  = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt16Low,  crvecVectorRefs[ szIdx ], _GetASTHelper().CreateIntegerLiteral( ciShuffleConstant ) );
            pShuffledVec        = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt16High, pShuffledVec,             _GetASTHelper().CreateIntegerLiteral( ciShuffleConstant ) );

            vecShuffledVectors.push_back( _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, pShuffledVec, _GetASTHelper().CreateIntegerLiteral( ciShuffleConstant ) ) );
          }

          // Merge the shuffled vectors
          return _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackLowInt64, vecShuffledVectors[0], vecShuffledVectors[1] );
        }
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          // Handle signed and unsigned up-conversion
          Expr *pInterleaveVector = CreateZeroVector(eSourceType);
          if (AST::BaseClasses::TypeInfo::IsSigned(eSourceType))
          {
            // Insert sign based high-word => Create the corresponding mask by a comparison
            pInterleaveVector = RelationalOperator( eSourceType, RelationalOperatorType::Less, crvecVectorRefs.front(), pInterleaveVector );
          }

          return _CreateFunctionCall( (uiGroupIndex == 0) ? IntrinsicsSSE2Enum::UnpackLowInt32 : IntrinsicsSSE2Enum::UnpackHighInt32, crvecVectorRefs.front(), pInterleaveVector );
        }
      }

      break;

    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

      switch (eTargetType)
      {
      case VectorElementTypes::Double:
        {
          // No SSE support for this conversion => Extract elements, convert them one by one and recreate the vector
          ClangASTHelper::ExpressionVectorType vecConvertedElements;

          // Vector creation needs elements in reversed order
          for (uint32_t uiIdx = 0; uiIdx < GetVectorElementCount(eTargetType); ++uiIdx)
          {
            vecConvertedElements.push_back( _CreateValueCast( ExtractElement(eSourceType, crvecVectorRefs.front(), uiIdx), _GetClangType(eTargetType), CK_IntegralToFloating ) );
          }

          return CreateVector( eTargetType, vecConvertedElements, false );
        }
      case VectorElementTypes::Float:
        {
          // Convert into the intermediate type "double" and then do the final conversion
          ClangASTHelper::ExpressionVectorType vecConvertedVectors;

          for (size_t szOutIdx = static_cast<size_t>(0); szOutIdx < crvecVectorRefs.size(); ++szOutIdx)
          {
            vecConvertedVectors.push_back( ConvertVectorSameSize( eSourceType, VectorElementTypes::Double, crvecVectorRefs[szOutIdx] ) );
          }

          return ConvertVectorDown( VectorElementTypes::Double, eTargetType, vecConvertedVectors );
        }
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return crvecVectorRefs.front();   // No difference between signed and unsigned vectors => nothing to do
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
        {
          // Shuffle each packed 2 low-DWords into the lower 8 byte
          const int32_t ciShuffleConstant = 0xD8;

          Expr *pValuesLow  = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, crvecVectorRefs[0], _GetASTHelper().CreateIntegerLiteral( ciShuffleConstant ) );
          Expr *pValuesHigh = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, crvecVectorRefs[1], _GetASTHelper().CreateIntegerLiteral( ciShuffleConstant ) );

          // Merge the shuffled vectors
          return _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackLowInt64, pValuesLow, pValuesHigh );
        }
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          // Convert into a signed / unsigned 32-bit integer intermediate type and then do the final conversion
          const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(4, AST::BaseClasses::TypeInfo::IsSigned(eSourceType)).GetType();

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;

          if ((crvecVectorRefs.size() & 1) != 0)
          {
            throw InternalErrorException("Expected a power of 2 as argument count for a downward conversion!");
          }

          for (size_t szOutIdx = static_cast<size_t>(0); szOutIdx < crvecVectorRefs.size(); szOutIdx += static_cast<size_t>(2))
          {
            ClangASTHelper::ExpressionVectorType vecConvArgs;

            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx ] );
            vecConvArgs.push_back( crvecVectorRefs[ szOutIdx + 1 ] );

            vecConvertedVectors.push_back( ConvertVectorDown(eSourceType, ceIntermediateType, vecConvArgs) );
          }

          return ConvertVectorDown( ceIntermediateType, eTargetType, vecConvertedVectors );
        }
      }

      break;
    }
  }

  // If the function has not returned earlier, let the base handle the conversion
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion); 
}

Expr* InstructionSetSSE2::_CompareInt64(VectorElementTypes eElementType, Expr *pExprLHS, Expr *pExprRHS, BinaryOperatorKind eOpKind)
{
  QualType qtBool = _GetClangType(VectorElementTypes::Bool);

  ClangASTHelper::ExpressionVectorType vecArgs;

  for (uint32_t uiIndex = 0; uiIndex < GetVectorElementCount(eElementType); ++uiIndex)
  {
    // Extract the elements and compare
    Expr  *pElement = _GetASTHelper().CreateBinaryOperator( ExtractElement(eElementType, pExprLHS, uiIndex), ExtractElement(eElementType, pExprRHS, uiIndex), eOpKind, qtBool );


    // Conditional set the correct mask value for each element
    pElement = _GetASTHelper().CreateConditionalOperator( _GetASTHelper().CreateParenthesisExpression(pElement), _GetASTHelper().CreateIntegerLiteral(-1L),
                                                          _GetASTHelper().CreateIntegerLiteral(0L), pElement->getType() );

    vecArgs.push_back( pElement );
  }

  return CreateVector( eElementType, vecArgs, false );
}

Expr* InstructionSetSSE2::_CreateFullBitMask(VectorElementTypes eElementType)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return RelationalOperator( eElementType, RelationalOperatorType::Equal, CreateZeroVector(eElementType), CreateZeroVector(eElementType) );
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return CreateOnesVector( VectorElementTypes::Int32, true );
  default:                                                          return BaseType::_CreateFullBitMask( eElementType );
  }
}

void InstructionSetSSE2::_InitIntrinsicsMap()
{
  // Addition functions
  _InitIntrinsic( IntrinsicsSSE2Enum::AddDouble, "add_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::AddInt8,   "add_epi8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::AddInt16,  "add_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::AddInt32,  "add_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::AddInt64,  "add_epi64" );

  // Bitwise "and" and "and not"
  _InitIntrinsic( IntrinsicsSSE2Enum::AndDouble,     "and_pd"       );
  _InitIntrinsic( IntrinsicsSSE2Enum::AndInteger,    "and_si128"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::AndNotDouble,  "andnot_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::AndNotInteger, "andnot_si128" );

  // Broadcast functions
  _InitIntrinsic( IntrinsicsSSE2Enum::BroadCastDouble, "set1_pd"     );
  _InitIntrinsic( IntrinsicsSSE2Enum::BroadCastInt8,   "set1_epi8"   );
  _InitIntrinsic( IntrinsicsSSE2Enum::BroadCastInt16,  "set1_epi16"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::BroadCastInt32,  "set1_epi32"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::BroadCastInt64,  "set1_epi64x" );

  // Vector cast functions (change bit-representation, no conversion)
  _InitIntrinsic( IntrinsicsSSE2Enum::CastDoubleToFloat,   "castpd_ps"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::CastDoubleToInteger, "castpd_si128" );
  _InitIntrinsic( IntrinsicsSSE2Enum::CastFloatToDouble,   "castps_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::CastFloatToInteger,  "castps_si128" );
  _InitIntrinsic( IntrinsicsSSE2Enum::CastIntegerToDouble, "castsi128_pd" );
  _InitIntrinsic( IntrinsicsSSE2Enum::CastIntegerToFloat,  "castsi128_ps" );

  // Comparison methods
  {
    // Compare equal
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareEqualDouble, "cmpeq_pd"    );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareEqualInt8,   "cmpeq_epi8"  );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareEqualInt16,  "cmpeq_epi16" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareEqualInt32,  "cmpeq_epi32" );

    // Compare "greater equal"
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareGreaterEqualDouble, "cmpge_pd" );

    // Compare "greater than"
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareGreaterThanDouble, "cmpgt_pd"    );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareGreaterThanInt8,   "cmpgt_epi8"  );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareGreaterThanInt16,  "cmpgt_epi16" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareGreaterThanInt32,  "cmpgt_epi32" );

    // Compare "less equal"
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareLessEqualDouble, "cmple_pd" );

    // Compare "less than"
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareLessThanDouble, "cmplt_pd"    );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareLessThanInt8,   "cmplt_epi8"  );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareLessThanInt16,  "cmplt_epi16" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareLessThanInt32,  "cmplt_epi32" );

    // Negated comparison methods
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareNotEqualDouble,        "cmpneq_pd" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareNotGreaterEqualDouble, "cmpnge_pd" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareNotGreaterThanDouble,  "cmpngt_pd" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareNotLessEqualDouble,    "cmpnle_pd" );
    _InitIntrinsic( IntrinsicsSSE2Enum::CompareNotLessThanDouble,     "cmpnlt_pd" );
  }

  // Convert functions
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertDoubleFloat,       "cvtpd_ps"     );
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertDoubleInt32,       "cvttpd_epi32" );   // Use truncation
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertFloatDouble,       "cvtps_pd"     );
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertFloatInt32,        "cvttps_epi32" );   // Use truncation
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertInt32Double,       "cvtepi32_pd"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertInt32Float,        "cvtepi32_ps"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::ConvertSingleDoubleInt64, "cvttsd_si64"  );   // Use truncation

  // Rounding functions
  _InitIntrinsic( IntrinsicsSSE2Enum::CeilDouble,  "ceil_pd"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::FloorDouble, "floor_pd" );

  // Division functions
  _InitIntrinsic( IntrinsicsSSE2Enum::DivideDouble, "div_pd" );

  // Extract functions
  _InitIntrinsic( IntrinsicsSSE2Enum::ExtractInt16,        "extract_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ExtractLowestDouble, "cvtsd_f64"     );
  _InitIntrinsic( IntrinsicsSSE2Enum::ExtractLowestInt32,  "cvtsi128_si32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ExtractLowestInt64,  "cvtsi128_si64" );

  // Insert functions
  _InitIntrinsic( IntrinsicsSSE2Enum::InsertInt16,        "insert_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::InsertLowestDouble, "move_sd"      );

  // Load functions
  _InitIntrinsic( IntrinsicsSSE2Enum::LoadDouble,  "loadu_pd"   );
  _InitIntrinsic( IntrinsicsSSE2Enum::LoadInteger, "loadu_si128" );

  // Maximum / Minimum functions
  _InitIntrinsic( IntrinsicsSSE2Enum::MaxDouble, "max_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::MaxUInt8,  "max_epu8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::MaxInt16,  "max_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::MinDouble, "min_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::MinUInt8,  "min_epu8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::MinInt16,  "min_epi16" );

  // Mask conversion functions
  _InitIntrinsic( IntrinsicsSSE2Enum::MoveMaskDouble, "movemask_pd"   );
  _InitIntrinsic( IntrinsicsSSE2Enum::MoveMaskInt8,   "movemask_epi8" );

  // Multiplication functions
  _InitIntrinsic( IntrinsicsSSE2Enum::MultiplyDouble, "mul_pd"      );
  _InitIntrinsic( IntrinsicsSSE2Enum::MultiplyInt16,  "mullo_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::MultiplyUInt32, "mul_epu32"   );

  // Bitwise "or" functions
  _InitIntrinsic( IntrinsicsSSE2Enum::OrDouble,  "or_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::OrInteger, "or_si128" );

  // Integer packing functions
  _InitIntrinsic( IntrinsicsSSE2Enum::PackInt16ToInt8,  "packs_epi16"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::PackInt16ToUInt8, "packus_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::PackInt32ToInt16, "packs_epi32"  );

  // Set methods
  _InitIntrinsic( IntrinsicsSSE2Enum::SetDouble, "set_pd"     );
  _InitIntrinsic( IntrinsicsSSE2Enum::SetInt8,   "set_epi8"   );
  _InitIntrinsic( IntrinsicsSSE2Enum::SetInt16,  "set_epi16"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::SetInt32,  "set_epi32"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::SetInt64,  "set_epi64x" );

  // Zero vector creation functions
  _InitIntrinsic( IntrinsicsSSE2Enum::SetZeroDouble,  "setzero_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::SetZeroInteger, "setzero_si128" );

  // Shift functions
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftLeftInt16,        "slli_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftLeftInt32,        "slli_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftLeftInt64,        "slli_epi64" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftLeftVectorBytes,  "slli_si128" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightArithInt16,  "srai_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightArithInt32,  "srai_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightLogInt16,    "srli_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightLogInt32,    "srli_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightLogInt64,    "srli_epi64" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShiftRightVectorBytes, "srli_si128" );

  // Shuffle functions
  _InitIntrinsic( IntrinsicsSSE2Enum::ShuffleDouble,    "shuffle_pd"      );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShuffleInt16High, "shufflehi_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShuffleInt16Low,  "shufflelo_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::ShuffleInt32,     "shuffle_epi32"   );

  // Square root functions
  _InitIntrinsic( IntrinsicsSSE2Enum::SqrtDouble, "sqrt_pd" );

  // Store functions
  _InitIntrinsic( IntrinsicsSSE2Enum::StoreDouble,             "storeu_pd"       );
  _InitIntrinsic( IntrinsicsSSE2Enum::StoreInteger,            "storeu_si128"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::StoreConditionalInteger, "maskmoveu_si128" );

  // Subtraction functions
  _InitIntrinsic( IntrinsicsSSE2Enum::SubtractDouble, "sub_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::SubtractInt8,   "sub_epi8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::SubtractInt16,  "sub_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::SubtractInt32,  "sub_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::SubtractInt64,  "sub_epi64" );

  // Un-packing function
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackHighDouble, "unpackhi_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackHighInt8,   "unpackhi_epi8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackHighInt16,  "unpackhi_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackHighInt32,  "unpackhi_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackHighInt64,  "unpackhi_epi64" );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackLowDouble,  "unpacklo_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackLowInt8,    "unpacklo_epi8"  );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackLowInt16,   "unpacklo_epi16" );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackLowInt32,   "unpacklo_epi32" );
  _InitIntrinsic( IntrinsicsSSE2Enum::UnpackLowInt64,   "unpacklo_epi64" );

  // Bitwise "xor" functions
  _InitIntrinsic( IntrinsicsSSE2Enum::XorDouble,  "xor_pd"    );
  _InitIntrinsic( IntrinsicsSSE2Enum::XorInteger, "xor_si128" );
}

Expr* InstructionSetSSE2::_RelationalOpInteger(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  if (eOpType == RelationalOperatorType::LogicalAnd)
  {
    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseAnd, pExprLHS, pExprRHS );
  }
  else if (eOpType == RelationalOperatorType::LogicalOr)
  {
    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseOr, pExprLHS, pExprRHS );
  }
  else if (eOpType == RelationalOperatorType::NotEqual)
  {
    return UnaryOperator( eElementType, UnaryOperatorType::LogicalNot, RelationalOperator(eElementType, RelationalOperatorType::Equal, pExprLHS, pExprRHS) );
  }
  else if (eOpType == RelationalOperatorType::Equal)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareEqualInt8,  pExprLHS, pExprRHS );
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareEqualInt16, pExprLHS, pExprRHS );
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareEqualInt32, pExprLHS, pExprRHS );
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CompareInt64( eElementType, pExprLHS, pExprRHS, BO_EQ );
    default:                                                          return BaseType::RelationalOperator( eElementType, eOpType, pExprLHS, pExprRHS );
    }
  }
  else if (! AST::BaseClasses::TypeInfo::IsSigned(eElementType))
  {
    // Convert vector elements such that an unsigned comparison is possible
    Expr *pSignMask = nullptr;

    switch (eElementType)
    {
    case VectorElementTypes::UInt8:    pSignMask = _GetASTHelper().CreateIntegerLiteral( 0x80                );  break;
    case VectorElementTypes::UInt16:   pSignMask = _GetASTHelper().CreateIntegerLiteral( 0x8000              );  break;
    case VectorElementTypes::UInt32:   pSignMask = _GetASTHelper().CreateIntegerLiteral( 0x80000000          );  break;
    case VectorElementTypes::UInt64:   pSignMask = _GetASTHelper().CreateIntegerLiteral( 0x8000000000000000L );  break;
    default:                          throw InternalErrorException("Unexpected vector element type detected!");
    }

    Expr *pConvLHS = ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseXOr, pExprLHS, BroadCast(eElementType, pSignMask) );
    Expr *pConvRHS = ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseXOr, pExprRHS, BroadCast(eElementType, pSignMask) );

    return RelationalOperator( AST::BaseClasses::TypeInfo::CreateSizedIntegerType(AST::BaseClasses::TypeInfo::GetTypeSize(eElementType), true).GetType(), eOpType, pConvLHS, pConvRHS );
  }
  else
  {
    switch (eOpType)
    {
    case RelationalOperatorType::GreaterEqual:
      switch (eElementType)
      {
      case VectorElementTypes::Int64:   return _CompareInt64( eElementType, pExprLHS, pExprRHS, BO_GE );
      default:                          return UnaryOperator( eElementType, UnaryOperatorType::LogicalNot, RelationalOperator(eElementType, RelationalOperatorType::Less, pExprLHS, pExprRHS) );
      }
    case RelationalOperatorType::LessEqual:
      switch (eElementType)
      {
      case VectorElementTypes::Int64:   return _CompareInt64( eElementType, pExprLHS, pExprRHS, BO_LE );
      default:                          return UnaryOperator( eElementType, UnaryOperatorType::LogicalNot, RelationalOperator(eElementType, RelationalOperatorType::Greater, pExprLHS, pExprRHS) );
      }
    case RelationalOperatorType::Greater:
      switch (eElementType)
      {
      case VectorElementTypes::Int8:    return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareGreaterThanInt8,  pExprLHS, pExprRHS );
      case VectorElementTypes::Int16:   return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareGreaterThanInt16, pExprLHS, pExprRHS );
      case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareGreaterThanInt32, pExprLHS, pExprRHS );
      case VectorElementTypes::Int64:   return _CompareInt64( eElementType, pExprLHS, pExprRHS, BO_GT );
      default:                          throw InternalErrorException("Unexpected vector element type detected!");
      }
    case RelationalOperatorType::Less:
      switch (eElementType)
      {
      case VectorElementTypes::Int8:    return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareLessThanInt8,  pExprLHS, pExprRHS );
      case VectorElementTypes::Int16:   return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareLessThanInt16, pExprLHS, pExprRHS );
      case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareLessThanInt32, pExprLHS, pExprRHS );
      case VectorElementTypes::Int64:   return _CompareInt64( eElementType, pExprLHS, pExprRHS, BO_LT );
      default:                          throw InternalErrorException("Unexpected vector element type detected!");
      }
    default:  throw InternalErrorException("Unexpected relational operation detected!");
    }
  }
}

Expr* InstructionSetSSE2::_SeparatedArithmeticOpInteger(VectorElementTypes eElementType, BinaryOperatorKind eOpKind, Expr *pExprLHS, Expr *pExprRHS)
{
  ClangASTHelper::ExpressionVectorType vecSeparatedExprs;

  // Extract all elements one by one and do the compuation (keep in mind that SSE expects the reversed order of creation args)
  for (uint32_t uiIndex = 0; uiIndex < GetVectorElementCount(eElementType); ++uiIndex)
  {
    Expr *pElemLHS = ExtractElement( eElementType, pExprLHS, uiIndex );
    Expr *pElemRHS = ExtractElement( eElementType, pExprRHS, uiIndex );

    vecSeparatedExprs.push_back( _GetASTHelper().CreateBinaryOperator(pElemLHS, pElemRHS, eOpKind, pElemLHS->getType()) );
  }

  // Rereate the vector
  return CreateVector( eElementType, vecSeparatedExprs, false );
}

Expr* InstructionSetSSE2::_ShiftIntegerVectorBytes(Expr *pVectorRef, uint32_t uiByteCount, bool bShiftLeft)
{
  if (uiByteCount == 0)
  {
    return pVectorRef;  // Nothing to do
  }
  else if (uiByteCount >= 16)
  {
    throw InternalErrorException("Cannot shift a vector by 16 bytes or more!");
  }
  else
  {
    const IntrinsicsSSE2Enum eShiftID = bShiftLeft ? IntrinsicsSSE2Enum::ShiftLeftVectorBytes : IntrinsicsSSE2Enum::ShiftRightVectorBytes;

    return _CreateFunctionCall( eShiftID, pVectorRef, _GetASTHelper().CreateIntegerLiteral( static_cast<int32_t>(uiByteCount) ) );
  }
}

Expr* InstructionSetSSE2::_UnpackVectors(VectorElementTypes eElementType, Expr *pVectorRef1, Expr *pVectorRef2, bool bLowHalf)
{
  IntrinsicsSSE2Enum eIntrinID = IntrinsicsSSE2Enum::UnpackHighInt8;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eIntrinID = bLowHalf ? IntrinsicsSSE2Enum::UnpackLowDouble : IntrinsicsSSE2Enum::UnpackHighDouble;  break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = bLowHalf ? IntrinsicsSSE2Enum::UnpackLowInt8   : IntrinsicsSSE2Enum::UnpackHighInt8;    break;
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = bLowHalf ? IntrinsicsSSE2Enum::UnpackLowInt16  : IntrinsicsSSE2Enum::UnpackHighInt16;   break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = bLowHalf ? IntrinsicsSSE2Enum::UnpackLowInt32  : IntrinsicsSSE2Enum::UnpackHighInt32;   break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = bLowHalf ? IntrinsicsSSE2Enum::UnpackLowInt64  : IntrinsicsSSE2Enum::UnpackHighInt64;   break;
  default:                                                          return BaseType::_UnpackVectors( eElementType, pVectorRef1, pVectorRef2, bLowHalf );
  }

  return _CreateFunctionCall( eIntrinID, pVectorRef1, pVectorRef2 );
}

Expr* InstructionSetSSE2::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    switch (eOpType)
    {
    case ArithmeticOperatorType::Add:         return _CreateFunctionCall( IntrinsicsSSE2Enum::AddDouble,      pExprLHS, pExprRHS );
    case ArithmeticOperatorType::BitwiseAnd:  return _CreateFunctionCall( IntrinsicsSSE2Enum::AndDouble,      pExprLHS, pExprRHS );
    case ArithmeticOperatorType::BitwiseOr:   return _CreateFunctionCall( IntrinsicsSSE2Enum::OrDouble,       pExprLHS, pExprRHS );
    case ArithmeticOperatorType::BitwiseXOr:  return _CreateFunctionCall( IntrinsicsSSE2Enum::XorDouble,      pExprLHS, pExprRHS );
    case ArithmeticOperatorType::Divide:      return _CreateFunctionCall( IntrinsicsSSE2Enum::DivideDouble,   pExprLHS, pExprRHS );
    case ArithmeticOperatorType::Multiply:    return _CreateFunctionCall( IntrinsicsSSE2Enum::MultiplyDouble, pExprLHS, pExprRHS );
    case ArithmeticOperatorType::Subtract:    return _CreateFunctionCall( IntrinsicsSSE2Enum::SubtractDouble, pExprLHS, pExprRHS );
    case ArithmeticOperatorType::Modulo:      throw RuntimeErrorException("Modulo operation is undefined for \"double\" data types!");
    case ArithmeticOperatorType::ShiftLeft:
    case ArithmeticOperatorType::ShiftRight:  throw RuntimeErrorException("Shift operations are undefined for \"double\" data types!");
    default:                                  throw InternalErrorException("Unsupported arithmetic operation detected!");
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _ArithmeticOpInteger( eElementType, eOpType, pExprLHS, pExprRHS );
  default:                                                          return BaseType::ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS );
  }
}

Expr* InstructionSetSSE2::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    {
      Expr *pSelectTrue  = _CreateFunctionCall( IntrinsicsSSE2Enum::AndDouble,    pMaskRef, pVectorTrue  );
      Expr *pSelectFalse = _CreateFunctionCall( IntrinsicsSSE2Enum::AndNotDouble, pMaskRef, pVectorFalse );

      return _CreateFunctionCall( IntrinsicsSSE2Enum::OrDouble, pSelectTrue, pSelectFalse );
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      Expr *pSelectTrue  = _CreateFunctionCall( IntrinsicsSSE2Enum::AndInteger,    pMaskRef, pVectorTrue  );
      Expr *pSelectFalse = _CreateFunctionCall( IntrinsicsSSE2Enum::AndNotInteger, pMaskRef, pVectorFalse );

      return _CreateFunctionCall( IntrinsicsSSE2Enum::OrInteger, pSelectTrue, pSelectFalse );
    }
  default:  return BaseType::BlendVectors( eElementType, pMaskRef, pVectorTrue, pVectorFalse );
  }
}

Expr* InstructionSetSSE2::BroadCast(VectorElementTypes eElementType, Expr *pBroadCastValue)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return _CreateFunctionCall(IntrinsicsSSE2Enum::BroadCastDouble, pBroadCastValue);
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _CreateFunctionCall(IntrinsicsSSE2Enum::BroadCastInt8,   pBroadCastValue);
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall(IntrinsicsSSE2Enum::BroadCastInt16,  pBroadCastValue);
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall(IntrinsicsSSE2Enum::BroadCastInt32,  pBroadCastValue);
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall(IntrinsicsSSE2Enum::BroadCastInt64,  pBroadCastValue);
  default:                                                          return BaseType::BroadCast(eElementType, pBroadCastValue);
  }
}

Expr* InstructionSetSSE2::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  const uint32_t cuiParamCount = static_cast< uint32_t >( crvecArguments.size() );

  if (! IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount))
  {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "SSE2");
  }


  switch (eElementType)
  {
  case VectorElementTypes::Double:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:
      {
        // No intrinsic for this => multiply with 1 or -1
        Expr *pMultiplier = RelationalOperator( eElementType, RelationalOperatorType::Less, crvecArguments[0], CreateZeroVector(eElementType) );
        pMultiplier       = BlendVectors( eElementType, pMultiplier, CreateOnesVector(eElementType, true), CreateOnesVector(eElementType, false) );

        return ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, crvecArguments[0], pMultiplier );
      }
    case BuiltinFunctionsEnum::Ceil:    return _CreateFunctionCall( IntrinsicsSSE2Enum::CeilDouble,  crvecArguments[0] );
    case BuiltinFunctionsEnum::Floor:   return _CreateFunctionCall( IntrinsicsSSE2Enum::FloorDouble, crvecArguments[0] );
    case BuiltinFunctionsEnum::Max:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MaxDouble,   crvecArguments[0], crvecArguments[1] );
    case BuiltinFunctionsEnum::Min:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MinDouble,   crvecArguments[0], crvecArguments[1] );
    case BuiltinFunctionsEnum::Sqrt:    return _CreateFunctionCall( IntrinsicsSSE2Enum::SqrtDouble,  crvecArguments[0] );
    }

    break;

  case VectorElementTypes::UInt8:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:
    case BuiltinFunctionsEnum::Ceil:
    case BuiltinFunctionsEnum::Floor:   return crvecArguments.front();  // Nothing to do for this functions
    case BuiltinFunctionsEnum::Max:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MaxUInt8, crvecArguments[0], crvecArguments[1] );
    case BuiltinFunctionsEnum::Min:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MinUInt8, crvecArguments[0], crvecArguments[1] );
    }

    break;

  case VectorElementTypes::Int16:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Ceil:
    case BuiltinFunctionsEnum::Floor:   return crvecArguments.front();  // Nothing to do for this functions
    case BuiltinFunctionsEnum::Max:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MaxInt16, crvecArguments[0], crvecArguments[1] );
    case BuiltinFunctionsEnum::Min:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MinInt16, crvecArguments[0], crvecArguments[1] );
    }

    break;

  case VectorElementTypes::Int8:  case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:
      if (! AST::BaseClasses::TypeInfo::IsSigned(eElementType))
      {
        return crvecArguments.front();  // Unsigned types already represent absolute values
      }
    case BuiltinFunctionsEnum::Ceil:
    case BuiltinFunctionsEnum::Floor:   return crvecArguments.front();  // Nothing to do for this functions
    }

    break;

  default:    return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
  }

  throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "SSE2");
}

Expr* InstructionSetSSE2::CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, Expr *pMaskExpr)
{
  int32_t             iTestValue      = 0;
  IntrinsicsSSE2Enum  eMoveMaskID     = IntrinsicsSSE2Enum::MoveMaskDouble;

  switch (eMaskElementType)
  {
  case VectorElementTypes::Double:
    {
      eMoveMaskID = IntrinsicsSSE2Enum::MoveMaskDouble;
      iTestValue  = (eCheckType == ActiveElementsCheckType::All) ? 0x3 : 0;
      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::Int16:  case VectorElementTypes::Int32:  case VectorElementTypes::Int64:
  case VectorElementTypes::UInt8: case VectorElementTypes::UInt16: case VectorElementTypes::UInt32: case VectorElementTypes::UInt64:
    {
      eMoveMaskID = IntrinsicsSSE2Enum::MoveMaskInt8;
      iTestValue  = (eCheckType == ActiveElementsCheckType::All) ? 0xFFFF : 0;
      break;
    }

  default:  return BaseType::CheckActiveElements(eMaskElementType, eCheckType, pMaskExpr);
  }

  CallExpr        *pMoveMask      = _CreateFunctionCall(eMoveMaskID, pMaskExpr);
  IntegerLiteral  *pTestConstant  = _GetASTHelper().CreateIntegerLiteral(iTestValue);

  BinaryOperatorKind  eCompareOpType = (eCheckType == ActiveElementsCheckType::Any) ? BO_NE : BO_EQ;

  return _GetASTHelper().CreateBinaryOperator( pMoveMask, pTestConstant, eCompareOpType, _GetClangType(VectorElementTypes::Bool) );
}

Expr* InstructionSetSSE2::CreateOnesVector(VectorElementTypes eElementType, bool bNegative)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return BroadCast( eElementType, _GetASTHelper().CreateLiteral( bNegative ? -1.0 : 1.0 ) );
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return BroadCast( eElementType, _GetASTHelper().CreateIntegerLiteral( bNegative ? -1 : 1 ) );
  default:                                                          return BaseType::CreateOnesVector( eElementType, bNegative );
  }
}

Expr* InstructionSetSSE2::CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder)
{
  if (! bReversedOrder)
  {
    // SSE vector creation methods expect the arguments in reversed order
    return CreateVector( eElementType, _SwapExpressionOrder( crvecElements ), true );
  }


  IntrinsicsSSE2Enum eIntrinID = IntrinsicsSSE2Enum::SetDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsSSE2Enum::SetDouble;  break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE2Enum::SetInt8;    break;
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsSSE2Enum::SetInt16;   break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE2Enum::SetInt32;   break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE2Enum::SetInt64;   break;
  default:                                                          return BaseType::CreateVector(eElementType, crvecElements, bReversedOrder);
  }

  if (crvecElements.size() != GetVectorElementCount(eElementType))
  {
    throw RuntimeErrorException("The number of init expressions must be equal to the vector element count!");
  }

  return _CreateFunctionCall( eIntrinID, crvecElements );
}

Expr* InstructionSetSSE2::CreateZeroVector(VectorElementTypes eElementType)
{
  switch ( eElementType )
  {
  case VectorElementTypes::Double:  return _CreateFunctionCall(IntrinsicsSSE2Enum::SetZeroDouble);
  case VectorElementTypes::Int8:
  case VectorElementTypes::Int16:
  case VectorElementTypes::Int32:
  case VectorElementTypes::Int64:
  case VectorElementTypes::UInt8:
  case VectorElementTypes::UInt16:
  case VectorElementTypes::UInt32:
  case VectorElementTypes::UInt64:  return _CreateFunctionCall(IntrinsicsSSE2Enum::SetZeroInteger);
  default:                          return BaseType::CreateZeroVector( eElementType );
  }
}

Expr* InstructionSetSSE2::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    {
      _CheckExtractIndex(eElementType, uiIndex);

      Expr *pIntermediateValue = nullptr;

      if (uiIndex == 0)
      {
        // The lowest element is requested => it can be extracted directly
        pIntermediateValue = pVectorRef;
      }
      else if (uiIndex == 1)
      {
        // Swap the highest and lowest vector element
        pIntermediateValue = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleDouble, pVectorRef, CreateZeroVector(eElementType), _GetASTHelper().CreateIntegerLiteral(1) );
      }

      return _CreateFunctionCall(IntrinsicsSSE2Enum::ExtractLowestDouble, pIntermediateValue);
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
    {
      _CheckExtractIndex(eElementType, uiIndex);

      Expr *pExtractExpr = _CreateFunctionCall( IntrinsicsSSE2Enum::ExtractInt16, pVectorRef, _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(uiIndex >> 1)) );

      if ((uiIndex & 1) != 0)
      {
        // Odd indices correspond to the upper byte of the 16bit word => Shift extracted value by 8 bits
        pExtractExpr = _GetASTHelper().CreateBinaryOperator(pExtractExpr, _GetASTHelper().CreateIntegerLiteral(8), BO_Shr, pExtractExpr->getType());
      }

      return _CreateValueCast( pExtractExpr, _GetClangType(eElementType), CK_IntegralCast );
    }
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
    {
      _CheckExtractIndex(eElementType, uiIndex);

      Expr *pExtractExpr = _CreateFunctionCall( IntrinsicsSSE2Enum::ExtractInt16, pVectorRef, _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(uiIndex)) );

      return _CreateValueCast( pExtractExpr, _GetClangType(eElementType), CK_IntegralCast );
    }
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    {
      _CheckExtractIndex(eElementType, uiIndex);

      Expr *pIntermediateValue = nullptr;

      if (uiIndex == 0)
      {
        // The lowest element is requested => it can be extracted directly
        pIntermediateValue = pVectorRef;
      }
      else
      {
        // Swap vector elements such that the desired value is in the lowest element
        int32_t iControlFlags = 0;

        switch (uiIndex)
        {
        case 1:   iControlFlags = 0xE1;   break;  // Swap element 0 and 1
        case 2:   iControlFlags = 0xC6;   break;  // Swap element 0 and 2
        case 3:   iControlFlags = 0x27;   break;  // Swap element 0 and 3
        default:  throw InternalErrorException("Unexpected index detected!");
        }

        pIntermediateValue = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, pVectorRef, _GetASTHelper().CreateIntegerLiteral(iControlFlags) );
      }

      pIntermediateValue = _CreateFunctionCall(IntrinsicsSSE2Enum::ExtractLowestInt32, pIntermediateValue);

      if (eElementType == VectorElementTypes::UInt32)
      {
        pIntermediateValue = _CreateValueCast( pIntermediateValue, _GetClangType(eElementType), CK_IntegralCast );
      }

      return pIntermediateValue;
    }
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      _CheckExtractIndex(eElementType, uiIndex);

      Expr *pIntermediateValue = nullptr;

      if (uiIndex == 0)
      {
        // The lowest element is requested => it can be extracted directly
        pIntermediateValue = pVectorRef;
      }
      else
      {
        // Swap the highest and lowest vector element
        pIntermediateValue = _CreateFunctionCall( IntrinsicsSSE2Enum::ShuffleInt32, pVectorRef, _GetASTHelper().CreateIntegerLiteral(0x4E) );
      }

      pIntermediateValue = _CreateFunctionCall( IntrinsicsSSE2Enum::ExtractLowestInt64, pIntermediateValue );

      if (eElementType == VectorElementTypes::UInt64)
      {
        pIntermediateValue = _CreateValueCast( pIntermediateValue, _GetClangType(eElementType), CK_IntegralCast );
      }

      return pIntermediateValue;
    }
  default:  return BaseType::ExtractElement(eElementType, pVectorRef, uiIndex);
  }
}

QualType InstructionSetSSE2::GetVectorType(VectorElementTypes eElementType)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return _GetFunctionReturnType(IntrinsicsSSE2Enum::SetZeroDouble);
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _GetFunctionReturnType(IntrinsicsSSE2Enum::SetZeroInteger);
  default:                                                          return BaseType::GetVectorType(eElementType);
  }
}

Expr* InstructionSetSSE2::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    {
      _CheckInsertIndex(eElementType, uiIndex);

      IntrinsicsSSE2Enum eIntrinID = (uiIndex == 0) ? IntrinsicsSSE2Enum::InsertLowestDouble : IntrinsicsSSE2Enum::UnpackLowDouble;

      return _CreateFunctionCall( eIntrinID, pVectorRef, BroadCast(eElementType, pElementValue) );
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
    {
      _CheckInsertIndex(eElementType, uiIndex);

      VectorElementTypes eSubElemType = (eElementType == VectorElementTypes::Int8) ? VectorElementTypes::Int16 : VectorElementTypes::UInt16;

      Expr *pCombinedValue = ExtractElement(eSubElemType, pVectorRef, uiIndex >> 1);

      if ((uiIndex & 1) == 0)
      {
        // Even indices are copied into the low byte
        pCombinedValue = _GetASTHelper().CreateBinaryOperator( pCombinedValue, _GetASTHelper().CreateIntegerLiteral(0xFF00), BO_And, pCombinedValue->getType() );
        pElementValue  = _GetASTHelper().CreateBinaryOperator( pElementValue,  _GetASTHelper().CreateIntegerLiteral(0x00FF), BO_And, pElementValue->getType() );

        pCombinedValue = _GetASTHelper().CreateParenthesisExpression( pCombinedValue );
        pElementValue  = _GetASTHelper().CreateParenthesisExpression( pElementValue );
      }
      else
      {
        // Odd indices are copied into the high byte
        pElementValue  = _CreateValueCast( pElementValue, _GetClangType(VectorElementTypes::Int32), CK_IntegralCast );

        pCombinedValue = _GetASTHelper().CreateBinaryOperator( pCombinedValue, _GetASTHelper().CreateIntegerLiteral(0xFF), BO_And, pCombinedValue->getType() );
        pElementValue  = _GetASTHelper().CreateBinaryOperator( pElementValue,  _GetASTHelper().CreateIntegerLiteral(0xFF), BO_And, pElementValue->getType() );

        pCombinedValue = _GetASTHelper().CreateParenthesisExpression( pCombinedValue );
        pElementValue  = _GetASTHelper().CreateParenthesisExpression( pElementValue );

        pElementValue  = _GetASTHelper().CreateBinaryOperator( pElementValue, _GetASTHelper().CreateIntegerLiteral(8), BO_Shl, pElementValue->getType() );
        pElementValue  = _GetASTHelper().CreateParenthesisExpression( pElementValue );
      }

      pCombinedValue = _GetASTHelper().CreateBinaryOperator( pCombinedValue, pElementValue, BO_Or, pCombinedValue->getType() );

      return InsertElement(eSubElemType, pVectorRef, pCombinedValue, uiIndex >> 1);
    }
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
    {
      _CheckInsertIndex(eElementType, uiIndex);

      return _CreateFunctionCall( IntrinsicsSSE2Enum::InsertInt16, pVectorRef, pElementValue, _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(uiIndex)) );
    }
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    {
      _CheckInsertIndex(eElementType, uiIndex);

      VectorElementTypes eSubElemType = (eElementType == VectorElementTypes::Int32) ? VectorElementTypes::Int16 : VectorElementTypes::UInt16;

      // Insert lower Word
      Expr *pInsertExpr = InsertElement(eSubElemType, pVectorRef, pElementValue, uiIndex << 1);

      // Insert upper Word
      Expr *pUpperWord = _GetASTHelper().CreateBinaryOperator(pElementValue, _GetASTHelper().CreateIntegerLiteral(16), BO_Shr, pElementValue->getType());

      return InsertElement( eSubElemType, pInsertExpr, pUpperWord, (uiIndex << 1) + 1 );
    }
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      _CheckInsertIndex(eElementType, uiIndex);

      Expr *pBroadCast = BroadCast( eElementType, pElementValue );

      if (uiIndex == 0)
      {
        return _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackHighInt64, pBroadCast, pVectorRef );
      }
      else
      {
        return _CreateFunctionCall( IntrinsicsSSE2Enum::UnpackLowInt64, pVectorRef, pBroadCast );
      }
    }
  default:  return BaseType::InsertElement(eElementType, pVectorRef, pElementValue, uiIndex);
  }
}

bool InstructionSetSSE2::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:     return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Ceil:    return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Floor:   return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Max:     return (uiParamCount == 2);
    case BuiltinFunctionsEnum::Min:     return (uiParamCount == 2);
    case BuiltinFunctionsEnum::Sqrt:    return (uiParamCount == 1);
    default:                            return false;
    }

  case VectorElementTypes::UInt8:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:     return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Ceil:    return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Floor:   return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Max:     return (uiParamCount == 2);
    case BuiltinFunctionsEnum::Min:     return (uiParamCount == 2);
    default:                            return false;
    }

  case VectorElementTypes::Int16:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Ceil:    return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Floor:   return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Max:     return (uiParamCount == 2);
    case BuiltinFunctionsEnum::Min:     return (uiParamCount == 2);
    default:                            return false;
    }

  case VectorElementTypes::Int8:  case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:     return ( (uiParamCount == 1) && (! AST::BaseClasses::TypeInfo::IsSigned(eElementType)) );
    case BuiltinFunctionsEnum::Ceil:    return (uiParamCount == 1);
    case BuiltinFunctionsEnum::Floor:   return (uiParamCount == 1);
    default:                            return false;
    }

  default:  return BaseType::IsBuiltinFunctionSupported( eElementType, eFunctionType, uiParamCount );
  }
}

bool InstructionSetSSE2::IsElementTypeSupported(VectorElementTypes eElementType) const
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
  case VectorElementTypes::Int8:
  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16:
  case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32:
  case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64:
  case VectorElementTypes::UInt64:    return true;
  default:                            return BaseType::IsElementTypeSupported( eElementType );
  }
}

Expr* InstructionSetSSE2::LoadVector(VectorElementTypes eElementType, Expr *pPointerRef)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return _CreateFunctionCall(IntrinsicsSSE2Enum::LoadDouble, pPointerRef);
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      CastExpr *pPointerCast = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType( GetVectorType(VectorElementTypes::Int32) ) );

      return _CreateFunctionCall(IntrinsicsSSE2Enum::LoadInteger, pPointerCast);
    }
  default:  return BaseType::LoadVector(eElementType, pPointerRef);
  }
}

Expr* InstructionSetSSE2::LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex)
{
  switch (eIndexElementType)
  {
  case VectorElementTypes::Int32: case VectorElementTypes::Int64:   break;
  default:  throw RuntimeErrorException( string("Only index element types \"") + AST::BaseClasses::TypeInfo::GetTypeString( VectorElementTypes::Int32 ) + ("\" and \"") +
                                         AST::BaseClasses::TypeInfo::GetTypeString( VectorElementTypes::Int64 ) + ("\" supported for gathered vector loads!") );
  }

  if ( GetVectorElementCount(eElementType) > (GetVectorElementCount(eIndexElementType) * crvecIndexExprs.size()) )
  {
    throw RuntimeErrorException( "The number of vector elements must be less or equal to the number of index elements times the number of index vectors for gathered vector loads!" );
  }
  else if ( (uiGroupIndex != 0) && (uiGroupIndex >= (GetVectorElementCount(eIndexElementType) / GetVectorElementCount(eElementType))) )
  {
    throw RuntimeErrorException( "The group index must be smaller than the size spread between the index element type and the vector element type!" );
  }


  const uint32_t cuiIndexOffset = uiGroupIndex * static_cast<uint32_t>( GetVectorElementCount(eIndexElementType) / GetVectorElementCount(eElementType) );

  ClangASTHelper::ExpressionVectorType vecLoadedElements;

  for (size_t szVecIdx = static_cast<size_t>(0); szVecIdx < crvecIndexExprs.size(); ++szVecIdx)
  {
    for (size_t szElemIdx = static_cast<size_t>(0); szElemIdx < GetVectorElementCount(eIndexElementType); ++szElemIdx)
    {
      if (vecLoadedElements.size() == GetVectorElementCount(eElementType))
      {
        // If more indices are given than required, break as soon as all elements are loaded
        break;
      }

      Expr *pCurrentOffset = ExtractElement( eIndexElementType, crvecIndexExprs[szVecIdx], static_cast<uint32_t>(szElemIdx) + cuiIndexOffset );

      vecLoadedElements.push_back( _GetASTHelper().CreateArraySubscriptExpression( pPointerRef, pCurrentOffset, _GetClangType(eElementType), false ) );
    }
  }

  return CreateVector( eElementType, vecLoadedElements, false );
}

Expr* InstructionSetSSE2::ShiftElements(VectorElementTypes eElementType, Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount)
{
  if (uiCount == 0)
  {
    return pVectorRef;  // Nothing to do
  }

  IntegerLiteral *pShiftCount = _GetASTHelper().CreateIntegerLiteral( static_cast<int32_t>(uiCount) );

  if (bShiftLeft)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      {
        // Convert vector to UInt16 data type, do the shift and convert back (there is no arithmetic left shift)
        ClangASTHelper::ExpressionVectorType vecShiftedVectors;

        for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx)
        {
          Expr *pConvertedVector = ConvertVectorUp( VectorElementTypes::UInt8, VectorElementTypes::UInt16, pVectorRef, uiGroupIdx );
          vecShiftedVectors.push_back( ShiftElements(VectorElementTypes::UInt16, pConvertedVector, bShiftLeft, uiCount) );
        }

        return ConvertVectorDown( VectorElementTypes::UInt16, VectorElementTypes::UInt8, vecShiftedVectors );
      }
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftLeftInt16, pVectorRef, pShiftCount );
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftLeftInt32, pVectorRef, pShiftCount );
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftLeftInt64, pVectorRef, pShiftCount );
    default:                                                          throw RuntimeErrorException("Shift operations are only defined for integer element types!");
    }
  }
  else
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8: case VectorElementTypes::UInt8:
      {
        // Convert vector to a signed / unsigned 16-bit integer data type, do the shift and convert back
        const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2, AST::BaseClasses::TypeInfo::IsSigned(eElementType)).GetType();

        ClangASTHelper::ExpressionVectorType vecShiftedVectors;

        for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx)
        {
          Expr *pConvertedVector = ConvertVectorUp( eElementType, ceIntermediateType, pVectorRef, uiGroupIdx );
          vecShiftedVectors.push_back( ShiftElements(ceIntermediateType, pConvertedVector, bShiftLeft, uiCount) );
        }

        return ConvertVectorDown( ceIntermediateType, VectorElementTypes::UInt8, vecShiftedVectors );
      }
    case VectorElementTypes::Int16:   return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightArithInt16, pVectorRef, pShiftCount );
    case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightLogInt16,   pVectorRef, pShiftCount );
    case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightArithInt32, pVectorRef, pShiftCount );
    case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightLogInt32,   pVectorRef, pShiftCount );
    case VectorElementTypes::Int64:
      {
        // This is unsupported by SSE => Extract elements and shift them separately
        ClangASTHelper::ExpressionVectorType vecElements;

        for (uint32_t uiIndex = 0; uiIndex < GetVectorElementCount(eElementType); ++uiIndex)
        {
          vecElements.push_back( _GetASTHelper().CreateBinaryOperator( ExtractElement(eElementType, pVectorRef, uiIndex), pShiftCount, BO_Shr, _GetClangType(eElementType) ) );
        }

        return CreateVector( eElementType, vecElements, false );
      }
    case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightLogInt64,   pVectorRef, pShiftCount );
    default:                          throw RuntimeErrorException("Shift operations are only defined for integer element types!");
    }
  }
}

Expr* InstructionSetSSE2::StoreVector(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return _CreateFunctionCall(IntrinsicsSSE2Enum::StoreDouble, pPointerRef, pVectorValue);
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      CastExpr *pPointerCast = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType( GetVectorType(VectorElementTypes::Int32) ) );

      return _CreateFunctionCall(IntrinsicsSSE2Enum::StoreInteger, pPointerCast, pVectorValue);
    }
  default:  return BaseType::StoreVector(eElementType, pPointerRef, pVectorValue);
  }
}

Expr* InstructionSetSSE2::StoreVectorMasked(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue, Expr *pMaskRef)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    {
      Expr *pCastedVector = _CreateFunctionCall( IntrinsicsSSE2Enum::CastDoubleToInteger, pVectorValue );
      Expr *pCastedMask   = _CreateFunctionCall( IntrinsicsSSE2Enum::CastDoubleToInteger, pMaskRef );

      return StoreVectorMasked( VectorElementTypes::UInt64, pPointerRef, pCastedVector, pCastedMask );
    }
  case VectorElementTypes::Float:
    {
      Expr *pCastedVector = _CreateFunctionCall( IntrinsicsSSE2Enum::CastFloatToInteger, pVectorValue );
      Expr *pCastedMask   = _CreateFunctionCall( IntrinsicsSSE2Enum::CastFloatToInteger, pMaskRef );

      return StoreVectorMasked( VectorElementTypes::UInt32, pPointerRef, pCastedVector, pCastedMask );
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      Expr *pCastedPointer = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType( _GetClangType(VectorElementTypes::Int8) ) );

      return _CreateFunctionCall( IntrinsicsSSE2Enum::StoreConditionalInteger, pVectorValue, pMaskRef, pCastedPointer );
    }
  default:    return BaseType::StoreVectorMasked( eElementType, pPointerRef, pVectorValue, pMaskRef );
  }
}

Expr* InstructionSetSSE2::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
    switch (eOpType)
    {
    case RelationalOperatorType::Equal:         return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareEqualDouble,         pExprLHS, pExprRHS );
    case RelationalOperatorType::Greater:       return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareGreaterThanDouble,   pExprLHS, pExprRHS );
    case RelationalOperatorType::GreaterEqual:  return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareGreaterEqualDouble,  pExprLHS, pExprRHS );
    case RelationalOperatorType::Less:          return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareLessThanDouble,      pExprLHS, pExprRHS );
    case RelationalOperatorType::LessEqual:     return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareLessEqualDouble,     pExprLHS, pExprRHS );
    case RelationalOperatorType::NotEqual:      return _CreateFunctionCall( IntrinsicsSSE2Enum::CompareNotEqualDouble,      pExprLHS, pExprRHS );
    case RelationalOperatorType::LogicalAnd:    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseAnd, pExprLHS, pExprRHS );
    case RelationalOperatorType::LogicalOr:     return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseOr,  pExprLHS, pExprRHS );
    default:                                    throw InternalErrorException("Unsupported relational operation detected!");
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _RelationalOpInteger( eElementType, eOpType, pExprLHS, pExprRHS );
  default:                                                          return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
  }
}

Expr* InstructionSetSSE2::UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, Expr *pSubExpr)
{
  if      (eOpType == UnaryOperatorType::AddressOf)
  {
    return _GetASTHelper().CreateUnaryOperator( pSubExpr, UO_AddrOf, _GetASTHelper().GetASTContext().getPointerType(pSubExpr->getType()) );
  }
  else if ( (eOpType == UnaryOperatorType::BitwiseNot) || (eOpType == UnaryOperatorType::LogicalNot) )
  {
    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseXOr, pSubExpr, _CreateFullBitMask(eElementType) );
  }
  else if (eOpType == UnaryOperatorType::Minus)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Double:                                  return ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, pSubExpr, CreateOnesVector(eElementType, true) );
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return ArithmeticOperator( eElementType, ArithmeticOperatorType::Subtract, CreateZeroVector(eElementType), pSubExpr );
    default:                                                          return BaseType::UnaryOperator(eElementType, eOpType, pSubExpr);
    }
  }
  else if (eOpType == UnaryOperatorType::Plus)
  {
    return pSubExpr;  // Nothing to do
  }
  else    // Expected a pre-/post-fixed operator here
  {
    IntrinsicsSSE2Enum  eIntrinID   = IntrinsicsSSE2Enum::AddDouble;
    bool                bPrefixedOp = true;

    switch (eOpType)
    {
    case UnaryOperatorType::PostDecrement:
      {
        bPrefixedOp = false;

        switch (eElementType)
        {
        case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsSSE2Enum::SubtractDouble;   break;
        case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE2Enum::SubtractInt8;     break;
        case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt16;    break;
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt32;    break;
        case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt64;    break;
        default:                                                          return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
        }

        break;
      }
    case UnaryOperatorType::PostIncrement:
      {
        bPrefixedOp = false;

        switch (eElementType)
        {
        case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsSSE2Enum::AddDouble;  break;
        case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE2Enum::AddInt8;    break;
        case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsSSE2Enum::AddInt16;   break;
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE2Enum::AddInt32;   break;
        case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE2Enum::AddInt64;   break;
        default:                                                          return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
        }

        break;
      }
    case UnaryOperatorType::PreDecrement:
      {
        bPrefixedOp = true;

        switch (eElementType)
        {
        case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsSSE2Enum::SubtractDouble;   break;
        case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE2Enum::SubtractInt8;     break;
        case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt16;    break;
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt32;    break;
        case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE2Enum::SubtractInt64;    break;
        default:                                                          return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
        }

        break;
      }
    case UnaryOperatorType::PreIncrement:
      {
        bPrefixedOp = true;

        switch (eElementType)
        {
        case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsSSE2Enum::AddDouble;  break;
        case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE2Enum::AddInt8;    break;
        case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsSSE2Enum::AddInt16;   break;
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE2Enum::AddInt32;   break;
        case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE2Enum::AddInt64;   break;
        default:                                                          return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
        }

        break;
      }
    default:    throw InternalErrorException("Unsupported unary operation detected!");
    }

    if (bPrefixedOp)
    {
      return _CreatePrefixedUnaryOp( eIntrinID, eElementType, pSubExpr );
    }
    else
    {
      return _CreatePostfixedUnaryOp( eIntrinID, eElementType, pSubExpr );
    }
  }
}



// Implementation of class InstructionSetSSE3
InstructionSetSSE3::InstructionSetSSE3(ASTContext &rAstContext) : BaseType(rAstContext)
{
  _InitIntrinsicsMap();

  _LookupIntrinsics();
}

Expr* InstructionSetSSE3::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion);
}

void InstructionSetSSE3::_InitIntrinsicsMap()
{
  _InitIntrinsic (IntrinsicsSSE3Enum::LoadInteger, "lddqu_si128" );
}

Expr* InstructionSetSSE3::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  return BaseType::ArithmeticOperator(eElementType, eOpType, pExprLHS, pExprRHS);
}

Expr* InstructionSetSSE3::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  return BaseType::BlendVectors(eElementType, pMaskRef, pVectorTrue, pVectorFalse);
}

Expr* InstructionSetSSE3::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
}

Expr* InstructionSetSSE3::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  return BaseType::ExtractElement(eElementType, pVectorRef, uiIndex);
}

Expr* InstructionSetSSE3::LoadVector(VectorElementTypes eElementType, Expr *pPointerRef)
{
  switch (eElementType)
  {
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      CastExpr *pPointerCast = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType( GetVectorType(VectorElementTypes::Int32) ) );

      return _CreateFunctionCall(IntrinsicsSSE3Enum::LoadInteger, pPointerCast);
    }
  default:  return BaseType::LoadVector(eElementType, pPointerRef);
  }
}

Expr* InstructionSetSSE3::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  return BaseType::InsertElement(eElementType, pVectorRef, pElementValue, uiIndex);
}

bool InstructionSetSSE3::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  return BaseType::IsBuiltinFunctionSupported(eElementType, eFunctionType, uiParamCount);
}

Expr* InstructionSetSSE3::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
}

Expr* InstructionSetSSE3::UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, Expr *pSubExpr)
{
  return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
}



// Implementation of class InstructionSetSSSE3
InstructionSetSSSE3::InstructionSetSSSE3(ASTContext &rAstContext) : BaseType(rAstContext)
{
  _InitIntrinsicsMap();

  _LookupIntrinsics();
}

Expr* InstructionSetSSSE3::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  if (! bMaskConversion)
  {
    // By-pass integer down-conversion chains by the fast SSSE3 8-bit integer shuffles (except mask conversion which are handled quickly by the SSE2 instruction set)

    switch (eSourceType)
    {
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:

      switch (eTargetType)
      {
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
        {
          // Shuffle the low-byte of each packed 32-integer into the low 4 bytes
          ClangASTHelper::ExpressionVectorType vecShuffledVectors;

          for (size_t szIdx = static_cast<size_t>(0); szIdx < crvecVectorRefs.size(); ++szIdx)
          {
            const int32_t ciShuffleConstant = 0x0C080400;

            Expr *pShuffleMask = BroadCast( VectorElementTypes::Int32, _GetASTHelper().CreateIntegerLiteral(ciShuffleConstant) );

            vecShuffledVectors.push_back( _CreateFunctionCall( IntrinsicsSSSE3Enum::ShuffleInt8, crvecVectorRefs[szIdx], pShuffleMask ) );
          }

          // Merge adjacent 4-element vector pairs into 8-element vectors
          vecShuffledVectors[0] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[0], vecShuffledVectors[1], true );
          vecShuffledVectors[1] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[2], vecShuffledVectors[3], true );

          // Merge two remaining 8-element vectors into a 16-element vector
          return _UnpackVectors( VectorElementTypes::Int64, vecShuffledVectors[0], vecShuffledVectors[1], true );
        }
      }

      break;

    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

      switch (eTargetType)
      {
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
        {
          // Shuffle the low-byte of each packed 64-integer into the low 2 bytes
          ClangASTHelper::ExpressionVectorType vecShuffledVectors;

          for (size_t szIdx = static_cast<size_t>(0); szIdx < crvecVectorRefs.size(); ++szIdx)
          {
            const int32_t ciShuffleConstant = 0x0800;

            Expr *pShuffleMask = BroadCast( VectorElementTypes::Int32, _GetASTHelper().CreateIntegerLiteral(ciShuffleConstant) );

            vecShuffledVectors.push_back( _CreateFunctionCall( IntrinsicsSSSE3Enum::ShuffleInt8, crvecVectorRefs[szIdx], pShuffleMask ) );
          }

          // Merge adjacent 2-element vector pairs into 4-element vectors
          for (size_t szIdx = static_cast<size_t>(0); szIdx < (crvecVectorRefs.size() >> 1); ++szIdx)
          {
            vecShuffledVectors[szIdx] = _UnpackVectors(VectorElementTypes::Int16, vecShuffledVectors[szIdx << 1], vecShuffledVectors[(szIdx << 1) + 1], true);
          }

          // Merge adjacent 4-element vector pairs into 8-element vectors
          vecShuffledVectors[0] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[0], vecShuffledVectors[1], true );
          vecShuffledVectors[1] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[2], vecShuffledVectors[3], true );

          // Merge two remaining 8-element vectors into a 16-element vector
          return _UnpackVectors( VectorElementTypes::Int64, vecShuffledVectors[0], vecShuffledVectors[1], true );
        }
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        {
          // Shuffle the low-word of each packed 64-integer into the low 4 bytes
          ClangASTHelper::ExpressionVectorType vecShuffledVectors;

          for (size_t szIdx = static_cast<size_t>(0); szIdx < crvecVectorRefs.size(); ++szIdx)
          {
            const int32_t ciShuffleConstant = 0x09080100;

            Expr *pShuffleMask = BroadCast( VectorElementTypes::Int32, _GetASTHelper().CreateIntegerLiteral(ciShuffleConstant) );

            vecShuffledVectors.push_back( _CreateFunctionCall( IntrinsicsSSSE3Enum::ShuffleInt8, crvecVectorRefs[szIdx], pShuffleMask ) );
          }

          // Merge adjacent 2-element vector pairs into 4-element vectors
          vecShuffledVectors[0] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[0], vecShuffledVectors[1], true );
          vecShuffledVectors[1] = _UnpackVectors( VectorElementTypes::Int32, vecShuffledVectors[2], vecShuffledVectors[3], true );

          // Merge two remaining 4-element vectors into a 8-element vector
          return _UnpackVectors( VectorElementTypes::Int64, vecShuffledVectors[0], vecShuffledVectors[1], true );
        }
      }

      break;
    }
  }

  // If the function has not returned earlier, let the base handle the conversion
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion);
}

void InstructionSetSSSE3::_InitIntrinsicsMap()
{
  // Absolute value computation functions
  _InitIntrinsic( IntrinsicsSSSE3Enum::AbsoluteInt8,  "abs_epi8"  );
  _InitIntrinsic( IntrinsicsSSSE3Enum::AbsoluteInt16, "abs_epi16" );
  _InitIntrinsic( IntrinsicsSSSE3Enum::AbsoluteInt32, "abs_epi32" );

  // Shuffle functions
  _InitIntrinsic( IntrinsicsSSSE3Enum::ShuffleInt8, "shuffle_epi8" );

  // Sign change functions
  _InitIntrinsic( IntrinsicsSSSE3Enum::SignInt8,  "sign_epi8"  );
  _InitIntrinsic( IntrinsicsSSSE3Enum::SignInt16, "sign_epi16" );
  _InitIntrinsic( IntrinsicsSSSE3Enum::SignInt32, "sign_epi32" );
}

Expr* InstructionSetSSSE3::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  return BaseType::ArithmeticOperator(eElementType, eOpType, pExprLHS, pExprRHS);
}

Expr* InstructionSetSSSE3::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  return BaseType::BlendVectors(eElementType, pMaskRef, pVectorTrue, pVectorFalse);
}

Expr* InstructionSetSSSE3::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  const uint32_t cuiParamCount = static_cast< uint32_t >(crvecArguments.size());

  if (! IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount))
  {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "SSSE3");
  }


  if (eFunctionType == BuiltinFunctionsEnum::Abs)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:    return _CreateFunctionCall( IntrinsicsSSSE3Enum::AbsoluteInt8,  crvecArguments[0] );
    case VectorElementTypes::Int16:   return _CreateFunctionCall( IntrinsicsSSSE3Enum::AbsoluteInt16, crvecArguments[0] );
    case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSSE3Enum::AbsoluteInt32, crvecArguments[0] );
    }
  }

  // If the function has not returned earlier, call the base implementation
  return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
}

Expr* InstructionSetSSSE3::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  return BaseType::ExtractElement(eElementType, pVectorRef, uiIndex);
}

Expr* InstructionSetSSSE3::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  return BaseType::InsertElement(eElementType, pVectorRef, pElementValue, uiIndex);
}

bool InstructionSetSSSE3::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  if (eFunctionType == BuiltinFunctionsEnum::Abs)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:
    case VectorElementTypes::Int16:
    case VectorElementTypes::Int32:   return (uiParamCount == 1);
    }
  }

  // If the function has not returned earlier, call the base implementation
  return BaseType::IsBuiltinFunctionSupported(eElementType, eFunctionType, uiParamCount);
}

Expr* InstructionSetSSSE3::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
}

Expr* InstructionSetSSSE3::UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, Expr *pSubExpr)
{
  if (eOpType == UnaryOperatorType::Minus)
  {
    // Boost integer negations by fast SSSE3 "sign" routines

    switch (eElementType)
    {
    case VectorElementTypes::Int8:
    case VectorElementTypes::UInt8:   return _CreateFunctionCall( IntrinsicsSSSE3Enum::SignInt8,  pSubExpr, BroadCast( eElementType, _GetASTHelper().CreateIntegerLiteral(-1) ) );
    case VectorElementTypes::Int16:
    case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSSE3Enum::SignInt16, pSubExpr, BroadCast( eElementType, _GetASTHelper().CreateIntegerLiteral(-1) ) );
    case VectorElementTypes::Int32:
    case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSSE3Enum::SignInt32, pSubExpr, BroadCast( eElementType, _GetASTHelper().CreateIntegerLiteral(-1) ) );
    }
  }

  // If the function has not returned earlier, let the base handle the unary operator
  return BaseType::UnaryOperator( eElementType, eOpType, pSubExpr );
}



// Implementation of class InstructionSetSSE4_1
InstructionSetSSE4_1::InstructionSetSSE4_1(ASTContext &rAstContext) : BaseType(rAstContext)
{
  _InitIntrinsicsMap();

  _CreateMissingIntrinsicsSSE4_1();

  _LookupIntrinsics();
}

Expr* InstructionSetSSE4_1::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  if (bMaskConversion)
  {
    // Boost upward mask conversions by fast SSE4.1 signed integer upward conversions (except 32-bit to 64-bit, this is done faster by SSE2 shuffle)
    switch (eSourceType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:

      switch (eTargetType)
      {
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
        {
          const size_t cszSourceSize = AST::BaseClasses::TypeInfo::GetTypeSize( eSourceType );
          const size_t cszTargetSize = AST::BaseClasses::TypeInfo::GetTypeSize( eTargetType );
          
          if (cszSourceSize < cszTargetSize)
          {
            const VectorElementTypes ceNewSourceType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType( cszSourceSize, true ).GetType();
            const VectorElementTypes ceNewTargetType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType( cszTargetSize, true ).GetType();

            return ConvertVectorUp(ceNewSourceType, ceNewTargetType, crvecVectorRefs.front(), uiGroupIndex);
          }
        }
      }
    }
  }
  else
  {
    // Handle all upward conversions from integer types to integer types
    bool                  bHandleConversion = true;
    IntrinsicsSSE4_1Enum  eConvertID        = IntrinsicsSSE4_1Enum::ConvertInt8Int32;

    switch (eTargetType)
    {
    case VectorElementTypes::Int16: case VectorElementTypes::UInt16:

      switch (eSourceType)
      {
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  // Special downward conversion
      {
        // We need to remove the high-word from all vector elements to avoid the saturation
        const int32_t cuiMaskConstant = 0xFFFF;

        Expr *pValuesLow  = ArithmeticOperator( eSourceType, ArithmeticOperatorType::BitwiseAnd, crvecVectorRefs[0], BroadCast( eSourceType, _GetASTHelper().CreateIntegerLiteral(cuiMaskConstant) ) );
        Expr *pValuesHigh = ArithmeticOperator( eSourceType, ArithmeticOperatorType::BitwiseAnd, crvecVectorRefs[1], BroadCast( eSourceType, _GetASTHelper().CreateIntegerLiteral(cuiMaskConstant) ) );

        return _CreateFunctionCall( IntrinsicsSSE4_1Enum::PackInt32ToUInt16, pValuesLow, pValuesHigh );
      }
      case VectorElementTypes::Int8:    eConvertID = IntrinsicsSSE4_1Enum::ConvertInt8Int16;    break;
      case VectorElementTypes::UInt8:   eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt8Int16;   break;
      default:                          bHandleConversion = false;
      }

      break;

    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:

      switch (eSourceType)
      {
      case VectorElementTypes::Int8:    eConvertID = IntrinsicsSSE4_1Enum::ConvertInt8Int32;    break;
      case VectorElementTypes::UInt8:   eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt8Int32;   break;
      case VectorElementTypes::Int16:   eConvertID = IntrinsicsSSE4_1Enum::ConvertInt16Int32;   break;
      case VectorElementTypes::UInt16:  eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt16Int32;  break;
      default:                          bHandleConversion = false;
      }

      break;

    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:

      switch (eSourceType)
      {
      case VectorElementTypes::Int8:    eConvertID = IntrinsicsSSE4_1Enum::ConvertInt8Int64;    break;
      case VectorElementTypes::UInt8:   eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt8Int64;   break;
      case VectorElementTypes::Int16:   eConvertID = IntrinsicsSSE4_1Enum::ConvertInt16Int64;   break;
      case VectorElementTypes::UInt16:  eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt16Int64;  break;
      case VectorElementTypes::Int32:   eConvertID = IntrinsicsSSE4_1Enum::ConvertInt32Int64;   break;
      case VectorElementTypes::UInt32:  eConvertID = IntrinsicsSSE4_1Enum::ConvertUInt32Int64;  break;
      default:                          bHandleConversion = false;
      }

      break;

    default:  bHandleConversion = false;
    }

    if (bHandleConversion)
    {
      const uint32_t cuiShiftMultiplier = 16 / static_cast< uint32_t >( AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) );

      return _CreateFunctionCall( eConvertID, _ShiftIntegerVectorBytes(crvecVectorRefs.front(), uiGroupIndex * cuiShiftMultiplier, false) );
    }
  }

  // If the function has not returned earlier, let the base handle the conversion
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion);
}

Expr* InstructionSetSSE4_1::_ExtractElement(VectorElementTypes eElementType, IntrinsicsSSE4_1Enum eIntrinType, Expr *pVectorRef, uint32_t uiIndex)
{
  _CheckExtractIndex(eElementType, uiIndex);

  Expr *pExtractExpr = _CreateFunctionCall( eIntrinType, pVectorRef, _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(uiIndex)) );

  QualType qtReturnType = _GetClangType(eElementType);
  if (qtReturnType != pExtractExpr->getType())
  {
    pExtractExpr = _CreateValueCast( pExtractExpr, qtReturnType, CK_IntegralCast );
  }

  return pExtractExpr;
}

void InstructionSetSSE4_1::_InitIntrinsicsMap()
{
  // Blending functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::BlendDouble,  "blendv_pd"   );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::BlendFloat,   "blendv_ps"   );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::BlendInteger, "blendv_epi8" );

  // Comparison functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::CompareEqualInt64, "cmpeq_epi64" );

  // Convert functions for signed integers
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt8Int16,  "cvtepi8_epi16"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt8Int32,  "cvtepi8_epi32"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt8Int64,  "cvtepi8_epi64"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt16Int32, "cvtepi16_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt16Int64, "cvtepi16_epi64" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertInt32Int64, "cvtepi32_epi64" );

  // Convert functions for unsigned integers
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt8Int16,  "cvtepu8_epi16"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt8Int32,  "cvtepu8_epi32"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt8Int64,  "cvtepu8_epi64"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt16Int32, "cvtepu16_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt16Int64, "cvtepu16_epi64" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ConvertUInt32Int64, "cvtepu32_epi64" );

  // Extract functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ExtractInt8,  "extract_epi8"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ExtractInt32, "extract_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::ExtractInt64, "extract_epi64" );

  // Insert functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::InsertFloat, "insert_ps"    );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::InsertInt8,  "insert_epi8"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::InsertInt32, "insert_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::InsertInt64, "insert_epi64" );

  // Maximum functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MaxInt8,   "max_epi8"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MaxInt32,  "max_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MaxUInt16, "max_epu16" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MaxUInt32, "max_epu32" );

  // Minimum functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MinInt8,   "min_epi8"  );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MinInt32,  "min_epi32" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MinUInt16, "min_epu16" );
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MinUInt32, "min_epu32" );

  // Multiply functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::MultiplyInt32, "mullo_epi32" );

  // Packing functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::PackInt32ToUInt16, "packus_epi32" );

  // Testing functions
  _InitIntrinsic( IntrinsicsSSE4_1Enum::TestControl, "testc_si128" );
}

Expr* InstructionSetSSE4_1::_InsertElement(VectorElementTypes eElementType, IntrinsicsSSE4_1Enum eIntrinType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  _CheckInsertIndex(eElementType, uiIndex);

  if (eElementType == VectorElementTypes::Float)
  {
    pElementValue   = BroadCast(eElementType, pElementValue);
    uiIndex       <<= 4;
  }

  return _CreateFunctionCall( eIntrinType, pVectorRef, pElementValue, _GetASTHelper().CreateIntegerLiteral(static_cast<int32_t>(uiIndex)) );
}

Expr* InstructionSetSSE4_1::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  if (eOpType == ArithmeticOperatorType::Multiply)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MultiplyInt32, pExprLHS, pExprRHS );
    default:                                                          return BaseType::ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS );
    }
  }
  else
  {
    return BaseType::ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS );
  }
}

Expr* InstructionSetSSE4_1::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::BlendDouble,  pVectorFalse, pVectorTrue, pMaskRef );
  case VectorElementTypes::Float:                                   return _CreateFunctionCall( IntrinsicsSSE4_1Enum::BlendFloat,   pVectorFalse, pVectorTrue, pMaskRef );
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::BlendInteger, pVectorFalse, pVectorTrue, pMaskRef );
  default:                                                          return BaseType::BlendVectors( eElementType, pMaskRef, pVectorTrue, pVectorFalse );
  }
}

Expr* InstructionSetSSE4_1::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  const uint32_t cuiParamCount = static_cast< uint32_t >(crvecArguments.size());

  if (!IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount))
  {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "SSSE3");
  }


  if (eFunctionType == BuiltinFunctionsEnum::Max)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:    return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MaxInt8,   crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MaxUInt16, crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MaxInt32,  crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MaxUInt32, crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::Int64:   case VectorElementTypes::UInt64:
      {
        Expr *pMask = RelationalOperator( eElementType, RelationalOperatorType::Greater, crvecArguments[0], crvecArguments[1] );

        return BlendVectors( eElementType, pMask, crvecArguments[0], crvecArguments[1] );
      }
    }
  }
  else if (eFunctionType == BuiltinFunctionsEnum::Min)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:    return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MinInt8,   crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MinUInt16, crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::Int32:   return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MinInt32,  crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::MinUInt32, crvecArguments[0], crvecArguments[1] );
    case VectorElementTypes::Int64:   case VectorElementTypes::UInt64:
      {
        Expr *pMask = RelationalOperator( eElementType, RelationalOperatorType::Greater, crvecArguments[0], crvecArguments[1] );

        return BlendVectors( eElementType, pMask, crvecArguments[1], crvecArguments[0] );
      }
    }
  }

  // If the function has not returned earlier, call the base implementation
  return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
}

Expr* InstructionSetSSE4_1::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  switch (eElementType)
  {
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _ExtractElement( eElementType, IntrinsicsSSE4_1Enum::ExtractInt8,  pVectorRef, uiIndex );
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _ExtractElement( eElementType, IntrinsicsSSE4_1Enum::ExtractInt32, pVectorRef, uiIndex );
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _ExtractElement( eElementType, IntrinsicsSSE4_1Enum::ExtractInt64, pVectorRef, uiIndex );
  default:                                                          return BaseType::ExtractElement( eElementType, pVectorRef, uiIndex );
  }
}

Expr* InstructionSetSSE4_1::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  switch (eElementType)
  {
  case VectorElementTypes::Float:                                   return _InsertElement( eElementType, IntrinsicsSSE4_1Enum::InsertFloat, pVectorRef, pElementValue, uiIndex );
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   return _InsertElement( eElementType, IntrinsicsSSE4_1Enum::InsertInt8,  pVectorRef, pElementValue, uiIndex );
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _InsertElement( eElementType, IntrinsicsSSE4_1Enum::InsertInt32, pVectorRef, pElementValue, uiIndex );
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _InsertElement( eElementType, IntrinsicsSSE4_1Enum::InsertInt64, pVectorRef, pElementValue, uiIndex );
  default:                                                          return BaseType::InsertElement( eElementType, pVectorRef, pElementValue, uiIndex );
  }
}

bool InstructionSetSSE4_1::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  if ( (eFunctionType == BuiltinFunctionsEnum::Max) || (eFunctionType == BuiltinFunctionsEnum::Min) )
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int8:  case VectorElementTypes::UInt16:
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return (uiParamCount == 2);
    }
  }

  // If the function has not returned earlier, call the base implementation
  return BaseType::IsBuiltinFunctionSupported(eElementType, eFunctionType, uiParamCount);
}

Expr* InstructionSetSSE4_1::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  if (eOpType == RelationalOperatorType::Equal)
  {
    switch (eElementType)
    {
    case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _CreateFunctionCall( IntrinsicsSSE4_1Enum::CompareEqualInt64, pExprLHS, pExprRHS );
    default:                                                          return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
    }
  }
  else
  {
    return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
  }
}



// Implementation of class InstructionSetSSE4_2
InstructionSetSSE4_2::InstructionSetSSE4_2(::clang::ASTContext &rAstContext) : BaseType(rAstContext)
{
  _InitIntrinsicsMap();

  _LookupIntrinsics();
}

void InstructionSetSSE4_2::_InitIntrinsicsMap()
{
  _InitIntrinsic( IntrinsicsSSE4_2Enum::CompareGreaterThanInt64, "cmpgt_epi64" );
}

Expr* InstructionSetSSE4_2::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  switch (eElementType)
  {
  case VectorElementTypes::Int64:
    switch (eOpType)
    {
    case RelationalOperatorType::Greater:       return _CreateFunctionCall( IntrinsicsSSE4_2Enum::CompareGreaterThanInt64, pExprLHS, pExprRHS );
    case RelationalOperatorType::GreaterEqual:
      {
        Expr *pEqualExpr    = RelationalOperator( eElementType, RelationalOperatorType::Equal,   pExprLHS, pExprRHS );
        Expr *pGreaterExpr  = RelationalOperator( eElementType, RelationalOperatorType::Greater, pExprLHS, pExprRHS );

        return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseOr, pEqualExpr, pGreaterExpr );
      }
    case RelationalOperatorType::Less:          return UnaryOperator( eElementType, UnaryOperatorType::LogicalNot, RelationalOperator(eElementType, RelationalOperatorType::GreaterEqual, pExprLHS, pExprRHS) );
    case RelationalOperatorType::LessEqual:     return UnaryOperator( eElementType, UnaryOperatorType::LogicalNot, RelationalOperator(eElementType, RelationalOperatorType::Greater,      pExprLHS, pExprRHS) );
    default:                                    return BaseType::RelationalOperator( eElementType, eOpType, pExprLHS, pExprRHS );
    }
  default:  return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS, pExprRHS);
  }
}



// vim: set ts=2 sw=2 sts=2 et ai:

