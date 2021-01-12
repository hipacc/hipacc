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


// Implementation of class InstructionSetExceptions
std::string InstructionSetExceptions::IndexOutOfRange::_ConvertLimit(uint32_t uiUpperLimit)
{
  std::stringstream streamOutput;

  streamOutput << uiUpperLimit;

  return streamOutput.str();
}

InstructionSetExceptions::IndexOutOfRange::IndexOutOfRange(std::string strMethodType, VectorElementTypes eElementType, uint32_t uiUpperLimit) :
      BaseType( std::string("The index for a \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + std::string("\" element ") +
                strMethodType + std::string(" must be in the range of [0; ") + _ConvertLimit(uiUpperLimit) + std::string("] !"))
{
}

std::string InstructionSetExceptions::UnsupportedBuiltinFunctionType::_ConvertParamCount(uint32_t uiParamCount)
{
  std::stringstream streamOutput;

  streamOutput << uiParamCount;

  return streamOutput.str();
}

InstructionSetExceptions::UnsupportedBuiltinFunctionType::UnsupportedBuiltinFunctionType( VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType,
                                                                                          uint32_t uiParamCount, std::string strInstructionSetName ) :
      BaseType( std::string("The built-in function \"") + GetBuiltinFunctionTypeString(eFunctionType) + std::string("(") + _ConvertParamCount(uiParamCount) +
                std::string(")\" is not supported for element type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + std::string("\" in the instruction set \"") +
                strInstructionSetName + std::string("\" !") )
{
}

InstructionSetExceptions::UnsupportedConversion::UnsupportedConversion(VectorElementTypes eSourceType, VectorElementTypes eTargetType, std::string strInstructionSetName) :
      BaseType( std::string("A conversion from type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eSourceType) + std::string("\" to type \"") +
                AST::BaseClasses::TypeInfo::GetTypeString(eTargetType) + std::string("\" is not supported in the instruction set \"") + 
                strInstructionSetName + std::string("\" !") )
{
}



// Implementation of class InstructionSetBase
InstructionSetBase::InstructionSetBase(ASTContext &rAstContext, std::string strFunctionNamePrefix) : _ASTHelper(rAstContext), _strIntrinsicPrefix(strFunctionNamePrefix)
{
  const size_t cszPrefixLength = strFunctionNamePrefix.size();
  ClangASTHelper::FunctionDeclarationVectorType vecFunctionDecls = _ASTHelper.GetKnownFunctionDeclarations();

  for (auto itFuncDecl : vecFunctionDecls)
  {
    std::string strFuncName = ClangASTHelper::GetFullyQualifiedFunctionName(itFuncDecl);

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

void InstructionSetBase::_CreateIntrinsicDeclaration(std::string strFunctionName, const QualType &crReturnType, const ClangASTHelper::QualTypeVectorType &crvecArgTypes, const ClangASTHelper::StringVectorType &crvecArgNames)
{
  if (_mapKnownFuncDecls.find(strFunctionName) == _mapKnownFuncDecls.end())
  {
    _mapKnownFuncDecls[ strFunctionName ].push_back( _ASTHelper.CreateFunctionDeclaration(strFunctionName, crReturnType, crvecArgNames, crvecArgTypes) );
  }
}

void InstructionSetBase::_CreateIntrinsicDeclaration(std::string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, std::string strArgName1)
{
  ClangASTHelper::QualTypeVectorType  vecArgTypes;
  ClangASTHelper::StringVectorType    vecArgNames;

  vecArgTypes.push_back( crArgType1 );
  vecArgNames.push_back( strArgName1 );

  _CreateIntrinsicDeclaration(strFunctionName, crReturnType, vecArgTypes, vecArgNames);
}

void InstructionSetBase::_CreateIntrinsicDeclaration(std::string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, std::string strArgName1, const QualType &crArgType2, std::string strArgName2)
{
  ClangASTHelper::QualTypeVectorType  vecArgTypes;
  ClangASTHelper::StringVectorType    vecArgNames;

  vecArgTypes.push_back( crArgType1 );
  vecArgNames.push_back( strArgName1 );

  vecArgTypes.push_back( crArgType2 );
  vecArgNames.push_back( strArgName2 );

  _CreateIntrinsicDeclaration(strFunctionName, crReturnType, vecArgTypes, vecArgNames);
}

void InstructionSetBase::_CreateIntrinsicDeclaration(std::string strFunctionName, const QualType &crReturnType, const QualType &crArgType1, std::string strArgName1, const QualType &crArgType2, std::string strArgName2, const QualType &crArgType3, std::string strArgName3)
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
  QualType  qtFloat       = _GetClangType(VectorElementTypes::Float);

  // Create missing SSE intrinsic functions
  _CreateIntrinsicDeclaration( "_mm_ceil_ps",     qtFloatVector, qtFloatVector, "a" );
  _CreateIntrinsicDeclaration( "_mm_floor_ps",    qtFloatVector, qtFloatVector, "a" );
  _CreateIntrinsicDeclaration( "_mm_set1_ps",     qtFloatVector, qtFloat,       "a" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_ps",  qtFloatVector, qtFloatVector, "a", qtFloatVector, "b", _GetClangType(VectorElementTypes::UInt32), "imm" );
}

void InstructionSetBase::_CreateMissingIntrinsicsSSE2()
{
  // Get required types
  QualType  qtDoubleVector  = _GetFunctionReturnType("_mm_setzero_pd");
  QualType  qtIntegerVector = _GetFunctionReturnType("_mm_setzero_si128");
  QualType  qtInt           = _GetClangType(VectorElementTypes::Int32);
  QualType  qtLong          = _GetClangType(VectorElementTypes::Int64);

  // Create missing SSE2 intrinsic functions
  _CreateIntrinsicDeclaration( "_mm_cvttsd_si64",     qtLong,          qtDoubleVector,  "a" );
  _CreateIntrinsicDeclaration( "_mm_cvtsi128_si64",   qtLong,          qtIntegerVector, "a" );
  _CreateIntrinsicDeclaration( "_mm_ceil_pd",         qtDoubleVector,  qtDoubleVector,  "a" );
  _CreateIntrinsicDeclaration( "_mm_floor_pd",        qtDoubleVector,  qtDoubleVector,  "a" );
  _CreateIntrinsicDeclaration( "_mm_slli_si128",      qtIntegerVector, qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_srli_si128",      qtIntegerVector, qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_set_epi64x",      qtIntegerVector, qtLong,          "e1", qtLong,         "e0"  );
  _CreateIntrinsicDeclaration( "_mm_set1_epi64x",     qtIntegerVector, qtLong,          "a" );
  _CreateIntrinsicDeclaration( "_mm_shufflehi_epi16", qtIntegerVector, qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shufflelo_epi16", qtIntegerVector, qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_epi32",   qtIntegerVector, qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_shuffle_pd",      qtDoubleVector,  qtDoubleVector,  "a",  qtDoubleVector, "b", qtInt, "imm" );
  _CreateIntrinsicDeclaration( "_mm_extract_epi16",   qtInt,           qtIntegerVector, "a",  qtInt,          "imm" );
  _CreateIntrinsicDeclaration( "_mm_insert_epi16",    qtIntegerVector, qtIntegerVector, "a",  qtInt,          "i", qtInt, "imm" );
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

void InstructionSetBase::_CreateMissingIntrinsicsAVX()
{
  // Get required types
  QualType  qtDoubleVectorAVX   = _GetFunctionReturnType( "_mm256_setzero_pd"      );
  QualType  qtDoubleVectorSSE   = _GetFunctionReturnType( "_mm256_castpd256_pd128" );
  QualType  qtFloatVectorAVX    = _GetFunctionReturnType( "_mm256_setzero_ps"      );
  QualType  qtFloatVectorSSE    = _GetFunctionReturnType( "_mm256_castps256_ps128" );
  QualType  qtIntegerVectorAVX  = _GetFunctionReturnType( "_mm256_setzero_si256"   );
  QualType  qtIntegerVectorSSE  = _GetFunctionReturnType( "_mm256_castsi256_si128" );
  QualType  qtInt               = _GetClangType( VectorElementTypes::Int32 );
  QualType  qtInt8              = _GetClangType( VectorElementTypes::Int8  );
  QualType  qtInt16             = _GetClangType( VectorElementTypes::Int16 );
  QualType  qtInt64             = _GetClangType( VectorElementTypes::Int64 );
  QualType  qtConstInt          = qtInt;
  qtConstInt.addConst();

  // Create missing AVX intrinsic functions
  _CreateIntrinsicDeclaration( "_mm256_ceil_pd",           qtDoubleVectorAVX,  qtDoubleVectorAVX,  "a" );
  _CreateIntrinsicDeclaration( "_mm256_ceil_ps",           qtFloatVectorAVX,   qtFloatVectorAVX,   "a" );
  _CreateIntrinsicDeclaration( "_mm256_cmp_pd",            qtDoubleVectorAVX,  qtDoubleVectorAVX,  "a",  qtDoubleVectorAVX,  "b",  qtConstInt, "imm" );
  _CreateIntrinsicDeclaration( "_mm256_cmp_ps",            qtFloatVectorAVX,   qtFloatVectorAVX,   "a",  qtFloatVectorAVX,   "b",  qtConstInt, "imm" );
  _CreateIntrinsicDeclaration( "_mm256_extract_epi8",      qtInt8,             qtIntegerVectorAVX, "a",  qtConstInt,         "index" );
  _CreateIntrinsicDeclaration( "_mm256_extract_epi16",     qtInt16,            qtIntegerVectorAVX, "a",  qtConstInt,         "index" );
  _CreateIntrinsicDeclaration( "_mm256_extract_epi32",     qtInt,              qtIntegerVectorAVX, "a",  qtConstInt,         "index" );
  _CreateIntrinsicDeclaration( "_mm256_extract_epi64",     qtInt64,            qtIntegerVectorAVX, "a",  qtConstInt,         "index" );
  _CreateIntrinsicDeclaration( "_mm256_extractf128_pd",    qtDoubleVectorSSE,  qtDoubleVectorAVX,  "a",  qtConstInt,         "imm" );
  _CreateIntrinsicDeclaration( "_mm256_extractf128_ps",    qtFloatVectorSSE,   qtFloatVectorAVX,   "a",  qtConstInt,         "imm" );
  _CreateIntrinsicDeclaration( "_mm256_extractf128_si256", qtIntegerVectorSSE, qtIntegerVectorAVX, "a",  qtConstInt,         "imm" );
  _CreateIntrinsicDeclaration( "_mm256_floor_pd",          qtDoubleVectorAVX,  qtDoubleVectorAVX,  "a" );
  _CreateIntrinsicDeclaration( "_mm256_floor_ps",          qtFloatVectorAVX,   qtFloatVectorAVX,   "a" );
  _CreateIntrinsicDeclaration( "_mm256_insertf128_pd",     qtDoubleVectorAVX,  qtDoubleVectorAVX,  "a",  qtDoubleVectorSSE,  "b",  qtInt,      "imm" );
  _CreateIntrinsicDeclaration( "_mm256_insertf128_ps",     qtFloatVectorAVX,   qtFloatVectorAVX,   "a",  qtFloatVectorSSE,   "b",  qtInt,      "imm" );
  _CreateIntrinsicDeclaration( "_mm256_insertf128_si256",  qtIntegerVectorAVX, qtIntegerVectorAVX, "a",  qtIntegerVectorSSE, "b",  qtInt,      "imm" );
  _CreateIntrinsicDeclaration( "_mm256_permute2f128_ps",   qtFloatVectorAVX,   qtFloatVectorAVX,   "a",  qtFloatVectorAVX,   "b",  qtInt,      "imm" );
  _CreateIntrinsicDeclaration( "_mm256_shuffle_pd",        qtDoubleVectorAVX,  qtDoubleVectorAVX,  "a",  qtDoubleVectorAVX,  "b",  qtConstInt, "imm" );
  _CreateIntrinsicDeclaration( "_mm256_shuffle_ps",        qtFloatVectorAVX,   qtFloatVectorAVX,   "a",  qtFloatVectorAVX,   "b",  qtConstInt, "imm" );
  _CreateIntrinsicDeclaration( "_mm256_set_m128d",         qtDoubleVectorAVX,  qtDoubleVectorSSE,  "hi", qtDoubleVectorSSE,  "lo"  );
  _CreateIntrinsicDeclaration( "_mm256_set_m128",          qtFloatVectorAVX,   qtFloatVectorSSE,   "hi", qtFloatVectorSSE,   "lo"  );
  _CreateIntrinsicDeclaration( "_mm256_set_m128i",         qtIntegerVectorAVX, qtIntegerVectorSSE, "hi", qtIntegerVectorSSE, "lo"  );
}

void InstructionSetBase::_CreateMissingIntrinsicsAVX2() {
  // Get required types
  QualType qtIntegerVectorAVX = _GetFunctionReturnType("_mm256_setzero_si256");
  QualType qtIntegerVectorSSE = _GetFunctionReturnType("_mm256_castsi256_si128");
  QualType qtInt = _GetClangType(VectorElementTypes::Int32);
  QualType qtInt32Pointer = _ASTHelper.GetPointerType(_GetClangType(VectorElementTypes::Int32));
  QualType qtInt64Pointer = _ASTHelper.GetPointerType(_GetClangType(VectorElementTypes::Int64));
  QualType qtFloatPointer = _ASTHelper.GetPointerType(_GetClangType(VectorElementTypes::Float));
  QualType qtDoublePointer = _ASTHelper.GetPointerType(_GetClangType(VectorElementTypes::Double));

  // Create missing AVX2 intrinsic functions
  _CreateIntrinsicDeclaration("_mm256_cmpeq_epi8", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpeq_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpeq_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpeq_epi64", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpgt_epi8", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpgt_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpgt_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cmpgt_epi64", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_cvtepi8_epi16", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepi8_epi32", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepi8_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu8_epi16", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu8_epi32", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu8_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepi16_epi32", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepi16_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu16_epi32", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu16_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepi32_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_cvtepu32_epi64", qtIntegerVectorAVX, qtIntegerVectorSSE, "a");
  _CreateIntrinsicDeclaration("_mm256_extracti128_si256", qtIntegerVectorSSE, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_i32gather_epi32", qtInt32Pointer, qtIntegerVectorAVX, "vindex", qtInt, "scale");
  _CreateIntrinsicDeclaration("_mm256_i32gather_epi64", qtInt64Pointer, qtIntegerVectorAVX, "vindex", qtInt, "scale");
  _CreateIntrinsicDeclaration("_mm256_i32gather_ps", qtFloatPointer, qtIntegerVectorAVX, "vindex", qtInt, "scale");
  _CreateIntrinsicDeclaration("_mm256_i32gather_pd", qtDoublePointer, qtIntegerVectorAVX, "vindex", qtInt, "scale");
  _CreateIntrinsicDeclaration("_mm256_slli_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_slli_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_slli_epi64", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_slli_si256", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srai_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srai_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srli_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srli_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srli_epi64", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_srli_si256", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_shuffle_epi8", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_shuffle_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtInt, "imm");
  _CreateIntrinsicDeclaration("_mm256_packs_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_packs_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_packus_epi16", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_packus_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "b");
  _CreateIntrinsicDeclaration("_mm256_permutevar8x32_epi32", qtIntegerVectorAVX, qtIntegerVectorAVX, "a", qtIntegerVectorAVX, "idx");
}

CastExpr* InstructionSetBase::_CreatePointerCast(Expr *pPointerRef, const QualType &crNewPointerType)
{
  return _GetASTHelper().CreateReinterpretCast(pPointerRef, crNewPointerType, CK_ArrayToPointerDecay);//CK_ReinterpretMemberPointer);
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

ClangASTHelper::FunctionDeclarationVectorType InstructionSetBase::_GetFunctionDecl(std::string strFunctionName)
{
  auto itFunctionDecl = _mapKnownFuncDecls.find(strFunctionName);

  if (itFunctionDecl != _mapKnownFuncDecls.end())
  {
    return itFunctionDecl->second;
  }
  else
  {
    throw InternalErrorException(std::string("Cannot find function \"") + strFunctionName + std::string("\" !"));
  }
}

QualType InstructionSetBase::_GetFunctionReturnType(std::string strFunctionName)
{
  auto vecFunctionDecls = _GetFunctionDecl(strFunctionName);

  if ( _GetASTHelper().AreSignaturesEqual( vecFunctionDecls ) )
  {
    return vecFunctionDecls.front()->getReturnType();
  }
  else
  {
    throw InternalErrorException(std::string("The function declaration \"") + strFunctionName + std::string("\" is ambiguous!"));
  }
}

QualType InstructionSetBase::_GetVectorType(VectorElementTypes eElementType, bool bIsConst)
{
  QualType qtReturnType = GetVectorType(eElementType);

  if (bIsConst)
  {
    qtReturnType.addConst();
  }

  return qtReturnType;
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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSE, "SSE" );
}

void InstructionSetSSE::_CheckElementType(VectorElementTypes eElementType) const
{
  if (eElementType != VectorElementTypes::Float)
  {
    throw RuntimeErrorException(std::string("The element type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eElementType) + std::string("\" is not supported in instruction set \"SSE\"!"));
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

Expr* InstructionSetSSE::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
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

  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

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

Expr* InstructionSetSSE::CheckSingleMaskElement(VectorElementTypes eMaskElementType, Expr *pMaskExpr, uint32_t uiIndex)
{
  _CheckElementType(eMaskElementType);

  if ( uiIndex >= static_cast<uint32_t>(GetVectorElementCount(eMaskElementType)) )
  {
    throw InternalErrorException("The index cannot exceed the vector element count!");
  }

  CallExpr        *pMoveMask      = _CreateFunctionCall(IntrinsicsSSEEnum::MoveMaskFloat, pMaskExpr);
  IntegerLiteral  *pTestConstant  = _GetASTHelper().CreateIntegerLiteral<int32_t>(1 << uiIndex);

  return _GetASTHelper().CreateBinaryOperator( pMoveMask, pTestConstant, BO_And, pMoveMask->getType() );
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
  default:                            break;    // Useless default branch avoiding GCC compiler warnings
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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSE2, "SSE2" );
}

Expr* InstructionSetSSE2::_ArithmeticOpInteger(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
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
  case ArithmeticOperatorType::ShiftLeft:
    if (!bIsRHSScalar)
    {
      return _SeparatedArithmeticOpInteger( eElementType, BO_Shl, pExprLHS, pExprRHS );
    }
    else
    {
      switch (eElementType)
      {
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
        {
          // Convert vector to a signed / unsigned 16-bit integer data type, do the shift and convert back
          const VectorElementTypes ceIntermediateType = AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2, AST::BaseClasses::TypeInfo::IsSigned(eElementType)).GetType();

          ClangASTHelper::ExpressionVectorType vecShiftedVectors;

          for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx)
          {
            Expr *pConvertedVector = ConvertVectorUp( eElementType, ceIntermediateType, pExprLHS, uiGroupIdx );
            vecShiftedVectors.push_back( _ArithmeticOpInteger(ceIntermediateType, eOpType, pConvertedVector, pExprRHS, bIsRHSScalar) );
          }

          return ConvertVectorDown( ceIntermediateType, eElementType, vecShiftedVectors );
        }
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftLeftInt16, pExprLHS, pExprRHS );
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftLeftInt32, pExprLHS, pExprRHS );
      default:    return _SeparatedArithmeticOpInteger(eElementType, BO_Shl, pExprLHS, pExprRHS);
      }
    }
  case ArithmeticOperatorType::ShiftRight:
    if (!bIsRHSScalar)
    {
      return _SeparatedArithmeticOpInteger( eElementType, BO_Shr, pExprLHS, pExprRHS );
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
            Expr *pConvertedVector = ConvertVectorUp( eElementType, ceIntermediateType, pExprLHS, uiGroupIdx );
            vecShiftedVectors.push_back( _ArithmeticOpInteger(ceIntermediateType, eOpType, pConvertedVector, pExprRHS, bIsRHSScalar) );
          }

          return ConvertVectorDown( ceIntermediateType, eElementType, vecShiftedVectors );
        }
      case VectorElementTypes::Int16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightArithInt16, pExprLHS, pExprRHS );
      case VectorElementTypes::Int32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightArithInt32, pExprLHS, pExprRHS );
      case VectorElementTypes::UInt16:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightLogInt16, pExprLHS, pExprRHS );
      case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsSSE2Enum::ShiftRightLogInt32, pExprLHS, pExprRHS );
      default:    return _SeparatedArithmeticOpInteger(eElementType, BO_Shr, pExprLHS, pExprRHS);
      }
    }
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

          for (auto itVec : crvecVectorRefs)
          {
            vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt64, itVec) );
          }

          return ConvertMaskDown( VectorElementTypes::UInt64, eTargetType, vecCastedVectors );
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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

          for (auto itVec : crvecVectorRefs)
          {
            vecCastedVectors.push_back( ConvertMaskSameSize(eSourceType, VectorElementTypes::UInt32, itVec) );
          }

          return _ConvertVector( VectorElementTypes::UInt32, eTargetType, vecCastedVectors, uiGroupIndex, bMaskConversion );
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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
              const uint32_t cuiSwapIndex = static_cast<uint32_t>(cszTargetSize / cszSourceSize) >> 1;

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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
      }

    default:  break;    // Useless default branch avoiding GCC compiler warnings
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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

          for (auto itVectorRef : crvecVectorRefs)
          {
            vecConvertedVectors.push_back( ConvertVectorSameSize(eSourceType, VectorElementTypes::Int32, itVectorRef) );
          }

          return ConvertVectorDown(VectorElementTypes::Int32, eTargetType, vecConvertedVectors);
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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
          const uint32_t            cuiSwapIndex        = static_cast<uint32_t>(AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType)) >> 1;

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

          const uint32_t cuiSwapIndex   = static_cast<uint32_t>(AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType)) >> 2;

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;
          vecConvertedVectors.push_back( ConvertVectorUp( eSourceType, eIntermediateType, crvecVectorRefs.front(), uiGroupIndex / cuiSwapIndex ) );

          if (uiGroupIndex >= cuiSwapIndex)
          {
            uiGroupIndex %= cuiSwapIndex;
          }

          return _ConvertVector( eIntermediateType, eTargetType, vecConvertedVectors, uiGroupIndex, bMaskConversion );
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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

          const uint32_t cuiSwapIndex   = static_cast<uint32_t>(AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType)) >> 1;

          ClangASTHelper::ExpressionVectorType vecConvertedVectors;
          vecConvertedVectors.push_back( ConvertVectorUp( eSourceType, eIntermediateType, crvecVectorRefs.front(), uiGroupIndex / cuiSwapIndex ) );

          if (uiGroupIndex >= cuiSwapIndex)
          {
            uiGroupIndex -= cuiSwapIndex;
          }

          return _ConvertVector( eIntermediateType, eTargetType, vecConvertedVectors, uiGroupIndex, bMaskConversion );
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
      }

      break;

    default:  break;    // Useless default branch avoiding GCC compiler warnings
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
    pElement = _GetASTHelper().CreateConditionalOperator( _GetASTHelper().CreateParenthesisExpression(pElement), _GetASTHelper().CreateIntegerLiteral<int64_t>(-1LL),
                                                          _GetASTHelper().CreateIntegerLiteral<int64_t>(0LL), pElement->getType() );

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
    case VectorElementTypes::UInt8:  pSignMask = _GetASTHelper().CreateIntegerLiteral<uint8_t> (0x80U                ); break;
    case VectorElementTypes::UInt16: pSignMask = _GetASTHelper().CreateIntegerLiteral<uint16_t>(0x8000U              ); break;
    case VectorElementTypes::UInt32: pSignMask = _GetASTHelper().CreateIntegerLiteral<uint32_t>(0x80000000U          ); break;
    case VectorElementTypes::UInt64: pSignMask = _GetASTHelper().CreateIntegerLiteral<uint64_t>(0x8000000000000000ULL); break;
    default:                         throw InternalErrorException("Unexpected vector element type detected!");
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

    return _CreateFunctionCall( eShiftID, pVectorRef, _GetASTHelper().CreateIntegerLiteral<int32_t>(uiByteCount) );
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

Expr* InstructionSetSSE2::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
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
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return _ArithmeticOpInteger( eElementType, eOpType, pExprLHS, pExprRHS, bIsRHSScalar );
  default:                                                          return BaseType::ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS, bIsRHSScalar );
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
  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

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
    default:                            break;    // Useless default branch avoiding GCC compiler warnings
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
    default:                            break;    // Useless default branch avoiding GCC compiler warnings
    }

    break;

  case VectorElementTypes::Int16:

    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Ceil:
    case BuiltinFunctionsEnum::Floor:   return crvecArguments.front();  // Nothing to do for this functions
    case BuiltinFunctionsEnum::Max:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MaxInt16, crvecArguments[0], crvecArguments[1] );
    case BuiltinFunctionsEnum::Min:     return _CreateFunctionCall( IntrinsicsSSE2Enum::MinInt16, crvecArguments[0], crvecArguments[1] );
    default:                            break;    // Useless default branch avoiding GCC compiler warnings
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
    default:                            break;    // Useless default branch avoiding GCC compiler warnings
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

Expr* InstructionSetSSE2::CheckSingleMaskElement(VectorElementTypes eMaskElementType, Expr *pMaskExpr, uint32_t uiIndex)
{
  if ( uiIndex >= static_cast<uint32_t>(GetVectorElementCount(eMaskElementType)) )
  {
    throw InternalErrorException("The index cannot exceed the vector element count!");
  }

  IntrinsicsSSE2Enum  eMoveMaskType = IntrinsicsSSE2Enum::MoveMaskInt8;
  int32_t             iTestValue    = 1;
  int32_t             iShiftValue   = 0;

  switch (eMaskElementType)
  {
  case VectorElementTypes::Double:
    {
      eMoveMaskType = IntrinsicsSSE2Enum::MoveMaskDouble;
      iTestValue    = 1;
      iShiftValue   = static_cast<int32_t>(uiIndex);
      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      const uint32_t cuiTypeSize = static_cast<uint32_t>(AST::BaseClasses::TypeInfo::GetTypeSize( eMaskElementType ));

      eMoveMaskType = IntrinsicsSSE2Enum::MoveMaskInt8;
      iTestValue    = static_cast<int32_t>(0xFF % (1 << cuiTypeSize));
      iShiftValue   = static_cast<int32_t>(uiIndex * cuiTypeSize);
      break;
    }
  default:                                                          return BaseType::CheckSingleMaskElement( eMaskElementType, pMaskExpr, uiIndex );
  }

  CallExpr        *pMoveMask      = _CreateFunctionCall(eMoveMaskType, pMaskExpr);
  IntegerLiteral  *pTestConstant  = _GetASTHelper().CreateIntegerLiteral(iTestValue << iShiftValue);

  return _GetASTHelper().CreateBinaryOperator( pMoveMask, pTestConstant, BO_And, pMoveMask->getType() );
}

Expr* InstructionSetSSE2::CreateOnesVector(VectorElementTypes eElementType, bool bNegative)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  return BroadCast( eElementType, _GetASTHelper().CreateLiteral(bNegative ? -1.0 : 1.0) );
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return BroadCast( eElementType, _GetASTHelper().CreateIntegerLiteral(bNegative ? -1 : 1) );
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
      QualType qtReturnType = _GetVectorType( eElementType, _GetASTHelper().IsPointerToConstType(pPointerRef->getType()) );

      CastExpr *pPointerCast = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType(qtReturnType) );

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
  default:  throw RuntimeErrorException( std::string("Only index element types \"") + AST::BaseClasses::TypeInfo::GetTypeString( VectorElementTypes::Int32 ) + ("\" and \"") +
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

  IntegerLiteral *pShiftCount = _GetASTHelper().CreateIntegerLiteral<int32_t>(uiCount);

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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSE3, "SSE3" );
}

Expr* InstructionSetSSE3::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion);
}

void InstructionSetSSE3::_InitIntrinsicsMap()
{
  _InitIntrinsic (IntrinsicsSSE3Enum::LoadInteger, "lddqu_si128" );
}

Expr* InstructionSetSSE3::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
{
  return BaseType::ArithmeticOperator(eElementType, eOpType, pExprLHS, pExprRHS, bIsRHSScalar);
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
      QualType qtReturnType = _GetVectorType( eElementType, _GetASTHelper().IsPointerToConstType(pPointerRef->getType()) );

      CastExpr *pPointerCast = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType(qtReturnType) );

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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSSE3, "SSSE3" );
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
      }

      break;

    default:  break;    // Useless default branch avoiding GCC compiler warnings
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

Expr* InstructionSetSSSE3::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
{
  return BaseType::ArithmeticOperator(eElementType, eOpType, pExprLHS, pExprRHS, bIsRHSScalar);
}

Expr* InstructionSetSSSE3::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  return BaseType::BlendVectors(eElementType, pMaskRef, pVectorTrue, pVectorFalse);
}

Expr* InstructionSetSSSE3::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

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
    default:                          break;    // Useless default branch avoiding GCC compiler warnings
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
    default:                          break;    // Useless default branch avoiding GCC compiler warnings
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
    default:                          break;    // Useless default branch avoiding GCC compiler warnings
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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSE4_1, "SSE4.1" );
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
      default:  break;    // Useless default branch avoiding GCC compiler warnings
      }

    default:  break;    // Useless default branch avoiding GCC compiler warnings
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
      const uint32_t cuiShiftMultiplier = 16 / static_cast<uint32_t>(AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) / AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType));

      return _CreateFunctionCall( eConvertID, _ShiftIntegerVectorBytes(crvecVectorRefs.front(), uiGroupIndex * cuiShiftMultiplier, false) );
    }
  }

  // If the function has not returned earlier, let the base handle the conversion
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion);
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

Expr* InstructionSetSSE4_1::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
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
    return BaseType::ArithmeticOperator( eElementType, eOpType, pExprLHS, pExprRHS, bIsRHSScalar );
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
  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

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
    default:  break;    // Useless default branch avoiding GCC compiler warnings
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
    default:  break;    // Useless default branch avoiding GCC compiler warnings
    }
  }

  // If the function has not returned earlier, call the base implementation
  return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
}

Expr* InstructionSetSSE4_1::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  _CheckExtractIndex( eElementType, uiIndex );

  // Select the correct extraction intrinsic
  IntrinsicsSSE4_1Enum eIntrinID = IntrinsicsSSE4_1Enum::ExtractInt8;

  switch (eElementType)
  {
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE4_1Enum::ExtractInt8;    break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE4_1Enum::ExtractInt32;   break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE4_1Enum::ExtractInt64;   break;
  default:                                                          return BaseType::ExtractElement( eElementType, pVectorRef, uiIndex );
  }

  // Perform the actual extraction
  Expr *pExtractExpr = _CreateFunctionCall( eIntrinID, pVectorRef, _GetASTHelper().CreateIntegerLiteral<int32_t>(uiIndex) );

  QualType qtReturnType = _GetClangType(eElementType);
  if (qtReturnType != pExtractExpr->getType())
  {
    pExtractExpr = _CreateValueCast( pExtractExpr, qtReturnType, CK_IntegralCast );
  }

  return pExtractExpr;
}

Expr* InstructionSetSSE4_1::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  _CheckInsertIndex( eElementType, uiIndex );

  // Select the correct insertion intrinsic
  IntrinsicsSSE4_1Enum eIntrinID = IntrinsicsSSE4_1Enum::InsertFloat;

  switch (eElementType)
  {
  case VectorElementTypes::Float:
    {
      // The insertion intrinsic for the element type "float" has a different syntax
      eIntrinID       = IntrinsicsSSE4_1Enum::InsertFloat;
      pElementValue   = BroadCast(eElementType, pElementValue);
      uiIndex       <<= 4;
      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsSSE4_1Enum::InsertInt8;   break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsSSE4_1Enum::InsertInt32;  break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsSSE4_1Enum::InsertInt64;  break;
  default:                                                          return BaseType::InsertElement( eElementType, pVectorRef, pElementValue, uiIndex );
  }

  // Perform the actual insertion
  return _CreateFunctionCall( eIntrinID, pVectorRef, pElementValue, _GetASTHelper().CreateIntegerLiteral<int32_t>(uiIndex) );
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
    default:  break;    // Useless default branch avoiding GCC compiler warnings
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

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsSSE4_2, "SSE4.2" );
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




// Implementation of class InstructionSetAVX
InstructionSetAVX::InstructionSetAVX(ASTContext &rAstContext) : InstructionSetBase(rAstContext, _GetIntrinsicPrefix())
{
  _spFallbackInstructionSet = InstructionSetBase::Create<InstructionSetSSE4_2>(rAstContext);

  _InitIntrinsicsMap();

  _CreateMissingIntrinsicsAVX();

  InstructionSetBase::_LookupIntrinsics( _mapIntrinsicsAVX, "AVX" );
}

Expr* InstructionSetAVX::_ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  if (bMaskConversion)
  {
    switch (eSourceType)
    {
    case VectorElementTypes::Double:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:  return crvecVectorRefs.front();   // Same type => nothing to do
        case VectorElementTypes::Float:
          {
            ClangASTHelper::ExpressionVectorType vecMergedLanes;

            for (auto itSourceVec : crvecVectorRefs)
            {
              Expr *pLaneSwapConstant = _GetASTHelper().CreateLiteral<int32_t>(0x01);
              Expr *pShuffleContant   = _GetASTHelper().CreateLiteral<int32_t>(0x88);

              Expr *pCastedVec   = _CreateFunctionCall( IntrinsicsAVXEnum::CastDoubleToFloat, itSourceVec );
              Expr *pSwappedLane = _CreateFunctionCall( IntrinsicsAVXEnum::PermuteLanesFloat, pCastedVec, CreateZeroVector(eTargetType), pLaneSwapConstant );

              vecMergedLanes.push_back( _CreateFunctionCall( IntrinsicsAVXEnum::ShuffleFloat, pCastedVec, pSwappedLane, pShuffleContant ) );
            }

            return _CreateFunctionCall( IntrinsicsAVXEnum::PermuteLanesFloat, vecMergedLanes[0], vecMergedLanes[1], _GetASTHelper().CreateLiteral<int32_t>(0x20) );
          }
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
          {
            Expr *pFloatMask = ConvertMaskDown( eSourceType, VectorElementTypes::Float, crvecVectorRefs );

            return ConvertMaskSameSize( VectorElementTypes::Float, eTargetType, pFloatMask );
          }
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Float:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:
          {
            Expr *pLaneSelectConstant = _GetASTHelper().CreateLiteral<int32_t>((uiGroupIndex == 0) ? 0x00 : 0x11);
            Expr *pSelectedLane       = _CreateFunctionCall( IntrinsicsAVXEnum::PermuteLanesFloat, crvecVectorRefs.front(), CreateZeroVector(eSourceType), pLaneSelectConstant );

            Expr *pElementsEven = _CreateFunctionCall( IntrinsicsAVXEnum::CastFloatToDouble, _CreateFunctionCall(IntrinsicsAVXEnum::DuplicateEvenFloat, pSelectedLane) );
            Expr *pElementsOdd  = _CreateFunctionCall( IntrinsicsAVXEnum::CastFloatToDouble, _CreateFunctionCall(IntrinsicsAVXEnum::DuplicateOddFloat,  pSelectedLane) );

            return _CreateFunctionCall( IntrinsicsAVXEnum::ShuffleDouble, pElementsEven, pElementsOdd, _GetASTHelper().CreateLiteral<int32_t>(0x0C) );
        }
        case VectorElementTypes::Float:                                   return crvecVectorRefs.front();   // Same type => nothing to do
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsAVXEnum::CastFloatToInteger, crvecVectorRefs.front() );
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:
          {
            Expr *pFloatMask = ConvertMaskSameSize( eSourceType, VectorElementTypes::Float, crvecVectorRefs.front() );

            return ConvertMaskUp( VectorElementTypes::Float, eTargetType, pFloatMask, uiGroupIndex );
          }
        case VectorElementTypes::Float:                                   return _CreateFunctionCall( IntrinsicsAVXEnum::CastIntegerToFloat, crvecVectorRefs.front() );
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return crvecVectorRefs.front();   // Same type => nothing to do
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Int8:   case VectorElementTypes::UInt8: case VectorElementTypes::Int16:
    case VectorElementTypes::UInt16: case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
      {
        switch (eTargetType)
        {
          case VectorElementTypes::Int8:   case VectorElementTypes::UInt8: case VectorElementTypes::Int16:
          case VectorElementTypes::UInt16: case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
            if ( AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) == AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) )
            {
              return crvecVectorRefs.front();   // There is no difference between signed and unsigned integer vector types
            }
          default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
       }
    default:  break;    // Useless default branch avoiding GCC compiler warnings
    }
  }
  else
  {
    switch (eSourceType)
    {
    case VectorElementTypes::Double:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:    return crvecVectorRefs.front();   // Same type => nothing to do
        case VectorElementTypes::Float:
          {
            Expr *pLowHalf  = _CreateFunctionCall( IntrinsicsAVXEnum::ConvertDoubleFloat, crvecVectorRefs[0] );
            Expr *pHighHalf = _CreateFunctionCall( IntrinsicsAVXEnum::ConvertDoubleFloat, crvecVectorRefs[1] );

            return _MergeSSEVectors( eTargetType, pLowHalf, pHighHalf );
          }
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
          {
            Expr *pLowHalf  = _CreateFunctionCall( IntrinsicsAVXEnum::ConvertDoubleInt32, crvecVectorRefs[0] );
            Expr *pHighHalf = _CreateFunctionCall( IntrinsicsAVXEnum::ConvertDoubleInt32, crvecVectorRefs[1] );

            return _MergeSSEVectors( eTargetType, pLowHalf, pHighHalf );
          }
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Float:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:
          {
            Expr *pSelectedGroup = _ExtractSSEVector( eSourceType, crvecVectorRefs.front(), (uiGroupIndex == 0) );

            return _CreateFunctionCall( IntrinsicsAVXEnum::ConvertFloatDouble, pSelectedGroup );
          }
        case VectorElementTypes::Float:                                   return crvecVectorRefs.front();   // Same type => nothing to do
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return _CreateFunctionCall( IntrinsicsAVXEnum::ConvertFloatInt32, crvecVectorRefs.front() );
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Int32:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Double:
          {
            Expr *pSelectedGroup = _ExtractSSEVector( eSourceType, crvecVectorRefs.front(), (uiGroupIndex == 0) );

            return _CreateFunctionCall( IntrinsicsAVXEnum::ConvertInt32Double, pSelectedGroup );
          }
        case VectorElementTypes::Float:                                   return _CreateFunctionCall( IntrinsicsAVXEnum::ConvertInt32Float, crvecVectorRefs.front() );
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  return crvecVectorRefs.front();   // Same type => nothing to do
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
      }
    case VectorElementTypes::Int8:   case VectorElementTypes::UInt8: case VectorElementTypes::Int16:  case VectorElementTypes::UInt16:
    case VectorElementTypes::UInt32: case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
      {
        switch (eTargetType)
        {
        case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
        case VectorElementTypes::Int32: case VectorElementTypes::UInt32: case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
          if ( AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) == AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) )
          {
            return crvecVectorRefs.front();   // There is no difference between signed and unsigned integer vector types
          }
        default:  break;    // Useless default branch avoiding GCC compiler warnings
        }

        break;
       }
    default:  break;    // Useless default branch avoiding GCC compiler warnings
    }
  }

  // Use SSE fallback for unsupported conversions
  return _ConvertVectorWithSSE( eSourceType, eTargetType, crvecVectorRefs, uiGroupIndex, bMaskConversion );
}

Expr* InstructionSetAVX::_ConvertVectorWithSSE(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, uint32_t uiGroupIndex, bool bMaskConversion)
{
  const size_t cszElementCountSource = GetVectorElementCount( eSourceType );
  const size_t cszElementCountTarget = GetVectorElementCount( eTargetType );

  Expr *pResultLow  = nullptr;
  Expr *pResultHigh = nullptr;

  if (cszElementCountSource == cszElementCountTarget)
  {
    // Same size conversion
    Expr *pVectorAVX = crvecVectorRefs.front();

    // Both SSE vector lane can be converted separately
    if (bMaskConversion)
    {
      pResultLow  = _GetFallback()->ConvertMaskSameSize( eSourceType, eTargetType, _ExtractSSEVector( eSourceType, pVectorAVX, true  ) );
      pResultHigh = _GetFallback()->ConvertMaskSameSize( eSourceType, eTargetType, _ExtractSSEVector( eSourceType, pVectorAVX, false ) );
    }
    else
    {
      pResultLow  = _GetFallback()->ConvertVectorSameSize( eSourceType, eTargetType, _ExtractSSEVector( eSourceType, pVectorAVX, true  ) );
      pResultHigh = _GetFallback()->ConvertVectorSameSize( eSourceType, eTargetType, _ExtractSSEVector( eSourceType, pVectorAVX, false ) );
    }
  }
  else if (cszElementCountSource < cszElementCountTarget)
  {
    // Downward conversion
    ClangASTHelper::ExpressionVectorType vecVectorRefsLow, vecVectorRefsHigh;
    {
      // Extract all SSE vectors from the source AVX vectors
      ClangASTHelper::ExpressionVectorType vecVectorsSSE;
      for (auto itVectorAVX : crvecVectorRefs)
      {
        vecVectorsSSE.push_back( _ExtractSSEVector( eSourceType, itVectorAVX, true  ) );
        vecVectorsSSE.push_back( _ExtractSSEVector( eSourceType, itVectorAVX, false ) );
      }

      // Split the SSE vectors into the low and high element-halfs
      const size_t cszSourceVectorCount = crvecVectorRefs.size();
      vecVectorRefsLow.insert ( vecVectorRefsLow.end(),  vecVectorsSSE.begin(),                        vecVectorsSSE.begin() + cszSourceVectorCount );
      vecVectorRefsHigh.insert( vecVectorRefsHigh.end(), vecVectorsSSE.begin() + cszSourceVectorCount, vecVectorsSSE.end()                          );
    }

    // Perform the downward conversions for both element-halfs
    if (bMaskConversion)
    {
      pResultLow  = _GetFallback()->ConvertMaskDown( eSourceType, eTargetType, vecVectorRefsLow  );
      pResultHigh = _GetFallback()->ConvertMaskDown( eSourceType, eTargetType, vecVectorRefsHigh );
    }
    else
    {
      pResultLow  = _GetFallback()->ConvertVectorDown( eSourceType, eTargetType, vecVectorRefsLow  );
      pResultHigh = _GetFallback()->ConvertVectorDown( eSourceType, eTargetType, vecVectorRefsHigh );
    }
  }
  else
  {
    // Upward conversion
    const uint32_t cuiSwapIndex     = static_cast<uint32_t>((cszElementCountSource / cszElementCountTarget) >> 1);
    const uint32_t cuiGroupIndexSSE = (uiGroupIndex % cuiSwapIndex) << 1;

    // Select the SSE vector lane which contains the requested group
    Expr *pVectorSSE = _ExtractSSEVector( eSourceType, crvecVectorRefs.front(), (uiGroupIndex < cuiSwapIndex) );

    // Convert two adjacent groups of the requested SSE lane
    if (bMaskConversion)
    {
      pResultLow  = _GetFallback()->ConvertMaskUp( eSourceType, eTargetType, pVectorSSE, cuiGroupIndexSSE );
      pResultHigh = _GetFallback()->ConvertMaskUp( eSourceType, eTargetType, pVectorSSE, cuiGroupIndexSSE + 1 );
    }
    else
    {
      pResultLow  = _GetFallback()->ConvertVectorUp( eSourceType, eTargetType, pVectorSSE, cuiGroupIndexSSE );
      pResultHigh = _GetFallback()->ConvertVectorUp( eSourceType, eTargetType, pVectorSSE, cuiGroupIndexSSE + 1 );
    }
  }

  // Merge both converted SSE vectors back into the final AVX vector
  return _MergeSSEVectors( eTargetType, pResultLow, pResultHigh );
}

void InstructionSetAVX::_InitIntrinsicsMap()
{
  // Addition functions
  _InitIntrinsic( IntrinsicsAVXEnum::AddDouble, "add_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::AddFloat,  "add_ps" );

  // Bitwise "and" functions
  _InitIntrinsic( IntrinsicsAVXEnum::AndDouble, "and_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::AndFloat,  "and_ps" );

  // Blending functions
  _InitIntrinsic( IntrinsicsAVXEnum::BlendDouble, "blendv_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::BlendFloat,  "blendv_ps" );

  // Broadcast functions
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastDouble, "set1_pd"     );
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastFloat,  "set1_ps"     );
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastInt8,   "set1_epi8"   );
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastInt16,  "set1_epi16"  );
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastInt32,  "set1_epi32"  );
  _InitIntrinsic( IntrinsicsAVXEnum::BroadCastInt64,  "set1_epi64x" );

  // Vector cast functions (change bit-representation, no conversion)
  _InitIntrinsic( IntrinsicsAVXEnum::CastDoubleToFloat,   "castpd_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::CastDoubleToInteger, "castpd_si256" );
  _InitIntrinsic( IntrinsicsAVXEnum::CastFloatToDouble,   "castps_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::CastFloatToInteger,  "castps_si256" );
  _InitIntrinsic( IntrinsicsAVXEnum::CastIntegerToDouble, "castsi256_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::CastIntegerToFloat,  "castsi256_ps" );

  // Rounding functions
  _InitIntrinsic( IntrinsicsAVXEnum::CeilDouble,  "ceil_pd"  );
  _InitIntrinsic( IntrinsicsAVXEnum::CeilFloat,   "ceil_ps"  );
  _InitIntrinsic( IntrinsicsAVXEnum::FloorDouble, "floor_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::FloorFloat,  "floor_ps" );

  // Comparison functions
  _InitIntrinsic( IntrinsicsAVXEnum::CompareDouble, "cmp_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::CompareFloat,  "cmp_ps" );

  // Convert functions
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertDoubleFloat, "cvtpd_ps"     );
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertDoubleInt32, "cvttpd_epi32" );    // Use truncation
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertFloatDouble, "cvtps_pd"     );
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertFloatInt32,  "cvttps_epi32" );    // Use truncation
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertInt32Double, "cvtepi32_pd"  );
  _InitIntrinsic( IntrinsicsAVXEnum::ConvertInt32Float,  "cvtepi32_ps"  );

  // Division functions
  _InitIntrinsic( IntrinsicsAVXEnum::DivideDouble, "div_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::DivideFloat,  "div_ps" );

  // Duplication functions
  _InitIntrinsic( IntrinsicsAVXEnum::DuplicateEvenFloat, "moveldup_ps" );
  _InitIntrinsic( IntrinsicsAVXEnum::DuplicateOddFloat,  "movehdup_ps" );

  // Integer extraction functions
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractInt8,  "extract_epi8"  );
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractInt16, "extract_epi16" );
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractInt32, "extract_epi32" );
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractInt64, "extract_epi64" );

  // Extract SSE vectors functions
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractSSEDouble,  "extractf128_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractSSEFloat,   "extractf128_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::ExtractSSEInteger, "extractf128_si256" );

  // Insert SSE vectors functions
  _InitIntrinsic( IntrinsicsAVXEnum::InsertSSEDouble,  "insertf128_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::InsertSSEFloat,   "insertf128_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::InsertSSEInteger, "insertf128_si256" );

  // Load functions
  _InitIntrinsic( IntrinsicsAVXEnum::LoadDouble,  "loadu_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::LoadFloat,   "loadu_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::LoadInteger, "lddqu_si256" );

  // Maximum / Minimum functions
  _InitIntrinsic( IntrinsicsAVXEnum::MaxDouble, "max_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::MaxFloat,  "max_ps" );
  _InitIntrinsic( IntrinsicsAVXEnum::MinDouble, "min_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::MinFloat,  "min_ps" );

  // Merge SSE vectors functions
  _InitIntrinsic( IntrinsicsAVXEnum::MergeDouble,  "set_m128d" );
  _InitIntrinsic( IntrinsicsAVXEnum::MergeFloat,   "set_m128"  );
  _InitIntrinsic( IntrinsicsAVXEnum::MergeInteger, "set_m128i" );

  // Mask conversion functions
  _InitIntrinsic( IntrinsicsAVXEnum::MoveMaskDouble, "movemask_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::MoveMaskFloat,  "movemask_ps" );

  // Multiplication functions
  _InitIntrinsic( IntrinsicsAVXEnum::MultiplyDouble, "mul_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::MultiplyFloat,  "mul_ps" );

  // Bitwise "or" functions
  _InitIntrinsic( IntrinsicsAVXEnum::OrDouble, "or_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::OrFloat,  "or_ps" );

  // Permute functions
  _InitIntrinsic( IntrinsicsAVXEnum::PermuteLanesFloat, "permute2f128_ps" );

  // Set methods
  _InitIntrinsic( IntrinsicsAVXEnum::SetDouble, "set_pd"     );
  _InitIntrinsic( IntrinsicsAVXEnum::SetFloat,  "set_ps"     );
  _InitIntrinsic( IntrinsicsAVXEnum::SetInt8,   "set_epi8"   );
  _InitIntrinsic( IntrinsicsAVXEnum::SetInt16,  "set_epi16"  );
  _InitIntrinsic( IntrinsicsAVXEnum::SetInt32,  "set_epi32"  );
  _InitIntrinsic( IntrinsicsAVXEnum::SetInt64,  "set_epi64x" );

  // Zero vector creation functions
  _InitIntrinsic( IntrinsicsAVXEnum::SetZeroDouble,  "setzero_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::SetZeroFloat,   "setzero_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::SetZeroInteger, "setzero_si256" );

  // Shuffle functions
  _InitIntrinsic( IntrinsicsAVXEnum::ShuffleDouble, "shuffle_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::ShuffleFloat,  "shuffle_ps" );

  // Store functions
  _InitIntrinsic( IntrinsicsAVXEnum::StoreDouble,  "storeu_pd"    );
  _InitIntrinsic( IntrinsicsAVXEnum::StoreFloat,   "storeu_ps"    );
  _InitIntrinsic( IntrinsicsAVXEnum::StoreInteger, "storeu_si256" );

  // Square root functions
  _InitIntrinsic( IntrinsicsAVXEnum::SqrtDouble, "sqrt_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::SqrtFloat,  "sqrt_ps" );

  // Subtraction functions
  _InitIntrinsic( IntrinsicsAVXEnum::SubtractDouble, "sub_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::SubtractFloat,  "sub_ps" );
  
  // Bitwise "xor" functions
  _InitIntrinsic( IntrinsicsAVXEnum::XorDouble, "xor_pd" );
  _InitIntrinsic( IntrinsicsAVXEnum::XorFloat,  "xor_ps" );
}

Expr* InstructionSetAVX::_CastVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, Expr *pVectorRef)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::CastDoubleToFloat;

  switch (eSourceType)
  {
  case VectorElementTypes::Double:
    {
      switch (eTargetType)
      {
      case VectorElementTypes::Double:                                  return pVectorRef;
      case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::CastDoubleToFloat;     break;
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::CastDoubleToInteger;   break;
      default:                                                          _ThrowUnsupportedType(eSourceType);
      }

      break;
    }
  case VectorElementTypes::Float:
    {
      switch (eTargetType)
      {
      case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::CastFloatToDouble;   break;
      case VectorElementTypes::Float:                                   return pVectorRef;
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::CastFloatToInteger;  break;
      default:                                                          _ThrowUnsupportedType(eSourceType);
      }

      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      switch (eTargetType)
      {
      case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::CastIntegerToDouble;   break;
      case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::CastIntegerToFloat;    break;
      case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  return pVectorRef;
      default:                                                          _ThrowUnsupportedType(eSourceType);
      }

      break;
    }
  default:  _ThrowUnsupportedType( eSourceType );
  }

  return _CreateFunctionCall( eFunctionID, pVectorRef );
}

Expr* InstructionSetAVX::_CreateFullBitMask(VectorElementTypes eElementType)
{
  return _CastVector( VectorElementTypes::Int32, eElementType, CreateOnesVector( VectorElementTypes::Int32, true ) );
}

Expr* InstructionSetAVX::_CreatePrePostFixedUnaryOp(VectorElementTypes eElementType, Expr *pVectorRef, bool bPrefixed, bool bIncrement)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::AddDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:  eFunctionID = bIncrement ? IntrinsicsAVXEnum::AddDouble : IntrinsicsAVXEnum::SubtractDouble;  break;
  case VectorElementTypes::Float:   eFunctionID = bIncrement ? IntrinsicsAVXEnum::AddFloat  : IntrinsicsAVXEnum::SubtractFloat;   break;
  default:
    {
      const ArithmeticOperatorType ceOpCode = bIncrement ? ArithmeticOperatorType::Add : ArithmeticOperatorType::Subtract;

      Expr *pReturnExpr = ArithmeticOperator( eElementType, ceOpCode, pVectorRef, CreateOnesVector( eElementType, false ) );
      pReturnExpr       = _GetASTHelper().CreateBinaryOperator( pVectorRef, pReturnExpr, BO_Assign, pVectorRef->getType() );

      if (! bPrefixed)
      {
        Expr *pRevertExpr = ArithmeticOperator( eElementType, ceOpCode, pVectorRef, CreateOnesVector( eElementType, true ) );
        pReturnExpr       = _GetASTHelper().CreateBinaryOperatorComma( _GetASTHelper().CreateParenthesisExpression( pReturnExpr ), pRevertExpr );
      }

      return _GetASTHelper().CreateParenthesisExpression( pReturnExpr );
    }
  }

  if (bPrefixed)
  {
    return _CreatePrefixedUnaryOp( _mapIntrinsicsAVX, eFunctionID, eElementType, pVectorRef );
  }
  else
  {
    return _CreatePostfixedUnaryOp( _mapIntrinsicsAVX, eFunctionID, eElementType, pVectorRef );
  }
}

Expr* InstructionSetAVX::_ExtractSSEVector(VectorElementTypes eElementType, Expr *pAVXVector, bool bLowHalf)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::ExtractSSEDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::ExtractSSEDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::ExtractSSEFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::ExtractSSEInteger;  break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pAVXVector, _GetASTHelper().CreateIntegerLiteral(bLowHalf ? 0 : 1) );
}

Expr* InstructionSetAVX::_InsertSSEVector(VectorElementTypes eElementType, Expr *pAVXVector, Expr *pSSEVector, bool bLowHalf)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::InsertSSEDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::InsertSSEDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::InsertSSEFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::InsertSSEInteger;  break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pAVXVector, pSSEVector, _GetASTHelper().CreateIntegerLiteral(bLowHalf ? 0 : 1) );
}

Expr* InstructionSetAVX::_MergeSSEVectors(VectorElementTypes eElementType, Expr *pSSEVectorLow, Expr *pSSEVectorHigh)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::MergeDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::MergeDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::MergeFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::MergeInteger;  break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pSSEVectorHigh, pSSEVectorLow );
}

Expr* InstructionSetAVX::ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::AddDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:
    {
      switch (eOpType)
      {
      case ArithmeticOperatorType::Add:           eFunctionID = IntrinsicsAVXEnum::AddDouble;       break;
      case ArithmeticOperatorType::BitwiseAnd:    eFunctionID = IntrinsicsAVXEnum::AndDouble;       break;
      case ArithmeticOperatorType::BitwiseOr:     eFunctionID = IntrinsicsAVXEnum::OrDouble;        break;
      case ArithmeticOperatorType::BitwiseXOr:    eFunctionID = IntrinsicsAVXEnum::XorDouble;       break;
      case ArithmeticOperatorType::Divide:        eFunctionID = IntrinsicsAVXEnum::DivideDouble;    break;
      case ArithmeticOperatorType::Multiply:      eFunctionID = IntrinsicsAVXEnum::MultiplyDouble;  break;
      case ArithmeticOperatorType::Subtract:      eFunctionID = IntrinsicsAVXEnum::SubtractDouble;  break;
      case ArithmeticOperatorType::Modulo:        throw RuntimeErrorException("Modulo operation is undefined for floating point data types!");
      case ArithmeticOperatorType::ShiftLeft:
      case ArithmeticOperatorType::ShiftRight:    throw RuntimeErrorException("Shift operations are undefined for floating point data types!");
      default:                                    throw InternalErrorException("Unsupported arithmetic operation detected!");
      }

      break;
    }
  case VectorElementTypes::Float:
    {
      switch (eOpType)
      {
      case ArithmeticOperatorType::Add:           eFunctionID = IntrinsicsAVXEnum::AddFloat;        break;
      case ArithmeticOperatorType::BitwiseAnd:    eFunctionID = IntrinsicsAVXEnum::AndFloat;        break;
      case ArithmeticOperatorType::BitwiseOr:     eFunctionID = IntrinsicsAVXEnum::OrFloat;         break;
      case ArithmeticOperatorType::BitwiseXOr:    eFunctionID = IntrinsicsAVXEnum::XorFloat;        break;
      case ArithmeticOperatorType::Divide:        eFunctionID = IntrinsicsAVXEnum::DivideFloat;     break;
      case ArithmeticOperatorType::Multiply:      eFunctionID = IntrinsicsAVXEnum::MultiplyFloat;   break;
      case ArithmeticOperatorType::Subtract:      eFunctionID = IntrinsicsAVXEnum::SubtractFloat;   break;
      case ArithmeticOperatorType::Modulo:        throw RuntimeErrorException("Modulo operation is undefined for floating point data types!");
      case ArithmeticOperatorType::ShiftLeft:
      case ArithmeticOperatorType::ShiftRight:    throw RuntimeErrorException("Shift operations are undefined for floating point data types!");
      default:                                    throw InternalErrorException("Unsupported arithmetic operation detected!");
      }

      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      // Only bit-wise operation can be sped up => All other operations fall through to the default case
      switch (eOpType)
      {
      case ArithmeticOperatorType::BitwiseAnd:
      case ArithmeticOperatorType::BitwiseOr:
      case ArithmeticOperatorType::BitwiseXOr:
        {
          Expr *pCastedLHS = _CastVector( eElementType, VectorElementTypes::Float, pExprLHS );
          Expr *pCastedRHS = _CastVector( eElementType, VectorElementTypes::Float, pExprRHS );

          return _CastVector( VectorElementTypes::Float, eElementType, ArithmeticOperator( VectorElementTypes::Float, eOpType, pCastedLHS, pCastedRHS ) );
        }
      default:  break;    // Useless default branch avoiding GCC compiler warnings
      }
    }
  default:
    {
      ClangASTHelper::ExpressionVectorType  vecSSEArithOps;

      for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
      {
        const bool cbLowHalf = (uiIdx == 0);

        Expr *pExprLHS_SSE = _ExtractSSEVector( eElementType, pExprLHS, cbLowHalf );
        Expr *pExprRHS_SSE = nullptr;

        if (bIsRHSScalar && (eOpType == ArithmeticOperatorType::ShiftLeft ||
                             eOpType == ArithmeticOperatorType::ShiftRight))
        {
          pExprRHS_SSE = pExprRHS;
        }
        else
        {
          pExprRHS_SSE = _ExtractSSEVector( eElementType, pExprRHS, cbLowHalf );
        }

        vecSSEArithOps.push_back( _GetFallback()->ArithmeticOperator( eElementType, eOpType, pExprLHS_SSE, pExprRHS_SSE, bIsRHSScalar ) );
      }

      return _MergeSSEVectors( eElementType, vecSSEArithOps[0], vecSSEArithOps[1] );
    }
  }

  return _CreateFunctionCall( eFunctionID, pExprLHS, pExprRHS );
}

Expr* InstructionSetAVX::BlendVectors(VectorElementTypes eElementType, Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse)
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:    return _CreateFunctionCall( IntrinsicsAVXEnum::BlendDouble, pVectorFalse, pVectorTrue, pMaskRef );
  case VectorElementTypes::Float:     return _CreateFunctionCall( IntrinsicsAVXEnum::BlendFloat,  pVectorFalse, pVectorTrue, pMaskRef );
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    {
      // Exploit blending routine for "float" element type
      const VectorElementTypes ceIntermediateType = VectorElementTypes::Float;

      Expr *pMaskRefCasted      = _CastVector( eElementType, ceIntermediateType, pMaskRef     );
      Expr *pVectorTrueCasted   = _CastVector( eElementType, ceIntermediateType, pVectorTrue  );
      Expr *pVectorFalseCasted  = _CastVector( eElementType, ceIntermediateType, pVectorFalse );

      return _CastVector( ceIntermediateType, eElementType, BlendVectors( ceIntermediateType, pMaskRefCasted, pVectorTrueCasted, pVectorFalseCasted ) );
    }
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      // Exploit blending routine for "double" element type
      const VectorElementTypes ceIntermediateType = VectorElementTypes::Double;

      Expr *pMaskRefCasted      = _CastVector( eElementType, ceIntermediateType, pMaskRef     );
      Expr *pVectorTrueCasted   = _CastVector( eElementType, ceIntermediateType, pVectorTrue  );
      Expr *pVectorFalseCasted  = _CastVector( eElementType, ceIntermediateType, pVectorFalse );

      return _CastVector( ceIntermediateType, eElementType, BlendVectors( ceIntermediateType, pMaskRefCasted, pVectorTrueCasted, pVectorFalseCasted ) );
    }
  default:
    {
      ClangASTHelper::ExpressionVectorType  vecSSEBlends;

      for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
      {
        const bool cbLowHalf = (uiIdx == 0);

        Expr *pMaskRefSSE     = _ExtractSSEVector( eElementType, pMaskRef,     cbLowHalf );
        Expr *pVectorTrueSSE  = _ExtractSSEVector( eElementType, pVectorTrue,  cbLowHalf );
        Expr *pVectorFalseSSE = _ExtractSSEVector( eElementType, pVectorFalse, cbLowHalf );

        vecSSEBlends.push_back( _GetFallback()->BlendVectors( eElementType, pMaskRefSSE, pVectorTrueSSE, pVectorFalseSSE ) );
      }

      return _MergeSSEVectors( eElementType, vecSSEBlends[0], vecSSEBlends[1] );
    }
  }
}

Expr* InstructionSetAVX::BroadCast(VectorElementTypes eElementType, Expr *pBroadCastValue)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::BroadCastDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::BroadCastDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::BroadCastFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eFunctionID = IntrinsicsAVXEnum::BroadCastInt8;     break;
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eFunctionID = IntrinsicsAVXEnum::BroadCastInt16;    break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eFunctionID = IntrinsicsAVXEnum::BroadCastInt32;    break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::BroadCastInt64;    break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pBroadCastValue );
}

Expr* InstructionSetAVX::BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments)
{
  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

  if (! IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount))
  {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType, eFunctionType, cuiParamCount, "AVX");
  }


  switch (eElementType)
  {
  case VectorElementTypes::Double: case VectorElementTypes::Float:
    {
      const bool cbIsDouble = (eElementType == VectorElementTypes::Double);

      if (cuiParamCount == 2)
      {
        switch (eFunctionType)
        {
        case BuiltinFunctionsEnum::Max:   return _CreateFunctionCall( cbIsDouble ? IntrinsicsAVXEnum::MaxDouble : IntrinsicsAVXEnum::MaxFloat, crvecArguments[0], crvecArguments[1] );
        case BuiltinFunctionsEnum::Min:   return _CreateFunctionCall( cbIsDouble ? IntrinsicsAVXEnum::MinDouble : IntrinsicsAVXEnum::MinFloat, crvecArguments[0], crvecArguments[1] );
        default:                          break;    // Useless default branch avoiding GCC compiler warnings
        }
      }
      else if (cuiParamCount == 1)
      {
        switch (eFunctionType)
        {
        case BuiltinFunctionsEnum::Abs:
          {
            Expr *pSubZeroMask  = RelationalOperator( eElementType, RelationalOperatorType::Less, crvecArguments.front(), CreateZeroVector(eElementType) );
            Expr *pMultiplier   = BlendVectors( eElementType, pSubZeroMask, CreateOnesVector(eElementType, true), CreateOnesVector(eElementType, false) );

            return ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, crvecArguments.front(), pMultiplier );
          }
        case BuiltinFunctionsEnum::Ceil:    return _CreateFunctionCall( cbIsDouble ? IntrinsicsAVXEnum::CeilDouble  : IntrinsicsAVXEnum::CeilFloat,  crvecArguments.front() );
        case BuiltinFunctionsEnum::Floor:   return _CreateFunctionCall( cbIsDouble ? IntrinsicsAVXEnum::FloorDouble : IntrinsicsAVXEnum::FloorFloat, crvecArguments.front() );
        case BuiltinFunctionsEnum::Sqrt:    return _CreateFunctionCall( cbIsDouble ? IntrinsicsAVXEnum::SqrtDouble  : IntrinsicsAVXEnum::SqrtFloat,  crvecArguments.front() );
        default:                            break;    // Useless default branch avoiding GCC compiler warnings
        }
      }

      break;
    }
  case VectorElementTypes::Int8:   case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16:  case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32:  case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64:  case VectorElementTypes::UInt64:
    {
      switch (eFunctionType)
      {
      case BuiltinFunctionsEnum::Abs:
        {
          if (! AST::BaseClasses::TypeInfo::IsSigned( eElementType ))
          {
            return crvecArguments.front();
          }

          break;
        }
      case BuiltinFunctionsEnum::Ceil:    return crvecArguments.front();
      case BuiltinFunctionsEnum::Floor:   return crvecArguments.front();
      default:                            break;    // Useless default branch avoiding GCC compiler warnings
      }
      
      break;
    }
  default:  break;    // Useless default branch avoiding GCC compiler warnings
  }


  // If the function has not returned earlier, use the SSE fallback
  ClangASTHelper::ExpressionVectorType  vecSSEBuiltinFuncs;

  for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
  {
    ClangASTHelper::ExpressionVectorType vecSSEArgs;

    for (auto itArg : crvecArguments)
    {
      vecSSEArgs.push_back( _ExtractSSEVector( eElementType, itArg, (uiIdx == 0) ) );
    }

    vecSSEBuiltinFuncs.push_back( _GetFallback()->BuiltinFunction( eElementType, eFunctionType, vecSSEArgs ) );
  }

  return _MergeSSEVectors( eElementType, vecSSEBuiltinFuncs[0], vecSSEBuiltinFuncs[1] );
}

Expr* InstructionSetAVX::CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, Expr *pMaskExpr)
{
  const QualType cqtBool = _GetClangType( VectorElementTypes::Bool );

  switch (eMaskElementType)
  {
  case VectorElementTypes::Double: case VectorElementTypes::Float:
    {
      int32_t iTestConstant = (eMaskElementType == VectorElementTypes::Double)    ? 0x0F          : 0xFF;
      iTestConstant         = (eCheckType       == ActiveElementsCheckType::All)  ? iTestConstant :    0;

      const IntrinsicsAVXEnum   ceFunctionID  = (eMaskElementType == VectorElementTypes::Double)    ? IntrinsicsAVXEnum::MoveMaskDouble : IntrinsicsAVXEnum::MoveMaskFloat;
      const BinaryOperatorKind  ceCompareOp   = (eCheckType       == ActiveElementsCheckType::Any)  ? BO_NE : BO_EQ;

      return _GetASTHelper().CreateBinaryOperator( _CreateFunctionCall( ceFunctionID, pMaskExpr ), _GetASTHelper().CreateLiteral(iTestConstant), ceCompareOp, cqtBool );
    }
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
    {
      // Exploit check routine for "float" element type
      const VectorElementTypes ceIntermediateType = VectorElementTypes::Float;

      return CheckActiveElements( ceIntermediateType, eCheckType, _CastVector( eMaskElementType, ceIntermediateType, pMaskExpr ) );
    }
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      // Exploit check routine for "double" element type
      const VectorElementTypes ceIntermediateType = VectorElementTypes::Double;

      return CheckActiveElements( ceIntermediateType, eCheckType, _CastVector( eMaskElementType, ceIntermediateType, pMaskExpr ) );
    }
  default:
    {
      ClangASTHelper::ExpressionVectorType  vecSSEChecks;

      for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
      {
        Expr *pMaskRefSSE = _ExtractSSEVector( eMaskElementType, pMaskExpr, (uiIdx == 0) );

        vecSSEChecks.push_back( _GetASTHelper().CreateParenthesisExpression( _GetFallback()->CheckActiveElements(eMaskElementType, eCheckType, pMaskRefSSE) ) );
      }

      return _GetASTHelper().CreateBinaryOperator( vecSSEChecks[0], vecSSEChecks[1], ((eCheckType == ActiveElementsCheckType::Any) ? BO_LOr : BO_LAnd), cqtBool );
    }
  }
}

Expr* InstructionSetAVX::CheckSingleMaskElement(VectorElementTypes eMaskElementType, Expr *pMaskExpr, uint32_t uiIndex)
{
  if ( uiIndex >= static_cast<uint32_t>(GetVectorElementCount(eMaskElementType)) )
  {
    throw InternalErrorException("The index cannot exceed the vector element count!");
  }

  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::MoveMaskDouble;

  switch (eMaskElementType)
  {
  case VectorElementTypes::Double:  eFunctionID = IntrinsicsAVXEnum::MoveMaskDouble;  break;
  case VectorElementTypes::Float:   eFunctionID = IntrinsicsAVXEnum::MoveMaskFloat;   break;
  default:
    {
      const uint32_t cuiElementCountSSE = static_cast<uint32_t>(_GetFallback()->GetVectorElementCount( eMaskElementType ));

      Expr *pMaskSSE = _ExtractSSEVector( eMaskElementType, pMaskExpr, (uiIndex < cuiElementCountSSE) );

      return _GetFallback()->CheckSingleMaskElement( eMaskElementType, pMaskSSE, uiIndex % cuiElementCountSSE );
    }
  }

  Expr *pMoveMask = _CreateFunctionCall( eFunctionID, pMaskExpr );

  return _GetASTHelper().CreateBinaryOperator( pMoveMask, _GetASTHelper().CreateLiteral<int32_t>(1 << uiIndex), BO_And, pMoveMask->getType() );
}

Expr* InstructionSetAVX::CreateOnesVector(VectorElementTypes eElementType, bool bNegative)
{
  Expr *pBroadCastValue = nullptr;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  pBroadCastValue = _GetASTHelper().CreateLiteral(bNegative ? -1.0  : 1.0 ); break;
  case VectorElementTypes::Float:                                   pBroadCastValue = _GetASTHelper().CreateLiteral(bNegative ? -1.0f : 1.0f); break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  pBroadCastValue = _GetASTHelper().CreateLiteral(bNegative ? -1    : 1   ); break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return BroadCast( eElementType, pBroadCastValue );
}

Expr* InstructionSetAVX::CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder)
{
  if (crvecElements.size() != GetVectorElementCount(eElementType))
  {
    throw RuntimeErrorException("The number of init expressions must be equal to the vector element count!");
  }

  if (! bReversedOrder)
  {
    // AVX vector creation methods expect the arguments in reversed order
    return CreateVector( eElementType, _SwapExpressionOrder( crvecElements ), true );
  }


  IntrinsicsAVXEnum eIntrinID = IntrinsicsAVXEnum::SetDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eIntrinID = IntrinsicsAVXEnum::SetDouble;   break;
  case VectorElementTypes::Float:                                   eIntrinID = IntrinsicsAVXEnum::SetFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eIntrinID = IntrinsicsAVXEnum::SetInt8;     break;
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eIntrinID = IntrinsicsAVXEnum::SetInt16;    break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eIntrinID = IntrinsicsAVXEnum::SetInt32;    break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eIntrinID = IntrinsicsAVXEnum::SetInt64;    break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eIntrinID, crvecElements );
}

Expr* InstructionSetAVX::CreateZeroVector(VectorElementTypes eElementType)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::SetZeroDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::SetZeroDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::SetZeroFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::SetZeroInteger;  break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID );
}

Expr* InstructionSetAVX::ExtractElement(VectorElementTypes eElementType, Expr *pVectorRef, uint32_t uiIndex)
{
  const uint32_t cuiElementCountAVX = static_cast<uint32_t>(GetVectorElementCount(eElementType));

  if (uiIndex >= cuiElementCountAVX)
  {
    throw InstructionSetExceptions::ExtractIndexOutOfRange( eElementType, cuiElementCountAVX );
  }

  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::ExtractInt8;

  switch (eElementType)
  {
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:   eFunctionID = IntrinsicsAVXEnum::ExtractInt8;  break;
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:  eFunctionID = IntrinsicsAVXEnum::ExtractInt16; break;
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:  eFunctionID = IntrinsicsAVXEnum::ExtractInt32; break;
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::ExtractInt64; break;
  default:
    {
      const uint32_t cuiElementCountSSE = static_cast<uint32_t>(_GetFallback()->GetVectorElementCount(eElementType));

      Expr *pVectorSSE = _ExtractSSEVector( eElementType, pVectorRef, (uiIndex < cuiElementCountSSE) );

      return _GetFallback()->ExtractElement( eElementType, pVectorSSE, uiIndex % cuiElementCountSSE );
    }
  }

  return _CreateFunctionCall( eFunctionID, pVectorRef, _GetASTHelper().CreateIntegerLiteral(uiIndex));
}

QualType InstructionSetAVX::GetVectorType(VectorElementTypes eElementType)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::SetZeroDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:                                  eFunctionID = IntrinsicsAVXEnum::SetZeroDouble;   break;
  case VectorElementTypes::Float:                                   eFunctionID = IntrinsicsAVXEnum::SetZeroFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:  eFunctionID = IntrinsicsAVXEnum::SetZeroInteger;  break;
  default:                                                          _ThrowUnsupportedType( eElementType );
  }

  return _GetFunctionReturnType( eFunctionID );
}

Expr* InstructionSetAVX::InsertElement(VectorElementTypes eElementType, Expr *pVectorRef, Expr *pElementValue, uint32_t uiIndex)
{
  const uint32_t cuiElementCountAVX = static_cast< uint32_t >( GetVectorElementCount(eElementType) );
  const uint32_t cuiElementCountSSE = static_cast< uint32_t >( _GetFallback()->GetVectorElementCount(eElementType) );

  if (uiIndex >= cuiElementCountAVX)
  {
    throw InstructionSetExceptions::InsertIndexOutOfRange( eElementType, cuiElementCountAVX );
  }

  const bool cbLowHalf = (uiIndex < cuiElementCountSSE);

  Expr *pVectorSSE  = _ExtractSSEVector( eElementType, pVectorRef, cbLowHalf );
  pVectorSSE        = _GetFallback()->InsertElement( eElementType, pVectorSSE, pElementValue, uiIndex % cuiElementCountSSE );

  return _InsertSSEVector( eElementType, pVectorRef, pVectorSSE, cbLowHalf );
}

bool InstructionSetAVX::IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, uint32_t uiParamCount) const
{
  bool bSupported = false;

  switch (eElementType)
  {
  case VectorElementTypes::Double: case VectorElementTypes::Float:
    {
      switch (eFunctionType)
      {
      case BuiltinFunctionsEnum::Abs:     bSupported = (uiParamCount == 1);   break;
      case BuiltinFunctionsEnum::Ceil:    bSupported = (uiParamCount == 1);   break;
      case BuiltinFunctionsEnum::Floor:   bSupported = (uiParamCount == 1);   break;
      case BuiltinFunctionsEnum::Max:     bSupported = (uiParamCount == 2);   break;
      case BuiltinFunctionsEnum::Min:     bSupported = (uiParamCount == 2);   break;
      case BuiltinFunctionsEnum::Sqrt:    bSupported = (uiParamCount == 1);   break;
      default:                            break;    // Useless default branch avoiding GCC compiler warnings
      }

      break;
    }
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
    const bool cbIsUnsigned = ( ! AST::BaseClasses::TypeInfo::IsSigned( eElementType ) );

      switch (eFunctionType)
      {
      case BuiltinFunctionsEnum::Abs:     bSupported = ( (uiParamCount == 1) && cbIsUnsigned );   break;
      case BuiltinFunctionsEnum::Ceil:    bSupported =   (uiParamCount == 1);                     break;
      case BuiltinFunctionsEnum::Floor:   bSupported =   (uiParamCount == 1);                     break;
      default:                            break;    // Useless default branch avoiding GCC compiler warnings
      }

      break;
    }
  default:  break;    // Useless default branch avoiding GCC compiler warnings
  }

  if (bSupported)
  {
    return true;
  }
  else
  {
    return _spFallbackInstructionSet->IsBuiltinFunctionSupported( eElementType, eFunctionType, uiParamCount );
  }
}

bool InstructionSetAVX::IsElementTypeSupported(VectorElementTypes eElementType) const
{
  switch (eElementType)
  {
  case VectorElementTypes::Double:
  case VectorElementTypes::Float:   return true;
  default:                          return _spFallbackInstructionSet->IsElementTypeSupported( eElementType );
  }
}

Expr* InstructionSetAVX::LoadVector(VectorElementTypes eElementType, Expr *pPointerRef)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::LoadDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:    eFunctionID = IntrinsicsAVXEnum::LoadDouble;   break;
  case VectorElementTypes::Float:     eFunctionID = IntrinsicsAVXEnum::LoadFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      eFunctionID = IntrinsicsAVXEnum::LoadInteger;

      QualType qtReturnType = _GetVectorType( eElementType, _GetASTHelper().IsPointerToConstType(pPointerRef->getType()) );

      pPointerRef = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType(qtReturnType) );
      break;
    }
  default:    _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pPointerRef );
}

Expr* InstructionSetAVX::LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex)
{
  switch (eIndexElementType)
  {
  case VectorElementTypes::Int32: case VectorElementTypes::Int64:   break;
  default:  throw RuntimeErrorException( std::string("Only index element types \"") + AST::BaseClasses::TypeInfo::GetTypeString( VectorElementTypes::Int32 ) + ("\" and \"") +
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


  const uint32_t cuiIndexOffset = uiGroupIndex * static_cast<uint32_t>(GetVectorElementCount(eIndexElementType) / GetVectorElementCount(eElementType));

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

Expr* InstructionSetAVX::RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS)
{
  if (eOpType == RelationalOperatorType::LogicalAnd)
  {
    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseAnd, pExprLHS, pExprRHS );
  }
  else if (eOpType == RelationalOperatorType::LogicalOr)
  {
    return ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseOr, pExprLHS, pExprRHS );
  }

  switch (eElementType)
  {
  case VectorElementTypes::Double: case VectorElementTypes::Float:
    {
      const IntrinsicsAVXEnum ceFunctionID  = (eElementType == VectorElementTypes::Double) ? IntrinsicsAVXEnum::CompareDouble : IntrinsicsAVXEnum::CompareFloat;
      int32_t                 iOpCode       = 0;

      switch (eOpType)
      {
      case RelationalOperatorType::Equal:         iOpCode = 0x00;   break;  // _CMP_EQ_OQ
      case RelationalOperatorType::Greater:       iOpCode = 0x1E;   break;  // _CMP_GT_OQ
      case RelationalOperatorType::GreaterEqual:  iOpCode = 0x1D;   break;  // _CMP_GE_OQ
      case RelationalOperatorType::Less:          iOpCode = 0x11;   break;  // _CMP_LT_OQ
      case RelationalOperatorType::LessEqual:     iOpCode = 0x12;   break;  // _CMP_LE_OQ
      case RelationalOperatorType::NotEqual:      iOpCode = 0x0C;   break;  // _CMP_NEQ_OQ
      default:                                    throw InternalErrorException("Unsupported relational operation detected!");
      }

      return _CreateFunctionCall( ceFunctionID, pExprLHS, pExprRHS, _GetASTHelper().CreateIntegerLiteral(iOpCode) );
    }
  default:
    {
      ClangASTHelper::ExpressionVectorType  vecSSERelOps;

      for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
      {
        const bool cbLowHalf = (uiIdx == 0);

        Expr *pExprLHS_SSE = _ExtractSSEVector( eElementType, pExprLHS, cbLowHalf );
        Expr *pExprRHS_SSE = _ExtractSSEVector( eElementType, pExprRHS, cbLowHalf );

        vecSSERelOps.push_back( _GetFallback()->RelationalOperator( eElementType, eOpType, pExprLHS_SSE, pExprRHS_SSE ) );
      }

      return _MergeSSEVectors( eElementType, vecSSERelOps[0], vecSSERelOps[1] );
    }
  }
}

Expr* InstructionSetAVX::ShiftElements(VectorElementTypes eElementType, Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount)
{
  if (uiCount == 0)
  {
    return pVectorRef;  // Nothing to do
  }

  ClangASTHelper::ExpressionVectorType  vecSSEShifts;

  for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
  {
    Expr *pVectorSSE = _ExtractSSEVector( eElementType, pVectorRef, (uiIdx == 0) );

    vecSSEShifts.push_back( _GetFallback()->ShiftElements( eElementType, pVectorSSE, bShiftLeft, uiCount ) );
  }

  return _MergeSSEVectors( eElementType, vecSSEShifts[0], vecSSEShifts[1] );
}

Expr* InstructionSetAVX::StoreVector(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue)
{
  IntrinsicsAVXEnum eFunctionID = IntrinsicsAVXEnum::StoreDouble;

  switch (eElementType)
  {
  case VectorElementTypes::Double:    eFunctionID = IntrinsicsAVXEnum::StoreDouble;   break;
  case VectorElementTypes::Float:     eFunctionID = IntrinsicsAVXEnum::StoreFloat;    break;
  case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
  case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
  case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
  case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    {
      eFunctionID = IntrinsicsAVXEnum::StoreInteger;
      pPointerRef = _CreatePointerCast( pPointerRef, _GetASTHelper().GetPointerType( GetVectorType( eElementType ) ) );
      break;
    }
  default:    _ThrowUnsupportedType( eElementType );
  }

  return _CreateFunctionCall( eFunctionID, pPointerRef, pVectorValue );
}

Expr* InstructionSetAVX::StoreVectorMasked(VectorElementTypes eElementType, Expr *pPointerRef, Expr *pVectorValue, Expr *pMaskRef)
{
  Expr *pReturnExpr = nullptr;

  switch (eElementType)
  {
  case VectorElementTypes::Double: case VectorElementTypes::Float:
    {
      Expr *pBlendedValue = BlendVectors( eElementType, pMaskRef, pVectorValue, LoadVector( eElementType, pPointerRef ) );
      pReturnExpr         = StoreVector( eElementType, pPointerRef, pBlendedValue );
      break;
    }
  default:
    {
      ClangASTHelper::ExpressionVectorType  vecSSEStores;

      for (uint32_t uiIdx = 0; uiIdx < 2; ++uiIdx)
      {
        const bool cbLowHalf = (uiIdx == 0);

        Expr *pVectorValueSSE = _ExtractSSEVector( eElementType, pVectorValue, cbLowHalf );
        Expr *pMaskRefSSE     = _ExtractSSEVector( eElementType, pMaskRef,     cbLowHalf );

        if (! cbLowHalf)
        {
          Expr *pOffset = _GetASTHelper().CreateIntegerLiteral<int32_t>(static_cast<int32_t>(_GetFallback()->GetVectorElementCount(eElementType)) );
          pPointerRef   = _GetASTHelper().CreateBinaryOperator( pPointerRef, pOffset, BO_Add, pPointerRef->getType() );
        }

        vecSSEStores.push_back( _GetFallback()->StoreVectorMasked( eElementType, pPointerRef, pVectorValueSSE, pMaskRefSSE ) );
      }

      vecSSEStores[0] = _GetASTHelper().CreateBinaryOperatorComma( vecSSEStores[0], vecSSEStores[1] );
      pReturnExpr     = _GetASTHelper().CreateParenthesisExpression( vecSSEStores[0] );
      break;
    }
  }

  return pReturnExpr;
}

Expr* InstructionSetAVX::UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, Expr *pSubExpr)
{
  Expr *pReturnExpr = nullptr;

  switch (eOpType)
  {
  case UnaryOperatorType::AddressOf:
    {
      pReturnExpr = _GetASTHelper().CreateUnaryOperator( pSubExpr, UO_AddrOf, _GetASTHelper().GetASTContext().getPointerType(pSubExpr->getType()) );
      break;
    }
  case UnaryOperatorType::BitwiseNot: case UnaryOperatorType::LogicalNot:
    {
      pReturnExpr = ArithmeticOperator( eElementType, ArithmeticOperatorType::BitwiseXOr, pSubExpr, _CreateFullBitMask(eElementType) );
      break;
    }
  case UnaryOperatorType::Minus:
    {
      switch (eElementType)
      {
      case VectorElementTypes::Double: case VectorElementTypes::Float:
        {
          pReturnExpr = ArithmeticOperator( eElementType, ArithmeticOperatorType::Multiply, pSubExpr, CreateOnesVector(eElementType, true) );
          break;
        }
      case VectorElementTypes::Int8:   case VectorElementTypes::UInt8:
      case VectorElementTypes::Int16:  case VectorElementTypes::UInt16:
      case VectorElementTypes::Int32:  case VectorElementTypes::UInt32:
      case VectorElementTypes::Int64:  case VectorElementTypes::UInt64:
        {
          pReturnExpr = ArithmeticOperator( eElementType, ArithmeticOperatorType::Subtract, CreateZeroVector(eElementType), pSubExpr );
          break;
        }
      default:  _ThrowUnsupportedType( eElementType );
      }

      break;
    }
  case UnaryOperatorType::Plus:
    {
      // Nothing to do
      pReturnExpr = pSubExpr;
      break;
    }
  case UnaryOperatorType::PostDecrement: case UnaryOperatorType::PostIncrement:
  case UnaryOperatorType::PreDecrement:  case UnaryOperatorType::PreIncrement:
    {
      const bool cbPrefixed   = ( (eOpType == UnaryOperatorType::PreDecrement)  || (eOpType == UnaryOperatorType::PreIncrement) );
      const bool cbIncrement  = ( (eOpType == UnaryOperatorType::PostIncrement) || (eOpType == UnaryOperatorType::PreIncrement) );

      pReturnExpr = _CreatePrePostFixedUnaryOp( eElementType, pSubExpr, cbPrefixed, cbIncrement );
      break;
    }
  default:  throw InternalErrorException("Unsupported unary operation detected!");
  }

  return pReturnExpr;
}



// Implementation of class InstructionSetAVX2
InstructionSetAVX2::InstructionSetAVX2(ASTContext &rAstContext)
    : BaseType(rAstContext) {
  _InitIntrinsicsMap();

  _CreateMissingIntrinsicsAVX2();

  InstructionSetBase::_LookupIntrinsics(_mapIntrinsicsAVX2, "AVX2");
}

void InstructionSetAVX2::_InitIntrinsicsMap() {
  // Abs functions
  _InitIntrinsic(IntrinsicsAVX2Enum::AbsInt8, "abs_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::AbsInt16, "abs_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::AbsInt32, "abs_epi32");

  // Add functions
  _InitIntrinsic(IntrinsicsAVX2Enum::AddInt8, "add_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::AddInt16, "add_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::AddInt32, "add_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::AddInt64, "add_epi64");

  // Bitwise "and" and "and not"
  _InitIntrinsic(IntrinsicsAVX2Enum::AndInteger, "and_si256");
  _InitIntrinsic(IntrinsicsAVX2Enum::AndNotInteger, "andnot_si256");

  // Blending function
  _InitIntrinsic(IntrinsicsAVX2Enum::BlendInteger, "blendv_epi8");

  // Compare functions
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareEqualInt8, "cmpeq_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareEqualInt16, "cmpeq_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareEqualInt32, "cmpeq_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareEqualInt64, "cmpeq_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareGreaterThanInt8, "cmpgt_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareGreaterThanInt16, "cmpgt_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareGreaterThanInt32, "cmpgt_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::CompareGreaterThanInt64, "cmpgt_epi64");

  // Convert functions for signed integers
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt8Int16, "cvtepi8_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt8Int32, "cvtepi8_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt8Int64, "cvtepi8_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt16Int32, "cvtepi16_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt16Int64, "cvtepi16_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertInt32Int64, "cvtepi32_epi64");

  // Convert functions for unsigned integers
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt8Int16, "cvtepu8_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt8Int32, "cvtepu8_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt8Int64, "cvtepu8_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt16Int32, "cvtepu16_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt16Int64, "cvtepu16_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ConvertUInt32Int64, "cvtepu32_epi64");

  // Extract SSE vectors functions
  _InitIntrinsic(IntrinsicsAVX2Enum::ExtractSSEIntegerInteger,
      "extracti128_si256");

  // Gather functions
  _InitIntrinsic(IntrinsicsAVX2Enum::GatherInt32, "i32gather_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::GatherInt64, "i32gather_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::GatherFloat, "i32gather_ps");
  _InitIntrinsic(IntrinsicsAVX2Enum::GatherDouble, "i32gather_pd");

  // Max functions
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxInt8, "max_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxInt16, "max_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxInt32, "max_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxUInt8, "max_epu8");
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxUInt16, "max_epu16");
  _InitIntrinsic(IntrinsicsAVX2Enum::MaxUInt32, "max_epu32");

  // Min functions
  _InitIntrinsic(IntrinsicsAVX2Enum::MinInt8, "min_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::MinInt16, "min_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::MinInt32, "min_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::MinUInt8, "min_epu8");
  _InitIntrinsic(IntrinsicsAVX2Enum::MinUInt16, "min_epu16");
  _InitIntrinsic(IntrinsicsAVX2Enum::MinUInt32, "min_epu32");

  // Mask conversion function
  _InitIntrinsic(IntrinsicsAVX2Enum::MoveMaskInt8, "movemask_epi8");

  // Multiplication functions
  _InitIntrinsic(IntrinsicsAVX2Enum::MultiplyInt16, "mullo_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::MultiplyInt32, "mullo_epi32");

  // Bitwise "or"
  _InitIntrinsic(IntrinsicsAVX2Enum::OrInteger, "or_si256");

  // Integer packing functions
  _InitIntrinsic(IntrinsicsAVX2Enum::PackInt16ToInt8, "packs_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::PackInt32ToInt16, "packs_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::PackInt16ToUInt8, "packus_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::PackInt32ToUInt16, "packus_epi32");

  // Permute function
  _InitIntrinsic(IntrinsicsAVX2Enum::PermuteCrossInt32, "permutevar8x32_epi32");

  // Shift functions
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftLeftInt16, "slli_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftLeftInt32, "slli_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftLeftInt64, "slli_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftLeftVectorBytes, "slli_si256");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightArithInt16, "srai_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightArithInt32, "srai_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightLogInt16, "srli_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightLogInt32, "srli_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightLogInt64, "srli_epi64");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShiftRightVectorBytes, "srli_si256");

  // Shuffle functions
  _InitIntrinsic(IntrinsicsAVX2Enum::ShuffleInt8, "shuffle_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::ShuffleInt32, "shuffle_epi32");

  // Subtraction functions
  _InitIntrinsic(IntrinsicsAVX2Enum::SubtractInt8, "sub_epi8");
  _InitIntrinsic(IntrinsicsAVX2Enum::SubtractInt16, "sub_epi16");
  _InitIntrinsic(IntrinsicsAVX2Enum::SubtractInt32, "sub_epi32");
  _InitIntrinsic(IntrinsicsAVX2Enum::SubtractInt64, "sub_epi64");

  // Bitwise "xor"
  _InitIntrinsic(IntrinsicsAVX2Enum::XorInteger, "xor_si256");
}

Expr* InstructionSetAVX2::_ShiftIntegerVectorBytes(Expr *pVectorRef,
    uint32_t uiByteCount, bool bShiftLeft) {
  if (uiByteCount == 0) {
    return pVectorRef;  // Nothing to do
  } else if (uiByteCount >= 32) {
    throw InternalErrorException("Cannot shift a vector by 32 bytes or more!");
  } else {
    const IntrinsicsAVX2Enum eShiftID = bShiftLeft ?
      IntrinsicsAVX2Enum::ShiftLeftVectorBytes :
      IntrinsicsAVX2Enum::ShiftRightVectorBytes;

    return _CreateFunctionCall(eShiftID, pVectorRef,
        _GetASTHelper().CreateIntegerLiteral<int32_t>(uiByteCount));
  }
}

Expr* InstructionSetAVX2::_ExtractSSEVector(VectorElementTypes eElementType,
    Expr *pAVXVector, bool bLowHalf) {
  IntrinsicsAVX2Enum eFunctionID = IntrinsicsAVX2Enum::ExtractSSEIntegerInteger;

  switch (eElementType) {
   case VectorElementTypes::Int8:  case VectorElementTypes::UInt8:
   case VectorElementTypes::Int16: case VectorElementTypes::UInt16:
   case VectorElementTypes::Int32: case VectorElementTypes::UInt32:
   case VectorElementTypes::Int64: case VectorElementTypes::UInt64:
    return _CreateFunctionCall(eFunctionID, pAVXVector,
        _GetASTHelper().CreateIntegerLiteral(bLowHalf ? 0 : 1));
   default:
    return BaseType::_ExtractSSEVector(eElementType, pAVXVector, bLowHalf);
  }

}

Expr* InstructionSetAVX2::_ConvertVector(VectorElementTypes eSourceType,
    VectorElementTypes eTargetType,
    const ClangASTHelper::ExpressionVectorType &crvecVectorRefs,
    uint32_t uiGroupIndex, bool bMaskConversion) {
  if (bMaskConversion) {
    switch (eSourceType) {
     case VectorElementTypes::Int8:
     case VectorElementTypes::UInt8:
     case VectorElementTypes::Int16:
     case VectorElementTypes::UInt16:
     case VectorElementTypes::Int32:
     case VectorElementTypes::UInt32:
     case VectorElementTypes::Int64:
     case VectorElementTypes::UInt64: {
      const size_t cszSourceSize =
        AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType);
      const size_t cszTargetSize =
        AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType);

      const VectorElementTypes ceNewSourceType =
        AST::BaseClasses::TypeInfo::CreateSizedIntegerType(
            cszSourceSize, true).GetType();
      const VectorElementTypes ceNewTargetType =
        AST::BaseClasses::TypeInfo::CreateSizedIntegerType(
            cszTargetSize, true).GetType();

      switch (eTargetType) {
       case VectorElementTypes::Int8:
       case VectorElementTypes::UInt8:
       case VectorElementTypes::Int16:
       case VectorElementTypes::UInt16:
       case VectorElementTypes::Int32:
       case VectorElementTypes::UInt32: {
        if (cszSourceSize > cszTargetSize) {
          // Boost downward mask conversions by special AVX2 signed integer
          // downward conversions
          return ConvertVectorDown(ceNewSourceType, ceNewTargetType,
              crvecVectorRefs);
        } else if (cszSourceSize < cszTargetSize) {
          return ConvertVectorUp(ceNewSourceType, ceNewTargetType,
              crvecVectorRefs.front(), uiGroupIndex);
        }
        break;
       }

       case VectorElementTypes::Float: {
        if (cszSourceSize < cszTargetSize) {
          Expr *pIntMask = ConvertVectorUp(ceNewSourceType,
              VectorElementTypes::Int32, crvecVectorRefs.front(), uiGroupIndex);

          // Just cast to float, instead of convert
          return ConvertMaskSameSize(VectorElementTypes::Int32, eTargetType,
              pIntMask);
        }
        break;
       }

       case VectorElementTypes::Double: {
        if (cszSourceSize < cszTargetSize) {
          Expr *pIntMask = ConvertVectorUp(ceNewSourceType,
              VectorElementTypes::Int64, crvecVectorRefs.front(), uiGroupIndex);

          // Just cast to double, instead of convert
          return ConvertMaskSameSize(VectorElementTypes::Int64, eTargetType,
              pIntMask);
        }
        break;
       }

       default: break; // Useless default branch avoiding GCC compiler warnings
      }
     }

     default: break; // Useless default branch avoiding GCC compiler warnings
    }
  } else {
    bool bHandleConversion = true;
    IntrinsicsAVX2Enum eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int32;

    switch (eTargetType) {
     case VectorElementTypes::Int8:
     case VectorElementTypes::UInt8: {

      IntrinsicsAVX2Enum ePack16Bit =
          eTargetType == VectorElementTypes::Int8 ?
            IntrinsicsAVX2Enum::PackInt16ToInt8 :
            IntrinsicsAVX2Enum::PackInt16ToUInt8;

      Expr* pClamp = BroadCast(eSourceType,
          _GetASTHelper().CreateIntegerLiteral<uint8_t>(-1));

      switch (eSourceType) {
       case VectorElementTypes::Int16:
       case VectorElementTypes::UInt16: {
        // Special downward conversion
        if (crvecVectorRefs.size() != 2) {
          throw InternalErrorException(
              "Element count mismatch for conversion.");
        }

        ClangASTHelper::ExpressionVectorType vecPermutationElements;
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(7));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(6));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(3));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(2));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(5));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(4));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(1));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(0));

        return _CreateFunctionCall(IntrinsicsAVX2Enum::PermuteCrossInt32,
            _CreateFunctionCall(ePack16Bit,
              _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[0], pClamp),
              _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[1], pClamp)),
            CreateVector(VectorElementTypes::Int32, vecPermutationElements,
                         true));
       }

       case VectorElementTypes::Int32:
       case VectorElementTypes::UInt32: {
        // Special downward conversion
        if (crvecVectorRefs.size() != 4) {
          throw InternalErrorException(
              "Element count mismatch for conversion.");
        }

        ClangASTHelper::ExpressionVectorType vecPermutationElements;
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(7));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(3));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(6));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(2));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(5));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(1));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(4));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(0));

        return _CreateFunctionCall(IntrinsicsAVX2Enum::PermuteCrossInt32,
            _CreateFunctionCall(ePack16Bit,
              _CreateFunctionCall(ePack16Bit,
                _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[0], pClamp),
                _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[1], pClamp)),
              _CreateFunctionCall(ePack16Bit,
                _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[2], pClamp),
                _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[3], pClamp))),
            CreateVector(VectorElementTypes::Int32, vecPermutationElements,
                         true));
       }

       case VectorElementTypes::Float: {
        // Special downward conversion
        ClangASTHelper::ExpressionVectorType vecValues;

        for (auto vecRef : crvecVectorRefs) {
          ClangASTHelper::ExpressionVectorType vecCurRef;
          vecCurRef.push_back(vecRef);
          vecValues.push_back(_ConvertVector(eSourceType,
                VectorElementTypes::Int32, vecCurRef, uiGroupIndex,
                bMaskConversion));
        }

        return _ConvertVector(VectorElementTypes::Int32, eTargetType,
            vecValues, uiGroupIndex, bMaskConversion);
       }

       case VectorElementTypes::Double: {
        // Special downward conversion
        ClangASTHelper::ExpressionVectorType vecValues;

        for (auto vecRef : crvecVectorRefs) {
          ClangASTHelper::ExpressionVectorType vecCurRef;
          vecCurRef.push_back(vecRef);
          vecValues.push_back(_ConvertVector(eSourceType,
                VectorElementTypes::Int32, vecCurRef, uiGroupIndex,
                bMaskConversion));
        }

        return _ConvertVector(VectorElementTypes::Int32, eTargetType,
            vecValues, uiGroupIndex, bMaskConversion);
       }

       default:
        bHandleConversion = false;
      }
      break;
     }

     case VectorElementTypes::Int16:
     case VectorElementTypes::UInt16:
      switch (eSourceType) {
       case VectorElementTypes::Int8:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int16;
        break;

       case VectorElementTypes::UInt8:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt8Int16;
        break;

       case VectorElementTypes::Int32:
       case VectorElementTypes::UInt32: {
        // Special downward conversion
        if (crvecVectorRefs.size() != 2) {
          throw InternalErrorException(
              "Element count mismatch for conversion.");
        }

        IntrinsicsAVX2Enum ePack32Bit =
            eTargetType == VectorElementTypes::Int16 ?
              IntrinsicsAVX2Enum::PackInt32ToInt16 :
              IntrinsicsAVX2Enum::PackInt32ToUInt16;

        Expr* pClamp = BroadCast(eSourceType,
            _GetASTHelper().CreateIntegerLiteral<uint16_t>(-1));

        ClangASTHelper::ExpressionVectorType vecPermutationElements;
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(7));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(6));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(3));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(2));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(5));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(4));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(1));
        vecPermutationElements.push_back(
          _GetASTHelper().CreateIntegerLiteral<int32_t>(0));

        return _CreateFunctionCall(IntrinsicsAVX2Enum::PermuteCrossInt32,
            _CreateFunctionCall(ePack32Bit,
              _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[0], pClamp),
              _CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger, crvecVectorRefs[1], pClamp)),
            CreateVector(VectorElementTypes::Int32, vecPermutationElements,
                         true));
       }

       case VectorElementTypes::Float: {
        // Special downward conversion
        ClangASTHelper::ExpressionVectorType vecValues;

        for (auto vecRef : crvecVectorRefs) {
          ClangASTHelper::ExpressionVectorType vecCurRef;
          vecCurRef.push_back(vecRef);
          vecValues.push_back(_ConvertVector(eSourceType,
                VectorElementTypes::Int32, vecCurRef, uiGroupIndex,
                bMaskConversion));
        }

        return _ConvertVector(VectorElementTypes::Int32, eTargetType,
            vecValues, uiGroupIndex, bMaskConversion);
       }

       case VectorElementTypes::Double: {
        // Special downward conversion
        ClangASTHelper::ExpressionVectorType vecValues;

        for (auto vecRef : crvecVectorRefs) {
          ClangASTHelper::ExpressionVectorType vecCurRef;
          vecCurRef.push_back(vecRef);
          vecValues.push_back(_ConvertVector(eSourceType,
                VectorElementTypes::Int32, vecCurRef, uiGroupIndex,
                bMaskConversion));
        }

        return _ConvertVector(VectorElementTypes::Int32, eTargetType,
            vecValues, uiGroupIndex, bMaskConversion);
       }


       default:
        bHandleConversion = false;
      }
      break;

     case VectorElementTypes::Int32:
     case VectorElementTypes::UInt32:
      switch (eSourceType) {
       case VectorElementTypes::Int8:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int32;
        break;

       case VectorElementTypes::UInt8:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt8Int32;
        break;

       case VectorElementTypes::Int16:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt16Int32;
        break;

       case VectorElementTypes::UInt16:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt16Int32;
        break;

       default:
        bHandleConversion = false;
      }
      break;

     case VectorElementTypes::Int64:
     case VectorElementTypes::UInt64:
      switch (eSourceType) {
       case VectorElementTypes::Int8:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int64;
        break;

       case VectorElementTypes::UInt8:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt8Int64;
        break;

       case VectorElementTypes::Int16:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt16Int64;
        break;

       case VectorElementTypes::UInt16:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt16Int64;
        break;

       case VectorElementTypes::Int32:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt32Int64;
        break;

       case VectorElementTypes::UInt32:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt32Int64;
        break;

       default:
        bHandleConversion = false;
      }
      break;

     case VectorElementTypes::Float:
      switch (eSourceType) {
       case VectorElementTypes::Int8:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int32;
        break;

       case VectorElementTypes::UInt8:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt8Int32;
        break;

       case VectorElementTypes::Int16:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt16Int32;
        break;

       case VectorElementTypes::UInt16:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt16Int32;
        break;

       default:
        bHandleConversion = false;
      }
      break;

     case VectorElementTypes::Double:
      switch (eSourceType) {
       case VectorElementTypes::Int8:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt8Int64;
        break;

       case VectorElementTypes::UInt8:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt8Int64;
        break;

       case VectorElementTypes::Int16:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt16Int64;
        break;

       case VectorElementTypes::UInt16:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt16Int64;
        break;

       case VectorElementTypes::Int32:
        eConvertID = IntrinsicsAVX2Enum::ConvertInt32Int64;
        break;

       case VectorElementTypes::UInt32:
        eConvertID = IntrinsicsAVX2Enum::ConvertUInt32Int64;
        break;

       default:
        bHandleConversion = false;
      }
      break;

     default:
      bHandleConversion = false;
    }

    if (bHandleConversion) {
      // Upward conversion
      const uint32_t cuiShiftMultiplier = 32 / static_cast<uint32_t>(
          AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType) /
          AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType));

      uiGroupIndex *= cuiShiftMultiplier;
      bool bLowHalf = uiGroupIndex < 16;
      uiGroupIndex = uiGroupIndex % 16;

      switch (eTargetType) {
       case VectorElementTypes::Float:
       case VectorElementTypes::Double: {
        Expr *pSelectedGroup = crvecVectorRefs.front();

        if (uiGroupIndex != 0) {
          pSelectedGroup =
            _ShiftIntegerVectorBytes(pSelectedGroup, uiGroupIndex, false);
        }

        pSelectedGroup =
          _ExtractSSEVector(eSourceType, pSelectedGroup, bLowHalf);

        ClangASTHelper::ExpressionVectorType vecSelectedGroup;
        vecSelectedGroup.push_back(
            _CreateFunctionCall(eConvertID, pSelectedGroup));

        return _ConvertVector(VectorElementTypes::Int32, eTargetType,
            vecSelectedGroup, uiGroupIndex, bMaskConversion);
       }

       default: {
        Expr* pExpr = _ShiftIntegerVectorBytes(crvecVectorRefs.front(),
            uiGroupIndex, false);

        if (AST::BaseClasses::TypeInfo::GetTypeSize(eSourceType) <
            AST::BaseClasses::TypeInfo::GetTypeSize(eTargetType)) {
          // Upward conversion requires SSE vector
          pExpr = _ExtractSSEVector(eSourceType, pExpr, bLowHalf);
        }

        return _CreateFunctionCall(eConvertID, pExpr);
       }
      }
    }
  }

  // If the function has not returned earlier, let the base handle the
  // conversion
  return BaseType::_ConvertVector(eSourceType, eTargetType, crvecVectorRefs,
      uiGroupIndex, bMaskConversion);
}

Expr* InstructionSetAVX2::ArithmeticOperator(
    VectorElementTypes eElementType, ArithmeticOperatorType eOpType,
    Expr *pExprLHS, Expr *pExprRHS, bool bIsRHSScalar) {
  IntrinsicsAVX2Enum eFunctionID = IntrinsicsAVX2Enum::AddInt8;

  switch (eElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
    switch (eOpType) {
     case ArithmeticOperatorType::Add:
      eFunctionID = IntrinsicsAVX2Enum::AddInt8;
      break;

     case ArithmeticOperatorType::BitwiseAnd:
      eFunctionID = IntrinsicsAVX2Enum::AndInteger;
      break;

     case ArithmeticOperatorType::BitwiseOr:
      eFunctionID = IntrinsicsAVX2Enum::OrInteger;
      break;

     case ArithmeticOperatorType::BitwiseXOr:
      eFunctionID = IntrinsicsAVX2Enum::XorInteger;
      break;

     case ArithmeticOperatorType::Divide:
     case ArithmeticOperatorType::Multiply:
      // TODO
      return BaseType::ArithmeticOperator(eElementType, eOpType,
                                          pExprLHS, pExprRHS);

     case ArithmeticOperatorType::Subtract:
      eFunctionID = IntrinsicsAVX2Enum::SubtractInt8;
      break;

     case ArithmeticOperatorType::ShiftLeft:
     case ArithmeticOperatorType::ShiftRight:
      throw RuntimeErrorException(
          "Shift operations are undefined for Int8 data types!");

     case ArithmeticOperatorType::Modulo:
      throw RuntimeErrorException(
          "Modulo operation is undefined for floating point data types!");

     default:
      throw InternalErrorException(
          "Unsupported arithmetic operation detected!");
    }
    return _CreateFunctionCall( eFunctionID, pExprLHS, pExprRHS );

   case VectorElementTypes::Int16:
   case VectorElementTypes::UInt16:
    switch (eOpType) {
     case ArithmeticOperatorType::Add:
      eFunctionID = IntrinsicsAVX2Enum::AddInt16;
      break;

     case ArithmeticOperatorType::BitwiseAnd:
      eFunctionID = IntrinsicsAVX2Enum::AndInteger;
      break;

     case ArithmeticOperatorType::BitwiseOr:
      eFunctionID = IntrinsicsAVX2Enum::OrInteger;
      break;

     case ArithmeticOperatorType::BitwiseXOr:
      eFunctionID = IntrinsicsAVX2Enum::XorInteger;
      break;

     case ArithmeticOperatorType::Divide:
      // TODO
      return BaseType::ArithmeticOperator(eElementType, eOpType,
                                          pExprLHS, pExprRHS);

     case ArithmeticOperatorType::Multiply:
      eFunctionID = IntrinsicsAVX2Enum::MultiplyInt16;
      break;

     case ArithmeticOperatorType::Subtract:
      eFunctionID = IntrinsicsAVX2Enum::SubtractInt16;
      break;

     case ArithmeticOperatorType::ShiftLeft:
      eFunctionID = IntrinsicsAVX2Enum::ShiftLeftInt16;
      break;

     case ArithmeticOperatorType::ShiftRight:
      eFunctionID = IntrinsicsAVX2Enum::ShiftRightArithInt16;
      break;

     case ArithmeticOperatorType::Modulo:
      throw RuntimeErrorException(
          "Modulo operation is undefined for floating point data types!");

     default:
      throw InternalErrorException(
          "Unsupported arithmetic operation detected!");
    }
    return _CreateFunctionCall( eFunctionID, pExprLHS, pExprRHS );

   case VectorElementTypes::Int32:
   case VectorElementTypes::UInt32:
    switch (eOpType) {
     case ArithmeticOperatorType::Add:
      eFunctionID = IntrinsicsAVX2Enum::AddInt32;
      break;

     case ArithmeticOperatorType::BitwiseAnd:
      eFunctionID = IntrinsicsAVX2Enum::AndInteger;
      break;

     case ArithmeticOperatorType::BitwiseOr:
      eFunctionID = IntrinsicsAVX2Enum::OrInteger;
      break;

     case ArithmeticOperatorType::BitwiseXOr:
      eFunctionID = IntrinsicsAVX2Enum::XorInteger;
      break;

     case ArithmeticOperatorType::Divide:
      // TODO
      return BaseType::ArithmeticOperator(eElementType, eOpType,
                                          pExprLHS, pExprRHS);

     case ArithmeticOperatorType::Multiply:
      eFunctionID = IntrinsicsAVX2Enum::MultiplyInt32;
      break;

     case ArithmeticOperatorType::Subtract:
      eFunctionID = IntrinsicsAVX2Enum::SubtractInt32;
      break;

     case ArithmeticOperatorType::ShiftLeft:
      eFunctionID = IntrinsicsAVX2Enum::ShiftLeftInt32;
      break;

     case ArithmeticOperatorType::ShiftRight:
      eFunctionID = IntrinsicsAVX2Enum::ShiftRightArithInt32;
      break;

     case ArithmeticOperatorType::Modulo:
      throw RuntimeErrorException(
          "Modulo operation is undefined for floating point data types!");

     default:
      throw InternalErrorException(
          "Unsupported arithmetic operation detected!");
    }
    return _CreateFunctionCall( eFunctionID, pExprLHS, pExprRHS );

   case VectorElementTypes::Int64:
   case VectorElementTypes::UInt64:
    switch (eOpType) {
     case ArithmeticOperatorType::Add:
      eFunctionID = IntrinsicsAVX2Enum::AddInt64;
      break;

     case ArithmeticOperatorType::BitwiseAnd:
      eFunctionID = IntrinsicsAVX2Enum::AndInteger;
      break;

     case ArithmeticOperatorType::BitwiseOr:
      eFunctionID = IntrinsicsAVX2Enum::OrInteger;
      break;

     case ArithmeticOperatorType::BitwiseXOr:
      eFunctionID = IntrinsicsAVX2Enum::XorInteger;
      break;

     case ArithmeticOperatorType::Divide:
     case ArithmeticOperatorType::Multiply:
      // TODO
      return BaseType::ArithmeticOperator(eElementType, eOpType,
                                          pExprLHS, pExprRHS);

     case ArithmeticOperatorType::Subtract:
      eFunctionID = IntrinsicsAVX2Enum::SubtractInt64;
      break;

     case ArithmeticOperatorType::ShiftLeft:
      eFunctionID = IntrinsicsAVX2Enum::ShiftLeftInt64;
      break;

     case ArithmeticOperatorType::ShiftRight:
      throw RuntimeErrorException(
          "Shift right operation is undefined for Int64 data types!");

     case ArithmeticOperatorType::Modulo:
      throw RuntimeErrorException(
          "Modulo operation is undefined for floating point data types!");

     default:
      throw InternalErrorException(
          "Unsupported arithmetic operation detected!");
    }
    return BaseType::ArithmeticOperator(eElementType, eOpType,
                                        pExprLHS, pExprRHS, bIsRHSScalar);
    break;

   case VectorElementTypes::Double:
   case VectorElementTypes::Float:
   default:
    return BaseType::ArithmeticOperator(eElementType, eOpType,
                                        pExprLHS, pExprRHS, bIsRHSScalar);
  }

}

Expr* InstructionSetAVX2::BlendVectors(VectorElementTypes eElementType,
    Expr *pMaskRef, Expr *pVectorTrue, Expr *pVectorFalse) {
  switch (eElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
    return _CreateFunctionCall(IntrinsicsAVX2Enum::BlendInteger, pVectorFalse,
        pVectorTrue, pMaskRef);

   default:
    return BaseType::BlendVectors(eElementType, pMaskRef, pVectorTrue,
        pVectorFalse);
  }
}

Expr* InstructionSetAVX2::BuiltinFunction(
    VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType,
    const ClangASTHelper::ExpressionVectorType &crvecArguments) {
  const uint32_t cuiParamCount = static_cast<uint32_t>(crvecArguments.size());

  if (!IsBuiltinFunctionSupported(eElementType, eFunctionType, cuiParamCount)) {
    throw InstructionSetExceptions::UnsupportedBuiltinFunctionType(eElementType,
        eFunctionType, cuiParamCount, "AVX2");
  }

  bool bHandleBuiltin = true;
  IntrinsicsAVX2Enum eFunctionID = IntrinsicsAVX2Enum::AbsInt8;

  switch (eFunctionType) {
   case BuiltinFunctionsEnum::Abs:
    switch (eElementType) {
     case VectorElementTypes::Int8:
      eFunctionID = IntrinsicsAVX2Enum::AbsInt8;
      break;

     case VectorElementTypes::Int16:
      eFunctionID = IntrinsicsAVX2Enum::AbsInt16;
      break;

     case VectorElementTypes::Int32:
      eFunctionID = IntrinsicsAVX2Enum::AbsInt32;
      break;

     default:
      bHandleBuiltin = false;
      break;
    }
    break;

   case BuiltinFunctionsEnum::Max:
    switch (eElementType) {
     case VectorElementTypes::Int8:
      eFunctionID = IntrinsicsAVX2Enum::MaxInt8;
      break;

     case VectorElementTypes::UInt8:
      eFunctionID = IntrinsicsAVX2Enum::MaxUInt8;
      break;

     case VectorElementTypes::Int16:
      eFunctionID = IntrinsicsAVX2Enum::MaxInt16;
      break;

     case VectorElementTypes::UInt16:
      eFunctionID = IntrinsicsAVX2Enum::MaxUInt16;
      break;

     case VectorElementTypes::Int32:
      eFunctionID = IntrinsicsAVX2Enum::MaxInt32;
      break;

     case VectorElementTypes::UInt32:
      eFunctionID = IntrinsicsAVX2Enum::MaxUInt32;
      break;

     default:
      bHandleBuiltin = false;
      break;
    }
    break;

   case BuiltinFunctionsEnum::Min:
    switch (eElementType) {
     case VectorElementTypes::Int8:
      eFunctionID = IntrinsicsAVX2Enum::MinInt8;
      break;

     case VectorElementTypes::UInt8:
      eFunctionID = IntrinsicsAVX2Enum::MinUInt8;
      break;

     case VectorElementTypes::Int16:
      eFunctionID = IntrinsicsAVX2Enum::MinInt16;
      break;

     case VectorElementTypes::UInt16:
      eFunctionID = IntrinsicsAVX2Enum::MinUInt16;
      break;

     case VectorElementTypes::Int32:
      eFunctionID = IntrinsicsAVX2Enum::MinInt32;
      break;

     case VectorElementTypes::UInt32:
      eFunctionID = IntrinsicsAVX2Enum::MinUInt32;
      break;

     default:
      bHandleBuiltin = false;
      break;
    }
    break;

   default:
    bHandleBuiltin = false;
    break;
  }

  if (bHandleBuiltin) {
    if (cuiParamCount == 1) {
      return _CreateFunctionCall(eFunctionID, crvecArguments.front());
    } else if (cuiParamCount == 2) {
      return _CreateFunctionCall(eFunctionID, crvecArguments[0],
                                              crvecArguments[1]);
    }
  }

  return BaseType::BuiltinFunction(eElementType, eFunctionType, crvecArguments);
}

Expr* InstructionSetAVX2::CheckActiveElements(
    VectorElementTypes eMaskElementType,
    ActiveElementsCheckType eCheckType, Expr *pMaskExpr) {
  const QualType cqtBool = _GetClangType(VectorElementTypes::Bool);

  switch (eMaskElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8: {
     int32_t iTestConstant =
       (eCheckType == ActiveElementsCheckType::All) ? 0xFFFF : 0;

     const IntrinsicsAVX2Enum ceFunctionID = IntrinsicsAVX2Enum::MoveMaskInt8;
     const BinaryOperatorKind ceCompareOp =
       (eCheckType == ActiveElementsCheckType::Any) ? BO_NE : BO_EQ;

     return _GetASTHelper().CreateBinaryOperator(
         _CreateFunctionCall(ceFunctionID, pMaskExpr),
         _GetASTHelper().CreateLiteral(iTestConstant), ceCompareOp, cqtBool);
    }

   default: {
     return BaseType::CheckActiveElements(eMaskElementType, eCheckType,
         pMaskExpr);
   }
  }
}

Expr* InstructionSetAVX2::CheckSingleMaskElement(
    VectorElementTypes eMaskElementType, Expr *pMaskExpr, uint32_t uiIndex) {
  if (uiIndex >= static_cast<uint32_t>(
        GetVectorElementCount(eMaskElementType))) {
    throw InternalErrorException(
        "The index cannot exceed the vector element count!");
  }

  IntrinsicsAVX2Enum eFunctionID = IntrinsicsAVX2Enum::MoveMaskInt8;

  switch (eMaskElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
    eFunctionID = IntrinsicsAVX2Enum::MoveMaskInt8;
    break;

   default:
    return BaseType::CheckSingleMaskElement(eMaskElementType, pMaskExpr,
        uiIndex);
  }

  Expr *pMoveMask = _CreateFunctionCall(eFunctionID, pMaskExpr);

  return _GetASTHelper().CreateBinaryOperator(pMoveMask,
      _GetASTHelper().CreateLiteral<int32_t>(1 << uiIndex),
      BO_And, pMoveMask->getType());
}

bool InstructionSetAVX2::IsBuiltinFunctionSupported(
    VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType,
    uint32_t uiParamCount) const {
  bool bSupported = false;

  switch (eElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
   case VectorElementTypes::Int16:
   case VectorElementTypes::UInt16:
   case VectorElementTypes::Int32:
   case VectorElementTypes::UInt32: {
    const bool cbIsUnsigned =
      (!AST::BaseClasses::TypeInfo::IsSigned(eElementType));

    switch (eFunctionType) {
     case BuiltinFunctionsEnum::Abs:
      bSupported = (uiParamCount == 1) && cbIsUnsigned;
      break;

     case BuiltinFunctionsEnum::Max:
      bSupported = (uiParamCount == 2);
      break;

     case BuiltinFunctionsEnum::Min:
      bSupported = (uiParamCount == 2);
      break;

     default:
      break;    // Useless default branch avoiding GCC compiler warnings
    }
    break;
   }

   default:
    break;    // Useless default branch avoiding GCC compiler warnings
  }

  if (bSupported) {
    return true;
  } else {
    return BaseType::IsBuiltinFunctionSupported(eElementType, eFunctionType,
        uiParamCount);
  }
}

::clang::Expr* InstructionSetAVX2::LoadVectorGathered(
    VectorElementTypes eElementType,
    VectorElementTypes eIndexElementType,
    ::clang::Expr *pPointerRef,
    const ClangASTHelper::ExpressionVectorType &crvecIndexExprs,
    uint32_t uiGroupIndex) {
  // TODO: Add support for 64 bit index element types
  switch (eIndexElementType) {
   case VectorElementTypes::Int32:
   //case VectorElementTypes::Int64:
    break;

   default:
    throw RuntimeErrorException(std::string("Only index element types \"")
        + AST::BaseClasses::TypeInfo::GetTypeString(VectorElementTypes::Int32)
        //+ ("\" and \"")
        //+ AST::BaseClasses::TypeInfo::GetTypeString( VectorElementTypes::Int64)
        + ("\" supported for gathered vector loads!"));
  }

  switch (eElementType) {
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
   case VectorElementTypes::Int16:
   case VectorElementTypes::UInt16:
   case VectorElementTypes::Int32:
   case VectorElementTypes::UInt32: {
    ClangASTHelper::ExpressionVectorType vecLoads;

    VectorElementTypes eIntermediateType = VectorElementTypes::Int32;
    QualType qtTarget = _GetClangType(eIntermediateType);
    qtTarget.addConst();
    QualType qtTargetPointer = _GetASTHelper().GetPointerType(qtTarget);

    const size_t cszElementSize =
      AST::BaseClasses::TypeInfo::GetTypeSize(eElementType);
    const size_t cszIntermediateSize =
      AST::BaseClasses::TypeInfo::GetTypeSize(eIntermediateType);

    for (auto idxExpr : crvecIndexExprs) {
      vecLoads.push_back(_CreateFunctionCall(IntrinsicsAVX2Enum::GatherInt32,
          _CreatePointerCast(pPointerRef, qtTargetPointer), idxExpr,
          _GetASTHelper().CreateIntegerLiteral<int32_t>(static_cast<int32_t>(cszElementSize))));
    }

    if (cszElementSize < cszIntermediateSize) {
      // Zero out superfluous bits for Int8, UInt8, Int16, and UInt16
      ClangASTHelper::ExpressionVectorType vecZeroed;
      for (auto loadExpr : vecLoads) {
        vecZeroed.push_back(_CreateFunctionCall(IntrinsicsAVX2Enum::AndInteger,
              loadExpr, BroadCast(eIntermediateType,
                _GetASTHelper().CreateIntegerLiteral<int32_t>(
                  0xffffffff >> ((cszIntermediateSize-cszElementSize)*8)))));
      }

      if (AST::BaseClasses::TypeInfo::IsSigned(eElementType)) {
        return _ConvertVector(VectorElementTypes::Int32, eElementType,
                vecZeroed, 0, false);
      } else {
        return _ConvertVector(VectorElementTypes::UInt32, eElementType,
                vecZeroed, 0, false);
      }
    } else {
      // Must be a single vector for Int32 and UInt32
      return vecLoads.front();
    }
   }

   case VectorElementTypes::Int64:
   case VectorElementTypes::UInt64: {
    VectorElementTypes eIntermediateType = VectorElementTypes::Int64;
    QualType qtTarget = _GetClangType(eIntermediateType);
    qtTarget.addConst();
    QualType qtTargetPointer = _GetASTHelper().GetPointerType(qtTarget);

    return _CreateFunctionCall(IntrinsicsAVX2Enum::GatherInt64,
          _CreatePointerCast(pPointerRef, qtTargetPointer),
          crvecIndexExprs.front(),
          _GetASTHelper().CreateIntegerLiteral<int32_t>(1));
   }

   case VectorElementTypes::Float:
    return _CreateFunctionCall(IntrinsicsAVX2Enum::GatherFloat,
        pPointerRef, crvecIndexExprs.front(),
        _GetASTHelper().CreateIntegerLiteral<int32_t>(4));

   case VectorElementTypes::Double:
    // TODO: Determine whether to extract upper or lower half from SSE vector
    return _CreateFunctionCall(IntrinsicsAVX2Enum::GatherDouble,
        pPointerRef,
        _ExtractSSEVector(eIndexElementType, crvecIndexExprs.front(), true),
        _GetASTHelper().CreateIntegerLiteral<int32_t>(8));

   default:
    break;
  }

  return BaseType::LoadVectorGathered(eElementType, eIndexElementType,
      pPointerRef, crvecIndexExprs, uiGroupIndex);
}

Expr* InstructionSetAVX2::RelationalOperator(VectorElementTypes eElementType,
    RelationalOperatorType eOpType, Expr *pExprLHS, Expr *pExprRHS) {
  bool bHandleOperator = true;
  bool bFlipAll = false;
  bool bFlipEqual = false;

  RelationalOperatorType eCompareOpType = RelationalOperatorType::Equal;
  IntrinsicsAVX2Enum eFunctionID = IntrinsicsAVX2Enum::CompareEqualInt8;
  IntrinsicsAVX2Enum eEqualFunctionID = IntrinsicsAVX2Enum::CompareEqualInt8;

  switch (eOpType) {
   case RelationalOperatorType::Equal:
    eCompareOpType = RelationalOperatorType::Equal;
    break;

   case RelationalOperatorType::NotEqual:
    eCompareOpType = RelationalOperatorType::Equal;
    bFlipAll = true;
    break;

   case RelationalOperatorType::Less:
    eCompareOpType = RelationalOperatorType::Greater;
    bFlipAll = true;
    bFlipEqual = true;
    break;

   case RelationalOperatorType::LessEqual:
    eCompareOpType = RelationalOperatorType::Greater;
    bFlipAll = true;
    break;

   case RelationalOperatorType::Greater:
    eCompareOpType = RelationalOperatorType::Greater;
    break;

   case RelationalOperatorType::GreaterEqual:
    eCompareOpType = RelationalOperatorType::Greater;
    bFlipEqual = true;
    break;

   default:
    bHandleOperator = false;
    break;
  }

  if (bHandleOperator) {
    switch (eElementType) {
     case VectorElementTypes::Int8:
     case VectorElementTypes::UInt8:
      eFunctionID = eCompareOpType == RelationalOperatorType::Equal ?
          IntrinsicsAVX2Enum::CompareEqualInt8 :
          IntrinsicsAVX2Enum::CompareGreaterThanInt8;
      eEqualFunctionID = IntrinsicsAVX2Enum::CompareEqualInt8;
      break;

     case VectorElementTypes::Int16:
     case VectorElementTypes::UInt16:
      eFunctionID = eCompareOpType == RelationalOperatorType::Equal ?
          IntrinsicsAVX2Enum::CompareEqualInt16 :
          IntrinsicsAVX2Enum::CompareGreaterThanInt16;
      eEqualFunctionID = IntrinsicsAVX2Enum::CompareEqualInt16;
      break;

     case VectorElementTypes::Int32:
     case VectorElementTypes::UInt32:
      eFunctionID = eCompareOpType == RelationalOperatorType::Equal ?
          IntrinsicsAVX2Enum::CompareEqualInt32 :
          IntrinsicsAVX2Enum::CompareGreaterThanInt32;
      eEqualFunctionID = IntrinsicsAVX2Enum::CompareEqualInt32;
      break;

     case VectorElementTypes::Int64:
     case VectorElementTypes::UInt64:
      eFunctionID = eCompareOpType == RelationalOperatorType::Equal ?
          IntrinsicsAVX2Enum::CompareEqualInt64 :
          IntrinsicsAVX2Enum::CompareGreaterThanInt64;
      eEqualFunctionID = IntrinsicsAVX2Enum::CompareEqualInt64;
      break;

     default:
      bHandleOperator = false;
      break;
    }
  }

  if (bHandleOperator) {
    Expr *exprOperator = _CreateFunctionCall(eFunctionID, pExprLHS, pExprRHS);

    if (bFlipAll) {
      exprOperator = _CreateFunctionCall(IntrinsicsAVX2Enum::XorInteger,
          exprOperator, BroadCast(VectorElementTypes::Int8,
            _GetASTHelper().CreateIntegerLiteral<int8_t>(-1)));
    }

    if (bFlipEqual) {
      exprOperator = _CreateFunctionCall(IntrinsicsAVX2Enum::XorInteger,
          exprOperator,
          _CreateFunctionCall(eEqualFunctionID, pExprLHS, pExprRHS));
    }

    return exprOperator;
  }

  return BaseType::RelationalOperator(eElementType, eOpType, pExprLHS,
      pExprRHS);
}

Expr* InstructionSetAVX2::ShiftElements(VectorElementTypes eElementType,
    Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) {
  if (uiCount == 0) {
    return pVectorRef;  // Nothing to do
  }

  IntegerLiteral *pShiftCount =
    _GetASTHelper().CreateIntegerLiteral<int32_t>(uiCount);

  if (bShiftLeft) {
    switch (eElementType) {
     case VectorElementTypes::Int8:
     case VectorElementTypes::UInt8: {
      // Convert vector to UInt16 data type, do the shift and convert back
      // (there is no arithmetic left shift)
      ClangASTHelper::ExpressionVectorType vecShiftedVectors;

      for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx) {
        Expr *pConvertedVector = ConvertVectorUp(VectorElementTypes::UInt8,
            VectorElementTypes::UInt16, pVectorRef, uiGroupIdx);
        vecShiftedVectors.push_back(ShiftElements(VectorElementTypes::UInt16,
              pConvertedVector, bShiftLeft, uiCount));
      }

      return ConvertVectorDown(VectorElementTypes::UInt16,
          VectorElementTypes::UInt8, vecShiftedVectors);
     }

     case VectorElementTypes::Int16:
     case VectorElementTypes::UInt16:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftLeftInt16,
          pVectorRef, pShiftCount);

     case VectorElementTypes::Int32:
     case VectorElementTypes::UInt32:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftLeftInt32,
          pVectorRef, pShiftCount);

     case VectorElementTypes::Int64:
     case VectorElementTypes::UInt64:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftLeftInt64,
          pVectorRef, pShiftCount);

     default:
      throw RuntimeErrorException(
          "Shift operations are only defined for integer element types!");
    }
  } else {
    switch (eElementType) {
     case VectorElementTypes::Int8:
     case VectorElementTypes::UInt8: {
       // Convert vector to a signed / unsigned 16-bit integer data type, do
       // the shift and convert back
       const VectorElementTypes ceIntermediateType =
         AST::BaseClasses::TypeInfo::CreateSizedIntegerType(2,
             AST::BaseClasses::TypeInfo::IsSigned(eElementType)).GetType();

       ClangASTHelper::ExpressionVectorType vecShiftedVectors;

       for (uint32_t uiGroupIdx = 0; uiGroupIdx <= 1; ++uiGroupIdx) {
         Expr *pConvertedVector = ConvertVectorUp(eElementType,
             ceIntermediateType, pVectorRef, uiGroupIdx);
         vecShiftedVectors.push_back(ShiftElements(ceIntermediateType,
               pConvertedVector, bShiftLeft, uiCount));
       }

       return ConvertVectorDown(ceIntermediateType, VectorElementTypes::UInt8,
           vecShiftedVectors);
     }

     case VectorElementTypes::Int16:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftRightArithInt16,
          pVectorRef, pShiftCount);

     case VectorElementTypes::UInt16:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftRightLogInt16,
          pVectorRef, pShiftCount);

     case VectorElementTypes::Int32:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftRightArithInt32,
          pVectorRef, pShiftCount);

     case VectorElementTypes::UInt32:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftRightLogInt32,
          pVectorRef, pShiftCount);

     case VectorElementTypes::Int64: {
      // This is unsupported by AVX2 => Extract elements and shift them
      // separately
      ClangASTHelper::ExpressionVectorType vecElements;

      for (uint32_t uiIndex = 0; uiIndex < GetVectorElementCount(eElementType);
           ++uiIndex) {
        vecElements.push_back(_GetASTHelper().CreateBinaryOperator(
              ExtractElement(eElementType, pVectorRef, uiIndex),
              pShiftCount, BO_Shr, _GetClangType(eElementType)));
      }

      return CreateVector(eElementType, vecElements, false);
     }

     case VectorElementTypes::UInt64:
      return _CreateFunctionCall(IntrinsicsAVX2Enum::ShiftRightLogInt64,
          pVectorRef, pShiftCount);

     default:
      throw RuntimeErrorException(
          "Shift operations are only defined for integer element types!");
    }
  }
}

Expr* InstructionSetAVX2::StoreVectorMasked(VectorElementTypes eElementType,
    Expr *pPointerRef, Expr *pVectorValue, Expr *pMaskRef) {
  Expr *pReturnExpr = nullptr;

  switch (eElementType) {
   case VectorElementTypes::Double:
   case VectorElementTypes::Float:
   case VectorElementTypes::Int8:
   case VectorElementTypes::UInt8:
   case VectorElementTypes::Int16:
   case VectorElementTypes::UInt16:
   case VectorElementTypes::Int32:
   case VectorElementTypes::UInt32:
   case VectorElementTypes::Int64:
   case VectorElementTypes::UInt64: {
    Expr *pBlendedValue = BlendVectors(eElementType, pMaskRef, pVectorValue,
        LoadVector(eElementType, pPointerRef));
    pReturnExpr = StoreVector(eElementType, pPointerRef, pBlendedValue);
    break;
   }

   default:
    return BaseType::StoreVectorMasked(eElementType, pPointerRef, pVectorValue,
        pMaskRef);
  }

  return pReturnExpr;
}


// vim: set ts=2 sw=2 sts=2 et ai:

