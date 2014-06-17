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

//===--- VectorizationAST.cpp - Implements a vectorizable syntax tree. ---------------===//
//
// This file implements the internally used vectorizable syntax tree (a simplification to clang's AST)
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/VectorizationAST.h"
#include <algorithm>
#include <sstream>

using namespace clang::hipacc::Backend::Vectorization;
using namespace std;

#define CHECK_NULL_POINTER(ptr)   if (ptr == nullptr)   { throw InternalErrors::NullPointerException(#ptr); }


class XMLSupport
{
public:

  typedef map< string, string >   AttributesMapType;


public:

  inline static string CreateXmlTag(const size_t cszIntend, string strName)
  {
    return CreateXmlTag(cszIntend, strName, string(""));
  }

  inline static string CreateXmlTag(const size_t cszIntend, string strName, const AttributesMapType &crmapAttributes)
  {
    return CreateXmlTag(cszIntend, strName, string(""), crmapAttributes);
  }

  inline static string CreateXmlTag(const size_t cszIntend, string strName, const string &crstrInternalText)
  {
    return CreateXmlTag(cszIntend, strName, crstrInternalText, AttributesMapType());
  }

  static string CreateXmlTag(const size_t cszIntend, string strName, const string &crstrInternalText, const AttributesMapType &crmapAttributes);


  inline static string GetPadString(const size_t cszIntend)    { return string(cszIntend, ' '); }


  template <typename ValueType>   inline static string ToString(ValueType TValue)
  {
    stringstream OutpuStream;
    OutpuStream << TValue;
    return OutpuStream.str();
  }
};

template <> inline string XMLSupport::ToString<bool>(bool TValue)
{
  return TValue ? "true" : "false";
}


// Implementation of class AST::XMLSupport
string XMLSupport::CreateXmlTag(const size_t cszIntend, string strName, const string &crstrInternalText, const AttributesMapType &crmapAttributes)
{
  string strAttributes("");

  for (auto itAttribute : crmapAttributes)
  {
    strAttributes += string(" ") + itAttribute.first + string("=\"") + itAttribute.second + string("\"");
  }


  if (crstrInternalText.empty())
  {
    return GetPadString(cszIntend) + string("<") + strName + strAttributes + string(" />\n");
  }
  else
  {
    string strXmlString("");

    strXmlString += GetPadString(cszIntend) + string("<") + strName + strAttributes + string(">\n");
    strXmlString += crstrInternalText;
    strXmlString += GetPadString(cszIntend) + string("</") + strName + string(">\n");

    return strXmlString;
  }
}



/***********************/
/***   BaseClasses   ***/
/***********************/

// Implementation of class AST::BaseClasses::TypeInfo
AST::BaseClasses::TypeInfo& AST::BaseClasses::TypeInfo::operator=(const TypeInfo &crRVal)
{
  _bIsConst   = crRVal._bIsConst;
  _bIsPointer = crRVal._bIsPointer;
  _eType      = crRVal._eType;

  _vecArrayDimensions.clear();
  _vecArrayDimensions.insert(_vecArrayDimensions.end(), crRVal._vecArrayDimensions.begin(), crRVal._vecArrayDimensions.end());

  return *this;
}

AST::BaseClasses::TypeInfo AST::BaseClasses::TypeInfo::CreateDereferencedType() const
{
  TypeInfo ReturnType(*this);

  if (ReturnType.IsArray())
  {
    ReturnType.GetArrayDimensions().erase( ReturnType.GetArrayDimensions().begin() );
  }
  else if (ReturnType.GetPointer())
  {
    ReturnType.SetPointer(false);
  }
  else
  {
    throw ASTExceptions::NonDereferencableType();
  }

  return ReturnType;
}

AST::BaseClasses::TypeInfo AST::BaseClasses::TypeInfo::CreatePointerType() const
{
  TypeInfo ReturnType(*this);

  if (ReturnType.IsSingleValue())
  {
    ReturnType.SetPointer(true);
  }
  else
  {
    throw RuntimeErrorException("Only single value types have a corresponding pointer type!");
  }

  return ReturnType;
}

AST::BaseClasses::TypeInfo AST::BaseClasses::TypeInfo::CreateSizedIntegerType(size_t szTypeSize, bool bSigned)
{
  TypeInfo ReturnType;

  ReturnType.SetConst(false);
  ReturnType.SetPointer(false);

  KnownTypes eType = KnownTypes::Unknown;

  switch (szTypeSize)
  {
  case sizeof( int8_t  ):   eType = bSigned ? KnownTypes::Int8  : KnownTypes::UInt8;    break;
  case sizeof( int16_t ):   eType = bSigned ? KnownTypes::Int16 : KnownTypes::UInt16;   break;
  case sizeof( int32_t ):   eType = bSigned ? KnownTypes::Int32 : KnownTypes::UInt32;   break;
  case sizeof( int64_t ):   eType = bSigned ? KnownTypes::Int64 : KnownTypes::UInt64;   break;
  }

  ReturnType.SetType(eType);

  return ReturnType;
}

string AST::BaseClasses::TypeInfo::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["Type"]       = GetTypeString( _eType );
  mapAttributes["is_const"]   = XMLSupport::ToString( GetConst() );
  mapAttributes["is_pointer"] = XMLSupport::ToString( GetPointer() );
  mapAttributes["is_array"]   = XMLSupport::ToString( IsArray() );

  if (IsArray())
  {
    string strDim("");

    for (auto itDim : _vecArrayDimensions)
    {
      strDim += string("[") + XMLSupport::ToString(itDim) + string("]");
    }

    mapAttributes["array_dim"] = strDim;
  }

  return XMLSupport::CreateXmlTag(cszIntend, "TypeInfo", mapAttributes);
}

AST::BaseClasses::TypeInfo::KnownTypes AST::BaseClasses::TypeInfo::GetPromotedType(KnownTypes eTypeLHS, KnownTypes eTypeRHS)
{
  if      ((eTypeLHS == KnownTypes::Unknown) || (eTypeRHS == KnownTypes::Unknown))
  {
    return KnownTypes::Unknown;
  }
  else if ((eTypeLHS == KnownTypes::Double)  || (eTypeRHS == KnownTypes::Double))
  {
    return KnownTypes::Double;
  }
  else if ((eTypeLHS == KnownTypes::Float)   || (eTypeRHS == KnownTypes::Float))
  {
    return KnownTypes::Float;
  }
  else if ((eTypeLHS == KnownTypes::Bool)    && (eTypeRHS == KnownTypes::Bool))
  {
    return KnownTypes::Bool;
  }
  else
  {
    // We have an integer type => Promote to the larger type and keep the sign
    size_t  szTypeSize  = std::max( TypeInfo::GetTypeSize(eTypeLHS), TypeInfo::GetTypeSize(eTypeRHS) );
    bool    bSigned     = TypeInfo::IsSigned(eTypeLHS) | TypeInfo::IsSigned(eTypeRHS);

    return TypeInfo::CreateSizedIntegerType(szTypeSize, bSigned).GetType();
  }
}


size_t AST::BaseClasses::TypeInfo::GetTypeSize(KnownTypes eType)
{
  switch (eType)
  {
  case KnownTypes::Bool:    return static_cast< size_t >( 1 );
  case KnownTypes::Int8:    return sizeof( int8_t );
  case KnownTypes::UInt8:   return sizeof( uint8_t );
  case KnownTypes::Int16:   return sizeof( int16_t );
  case KnownTypes::UInt16:  return sizeof( uint16_t );
  case KnownTypes::Int32:   return sizeof( int32_t );
  case KnownTypes::UInt32:  return sizeof( uint32_t );
  case KnownTypes::Int64:   return sizeof( int64_t );
  case KnownTypes::UInt64:  return sizeof( uint64_t );
  case KnownTypes::Float:   return sizeof( float );
  case KnownTypes::Double:  return sizeof( double );
  case KnownTypes::Unknown: return static_cast< size_t >( 0 );
  default:                  throw InternalErrorException("Unknown type!"); 
  }
}

string AST::BaseClasses::TypeInfo::GetTypeString(KnownTypes eType)
{
  switch (eType)
  {
  case KnownTypes::Bool:    return "Bool";
  case KnownTypes::Int8:    return "Int8";
  case KnownTypes::UInt8:   return "UInt8";
  case KnownTypes::Int16:   return "Int16";
  case KnownTypes::UInt16:  return "UInt16";
  case KnownTypes::Int32:   return "Int32";
  case KnownTypes::UInt32:  return "UInt32";
  case KnownTypes::Int64:   return "Int64";
  case KnownTypes::UInt64:  return "UInt64";
  case KnownTypes::Float:   return "Float";
  case KnownTypes::Double:  return "Double";
  case KnownTypes::Unknown: return "Unknown";
  default:                  throw InternalErrorException("Unknown type!");
  }
}

bool AST::BaseClasses::TypeInfo::IsEqual(const TypeInfo &crRVal, bool bIgnoreConstQualifier)
{
  // Check array dimensions
  const size_t cszArrayDimCount = GetArrayDimensions().size();
  if (cszArrayDimCount != crRVal.GetArrayDimensions().size())
  {
    return false;
  }

  for (size_t szDim = static_cast<size_t>(0); szDim < cszArrayDimCount; ++szDim)
  {
    if (GetArrayDimensions()[szDim] != crRVal.GetArrayDimensions()[szDim])
    {
      return false;
    }
  }

  // Check pointer declaration
  if (GetPointer() != crRVal.GetPointer())
  {
    return false;
  }

  // Check element type
  if (GetType() != crRVal.GetType())
  {
    return false;
  }

  // Check const qualifier
  if (bIgnoreConstQualifier)
  {
    return true;
  }
  else
  {
    return (GetConst() == crRVal.GetConst());
  }
}

bool AST::BaseClasses::TypeInfo::IsSigned(KnownTypes eType)
{
  switch (eType)
  {
  case KnownTypes::Int8:
  case KnownTypes::Int16:
  case KnownTypes::Int32:
  case KnownTypes::Int64:
  case KnownTypes::Float:
  case KnownTypes::Double:  return true;
  case KnownTypes::Bool:
  case KnownTypes::UInt8:
  case KnownTypes::UInt16:
  case KnownTypes::UInt32:
  case KnownTypes::UInt64:
  case KnownTypes::Unknown: return false;
  default:                  throw InternalErrorException("Unknown type!");
  }
}


// Implementation of class AST::BaseClasses::VariableInfo
AST::BaseClasses::VariableInfoPtr AST::BaseClasses::VariableInfo::Create(string strName, const TypeInfo &crTypeInfo, bool bVectorize)
{
  AST::BaseClasses::VariableInfoPtr spVariableInfo( new AST::BaseClasses::VariableInfo );

  spVariableInfo->GetTypeInfo() = crTypeInfo;
  spVariableInfo->SetName(strName);
  spVariableInfo->SetVectorize(bVectorize);

  return spVariableInfo;
}

string AST::BaseClasses::VariableInfo::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["name"]       = GetName();
  mapAttributes["vectorize"]  = XMLSupport::ToString( GetVectorize() );


  return XMLSupport::CreateXmlTag(cszIntend, "Variable", _Type.DumpToXML(cszIntend + 2), mapAttributes);
}


// Implementation of class AST::BaseClasses::Node
string AST::BaseClasses::Node::_DumpChildToXml(const NodeConstPtr spChild, const size_t cszIntend)
{
  return spChild ? spChild->DumpToXML(cszIntend) : "";
}

void AST::BaseClasses::Node::_SetParentToChild(NodePtr spChild) const
{
  if (spChild)
  {
    spChild->_SetParent(_wpThis.lock());
  }
}

AST::IndexType AST::BaseClasses::Node::GetHierarchyLevel() const
{
  IndexType iHierarchyLevel = static_cast< IndexType >( 0 );

  NodeConstPtr spCurrentNode = GetThis();
  while (true)
  {
    spCurrentNode = spCurrentNode->GetParent();

    if (spCurrentNode)
    {
      ++iHierarchyLevel;
    }
    else
    {
      break;
    }
  }

  return iHierarchyLevel;
}

AST::BaseClasses::NodePtr AST::BaseClasses::Node::GetParent()
{
  return _wpParent.expired() ? nullptr : _wpParent.lock();
}

AST::ScopePosition AST::BaseClasses::Node::GetScopePosition()
{
  NodePtr spCurrentNode = GetThis();
  while (spCurrentNode)
  {
    NodePtr spParent = spCurrentNode->GetParent();
    if (spParent->IsType<AST::Scope>())
    {
      AST::ScopePtr spScope = spParent->CastToType<AST::Scope>();

      return AST::ScopePosition( spScope, spScope->GetChildIndex(spCurrentNode) );
    }

    spCurrentNode = spParent;
  }

  throw InternalErrorException("Cannot find parent scope!");
}


// Implementation of class AST::BaseClasses::Expression
AST::IndexType AST::BaseClasses::Expression::_FindSubExpressionIndex(ExpressionConstPtr spSubExpression) const
{
  for (IndexType iExprIndex = static_cast<IndexType>(0); iExprIndex < GetSubExpressionCount(); ++iExprIndex)
  {
    if (spSubExpression == GetSubExpression(iExprIndex))
    {
      return iExprIndex;
    }
  }

  throw InternalErrorException("Could not find the specified expression in the list of sub-expressions!");
}

string AST::BaseClasses::Expression::_DumpResultTypeToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["vectorize"] = XMLSupport::ToString( IsVectorized() );

  return XMLSupport::CreateXmlTag(cszIntend, "ResultType", GetResultType().DumpToXML(cszIntend + 2), mapAttributes);
}

AST::IndexType AST::BaseClasses::Expression::GetParentIndex() const
{
  if (! IsSubExpression())
  {
    throw InternalErrorException("The current expression is not a sub-expression of another expression!");
  }

  return GetParent()->CastToType<Expression>()->_FindSubExpressionIndex( GetThis()->CastToType<Expression>() );
}

bool AST::BaseClasses::Expression::IsSubExpression() const
{
  if (! GetParent())
  {
    return false;
  }
  else
  {
    return GetParent()->IsType<Expression>();
  }
}

bool AST::BaseClasses::Expression::IsVectorized()
{
  bool bIsVectorized = false;

  for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < GetSubExpressionCount(); ++iChildIdx)
  {
    ExpressionPtr spChild = GetSubExpression(iChildIdx);

    if (spChild)
    {
      bIsVectorized |= spChild->IsVectorized();
    }
  }

  return bIsVectorized;
}



/***********************/
/***   ControlFlow   ***/
/***********************/

// Implementation of class AST::ControlFlow::Loop
string AST::ControlFlow::Loop::_GetLoopTypeString(LoopType eType)
{
  switch (eType)
  {
  case LoopType::TopControlled:     return "TopControlled";
  case LoopType::BottomControlled:  return "BottomControlled";
  default:                          throw InternalErrorException("Unknown loop type!");
  }
}

AST::ControlFlow::LoopPtr AST::ControlFlow::Loop::Create(LoopType eType, BaseClasses::ExpressionPtr spCondition, BaseClasses::ExpressionPtr spIncrement)
{
  LoopPtr spNewLoop = AST::CreateNode< Loop >();

  spNewLoop->SetLoopType(eType);
  spNewLoop->SetCondition(spCondition);
  spNewLoop->SetIncrement(spIncrement);

  // Initialize loop body
  spNewLoop->GetBody();

  return spNewLoop;
}

string AST::ControlFlow::Loop::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["hierarchy_level"]  = XMLSupport::ToString( GetHierarchyLevel() );
  mapAttributes["type"]             = _GetLoopTypeString(GetLoopType());
  mapAttributes["vectorize"]        = XMLSupport::ToString( IsVectorized() );
  mapAttributes["force_vectorize"]  = XMLSupport::ToString( GetForcedVectorization() );

  string strXmlString("");

  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Condition", _DumpChildToXml(GetCondition(), cszIntend + 4) );
  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Increment", _DumpChildToXml(GetIncrement(), cszIntend + 4) );
  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Body",      _DumpChildToXml(GetBody(), cszIntend + 4) );

  return XMLSupport::CreateXmlTag(cszIntend, "Loop", strXmlString, mapAttributes);
}

AST::ScopePtr AST::ControlFlow::Loop::GetBody()
{
  if (! _spBody)
  {
    _SetChildPtr(_spBody, Scope::Create());
  }

  return _spBody;
}

AST::ScopeConstPtr AST::ControlFlow::Loop::GetBody() const
{
  CHECK_NULL_POINTER(_spBody);

  return _spBody;
}

AST::BaseClasses::NodePtr AST::ControlFlow::Loop::GetChild(IndexType ChildIndex)
{
  switch (ChildIndex)
  {
  case 0:   return GetCondition();
  case 1:   return GetIncrement();
  case 2:   return GetBody();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

bool AST::ControlFlow::Loop::IsVectorized() const
{
  if (GetForcedVectorization())
  {
    return true;
  }
  else if (GetCondition())
  {
    return GetCondition()->IsVectorized();
  }

  return false;
}


// Implementation of class AST::ControlFlow::LoopControlStatement
string AST::ControlFlow::LoopControlStatement::_GetLoopControlTypeString(LoopControlType eType)
{
  switch (eType)
  {
  case LoopControlType::Break:      return "Break";
  case LoopControlType::Continue:   return "Continue";
  default:                          throw InternalErrorException("Unknown loop control statement type!");
  }
}

AST::ControlFlow::LoopControlStatementPtr AST::ControlFlow::LoopControlStatement::Create(LoopControlType eCtrlType)
{
  LoopControlStatementPtr spLoopCtrlStatement = AST::CreateNode<LoopControlStatement>();

  spLoopCtrlStatement->SetControlType(eCtrlType);

  return spLoopCtrlStatement;
}

string AST::ControlFlow::LoopControlStatement::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["type"]       = _GetLoopControlTypeString( GetControlType() );
  mapAttributes["vectorize"]  = XMLSupport::ToString( IsVectorized() );

  return XMLSupport::CreateXmlTag(cszIntend, "LoopControlStatement", mapAttributes);
}

AST::ControlFlow::LoopPtr AST::ControlFlow::LoopControlStatement::GetControlledLoop()
{
  AST::BaseClasses::NodePtr spCurrentNode = GetThis();
  while (true)
  {
    spCurrentNode = spCurrentNode->GetParent();
    CHECK_NULL_POINTER( spCurrentNode );

    if (spCurrentNode->IsType<AST::ControlFlow::Loop>())
    {
      return spCurrentNode->CastToType<AST::ControlFlow::Loop>();
    }
  }
}

bool AST::ControlFlow::LoopControlStatement::IsVectorized() const
{
  LoopConstPtr spControlledLoop = GetControlledLoop();
  bool bVectorized = false;

  AST::BaseClasses::NodeConstPtr spCurrentNode = GetThis();
  while (spCurrentNode != spControlledLoop)
  {
    spCurrentNode = spCurrentNode->GetParent();
    CHECK_NULL_POINTER( spCurrentNode );

    if (spCurrentNode->IsType<AST::BaseClasses::ControlFlowStatement>())
    {
      bVectorized |= spCurrentNode->CastToType<AST::BaseClasses::ControlFlowStatement>()->IsVectorized();
    }
  }

  return bVectorized;
}


// Implementation of class AST::ControlFlow::ConditionalBranch
AST::ControlFlow::ConditionalBranchPtr AST::ControlFlow::ConditionalBranch::Create(ExpressionPtr spCondition)
{
  ConditionalBranchPtr spCondBranch = AST::CreateNode<ConditionalBranch>();

  spCondBranch->SetCondition(spCondition);

  // Initialize branch body
  spCondBranch->GetBody();

  return spCondBranch;
}

string AST::ControlFlow::ConditionalBranch::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["vectorize"] = XMLSupport::ToString(IsVectorized());

  string strXmlString("");

  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Condition", _DumpChildToXml(GetCondition(), cszIntend + 4) );
  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Body",      _DumpChildToXml(GetBody(), cszIntend + 4) );

  return XMLSupport::CreateXmlTag(cszIntend, "ConditionalBranch", strXmlString, mapAttributes);
}

AST::ScopePtr AST::ControlFlow::ConditionalBranch::GetBody()
{
  if (! _spBody)
  {
    _SetChildPtr(_spBody, Scope::Create());
  }

  return _spBody;
}

AST::ScopeConstPtr AST::ControlFlow::ConditionalBranch::GetBody() const
{
  CHECK_NULL_POINTER(_spBody);

  return _spBody;
}

AST::BaseClasses::NodePtr AST::ControlFlow::ConditionalBranch::GetChild(IndexType ChildIndex)
{
  switch (ChildIndex)
  {
  case 0:   return GetCondition();
  case 1:   return GetBody();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

bool AST::ControlFlow::ConditionalBranch::IsVectorized() const
{
  if (GetCondition())
  {
    return GetCondition()->IsVectorized();
  }

  return false;
}


// Implementation of class AST::ControlFlow::BranchingStatement
void AST::ControlFlow::BranchingStatement::AddConditionalBranch(ConditionalBranchPtr spBranch)
{
  CHECK_NULL_POINTER(spBranch);

  _SetParentToChild(spBranch);
  _vecBranches.push_back(spBranch);
}

AST::ControlFlow::BranchingStatementPtr AST::ControlFlow::BranchingStatement::Create()
{
  BranchingStatementPtr spBranchingStatement = AST::CreateNode<BranchingStatement>();

  // Initialize body of default branch
  spBranchingStatement->GetDefaultBranch();

  return spBranchingStatement;
}

string AST::ControlFlow::BranchingStatement::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["hierarchy_level"]  = XMLSupport::ToString(GetHierarchyLevel());
  mapAttributes["vectorize"]        = XMLSupport::ToString(IsVectorized());

  string strXmlString("");

  // Dump conditional branches
  for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < GetConditionalBranchesCount(); ++iBranchIdx)
  {
    XMLSupport::AttributesMapType mapIndexAttributes;

    mapIndexAttributes["index"] = XMLSupport::ToString(iBranchIdx);

    strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Branch", GetConditionalBranch(iBranchIdx)->DumpToXML(cszIntend + 4), mapIndexAttributes );
  }

  // Dump default branch
  strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "DefaultBranch", GetDefaultBranch()->DumpToXML(cszIntend + 4));

  return XMLSupport::CreateXmlTag(cszIntend, "BranchingStatement", strXmlString, mapAttributes);
}

AST::BaseClasses::NodePtr AST::ControlFlow::BranchingStatement::GetChild(IndexType ChildIndex)
{
  if (ChildIndex < GetConditionalBranchesCount())
  {
    return GetConditionalBranch(ChildIndex);
  }
  else if (ChildIndex == GetConditionalBranchesCount())
  {
    return GetDefaultBranch();
  }
  else
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
}

AST::ControlFlow::ConditionalBranchPtr AST::ControlFlow::BranchingStatement::GetConditionalBranch(IndexType BranchIndex)
{
  if (BranchIndex >= GetConditionalBranchesCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }

  return _vecBranches[BranchIndex];
}

AST::ScopePtr AST::ControlFlow::BranchingStatement::GetDefaultBranch()
{
  if (! _spDefaultBranch)
  {
    _SetChildPtr(_spDefaultBranch, Scope::Create());
  }

  return _spDefaultBranch;
}

AST::ScopeConstPtr AST::ControlFlow::BranchingStatement::GetDefaultBranch() const
{
  CHECK_NULL_POINTER(_spDefaultBranch);

  return _spDefaultBranch;
}

bool AST::ControlFlow::BranchingStatement::IsVectorized() const
{
  bool bVectorized = false;

  for (IndexType iBranchIdx = static_cast<IndexType>(0); iBranchIdx < GetConditionalBranchesCount(); ++iBranchIdx)
  {
    bVectorized |= GetConditionalBranch(iBranchIdx)->IsVectorized();
  }

  return bVectorized;
}

void AST::ControlFlow::BranchingStatement::RemoveConditionalBranch(IndexType BranchIndex)
{
  if (BranchIndex >= GetConditionalBranchesCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }

  _vecBranches.erase( _vecBranches.begin() + BranchIndex );
}


// Implementation of class AST::ControlFlow::ReturnStatement
AST::ControlFlow::ReturnStatementPtr AST::ControlFlow::ReturnStatement::Create()
{
  return AST::CreateNode< ReturnStatement >();
}

string AST::ControlFlow::ReturnStatement::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["vectorize"] = XMLSupport::ToString( IsVectorized() );

  return XMLSupport::CreateXmlTag(cszIntend, "ReturnStatement", mapAttributes);
}

bool AST::ControlFlow::ReturnStatement::IsVectorized() const
{
  AST::BaseClasses::NodeConstPtr spCurrentNode = GetThis();
  while (true)
  {
    spCurrentNode = spCurrentNode->GetParent();
    if (! spCurrentNode)
    {
      return false;
    }

    if (spCurrentNode->IsType<AST::BaseClasses::ControlFlowStatement>())
    {
      if (spCurrentNode->CastToType<AST::BaseClasses::ControlFlowStatement>()->IsVectorized())
      {
        return true;
      }
    }
  }
}



/***********************/
/***   Expressions   ***/
/***********************/

// Implementation of class AST::Expressions::Constant
string AST::Expressions::Constant::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["type"]   = BaseClasses::TypeInfo::GetTypeString(_eType);
  mapAttributes["value"]  = GetAsString();

  return XMLSupport::CreateXmlTag(cszIntend, "Constant", mapAttributes);
}

void AST::Expressions::Constant::ChangeType(KnownTypes eNewType)
{
  switch (eNewType)
  {
  case KnownTypes::Bool:    _ChangeType< bool     >( eNewType );  break;
  case KnownTypes::Int8:    _ChangeType< int8_t   >( eNewType );  break;
  case KnownTypes::UInt8:   _ChangeType< uint8_t  >( eNewType );  break;
  case KnownTypes::Int16:   _ChangeType< int16_t  >( eNewType );  break;
  case KnownTypes::UInt16:  _ChangeType< uint16_t >( eNewType );  break;
  case KnownTypes::Int32:   _ChangeType< int32_t  >( eNewType );  break;
  case KnownTypes::UInt32:  _ChangeType< uint32_t >( eNewType );  break;
  case KnownTypes::Int64:   _ChangeType< int64_t  >( eNewType );  break;
  case KnownTypes::UInt64:  _ChangeType< uint64_t >( eNewType );  break;
  case KnownTypes::Float:   _ChangeType< float    >( eNewType );  break;
  case KnownTypes::Double:  _ChangeType< double   >( eNewType );  break;
  default:                  throw RuntimeErrorException( string("Invalid constant type: ") + BaseClasses::TypeInfo::GetTypeString(eNewType) );
  }
}

string AST::Expressions::Constant::GetAsString() const
{
  switch (_eType)
  {
  case KnownTypes::Bool:    return XMLSupport::ToString( GetValue< bool     >() );
  case KnownTypes::Int8:    return XMLSupport::ToString( GetValue< int16_t  >() );  // Avoid automatic char conversion
  case KnownTypes::UInt8:   return XMLSupport::ToString( GetValue< uint16_t >() );  // Avoid automatic char conversion
  case KnownTypes::Int16:   return XMLSupport::ToString( GetValue< int16_t  >() );
  case KnownTypes::UInt16:  return XMLSupport::ToString( GetValue< uint16_t >() );
  case KnownTypes::Int32:   return XMLSupport::ToString( GetValue< int32_t  >() );
  case KnownTypes::UInt32:  return XMLSupport::ToString( GetValue< uint32_t >() );
  case KnownTypes::Int64:   return XMLSupport::ToString( GetValue< int64_t  >() );
  case KnownTypes::UInt64:  return XMLSupport::ToString( GetValue< uint64_t >() );
  case KnownTypes::Float:   return XMLSupport::ToString( GetValue< float    >() );
  case KnownTypes::Double:  return XMLSupport::ToString( GetValue< double   >() );
  default:                  throw InternalErrorException("Unexpected constant data type!");
  }
}

AST::BaseClasses::TypeInfo AST::Expressions::Constant::GetResultType() const
{
  BaseClasses::TypeInfo ResultType;

  ResultType.SetConst(true);
  ResultType.SetPointer(false);
  ResultType.SetType(GetValueType());

  return ResultType;
}


// Implementation of class AST::Expressions::Identifier
AST::Expressions::IdentifierPtr AST::Expressions::Identifier::Create(string strName)
{
  IdentifierPtr spIdentifier = AST::CreateNode<Identifier>();

  spIdentifier->SetName( strName );

  return spIdentifier;
}

string AST::Expressions::Identifier::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["name"] = GetName();

  return XMLSupport::CreateXmlTag(cszIntend, "Identifier", mapAttributes);
}

AST::BaseClasses::TypeInfo AST::Expressions::Identifier::GetResultType() const
{
  BaseClasses::VariableInfoConstPtr spVariableInfo = LookupVariableInfo();

  if (spVariableInfo)
  {
    return spVariableInfo->GetTypeInfo();
  }
  else
  {
    return BaseClasses::TypeInfo();
  }
}

bool AST::Expressions::Identifier::IsVectorized()
{
  BaseClasses::VariableInfoPtr spVariableInfo = LookupVariableInfo();

  if (spVariableInfo)
  {
    return spVariableInfo->GetVectorize();
  }
  else
  {
    return false;
  }
}

AST::BaseClasses::VariableInfoPtr AST::Expressions::Identifier::LookupVariableInfo()
{
  BaseClasses::NodePtr spParent = GetThis();

  while (spParent)
  {
    if (spParent->IsType<AST::IVariableContainer>())
    {
      return spParent->CastToType< AST::IVariableContainer >()->GetVariableInfo(GetName());
    }
    else
    {
      spParent = spParent->GetParent();
    }
  }

  return nullptr;
}


// Implementation of class AST::Expressions::MemoryAccess
AST::Expressions::MemoryAccessPtr AST::Expressions::MemoryAccess::Create(ExpressionPtr spMemoryReference, ExpressionPtr spIndexExpression)
{
  MemoryAccessPtr spNewMemAccess = AST::CreateNode< MemoryAccess >();

  spNewMemAccess->SetMemoryReference(spMemoryReference);
  spNewMemAccess->SetIndexExpression(spIndexExpression);

  return spNewMemAccess;
}

string AST::Expressions::MemoryAccess::DumpToXML(const size_t cszIntend) const
{
  string strXmlString  = _DumpResultTypeToXML(cszIntend + 2);
  strXmlString        += XMLSupport::CreateXmlTag( cszIntend + 2, "MemoryRef", _DumpChildToXml(GetMemoryReference(), cszIntend + 4) );
  strXmlString        += XMLSupport::CreateXmlTag( cszIntend + 2, "Index",     _DumpChildToXml(GetIndexExpression(), cszIntend + 4) );

  return XMLSupport::CreateXmlTag(cszIntend, "MemoryAccess", strXmlString);
}

AST::BaseClasses::ExpressionPtr AST::Expressions::MemoryAccess::GetSubExpression(IndexType SubExprIndex)
{
  switch (SubExprIndex)
  {
  case 0:   return GetMemoryReference();
  case 1:   return GetIndexExpression();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

AST::BaseClasses::TypeInfo AST::Expressions::MemoryAccess::GetResultType() const
{
  if (GetMemoryReference())
  {
    return GetMemoryReference()->GetResultType().CreateDereferencedType();
  }
  else
  {
    return BaseClasses::TypeInfo();
  }
}

void AST::Expressions::MemoryAccess::SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr)
{
  switch (SubExprIndex)
  {
  case 0:   SetMemoryReference(spSubExpr);  break;
  case 1:   SetIndexExpression(spSubExpr);  break;
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::Expressions::UnaryExpression
string AST::Expressions::UnaryExpression::_DumpSubExpressionToXML(const size_t cszIntend) const
{
  string strXmlString  = _DumpResultTypeToXML(cszIntend);
  strXmlString        += XMLSupport::CreateXmlTag( cszIntend, "SubExpression", _DumpChildToXml(GetSubExpression(), cszIntend + 2) );
  return strXmlString;
}

AST::BaseClasses::ExpressionPtr AST::Expressions::UnaryExpression::GetSubExpression(IndexType SubExprIndex)
{
  switch (SubExprIndex)
  {
  case 0:   return GetSubExpression();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

void AST::Expressions::UnaryExpression::SetSubExpression(IndexType SubExprIndex, BaseClasses::ExpressionPtr spSubExpr)
{
  switch (SubExprIndex)
  {
  case 0:   SetSubExpression(spSubExpr);  break;
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::Expressions::Conversion
AST::Expressions::ConversionPtr AST::Expressions::Conversion::Create(const BaseClasses::TypeInfo &crConvertType, BaseClasses::ExpressionPtr spSubExpression, bool bExplicit)
{
  ConversionPtr spNewConversion = AST::CreateNode<Conversion>();

  spNewConversion->SetConvertType(crConvertType);
  spNewConversion->SetExplicit(bExplicit);
  spNewConversion->SetSubExpression(spSubExpression);

  return spNewConversion;
}

string AST::Expressions::Conversion::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["explicit"] = XMLSupport::ToString( GetExplicit() );

  return XMLSupport::CreateXmlTag(cszIntend, "Conversion", _DumpSubExpressionToXML(cszIntend + 2), mapAttributes);
}


// Implementation of class AST::Expressions::Parenthesis
AST::Expressions::ParenthesisPtr AST::Expressions::Parenthesis::Create(BaseClasses::ExpressionPtr spSubExpression)
{
  ParenthesisPtr spNewParenthesis = AST::CreateNode<Parenthesis>();

  spNewParenthesis->SetSubExpression(spSubExpression);

  return spNewParenthesis;
}

string AST::Expressions::Parenthesis::DumpToXML(const size_t cszIntend) const
{
  return XMLSupport::CreateXmlTag(cszIntend, "Parenthesis", _DumpSubExpressionToXML(cszIntend + 2));
}

AST::BaseClasses::TypeInfo AST::Expressions::Parenthesis::GetResultType() const
{
  if (GetSubExpression())
  {
    return GetSubExpression()->GetResultType();
  }
  else
  {
    return BaseClasses::TypeInfo();
  }
}


// Implementation of class AST::Expressions::UnaryOperator
AST::Expressions::UnaryOperatorPtr AST::Expressions::UnaryOperator::Create(UnaryOperatorType eType, BaseClasses::ExpressionPtr spSubExpression)
{
  UnaryOperatorPtr spNewUnaryOp = AST::CreateNode<UnaryOperator>();

  spNewUnaryOp->SetOperatorType(eType);
  spNewUnaryOp->SetSubExpression(spSubExpression);

  return spNewUnaryOp;
}

string AST::Expressions::UnaryOperator::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;
  mapAttributes["type"] = GetOperatorTypeString( GetOperatorType() );
  return XMLSupport::CreateXmlTag(cszIntend, "UnaryOperator", _DumpSubExpressionToXML(cszIntend + 2), mapAttributes);
}

string AST::Expressions::UnaryOperator::GetOperatorTypeString(UnaryOperatorType eType)
{
  switch (eType)
  {
  case UnaryOperatorType::AddressOf:        return "AddressOf";
  case UnaryOperatorType::BitwiseNot:       return "BitwiseNot";
  case UnaryOperatorType::LogicalNot:       return "LogicalNot";
  case UnaryOperatorType::Minus:            return "Minus";
  case UnaryOperatorType::Plus:             return "Plus";
  case UnaryOperatorType::PostDecrement:    return "PostDecrement";
  case UnaryOperatorType::PostIncrement:    return "PostIncrement";
  case UnaryOperatorType::PreDecrement:     return "PreDecrement";
  case UnaryOperatorType::PreIncrement:     return "PreIncrement";
  default:                                  throw InternalErrorException("Unknown unary operator type!");
  }
}

AST::BaseClasses::TypeInfo AST::Expressions::UnaryOperator::GetResultType() const
{
  if (! GetSubExpression())
  {
    return BaseClasses::TypeInfo(BaseClasses::TypeInfo::KnownTypes::Unknown);
  }

  if (GetOperatorType() == UnaryOperatorType::AddressOf)
  {
    return GetSubExpression()->GetResultType().CreatePointerType();
  }
  else if (GetOperatorType() == UnaryOperatorType::LogicalNot)
  {
    return BaseClasses::TypeInfo(BaseClasses::TypeInfo::KnownTypes::Bool, true, false);
  }
  else
  {
    BaseClasses::TypeInfo ReturnType = GetSubExpression()->GetResultType();

    if (! ReturnType.GetPointer())
    {
      ReturnType.SetConst(true);
    }

    return ReturnType;
  }
}


// Implementation of class AST::Expressions::BinaryOperator
string AST::Expressions::BinaryOperator::_DumpSubExpressionsToXML(const size_t cszIntend) const
{
  string strXmlString = _DumpResultTypeToXML(cszIntend);

  strXmlString += XMLSupport::CreateXmlTag( cszIntend, "LHS", _DumpChildToXml(GetLHS(), cszIntend + 2) );
  strXmlString += XMLSupport::CreateXmlTag( cszIntend, "RHS", _DumpChildToXml(GetRHS(), cszIntend + 2) );

  return strXmlString;
}

AST::BaseClasses::ExpressionPtr AST::Expressions::BinaryOperator::GetSubExpression(IndexType SubExprIndex)
{
  switch (SubExprIndex)
  {
  case 0:   return GetLHS();
  case 1:   return GetRHS();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

void AST::Expressions::BinaryOperator::SetSubExpression(IndexType SubExprIndex, BaseClasses::ExpressionPtr spSubExpr)
{
  switch (SubExprIndex)
  {
  case 0:   SetLHS(spSubExpr);  break;
  case 1:   SetRHS(spSubExpr);  break;
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::Expressions::ArithmeticOperator
AST::Expressions::ArithmeticOperatorPtr AST::Expressions::ArithmeticOperator::Create(ArithmeticOperatorType eOpType, ExpressionPtr spLHS, ExpressionPtr spRHS)
{
  ArithmeticOperatorPtr spArithmeticOp = AST::CreateNode<ArithmeticOperator>();

  spArithmeticOp->SetOperatorType(eOpType);
  spArithmeticOp->SetLHS(spLHS);
  spArithmeticOp->SetRHS(spRHS);

  return spArithmeticOp;
}

string AST::Expressions::ArithmeticOperator::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["type"] = GetOperatorTypeString(_eOpType);

  return XMLSupport::CreateXmlTag(cszIntend, "ArithmeticOperator", _DumpSubExpressionsToXML(cszIntend + 2), mapAttributes);
}

string AST::Expressions::ArithmeticOperator::GetOperatorTypeString(ArithmeticOperatorType eType)
{
  switch (eType)
  {
  case ArithmeticOperatorType::Add:           return "Add";
  case ArithmeticOperatorType::BitwiseAnd:    return "BitwiseAnd";
  case ArithmeticOperatorType::BitwiseOr:     return "BitwiseOr";
  case ArithmeticOperatorType::BitwiseXOr:    return "BitwiseXOr";
  case ArithmeticOperatorType::Divide:        return "Divide";
  case ArithmeticOperatorType::Modulo:        return "Modulo";
  case ArithmeticOperatorType::Multiply:      return "Multiply";
  case ArithmeticOperatorType::ShiftLeft:     return "ShiftLeft";
  case ArithmeticOperatorType::ShiftRight:    return "ShiftRight";
  case ArithmeticOperatorType::Subtract:      return "Subtract";
  default:                                    throw InternalErrorException("Unknown arithmetic operator type!");
  }
}

AST::BaseClasses::TypeInfo AST::Expressions::ArithmeticOperator::GetResultType() const
{
  typedef BaseClasses::TypeInfo   TypeInfo;

  if ( GetLHS() && GetRHS() )     // Check if both children are set
  {
    TypeInfo TypeLHS = GetLHS()->GetResultType();
    TypeInfo TypeRHS = GetRHS()->GetResultType();

    if ( (TypeLHS.GetType() == TypeInfo::KnownTypes::Unknown) || (TypeRHS.GetType() == TypeInfo::KnownTypes::Unknown) )
    {
      // Cannot do arithmetic with unknown types => Return type is unknown
      return TypeInfo();
    }
    else if (! TypeRHS.IsSingleValue())
    {
      // Expected single value for right operand => Unknown type
      return TypeInfo();
    }
    else if (TypeLHS.IsArray())
    {
      // Array arithmetic is forbidden => Unknown type
      return TypeInfo();
    }
    else if (TypeLHS.GetPointer())
    {
      // Pointer arithmetic is only allowed for the "Add" and "Subtract" operator
      if ( (GetOperatorType() == ArithmeticOperatorType::Add) || (GetOperatorType() == ArithmeticOperatorType::Subtract) )
      {
        return TypeLHS;
      }
      else
      {
        return TypeInfo();
      }
    }
    else
    {
      // Both operands are single values => Return the promoted type
      TypeInfo ReturnType;

      ReturnType.SetConst(true);
      ReturnType.SetPointer(false);
      ReturnType.SetType( BaseClasses::TypeInfo::GetPromotedType(TypeLHS.GetType(), TypeRHS.GetType()) );

      return ReturnType;
    }
  }
  else                            // Incomplete statement => Return type cannot be created
  {
    return TypeInfo();
  }
}


// Implementation of class AST::Expressions::AssignmentOperator
AST::Expressions::AssignmentOperatorPtr AST::Expressions::AssignmentOperator::Create(ExpressionPtr spLHS, ExpressionPtr spRHS, IdentifierPtr spMask)
{
  AssignmentOperatorPtr spAssignment = AST::CreateNode<AssignmentOperator>();

  spAssignment->SetLHS(spLHS);
  spAssignment->SetRHS(spRHS);
  spAssignment->SetMask(spMask);

  return spAssignment;
}

string AST::Expressions::AssignmentOperator::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  string strXmlString = _DumpSubExpressionsToXML(cszIntend + 2);

  if (IsMasked())
  {
    mapAttributes["masked"] = XMLSupport::ToString( IsMasked() );

    strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Mask", _DumpChildToXml(GetMask(), cszIntend + 4) );
  }

  return XMLSupport::CreateXmlTag( cszIntend, "AssignmentOperator", strXmlString, mapAttributes );
}

AST::BaseClasses::TypeInfo AST::Expressions::AssignmentOperator::GetResultType() const
{
  if (GetLHS())
  {
    BaseClasses::TypeInfo ResultType(GetLHS()->GetResultType());

    if ( (! ResultType.GetPointer()) && (! ResultType.IsArray()) )
    {
      ResultType.SetConst(true);
    }

    return ResultType;
  }
  else
  {
    return BaseClasses::TypeInfo();
  }
}

AST::BaseClasses::ExpressionPtr AST::Expressions::AssignmentOperator::GetSubExpression(IndexType SubExprIndex)
{
  const IndexType ciBaseSubExprCount = BaseType::GetSubExpressionCount();

  if (SubExprIndex < ciBaseSubExprCount)
  {
    return BaseType::GetSubExpression(SubExprIndex);
  }
  else if (SubExprIndex == ciBaseSubExprCount)
  {
    return GetMask();
  }
  else
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
}

AST::IndexType AST::Expressions::AssignmentOperator::GetSubExpressionCount() const
{
  const IndexType ciBaseSubExprCount = BaseType::GetSubExpressionCount();

  return IsMasked() ? (ciBaseSubExprCount + 1) : ciBaseSubExprCount;
}

void AST::Expressions::AssignmentOperator::SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr)
{
  const IndexType ciBaseSubExprCount = BaseType::GetSubExpressionCount();

  if (SubExprIndex < ciBaseSubExprCount)
  {
    BaseType::SetSubExpression(SubExprIndex, spSubExpr);
  }
  else if (SubExprIndex == ciBaseSubExprCount)
  {
    IdentifierPtr spNewMask = nullptr;

    if (spSubExpr)
    {
      if (spSubExpr->IsType<Identifier>())
      {
        SetMask( spSubExpr->CastToType<Identifier>() );
      }
      else
      {
        throw InternalErrorException("Expected an identifier expression for the mask parameter!");
      }
    }

    SetMask(spNewMask);
  }
  else
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::Expressions::RelationalOperator
AST::Expressions::RelationalOperatorPtr AST::Expressions::RelationalOperator::Create(RelationalOperatorType eOpType, ExpressionPtr spLHS, ExpressionPtr spRHS)
{
  RelationalOperatorPtr spNewRelOp = AST::CreateNode< RelationalOperator >();

  spNewRelOp->SetOperatorType(eOpType);
  spNewRelOp->SetLHS(spLHS);
  spNewRelOp->SetRHS(spRHS);

  return spNewRelOp;
}

string AST::Expressions::RelationalOperator::DumpToXML(const size_t cszIntend) const
{
  string strXmlString("");
  
  strXmlString  += XMLSupport::CreateXmlTag(cszIntend + 2, "ComparisonType", GetComparisonType().DumpToXML(cszIntend + 4));
  strXmlString  += _DumpSubExpressionsToXML(cszIntend + 2);

  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["type"] = GetOperatorTypeString(_eOpType);

  return XMLSupport::CreateXmlTag( cszIntend, "RelationalOperator", strXmlString, mapAttributes );
}

AST::BaseClasses::TypeInfo AST::Expressions::RelationalOperator::GetComparisonType() const
{
  typedef BaseClasses::TypeInfo::KnownTypes  KnownTypes;

  if ((! GetLHS()) || (! GetRHS()))
  {
    return BaseClasses::TypeInfo();
  }

  BaseClasses::TypeInfo TypeLHS = GetLHS()->GetResultType();
  BaseClasses::TypeInfo TypeRHS = GetRHS()->GetResultType();

  if ( (TypeLHS.GetType() == KnownTypes::Unknown) || (TypeRHS.GetType() == KnownTypes::Unknown) )
  {
    return BaseClasses::TypeInfo();
  }
  else if ( (! TypeLHS.IsSingleValue()) || (! TypeRHS.IsSingleValue()) )
  {
    return BaseClasses::TypeInfo();
  }

  KnownTypes eCompType = BaseClasses::TypeInfo::GetPromotedType(TypeLHS.GetType(), TypeRHS.GetType());

  return BaseClasses::TypeInfo(eCompType, true, false);
}

string AST::Expressions::RelationalOperator::GetOperatorTypeString(RelationalOperatorType eType)
{
  switch (eType)
  {
  case RelationalOperatorType::Equal:         return "Equal";
  case RelationalOperatorType::Greater:       return "Greater";
  case RelationalOperatorType::GreaterEqual:  return "GreaterEqual";
  case RelationalOperatorType::Less:          return "Less";
  case RelationalOperatorType::LessEqual:     return "LessEqual";
  case RelationalOperatorType::LogicalAnd:    return "LogicalAnd";
  case RelationalOperatorType::LogicalOr:     return "LogicalOr";
  case RelationalOperatorType::NotEqual:      return "NotEqual";
  default:                                    throw InternalErrorException("Unknown relational operator type!");
  }
}

AST::BaseClasses::TypeInfo AST::Expressions::RelationalOperator::GetResultType() const
{
  return BaseClasses::TypeInfo( BaseClasses::TypeInfo::KnownTypes::Bool, true, false );
}


// Implementation of class AST::Expressions::FunctionCall
void AST::Expressions::FunctionCall::AddCallParameter(ExpressionPtr spCallParam)
{
  CHECK_NULL_POINTER(spCallParam);

  _SetParentToChild(spCallParam);
  _vecCallParams.push_back(spCallParam);
}

AST::Expressions::FunctionCallPtr AST::Expressions::FunctionCall::Create(std::string strFunctionName, const BaseClasses::TypeInfo &crReturnType)
{
  FunctionCallPtr spNewFunctionCall = AST::CreateNode<FunctionCall>();

  spNewFunctionCall->SetName(strFunctionName);
  spNewFunctionCall->SetReturnType(crReturnType);

  return spNewFunctionCall;
}

string AST::Expressions::FunctionCall::DumpToXML(const size_t cszIntend) const
{
  // Dump return type
  string strXmlString = _DumpResultTypeToXML(cszIntend + 2);

  // Dump call parameters
  for (IndexType i = 0; i < GetCallParameterCount(); ++i)
  {
    XMLSupport::AttributesMapType mapParamAttributes;

    mapParamAttributes["index"] = XMLSupport::ToString(static_cast<unsigned int>(i));

    ExpressionConstPtr spParam = GetCallParameter(i);
    strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "Param", spParam->DumpToXML(cszIntend + 4), mapParamAttributes);
  }

  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["name"] = GetName();

  return XMLSupport::CreateXmlTag(cszIntend, "FunctionCall", strXmlString, mapAttributes);
}

AST::BaseClasses::ExpressionPtr AST::Expressions::FunctionCall::GetCallParameter(IndexType CallParamIndex)
{
  if (CallParamIndex >= GetSubExpressionCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
  else
  {
    return _vecCallParams[ CallParamIndex ];
  }
}

void AST::Expressions::FunctionCall::SetCallParameter(IndexType CallParamIndex, ExpressionPtr spCallParam)
{
  if (CallParamIndex >= GetSubExpressionCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
  else
  {
    _SetParentToChild(spCallParam);
    _vecCallParams[CallParamIndex] = spCallParam;
  }
}



/*************************/
/***   VectorSupport   ***/
/*************************/

// Implementation of class AST::VectorSupport::BroadCast
AST::VectorSupport::BroadCastPtr AST::VectorSupport::BroadCast::Create(ExpressionPtr spSubExpression)
{
  BroadCastPtr spBroadCast = AST::CreateNode<BroadCast>();

  spBroadCast->SetSubExpression(spSubExpression);

  return spBroadCast;
}

string AST::VectorSupport::BroadCast::DumpToXML(const size_t cszIntend) const
{
  string strXmlString("");

  strXmlString += _DumpResultTypeToXML(cszIntend + 2);
  strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "SubExpression", _DumpChildToXml(GetSubExpression(), cszIntend + 4));

  return XMLSupport::CreateXmlTag(cszIntend, "BroadCast", strXmlString);
}

AST::BaseClasses::TypeInfo AST::VectorSupport::BroadCast::GetResultType() const
{
  if (GetSubExpression())
  {
    return GetSubExpression()->GetResultType();
  }
  else
  {
    return BaseClasses::TypeInfo();
  }
}

AST::BaseClasses::ExpressionPtr AST::VectorSupport::BroadCast::GetSubExpression(IndexType SubExprIndex)
{
  switch (SubExprIndex)
  {
  case 0:   return GetSubExpression();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

void AST::VectorSupport::BroadCast::SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr)
{
  switch (SubExprIndex)
  {
  case 0:   SetSubExpression(spSubExpr);  break;
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::VectorSupport::CheckActiveElements
AST::VectorSupport::CheckActiveElementsPtr AST::VectorSupport::CheckActiveElements::Create(CheckType eCheckType, ExpressionPtr spSubExpression)
{
  CheckActiveElementsPtr spCheckElements = AST::CreateNode<CheckActiveElements>();

  spCheckElements->SetCheckType(eCheckType);
  spCheckElements->SetSubExpression(spSubExpression);

  return spCheckElements;
}

string AST::VectorSupport::CheckActiveElements::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["check_type"] = GetCheckTypeString( GetCheckType() );

  string strXmlString("");

  strXmlString += _DumpResultTypeToXML(cszIntend + 2);
  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "SubExpression", _DumpChildToXml(GetSubExpression(), cszIntend + 4) );

  return XMLSupport::CreateXmlTag(cszIntend, "CheckActiveElements", strXmlString, mapAttributes);
}

string AST::VectorSupport::CheckActiveElements::GetCheckTypeString(CheckType eType)
{
  switch (eType)
  {
  case CheckType::All:    return "All";
  case CheckType::Any:    return "Any";
  case CheckType::None:   return "None";
  default:                throw InternalErrorException("Unknown check type!");
  }
}

AST::BaseClasses::ExpressionPtr AST::VectorSupport::CheckActiveElements::GetSubExpression(IndexType SubExprIndex)
{
  switch (SubExprIndex)
  {
  case 0:   return GetSubExpression();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

void AST::VectorSupport::CheckActiveElements::SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr)
{
  switch (SubExprIndex)
  {
  case 0:   return SetSubExpression(spSubExpr);   break;
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}


// Implementation of class AST::VectorSupport::VectorIndex
AST::VectorSupport::VectorIndexPtr AST::VectorSupport::VectorIndex::Create(KnownTypes eType)
{
  VectorIndexPtr spNewVecIndex = AST::CreateNode<VectorIndex>();

  spNewVecIndex->SetType(eType);

  return spNewVecIndex;
}

string AST::VectorSupport::VectorIndex::DumpToXML(const size_t cszIntend) const
{
  return XMLSupport::CreateXmlTag(cszIntend, "VectorIndex", _DumpResultTypeToXML(cszIntend + 2));
}

AST::BaseClasses::TypeInfo AST::VectorSupport::VectorIndex::GetResultType() const
{
  return AST::BaseClasses::TypeInfo(GetType(), true, false);
}



/*************************/
/***   Other classes   ***/
/*************************/

// Implementation of class AST::Scope
AST::IVariableContainerPtr AST::Scope::_GetParentVariableContainer()
{
  BaseClasses::NodePtr spParent = GetThis();

  while (true)
  {
    spParent = spParent->GetParent();

    if (! spParent)
    {
      break;
    }
    else if (spParent->IsType<AST::IVariableContainer>())
    {
      return spParent->CastToType<AST::IVariableContainer>();
    }
  }

  return nullptr;
}

void AST::Scope::AddChild(NodePtr spChild)
{
  CHECK_NULL_POINTER(spChild);

  _SetParentToChild(spChild);
  _Children.push_back(spChild);
}

void AST::Scope::AddVariable(BaseClasses::VariableInfoPtr spVariableInfo)
{
  CHECK_NULL_POINTER(spVariableInfo);

  AST::IVariableContainerPtr spParentVarContainer = _GetParentVariableContainer();
  CHECK_NULL_POINTER(spParentVarContainer);

  spParentVarContainer->AddVariable(spVariableInfo);
}

void AST::Scope::AddVariableDeclaration(BaseClasses::VariableInfoPtr spVariableInfo)
{
  CHECK_NULL_POINTER(spVariableInfo);

  _setDeclaredVariables.insert( spVariableInfo->GetName() );

  AddVariable(spVariableInfo);
}

AST::ScopePtr AST::Scope::Create()
{
  return AST::CreateNode<Scope>();
}

string AST::Scope::DumpToXML(const size_t cszIntend) const
{
  string strXmlString("");

  // Dump declared variables
  if (! _setDeclaredVariables.empty())
  {
    string strDeclarations("");

    for (auto itVar : _setDeclaredVariables)
    {
      XMLSupport::AttributesMapType mapDeclAttributes;

      mapDeclAttributes["name"] = itVar;

      strDeclarations += XMLSupport::CreateXmlTag(cszIntend + 4, "Variable", "", mapDeclAttributes);
    }

    strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "DeclaredVariables", strDeclarations);
  }

  // Dump children
  for (auto itNode : _Children)
  {
    strXmlString += itNode->DumpToXML(cszIntend + 2);
  }

  return XMLSupport::CreateXmlTag(cszIntend, "Scope", strXmlString);
}

AST::BaseClasses::NodePtr AST::Scope::GetChild(IndexType ChildIndex)
{
  if (ChildIndex >= GetChildCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
  else
  {
    return _Children[ChildIndex];
  }
}

AST::IndexType AST::Scope::GetChildIndex(NodePtr spChildNode)
{
  CHECK_NULL_POINTER(spChildNode);

  for (IndexType iChildIndex = static_cast<IndexType>(0); iChildIndex < GetChildCount(); ++iChildIndex)
  {
    if (spChildNode == GetChild(iChildIndex))
    {
      return iChildIndex;
    }
  }

  throw InternalErrorException("Could not find the specified child!");
}

AST::Scope::VariableDeclarationVectorType AST::Scope::GetVariableDeclarations() const
{
  VariableDeclarationVectorType vecDeclarations;

  vecDeclarations.reserve( _setDeclaredVariables.size() );

  for (auto itVariable = _setDeclaredVariables.begin(); itVariable != _setDeclaredVariables.end(); itVariable++)
  {
    Expressions::IdentifierPtr spVariable = Expressions::Identifier::Create( *itVariable );

    _SetParentToChild(spVariable);
    vecDeclarations.push_back(spVariable);
  }

  return std::move( vecDeclarations );
}

AST::BaseClasses::VariableInfoPtr AST::Scope::GetVariableInfo(std::string strVariableName)
{
  BaseClasses::NodePtr spParent = GetThis();

  while (true)
  {
    spParent = spParent->GetParent();
    CHECK_NULL_POINTER(spParent);

    if (spParent->IsType<AST::IVariableContainer>())
    {
      return spParent->CastToType<AST::IVariableContainer>()->GetVariableInfo(strVariableName);
      break;
    }
  }

  return nullptr;
}

void AST::Scope::ImportScope(ScopePtr spOtherScope)
{
  CHECK_NULL_POINTER(spOtherScope);

  ImportVariableDeclarations(spOtherScope);

  for (auto itChild : spOtherScope->_Children)
  {
    AddChild(itChild);
  }

  spOtherScope->_Children.clear();
}

void AST::Scope::ImportVariableDeclarations(ScopePtr spOtherScope)
{
  CHECK_NULL_POINTER( spOtherScope );

  for (auto itDecl : spOtherScope->_setDeclaredVariables)
  {
    _setDeclaredVariables.insert( itDecl );
  }

  spOtherScope->_setDeclaredVariables.clear();
}

void AST::Scope::InsertChild(IndexType ChildIndex, NodePtr spChildNode)
{
  CHECK_NULL_POINTER(spChildNode);

  if (ChildIndex > GetChildCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }

  _SetParentToChild(spChildNode);
  _Children.insert(_Children.begin() + ChildIndex, spChildNode);
}

bool AST::Scope::IsVariableUsed(const std::string &crstrVariableName) const
{
  AST::IVariableContainerConstPtr spParentVariableContainer = _GetParentVariableContainer();
  CHECK_NULL_POINTER(spParentVariableContainer);

  return spParentVariableContainer->IsVariableUsed(crstrVariableName);
}

void AST::Scope::RemoveChild(IndexType ChildIndex)
{
  if (ChildIndex >= GetChildCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
  else
  {
    _Children.erase(_Children.begin() + ChildIndex);
  }
}

void AST::Scope::SetChild(IndexType ChildIndex, NodePtr spChildNode)
{
  if (ChildIndex >= GetChildCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }
  else
  {
    _SetParentToChild(spChildNode);
    _Children[ChildIndex] = spChildNode;
  }
}


// Implementation of class AST::FunctionDeclaration
void AST::FunctionDeclaration::AddParameter(BaseClasses::VariableInfoPtr spVariableInfo)
{
  CHECK_NULL_POINTER(spVariableInfo);

  AddVariable(spVariableInfo);

  Expressions::IdentifierPtr spParameter = Expressions::Identifier::Create( spVariableInfo->GetName() );

  _SetParentToChild(spParameter);
  _Parameters.push_back(spParameter);
}

void AST::FunctionDeclaration::AddVariable(BaseClasses::VariableInfoPtr spVariableInfo)
{
  CHECK_NULL_POINTER(spVariableInfo);

  string strVariableName = spVariableInfo->GetName();

  if (IsVariableUsed(strVariableName))
  {
    throw ASTExceptions::DuplicateVariableName(strVariableName);
  }

  _mapKnownVariables[strVariableName] = spVariableInfo;
}

AST::FunctionDeclarationPtr AST::FunctionDeclaration::Create(string strFunctionName)
{
  FunctionDeclarationPtr spNewFunction = AST::CreateNode<FunctionDeclaration>();

  spNewFunction->SetName(strFunctionName);

  // Initialize function body
  spNewFunction->GetBody();

  return spNewFunction;
}

string AST::FunctionDeclaration::DumpToXML(const size_t cszIntend) const
{
  XMLSupport::AttributesMapType mapAttributes;

  mapAttributes["name"] = GetName();

  string strXmlString("");

  // Dump known variables
  {
    string strXmlVariables("");

    for (auto itVariable : _mapKnownVariables)
    {
      strXmlVariables += itVariable.second->DumpToXML(cszIntend + 4);
    }

    strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "KnownVariables", strXmlVariables);
  }

  // Dump parameters
  {
    string strXmlParams("");

    for (auto itParameter : _Parameters)
    {
      strXmlParams += itParameter->DumpToXML(cszIntend + 4);
    }

    strXmlString += XMLSupport::CreateXmlTag(cszIntend + 2, "Parameters", strXmlParams);
  }

  // Dump body
  strXmlString += XMLSupport::CreateXmlTag( cszIntend + 2, "Body", _DumpChildToXml(GetBody(), cszIntend + 4) );

  return XMLSupport::CreateXmlTag(cszIntend, "FunctionDeclaration", strXmlString, mapAttributes);
}

AST::ScopePtr AST::FunctionDeclaration::GetBody()
{
  if (!_spBody)
  {
    _SetChildPtr(_spBody, Scope::Create());
  }

  return _spBody;
}

AST::ScopeConstPtr AST::FunctionDeclaration::GetBody() const
{
  CHECK_NULL_POINTER(_spBody);

  return _spBody;
}

AST::BaseClasses::NodePtr AST::FunctionDeclaration::GetChild(IndexType ChildIndex)
{
  switch (ChildIndex)
  {
  case 0:   return GetBody();
  default:  throw ASTExceptions::ChildIndexOutOfRange();
  }
}

vector< string > AST::FunctionDeclaration::GetKnownVariableNames() const
{
  vector< string > vecVariableNames;

  for (auto itKnownVariable : _mapKnownVariables)
  {
    vecVariableNames.push_back( itKnownVariable.first );
  }

  return move( vecVariableNames );
}

AST::Expressions::IdentifierPtr AST::FunctionDeclaration::GetParameter(IndexType iParamIndex)
{
  if (iParamIndex >= GetParameterCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }

  return _Parameters[ iParamIndex ];
}

AST::BaseClasses::VariableInfoPtr AST::FunctionDeclaration::GetVariableInfo(std::string strVariableName)
{
  auto itVariableEntry = _mapKnownVariables.find(strVariableName);

  if (itVariableEntry != _mapKnownVariables.end())
  {
    return itVariableEntry->second;
  }
  else
  {
    return nullptr;
  }
}

bool AST::FunctionDeclaration::IsVariableUsed(const std::string &crstrVariableName) const
{
  auto itVariable = _mapKnownVariables.find(crstrVariableName);

  return (itVariable != _mapKnownVariables.end());
}

void AST::FunctionDeclaration::SetParameter(IndexType iParamIndex, BaseClasses::VariableInfoPtr spVariableInfo)
{
  if (iParamIndex >= GetParameterCount())
  {
    throw ASTExceptions::ChildIndexOutOfRange();
  }

  string strNewParamName = spVariableInfo->GetName();
  if (IsVariableUsed(strNewParamName))
  {
    throw ASTExceptions::DuplicateVariableName(strNewParamName);
  }

  // Erase the old parameter
  Expressions::IdentifierPtr spOldParam = GetParameter(iParamIndex);
  _mapKnownVariables.erase( _mapKnownVariables.find(spOldParam->GetName()) );

  // Set the new parameter
  Expressions::IdentifierPtr spNewParameter = Expressions::Identifier::Create( strNewParamName );

  _SetParentToChild(spNewParameter);
  _Parameters[iParamIndex] = spNewParameter;

  _mapKnownVariables[strNewParamName] = spVariableInfo;
}


// vim: set ts=2 sw=2 sts=2 et ai:

