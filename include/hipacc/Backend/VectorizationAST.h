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

//===--- VectorizationAST.h - Implements a vectorizable syntax tree. -----------------===//
//
// This file implements the internally used vectorizable syntax tree (a simplification to clang's AST)
//
//===---------------------------------------------------------------------------------===//

#ifndef _BACKEND_VECTORIZATION_AST_H_
#define _BACKEND_VECTORIZATION_AST_H_

#include "llvm/Support/Casting.h"
#include "BackendExceptions.h"
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <type_traits>

namespace clang
{
namespace hipacc
{
namespace Backend
{
namespace Vectorization
{
  class ASTExceptions
  {
  public:

    class ChildIndexOutOfRange : public InternalErrorException
    {
    private:

      typedef InternalErrorException  BaseType;   //!< The base type of this class.

    public:

      inline ChildIndexOutOfRange() : BaseType("The index for the child node is out of range!")  {}
    };

    class DuplicateVariableName : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline DuplicateVariableName(std::string strVarName) : BaseType(std::string("The variable name \"") + strVarName + std::string("\" is not unique!"))  {}
    };

    class NonDereferencableType : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline NonDereferencableType() : BaseType("The specified type cannot be dereferenced!")  {}
    };

    class UnknownExpressionClass : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline UnknownExpressionClass(std::string strExprClassName) : BaseType( std::string( "The expression class \"") + strExprClassName + std::string("\" is unknown!") )  {}
    };

    class UnknownStatementClass : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline UnknownStatementClass(std::string strStmtClassName) : BaseType(std::string("The statement class \"") + strStmtClassName + std::string("\" is unknown!"))  {}
    };
  };


  class AST
  {
  // Forward declarations and type definitions
  public:

    typedef size_t    IndexType;


    class IVariableContainer;
    class FunctionDeclaration;
    class Scope;

    typedef std::shared_ptr< IVariableContainer >   IVariableContainerPtr;
    typedef std::shared_ptr< FunctionDeclaration >  FunctionDeclarationPtr;
    typedef std::shared_ptr< Scope >                ScopePtr;


  // Class definitions
  public:

    class ScopePosition final
    {
    private:

      ScopePtr    _spScope;
      IndexType   _ChildIndex;

    public:

      inline ScopePosition(ScopePtr spScope, IndexType ChildIndex) : _spScope(spScope), _ChildIndex(ChildIndex)   {}
      inline ScopePosition(const ScopePosition &crRVal)   { *this = crRVal; }
      inline ScopePosition& operator=(const ScopePosition &crRVal)
      {
        _spScope    = crRVal._spScope;
        _ChildIndex = crRVal._ChildIndex;

        return *this;
      }

      inline ScopePtr    GetScope()       { return _spScope; }
      inline IndexType   GetChildIndex()  { return _ChildIndex; }
    };


    class BaseClasses final
    {
    // Forward declarations and type definitions
    public:

      class ControlFlowStatement;
      class Node;
      class Expression;
      class VariableInfo;

      typedef std::shared_ptr< ControlFlowStatement >   ControlFlowStatementPtr;
      typedef std::shared_ptr< Node >                   NodePtr;
      typedef std::shared_ptr< Expression >             ExpressionPtr;
      typedef std::shared_ptr< VariableInfo >           VariableInfoPtr;


    public:

      class TypeInfo
      {
      public:

        typedef std::vector< size_t >    ArrayDimensionVectorType;

        enum class KnownTypes
        {
          Bool,
          Int8,
          UInt8,
          Int16,
          UInt16,
          Int32,
          UInt32,
          Int64,
          UInt64,
          Float,
          Double,
          Unknown
        };


      private:

        KnownTypes                _eType;
        bool                      _bIsConst;
        bool                      _bIsPointer;
        ArrayDimensionVectorType  _vecArrayDimensions;


      public:

        inline TypeInfo(KnownTypes eType = KnownTypes::Unknown, bool bIsConst = false, bool bIsPointer = false)
        {
          _eType      = eType;
          _bIsConst   = bIsConst;
          _bIsPointer = bIsPointer;
        }

        inline TypeInfo(const TypeInfo &crRVal)   { *this = crRVal; }
        TypeInfo& operator=(const TypeInfo &crRVal);


        TypeInfo CreateDereferencedType() const;
        TypeInfo CreatePointerType() const;


        inline ArrayDimensionVectorType&        GetArrayDimensions()        { return _vecArrayDimensions; }
        inline const ArrayDimensionVectorType&  GetArrayDimensions() const  { return _vecArrayDimensions; }

        inline bool GetConst() const          { return _bIsConst; }
        inline void SetConst(bool bIsConst)   { _bIsConst = bIsConst; }

        inline bool GetPointer() const            { return _bIsPointer; }
        inline void SetPointer(bool bIsPointer)   { _bIsPointer = bIsPointer; }

        inline KnownTypes GetType() const             { return _eType; }
        inline void       SetType(KnownTypes eType)   { _eType = eType; }

        inline bool IsArray() const           { return (!_vecArrayDimensions.empty()); }
        inline bool IsDereferencable() const  { return (IsArray() || GetPointer()); }
        inline bool IsSingleValue() const     { return (! IsDereferencable()); }


        bool IsEqual(const TypeInfo &crRVal, bool bIgnoreConstQualifier);

        std::string DumpToXML(const size_t cszIntend) const;


        inline bool operator==(const TypeInfo &crRVal)    { return IsEqual(crRVal, false); }


      public:

        static TypeInfo     CreateSizedIntegerType(size_t szTypeSize, bool bSigned);

        static KnownTypes   GetPromotedType(KnownTypes eTypeLHS, KnownTypes eTypeRHS);

        static size_t       GetTypeSize(KnownTypes eType);
        static std::string  GetTypeString(KnownTypes eType);

        static bool         IsSigned(KnownTypes eType);
      };

      class VariableInfo
      {
      private:

        std::string _strName;
        TypeInfo    _Type;
        bool        _bVectorize;

      public:

        static VariableInfoPtr Create(std::string strName, const TypeInfo &crTypeInfo, bool bVectorize = false);

        inline VariableInfo() : _strName(""), _bVectorize(false)    {}

        inline std::string  GetName() const               { return _strName; }
        inline void         SetName(std::string strName)  { _strName = strName; }

        inline TypeInfo&        GetTypeInfo()       { return _Type; }
        inline const TypeInfo&  GetTypeInfo() const { return _Type; }

        inline bool GetVectorize() const            { return _bVectorize; }
        inline void SetVectorize(bool bVectorize)   { _bVectorize = bVectorize; }

        std::string DumpToXML(const size_t cszIntend) const;
      };

      class Node
      {
      private:

        friend class AST;

      public:

        enum class NodeType
        {
          ControlFlowStatement,
          Expression,
          FunctionDeclaration,
          Scope,
          Value
        };


      private:

        const NodeType          _ceNodeType;
        std::weak_ptr< Node >   _wpParent;
        std::weak_ptr< Node >   _wpThis;


        inline void _SetParent(NodePtr spParent)        { _wpParent = spParent; }


      protected:

        inline Node(NodeType eType) : _ceNodeType(eType)    {}


        static std::string _DumpChildToXml(const NodePtr spChild, const size_t cszIntend);

        template < typename NodeClassPtr >
        inline void _SetChildPtr(NodeClassPtr &rDestinationPtr, const NodeClassPtr &crSourcePtr)
        {
          // If the child is set, remove its parent pointer
          if (rDestinationPtr)
          {
            // Only remove the parent of the child, if this node is the parent
            if (rDestinationPtr->GetParent() == GetThis())
            {
              rDestinationPtr->_SetParent(nullptr);
            }
          }

          // Set the child pointer and set the current node as its parent
          rDestinationPtr = crSourcePtr;
          _SetParentToChild(rDestinationPtr);
        }

        void _SetParentToChild(NodePtr spChild) const;


      public:

        virtual ~Node() {}


        IndexType GetHierarchyLevel() const;


        inline NodeType GetNodeType() const    { return _ceNodeType; }

        NodePtr               GetParent();
        inline const NodePtr  GetParent() const   { return const_cast<Node*>(this)->GetParent(); }

        ScopePosition GetScopePosition() const;

        inline NodePtr        GetThis()         { return _wpThis.lock(); }
        inline const NodePtr  GetThis() const   { return _wpThis.lock(); }

        inline bool IsLeafNode() const  { return (GetChildCount() == static_cast<IndexType>(0)); }


        template <class NodeClass>
        inline std::shared_ptr< NodeClass > CastToType()
        {
          if (! IsType<NodeClass>())
          {
            throw RuntimeErrorException("Invalid node cast type!");
          }

          return std::dynamic_pointer_cast< NodeClass >( GetThis() );
        }

        template <class NodeClass>
        inline bool IsType() const
        {
          static_assert( std::is_base_of< Node, NodeClass >::value, "All VAST nodes must be derived from class \"Node\"!" );

          return (dynamic_cast< const NodeClass* >(this) != nullptr);
        }


      public:

        virtual NodePtr     GetChild(IndexType ChildIndex) = 0;
        virtual IndexType   GetChildCount() const = 0;

        virtual std::string DumpToXML(const size_t cszIntend) const = 0;
      };

      class ControlFlowStatement : public Node
      {
      private:

        typedef Node    BaseType;

      public:

        enum class ControlFlowType
        {
          BranchingStatement,
          ConditionalBranch,
          Loop,
          LoopControlStatement,
          ReturnStatement
        };
        
      private:

        const ControlFlowType    _ceControlFlowType;

      protected:

        inline ControlFlowStatement(ControlFlowType eCtlFlowType) : BaseType(Node::NodeType::ControlFlowStatement), _ceControlFlowType(eCtlFlowType)  {}

      public:

        virtual ~ControlFlowStatement() {}

        virtual bool IsVectorized() const = 0;
      };

      class Expression : public Node
      {
      private:

        typedef Node    BaseType;

      public:

        enum class ExpressionType
        {
          BinaryOperator,
          FunctionCall,
          UnaryExpression,
          Value,
          VectorExpression
        };


      private:

        const ExpressionType    _ceExprType;

      protected:

        std::string _DumpResultTypeToXML(const size_t cszIntend) const;


        inline Expression(ExpressionType eExprType) : BaseType(Node::NodeType::Expression), _ceExprType(eExprType)  {}

        IndexType _FindSubExpressionIndex(ExpressionPtr spSubExpression) const;

      public:

        virtual ~Expression() {}


        virtual NodePtr   GetChild(IndexType ChildIndex) final override   { return GetSubExpression(ChildIndex); }
        virtual IndexType GetChildCount() const final override            { return GetSubExpressionCount(); }

        IndexType GetParentIndex() const;
        bool      IsSubExpression() const;

        virtual bool      IsVectorized();
        inline  bool      IsVectorized() const  { return const_cast< Expression* >(this)->IsVectorized(); }

        virtual TypeInfo  GetResultType() const = 0;

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) = 0;
        virtual IndexType       GetSubExpressionCount() const = 0;
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) = 0;

        inline const ExpressionPtr  GetSubExpression(IndexType SubExprIndex) const  { return const_cast< Expression* >(this)->GetSubExpression(SubExprIndex); }
      };
    };


    class ControlFlow final
    {
      // Forward declarations and type definitions
    public:

      class Loop;
      class LoopControlStatement;
      class ConditionalBranch;
      class BranchingStatement;
      class ReturnStatement;

      typedef std::shared_ptr< Loop >                   LoopPtr;
      typedef std::shared_ptr< LoopControlStatement >   LoopControlStatementPtr;
      typedef std::shared_ptr< ConditionalBranch >      ConditionalBranchPtr;
      typedef std::shared_ptr< BranchingStatement >     BranchingStatementPtr;
      typedef std::shared_ptr< ReturnStatement >        ReturnStatementPtr;


    public:

      class Loop final : public BaseClasses::ControlFlowStatement
      {
      private:

        typedef BaseClasses::ControlFlowStatement   BaseType;

      public:
        
        enum class LoopType
        {
          TopControlled,
          BottomControlled
        };

      private:

        LoopType                    _eLoopType;
        BaseClasses::ExpressionPtr  _spConditionExpr;
        BaseClasses::ExpressionPtr  _spIncrementExpr;
        ScopePtr                    _spBody;


        static std::string _GetLoopTypeString(LoopType eType);


      public:

        static LoopPtr Create(LoopType eType = LoopType::TopControlled, BaseClasses::ExpressionPtr spCondition = nullptr, BaseClasses::ExpressionPtr spIncrement = nullptr);

        inline Loop() : BaseType(BaseType::ControlFlowType::Loop)   {}

        virtual ~Loop() {}


        ScopePtr        GetBody();
        const ScopePtr  GetBody() const;

        inline BaseClasses::ExpressionPtr         GetCondition()                                        { return _spConditionExpr; }
        inline const BaseClasses::ExpressionPtr   GetCondition() const                                  { return _spConditionExpr; }
        inline void                               SetCondition(BaseClasses::ExpressionPtr spCondition)  { _SetChildPtr(_spConditionExpr, spCondition); }

        inline BaseClasses::ExpressionPtr         GetIncrement()                                        { return _spIncrementExpr; }
        inline const BaseClasses::ExpressionPtr   GetIncrement() const                                  { return _spIncrementExpr; }
        inline void                               SetIncrement(BaseClasses::ExpressionPtr spIncrement)  { _SetChildPtr(_spIncrementExpr, spIncrement); }

        inline LoopType GetLoopType() const           { return _eLoopType; }
        inline void     SetLoopType(LoopType eType)   { _eLoopType = eType; }


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override    { return static_cast< IndexType >( 3 ); }

        virtual bool IsVectorized() const final override;
      };

      class LoopControlStatement final : public BaseClasses::ControlFlowStatement
      {
      private:

        typedef BaseClasses::ControlFlowStatement   BaseType;

      public:

        enum class LoopControlType
        {
          Break,
          Continue
        };


      private:

        LoopControlType _eControlType;

        static std::string _GetLoopControlTypeString(LoopControlType eType);

      public:

        static LoopControlStatementPtr Create(LoopControlType eCtrlType);

        inline LoopControlStatement() : BaseType(BaseType::ControlFlowType::LoopControlStatement), _eControlType(LoopControlType::Break)  {}

        virtual ~LoopControlStatement() {}


        inline LoopControlType  GetControlType() const                        { return _eControlType; }
        inline void             SetControlType(LoopControlType eNewCtrlType)  { _eControlType = eNewCtrlType; }

        LoopPtr GetControlledLoop() const;

      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override   { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType             GetChildCount() const final override            { return static_cast< IndexType >(0); }

        virtual bool IsVectorized() const final override;
      };

      class ConditionalBranch final : public BaseClasses::ControlFlowStatement
      {
      private:

        typedef BaseClasses::ControlFlowStatement   BaseType;
        typedef BaseClasses::ExpressionPtr          ExpressionPtr;

      private:

        ExpressionPtr   _spCondition;
        ScopePtr        _spBody;

      public:

        static ConditionalBranchPtr Create(ExpressionPtr spCondition = nullptr);

        ConditionalBranch() : BaseType(BaseType::ControlFlowType::ConditionalBranch), _spCondition(nullptr), _spBody(nullptr)   {}

        virtual ~ConditionalBranch()  {}


        ScopePtr        GetBody();
        const ScopePtr  GetBody() const;

        inline ExpressionPtr        GetCondition()                            { return _spCondition; }
        inline const ExpressionPtr  GetCondition() const                      { return _spCondition; }
        inline void                 SetCondition(ExpressionPtr spCondition)   { _SetChildPtr(_spCondition, spCondition); }


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override    { return static_cast< IndexType >( 2 ); }

        virtual bool IsVectorized() const final override;
      };

      class BranchingStatement final : public BaseClasses::ControlFlowStatement
      {
      private:

        typedef BaseClasses::ControlFlowStatement   BaseType;

      private:

        std::vector< ConditionalBranchPtr >   _vecBranches;
        ScopePtr                              _spDefaultBranch;

      public:

        static BranchingStatementPtr Create();

        BranchingStatement() : BaseType(BaseType::ControlFlowType::BranchingStatement), _spDefaultBranch(nullptr)   {}

        virtual ~BranchingStatement() {}


        void                  AddConditionalBranch(ConditionalBranchPtr spBranch);
        ConditionalBranchPtr  GetConditionalBranch(IndexType BranchIndex) const;
        inline IndexType      GetConditionalBranchesCount() const   { return static_cast< IndexType >( _vecBranches.size() ); }
        void                  RemoveConditionalBranch(IndexType BranchIndex);

        ScopePtr        GetDefaultBranch();
        const ScopePtr  GetDefaultBranch() const;


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override    { return GetConditionalBranchesCount() + 1; }

        virtual bool IsVectorized() const final override;
      };

      class ReturnStatement final : public BaseClasses::ControlFlowStatement
      {
      private:

        typedef BaseClasses::ControlFlowStatement   BaseType;

      public:

        static ReturnStatementPtr Create();

        ReturnStatement() : BaseType(BaseType::ControlFlowType::ReturnStatement)   {}

        virtual ~ReturnStatement() {}


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override   { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType             GetChildCount() const final override            { return static_cast< IndexType >( 0 ); }

        virtual bool IsVectorized() const final override;
      };
    };


    class Expressions final
    {
    // Forward declarations and type definitions
    public:

      class Value;
      class Constant;
      class Identifier;
      class MemoryAccess;
      class UnaryExpression;
      class Conversion;
      class Parenthesis;
      class UnaryOperator;
      class BinaryOperator;
      class ArithmeticOperator;
      class AssignmentOperator;
      class RelationalOperator;
      class FunctionCall;

      typedef std::shared_ptr< Value >                ValuePtr;
      typedef std::shared_ptr< Constant >             ConstantPtr;
      typedef std::shared_ptr< Identifier >           IdentifierPtr;
      typedef std::shared_ptr< MemoryAccess >         MemoryAccessPtr;
      typedef std::shared_ptr< UnaryExpression >      UnaryExpressionPtr;
      typedef std::shared_ptr< Conversion >           ConversionPtr;
      typedef std::shared_ptr< Parenthesis >          ParenthesisPtr;
      typedef std::shared_ptr< UnaryOperator >        UnaryOperatorPtr;
      typedef std::shared_ptr< BinaryOperator >       BinaryOperatorPtr;
      typedef std::shared_ptr< ArithmeticOperator >   ArithmeticOperatorPtr;
      typedef std::shared_ptr< AssignmentOperator >   AssignmentOperatorPtr;
      typedef std::shared_ptr< RelationalOperator >   RelationalOperatorPtr;
      typedef std::shared_ptr< FunctionCall >         FunctionCallPtr;


    public:

      class Value : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::Expression     BaseType;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;

      public:

        enum class ValueType
        {
          Constant,
          Identifier,
          MemoryAccess
        };

      private:

        const ValueType   _ceValueType;

      protected:

        inline  Value(ValueType eValueType) : BaseType(BaseType::ExpressionType::Value), _ceValueType(eValueType)   {}

      public:

        virtual ~Value()  {}


        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) override                           { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType       GetSubExpressionCount() const override                                      { return static_cast< IndexType >(0); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) override  { throw ASTExceptions::ChildIndexOutOfRange(); }
      };

      class Constant final : public Value
      {
      private:

        typedef Value  BaseType;

        typedef BaseClasses::TypeInfo::KnownTypes   KnownTypes;

        union
        {
          std::uint64_t ui64IntegralValue;
          double        dFloatingPointValue;
        } _unionValues;

        KnownTypes    _eType;


        template < typename SourceValueType >
        inline void _ChangeType(KnownTypes eNewType)
        {
          SourceValueType TValue = GetValue< SourceValueType >();

          switch (eNewType)
          {
          case KnownTypes::Bool:    SetValue( TValue != static_cast< SourceValueType >( 0 ) );  break;
          case KnownTypes::Int8:    SetValue( static_cast< std::int8_t   >( TValue ) );         break;
          case KnownTypes::UInt8:   SetValue( static_cast< std::uint8_t  >( TValue ) );         break;
          case KnownTypes::Int16:   SetValue( static_cast< std::int16_t  >( TValue ) );         break;
          case KnownTypes::UInt16:  SetValue( static_cast< std::uint16_t >( TValue ) );         break;
          case KnownTypes::Int32:   SetValue( static_cast< std::int32_t  >( TValue ) );         break;
          case KnownTypes::UInt32:  SetValue( static_cast< std::uint32_t >( TValue ) );         break;
          case KnownTypes::Int64:   SetValue( static_cast< std::int64_t  >( TValue ) );         break;
          case KnownTypes::UInt64:  SetValue( static_cast< std::uint64_t >( TValue ) );         break;
          case KnownTypes::Float:   SetValue( static_cast< float         >( TValue ) );         break;
          case KnownTypes::Double:  SetValue( static_cast< double        >( TValue ) );         break;
          default:                  throw RuntimeErrorException(std::string("Invalid constant type: ") + BaseClasses::TypeInfo::GetTypeString(eNewType));
          }
        }


      public:

        template <typename ValueType>
        static ConstantPtr Create(ValueType TValue)
        {
          ConstantPtr spConstant = AST::CreateNode<Constant>();

          spConstant->SetValue(TValue);

          return spConstant;
        }


        inline Constant() : BaseType(BaseType::ValueType::Constant)   {}

        virtual ~Constant() {}


        inline BaseClasses::TypeInfo::KnownTypes  GetValueType() const    { return _eType; }


        void  ChangeType(KnownTypes eNewType);

        template <typename ValueType> inline ValueType GetValue() const
        {
          static_assert(std::is_arithmetic< ValueType >::value, "Expected a numeric value type!");

          switch (_eType)
          {
          case KnownTypes::Float: case KnownTypes::Double:
            return static_cast< ValueType >( _unionValues.dFloatingPointValue );
          default:
            return static_cast< ValueType >( _unionValues.ui64IntegralValue );
          }
        }

        template <>                   inline bool      GetValue<bool>() const
        {
          switch (_eType)
          {
          case KnownTypes::Float: case KnownTypes::Double:
            return ( _unionValues.dFloatingPointValue != 0. );
          default:
            return ( _unionValues.ui64IntegralValue   != static_cast< std::uint64_t >( 0 ) );
          }
        }


        template <typename ValueType> inline void SetValue(ValueType TValue)
        {
          static_assert(std::is_arithmetic< ValueType >::value, "Expected a numeric value type!");

          if (std::is_integral<ValueType>::value)
          {
            _unionValues.ui64IntegralValue  = static_cast< std::uint64_t >( TValue );

            bool bSigned = std::numeric_limits< ValueType >::is_signed;

            switch (sizeof(ValueType))
            {
            case 1:   _eType = bSigned ? KnownTypes::Int8  : KnownTypes::UInt8;   break;
            case 2:   _eType = bSigned ? KnownTypes::Int16 : KnownTypes::UInt16;  break;
            case 4:   _eType = bSigned ? KnownTypes::Int32 : KnownTypes::UInt32;  break;
            default:  _eType = bSigned ? KnownTypes::Int64 : KnownTypes::UInt64;  break;
            }
          }
          else
          {
            _unionValues.dFloatingPointValue = static_cast< double >( TValue );

            _eType = (sizeof(ValueType) == 4) ? KnownTypes::Float : KnownTypes::Double;
          }
        }

        template<>                    inline void SetValue<bool>(bool TValue)
        {
          _unionValues.ui64IntegralValue  = static_cast< std::uint64_t >( TValue ? 1 : 0 );
          _eType                          = KnownTypes::Bool;
        }



        std::string GetAsString() const;

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class Identifier final : public Value
      {
      private:

        typedef Value  BaseType;

        std::string   _strName;

      public:

        static IdentifierPtr Create( std::string strName );

        inline Identifier() : BaseType(BaseType::ValueType::Identifier)   {}

        virtual ~Identifier() {}


        inline std::string  GetName() const               { return _strName; }
        inline void         SetName(std::string strName)  { _strName = strName; }

        BaseClasses::VariableInfoPtr LookupVariableInfo() const;


        virtual bool IsVectorized() final override;

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class MemoryAccess final : public Value
      {
      private:

        typedef Value                         BaseType;
        typedef BaseClasses::ExpressionPtr    ExpressionPtr;

        ExpressionPtr   _spMemoryRef;
        ExpressionPtr   _spIndexExpr;

      public:

        static MemoryAccessPtr Create(ExpressionPtr spMemoryReference = nullptr, ExpressionPtr spIndexExpression = nullptr);

        inline MemoryAccess() : BaseType(BaseType::ValueType::MemoryAccess), _spMemoryRef(nullptr), _spIndexExpr(nullptr)   {}

        virtual ~MemoryAccess() {}


        inline ExpressionPtr        GetIndexExpression()                                  { return _spIndexExpr; }
        inline const ExpressionPtr  GetIndexExpression() const                            { return _spIndexExpr; }
        inline void                 SetIndexExpression(ExpressionPtr spIndexExpression)   { _SetChildPtr(_spIndexExpr, spIndexExpression); }

        inline ExpressionPtr        GetMemoryReference()                                  { return _spMemoryRef; }
        inline const ExpressionPtr  GetMemoryReference() const                            { return _spMemoryRef; }
        inline void                 SetMemoryReference(ExpressionPtr spMemoryReference)   { _SetChildPtr(_spMemoryRef, spMemoryReference); }


        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType       GetSubExpressionCount() const final override  { return static_cast< IndexType >(2); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;


        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };


      class UnaryExpression : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::Expression   BaseType;

      public:

        enum class UnaryExpressionType
        {
          Conversion,
          Parenthesis,
          UnaryOperator
        };

      private:

        const UnaryExpressionType   _ceUnaryExprType;
        BaseClasses::ExpressionPtr  _spSubExpression;

      protected:

        std::string _DumpSubExpressionToXML(const size_t cszIntend) const;

        inline UnaryExpression(UnaryExpressionType eType) : BaseType(BaseType::ExpressionType::UnaryExpression), _ceUnaryExprType(eType), _spSubExpression(nullptr)  {}

      public:

        virtual ~UnaryExpression()  {}


        inline BaseClasses::ExpressionPtr         GetSubExpression()                                      { return _spSubExpression; }
        inline const BaseClasses::ExpressionPtr   GetSubExpression() const                                { return _spSubExpression; }
        inline void                               SetSubExpression(BaseClasses::ExpressionPtr spSubExpr)  { _SetChildPtr(_spSubExpression, spSubExpr); }

        virtual BaseClasses::ExpressionPtr  GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType                   GetSubExpressionCount() const final override  { return static_cast< IndexType >( 1 ); }
        virtual void                        SetSubExpression(IndexType SubExprIndex, BaseClasses::ExpressionPtr spSubExpr) final override;
      };

      class Conversion final : public UnaryExpression
      {
      private:

        typedef UnaryExpression   BaseType;


        BaseClasses::TypeInfo   _ConvertType;
        bool                    _bIsExplicit;

      public:

        static ConversionPtr Create(const BaseClasses::TypeInfo &crConvertType, BaseClasses::ExpressionPtr spSubExpression = nullptr, bool bExplicit = true);

        inline Conversion() : BaseType(BaseType::UnaryExpressionType::Conversion), _bIsExplicit(true)   {}

        virtual ~Conversion() {}


        inline BaseClasses::TypeInfo  GetConvertType() const                                    { return _ConvertType; }
        inline void                   SetConvertType(const BaseClasses::TypeInfo &crConvType)   { _ConvertType = crConvType; }

        inline bool GetExplicit() const             { return _bIsExplicit; }
        inline void SetExplicit(bool bIsExplicit)   { _bIsExplicit = bIsExplicit; }

      public:

        virtual BaseClasses::TypeInfo GetResultType() const final override    { return GetConvertType(); }

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class Parenthesis final : public UnaryExpression
      {
      private:

        typedef UnaryExpression   BaseType;

      public:

        static ParenthesisPtr Create(BaseClasses::ExpressionPtr spSubExpression = nullptr);

        inline Parenthesis() : BaseType(BaseType::UnaryExpressionType::Parenthesis)   {}

        virtual ~Parenthesis()  {}


      public:

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class UnaryOperator final : public UnaryExpression
      {
      private:

        typedef UnaryExpression   BaseType;


      public:

        enum class UnaryOperatorType
        {
          AddressOf,
          BitwiseNot,
          LogicalNot,
          Minus,
          Plus,
          PostDecrement,
          PostIncrement,
          PreDecrement,
          PreIncrement
        };

      private:

        UnaryOperatorType    _eOpType;

      public:

        static UnaryOperatorPtr Create(UnaryOperatorType eType = UnaryOperatorType::Plus, BaseClasses::ExpressionPtr spSubExpression = nullptr);

        inline UnaryOperator() : BaseType(BaseType::UnaryExpressionType::UnaryOperator), _eOpType(UnaryOperatorType::Plus)  {}

        virtual ~UnaryOperator()  {}


        static std::string GetOperatorTypeString(UnaryOperatorType eType);


        inline UnaryOperatorType  GetOperatorType() const                     { return _eOpType; }
        inline void               SetOperatorType(UnaryOperatorType eOpType)  { _eOpType = eOpType; }


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::TypeInfo GetResultType() const final override;
      };


      class BinaryOperator : public BaseClasses::Expression
      {
      protected:

        typedef BaseClasses::Expression     BaseType;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;

      public:

        enum class BinaryOperatorType
        {
          ArithmeticOperator,
          AssignmentOperator,
          RelationalOperator
        };


      private:

        const BinaryOperatorType  _ceBinOpType;
        ExpressionPtr             _spLHS;
        ExpressionPtr             _spRHS;

      protected:

        std::string _DumpSubExpressionsToXML(const size_t cszIntend) const;

        inline BinaryOperator(BinaryOperatorType eBinOpType) : BaseType(BaseType::ExpressionType::BinaryOperator), _ceBinOpType(eBinOpType)   {}

      public:

        virtual ~BinaryOperator() {}


        inline ExpressionPtr        GetLHS()                        { return _spLHS; }
        inline const ExpressionPtr  GetLHS() const                  { return _spLHS; }
        inline void                 SetLHS(ExpressionPtr spNewLHS)  { _SetChildPtr(_spLHS, spNewLHS); }

        inline ExpressionPtr        GetRHS()                        { return _spRHS; }
        inline const ExpressionPtr  GetRHS() const                  { return _spRHS; }
        inline void                 SetRHS(ExpressionPtr spNewRHS)  { _SetChildPtr(_spRHS, spNewRHS); }


      public:

        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) override;
        virtual IndexType     GetSubExpressionCount() const override      { return static_cast< IndexType >(2); }
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) override;
      };

      class ArithmeticOperator final : public BinaryOperator
      {
      private:

        typedef BinaryOperator          BaseType;
        typedef BaseClasses::TypeInfo   TypeInfo;
        typedef TypeInfo::KnownTypes    KnownTypes;

      public:

        enum class ArithmeticOperatorType
        {
          Add,
          BitwiseAnd,
          BitwiseOr,
          BitwiseXOr,
          Divide,
          Modulo,
          Multiply,
          ShiftLeft,
          ShiftRight,
          Subtract
        };


      private:

        ArithmeticOperatorType    _eOpType;

      public:

        static ArithmeticOperatorPtr Create(ArithmeticOperatorType eOpType = ArithmeticOperatorType::Add, ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr);

        inline ArithmeticOperator() : BaseType(BaseType::BinaryOperatorType::ArithmeticOperator), _eOpType(ArithmeticOperatorType::Add)  {}

        virtual ~ArithmeticOperator() {}


        static std::string GetOperatorTypeString(ArithmeticOperatorType eType);


        inline ArithmeticOperatorType GetOperatorType() const                           { return _eOpType; }
        inline void                   SetOperatorType(ArithmeticOperatorType eOpType)   { _eOpType = eOpType; }


      public:

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

      };

      class AssignmentOperator final : public BinaryOperator
      {
      private:

        typedef BinaryOperator            BaseType;
        typedef BaseType::ExpressionPtr   ExpressionPtr;

      private:

        IdentifierPtr   _spMask;

      public:

        static AssignmentOperatorPtr Create(ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr, IdentifierPtr spMask = nullptr);

        inline AssignmentOperator() : BaseType(BaseType::BinaryOperatorType::AssignmentOperator), _spMask(nullptr)  {}

        virtual ~AssignmentOperator() {}

        inline IdentifierPtr        GetMask()                         { return _spMask; }
        inline const IdentifierPtr  GetMask() const                   { return _spMask; }
        inline void                 SetMask(IdentifierPtr spNewMask)  { _SetChildPtr(_spMask, spNewMask); }


        inline bool IsMasked() const  { return static_cast<bool>( GetMask() ); }


      public:

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;


        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType     GetSubExpressionCount() const final override;
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;
      };

      class RelationalOperator final : public BinaryOperator
      {
      private:

        typedef BinaryOperator   BaseType;

      public:

        enum class RelationalOperatorType
        {
          Equal,
          Greater,
          GreaterEqual,
          Less,
          LessEqual,
          LogicalAnd,
          LogicalOr,
          NotEqual
        };

      private:

        RelationalOperatorType  _eOpType;

      public:

        static RelationalOperatorPtr  Create(RelationalOperatorType eOpType = RelationalOperatorType::Equal, ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr);

        inline RelationalOperator() : BaseType(BaseType::BinaryOperatorType::RelationalOperator), _eOpType(RelationalOperatorType::Equal)  {}

        virtual ~RelationalOperator() {}


        static std::string GetOperatorTypeString(RelationalOperatorType eType);


        BaseClasses::TypeInfo GetComparisonType() const;

        inline RelationalOperatorType GetOperatorType() const                           { return _eOpType; }
        inline void                   SetOperatorType(RelationalOperatorType eOpType)   { _eOpType = eOpType; }


      public:

        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

      };


      class FunctionCall : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::Expression     BaseType;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;

      private:

        std::string                     _strName;
        BaseClasses::TypeInfo           _ReturnType;
        std::vector< ExpressionPtr >    _vecCallParams;

      public:

        static FunctionCallPtr Create(std::string strFunctionName, const BaseClasses::TypeInfo &crReturnType);

        inline FunctionCall() : BaseType(BaseType::ExpressionType::FunctionCall)    {}

        virtual ~FunctionCall() {}


        void          AddCallParameter(ExpressionPtr spCallParam);
        ExpressionPtr GetCallParameter(IndexType CallParamIndex) const;
        void          SetCallParameter(IndexType CallParamIndex, ExpressionPtr spCallParam);

        inline IndexType GetCallParameterCount() const    { return static_cast< IndexType >(_vecCallParams.size()); }

        inline std::string  GetName() const                   { return _strName; }
        inline void         SetName(std::string strNewName)   { _strName = strNewName; }

        inline BaseClasses::TypeInfo GetReturnType() const                                      { return _ReturnType; }
        inline void                  SetReturnType(const BaseClasses::TypeInfo &crReturnType)   { _ReturnType = crReturnType; }


      public:

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::TypeInfo GetResultType() const final override    { return GetReturnType(); }

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex)                                          { return GetCallParameter(SubExprIndex); }
        virtual IndexType       GetSubExpressionCount() const final override                                      { return GetCallParameterCount(); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override  { SetCallParameter(SubExprIndex, spSubExpr); }
      };
    };


    class VectorSupport final
    {
      // Forward declarations and type definitions
    public:

      class VectorExpression;
      class BroadCast;
      class CheckActiveElements;
      class VectorIndex;

      typedef std::shared_ptr< VectorExpression >       VectorExpressionPtr;
      typedef std::shared_ptr< CheckActiveElements >    CheckActiveElementsPtr;
      typedef std::shared_ptr< BroadCast >              BroadCastPtr;
      typedef std::shared_ptr< VectorIndex >            VectorIndexPtr;


    public:

      class VectorExpression : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::Expression     BaseType;

      public:

        enum class VectorExpressionType
        {
          BroadCast,
          CheckActiveElements,
          VectorIndex
        };

      private:

        const VectorExpressionType   _ceVectorExpressionType;

      protected:

        inline VectorExpression(VectorExpressionType eVecExprType) : BaseType(BaseType::ExpressionType::VectorExpression), _ceVectorExpressionType(eVecExprType)   {}

      public:

        virtual ~VectorExpression()  {}
      };

      class BroadCast final : public VectorExpression
      {
      private:

        typedef VectorExpression            BaseType;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;

      private:

        ExpressionPtr _spSubExpression;

      public:

        static BroadCastPtr Create(ExpressionPtr spSubExpression = nullptr);

        inline BroadCast() : BaseType(BaseType::VectorExpressionType::BroadCast), _spSubExpression(nullptr)   {}

        virtual ~BroadCast()  {}


        inline ExpressionPtr        GetSubExpression()                          { return _spSubExpression; }
        inline const ExpressionPtr  GetSubExpression() const                    { return _spSubExpression; }
        inline void                 SetSubExpression(ExpressionPtr spSubExpr)   { _SetChildPtr(_spSubExpression, spSubExpr); }


      public:

        virtual bool IsVectorized() final override      { return true; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override;

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType       GetSubExpressionCount() const final override      { return static_cast< IndexType >( 1 ); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class CheckActiveElements final : public VectorExpression
      {
      private:

        typedef VectorExpression                    BaseType;
        typedef BaseClasses::ExpressionPtr          ExpressionPtr;

      public:

        enum class CheckType
        {
          All,
          Any,
          None
        };

      private:

        CheckType     _eCheckType;
        ExpressionPtr _spSubExpression;

      public:

        static CheckActiveElementsPtr Create(CheckType eCheckType = CheckType::All, ExpressionPtr spSubExpression = nullptr);

        inline CheckActiveElements() : BaseType(BaseType::VectorExpressionType::CheckActiveElements), _eCheckType(CheckType::All), _spSubExpression(nullptr)  {}

        virtual ~CheckActiveElements()  {}


        static std::string GetCheckTypeString(CheckType eType);


        inline CheckType  GetCheckType() const                    { return _eCheckType; }
        inline void       SetCheckType(CheckType eNewCheckType)   { _eCheckType = eNewCheckType; }

        inline ExpressionPtr        GetSubExpression()                          { return _spSubExpression; }
        inline const ExpressionPtr  GetSubExpression() const                    { return _spSubExpression; }
        inline void                 SetSubExpression(ExpressionPtr spSubExpr)   { _SetChildPtr(_spSubExpression, spSubExpr); }

      public:

        virtual bool IsVectorized() final override                            { return false; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override   { return BaseClasses::TypeInfo( BaseClasses::TypeInfo::KnownTypes::Bool, true, false ); }

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType       GetSubExpressionCount() const final override                                      { return static_cast< IndexType >(1); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };

      class VectorIndex final : public VectorExpression
      {
      private:

        typedef VectorExpression                    BaseType;
        typedef BaseClasses::ExpressionPtr          ExpressionPtr;
        typedef BaseClasses::TypeInfo::KnownTypes   KnownTypes;

      private:

        KnownTypes  _eType;

      public:

        static VectorIndexPtr Create(KnownTypes eType = KnownTypes::Int32);

        inline VectorIndex() : BaseType(BaseType::VectorExpressionType::VectorIndex), _eType(KnownTypes::Int32)  {}

        virtual ~VectorIndex()  {}


        inline KnownTypes   GetType() const                 { return _eType; }
        inline void         SetType(KnownTypes eNewType)    { _eType = eNewType; }


      public:

        virtual bool IsVectorized() final override      { return true; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override;

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override                           { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType       GetSubExpressionCount() const final override                                      { return static_cast< IndexType >(0); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override  { throw ASTExceptions::ChildIndexOutOfRange(); }

        virtual std::string DumpToXML(const size_t cszIntend) const final override;
      };
    };


    class IVariableContainer : public BaseClasses::Node
    {
    private:

      typedef BaseClasses::Node       BaseType;

    protected:

      inline IVariableContainer(BaseType::NodeType eNodeType) : BaseType(eNodeType)   {}

    public:

      virtual ~IVariableContainer() {}


      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) = 0;
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) const = 0;
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const = 0;
    };


    class Scope final : public IVariableContainer
    {
    public:

      typedef std::vector< Expressions::IdentifierPtr >   VariableDeclarationVectorType;

    private:

      typedef IVariableContainer      BaseType;
      typedef BaseClasses::NodePtr    NodePtr;

      typedef std::vector< NodePtr >  ChildrenContainerType;


    private:

      ChildrenContainerType     _Children;
      std::set< std::string >   _setDeclaredVariables;


      IVariableContainerPtr               _GetParentVariableContainer();
      inline const IVariableContainerPtr  _GetParentVariableContainer() const { return const_cast< Scope* >( this )->_GetParentVariableContainer(); }


    public:

      static ScopePtr Create();

      inline Scope() : BaseType(Node::NodeType::Scope)   {}

      virtual ~Scope()  {}


      void      AddChild(NodePtr spChild);
      IndexType GetChildIndex(NodePtr spChildNode);
      void      InsertChild(IndexType ChildIndex, NodePtr spChildNode);
      void      RemoveChild(IndexType ChildIndex);
      void      SetChild(IndexType ChildIndex, NodePtr spChildNode);

      void                            AddVariableDeclaration(BaseClasses::VariableInfoPtr spVariableInfo);
      VariableDeclarationVectorType   GetVariableDeclarations() const;
      inline bool                     HasVariableDeclaration(std::string strVariableName) const   { return (_setDeclaredVariables.count(strVariableName) != 0); }
      void                            ImportVariableDeclarations(ScopePtr spOtherScope);

      void ImportScope(ScopePtr spOtherScope);

      inline bool IsEmpty() const   { return (GetChildCount() == static_cast<IndexType>(0)); }


    public:

      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) final override;
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) const final override;
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const final override;

      virtual NodePtr       GetChild(IndexType ChildIndex) final override;
      virtual IndexType     GetChildCount() const final override  { return static_cast< IndexType >(_Children.size()); }


      virtual std::string DumpToXML(const size_t cszIntend) const final override;
    };


    class FunctionDeclaration final : public IVariableContainer
    {
    private:

      typedef IVariableContainer    BaseType;
      typedef BaseClasses::NodePtr  NodePtr;

      typedef std::vector< Expressions::IdentifierPtr >              ParameterContainerType;
      typedef std::map< std::string, BaseClasses::VariableInfoPtr >  KnownVariablesMapType;


    private:

      ParameterContainerType  _Parameters;
      ScopePtr                _spBody;
      KnownVariablesMapType   _mapKnownVariables;
      std::string             _strName;


    public:

      static FunctionDeclarationPtr Create(std::string strFunctionName);

      inline FunctionDeclaration() : BaseType(Node::NodeType::FunctionDeclaration), _spBody(nullptr)  {}

      virtual ~FunctionDeclaration()  {}


      void                        AddParameter(BaseClasses::VariableInfoPtr spVariableInfo);
      Expressions::IdentifierPtr  GetParameter(IndexType iParamIndex);
      inline IndexType            GetParameterCount() const   { return static_cast< IndexType >( _Parameters.size() ); }
      void                        SetParameter(IndexType iParamIndex, BaseClasses::VariableInfoPtr spVariableInfo);


      ScopePtr        GetBody();
      const ScopePtr  GetBody() const;

      inline std::string  GetName() const               { return _strName; }
      inline void         SetName(std::string strName)  { _strName = strName; }

    public:

      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) final override;
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) const final override;
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const final override;


      virtual NodePtr       GetChild(IndexType ChildIndex) final override;
      virtual IndexType     GetChildCount() const final override  { return static_cast< IndexType >(1); }


      virtual std::string DumpToXML(const size_t cszIntend) const final override;
    };



    public:

      template < class NodeClass >
      inline static std::shared_ptr< NodeClass > CreateNode()
      {
        static_assert(std::is_base_of< BaseClasses::Node, NodeClass >::value, "All nodes of the vectorizable AST must be derived from class \"Node\"");

        std::shared_ptr< NodeClass > spNode = std::make_shared< NodeClass >();

        spNode->_wpThis = BaseClasses::NodePtr( spNode );

        return spNode;
      }
  };
} // end namespace Vectorization
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_VECTORIZATION_AST_H_

// vim: set ts=2 sw=2 sts=2 et ai:

