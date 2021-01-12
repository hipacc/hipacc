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

#ifndef _HIPACC_BACKEND_VECTORIZATION_AST_H_
#define _HIPACC_BACKEND_VECTORIZATION_AST_H_

#include "hipacc/Backend/BackendExceptions.h"

#include <llvm/Support/Casting.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <type_traits>
#include <limits>

namespace clang
{
namespace hipacc
{
namespace Backend
{
namespace Vectorization
{
  /** \brief  Contains common exceptions, which can be thrown by the abstract vectorized AST. */
  class ASTExceptions
  {
  public:

    /** \brief  Indicates that a specified child node index has been out of range. */
    class ChildIndexOutOfRange : public InternalErrorException
    {
    private:

      typedef InternalErrorException  BaseType;   //!< The base type of this class.

    public:

      inline ChildIndexOutOfRange() : BaseType("The index for the child node is out of range!")  {}
    };

    /** \brief  Indicates that newly specified variable declaration has a conflicting name. */
    class DuplicateVariableName : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline DuplicateVariableName(std::string strVarName) : BaseType(std::string("The variable name \"") + strVarName + std::string("\" is not unique!"))  {}
    };

    /** \brief  Indicates that an expression is deferenced, which is neither a pointer nor an array. */
    class NonDereferencableType : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline NonDereferencableType() : BaseType("The specified type cannot be dereferenced!")  {}
    };


    /** \brief  Externally used exception, indicating that a specific expression class is unknown. */
    class UnknownExpressionClass : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline UnknownExpressionClass(std::string strExprClassName) : BaseType( std::string( "The expression class \"") + strExprClassName + std::string("\" is unknown!") )  {}
    };

    /** \brief  Externally used exception, indicating that a specific statement class is unknown. */
    class UnknownStatementClass : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException  BaseType;   //!< The base type of this class.

    public:

      inline UnknownStatementClass(std::string strStmtClassName) : BaseType(std::string("The statement class \"") + strStmtClassName + std::string("\" is unknown!"))  {}
    };
  };


  /** \brief  Contains all class definitions for the abstract <b>vectorized Annotated Syntax Tree (VAST)</b>. */
  class AST
  {
  public:

    typedef size_t    IndexType;    //!< Type defintions for the internally used indices

    /** \name Shared pointer type definitions */
    //@{

    class IVariableContainer;
    typedef std::shared_ptr<IVariableContainer>       IVariableContainerPtr;      //!< Shared pointer type for objects of class IVariableContainer
    typedef std::shared_ptr<const IVariableContainer> IVariableContainerConstPtr; //!< Shared pointer type for constant objects of class IVariableContainer

    class FunctionDeclaration;
    typedef std::shared_ptr<FunctionDeclaration>       FunctionDeclarationPtr;      //!< Shared pointer type for objects of class FunctionDeclaration
    typedef std::shared_ptr<const FunctionDeclaration> FunctionDeclarationConstPtr; //!< Shared pointer type for constant objects of class FunctionDeclaration

    class Scope;
    typedef std::shared_ptr<Scope>       ScopePtr;      //!< Shared pointer type for objects of class Scope
    typedef std::shared_ptr<const Scope> ScopeConstPtr; //!< Shared pointer type for constant objects of class Scope

    //@}


  public:

    /** \brief  Helper class, which stores the position of a statement inside its enclosing scope. */
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


      /** \brief  Returns a shared pointer to the enclosing scope. */
      inline ScopePtr    GetScope()       { return _spScope; }

      /** \brief  Returns the referenced child node index in the enclosing scope. */
      inline IndexType   GetChildIndex()  { return _ChildIndex; }
    };


    /** \brief  Contains abstract base classes as well as commonly used node types. */
    class BaseClasses final
    {
    public:

      /** \name Shared pointer type definitions */
      //@{

      class VariableInfo;
      typedef std::shared_ptr<VariableInfo>       VariableInfoPtr;      //!< Shared pointer type for objects of class VariableInfo
      typedef std::shared_ptr<const VariableInfo> VariableInfoConstPtr; //!< Shared pointer type for constant objects of class VariableInfo

      class Node;
      typedef std::shared_ptr<Node>       NodePtr;      //!< Shared pointer type for objects of class Node
      typedef std::shared_ptr<const Node> NodeConstPtr; //!< Shared pointer type for constant objects of class Node

      class ControlFlowStatement;
      typedef std::shared_ptr<ControlFlowStatement>       ControlFlowStatementPtr;      //!< Shared pointer type for objects of class ControlFlowStatement
      typedef std::shared_ptr<const ControlFlowStatement> ControlFlowStatementConstPtr; //!< Shared pointer type for constant objects of class ControlFlowStatement

      class Expression;
      typedef std::shared_ptr<Expression>       ExpressionPtr;      //!< Shared pointer type for objects of class Expression
      typedef std::shared_ptr<const Expression> ExpressionConstPtr; //!< Shared pointer type for constant objects of class Expression

      //@}


    public:

      /** \brief    Encapsulates the information about qualiified type.
       *  \remarks  A the current stage, only the following kinds of types are suupported:<BR>
                    <UL>
                      <LI>Native element types</LI>
                      <LI>Pointer types to native elements</LI>
                      <LI>Multi-dimensional array types of native elements</LI>
                      <LI>Multi-dimensional arrays of pointer types of native elements</LI>
                    </UL> */
      class TypeInfo
      {
      public:

        typedef std::vector<size_t> ArrayDimensionVectorType; //!< Type definition for a list of array dimensions

        enum class KnownTypes
        {
          Bool,     //!< Internal ID for a boolean type
          Int8,     //!< Internal ID for a signed 8-bit integer type
          UInt8,    //!< Internal ID for an unsigned 8-bit integer type
          Int16,    //!< Internal ID for a signed 16-bit integer type
          UInt16,   //!< Internal ID for an unsigned 16-bit integer type
          Int32,    //!< Internal ID for a signed 32-bit integer type
          UInt32,   //!< Internal ID for an unsigned 32-bit integer type
          Int64,    //!< Internal ID for a signed 64-bit integer type
          UInt64,   //!< Internal ID for an unsigned 64-bit integer type
          Float,    //!< Internal ID for a single-precision floating-point type
          Double,   //!< Internal ID for a double-precision floating-point type
          Unknown   //!< Internal ID for all currently unknown types
        };


      private:

        KnownTypes                _eType;
        bool                      _bIsConst;
        bool                      _bIsPointer;
        ArrayDimensionVectorType  _vecArrayDimensions;


      public:

        /** \brief  Constructs a new TypeInfo object.
         *  \param  eType       The requested native element type.
         *  \param  bIsConst    A flag indicating, whether the specified type is marked as constant.
         *  \param  bIsPointer  A flag indicating, whether the specified type is a pointer. */
        inline TypeInfo(KnownTypes eType = KnownTypes::Unknown, bool bIsConst = false, bool bIsPointer = false)
        {
          _eType      = eType;
          _bIsConst   = bIsConst;
          _bIsPointer = bIsPointer;
        }

        inline TypeInfo(const TypeInfo &crRVal)   { *this = crRVal; }
        TypeInfo& operator=(const TypeInfo &crRVal);


        /** \brief  Returns the type, which would be created by dereferencing this type a single time. */
        TypeInfo CreateDereferencedType() const;

        /** \brief  Returns the type, which corresponds to a pointer to this type. */
        TypeInfo CreatePointerType() const;


        /** \brief  Returns a reference to the array dimensions list of this type. */
        inline ArrayDimensionVectorType&        GetArrayDimensions()        { return _vecArrayDimensions; }

        /** \brief  Returns a constant reference to the array dimensions list of this type. */
        inline const ArrayDimensionVectorType&  GetArrayDimensions() const  { return _vecArrayDimensions; }


        /** \brief  Returns the currently set <b>constant</b> marker of this type. */
        inline bool GetConst() const          { return _bIsConst; }

        /** \brief  Changes the <b>constant</b> marker of this type.
         *  \param  bIsConst  The new constant marker. */
        inline void SetConst(bool bIsConst)   { _bIsConst = bIsConst; }


        /** \brief  Returns the currently set <b>pointer</b> marker of this type. */
        inline bool GetPointer() const            { return _bIsPointer; }

        /** \brief  Changes the <b>pointer</b> marker of this type.
         *  \param  bIsPointer  The new pointer marker. */
        inline void SetPointer(bool bIsPointer)   { _bIsPointer = bIsPointer; }


        /** \brief  Returns the native element type of this type. */
        inline KnownTypes GetType() const             { return _eType; }

        /** \brief  Changes the native element type of this type.
         *  \param  eType   The requested new native element type. */
        inline void       SetType(KnownTypes eType)   { _eType = eType; }


        /** \brief  Returns, whether this type is an array type. */
        inline bool IsArray() const           { return (!_vecArrayDimensions.empty()); }

        /** \brief  Returns, whether this type is dereferencable, i.e. whether it is an pointer or an array. */
        inline bool IsDereferencable() const  { return (IsArray() || GetPointer()); }

        /** \brief  Returns, whether this type is a native element type. */
        inline bool IsSingleValue() const     { return (!IsDereferencable()); }


        /** \brief  Checks, whether this TypeInfo object is identical with another one.
         *  \param  crRVal                  The other TypeInfo object, the current one shall be compared with.
         *  \param  bIgnoreConstQualifier   A flag indicating, whether the <b>const</b> flag of both types shall be ignored during the comparison. */
        bool IsEqual(const TypeInfo &crRVal, bool bIgnoreConstQualifier);

        /** \brief  Checks, whether two TypeInfo objects describe the identical type.
         *  \param  crRVal  The other TypeInfo object, the current one shall be compared with. */
        inline bool operator==(const TypeInfo &crRVal)    { return IsEqual(crRVal, false); }


        /** \brief  Dumps the contents of this object into an XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        std::string DumpToXML(const size_t cszIntend) const;


      public:

        /** \brief  Creates a TypeInfo object representing an integer type with a specified size.
         *  \param  szTypeSize  The requested size of the integer element type in bytes.
         *  \param  bSigned     A flag indicating, whether the created integer type shall be signed. */
        static TypeInfo     CreateSizedIntegerType(size_t szTypeSize, bool bSigned);

        /** \brief  Returns the promoted element type, which would result from an operation on the two input element types.
         *  \param  eTypeLHS  The first input element type.
         *  \param  eTypeRHS  The second input element type. */
        static KnownTypes   GetPromotedType(KnownTypes eTypeLHS, KnownTypes eTypeRHS);

        /** \brief  Returns the size of a specific element type in bytes.
         *  \param  eType   The element type, whose size shall be returned. */
        static size_t       GetTypeSize(KnownTypes eType);

        /** \brief  Returns the string identifier of a specific element type.
         *  \param  eType   The element type, whose string identifier shall be returned. */
        static std::string  GetTypeString(KnownTypes eType);

        /** \brief  Checks, whether a certain element type is a signed type.
         *  \param  eType   The element type, which shall be checked. */
        static bool         IsSigned(KnownTypes eType);
      };

      /** \brief  Describes a variable declaration. */
      class VariableInfo
      {
      private:

        friend class AST;

        std::string _strName;
        TypeInfo    _Type;
        bool        _bVectorize;

        inline VariableInfo() : _strName(""), _bVectorize(false)    {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  strName     The requested name of the new declared variable.
         *  \param  crTypeInfo  The TypeInfo object for this declaration (its contents will be copied).
         *  \param  bVectorize  A flag indicating, whether the newly declared variable is vectorized.
         *  \return A shared pointer to the newly created VariableInfo object. */
        static VariableInfoPtr Create(std::string strName, const TypeInfo &crTypeInfo, bool bVectorize = false);


        /** \brief  Returns the currently set variable name of this declaration. */
        inline std::string  GetName() const               { return _strName; }

        /** \brief  Changes the name of the declared variable.
         *  \param  strName  The new variable name. */
        inline void         SetName(std::string strName)  { _strName = strName; }


        /** \brief  Returns a reference to the TypeInfo object of this variable declaration. */
        inline TypeInfo&        GetTypeInfo()       { return _Type; }

        /** \brief  Returns a constant reference to the TypeInfo object of this variable declaration. */
        inline const TypeInfo&  GetTypeInfo() const { return _Type; }


        /** \brief  Returns the currently set <b>vectorization</b> marker of this variable declaration. */
        inline bool GetVectorize() const            { return _bVectorize; }

        /** \brief  Changes the <b>vectorization</b> marker of this variable declaration.
         *  \param  bVectorize  The new vectorization marker. */
        inline void SetVectorize(bool bVectorize)   { _bVectorize = bVectorize; }


        /** \brief  Dumps the contents of this object into an XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        std::string DumpToXML(const size_t cszIntend) const;
      };

      /** \brief  Base class for all VAST nodes describing any kind of statements. */
      class Node
      {
      private:

        friend class AST;

        std::weak_ptr<Node> _wpParent;
        std::weak_ptr<Node> _wpThis;


        /** \brief  Sets the parent of a VAST node.
         *  \param  spParent  A shared pointer to the new parent VAST node. */
        inline void _SetParent(NodePtr spParent)        { _wpParent = spParent; }

      protected:

        inline Node()   {}


        /** \brief  Dumps the contents of a child node into an XML string.
         *  \param  spChild     A constant shared pointer to the child node, which shall be dumped.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        static std::string _DumpChildToXml(const NodeConstPtr spChild, const size_t cszIntend);


        /** \brief  Links a specific child node pointer to another child node, and updates the parent pointers.
         *  \tparam NodeClassPtr      The shared pointer type of the child node pointers.
         *  \param  rDestinationPtr   A reference to the shared pointer of the child node, which shall be exchanged.
         *  \param  crSourcePtr       A constant reference to the shared pointer, which points to the new child node. */
        template <typename NodeClassPtr>
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

        /** \brief  Sets this node as the parent node to a child node.
         *  \param  spChild   A shared pointer to the child node, whose parent pointer shall be updated. */
        void _SetParentToChild(NodePtr spChild) const;


      public:

        virtual ~Node() {}


        /** \brief  Returns the index of this node in the current AST, i.e. the number of parents. */
        IndexType GetHierarchyLevel() const;


        /** \brief  Returns a shared pointer to the direct parent of this node. */
        NodePtr               GetParent();

        /** \brief  Returns a constant shared pointer to the direct parent of this node. */
        inline NodeConstPtr   GetParent() const   { return const_cast<Node*>(this)->GetParent(); }


        /** \brief  Returns the current position of this node in its enclosing scope. */
        ScopePosition GetScopePosition();


        /** \brief  Returns a shared pointer to the current node. */
        inline NodePtr        GetThis()         { return _wpThis.lock(); }

        /** \brief  Returns a constant shared pointer to the current node. */
        inline NodeConstPtr   GetThis() const   { return _wpThis.lock(); }


        /** \brief  Checks, whether the current node is a leaf node, i.e. if it has no children. */
        inline bool IsLeafNode() const  { return (GetChildCount() == static_cast<IndexType>(0)); }



        /** \brief    Cast this node into another node type.
         *  \tparam   NodeClass   The requested target type of the cast.
         *  \remarks  If the current object is not implementing the requested type, an exception will be thrown. */
        template <class NodeClass>
        inline std::shared_ptr<NodeClass> CastToType()
        {
          if (! IsType<NodeClass>())
          {
            throw RuntimeErrorException("Invalid node cast type!");
          }

          return std::dynamic_pointer_cast<NodeClass>( GetThis() );
        }

        /** \brief    Cast this node into another constant node type.
         *  \tparam   NodeClass   The requested target type of the cast.
         *  \remarks  If the current object is not implementing the requested type, an exception will be thrown. */
        template <class NodeClass>
        inline std::shared_ptr<const NodeClass> CastToType() const
        {
          return const_cast<Node*>( this )->CastToType<const NodeClass>();
        }

        /** \brief    Checks, whether this node is implementing a specific node type.
         *  \tparam   NodeClass   The node type, which shall be checked. */
        template <class NodeClass>
        inline bool IsType() const
        {
          static_assert( std::is_base_of<Node, NodeClass>::value, "All VAST nodes must be derived from class \"Node\"!" );

          return (dynamic_cast<const NodeClass*>(this) != nullptr);
        }


      public:

        /** \name Abstract methods implemented by the derived classes. */
        //@{

        /** \brief    Returns a shared pointer to a specific child node of the current one.
         *  \param    ChildIndex  The index of the requested child node.
         *  \remarks  If the child index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
        virtual NodePtr     GetChild(IndexType ChildIndex) = 0;

        /** \brief  Returns the number of children of this node. */
        virtual IndexType   GetChildCount() const = 0;


        /** \brief  Dumps the contents of this node into an XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        virtual std::string DumpToXML(const size_t cszIntend) const = 0;

        //@}
      };

      /** \brief  Base class for all VAST nodes describing control-flow statements. */
      class ControlFlowStatement : public Node
      {
      protected:

        inline ControlFlowStatement()   {}

      public:

        virtual ~ControlFlowStatement() {}


        /** \brief  Returns, whether this control-flow statement needs to be vectorized. */
        virtual bool IsVectorized() const = 0;
      };

      /** \brief  Base class for all VAST nodes describing expressions. */
      class Expression : public Node
      {
      protected:

        inline Expression()   {}


        /** \brief  Dumps the result type of this expression into a XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        std::string _DumpResultTypeToXML(const size_t cszIntend) const;

        /** \brief    Returns the sub-expression index of a specified sub-expression.
         *  \param    spSubExpression   A constant shared pointer to the sub-expression, whose index shall be retrived.
         *  \remarks  If the specified sub-expression is not a child of the current expression, an exception will be thrown. */
        IndexType _FindSubExpressionIndex(ExpressionConstPtr spSubExpression) const;

      public:

        virtual ~Expression() {}


        /** \name Public methods inherited from class Node. */
        //@{

        virtual NodePtr   GetChild(IndexType ChildIndex) final override   { return GetSubExpression(ChildIndex); }
        virtual IndexType GetChildCount() const final override            { return GetSubExpressionCount(); }

        //@}


        /** \brief  Returns the sub-expression index of this expression in its parent expression, if the current expression is a sub-expression.
         *  \sa     IsSubExpression(). */
        IndexType GetParentIndex() const;

        /** \brief  Returns, whether the current expression is a sub-expression of another one. */
        bool      IsSubExpression() const;


        /** \brief  Returns, whether this expression is vectorized. */
        virtual bool      IsVectorized();

        /** \brief  Returns, whether this expression is vectorized. */
        inline  bool      IsVectorized() const  { return const_cast<Expression*>(this)->IsVectorized(); }


        /** \name Abstract methods implemented by the derived expression classes. */
        //@{

        /** \brief  Returns the result type of this expression. */
        virtual TypeInfo  GetResultType() const = 0;

        /** \brief  Returns a shared pointer to the sub-expression with the specified index.
         *  \param  SubExprIndex  The index of the requested sub-expression. */
        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) = 0;

        /** \brief  Returns the number of sub-expressions for this expression. */
        virtual IndexType       GetSubExpressionCount() const = 0;

        /** \brief  Replaces a specific sub-expression with another one.
         *  \param  SubExprIndex  The index of the sub-expression, which shall be replaced.
         *  \param  spSubExpr     A shared pointer to new sub-expression. */
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) = 0;

        //@}


        /** \brief  Returns a constant shared pointer to the sub-expression with the specified index.
         *  \param  SubExprIndex  The index of the requested sub-expression. */
        inline ExpressionConstPtr GetSubExpression(IndexType SubExprIndex) const  { return const_cast<Expression*>(this)->GetSubExpression(SubExprIndex); }
      };
    };


    /** \brief  Contains all class definitions for VAST nodes describing control-flow statements. */
    class ControlFlow final
    {
    public:

      /** \name Shared pointer type definitions */
      //@{

      class Loop;
      typedef std::shared_ptr<Loop>       LoopPtr;      //!< Shared pointer type for objects of class Loop
      typedef std::shared_ptr<const Loop> LoopConstPtr; //!< Shared pointer type for constant objects of class Loop

      class LoopControlStatement;
      typedef std::shared_ptr<LoopControlStatement>       LoopControlStatementPtr;      //!< Shared pointer type for objects of class LoopControlStatement
      typedef std::shared_ptr<const LoopControlStatement> LoopControlStatementConstPtr; //!< Shared pointer type for constant objects of class LoopControlStatement

      class ConditionalBranch;
      typedef std::shared_ptr<ConditionalBranch>       ConditionalBranchPtr;      //!< Shared pointer type for objects of class ConditionalBranch
      typedef std::shared_ptr<const ConditionalBranch> ConditionalBranchConstPtr; //!< Shared pointer type for constant objects of class ConditionalBranch

      class BranchingStatement;
      typedef std::shared_ptr<BranchingStatement>       BranchingStatementPtr;      //!< Shared pointer type for objects of class BranchingStatement
      typedef std::shared_ptr<const BranchingStatement> BranchingStatementConstPtr; //!< Shared pointer type for constant objects of class BranchingStatement

      class ReturnStatement;
      typedef std::shared_ptr<ReturnStatement>       ReturnStatementPtr;      //!< Shared pointer type for objects of class ReturnStatement
      typedef std::shared_ptr<const ReturnStatement> ReturnStatementConstPtr; //!< Shared pointer type for constant objects of class ReturnStatement

      //@}


    public:

      /** \brief  Describes all kinds of loop statements. */
      class Loop final : public BaseClasses::ControlFlowStatement
      {
      public:

        /** \brief  Enumeration of all supported loop types. */
        enum class LoopType
        {
          TopControlled,      //!< Internal ID for top-controlled loops, like <b>while-</b> and <b>for-</b>loops.
          BottomControlled    //!< Internal ID for bottom-controlled loops, like <b>do-while-</b>loops.
        };

      private:

        friend class AST;

        LoopType                    _eLoopType;
        BaseClasses::ExpressionPtr  _spConditionExpr;
        BaseClasses::ExpressionPtr  _spIncrementExpr;
        ScopePtr                    _spBody;
        bool                        _bForceVectorization;


        inline Loop() : _eLoopType(LoopType::TopControlled), _bForceVectorization(false)   {}

        /** \brief  Returns the string identifier of a specific loop type.
         *  \param  eType   The loop type, whose string identifier shall be returned. */
        static std::string _GetLoopTypeString(LoopType eType);


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eType         The requested type of this loop.
         *  \param  spCondition   A shared pointer to expression object, which shall be used as the condition.
         *  \param  spIncrement   A shared pointer to expression object, which describes the increment after each iteration. */
        static LoopPtr Create(LoopType eType = LoopType::TopControlled, BaseClasses::ExpressionPtr spCondition = nullptr, BaseClasses::ExpressionPtr spIncrement = nullptr);

        virtual ~Loop() {}


        /** \brief  Returns a shared pointer to the scope object, which encapsulates the loop body. */
        ScopePtr        GetBody();

        /** \brief  Returns a constant shared pointer to the scope object, which encapsulates the loop body. */
        ScopeConstPtr   GetBody() const;


        /** \brief  Returns a shared pointer to the condition expression object. */
        inline BaseClasses::ExpressionPtr         GetCondition()                                        { return _spConditionExpr; }

        /** \brief  Returns a constant shared pointer to the condition expression object. */
        inline BaseClasses::ExpressionConstPtr    GetCondition() const                                  { return _spConditionExpr; }

        /** \brief  Replaces the currently set condition expression object.
         *  \param  spCondition   A shared pointer to the expression object, which shall be used as the new condition. */
        inline void                               SetCondition(BaseClasses::ExpressionPtr spCondition)  { _SetChildPtr(_spConditionExpr, spCondition); }


        /** \brief  Returns a shared pointer to the increment expression object. */
        inline BaseClasses::ExpressionPtr         GetIncrement()                                        { return _spIncrementExpr; }

        /** \brief  Returns a constant shared pointer to the increment expression object. */
        inline BaseClasses::ExpressionConstPtr    GetIncrement() const                                  { return _spIncrementExpr; }

        /** \brief    Replaces the currently set increment expression object.
         *  \param    spIncrement   A shared pointer to the expression object, which shall be used as the new increment expression.
         *  \remarks  The increment expression is the expression, which will be called after each loop iteration. If this is not required,
                      it can be set to <b>nullptr</b>. */
        inline void                               SetIncrement(BaseClasses::ExpressionPtr spIncrement)  { _SetChildPtr(_spIncrementExpr, spIncrement); }


        /** \brief  Returns the currently set <b>forced vectorization</b> flag. */
        inline bool GetForcedVectorization() const          { return _bForceVectorization; }

        /** \brief  Sets a flag, which indicates whether the loop must be vectorized, even if its condition expression is not.
         *  \param  bForceVec   The new forced vectorization flag. */
        inline void SetForcedVectorization(bool bForceVec)  { _bForceVectorization = bForceVec; }


        /** \brief  Returns the currently set loop type. */
        inline LoopType GetLoopType() const           { return _eLoopType; }

        /** \brief  Changes the loop type.
         *  \param  eType   The new loop type. */
        inline void     SetLoopType(LoopType eType)   { _eLoopType = eType; }


      public:

        /** \name Public methods inherited from class BaseClasses::Node. */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override    { return static_cast< IndexType >( 3 ); }

        //@}


        /** \name Public methods inherited from class BaseClasses::ControlFlowStatement. */
        //@{

        virtual bool IsVectorized() const final override;

        //@}
      };

      /** \brief  Describes loop control statements, i.e. <b>break</b> and <b>continue</b> statements. */
      class LoopControlStatement final : public BaseClasses::ControlFlowStatement
      {
      public:

        /** \brief  Enumeration of all supported loop control types. */
        enum class LoopControlType
        {
          Break,    //!< Internal ID of the <b>break</b> statement.
          Continue  //!< Internal ID of the <b>continue</b> statement.
        };


      private:

        friend class AST;

        LoopControlType _eControlType;


        inline LoopControlStatement() : _eControlType(LoopControlType::Break)  {}


        /** \brief  Returns the string identifier of a specific loop control type.
         *  \param  eType   The loop control type, whose string identifier shall be returned. */
        static std::string _GetLoopControlTypeString(LoopControlType eType);


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eCtrlType   The requested type of this loop control statement. */
        static LoopControlStatementPtr Create(LoopControlType eCtrlType);

        virtual ~LoopControlStatement() {}


        /** \brief  Returns the currently set loop control type. */
        inline LoopControlType  GetControlType() const                        { return _eControlType; }

        /** \brief  Changes the loop control type.
         *  \param  eNewCtrlType   The new loop control type. */
        inline void             SetControlType(LoopControlType eNewCtrlType)  { _eControlType = eNewCtrlType; }


        /** \brief  Returns a shared pointer to the enclosing loop of this statement. */
        LoopPtr               GetControlledLoop();

        /** \brief  Returns a constant shared pointer to the enclosing loop of this statement. */
        inline LoopConstPtr   GetControlledLoop() const   { return const_cast<LoopControlStatement*>(this)->GetControlledLoop(); }

      public:

        /** \name Public methods inherited from class BaseClasses::Node. */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override   { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType             GetChildCount() const final override            { return static_cast<IndexType>(0); }

        //@}


        /** \name Public methods inherited from class BaseClasses::ControlFlowStatement. */
        //@{

        virtual bool IsVectorized() const final override;

        //@}
      };

      /** \brief  Describes a conditional branch of a multi-branch control-flow statement. */
      class ConditionalBranch final : public BaseClasses::ControlFlowStatement
      {
      private:

        friend class AST;

        typedef BaseClasses::ExpressionPtr          ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr     ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

      private:

        ExpressionPtr   _spCondition;
        ScopePtr        _spBody;

        inline ConditionalBranch() : _spCondition(nullptr), _spBody(nullptr)   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  spCondition   A shared pointer to the expression object, which shall be used as the condition. */
        static ConditionalBranchPtr Create(ExpressionPtr spCondition = nullptr);

        virtual ~ConditionalBranch()  {}


        /** \brief  Returns a shared pointer to the scope object, which encapsulates the branch body. */
        ScopePtr        GetBody();

        /** \brief  Returns a constant shared pointer to the scope object, which encapsulates the branch body. */
        ScopeConstPtr   GetBody() const;


        /** \brief  Returns a shared pointer to the condition expression object. */
        inline ExpressionPtr        GetCondition()                            { return _spCondition; }

        /** \brief  Returns a constant shared pointer to the condition expression object. */
        inline ExpressionConstPtr   GetCondition() const                      { return _spCondition; }

        /** \brief  Replaces the currently set condition expression object.
         *  \param  spCondition   A shared pointer to the expression object, which shall be used as the new condition. */
        inline void                 SetCondition(ExpressionPtr spCondition)   { _SetChildPtr(_spCondition, spCondition); }


      public:

        /** \name Public methods inherited from class BaseClasses::Node. */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override { return static_cast<IndexType>(2); }

        //@}


        /** \name Public methods inherited from class BaseClasses::ControlFlowStatement. */
        //@{

        virtual bool IsVectorized() const final override;

        //@}
      };

      /** \brief  Describes a multi-branch control-flow statement, e.g. an <b>if-statement</b>. */
      class BranchingStatement final : public BaseClasses::ControlFlowStatement
      {
      private:

        friend class AST;

        std::vector<ConditionalBranchPtr> _vecBranches;
        ScopePtr                          _spDefaultBranch;


        inline BranchingStatement() : _spDefaultBranch(nullptr)   {}


      public:

        /** \brief  Creates a new object of this class. */
        static BranchingStatementPtr Create();

        virtual ~BranchingStatement() {}


        /** \brief  Adds another conditional branch at the end of this statement.
         *  \param  spBranch  A shared pointer to the conditional branch, which shall be added. */
        void                  AddConditionalBranch(ConditionalBranchPtr spBranch);

        /** \brief  Returns a shared pointer to the conditional branch object with a specified index.
         *  \param  BranchIndex   The index of the requested conditional branch. */
        ConditionalBranchPtr  GetConditionalBranch(IndexType BranchIndex);

        /** \brief  Returns the number of conditional branches in this statement. */
        inline IndexType      GetConditionalBranchesCount() const { return static_cast<IndexType>(_vecBranches.size()); }

        /** \brief  Removes the conditional branch with a specified index for this statement.
         *  \param  BranchIndex   The index of the conditional branch, which shall be removed. */
        void                  RemoveConditionalBranch(IndexType BranchIndex);


        /** \brief  Returns a shared pointer to the scope object, which encapsulates the unconditional default branch. */
        ScopePtr        GetDefaultBranch();

        /** \brief  Returns a constant shared pointer to the scope object, which encapsulates the unconditional default branch. */
        ScopeConstPtr   GetDefaultBranch() const;


        /** \brief  Returns a constant shared pointer to the conditional branch object with a specified index.
         *  \param  BranchIndex   The index of the requested conditional branch. */
        inline ConditionalBranchConstPtr  GetConditionalBranch(IndexType BranchIndex) const
        {
          return const_cast<BranchingStatement*>(this)->GetConditionalBranch(BranchIndex);
        }


      public:

        /** \name Public methods inherited from class BaseClasses::Node. */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override;
        virtual IndexType             GetChildCount() const final override { return GetConditionalBranchesCount() + 1; }

        //@}


        /** \name Public methods inherited from class BaseClasses::ControlFlowStatement. */
        //@{

        virtual bool IsVectorized() const final override;

        //@}
      };

      /** \brief    Describes a <b>return</b> statement.
       *  \remarks  Since at the current stage of development only void-functions can be expressed, these statements do not contain a return expression. */
      class ReturnStatement final : public BaseClasses::ControlFlowStatement
      {
      private:

        friend class AST;

        inline ReturnStatement()  {}

      public:

        /** \brief  Creates a new object of this class. */
        static ReturnStatementPtr Create();

        virtual ~ReturnStatement() {}

      public:

        /** \name Public methods inherited from class BaseClasses::Node. */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        virtual BaseClasses::NodePtr  GetChild(IndexType ChildIndex) final override { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType             GetChildCount() const final override          { return static_cast<IndexType>(0); }

        //@}


        /** \name Public methods inherited from class BaseClasses::ControlFlowStatement. */
        //@{

        virtual bool IsVectorized() const final override;

        //@}
      };
    };


    /** \brief  Contains all class definitions for VAST nodes describing expressions. */
    class Expressions final
    {
    public:

      /** \name Shared pointer type definitions */
      //@{

      class Value;
      typedef std::shared_ptr<Value>       ValuePtr;      //!< Shared pointer type for objects of class Value
      typedef std::shared_ptr<const Value> ValueConstPtr; //!< Shared pointer type for constant objects of class Value

      class Constant;
      typedef std::shared_ptr<Constant>       ConstantPtr;      //!< Shared pointer type for objects of class Constant
      typedef std::shared_ptr<const Constant> ConstantConstPtr; //!< Shared pointer type for constant objects of class Constant

      class Identifier;
      typedef std::shared_ptr<Identifier>       IdentifierPtr;      //!< Shared pointer type for objects of class Identifier
      typedef std::shared_ptr<const Identifier> IdentifierConstPtr; //!< Shared pointer type for constant objects of class Identifier

      class MemoryAccess;
      typedef std::shared_ptr<MemoryAccess>       MemoryAccessPtr;      //!< Shared pointer type for objects of class MemoryAccess
      typedef std::shared_ptr<const MemoryAccess> MemoryAccessConstPtr; //!< Shared pointer type for constant objects of class MemoryAccess


      class UnaryExpression;
      typedef std::shared_ptr<UnaryExpression>       UnaryExpressionPtr;      //!< Shared pointer type for objects of class UnaryExpression
      typedef std::shared_ptr<const UnaryExpression> UnaryExpressionConstPtr; //!< Shared pointer type for constant objects of class UnaryExpression

      class Conversion;
      typedef std::shared_ptr<Conversion>       ConversionPtr;      //!< Shared pointer type for objects of class Conversion
      typedef std::shared_ptr<const Conversion> ConversionConstPtr; //!< Shared pointer type for constant objects of class Conversion

      class Parenthesis;
      typedef std::shared_ptr<Parenthesis>       ParenthesisPtr;      //!< Shared pointer type for objects of class Parenthesis
      typedef std::shared_ptr<const Parenthesis> ParenthesisConstPtr; //!< Shared pointer type for constant objects of class Parenthesis

      class UnaryOperator;
      typedef std::shared_ptr<UnaryOperator>       UnaryOperatorPtr;      //!< Shared pointer type for objects of class UnaryOperator
      typedef std::shared_ptr<const UnaryOperator> UnaryOperatorConstPtr; //!< Shared pointer type for constant objects of class UnaryOperator


      class BinaryOperator;
      typedef std::shared_ptr<BinaryOperator>       BinaryOperatorPtr;      //!< Shared pointer type for objects of class BinaryOperator
      typedef std::shared_ptr<const BinaryOperator> BinaryOperatorConstPtr; //!< Shared pointer type for constant objects of class BinaryOperator

      class ArithmeticOperator;
      typedef std::shared_ptr<ArithmeticOperator>       ArithmeticOperatorPtr;      //!< Shared pointer type for objects of class ArithmeticOperator
      typedef std::shared_ptr<const ArithmeticOperator> ArithmeticOperatorConstPtr; //!< Shared pointer type for constant objects of class ArithmeticOperator

      class AssignmentOperator;
      typedef std::shared_ptr<AssignmentOperator>       AssignmentOperatorPtr;     //!< Shared pointer type for objects of class AssignmentOperator
      typedef std::shared_ptr<const AssignmentOperator> AssignmentOperatorConstPtr; //!< Shared pointer type for constant objects of class AssignmentOperator

      class RelationalOperator;
      typedef std::shared_ptr<RelationalOperator>       RelationalOperatorPtr;      //!< Shared pointer type for objects of class RelationalOperator
      typedef std::shared_ptr<const RelationalOperator> RelationalOperatorConstPtr; //!< Shared pointer type for constant objects of class RelationalOperator


      class FunctionCall;
      typedef std::shared_ptr<FunctionCall>       FunctionCallPtr;      //!< Shared pointer type for objects of class FunctionCall
      typedef std::shared_ptr<const FunctionCall> FunctionCallConstPtr; //!< Shared pointer type for constant objects of class FunctionCall

      //@}


    public:

      /** \brief  Base class for all VAST nodes, which specify or access value. */
      class Value : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::ExpressionPtr  ExpressionPtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.

      protected:

        inline Value()  {}

      public:

        virtual ~Value()  {}


        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) override                           { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType       GetSubExpressionCount() const override                                      { return static_cast<IndexType>(0); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) override  { throw ASTExceptions::ChildIndexOutOfRange(); }
        //@}
      };

      /** \brief  Describes all kinds of compile-time constant values, e.g. integer literals. */
      class Constant final : public Value
      {
      private:

        friend class AST;
        typedef BaseClasses::TypeInfo::KnownTypes   KnownTypes;   //!< Type alias for the enumeration of known element types.

        union
        {
          std::uint64_t ui64IntegralValue;
          double        dFloatingPointValue;
        } _unionValues;

        KnownTypes    _eType;


        inline Constant()   {}


        /** \brief    Generic internal method, which handles changes of the numeric type of this constant.
         *  \tparam   SourceValueType   The internal numeric type of this constant before the type-changing action.
         *  \param    eNewType          The requested new element type of this constant. */
        template <typename SourceValueType>
        inline void _ChangeType(KnownTypes eNewType)
        {
          SourceValueType TValue = GetValue<SourceValueType>();

          switch (eNewType)
          {
          case KnownTypes::Bool:    SetValue( TValue != static_cast<SourceValueType>(0) ); break;
          case KnownTypes::Int8:    SetValue( static_cast<std::int8_t  >( TValue ) );      break;
          case KnownTypes::UInt8:   SetValue( static_cast<std::uint8_t >( TValue ) );      break;
          case KnownTypes::Int16:   SetValue( static_cast<std::int16_t >( TValue ) );      break;
          case KnownTypes::UInt16:  SetValue( static_cast<std::uint16_t>( TValue ) );      break;
          case KnownTypes::Int32:   SetValue( static_cast<std::int32_t >( TValue ) );      break;
          case KnownTypes::UInt32:  SetValue( static_cast<std::uint32_t>( TValue ) );      break;
          case KnownTypes::Int64:   SetValue( static_cast<std::int64_t >( TValue ) );      break;
          case KnownTypes::UInt64:  SetValue( static_cast<std::uint64_t>( TValue ) );      break;
          case KnownTypes::Float:   SetValue( static_cast<float        >( TValue ) );      break;
          case KnownTypes::Double:  SetValue( static_cast<double       >( TValue ) );      break;
          default:                  throw RuntimeErrorException(std::string("Invalid constant type: ") + BaseClasses::TypeInfo::GetTypeString(eNewType));
          }
        }

      public:

        /** \brief  Creates a new object of this class.
         *  \tparam ValueType   The numeric type, the Constant object shall represent. Allowed types are integral and floating-point types, as well as <b>bool</b>.
         *  \param  TValue      The actual value, which shall be stored by this constant. */
        template <typename ValueType>
        static ConstantPtr Create(ValueType TValue)
        {
          ConstantPtr spConstant = AST::CreateNode<Constant>();

          spConstant->SetValue(TValue);

          return spConstant;
        }

        virtual ~Constant() {}


        /** \brief  Returns the currently set element type of this constant. */
        inline BaseClasses::TypeInfo::KnownTypes GetValueType() const { return _eType; }


        /** \brief    Changes the internal numeric type of this constant.
         *  \param    eNewType  The requested new element type of this constant.
         *  \remarks  This method can be used to implement cast operations on constants. */
        void  ChangeType(KnownTypes eNewType);

        /** \brief    Returns the internally stored constant value.
         *  \tparam   ValueType   The requested numeric type of the return value. Allowed types are integral and floating-point types, as well as <b>bool</b>.
         *  \remarks  If the requested output value type does not equal the internal constant type, a loss of information can occur in the return value due to truncation. */
        template <typename ValueType> inline ValueType GetValue() const
        {
          static_assert(std::is_arithmetic<ValueType>::value, "Expected a numeric value type!");

          switch (_eType)
          {
          case KnownTypes::Float: case KnownTypes::Double:
            return static_cast<ValueType>(_unionValues.dFloatingPointValue);
          default:
            return static_cast<ValueType>(_unionValues.ui64IntegralValue);
          }
        }

        /** \brief  Sets a new internal value.
         *  \tparam ValueType   The numeric type, the Constant object shall represent. Allowed types are integral and floating-point types, as well as <b>bool</b>.
         *  \param  TValue      The requested new value, which shall be stored by this constant. */
        template <typename ValueType> inline void SetValue(ValueType TValue)
        {
          static_assert(std::is_arithmetic<ValueType>::value, "Expected a numeric value type!");

          if (std::is_integral<ValueType>::value)
          {
            _unionValues.ui64IntegralValue  = static_cast<std::uint64_t>(TValue);

            bool bSigned = std::numeric_limits<ValueType>::is_signed;

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
            _unionValues.dFloatingPointValue = static_cast<double>(TValue);

            _eType = (sizeof(ValueType) == 4) ? KnownTypes::Float : KnownTypes::Double;
          }
        }

      public:

        /** \brief  Returns the string representation of the internally stored constant value. */
        std::string GetAsString() const;

      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };

      /** \brief  Describes a simple reference to a declared variable in the current AST. */
      class Identifier final : public Value
      {
      private:

        friend class AST;

      private:

        std::string   _strName;

        inline Identifier()   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  strName   The unique name of the referenced variable. */
        static IdentifierPtr Create( std::string strName );

        virtual ~Identifier() {}


        /** \brief  Returns the name of the currently referenced variable. */
        inline std::string  GetName() const               { return _strName; }

        /** \brief  Sets a name referenced variable name.
         *  \param  strName   The new unique name of the variable, which shall be referenced. */
        inline void         SetName(std::string strName)  { _strName = strName; }


        /** \brief    Returns a shared pointer to the VariableInfo object, which describes the declaration of the referenced variable.
         *  \remarks  This method requires the VAST node to be correctly linked into an enclosing AST, because it retrives the variable declaration
         *            from the root <b>FunctionDeclaration</b> object. */
        BaseClasses::VariableInfoPtr LookupVariableInfo();

        /** \brief    Returns a constant shared pointer to the VariableInfo object, which describes the declaration of the referenced variable.
         *  \remarks  This method requires the VAST node to be correctly linked into an enclosing AST, because it retrives the variable declaration
         *            from the root <b>FunctionDeclaration</b> object. */
        inline BaseClasses::VariableInfoConstPtr LookupVariableInfo() const
        {
          return const_cast< Identifier* >( this )->LookupVariableInfo();
        }

      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        virtual bool                  IsVectorized() final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };

      /** \brief    Describes all kinds of memory accesses.
       *  \remarks  This class is used to describe array accesses as well as memory transactions with pointers and offsets.
                    The unary dereferencing operator can be described by setting a Constant object with <b>zero</b> value as index expression. */
      class MemoryAccess final : public Value
      {
      private:

        friend class AST;
        typedef BaseClasses::ExpressionPtr        ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr   ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

      private:

        ExpressionPtr   _spMemoryRef;
        ExpressionPtr   _spIndexExpr;

        inline MemoryAccess() : _spMemoryRef(nullptr), _spIndexExpr(nullptr)   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  spMemoryReference   A shared pointer to the expression object, which returns the memory reference. It must evaluate to a dereferencable type.
         *  \param  spIndexExpression   A shared pointer to the expression object, which returns the offset to the memory address. It must evaluate to a native element type. */
        static MemoryAccessPtr Create(ExpressionPtr spMemoryReference = nullptr, ExpressionPtr spIndexExpression = nullptr);

        virtual ~MemoryAccess() {}


        /** \brief  Returns a shared pointer to the currently set index expression object. */
        inline ExpressionPtr        GetIndexExpression()                                  { return _spIndexExpr; }

        /** \brief  Returns a constant shared pointer to the currently set index expression object. */
        inline ExpressionConstPtr   GetIndexExpression() const                            { return _spIndexExpr; }

        /** \brief  Sets a new expression, which returns the offset to the memory address. It must evaluate to a native element type.
         *  \param  spIndexExpression   A shared pointer to the new <b>index</b> expression. */
        inline void                 SetIndexExpression(ExpressionPtr spIndexExpression)   { _SetChildPtr(_spIndexExpr, spIndexExpression); }


        /** \brief  Returns a shared pointer to the currently set expression object, which returns the memory reference. */
        inline ExpressionPtr        GetMemoryReference()                                  { return _spMemoryRef; }

        /** \brief  Returns a constant shared pointer to the currently set expression object, which returns the memory reference. */
        inline ExpressionConstPtr   GetMemoryReference() const                            { return _spMemoryRef; }

        /** \brief  Sets a new expression, which returns the memory reference. It must evaluate to a dereferencable type.
         *  \param  spMemoryReference   A shared pointer to the new <b>memory reference</b> expression. */
        inline void                 SetMemoryReference(ExpressionPtr spMemoryReference)   { _SetChildPtr(_spMemoryRef, spMemoryReference); }

      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType       GetSubExpressionCount() const final override { return static_cast<IndexType>(2); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };


      /** \brief    Base class for all VAST nodes describing typical unary expressions.
       *  \remarks  All unary expressions contain exactly one sub-expression. */
      class UnaryExpression : public BaseClasses::Expression
      {
      private:

        BaseClasses::ExpressionPtr  _spSubExpression;

      protected:

        /** \brief  Internal method, which dumps the sub-expression into a XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        std::string _DumpSubExpressionToXML(const size_t cszIntend) const;

        inline UnaryExpression() : _spSubExpression(nullptr)  {}

      public:

        virtual ~UnaryExpression()  {}


        /** \brief  Returns a shared pointer to the currently referenced sub-expression. */
        inline BaseClasses::ExpressionPtr       GetSubExpression()                                      { return _spSubExpression; }

        /** \brief  Returns a constant shared pointer to the currently referenced sub-expression. */
        inline BaseClasses::ExpressionConstPtr  GetSubExpression() const                                { return _spSubExpression; }

        /** \brief    Sets a new sub-expression.
         *  \param    spSubExpr   A shared pointer to the new sub-expression. */
        inline void                             SetSubExpression(BaseClasses::ExpressionPtr spSubExpr)  { _SetChildPtr(_spSubExpression, spSubExpr); }


        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::ExpressionPtr GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType                  GetSubExpressionCount() const final override { return static_cast<IndexType>(1); }
        virtual void                       SetSubExpression(IndexType SubExprIndex, BaseClasses::ExpressionPtr spSubExpr) final override;
        //@}
      };

      /** \brief  Describes all kinds of conversion expressions, including casts. */
      class Conversion final : public UnaryExpression
      {
      private:

        friend class AST;

      private:

        BaseClasses::TypeInfo   _ConvertType;
        bool                    _bIsExplicit;

        inline Conversion() : _bIsExplicit(true)   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  crConvertType     A constant reference to the TypeInfo object describing the return type of the conversion. Its contents will be copied.
         *  \param  spSubExpression   A shared pointer to the expression object, which shall be used as the sub-expression.
         *  \param  bExplicit         A flag indicating, whether this conversion expression has been explicitly stated by the programmer. */
        static ConversionPtr Create(const BaseClasses::TypeInfo &crConvertType, BaseClasses::ExpressionPtr spSubExpression = nullptr, bool bExplicit = true);

        virtual ~Conversion() {}


        /** \brief  Returns the currently set target type of the conversion. */
        inline BaseClasses::TypeInfo  GetConvertType() const                                    { return _ConvertType; }

        /** \brief  Sets a new target type for the conversion.
         *  \param  crConvType    A constant reference to the TypeInfo object describing the return type of the conversion. Its contents will be copied. */
        inline void                   SetConvertType(const BaseClasses::TypeInfo &crConvType)   { _ConvertType = crConvType; }


        /** \brief  Returns the currently set <b>explicit conversion</b> flag. */
        inline bool GetExplicit() const             { return _bIsExplicit; }

        /** \brief  Sets a flag, which indicates whether this conversion has been explicitly stated by the programmer.
         *  \param  bIsExplicit   The new explicit conversion flag. */
        inline void SetExplicit(bool bIsExplicit)   { _bIsExplicit = bIsExplicit; }

      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override    { return GetConvertType(); }
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };

      /** \brief  Describes the encapsulation of the sub-expression into a parenthesis. */
      class Parenthesis final : public UnaryExpression
      {
      private:

        friend class AST;

        inline Parenthesis()  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  spSubExpression   A shared pointer to the expression object, which shall be used as the sub-expression. */
        static ParenthesisPtr Create(BaseClasses::ExpressionPtr spSubExpression = nullptr);

        virtual ~Parenthesis()  {}


      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };

      /** \brief  Describes all kinds of typical unary operators, except the unary dereferencing operator, which is handled by class <b>MemoryAccess</b>. */
      class UnaryOperator final : public UnaryExpression
      {
      private:

        friend class AST;

      public:

        /** \brief  Enumeration of all supported unary operator types. */
        enum class UnaryOperatorType
        {
          AddressOf,        //!< Internal ID of the address-of operator, i.e. <b>&amp;x</b>. The return type is a pointer to the sub-expression return type.
          BitwiseNot,       //!< Internal ID of the <b>bit-wise</b> not-operator, i.e. <b>!x</b>. The return type of the sub-expression is kept.
          LogicalNot,       //!< Internal ID of the <b>logical</b> not-operator, i.e. <b>!x</b>. The return type of this operator is <b>boolean</b>.
          Minus,            //!< Internal ID of the unary sign operator <b>minus</b>, i.e. <b>-x</b>.
          Plus,             //!< Internal ID of the unary sign operator <b>plus</b>, i.e. <b>+x</b>.
          PostDecrement,    //!< Internal ID of the post-decrement operator, i.e. <b>x--</b>.
          PostIncrement,    //!< Internal ID of the post-increment operator, i.e. <b>x++</b>.
          PreDecrement,     //!< Internal ID of the pre-decrement operator, i.e. <b>--x</b>.
          PreIncrement      //!< Internal ID of the pre-increment operator, i.e. <b>++x</b>.
        };

      private:

        UnaryOperatorType    _eOpType;

        inline UnaryOperator() : _eOpType(UnaryOperatorType::Plus)  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eType             The requested type of the unary operator.
         *  \param  spSubExpression   A shared pointer to the expression object, which shall be used as the sub-expression. */
        static UnaryOperatorPtr Create(UnaryOperatorType eType = UnaryOperatorType::Plus, BaseClasses::ExpressionPtr spSubExpression = nullptr);

        virtual ~UnaryOperator()  {}


        /** \brief  Returns the string identifier of a specific unary operator type.
         *  \param  eType   The unary operator type, whose string identifier shall be returned. */
        static std::string GetOperatorTypeString(UnaryOperatorType eType);


        /** \brief  Returns the currently set unary operator type. */
        inline UnaryOperatorType  GetOperatorType() const                     { return _eOpType; }

        /** \brief  Changes the unary operator type.
         *  \param  eOpType   The requested new unary operator type. */
        inline void               SetOperatorType(UnaryOperatorType eOpType)  { _eOpType = eOpType; }


      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}
      };


      /** \brief  Base class for all VAST nodes describing typical binary operators. */
      class BinaryOperator : public BaseClasses::Expression
      {
      private:

        typedef BaseClasses::ExpressionPtr        ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr   ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

        ExpressionPtr             _spLHS;
        ExpressionPtr             _spRHS;

      protected:

        /** \brief  Internal method, which dumps both sub-expressions into a XML string.
         *  \param  cszIntend   The intendation level, which shall be used for each line in the XML string, in space characters. */
        std::string _DumpSubExpressionsToXML(const size_t cszIntend) const;

        inline BinaryOperator()   {}

      public:

        virtual ~BinaryOperator() {}


        /** \brief  Returns a shared pointer to the currently referenced sub-expression, which is used as the left-hand-side. */
        inline ExpressionPtr        GetLHS()                        { return _spLHS; }

        /** \brief  Returns a constant shared pointer to the currently referenced sub-expression, which is used as the left-hand-side. */
        inline ExpressionConstPtr   GetLHS() const                  { return _spLHS; }

        /** \brief  Sets a new sub-expression as the left-hand-side.
         *  \param  spNewLHS  A shared pointer to the new sub-expression. */
        inline void                 SetLHS(ExpressionPtr spNewLHS)  { _SetChildPtr(_spLHS, spNewLHS); }


        /** \brief  Returns a shared pointer to the currently referenced sub-expression, which is used as the right-hand-side. */
        inline ExpressionPtr        GetRHS()                        { return _spRHS; }

        /** \brief  Returns a constant shared pointer to the currently referenced sub-expression, which is used as the right-hand-side. */
        inline ExpressionConstPtr   GetRHS() const                  { return _spRHS; }

        /** \brief  Sets a new sub-expression as the right-hand-side.
         *  \param  spNewRHS  A shared pointer to the new sub-expression. */
        inline void                 SetRHS(ExpressionPtr spNewRHS)  { _SetChildPtr(_spRHS, spNewRHS); }

      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) override;
        virtual IndexType     GetSubExpressionCount() const override { return static_cast<IndexType>(2); }
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) override;
        //@}
      };

      /** \brief  Describes all kinds of binary arithmetic operators. */
      class ArithmeticOperator final : public BinaryOperator
      {
      private:

        friend class AST;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.

      public:

        /** \brief  Enumeration of all supported arithmetic operator types. */
        enum class ArithmeticOperatorType
        {
          Add,            //!< Internal ID of the addition operation, i.e. <b>a + b</b>.
          BitwiseAnd,     //!< Internal ID of the bit-wise <b>and</b> operation, i.e. <b>a &amp; b</b>.
          BitwiseOr,      //!< Internal ID of the bit-wise <b>or</b> operation, i.e. <b>a | b</b>.
          BitwiseXOr,     //!< Internal ID of the bit-wise <b>exclusive-or</b> operation, i.e. <b>a ^ b</b>.
          Divide,         //!< Internal ID of the division operation, i.e. <b>a / b</b>.
          Modulo,         //!< Internal ID of the modulo operation, i.e. <b>a % b</b>.
          Multiply,       //!< Internal ID of the multiplication operation, i.e. <b>a * b</b>.
          ShiftLeft,      //!< Internal ID of the left-shift operation, i.e. <b>a &lt;&lt; b</b>.
          ShiftRight,     //!< Internal ID of the right-shift operation, i.e. <b>a &gt;&gt; b</b>.
          Subtract        //!< Internal ID of the subtraction operation, i.e. <b>a - b</b>.
        };


      private:

        ArithmeticOperatorType    _eOpType;

        inline ArithmeticOperator() : _eOpType(ArithmeticOperatorType::Add)  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eOpType   The requested type of the arithmetic operator.
         *  \param  spLHS     A shared pointer to the expression object, which shall be used as left-hand-side of this binary operator.
         *  \param  spRHS     A shared pointer to the expression object, which shall be used as right-hand-side of this binary operator. */
        static ArithmeticOperatorPtr Create(ArithmeticOperatorType eOpType = ArithmeticOperatorType::Add, ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr);

        virtual ~ArithmeticOperator() {}


        /** \brief  Returns the string identifier of a specific arithmetic operator type.
         *  \param  eType   The arithmetic operator type, whose string identifier shall be returned. */
        static std::string GetOperatorTypeString(ArithmeticOperatorType eType);


        /** \brief  Returns the currently set arithmetic operator type. */
        inline ArithmeticOperatorType GetOperatorType() const                           { return _eOpType; }

        /** \brief  Changes the arithmetic operator type.
         *  \param  eOpType   The requested new arithmetic operator type. */
        inline void                   SetOperatorType(ArithmeticOperatorType eOpType)   { _eOpType = eOpType; }

      public:

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        //@}
      };

      /** \brief    Describes the assignment operator.
       *  \remarks  This operator can receive an additional Identifier object referencing a <b>boolean</b> mask. If this is the case,
       *            the assignment shall only be performed, when the mask evaluates to <b>true</b>. */
      class AssignmentOperator final : public BinaryOperator
      {
      private:

        friend class AST;
        typedef BinaryOperator              BaseType;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.

      private:

        IdentifierPtr   _spMask;

        inline AssignmentOperator() : _spMask(nullptr)  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  spLHS   A shared pointer to the expression object, which shall be used as left-hand-side of this binary operator.
         *  \param  spRHS   A shared pointer to the expression object, which shall be used as right-hand-side of this binary operator.
         *  \param  spMask  A shared pointer to the optional identifier object, which shall be used as a condition mask for the assignment. */
        static AssignmentOperatorPtr Create(ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr, IdentifierPtr spMask = nullptr);

        virtual ~AssignmentOperator() {}


        /** \brief  Returns a shared pointer to the currently set mask identifier. */
        inline IdentifierPtr        GetMask()                         { return _spMask; }

        /** \brief  Returns a constant shared pointer to the currently set mask identifier. */
        inline IdentifierConstPtr   GetMask() const                   { return _spMask; }

        /** \brief    Sets a new mask identifier for conditional assignments.
         *  \param    spNewMask   A shared pointer to the new mask identifier.
         *  \remarks  If the assignment shall not be masked, set a <b>nullptr</b> as mask identifier. */
        inline void                 SetMask(IdentifierPtr spNewMask)  { _SetChildPtr(_spMask, spNewMask); }


        /** \brief  Returns, whether the assignment is conditional, i.e. whether a mask identifier has been set. */
        inline bool IsMasked() const { return static_cast<bool>(GetMask()); }

      public:

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;

        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType     GetSubExpressionCount() const final override;
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;
        //@}
      };

      /** \brief  Describes all kinds of binary relational operators, meaning operators returning <b>boolean</b> values regardless of the input types. */
      class RelationalOperator final : public BinaryOperator
      {
      private:

        friend class AST;
        typedef BaseClasses::ExpressionPtr  ExpressionPtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.

      public:

        /** \brief  Enumeration of all supported relational operator types. */
        enum class RelationalOperatorType
        {
          Equal,          //!< Internal ID of the equal-to comparison, i.e. <b>a == b</b>.
          Greater,        //!< Internal ID of the greater-than comparison, i.e. <b>a &gt; b</b>.
          GreaterEqual,   //!< Internal ID of the greater-than-or-equal-to comparison, i.e. <b>a &gt;= b</b>.
          Less,           //!< Internal ID of the less-than comparison, i.e. <b>a &lt; b</b>.
          LessEqual,      //!< Internal ID of the less-than-or-equal-to comparison, i.e. <b>a &lt;= b</b>.
          LogicalAnd,     //!< Internal ID of the logical-and combination, i.e. <b>a &amp;&amp; b</b>.
          LogicalOr,      //!< Internal ID of the logical-or combination, i.e. <b>a || b</b>.
          NotEqual        //!< Internal ID of the not-equal-to comparison, i.e. <b>a != b</b>.
        };

      private:

        RelationalOperatorType  _eOpType;

        inline RelationalOperator() : _eOpType(RelationalOperatorType::Equal)  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eOpType   The requested type of the relational operator.
         *  \param  spLHS     A shared pointer to the expression object, which shall be used as left-hand-side of this binary operator.
         *  \param  spRHS     A shared pointer to the expression object, which shall be used as right-hand-side of this binary operator. */
        static RelationalOperatorPtr  Create(RelationalOperatorType eOpType = RelationalOperatorType::Equal, ExpressionPtr spLHS = nullptr, ExpressionPtr spRHS = nullptr);

        virtual ~RelationalOperator() {}


        /** \brief  Returns the string identifier of a specific relational operator type.
         *  \param  eType   The relational operator type, whose string identifier shall be returned. */
        static std::string GetOperatorTypeString(RelationalOperatorType eType);


        /** \brief    Returns the internal type used during the relational operation.
         *  \remarks  The internal comparison type depends on the return types of the sub-expressions. */
        BaseClasses::TypeInfo GetComparisonType() const;


        /** \brief  Returns the currently set relational operator type. */
        inline RelationalOperatorType GetOperatorType() const                           { return _eOpType; }

        /** \brief  Changes the relational operator type.
         *  \param  eOpType   The requested new relational operator type. */
        inline void                   SetOperatorType(RelationalOperatorType eOpType)   { _eOpType = eOpType; }

      public:

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override;
        //@}
      };


      /** \brief    Describes a call to another function.
       *  \remarks  The number of sub-expressions of this VAST node is equal to the number of call parameters of the referenced function. */
      class FunctionCall : public BaseClasses::Expression
      {
      private:

        friend class AST;

        typedef BaseClasses::ExpressionPtr        ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr   ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

      private:

        std::string                     _strName;
        BaseClasses::TypeInfo           _ReturnType;
        std::vector< ExpressionPtr >    _vecCallParams;

        inline FunctionCall()   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  strFunctionName   The fully qualified name of the function, which shall be called.
         *  \param  crReturnType      A constant reference to the TypeInfo object describing the return type of the function call. Its contents will be copied. */
        static FunctionCallPtr Create(std::string strFunctionName, const BaseClasses::TypeInfo &crReturnType);

        virtual ~FunctionCall() {}


        /** \name Public methods for accessing the call parameter expression container. */
        //@{

        /** \brief  Adds a new function call parameter at the end of the current parameter list.
         *  \param  spCallParam   A shared pointer to the expression object, describing the new function call parameter. */
        void          AddCallParameter(ExpressionPtr spCallParam);

        /** \brief    Returns a shared pointer to the expression object, which is used as a specific function call parameter.
         *  \param    CallParamIndex  The index of the function call parameter, which shall be returned.
         *  \remarks  If the function call parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
        ExpressionPtr GetCallParameter(IndexType CallParamIndex);

        /** \brief    Replaces an existing function call parameter with a new one.
         *  \param    CallParamIndex  The index of the function call parameter, which shall be replaced.
         *  \param    spCallParam     A shared pointer to the expression object, describing the new function call parameter.
         *  \remarks  If the function call parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
        void          SetCallParameter(IndexType CallParamIndex, ExpressionPtr spCallParam);


        /** \brief  Returns the number of currently set call parameter expressions. */
        inline IndexType          GetCallParameterCount() const    { return static_cast< IndexType >(_vecCallParams.size()); }

        /** \brief    Returns a constant shared pointer to the expression object, which is used as a specific function call parameter.
         *  \param    CallParamIndex  The index of the function call parameter, which shall be returned.
         *  \remarks  If the function call parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
        inline ExpressionConstPtr GetCallParameter(IndexType CallParamIndex) const
        {
          return const_cast< FunctionCall* >( this )->GetCallParameter( CallParamIndex );
        }

        //@}


        /** \brief  Returns the currently set fully qualified name of the function, which shall be called. */
        inline std::string  GetName() const                   { return _strName; }

        /** \brief    Sets a new function name.
         *  \param    strNewName  The requested new function name.
         *  \remarks  The function name is expected to be fully qualified, i.e. it should contain the name of all enclosing namespaces. */
        inline void         SetName(std::string strNewName)   { _strName = strNewName; }


        /** \brief  Returns the currently set return type of the function call. */
        inline BaseClasses::TypeInfo GetReturnType() const                                      { return _ReturnType; }

        /** \brief  Sets a new return type of the function call.
         *  \param  crReturnType    A constant reference to the TypeInfo object describing the return type of the function call. Its contents will be copied. */
        inline void                  SetReturnType(const BaseClasses::TypeInfo &crReturnType)   { _ReturnType = crReturnType; }

      public:

        /** \name Public methods inherited from class BaseClasses::Node */
        //@{
        virtual std::string DumpToXML(const size_t cszIntend) const final override;
        //@}

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{
        virtual BaseClasses::TypeInfo GetResultType() const final override    { return GetReturnType(); }

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override                           { return GetCallParameter(SubExprIndex); }
        virtual IndexType       GetSubExpressionCount() const final override                                      { return GetCallParameterCount(); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override  { SetCallParameter(SubExprIndex, spSubExpr); }
        //@}
      };
    };


    /** \brief  Contains all class definitions for VAST nodes describing special vector expressions. */
    class VectorSupport final
    {
    public:

      /** \name Shared pointer type definitions */
      //@{

      class VectorExpression;
      typedef std::shared_ptr<VectorExpression>       VectorExpressionPtr;      //!< Shared pointer type for objects of class VectorExpression
      typedef std::shared_ptr<const VectorExpression> VectorExpressionConstPtr; //!< Shared pointer type for constant objects of class VectorExpression

      class BroadCast;
      typedef std::shared_ptr<BroadCast>       BroadCastPtr;      //!< Shared pointer type for objects of class BroadCast
      typedef std::shared_ptr<const BroadCast> BroadCastConstPtr; //!< Shared pointer type for constant objects of class BroadCast

      class CheckActiveElements;
      typedef std::shared_ptr<CheckActiveElements>       CheckActiveElementsPtr;      //!< Shared pointer type for objects of class CheckActiveElements
      typedef std::shared_ptr<const CheckActiveElements> CheckActiveElementsConstPtr; //!< Shared pointer type for constant objects of class CheckActiveElements

      class VectorIndex;
      typedef std::shared_ptr<VectorIndex>       VectorIndexPtr;      //!< Shared pointer type for objects of class VectorIndex
      typedef std::shared_ptr<const VectorIndex> VectorIndexConstPtr; //!< Shared pointer type for constant objects of class VectorIndex

      //@}


    public:

      /** \brief  Base class for all special vector expression classes. */
      class VectorExpression : public BaseClasses::Expression
      {
      protected:

        inline VectorExpression()   {}

      public:

        virtual ~VectorExpression()  {}
      };

      /** \brief    Describes an unary expression, which broadcasts the result of a scalar sub-expression into all elements of a vector.
       *  \remarks  The result type of this expression is identical to the one of its sub-expression, but the vectorization marker will be set. */
      class BroadCast final : public VectorExpression
      {
      private:

        friend class AST;

        typedef BaseClasses::ExpressionPtr        ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr   ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

      private:

        ExpressionPtr _spSubExpression;

        inline BroadCast() : _spSubExpression(nullptr)   {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  spSubExpression   A shared pointer to the expression object, which shall be used as the sub-expression. */
        static BroadCastPtr Create(ExpressionPtr spSubExpression = nullptr);

        virtual ~BroadCast()  {}


        /** \brief  Returns a shared pointer to the currently referenced sub-expression. */
        inline ExpressionPtr        GetSubExpression()                          { return _spSubExpression; }

        /** \brief  Returns a constant shared pointer to the currently referenced sub-expression. */
        inline ExpressionConstPtr   GetSubExpression() const                    { return _spSubExpression; }

        /** \brief    Sets a new sub-expression.
         *  \param    spSubExpr   A shared pointer to the new sub-expression.
         *  \remarks  The sub-expression should return a scalar element value, which can be broadcasted into all elements of a vector. */
        inline void                 SetSubExpression(ExpressionPtr spSubExpr)   { _SetChildPtr(_spSubExpression, spSubExpr); }


      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{

        virtual bool IsVectorized() final override      { return true; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override;

        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType     GetSubExpressionCount() const final override { return static_cast<IndexType>(1); }
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;

        // @}


        /** \name Public methods inherited from class BaseClasses::Node */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        // @}
      };

      /** \brief    Describes an unary expression, which checks whether the elements inside of a vector mask fulfil a certain criterion.
       *  \remarks  The result type of this expression is always a scalar boolean value. */
      class CheckActiveElements final : public VectorExpression
      {
      private:

        friend class AST;

        typedef BaseClasses::ExpressionPtr        ExpressionPtr;        //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::ExpressionConstPtr   ExpressionConstPtr;   //!< Type alias for constant shared pointers to class <b>BaseClasses::Expression</b>.

      public:

        /** \brief  Enumeration of all supported mask element checking types. */
        enum class CheckType
        {
          All,    //!< Internal ID which indicates, that all mask elements must be set.
          Any,    //!< Internal ID which indicates, that at least one of the mask elements must be set.
          None    //!< Internal ID which indicates, that none of the mask elements is allowed to be set.
        };

      private:

        CheckType     _eCheckType;
        ExpressionPtr _spSubExpression;

        inline CheckActiveElements() : _eCheckType(CheckType::All), _spSubExpression(nullptr)  {}


      public:

        /** \brief  Creates a new object of this class.
         *  \param  eCheckType        The requested type of the mask element checking routine.
         *  \param  spSubExpression   A shared pointer to the expression object, which shall be used as the sub-expression. */
        static CheckActiveElementsPtr Create(CheckType eCheckType = CheckType::All, ExpressionPtr spSubExpression = nullptr);

        virtual ~CheckActiveElements()  {}


        /** \brief  Returns the string identifier of a specific mask element checking type.
         *  \param  eType   The mask element checking type, whose string identifier shall be returned. */
        static std::string GetCheckTypeString(CheckType eType);


        /** \brief  Returns the currently set mask element checking type. */
        inline CheckType  GetCheckType() const                    { return _eCheckType; }

        /** \brief  Changes the mask element checking type.
         *  \param  eNewCheckType   The requested new mask element checking type. */
        inline void       SetCheckType(CheckType eNewCheckType)   { _eCheckType = eNewCheckType; }


        /** \brief  Returns a shared pointer to the currently referenced sub-expression. */
        inline ExpressionPtr        GetSubExpression()                          { return _spSubExpression; }

        /** \brief  Returns a constant shared pointer to the currently referenced sub-expression. */
        inline ExpressionConstPtr   GetSubExpression() const                    { return _spSubExpression; }

        /** \brief    Sets a new sub-expression.
         *  \param    spSubExpr   A shared pointer to the new sub-expression.
         *  \remarks  The sub-expression should evaluate to a vector mask, whose elements can then be checked. */
        inline void                 SetSubExpression(ExpressionPtr spSubExpr)   { _SetChildPtr(_spSubExpression, spSubExpr); }


      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{

        virtual bool IsVectorized() final override                            { return false; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override   { return BaseClasses::TypeInfo( BaseClasses::TypeInfo::KnownTypes::Bool, true, false ); }

        virtual ExpressionPtr   GetSubExpression(IndexType SubExprIndex) final override;
        virtual IndexType       GetSubExpressionCount() const final override                                      { return static_cast< IndexType >(1); }
        virtual void            SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override;

        //@}


        /** \name Public methods inherited from class BaseClasses::Node */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        //@}
      };

      /** \brief    Describes that an index vector shall be created with element values ranging from <b>zero</b> to <b>vector width - 1</b>.
       *  \remarks  This expression is a leaf node, which always returns a vector of a specified type. */
      class VectorIndex final : public VectorExpression
      {
      private:

        friend class AST;

        typedef BaseClasses::ExpressionPtr          ExpressionPtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Expression</b>.
        typedef BaseClasses::TypeInfo::KnownTypes   KnownTypes;       //!< Type alias for the enumeration of known element types.

      private:

        KnownTypes  _eType;

        inline VectorIndex() : _eType(KnownTypes::Int32)  {}

      public:

        /** \brief  Creates a new object of this class.
         *  \param  eType   The requested element type used for the index vector. */
        static VectorIndexPtr Create(KnownTypes eType = KnownTypes::Int32);

        virtual ~VectorIndex()  {}


        /** \brief  Returns the currently set index element type. */
        inline KnownTypes   GetType() const                 { return _eType; }

        /** \brief  Changes the element type for the index vector.
         *  \param  eNewType  The requested new index element type. */
        inline void         SetType(KnownTypes eNewType)    { _eType = eNewType; }


      public:

        /** \name Public methods inherited from class BaseClasses::Expression */
        //@{

        virtual bool IsVectorized() final override      { return true; }

        virtual BaseClasses::TypeInfo  GetResultType() const final override;

        virtual ExpressionPtr GetSubExpression(IndexType SubExprIndex) final override                          { throw ASTExceptions::ChildIndexOutOfRange(); }
        virtual IndexType     GetSubExpressionCount() const final override                                     { return static_cast<IndexType>(0); }
        virtual void          SetSubExpression(IndexType SubExprIndex, ExpressionPtr spSubExpr) final override { throw ASTExceptions::ChildIndexOutOfRange(); }

        //@}


        /** \name Public methods inherited from class BaseClasses::Node */
        //@{

        virtual std::string DumpToXML(const size_t cszIntend) const final override;

        //@}
      };
    };



    /** \brief  Base class for all VAST nodes, which may contain variable declarations. */
    class IVariableContainer : public BaseClasses::Node
    {
    protected:

      inline IVariableContainer()   {}

    public:

      virtual ~IVariableContainer() {}


      /** \brief  Adds a new variable declaration to the list of known variables.
       *  \param  spVariableInfo  A shared pointer to the VariableInfo object, describing the new variable declaration. */
      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) = 0;

      /** \brief    Returns a shared pointer to a VariableInfo object for a known variable declaration.
       *  \param    strVariableName   The unique name of the variable, whose declaration shall be looked up.
       *  \remarks  If the specified variable declaration is not known, <b>nullptr</b> will be returned. */
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) = 0;

      /** \brief  Checks, whether a sepcific variable is already known.
       *  \param  crstrVariableName   The unique name of the variable, whose presence shall be checked. */
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const = 0;


      /** \brief    Returns a constant shared pointer to a VariableInfo object for a known variable declaration.
       *  \param    strVariableName   The unique name of the variable, whose declaration shall be looked up.
       *  \remarks  If the specified variable declaration is not known, <b>nullptr</b> will be returned. */
      inline BaseClasses::VariableInfoConstPtr  GetVariableInfo(std::string strVariableName) const
      {
        return const_cast<IVariableContainer*>(this)->GetVariableInfo(strVariableName);
      }
    };

    /** \brief  Describes a syntactic scope, i.e. a compound of statements. */
    class Scope final : public IVariableContainer
    {
    public:

      typedef std::vector<Expressions::IdentifierPtr> VariableDeclarationVectorType; //!< Type definition for a list of declared variables.

    private:

      friend class AST;

      typedef BaseClasses::NodePtr NodePtr; //!< Type alias for shared pointers to class <b>BaseClasses::Node</b>.

      typedef std::vector<NodePtr> ChildrenContainerType;


    private:

      ChildrenContainerType _Children;
      std::set<std::string> _setDeclaredVariables;

      inline Scope() {}


      /** \brief  Returns a shared pointer to the first parent VAST node, which is derived from the class IVariableContainer. */
      IVariableContainerPtr             _GetParentVariableContainer();

      /** \brief  Returns a constant shared pointer to the first parent VAST node, which is derived from the class IVariableContainer. */
      inline IVariableContainerConstPtr _GetParentVariableContainer() const { return const_cast<Scope*>(this)->_GetParentVariableContainer(); }


    public:

      /** \brief  Creates a new object of this class. */
      static ScopePtr Create();

      virtual ~Scope()  {}


      /** \name Public methods for accessing the child node container. */
      //@{

      /** \brief  Adds another VAST node at the end of the node container.
       *  \param  spChild   A shared pointer to the node, which shall be added to the scope. */
      void      AddChild(NodePtr spChild);

      /** \brief    Returns the position index of a specific child node inside the scope.
       *  \param    spChildNode   A shared pointer to the child node, whose position index shall be retrived.
       *  \remarks  If the specified VAST node cannot be found inside the scope, an exception will be thrown. */
      IndexType GetChildIndex(NodePtr spChildNode);

      /** \brief    Inserts a new VAST node at a certain position inside the scope.
       *  \param    ChildIndex    The index of the position, the new VAST node shall be inserted to.
       *  \param    spChildNode   The new VAST node, that shall be inserted into the scope.
       *  \remarks  If the index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      void      InsertChild(IndexType ChildIndex, NodePtr spChildNode);

      /** \brief    Removes a specific child node from the scope.
       *  \param    ChildIndex  The index of the child node, which shall be removed.
       *  \remarks  If the index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      void      RemoveChild(IndexType ChildIndex);

      /** \brief    Replaces a specific child node of the scope with another VAST node.
       *  \param    ChildIndex    The index of the child node, which shall be replaced.
       *  \param    spChildNode   The new VAST node, that shall be inserted into the scope.
       *  \remarks  If the index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      void      SetChild(IndexType ChildIndex, NodePtr spChildNode);

      //@}



      /** \name Public methods for accessing the child node container. */
      //@{

      /** \brief    Adds a new variable declaration to this scope.
       *  \param    spVariableInfo  A shared pointer to the VariableInfo object, describing the variable declaration that shall be added.
       *  \remarks  This function automatically adds the variable declaration into the enclosing FunctionDeclaration object of this scope. */
      void                            AddVariableDeclaration(BaseClasses::VariableInfoPtr spVariableInfo);

      /** \brief  Returns a list of all variables, which are declared exactly in the current scope. */
      VariableDeclarationVectorType   GetVariableDeclarations() const;

      /** \brief  Checks, whether a certain variable is declared exactly in the current scope.
       *  \param  strVariableName   The unique name of the variable, which shall be checked. */
      inline bool                     HasVariableDeclaration(std::string strVariableName) const   { return (_setDeclaredVariables.count(strVariableName) != 0); }

      /** \brief    Imports all variable declarations of another scope into the current one.
       *  \param    spOtherScope  A shared pointer to the Scope object, whose variable declarations shall be imported.
       *  \remarks  All variable declarations of the imported scope will be cleared by this function. */
      void                            ImportVariableDeclarations(ScopePtr spOtherScope);

      //@}

      /** \brief  Fully devours another scope, i.e. all its internal child nodes and variable declarations will be appended to the current scope.
       *  \param  spOtherScope  A shared pointer to the Scope object, which shall be imported. Its contents will be cleared in that process. */
      void ImportScope(ScopePtr spOtherScope);

      /** \brief  Returns <b>true</b> if and only if this scope has no child nodes. */
      inline bool IsEmpty() const   { return (GetChildCount() == static_cast<IndexType>(0)); }


    public:

      /** \name Public methods inherited from class IVariableContainer. */
      //@{

      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) final override;
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) final override;
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const final override;

      //@}


      /** \name Public methods inherited from class BaseClasses::Node. */
      //@{

      virtual NodePtr     GetChild(IndexType ChildIndex) final override;
      virtual IndexType   GetChildCount() const final override  { return static_cast<IndexType>(_Children.size()); }

      virtual std::string DumpToXML(const size_t cszIntend) const final override;

      //@}
    };

    /** \brief    Describes a function declaration.
     *  \remarks  This class is the main declaration context for all variables declared inside a function. */
    class FunctionDeclaration final : public IVariableContainer
    {
    private:

      friend class AST;

      typedef BaseClasses::NodePtr  NodePtr;    //!< Type alias for shared pointers to class <b>BaseClasses::Node</b>.

      typedef std::vector<Expressions::IdentifierPtr>             ParameterContainerType;
      typedef std::map<std::string, BaseClasses::VariableInfoPtr> KnownVariablesMapType;


    private:

      ParameterContainerType  _Parameters;
      ScopePtr                _spBody;
      KnownVariablesMapType   _mapKnownVariables;
      std::string             _strName;

      inline FunctionDeclaration() : _spBody(nullptr)  {}


    public:

      /** \brief  Creates a new object of this class.
       *  \param  strFunctionName   The requested name of the function declaration. */
      static FunctionDeclarationPtr Create(std::string strFunctionName);

      virtual ~FunctionDeclaration()  {}


      /** \name Public methods for accessing the function paramater container. */
      //@{

      /** \brief  Adds a new function parameter at the end of the current parameter list.
       *  \param  spVariableInfo    A shared pointer to the VariableInfo object, describing the new parameter declaration. */
      void                        AddParameter(BaseClasses::VariableInfoPtr spVariableInfo);

      /** \brief    Returns a shared pointer to the identifier object of a specific function parameter.
       *  \param    iParamIndex   The index of the function parameter, which shall be returned.
       *  \remarks  If the function parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      Expressions::IdentifierPtr  GetParameter(IndexType iParamIndex);

      /** \brief  Returns the number of parameters for this function declaration. */
      inline IndexType            GetParameterCount() const   { return static_cast<IndexType>(_Parameters.size()); }

      /** \brief    Replaces an existing function parameter with a new one.
       *  \param    iParamIndex       The index of the function parameter, which shall be replaced.
       *  \param    spVariableInfo    A shared pointer to the VariableInfo object, describing the new parameter declaration.
       *  \remarks  If the function parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      void                        SetParameter(IndexType iParamIndex, BaseClasses::VariableInfoPtr spVariableInfo);


      /** \brief    Returns a constant shared pointer to the identifier object of a specific function parameter.
       *  \param    iParamIndex   The index of the function parameter, which shall be returned.
       *  \remarks  If the function parameter index is out of range, a <b>ASTExceptions::ChildIndexOutOfRange</b> exception will be thrown. */
      inline Expressions::IdentifierConstPtr  GetParameter(IndexType iParamIndex) const
      {
        return const_cast<FunctionDeclaration*>(this)->GetParameter(iParamIndex);
      }

      //@}


      /** \brief  Returns a shared pointer to the scope, which contains the whole function body. */
      ScopePtr        GetBody();

      /** \brief  Returns a constant shared pointer to the scope, which contains the whole function body. */
      ScopeConstPtr   GetBody() const;


      /** \brief  Returns the name of the function declaration. */
      inline std::string  GetName() const               { return _strName; }

      /** \brief  Changes the name of the function declaration.
       *  \param  strName   The requested new function name. */
      inline void         SetName(std::string strName)  { _strName = strName; }

    public:

      /** \brief  Returns the names of all known variable declarations inside the function, including its parameters. */
      std::vector<std::string> GetKnownVariableNames() const;


      /** \name Public methods inherited from class IVariableContainer. */
      //@{

      virtual void                          AddVariable(BaseClasses::VariableInfoPtr spVariableInfo) final override;
      virtual BaseClasses::VariableInfoPtr  GetVariableInfo(std::string strVariableName) final override;
      virtual bool                          IsVariableUsed(const std::string &crstrVariableName) const final override;

      //@}


      /** \name Public methods inherited from class BaseClasses::Node. */
      //@{

      virtual NodePtr   GetChild(IndexType ChildIndex) final override;
      virtual IndexType GetChildCount() const final override { return static_cast<IndexType>(1); }

      virtual std::string DumpToXML(const size_t cszIntend) const final override;

      //@}
    };

  private:

    /** \brief    Generic internal method, which creates a new object of a VAST class.
     *  \tparam   NodeClass   The type of the VAST object, which shall be created. It must be derived from class <b>BaseClasses::Node</b>.
     *  \return   A shared pointer to the newly created VAST object. */
    template <class NodeClass>
    inline static std::shared_ptr<NodeClass> CreateNode()
    {
      static_assert(std::is_base_of<BaseClasses::Node, NodeClass>::value, "All nodes of the vectorizable AST must be derived from class \"Node\"");

      std::shared_ptr<NodeClass> spNode( new NodeClass );

      spNode->_wpThis = BaseClasses::NodePtr( spNode );

      return spNode;
    }
  };


  template <> inline bool AST::Expressions::Constant::GetValue<bool>() const
  {
    switch (_eType)
    {
    case KnownTypes::Float: case KnownTypes::Double:
      return (_unionValues.dFloatingPointValue != 0.);
    default:
      return (_unionValues.ui64IntegralValue != static_cast<std::uint64_t>(0));
    }
  }

  template <> inline void AST::Expressions::Constant::SetValue<bool>(bool TValue)
  {
    _unionValues.ui64IntegralValue = static_cast<std::uint64_t>(TValue ? 1 : 0);
    _eType                         = KnownTypes::Bool;
  }
} // end namespace Vectorization
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _HIPACC_BACKEND_VECTORIZATION_AST_H_

// vim: set ts=2 sw=2 sts=2 et ai:

