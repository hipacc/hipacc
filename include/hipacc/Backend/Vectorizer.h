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

//===--- Vectorizer.h - Implements a vectorizing component for clang's syntax trees. -===//
//
// This file implements a vectorizing component for clang's syntax trees
//
//===---------------------------------------------------------------------------------===//

#ifndef _HIPACC_BACKEND_VECTORIZER_H_
#define _HIPACC_BACKEND_VECTORIZER_H_

#include "hipacc/Backend/ClangASTHelper.h"
#include "hipacc/Backend/CommonDefines.h"
#include "hipacc/Backend/VectorizationAST.h"

#include <list>
#include <map>
#include <type_traits>
#include <stdio.h>

namespace clang
{
namespace hipacc
{
namespace Backend
{
/** \brief  Contains all classes and type definitions, which are used during the vectorization process. */
namespace Vectorization
{
  /** \brief  Implements the whole algorithmics required for the vectorization of a function. */
  class Vectorizer final
  {
  public:

    /** \brief  Base class for all classes, which export an abstract vectorized AST back into a Clang-specific AST. */
    class VASTExporterBase
    {
    protected:

      typedef ClangASTHelper::FunctionDeclarationVectorType   FunctionDeclVectorType;   //!< Type alias for a vector of function declarations.
      typedef ClangASTHelper::QualTypeVectorType              QualTypeVectorType;       //!< Type alias for a vector of qualified types.

    private:

      typedef std::map<unsigned int, FunctionDeclVectorType>       FunctionDeclParamCountMapType;
      typedef std::map<std::string, FunctionDeclParamCountMapType> FunctionDeclNameMapType;


      ClangASTHelper            _ASTHelper;
      ::clang::DeclContext      *_pDeclContext;

      FunctionDeclNameMapType                    _mapKnownFunctions;
      std::map<std::string, ::clang::ValueDecl*> _mapKnownDeclarations;


    private:

      VASTExporterBase(const VASTExporterBase &)            = delete;
      VASTExporterBase& operator=(const VASTExporterBase &) = delete;


      /** \brief  Returns the Clang type-equivalent of a VAST variable variable declaration. */
      ::clang::QualType _GetVariableType(AST::BaseClasses::VariableInfoPtr spVariableInfo);


    protected:

      /** \brief  Constructor.
       *  \param  rAstContext   A reference to the current Clang AST context.  */
      VASTExporterBase(::clang::ASTContext &rAstContext);


      /** \brief  Returns a reference to the currently used ClangASTHelper object. */
      inline ClangASTHelper&      _GetASTHelper()     { return _ASTHelper; }

      /** \brief  Returns a reference to the current Clang AST context. */
      inline ::clang::ASTContext& _GetASTContext()    { return _GetASTHelper().GetASTContext(); }


      /** \brief    Resets the internal state of the VAST exporter.
       *  \remarks  This function should be called everytime in between the exports of two different VAST function declarations. */
      void _Reset();


      /** \brief  Adds a new Clang function declaration to the current AST context.
       *  \param  pFunctionDecl   A pointer to the function declaration object, which shall be added to the AST context. */
      void _AddKnownFunctionDeclaration(::clang::FunctionDecl *pFunctionDecl);


      /** \brief  Converts a VAST constant into a corresponding Clang literal.
       *  \param  spConstant  A shared pointer to the VAST constant node, which shall be converted. */
      ::clang::Expr*          _BuildConstant(AST::Expressions::ConstantPtr spConstant);

      /** \brief    Converts a VAST function declaration into a Clang function declaration and adds it to the current AST context.
       *  \param    spFunction  A shared pointer to the VAST function declaration node, which shall be converted.
       *  \remarks  This method does not process the function body, but only its declaration header. */
      ::clang::FunctionDecl*  _BuildFunctionDeclaration(AST::FunctionDeclarationPtr spFunction);

      /** \brief  Creates a Clang loop statement object.
       *  \param  eLoopType     The VAST-specific type of the loop. It affects the type of the created Clang loop statement.
       *  \param  pCondition    A pointer to the Clang expression object, which shall be used as the loop condition.
       *  \param  pBody         A pointer to the Clang statement object, which describes the body of the loop.
       *  \param  pIncrement    A pointer to an optional Clang expression object, which describes the increment expression for each loop iteration. */
      ::clang::Stmt*          _BuildLoop(AST::ControlFlow::Loop::LoopType eLoopType, ::clang::Expr *pCondition, ::clang::Stmt *pBody, ::clang::Expr *pIncrement = nullptr);

      /** \brief  Converts a VAST loop control statement into its Clang counterpart.
       *  \param  spLoopControl   A shared pointer to the VAST loop control statement node, which shall be converted. */
      ::clang::Stmt*          _BuildLoopControlStatement(AST::ControlFlow::LoopControlStatementPtr spLoopControl);

      /** \brief  Creates a Clang declaration statement object for a specific VAST variable.
       *  \param  spIdentifier      A shared pointer to the VAST indentifier node, which references the variable that shall be declared.
       *  \param  pInitExpression   A pointer to an optional Clang expression object, which can be used for the initialization of the variable declaration. */
      ::clang::ValueDecl*     _BuildValueDeclaration(AST::Expressions::IdentifierPtr spIdentifier, ::clang::Expr *pInitExpression = nullptr);


      /** \brief  Converts a VAST-specific arithmetic operator type into its Clang counterpart.
       *  \param  eOpType   The VAST-specific arithmetic operator type, which shall be converted. */
      static ::clang::BinaryOperatorKind  _ConvertArithmeticOperatorType(AST::Expressions::ArithmeticOperator::ArithmeticOperatorType eOpType);

      /** \brief  Converts a VAST-specific relational operator type into its Clang counterpart.
       *  \param  eOpType   The VAST-specific relational operator type, which shall be converted. */
      static ::clang::BinaryOperatorKind  _ConvertRelationalOperatorType(AST::Expressions::RelationalOperator::RelationalOperatorType eOpType);

      /** \brief  Converts a VAST-specific unary operator type into its Clang counterpart.
       *  \param  eOpType   The VAST-specific unary operator type, which shall be converted. */
      static ::clang::UnaryOperatorKind   _ConvertUnaryOperatorType(AST::Expressions::UnaryOperator::UnaryOperatorType eOpType);


      /** \brief  Converts a VAST-specific type information into a qualified Clang type.
       *  \param  crTypeInfo  A constant reference to the VAST type information object, which shall be converted. */
      ::clang::QualType       _ConvertTypeInfo(const AST::BaseClasses::TypeInfo &crTypeInfo);


      /** \brief  Creates all kinds of explicit Clang cast expression objects.
       *  \param  crSourceType  The VAST-specific type of the sub-expression for the cast expression.
       *  \param  crTargetType  The requested VAST-specific target type of the cast expression.
       *  \param  pSubExpr      A pointer to the Clang expression object, whose return value shall be casted. */
      ::clang::CastExpr*      _CreateCast(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);

      /** \brief  Creates Clang cast expression objects for pointer casts.
       *  \param  crSourceType  The VAST-specific type of the sub-expression for the cast expression. It must be a pointer type.
       *  \param  crTargetType  The requested VAST-specific target type of the cast expression. It must be a pointer type.
       *  \param  pSubExpr      A pointer to the Clang expression object, whose return value shall be casted. */
      ::clang::CastExpr*      _CreateCastPointer(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);

      /** \brief  Creates all kinds of explicit Clang cast expression objects for scalar single values, i.e. no pointers or arrays.
       *  \param  crSourceType  The VAST-specific type of the sub-expression for the cast expression. It must be a single element type.
       *  \param  crTargetType  The requested VAST-specific target type of the cast expression. It must be a single element type.
       *  \param  pSubExpr      A pointer to the Clang expression object, whose return value shall be casted. */
      ::clang::CastExpr*      _CreateCastSingleValue(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);


      /** \brief  Creates a Clang declaration reference expression object for a particular variable.
       *  \param  strValueName  The unique name of the variable, which shall be referenced. */
      ::clang::DeclRefExpr*   _CreateDeclarationReference(std::string strValueName);

      /** \brief  Creates a Clang parenthesis expression around another expression.
       *  \param  pSubExpr  A pointer to the Clang expression object, which shall be wrapped into a parenthesis. */
      ::clang::ParenExpr*     _CreateParenthesis(::clang::Expr *pSubExpr);


      /** \brief    Returns the known Clang function declaration object, which is the best match for a specific calling syntax.
       *  \param    strFunctionName   The fully qualified name of the function, which shall be looked up.
       *  \param    crvecArgTypes     A vector of the qualified Clang types of all call parameters, which is used to select the correct function overload.
       *  \remarks  If no matching function declaration could be found, <b>nullptr</b> will be returned. */
      ::clang::FunctionDecl*  _GetFirstMatchingFunctionDeclaration(std::string strFunctionName, const QualTypeVectorType &crvecArgTypes);

      /** \brief  Returns a list of all known function declaration objects, which match a specified function name and call parameter count.
       *  \param  strFunctionName   The fully qualified name of the functions, which shall be looked up.
       *  \param  uiParamCount      The number of requested call parameters for the function lookup. */
      FunctionDeclVectorType  _GetMatchingFunctionDeclarations(std::string strFunctionName, unsigned int uiParamCount);

      /** \brief  Checks, whether a specific variable declaration is already known.
       *  \param  strDeclName   The unique name of the variable, which shall be checked for a conflicting declaration. */
      bool _HasValueDeclaration(std::string strDeclName);


      /** \brief    Abstract method, which returns the qualified Clang type of a <b>vectorized</b> VAST-specific type.
       *  \param    crOriginalTypeInfo  A constant reference to the VAST-specific type information object, whose vectorized Clang type counterpart shall be returned.
       *  \remarks  As this type matching depends on the target architecture, this method must be implemented by the derived classes. */
      virtual ::clang::QualType _GetVectorizedType(AST::BaseClasses::TypeInfo &crOriginalTypeInfo) = 0;

    public:

      virtual ~VASTExporterBase()
      {
        _Reset();

        _mapKnownFunctions.clear();
      }
    };


  private:

    typedef AST::IndexType IndexType;   //!< Type alias for the internally used index type of the VAST.

    /** \brief  Internal class, which handles the conversion of a Clang-specific AST into an abstract vectorized AST. */
    class VASTBuilder
    {
    private:

      /** \brief    Internal helper class, which handles the mapping of declared variable names to unique variable names.
       *  \remarks  The C++ language allows variable hiding by a re-declaration of the same variable name in a nested scope.
       *            Since the VAST requires unique variable names throughout the whole function, this mapping is required. */
      class VariableNameTranslator final
      {
      private:

        typedef std::map<std::string, std::string> RenameMapType;

        std::list<RenameMapType> _lstRenameStack;

      public:

        /** \brief    Adds a new layer to the declaration stack.
         *  \remarks  This method should be called, whenever the AST converter enters a scope. */
        inline void AddLayer()  { _lstRenameStack.push_front(RenameMapType()); }

        /** \brief    Removes the current layer from the declaration stack.
         *  \remarks  This method should be called, whenever the AST converter leaves a scope. */
        inline void PopLayer()  { _lstRenameStack.pop_front(); }


        /** \brief  Adds a new variable name mapping into the lookup table.
         *  \param  strOriginalName   The name of the variable in its original declaration.
         *  \param  strNewName        The requested unique name for this variable. */
        void          AddRenameEntry( std::string strOriginalName, std::string strNewName );

        /** \brief  Returns the mapped unique name for a specific variable.
         *  \param  strOriginalName   The name of the variable in its original declaration. */
        std::string   TranslateName( std::string strOriginalName ) const;
      };


    private:

      /** \brief  Returns the VAST-specific counterpart of a qualified Clang type.
       *  \param  qtSourceType  The qualified Clang type, which shall be converted. */
      inline static AST::BaseClasses::TypeInfo _ConvertTypeInfo(::clang::QualType qtSourceType)
      {
        AST::BaseClasses::TypeInfo ReturnType;

        _ConvertTypeInfo(ReturnType, qtSourceType);

        return ReturnType;
      }

      /** \brief  Converts a qualified Clang type and writes its contents to a VAST-specific type information object.
       *  \param  rTypeInfo     A reference to the VAST type information object, which shall be filled with the converted type information.
       *  \param  qtSourceType  The qualified Clang type, which shall be converted. */
      static void _ConvertTypeInfo(AST::BaseClasses::TypeInfo &rTypeInfo, ::clang::QualType qtSourceType);



      /** \brief  Converts a Clang binary operator object (and its sub-expressions) into a VAST binary operator node.
       *  \param  pExprLHS  A pointer to the Clang expression object, which describes the left-hand-side of the binary operator.
       *  \param  pExprRHS  A pointer to the Clang expression object, which describes the right-hand-side of the binary operator.
       *  \param  eOpKind   The Clang-specific operator code of the binary operator. */
      AST::Expressions::BinaryOperatorPtr   _BuildBinaryOperatorExpression(::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, ::clang::BinaryOperatorKind eOpKind);

      /** \brief    Converts a Clang if-statement object into a VAST branching statement node.
       *  \param    pIfStmt           A pointer to the Clang if-statement object, which shall be converted.
       *  \param    spEnclosingScope  A shared pointer to the VAST scope node, the new branching statement shall be nested into.
       *  \remarks  If possible, this function will flatten the Clang-specific if-cascades into multi-branch statements. */
      void                                  _BuildBranchingStatement(::clang::IfStmt *pIfStmt, AST::ScopePtr spEnclosingScope);

      /** \brief  Converts the conditional branch of a Clang if-statement object into a VAST conditional branch node.
       *  \param  pIfStmt               A pointer to the Clang if-statement object, whose conditional branch shall be converted.
       *  \param  spBranchingStatement  A shared pointer to the VAST branching statement node, the new conditional branch shall be added to.
       *  \return A pointer to the contents of the <b>else</b> branch of the input if-statement. */
      ::clang::Stmt*                        _BuildConditionalBranch(::clang::IfStmt *pIfStmt, AST::ControlFlow::BranchingStatementPtr spBranchingStatement);

      /** \brief  Converts a Clang literal expression object into a VAST constant node.
       *  \param  pExpression   A pointer to the Clang expression object, which shall be converted. It must have a <b>literal</b> type. */
      AST::Expressions::ConstantPtr         _BuildConstantExpression(::clang::Expr *pExpression);

      /** \brief  Converts a Clang cast expression object into a VAST conversion node.
       *  \param  pCastExpr   A pointer to the Clang cast expression object, which shall be converted. */
      AST::Expressions::ConversionPtr       _BuildConversionExpression(::clang::CastExpr *pCastExpr);

      /** \brief  Base method, which converts any kinds of Clang expression objects into their VAST counterparts.
       *  \param  pExpression   A pointer to the Clang expression object, which shall be converted. */
      AST::BaseClasses::ExpressionPtr       _BuildExpression(::clang::Expr *pExpression);

      /** \brief    Creates a VAST identifier node for a specific variable name.
       *  \param    strIdentifierName   The originally declared name of the requested variable.
       *  \remarks  This function performs a lookup of the original Clang variable name to the unique VAST variable name. */
      AST::Expressions::IdentifierPtr       _BuildIdentifier(std::string strIdentifierName);

      /** \brief  Converts any kinds of Clang loop statement objects into a VAST loop node.
       *  \param  pLoopStatement    A pointer to the Clang statement object, which shall be converted. It must have a <b>loop</b> type.
       *  \param  spEnclosingScope  A shared pointer to the VAST scope node, the new loop shall be nested into. */
      void                                  _BuildLoop(::clang::Stmt *pLoopStatement, AST::ScopePtr spEnclosingScope);

      /** \brief  Base method, which converts any kinds of Clang statement objects into their VAST counterparts.
       *  \param  pStatement        A pointer to the Clang statement object, which shall be converted.
       *  \param  spEnclosingScope  A shared pointer to the VAST scope node, the new statement shall be added to.
       *  \return A shared pointer to the newly created VAST node, if it has to be added to the enclosing scope, or <b>nullptr</b> if the converted
       *          statement has already been linked into the VAST. */
      AST::BaseClasses::NodePtr             _BuildStatement(::clang::Stmt *pStatement, AST::ScopePtr spEnclosingScope);

      /** \brief  Converts a Clang unary operator object (and its sub-expression) into a VAST unary operator node.
       *  \param  pSubExpr  A pointer to the Clang expression object, which describes the sub-expression of the unary operator.
       *  \param  eOpKind   The Clang-specific operator code of the unary operator. */
      AST::Expressions::UnaryOperatorPtr    _BuildUnaryOperatorExpression(::clang::Expr *pSubExpr, ::clang::UnaryOperatorKind eOpKind);

      /** \brief  Converts a Clang-specific value declaration object into a VAST variable info node.
       *  \param  pVarDecl              A pointer to the Clang value declaration object, which shall be converted.
       *  \param  spVariableContainer   A shared pointer to the VAST variable container node, whose context the variable shall be declared in. */
      AST::BaseClasses::VariableInfoPtr     _BuildVariableInfo(::clang::VarDecl *pVarDecl, AST::IVariableContainerPtr spVariableContainer);


      /** \brief  Converts the whole contents of a Clang compound statement and adds them to a VAST scope node.
       *  \param  spScope             A shared pointer to the VAST scope node, the converted contents shall be added to.
       *  \param  pCompoundStatement  A pointer to the Clang compound statement object, which shall be converted. */
      void _ConvertScope(AST::ScopePtr spScope, ::clang::CompoundStmt *pCompoundStatement);


    private:

      VariableNameTranslator  _VarTranslator;

      ::clang::ASTContext &_rASTContext;


    public:
      VASTBuilder(::clang::ASTContext &rASTContext)
          : _rASTContext(rASTContext) {
      }

      /** \brief  Converts a Clang function declaration object into a VAST function declaration node.
       *  \param  pFunctionDeclaration  A pointer to the Clang function declaration object, whicc shall be converted. */
      AST::FunctionDeclarationPtr BuildFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration);


      /** \brief    Returns the next free unique variable name for a new variable declaration.
       *  \param    spVariableContainer   A shared pointer to the VAST variable container, where the variable shall be declared in.
       *  \param    strRootName           The requested root name of the variable declaration. It will be used to create the unique name.
       *  \remarks  Unique names will be created by concatenating the root name with the suffix <b>_&lt;index&gt;</b>, unless the root name itself is still free. */
      static std::string          GetNextFreeVariableName(AST::IVariableContainerPtr spVariableContainer, std::string strRootName);

      /** \brief  Returns the prefix, which shall be used for all temporary variable names. */
      inline static std::string   GetTemporaryNamePrefix()   { return "_temp"; }

      /** \brief  Returns the prefix, which shall be used for all index variable names. */
      inline static std::string   GetTemporaryIndexName()       { return GetTemporaryNamePrefix() + "_index"; }
    };

    class VASTExportArray final : public VASTExporterBase
    {
    private:

      typedef VASTExporterBase    BaseType;


      class VectorIndex final
      {
      private:

        enum class VectorIndexType
        {
          Constant,
          Identifier
        };


        const VectorIndexType     _ceIndexType;
        const IndexType           _ciVectorIndex;
        const ::clang::ValueDecl  *_cpIndexExprDecl;


        VectorIndex(const VectorIndex &) = delete;
        VectorIndex& operator=(const VectorIndex &) = delete;

      public:

        inline VectorIndex(IndexType iVecIdx = static_cast<IndexType>(0)) : _ceIndexType(VectorIndexType::Constant), _ciVectorIndex(iVecIdx), _cpIndexExprDecl(nullptr)  {}

        inline VectorIndex(::clang::ValueDecl *pIndexDecl) : _ceIndexType(VectorIndexType::Identifier), _ciVectorIndex(0), _cpIndexExprDecl(pIndexDecl)  {}


        ::clang::Expr* CreateIndexExpression(ClangASTHelper &rASTHelper) const;
      };


    private:

      const IndexType       _VectorWidth;

      ::clang::ValueDecl    *_pVectorIndexExpr;



    private:

      ::clang::CompoundStmt*  _BuildCompoundStatement(AST::ScopePtr spScope);

      ::clang::Expr*          _BuildExpression(AST::BaseClasses::ExpressionPtr spExpression, const VectorIndex &crVectorIndex);

      ::clang::Stmt*          _BuildExpressionStatement(AST::BaseClasses::ExpressionPtr spExpression);

      ::clang::Expr*          _BuildFunctionCall(AST::Expressions::FunctionCallPtr spFunctionCall, const VectorIndex &crVectorIndex);

      ::clang::IfStmt*        _BuildIfStatement(AST::ControlFlow::BranchingStatementPtr spBranchingStatement);

      ::clang::Stmt*          _BuildLoop(AST::ControlFlow::LoopPtr spLoop);


      virtual ::clang::QualType _GetVectorizedType(AST::BaseClasses::TypeInfo &crOriginalTypeInfo) final override;


    public:

      VASTExportArray(IndexType VectorWidth, ::clang::ASTContext &rAstContext);


      ::clang::FunctionDecl* ExportVASTFunction(AST::FunctionDeclarationPtr spVASTFunction, bool bUnrollVectorLoops);

    };


    /** \brief  Contains the implementations of all supported AST transformations. */
    class Transformations final
    {
    public:

      /** \brief  Enumeration of the supported traversing directions for the AST transformations. */
      enum class DirectionType
      {
        BottomUp,   //!< Indicates that all children of a node shall be processed before the parent node.
        TopDown     //!< Indicates that the parent node shall be processed before any of its children.
      };

    private:

      /** \brief    Iterates over the direct child nodes of a VAST node and performs a transformation for each node of the specified <b>child target type</b>.
       *  \tparam   TransformationType  The type of the requested transformation. It must be derived from <b>TransformationBase</b>.
       *  \param    spCurrentNode       A shared pointer to the VAST node, whose children shall be traversed.
       *  \param    rTransformation     A reference to an object of the requested transformation type.
       *  \remarks  This method relies on static polymorphism. Thus, the selected transformation must define the <b>ChildTargetType</b> type and implement the <b>ProcessChild()</b> method. */
      template <class TransformationType> inline static void _ParseChildren(typename TransformationType::TargetTypePtr spCurrentNode, TransformationType &rTransformation)
      {
        typedef typename TransformationType::ChildTargetType   ChildTargetType;
        static_assert(std::is_base_of<AST::BaseClasses::Node, ChildTargetType>::value, "The child target type of the VAST transformation must be derived from class\"AST::BaseClasses::Node\"!");

        for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < spCurrentNode->GetChildCount(); ++iChildIdx)
        {
          AST::BaseClasses::NodePtr spChildNode = spCurrentNode->GetChild(iChildIdx);
          if (! spChildNode)
          {
            continue;
          }

          if (spChildNode->IsType < ChildTargetType>())
          {
            iChildIdx = rTransformation.ProcessChild(spCurrentNode, iChildIdx, spChildNode->CastToType<ChildTargetType>());
          }
        }
      }

      /** \brief  Base class for all AST transformation implementations. */
      class TransformationBase
      {
      public:

        virtual ~TransformationBase()   {}

        /** \brief  Returns the specified traversing direction for this transformation. */
        virtual DirectionType GetSearchDirection() const    { return DirectionType::BottomUp; }
      };


    public:

      /** \brief    Traverses the whole child node hierarchy of a VAST node and performs a transformation for each node of the specified <b>target type</b>.
       *  \tparam   TransformationType  The type of the requested transformation. It must be derived from <b>TransformationBase</b>.
       *  \param    spCurrentNode       A shared pointer to the VAST node, which shall be traversed.
       *  \param    rTransformation     A reference to an object of the requested transformation type.
       *  \remarks  This method relies on static polymorphism. Thus, the selected transformation must define the <b>TargetType</b> type and implement the <b>Execute()</b> method. */
      template <class TransformationType> inline static void Run(AST::BaseClasses::NodePtr spCurrentNode, TransformationType &rTransformation)
      {
        typedef typename TransformationType::TargetType   TargetType;

        static_assert(std::is_base_of<AST::BaseClasses::Node, TargetType>::value,     "The target type of the VAST transformation must be derived from class\"AST::BaseClasses::Node\"!");
        static_assert(std::is_base_of<TransformationBase, TransformationType>::value, "The transformation class must be derived from class\"TransformationBase\"!");

        // Skip unset children
        if (! spCurrentNode)
        {
          return;
        }

        // Check if the current node is of the target type
        const bool cbIsTarget = spCurrentNode->IsType<TargetType>();

        // Get the search direction
        DirectionType eSearchDirection  = rTransformation.GetSearchDirection();
        if ((eSearchDirection != DirectionType::BottomUp) && (eSearchDirection != DirectionType::TopDown))
        {
          throw InternalErrorException("Invalid VAST transformation search direction type!");
        }


        // Execute the transformation before parsing the children for "top-down" search
        if (cbIsTarget && (eSearchDirection == DirectionType::TopDown))
        {
          rTransformation.Execute( spCurrentNode->CastToType<TargetType>() );
        }

        // Parse all children of the current node
        for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < spCurrentNode->GetChildCount(); ++iChildIdx)
        {
          Run(spCurrentNode->GetChild(iChildIdx), rTransformation);
        }

        // Execute the transformation after parsing the children for "bottom-up" search
        if (cbIsTarget && (eSearchDirection == DirectionType::BottomUp))
        {
          rTransformation.Execute( spCurrentNode->CastToType<TargetType>() );
        }
      }

      /** \brief    Traverses the whole child node hierarchy of a VAST node and performs a transformation for each node of the specified <b>target type</b>.
       *  \tparam   TransformationType  The type of the requested transformation. It must be derived from <b>TransformationBase</b>.
       *  \param    spCurrentNode       A shared pointer to the VAST node, which shall be traversed.
       *  \remarks  This method exists only for compatibility reasons with the <b>GCC</b> compiler (it cannot handle references to temporary objects).
       *  \sa       Run() */
      template <class TransformationType> inline static void RunSimple(AST::BaseClasses::NodePtr spCurrentNode)
      {
        // Required by GCC
        TransformationType Transform;
        Run( spCurrentNode, Transform );
      }


    public:

      /** \brief  Generic query transformation, which returns a list of VAST nodes with a specified type.
       *  \tparam NodeClass   The VAST node type, which shall be looked for. It must be derived from <b>AST::BaseClasses::Node</b>. */
      template <class NodeClass> class FindNodes final : public TransformationBase
      {
      private:

        static_assert( std::is_base_of<AST::BaseClasses::Node, NodeClass>::value, "The NodeClass type must be derived from class \"AST::BaseClasses::Node\" !" );

        const DirectionType _ceSearchDirection;

      public:

        typedef NodeClass                   TargetType;    //!< Type definition for the target node type.
        typedef std::shared_ptr<TargetType> TargetTypePtr; //!< Type definition for shared pointers to the target node type.

        std::list<TargetTypePtr> lstFoundNodes; //!< The list of found nodes with the specified type.


        /** \brief  Constructs a new transformation object.
         *  \param  eSearchDirection  The requested traversing direction of this query. It affects the order in which the found nodes will be added to the result list. */
        FindNodes(DirectionType eSearchDirection = DirectionType::BottomUp) : _ceSearchDirection(eSearchDirection)  {}


        virtual DirectionType GetSearchDirection() const    { return _ceSearchDirection; }

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spFoundNode   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        inline void Execute(TargetTypePtr spFoundNode)    { lstFoundNodes.push_back(spFoundNode); }
      };


      /** \brief    Implements a query transformation, which checks whether a specific variable is an internal variable of a scope hierarchy.
       *  \remarks  The information about whether a variable is visible outside of a scope or only inside of it is very important for the propagation of
                    vectorization markers for control-flow statements. */
      class CheckInternalDeclaration final : public TransformationBase
      {
      public:

        typedef AST::Scope    TargetType;   //!< Type definition for the target node type.

      private:

        std::string _strDeclName;
        bool        _bFound;

      public:

        /** \brief  Constructs a new transformation object.
         *  \param  strDeclName   The unique name of the variable, whose visibility scope shall be checked. */
        inline CheckInternalDeclaration(std::string strDeclName) : _strDeclName(strDeclName), _bFound(false)  {}


        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spScope   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::ScopePtr spScope);

        /** \brief  Returns, whether the variable in question is an internal variable of the input scope hierarchy. */
        inline bool Found() const   { return _bFound; }
      };

      /** \brief    Implements a transformation, which detects all vectorization dependencies inside of a branching statement.
       *  \remarks  The vectorization of all assignments inside of a conditional branch depend on the branch condition if the assignee is visible outside of the branch.
                    This transformation detects all these dependend assignments inside of a branching statement and correctly handles the cross-branch dependencies. */
      class FindBranchingInternalAssignments final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::BranchingStatement      TargetType;   //!< Type definition for the target node type.

        /** \brief  The dependency table of all assignments and their dependent expressions. */
        std::map<AST::Expressions::AssignmentOperatorPtr, std::list<AST::BaseClasses::ExpressionPtr>> mapConditionalAssignments;


        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spBranchingStmt   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt);
      };

      /** \brief    Implements a transformation, which detects all vectorization dependencies inside of a loop.
       *  \remarks  The vectorization of all assignments inside of a loop body depend on the loop condition if the assignee is visible outside of the loop.
                    This transformation detects all these dependend assignments inside of a loop. */
      class FindLoopInternalAssignments final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::Loop      TargetType;   //!< Type definition for the target node type.

        /** \brief  The dependency table of all assignments and their dependent expressions. */
        std::map<AST::Expressions::AssignmentOperatorPtr, std::list<AST::BaseClasses::ExpressionPtr>> mapConditionalAssignments;


        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spLoop  A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::ControlFlow::LoopPtr spLoop);
      };

      /** \brief    Implements a transformation, which flattens vectorized memory accesses.
       *  \remarks  This transformation extracts the index expressions of vectorized memory accesses into a temporary variable, if the index expression is not
                    already a leaf node. This is done as a <b>common sub-expresison elimination step</b>, because the memory accesses might get replicated depending
                    on the used virtual vector width. */
      class FlattenMemoryAccesses final : public TransformationBase
      {
      public:

        typedef AST::Expressions::MemoryAccess  TargetType;   //!< Type definition for the target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spMemoryAccess  A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::Expressions::MemoryAccessPtr spMemoryAccess);
      };

      /** \brief    Implements a transformation, which flattens scope trees.
       *  \remarks  The purpose of this transformation is only a code clean-up. Unrequired empty scopes will be removed and scope, which contain only one
                    child node, will be flushed into their parent scope. */
      class FlattenScopes final : public TransformationBase
      {
      public:

        typedef AST::Scope      TargetType;       //!< Type definition for the target node type.
        typedef AST::ScopePtr   TargetTypePtr;    //!< Type definition for shared pointers to the target node type.
        typedef AST::Scope      ChildTargetType;  //!< Type definition for the child target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spCurrentScope  A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        inline void Execute(AST::ScopePtr spCurrentScope)   { _ParseChildren(spCurrentScope, *this); }

        /** \brief    Internal method, which is called for every detected child node of each processed target node.
         *  \param    spParentScope   A shared pointer to the currently processed VAST node, whose children are parsed.
         *  \param    iChildIndex     The index of the detected child node inside its parent node.
         *  \param    spChildScope    A shared pointer to the currently detected child VAST node.
         *  \remarks  This method implements the actual functionality, which shall be processed for every child node with the child target type. */
        IndexType ProcessChild(AST::ScopePtr spParentScope, IndexType iChildIndex, AST::ScopePtr spChildScope);
      };

      /** \brief    Implements a transformation, which inserts required <b>broadcast</b> expressions.
       *  \remarks  Whenever a binary operator uses one scalar and one vectorized expression, the return value of the scalar expression must be broadcasted
                    into a vector, which is ensured by this transformation. */
      class InsertRequiredBroadcasts final : public TransformationBase
      {
      public:

        typedef AST::Expressions::BinaryOperator    TargetType;   //!< Type definition for the target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spCurrentBinOp  A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp);
      };

      /** \brief    Implements a transformation, which inserts conversion expressions where required.
       *  \remarks  This transformation ressembles the internal type promotion system of this compilation stage. The inserted conversion expressions ensure
                    that the used element type is as small as possible in order to use the largest possible vector width for each operation. */
      class InsertRequiredConversions final : public TransformationBase
      {
      public:

        typedef AST::Expressions::BinaryOperator    TargetType;   //!< Type definition for the target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spCurrentBinOp  A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp);
      };

      /** \brief    Implements a transformation, which removes all implicit conversion expressions.
       *  \remarks  By removing implicit conversions, the type promotion system of the used front-end is reverted, but the user-defined conversions remain. */
      class RemoveImplicitConversions final : public TransformationBase
      {
      public:

        typedef AST::BaseClasses::Expression      TargetType;       //!< Type definition for the target node type.
        typedef AST::BaseClasses::ExpressionPtr   TargetTypePtr;    //!< Type definition for shared pointers to the target node type.
        typedef AST::Expressions::Conversion      ChildTargetType;  //!< Type definition for the child target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spCurrentExpression   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::BaseClasses::ExpressionPtr spCurrentExpression)   { _ParseChildren(spCurrentExpression, *this); }

        /** \brief    Internal method, which is called for every detected child node of each processed target node.
         *  \param    spParentExpression  A shared pointer to the currently processed VAST node, whose children are parsed.
         *  \param    iChildIndex         The index of the detected child node inside its parent node.
         *  \param    spConversion        A shared pointer to the currently detected child VAST node.
         *  \remarks  This method implements the actual functionality, which shall be processed for every child node with the child target type. */
        IndexType ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion);
      };

      /** \brief    Implements a transformation, which removes all unnecessary conversion expressions.
       *  \remarks  This transformation executes conversions on constants (by changing their numeric type) and collapses conversion chains, where all elements
                    convert to the same type (this is a specialty of the Clang front-end). */
      class RemoveUnnecessaryConversions final : public TransformationBase
      {
      public:

        typedef AST::BaseClasses::Expression      TargetType;       //!< Type definition for the target node type.
        typedef AST::BaseClasses::ExpressionPtr   TargetTypePtr;    //!< Type definition for shared pointers to the target node type.
        typedef AST::Expressions::Conversion      ChildTargetType;  //!< Type definition for the child target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spCurrentExpression   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::BaseClasses::ExpressionPtr spCurrentExpression)   { _ParseChildren(spCurrentExpression, *this); }

        /** \brief    Internal method, which is called for every detected child node of each processed target node.
         *  \param    spParentExpression  A shared pointer to the currently processed VAST node, whose children are parsed.
         *  \param    iChildIndex         The index of the detected child node inside its parent node.
         *  \param    spConversion        A shared pointer to the currently detected child VAST node.
         *  \remarks  This method implements the actual functionality, which shall be processed for every child node with the child target type. */
        IndexType ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion);
      };

      /** \brief    Implements a transformation, which splits branching statements with mixed scalar and vectorized conditions.
       *  \remarks  If a branching statement uses only scalar conditions in the first branches, which are followed by at least one vectorized condition, it
                    will be seperated into a fully scalar branching statement and a fully vectorized branching statement in the scalar <b>else</b>-path. */
      class SeparateBranchingStatements final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::BranchingStatement    TargetType;   //!< Type definition for the target node type.

        /** \brief    Internal method, which is called for every detected node with the target type.
         *  \param    spBranchingStmt   A shared pointer to the currently detected VAST node.
         *  \remarks  This method implements the actual functionality of the transformation. */
        void Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt);
      };


    public:

      typedef FindNodes<AST::Expressions::AssignmentOperator> FindAssignments; //!< Type definition for a query transformation of VAST assignment operator nodes.
    };



    /** \brief  Returns the VAST variable info object for the left-hand-side of an assignment operator.
     *  \param  spAssignment  A shared pointer to the VAST assignment operator node in question. */
    static AST::BaseClasses::VariableInfoPtr _GetAssigneeInfo(AST::Expressions::AssignmentOperatorPtr spAssignment);

    /** \brief  Computes a local vector mask for a vectorized control-flow statement.
     *  \param  spParentScope       A shared pointer to the VAST scope object, all created statements shall be added to.
     *  \param  spCondition         A shared pointer to the VAST expression object, which describes the condition for the local mask computation.
     *  \param  strLocalMaskName    The requested name of the computed local vector mask.
     *  \param  strGlobalMaskName   The name of the currently active global vector mask at that vectorized control-flow hierarchy level.
     *  \param  bExclusiveBranches  A flag indicating, whether the newly created local mask shall be excluded from the global mask. */
    static void _CreateLocalMaskComputation(AST::ScopePtr spParentScope, AST::BaseClasses::ExpressionPtr spCondition, std::string strLocalMaskName, std::string strGlobalMaskName, bool bExclusiveBranches);

    /** \brief  Creates a new vectorized if-branch and adds it to the scope.
     *  \param  spParentScope   A shared pointer to the VAST scope object, where the new if-branch shall be added to.
     *  \param  spBranchScope   A shared pointer to the VAST scope object, which contains the branch body.
     *  \param  strMaskName     The name of the vector mask, the new branch does depend upon. */
    static void _CreateVectorizedConditionalBranch(AST::ScopePtr spParentScope, AST::ScopePtr spBranchScope, std::string strMaskName);

    /** \brief  Extracts a specific sub-expression from an expression tree and replaces it by a temporary variable.
     *  \param  crstrTempVarNameRoot  The requested root name for the newly created temporary variable.
     *  \param  spSubExpression       A shared pointer to the VAST expression object, which shall be extracted. */
    static void _FlattenSubExpression(const std::string &crstrTempVarNameRoot, AST::BaseClasses::ExpressionPtr spSubExpression);

  public:

    /** \brief  Converts a Clang-specific function declaration object into its VAST counterpart.
     *  \param  pFunctionDeclaration  A pointer to the Clang function declaration object, which shall be converted. */
    AST::FunctionDeclarationPtr ConvertClangFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration);

    ::clang::FunctionDecl*      ConvertVASTFunctionDecl(AST::FunctionDeclarationPtr spVASTFunction, const size_t cszVectorWidth, ::clang::ASTContext &rASTContext, bool bUnrollVectorLoops);


    /** \name Possible AST transformations */
    //@{

    /** \brief    Runs a AST transformation, which flattens the vectorized memory accesses present inside a VAST node.
     *  \param    spRootNode  A shared pointer to the top-level VAST node, this transformation shall be applied to.
     *  \remarks  This transformation extracts the index expressions of vectorized memory accesses into a temporary variable, if the index expression is not
                  already a leaf node. This is done as a <b>common sub-expresison elimination step</b>, because the memory accesses might get replicated depending
                  on the used virtual vector width. */
    inline void FlattenMemoryAccesses(AST::BaseClasses::NodePtr spRootNode) { Transformations::RunSimple<Transformations::FlattenMemoryAccesses>(spRootNode); }

    /** \brief    Runs a AST transformation, which flattens the scope trees present inside a VAST node.
     *  \param    spRootNode  A shared pointer to the top-level VAST node, this transformation shall be applied to.
     *  \remarks  The purpose of this transformation is only a code clean-up. Unrequired empty scopes will be removed and scope, which contain only one
                  child node, will be flushed into their parent scope. */
    inline void FlattenScopeTrees(AST::BaseClasses::NodePtr spRootNode) { Transformations::RunSimple<Transformations::FlattenScopes>(spRootNode); }

    /** \brief    Runs a AST transformation, which removes all unnecessary conversion expressions present inside a VAST node.
     *  \param    spRootNode  A shared pointer to the top-level VAST node, this transformation shall be applied to.
     *  \remarks  This transformation executes conversions on constants (by changing their numeric type) and collapses conversion chains, where all elements
                  convert to the same type (this is a specialty of the Clang front-end). */
    inline void RemoveUnnecessaryConversions(AST::BaseClasses::NodePtr spRootNode) { Transformations::RunSimple<Transformations::RemoveUnnecessaryConversions>(spRootNode); }

    /** \brief    Runs a AST transformation, which splits branching statements with mixed scalar and vectorized conditions present inside a VAST node.
     *  \param    spRootNode  A shared pointer to the top-level VAST node, this transformation shall be applied to.
     *  \remarks  If a branching statement uses only scalar conditions in the first branches, which are followed by at least one vectorized condition, it
                  will be seperated into a fully scalar branching statement and a fully vectorized branching statement in the scalar <b>else</b>-path. */
    inline void SeparateBranchingStatements(AST::BaseClasses::NodePtr spRootNode) { Transformations::RunSimple<Transformations::SeparateBranchingStatements>(spRootNode); }


    /** \brief  Rebuilds the vectorized control-flow of a function with the use of vector masks.
     *  \param  spFunction  A shared pointer to the VAST function declaration node, whose body shall be processed. */
    void RebuildControlFlow(AST::FunctionDeclarationPtr spFunction);

    /** \brief  Rebuilds the data-flow of a vectorized function such that the instruction selection can easily be performed.
     *  \param  spFunction                        A shared pointer to the VAST function declaration node, whose body shall be processed.
     *  \param  bEnsureMonoTypeVectorExpressions  A flag indicating, whether all vectorized expression trees have to be split such that only one element type is used. */
    void RebuildDataFlow(AST::FunctionDeclarationPtr spFunction, bool bEnsureMonoTypeVectorExpressions = false);

    //@}


    /** \brief    Runs the vectorization analysis algorithm on a specific VAST function declaration node.
     *  \param    spFunction  A shared pointer to the VAST function declaration node, the analysis shall be performed at.
     *  \remarks  This algorithm is propagating all already present vectorization markers to all required variables. Therefore,
                  the initial vectorization markers have to be set before calling this method. */
    void VectorizeFunction(AST::FunctionDeclarationPtr spFunction);


    /** \brief    Dumps the contents of a particular VAST node into a XML-file.
     *  \param    spVastNode      A shared pointer to the VAST node, which shall be dunped.
     *  \param    strXmlFilename  The requested name of the XML-file, the VAST node contents shall be written to.
     *  \remarks  If the specified XML-file already exists, it will be overwritten. */
    static void DumpVASTNodeToXML(AST::BaseClasses::NodePtr spVastNode, std::string strXmlFilename);

    class VASTHelper {
     public:
      inline static std::string GetTemporaryIndexName() { return VASTBuilder::GetTemporaryIndexName(); }

      inline static void ReplaceIdentifierByExpression(AST::BaseClasses::ExpressionPtr spParent, std::string strIdentifierName, AST::BaseClasses::ExpressionPtr spReplacement) {
        for (AST::IndexType iChildIdx = 0; iChildIdx < spParent->GetChildCount(); ++iChildIdx) {
          AST::BaseClasses::NodePtr spChild = spParent->GetChild(iChildIdx);
          if (spChild->IsType<AST::Expressions::Identifier>()) {
            AST::Expressions::IdentifierPtr spIdentifier = spChild->CastToType<AST::Expressions::Identifier>();
            if (spIdentifier->GetName().compare(strIdentifierName) == 0) {
              spParent->SetSubExpression(iChildIdx, spReplacement);
              continue;
            }
          }

          if (!spChild->IsLeafNode() &&
              spChild->IsType<AST::BaseClasses::Expression>()) {
            ReplaceIdentifierByExpression(spChild->CastToType<AST::BaseClasses::Expression>(), strIdentifierName, spReplacement);
          }
        }
      }

      inline static std::vector<AST::Expressions::AssignmentOperatorPtr> FindVariableAssignments(AST::BaseClasses::NodePtr spParent, std::string strVarName) {
        std::vector<AST::Expressions::AssignmentOperatorPtr> vecAssignments;

        for (AST::IndexType iChildIdx = 0; iChildIdx < spParent->GetChildCount(); ++iChildIdx) {
          AST::BaseClasses::NodePtr spChild = spParent->GetChild(iChildIdx);

          if (!spChild) continue; // somewhere a node containing null has been inserted

          if (spChild->IsType<AST::Expressions::AssignmentOperator>()) {
            AST::Expressions::AssignmentOperatorPtr spAssignment = spChild->CastToType<AST::Expressions::AssignmentOperator>();
            if (spAssignment->GetLHS() &&
                spAssignment->GetLHS()->IsType<AST::Expressions::Identifier>()) {
              AST::Expressions::IdentifierPtr spIdentifier = spAssignment->GetLHS()->CastToType<AST::Expressions::Identifier>();
              std::string strName = spIdentifier->GetName();
              size_t szMatch = strName.find(strVarName);
              if (szMatch == 0) {
                vecAssignments.push_back(spAssignment);
                continue;
              }
            }
          }

          if (!spChild->IsLeafNode()) {
            std::vector<AST::Expressions::AssignmentOperatorPtr> vecResult = FindVariableAssignments(spChild, strVarName);
            vecAssignments.insert(vecAssignments.end(), vecResult.begin(), vecResult.end());
          }
        }

        return vecAssignments;
      }
    };
  };
} // end namespace Vectorization
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _HIPACC_BACKEND_VECTORIZER_H_

// vim: set ts=2 sw=2 sts=2 et ai:

