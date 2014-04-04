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

#ifndef _BACKEND_VECTORIZER_H_
#define _BACKEND_VECTORIZER_H_

#include <clang/AST/StmtVisitor.h>
#include <list>
#include <map>
#include <type_traits>
#include "ClangASTHelper.h"
#include "CommonDefines.h"
#include "VectorizationAST.h"

#include "stdio.h"

namespace clang
{
namespace hipacc
{
namespace Backend
{
namespace Vectorization
{
  class Vectorizer final
  {
  public:

    class VASTExporterBase
    {
    protected:

      typedef ClangASTHelper::FunctionDeclarationVectorType   FunctionDeclVectorType;
      typedef ClangASTHelper::QualTypeVectorType              QualTypeVectorType;

    private:

      typedef std::map< unsigned int, FunctionDeclVectorType >        FunctionDeclParamCountMapType;
      typedef std::map< std::string, FunctionDeclParamCountMapType >  FunctionDeclNameMapType;


      ClangASTHelper            _ASTHelper;
      ::clang::DeclContext      *_pDeclContext;

      FunctionDeclNameMapType                       _mapKnownFunctions;
      std::map< std::string, ::clang::ValueDecl* >  _mapKnownDeclarations;


    private:

      VASTExporterBase(const VASTExporterBase &)            = delete;
      VASTExporterBase& operator=(const VASTExporterBase &) = delete;


      ::clang::QualType _GetVariableType(AST::BaseClasses::VariableInfoPtr spVariableInfo);


    protected:

      VASTExporterBase(::clang::ASTContext &rAstContext);

      inline ClangASTHelper&      _GetASTHelper()     { return _ASTHelper; }
      inline ::clang::ASTContext& _GetASTContext()    { return _GetASTHelper().GetASTContext(); }

      void _Reset();


      void _AddKnownFunctionDeclaration(::clang::FunctionDecl *pFunctionDecl);


      ::clang::Expr*          _BuildConstant(AST::Expressions::ConstantPtr spConstant);

      ::clang::FunctionDecl*  _BuildFunctionDeclaration(AST::FunctionDeclarationPtr spFunction);

      ::clang::Stmt*          _BuildLoop(AST::ControlFlow::Loop::LoopType eLoopType, ::clang::Expr *pCondition, ::clang::Stmt *pBody, ::clang::Expr *pIncrement = nullptr);

      ::clang::Stmt*          _BuildLoopControlStatement(AST::ControlFlow::LoopControlStatementPtr spLoopControl);

      ::clang::ValueDecl*     _BuildValueDeclaration(AST::Expressions::IdentifierPtr spIdentifier, ::clang::Expr *pInitExpression = nullptr);


      static ::clang::BinaryOperatorKind  _ConvertArithmeticOperatorType(AST::Expressions::ArithmeticOperator::ArithmeticOperatorType eOpType);

      static ::clang::BinaryOperatorKind  _ConvertRelationalOperatorType(AST::Expressions::RelationalOperator::RelationalOperatorType eOpType);

      static ::clang::UnaryOperatorKind   _ConvertUnaryOperatorType(AST::Expressions::UnaryOperator::UnaryOperatorType eOpType);


      ::clang::QualType       _ConvertTypeInfo(const AST::BaseClasses::TypeInfo &crTypeInfo);


      ::clang::CastExpr*      _CreateCast(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);

      ::clang::CastExpr*      _CreateCastPointer(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);

      ::clang::CastExpr*      _CreateCastSingleValue(const AST::BaseClasses::TypeInfo &crSourceType, const AST::BaseClasses::TypeInfo &crTargetType, ::clang::Expr *pSubExpr);


      ::clang::DeclRefExpr*   _CreateDeclarationReference(std::string strValueName);

      ::clang::ParenExpr*     _CreateParenthesis(::clang::Expr *pSubExpr);


      ::clang::FunctionDecl*  _GetFirstMatchingFunctionDeclaration(std::string strFunctionName, const QualTypeVectorType &crvecArgTypes);

      FunctionDeclVectorType  _GetMatchingFunctionDeclarations(std::string strFunctionName, unsigned int uiParamCount);


      virtual ::clang::QualType _GetVectorizedType(AST::BaseClasses::TypeInfo &crOriginalTypeInfo) = 0;


      bool _HasValueDeclaration(std::string strDeclName);


    public:

      virtual ~VASTExporterBase()
      {
        _Reset();

        _mapKnownFunctions.clear();
      }
    };



  private:

    typedef AST::IndexType IndexType;


    class VASTBuilder : public ::clang::StmtVisitor< VASTBuilder >
    {
    private:

      class VariableNameTranslator final
      {
      private:

        typedef std::map< std::string, std::string >  RenameMapType;

        std::list< RenameMapType >    _lstRenameStack;

      public:

        inline void AddLayer()  { _lstRenameStack.push_front(RenameMapType()); }
        inline void PopLayer()  { _lstRenameStack.pop_front(); }

        void          AddRenameEntry( std::string strOriginalName, std::string strNewName );
        std::string   TranslateName( std::string strOriginalName ) const;
      };


    private:

      inline static AST::BaseClasses::TypeInfo _ConvertTypeInfo(::clang::QualType qtSourceType)
      {
        AST::BaseClasses::TypeInfo ReturnType;

        _ConvertTypeInfo(ReturnType, qtSourceType);

        return ReturnType;
      }

      static void _ConvertTypeInfo(AST::BaseClasses::TypeInfo &rTypeInfo, ::clang::QualType qtSourceType);



      AST::Expressions::BinaryOperatorPtr   _BuildBinaryOperatorExpression(::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, ::clang::BinaryOperatorKind eOpKind);

      void                                  _BuildBranchingStatement(::clang::IfStmt *pIfStmt, AST::ScopePtr spEnclosingScope);

      ::clang::Stmt*                        _BuildConditionalBranch(::clang::IfStmt *pIfStmt, AST::ControlFlow::BranchingStatementPtr spBranchingStatement);

      AST::Expressions::ConstantPtr         _BuildConstantExpression(::clang::Expr *pExpression);

      AST::Expressions::ConversionPtr       _BuildConversionExpression(::clang::CastExpr *pCastExpr);

      AST::BaseClasses::ExpressionPtr       _BuildExpression(::clang::Expr *pExpression);

      AST::Expressions::IdentifierPtr       _BuildIdentifier(std::string strIdentifierName);

      void                                  _BuildLoop(::clang::Stmt *pLoopStatement, AST::ScopePtr spEnclosingScope);

      AST::BaseClasses::NodePtr             _BuildStatement(::clang::Stmt *pStatement, AST::ScopePtr spEnclosingScope);

      AST::Expressions::UnaryOperatorPtr    _BuildUnaryOperatorExpression(::clang::Expr *pSubExpr, ::clang::UnaryOperatorKind eOpKind);

      AST::BaseClasses::VariableInfoPtr     _BuildVariableInfo(::clang::VarDecl *pVarDecl, AST::IVariableContainerPtr spVariableContainer);

      void _ConvertScope(AST::ScopePtr spScope, ::clang::CompoundStmt *pCompoundStatement);


    private:

      VariableNameTranslator  _VarTranslator;


    public:

      AST::FunctionDeclarationPtr BuildFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration);


      static std::string          GetNextFreeVariableName(AST::IVariableContainerPtr spVariableContainer, std::string strRootName);

      inline static std::string   GetTemporaryNamePrefix()   { return "_temp"; }



    // Debug stuff
    private:

      unsigned int _uiIntend = 2;

    public:

      void Import(::clang::FunctionDecl *pFunctionDeclaration)
      {
        printf("\n\nImport function decl:\n");
        Visit(pFunctionDeclaration->getBody());

        printf("\n\nImport finished!");
      }

      void VisitExpr(::clang::Expr *E)
      {
        printf("  %s\n", E->getStmtClassName() );
      }

      void VisitStmt(::clang::Stmt *S)
      {
        if (S == nullptr)
          return;

        for (unsigned int i = 0; i < _uiIntend; ++i)
        {
          printf(" ");
        }

        printf("%s\n", S->getStmtClassName());

        _uiIntend += 2;

        for (::clang::Stmt::child_iterator itChild = S->child_begin(); itChild != S->child_end(); itChild++)
        {
           VisitStmt(*itChild);
        }

        _uiIntend -= 2;
      }

//      void VisitCompoundStmt(::clang::CompoundStmt *S)
//      {
//        printf("  %s\n", S->getStmtClassName());
//      }

      void VisitBinaryOperator(BinaryOperator *E)
      {
        printf("  %s\n", E->getStmtClassName());
      }
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



    class Transformations final
    {
    public:

      enum class DirectionType
      {
        BottomUp,
        TopDown
      };

    private:

      template < class TransformationType >
      inline static void _ParseChildren(typename TransformationType::TargetTypePtr spCurrentNode, TransformationType &rTransformation)
      {
        typedef typename TransformationType::ChildTargetType   ChildTargetType;
        static_assert(std::is_base_of< AST::BaseClasses::Node, ChildTargetType >::value, "The child target type of the VAST transformation must be derived from class\"AST::BaseClasses::Node\"!");

        for (IndexType iChildIdx = static_cast<IndexType>(0); iChildIdx < spCurrentNode->GetChildCount(); ++iChildIdx)
        {
          AST::BaseClasses::NodePtr spChildNode = spCurrentNode->GetChild(iChildIdx);
          if (! spChildNode)
          {
            continue;
          }

          if (spChildNode->IsType<ChildTargetType>())
          {
            iChildIdx = rTransformation.ProcessChild(spCurrentNode, iChildIdx, spChildNode->CastToType<ChildTargetType>());
          }
        }
      }

      class TransformationBase
      {
      public:

        virtual ~TransformationBase()   {}

        virtual DirectionType GetSearchDirection() const    { return DirectionType::BottomUp; }
      };


    public:

      template < class TransformationType >
      inline static void Run(AST::BaseClasses::NodePtr spCurrentNode, TransformationType &rTransformation)
      {
        typedef typename TransformationType::TargetType   TargetType;

        static_assert(std::is_base_of< AST::BaseClasses::Node, TargetType >::value,     "The target type of the VAST transformation must be derived from class\"AST::BaseClasses::Node\"!");
        static_assert(std::is_base_of< TransformationBase, TransformationType >::value, "The transformation class must be derived from class\"TransformationBase\"!");

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


    public:

      template < class NodeClass >
      class FindNodes final : public TransformationBase
      {
      private:

        static_assert( std::is_base_of< AST::BaseClasses::Node, NodeClass >::value, "The NodeClass type must be derived from class \"AST::BaseClasses::Node\" !" );

        const DirectionType _ceSearchDirection;

      public:

        typedef NodeClass                       TargetType;
        typedef std::shared_ptr< TargetType >   TargetTypePtr;

        std::list< TargetTypePtr >  lstFoundNodes;


        FindNodes(DirectionType eSearchDirection = DirectionType::BottomUp) : _ceSearchDirection(eSearchDirection)  {}

        virtual DirectionType GetSearchDirection() const    { return _ceSearchDirection; }


        inline void Execute(TargetTypePtr spFoundNode)    { lstFoundNodes.push_back(spFoundNode); }
      };


      class CheckInternalDeclaration final : public TransformationBase
      {
      public:

        typedef AST::Scope    TargetType;

      private:

        std::string _strDeclName;
        bool        _bFound;

      public:

        inline CheckInternalDeclaration(std::string strDeclName) : _strDeclName(strDeclName), _bFound(false)  {}

        void Execute(AST::ScopePtr spScope);

        inline bool Found() const   { return _bFound; }
      };

      class FindBranchingInternalAssignments final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::BranchingStatement      TargetType;

        std::map< AST::Expressions::AssignmentOperatorPtr, std::list< AST::BaseClasses::ExpressionPtr > >  mapConditionalAssignments;

        void Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt);
      };

      class FindLoopInternalAssignments final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::Loop      TargetType;

        std::map< AST::Expressions::AssignmentOperatorPtr, std::list< AST::BaseClasses::ExpressionPtr > >  mapConditionalAssignments;

        void Execute(AST::ControlFlow::LoopPtr spLoop);
      };

      class FlattenMemoryAccesses final : public TransformationBase
      {
      public:

        typedef AST::Expressions::MemoryAccess  TargetType;

        void Execute(AST::Expressions::MemoryAccessPtr spMemoryAccess);
      };

      class FlattenScopes final : public TransformationBase
      {
      public:

        typedef AST::Scope      TargetType;
        typedef AST::ScopePtr   TargetTypePtr;
        typedef AST::Scope      ChildTargetType;

        inline void Execute(AST::ScopePtr spCurrentScope)   { _ParseChildren(spCurrentScope, *this); }

        IndexType ProcessChild(AST::ScopePtr spParentScope, IndexType iChildIndex, AST::ScopePtr spChildScope);
      };

      class InsertRequiredBroadcasts final : public TransformationBase
      {
      public:

        typedef AST::Expressions::BinaryOperator    TargetType;

        void Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp);
      };

      class InsertRequiredConversions final : public TransformationBase
      {
      public:

        typedef AST::Expressions::BinaryOperator    TargetType;

        void Execute(AST::Expressions::BinaryOperatorPtr spCurrentBinOp);
      };

      class RemoveImplicitConversions final : public TransformationBase
      {
      public:

        typedef AST::BaseClasses::Expression      TargetType;
        typedef AST::BaseClasses::ExpressionPtr   TargetTypePtr;
        typedef AST::Expressions::Conversion      ChildTargetType;

        void Execute(AST::BaseClasses::ExpressionPtr spCurrentExpression)   { _ParseChildren(spCurrentExpression, *this); }

        IndexType ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion);
      };

      class RemoveUnnecessaryConversions final : public TransformationBase
      {
      public:

        typedef AST::BaseClasses::Expression      TargetType;
        typedef AST::BaseClasses::ExpressionPtr   TargetTypePtr;
        typedef AST::Expressions::Conversion      ChildTargetType;

        void Execute(AST::BaseClasses::ExpressionPtr spCurrentExpression)   { _ParseChildren(spCurrentExpression, *this); }

        IndexType ProcessChild(AST::BaseClasses::ExpressionPtr spParentExpression, IndexType iChildIndex, AST::Expressions::ConversionPtr spConversion);
      };

      class SeparateBranchingStatements final : public TransformationBase
      {
      public:

        typedef AST::ControlFlow::BranchingStatement    TargetType;

        void Execute(AST::ControlFlow::BranchingStatementPtr spBranchingStmt);
      };


    public:

      typedef FindNodes< AST::Expressions::AssignmentOperator >   FindAssignments;
    };


    static AST::BaseClasses::VariableInfoPtr _GetAssigneeInfo(AST::Expressions::AssignmentOperatorPtr spAssignment);

    static void _CreateLocalMaskComputation(AST::ScopePtr spParentScope, AST::BaseClasses::ExpressionPtr spCondition, std::string strLocalMaskName, std::string strGlobalMaskName, bool bExclusiveBranches);

    static void _CreateVectorizedConditionalBranch(AST::ScopePtr spParentScope, AST::ScopePtr spBranchScope, std::string strMaskName);

    static void _FlattenSubExpression(const std::string &crstrTempVarNameRoot, AST::BaseClasses::ExpressionPtr spSubExpression);

  public:

    void Import(::clang::FunctionDecl *pFunctionDeclaration)
    {
      VASTBuilder b;
      b.Import(pFunctionDeclaration);
    }


    AST::FunctionDeclarationPtr ConvertClangFunctionDecl(::clang::FunctionDecl *pFunctionDeclaration);

    ::clang::FunctionDecl*      ConvertVASTFunctionDecl(AST::FunctionDeclarationPtr spVASTFunction, const size_t cszVectorWidth, ::clang::ASTContext &rASTContext, bool bUnrollVectorLoops);


    inline void FlattenMemoryAccesses(AST::BaseClasses::NodePtr spRootNode)         { Transformations::Run(spRootNode, Transformations::FlattenMemoryAccesses()); }
    inline void FlattenScopeTrees(AST::BaseClasses::NodePtr spRootNode)             { Transformations::Run(spRootNode, Transformations::FlattenScopes()); }
    inline void RemoveUnnecessaryConversions(AST::BaseClasses::NodePtr spRootNode)  { Transformations::Run(spRootNode, Transformations::RemoveUnnecessaryConversions()); }
    inline void SeparateBranchingStatements(AST::BaseClasses::NodePtr spRootNode)   { Transformations::Run(spRootNode, Transformations::SeparateBranchingStatements()); }


    void RebuildControlFlow(AST::FunctionDeclarationPtr spFunction);

    void RebuildDataFlow(AST::FunctionDeclarationPtr spFunction, bool bEnsureMonoTypeVectorExpressions = false);

    void VectorizeFunction(AST::FunctionDeclarationPtr spFunction);

    static void DumpVASTNodeToXML(AST::BaseClasses::NodePtr spVastNode, std::string strXmlFilename);
  };
} // end namespace Vectorization
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_VECTORIZER_H_

// vim: set ts=2 sw=2 sts=2 et ai:

