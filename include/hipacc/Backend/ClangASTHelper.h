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

//===--- ClangASTHelper.h - Implements helper class for easy clang AST handling. -----===//
//
// This file implements a helper class which contains a few methods for easy clang AST handling.
//
//===---------------------------------------------------------------------------------===//

#ifndef _BACKEND_CLANG_AST_HELPER_H_
#define _BACKEND_CLANG_AST_HELPER_H_

#include "hipacc/AST/ASTNode.h"
#include <clang/AST/ExprCXX.h>
#include <limits>
#include <string>

namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \brief  Helper class which contains a few methods for easy clang AST handling. */
  class ClangASTHelper final
  {
  public:

    template <typename ElementType>   using VectorType = ::llvm::SmallVector< ElementType, 16U >;   // Type alias for LLVM SmallVector type

    typedef VectorType< ::clang::Expr* >          ExpressionVectorType;             //!< Type definition for a vector of expressions.
    typedef VectorType< ::clang::FunctionDecl* >  FunctionDeclarationVectorType;    //!< Type definition for a vector of function declarations.
    typedef VectorType< ::clang::QualType >       QualTypeVectorType;               //!< Type definition for a vector of qualified types.
    typedef VectorType< ::clang::Stmt* >          StatementVectorType;              //!< Type definition for a vector of statements.
    typedef VectorType< std::string >             StringVectorType;                 //!< Type definition for a vector of strings.

  private:

    ::clang::ASTContext   &_rCtx;   //!< A reference to the current AST context.


    ClangASTHelper(const ClangASTHelper &) = delete;
    ClangASTHelper& operator=(const ClangASTHelper &) = delete;

  public:

    /** \brief  Constructor.
     *  \param  rAstContext   A reference to the current AST context. */
    ClangASTHelper(::clang::ASTContext &rAstContext) : _rCtx(rAstContext)   {}

    /** \brief  Returns a reference to the current AST context. */
    inline ::clang::ASTContext& GetASTContext()   { return _rCtx; }

    /** \brief  Returns the corresponding array type for a qualified clang type.
     *  \param  crElementType   A reference to the qualified type whose array type shall be returned.
     *  \param  cszDimension    The dimension of the array. */
    ::clang::QualType           GetConstantArrayType(const ::clang::QualType &crElementType, const size_t cszDimension);

    /** \brief  Returns the corresponding pointer type for a qualified clang type.
     *  \param  crPointeeType   A reference to the qualified type whose pointer type shall be returned. */
    inline ::clang::QualType    GetPointerType(const ::clang::QualType &crPointeeType)  { return GetASTContext().getPointerType(crPointeeType); }


    /** \name AST node creation methods */
    //@{

    /** \brief  Creates an subscript expression.
     *  \param  pArrayRef         A pointer to the expression which represents the array.
     *  \param  pIndexExpression  A pointer to the expression object, which returns the index of the subscript.
     *  \param  crReturnType      The return type of the array subscript.
     *  \param  bIsLValue         Specifies, whether the array subscript expression is used as a L-value of another expression. */
    ::clang::ArraySubscriptExpr*      CreateArraySubscriptExpression(::clang::Expr *pArrayRef, ::clang::Expr *pIndexExpression, const ::clang::QualType &crReturnType, bool bIsLValue = false);

    /** \brief  Creates a binary operator object of a specified type.
     *  \param  pLhs            A pointer to the expression object, which shall be on the left-hand-side.
     *  \param  pRhs            A pointer to the expression object, which shall be on the right-hand-side.
     *  \param  eOperatorKind   The type of the binary operator.
     *  \param  crReturnType    The return type of the operator expression. */
    ::clang::BinaryOperator*          CreateBinaryOperator(::clang::Expr *pLhs, ::clang::Expr *pRhs, ::clang::BinaryOperatorKind eOperatorKind, const ::clang::QualType &crReturnType);

    /** \brief  Creates a binary operator object which represents the "comma" operator.
     *  \param  pLhs  A pointer to the expression object, which shall be on the left-hand-side.
     *  \param  pRhs  A pointer to the expression object, which shall be on the right-hand-side. */
    ::clang::BinaryOperator*          CreateBinaryOperatorComma(::clang::Expr *pLhs, ::clang::Expr *pRhs);

    /** \brief  Creates a binary operator object which represents a "less than" comparison.
     *  \param  pLhs  A pointer to the expression object, which shall be on the left-hand-side.
     *  \param  pRhs  A pointer to the expression object, which shall be on the right-hand-side. */
    ::clang::BinaryOperator*          CreateBinaryOperatorLessThan(::clang::Expr *pLhs, ::clang::Expr *pRhs);

    /** \brief  Creates a bool literal expression (i.e. a compile time constant).
     *  \param  bValue  The value of the bool literal. */
    ::clang::CXXBoolLiteralExpr*      CreateBoolLiteral(bool bValue);

    /** \brief  Creates a <b>break</b> statement. */
    ::clang::BreakStmt*               CreateBreakStatement();

    /** \brief  Wraps a statement object into a compound statement object.
     *  \param  pStatement  A pointer to the statement object, which shall be encapsulated into an compound statement. */
    ::clang::CompoundStmt*            CreateCompoundStatement(::clang::Stmt *pStatement);

    /** \brief  Constructs a compound statement object around a vector of statement objects.
     *  \param  crvecStatements   A reference to the statement vector. */
    ::clang::CompoundStmt*            CreateCompoundStatement(const StatementVectorType &crvecStatements);

    /** \brief  Constructs a conditional operator expression object (i.e. the "<cond> ? <expr_1> : <expr_2>" operator).
     *  \param  pCondition    A pointer to the expression object, which represents the condition.
     *  \param  pThenExpr     A pointer to the expression object, which will be returned when the condition is evaluated to <b>true</b>.
     *  \param  pElseExpr     A pointer to the expression object, which will be returned when the condition is evaluated to <b>false</b>.
     *  \param  crReturnType  The return type of the operator expression. */
    ::clang::ConditionalOperator*     CreateConditionalOperator(::clang::Expr *pCondition, ::clang::Expr *pThenExpr, ::clang::Expr *pElseExpr, const ::clang::QualType &crReturnType);

    /** \brief  Creates a <b>continue</b> statement. */
    ::clang::ContinueStmt*            CreateContinueStatement();

    /** \brief  Constructs a declaration reference expression which points to a specific declaration.
     *  \param  pValueDecl  A pointer to the value declaration object. */
    ::clang::DeclRefExpr*             CreateDeclarationReferenceExpression(::clang::ValueDecl *pValueDecl);

    /** \brief  Constructs a declaration statement for a specific declaration.
     *  \param  pDeclRef  A pointer to a declaration reference expression object which points to the specific declaration. */
    ::clang::DeclStmt*                CreateDeclarationStatement(::clang::DeclRefExpr *pDeclRef);

    /** \brief  Constructs a declaration statement for a specific declaration.
     *  \param  pValueDecl  A pointer to the value declaration object. */
    ::clang::DeclStmt*                CreateDeclarationStatement(::clang::ValueDecl *pValueDecl);

    /** \brief    Creates a floating point literal expression (i.e. a compile time constant).
     *  \tparam   ValueType The value type of the floating point literal (must be <b>float</b> or <b>double</b).
     *  \param    TValue    The value of the floating point literal. */
    template <typename ValueType>
    ::clang::FloatingLiteral*         CreateFloatingLiteral(ValueType TValue)
    {
      static_assert( ! std::numeric_limits< ValueType >::is_integer, "The value type of a floating point literal cannot be of an integer type!" );

      return ASTNode::createFloatingLiteral(GetASTContext(), TValue);
    }

    /** \brief  Constructs a function call expression.
     *  \param  pFunctionDecl   A pointer to the function declaration which the constructed call shall point to.
     *  \param  crvecArguments  A vector containing the argument expressions for the function call. */
    ::clang::CallExpr*                CreateFunctionCall(::clang::FunctionDecl *pFunctionDecl, const ExpressionVectorType &crvecArguments);

    /** \brief  Constructs a function declaration statement.
     *  \param  strFunctionName     The desired name of the newly declared function.
     *  \param  crReturnType        The qualified return type of the function.
     *  \param  crvecArgumentNames  A vector containing the names of the function arguments.
     *  \param  crvecArgumentTypes  A vector containing the qualified types of the function arguments. */
    ::clang::FunctionDecl*            CreateFunctionDeclaration(std::string strFunctionName, const ::clang::QualType &crReturnType, const StringVectorType &crvecArgumentNames, const QualTypeVectorType &crvecArgumentTypes);

    /** \brief  Constructs an <b>"if-then-else"</b>-statement.
     *  \param  pCondition    A pointer to the condition expression of the <b>if</b>-branch.
     *  \param  pThenBranch   A pointer to the body statement of the <b>if</b>-branch.
     *  \param  pElseBranch   A pointer to the body statement of the <b>else</b>-branch. If set to <b>nullptr</b>, no <b>else</b>-branch will be created. */
    ::clang::IfStmt*                  CreateIfStatement(::clang::Expr *pCondition, ::clang::Stmt *pThenBranch, ::clang::Stmt *pElseBranch = nullptr);

    /** \brief    Constructs a multi-branch <b>if</b>-statement (i.e. a <b>"if-{else if}-else"</b>-statement).
     *  \param    crvecConditions     A vector containing the conditions of all <b>if / else if</b> branches.
     *  \param    crvecBranchBodies   A vector containing the body statements of all conditional branches.
     *  \param    pDefaultBranch      A pointer to the body statement of the final <b>else</b>-branch. If set to <b>nullptr</b>, no <b>else</b>-branch will be created.
     *  \remarks  The number of conditions must be equal to the number of branch bodies. */
    ::clang::IfStmt*                  CreateIfStatement(const ExpressionVectorType &crvecConditions, const StatementVectorType &crvecBranchBodies, ::clang::Stmt *pDefaultBranch = nullptr);

    /** \brief  Creates an implicit cast expression object.
     *  \param  pOperandExpression  A pointer to the expression object whose return type shall be implicitly casted.
     *  \param  crReturnType        The qualified return type of the cast.
     *  \param  eCastKind           The internal kind of the cast.
     *  \param  bIsLValue           Specifies, whether the implicit cast expression is used as a L-value of another expression. */
    ::clang::ImplicitCastExpr*        CreateImplicitCastExpression(::clang::Expr *pOperandExpression, const ::clang::QualType &crReturnType, ::clang::CastKind eCastKind, bool bIsLValue = false);

    /** \brief  Constructs an init list expression object around a vector of expressions.
     *  \param  crvecExpressions  A reference to the expression vector. */
    ::clang::InitListExpr*            CreateInitListExpression(const ExpressionVectorType &crvecExpressions);

    /** \brief    Creates an integer literal expression (i.e. a compile time constant).
     *  \tparam   ValueType The value type of the integer literal (must be integral).
     *  \param    TValue    The value of the integer literal. */
    template <typename ValueType>
    ::clang::IntegerLiteral*          CreateIntegerLiteral(ValueType TValue)
    {
      static_assert( std::numeric_limits< ValueType >::is_integer, "The value type of an integer literal must be of an integer type!" );

      return ASTNode::createIntegerLiteral(GetASTContext(), TValue);
    }

    /** \brief    Creates a literal expression (i.e. a compile time constant).
     *  \tparam   ValueType   The value type of the literal.
     *  \param    TValue      The value of the literal.
     *  \remarks  Depending on the value type, this function construct a bool, integer or floating point literal. */
    template <typename ValueType>
    ::clang::Expr*                    CreateLiteral(ValueType TValue);

    /** \brief  Creates a <b>do-while</b>-loop statement.
     *  \param  pCondition  The condition expression of the loop.
     *  \param  pBody       The statement which represents the loop body. */
    ::clang::DoStmt*                  CreateLoopDoWhile(::clang::Expr *pCondition, ::clang::Stmt *pBody);

    /** \brief  Creates a <b>for</b>-loop statement.
     *  \param  pCondition    The condition expression of the loop.
     *  \param  pBody         The statement which represents the loop body.
     *  \param  pInitializer  The initializer statement of the for-loop (can be <b>NULL</b>).
     *  \param  pIncrement    The increment expression of the for-loop, i.e. the expression which will be evaluated after each iteration (can be <b>NULL</b>). */
    ::clang::ForStmt*                 CreateLoopFor(::clang::Expr *pCondition, ::clang::Stmt *pBody, ::clang::Stmt *pInitializer = nullptr, ::clang::Expr *pIncrement = nullptr);

    /** \brief  Creates a <b>while</b>-loop statement.
     *  \param  pCondition  The condition expression of the loop.
     *  \param  pBody       The statement which represents the loop body. */
    ::clang::WhileStmt*               CreateLoopWhile(::clang::Expr *pCondition, ::clang::Stmt *pBody);


    /** \brief  Creates a parenthesis expression around another expression.
     *  \param  pSubExpression  A pointer to the expression object which shall be encapsulated into a parenthesis expression. */
    ::clang::ParenExpr*               CreateParenthesisExpression(::clang::Expr *pSubExpression);

    /** \brief  Constructs a post increment statement for a declaration reference expression object.
     *  \param  pDeclRef  A pointer to the declaration reference expression, which shall be used in the post increment operator. */
    ::clang::UnaryOperator*           CreatePostIncrementOperator(::clang::DeclRefExpr *pDeclRef);

    /** \brief  Creates a reinterpret cast expression object.
     *  \param  pOperandExpression  A pointer to the expression object whose return type shall be implicitly casted.
     *  \param  crReturnType        The qualified return type of the cast.
     *  \param  eCastKind           The internal kind of the cast.
     *  \param  bIsLValue           Specifies, whether the reinterpret cast expression is used as a L-value of another expression. */
    ::clang::CXXReinterpretCastExpr*  CreateReinterpretCast(::clang::Expr *pOperandExpression, const ::clang::QualType &crReturnType, ::clang::CastKind eCastKind, bool bIsLValue = false);

    /** \brief  Creates a <b>return</b> statement. 
     *  \param  pReturnValue  A pointer to an expression object whose result shall be returned by the <b>return</b> statement (if set to <b>nullptr</b>, nothing will be returned). */
    ::clang::ReturnStmt*              CreateReturnStatement(::clang::Expr *pReturnValue = nullptr);

    /** \brief  Creates a static cast expression object.
     *  \param  pOperandExpression  A pointer to the expression object whose return type shall be implicitly casted.
     *  \param  crReturnType        The qualified return type of the cast.
     *  \param  eCastKind           The internal kind of the cast.
     *  \param  bIsLValue           Specifies, whether the static cast expression is used as a L-value of another expression. */
    ::clang::CXXStaticCastExpr*       CreateStaticCast(::clang::Expr *pOperandExpression, const ::clang::QualType &crReturnType, ::clang::CastKind eCastKind, bool bIsLValue = false);

    /** \brief  Creates a string literal expression (i.e. a constant C-string).
     *  \param  strValue  The value of the string literal. */
    ::clang::StringLiteral*           CreateStringLiteral(std::string strValue);

    /** \brief  Creates an unary operator object of a specified type.
     *  \param  pSubExpression  A pointer to the expression object, which shall be the sub-expression of the operator.
     *  \param  eOperatorKind   The type of the unary operator.
     *  \param  crReturnType    The return type of the operator expression. */
    ::clang::UnaryOperator*           CreateUnaryOperator(::clang::Expr *pSubExpression, ::clang::UnaryOperatorKind eOperatorKind, const ::clang::QualType &crResultType);

    /** \brief    Creates a new variable declaration object.
     *  \param    pParentFunction     A pointer to the declaration context which the new variable shall be declared in.
     *  \param    crstrVariableName   The name of the newly declared variable.
     *  \param    crVariableType      The qualified type of newly declared variable.
     *  \param    pInitExpression     A pointer to the initialization expression object for the variable declaration (i.e. the R-value of the assignment).
     *  \remarks  The created variable declaration is automatically added to the declaration context of the specified function declaration. */
    ::clang::VarDecl*                 CreateVariableDeclaration(::clang::DeclContext *pDeclContext, const std::string &crstrVariableName, const ::clang::QualType &crVariableType, ::clang::Expr *pInitExpression);

    /** \brief    Creates a new variable declaration object.
     *  \param    pParentFunction     A pointer to the function declaration object in whose context the new variable shall be declared.
     *  \param    crstrVariableName   The name of the newly declared variable.
     *  \param    crVariableType      The qualified type of newly declared variable.
     *  \param    pInitExpression     A pointer to the initialization expression object for the variable declaration (i.e. the R-value of the assignment).
     *  \remarks  The created variable declaration is automatically added to the declaration context of the specified function declaration. */
    ::clang::VarDecl*                 CreateVariableDeclaration(::clang::FunctionDecl *pParentFunction, const std::string &crstrVariableName, const ::clang::QualType &crVariableType, ::clang::Expr *pInitExpression);

    //@}


  public:


    /** \brief  Counts the number of declaration references to a specific declaration inside a statement tree.
     *  \param  pStatement          A pointer to the root of the statement tree which shall be parsed for the specified declaration references.
     *  \param  crstrReferenceName  The name of the declaration reference whose appearances shall be counted. */
    static unsigned int             CountNumberOfReferences(::clang::Stmt *pStatement, const std::string &crstrReferenceName);

    /** \brief    Looks up a specific declaration.
     *  \param    pFunction       A pointer to the function declaration object whose declaration context will be searched for the specified declaration.
     *  \param    crstrDeclName   The name of the declaration which shall be searched for.
     *  \return   If successful, a pointer to a newly created declaration reference expression for the found declaration, and zero otherwise. */
    ::clang::DeclRefExpr*           FindDeclaration(::clang::FunctionDecl *pFunction, const std::string &crstrDeclName);

    /** \brief  Returns the fully qualified name of a function declaration, i.e. the function name with all preceding namespace names.
     *  \param  pFunctionDecl   A pointer to the function declaration object, whose fully qualified name shall be retrieved. */
    static std::string              GetFullyQualifiedFunctionName(::clang::FunctionDecl *pFunctionDecl);

    /** \brief    Returns a vector of all known function declarations in the encapsulated AST context.
     *  \remarks  This method parses all namespaces, beginning with the global namespace. */
    FunctionDeclarationVectorType   GetKnownFunctionDeclarations();

    /** \brief    Returns a vector of all known function declarations inside a namespace declaration.
     *  \param    pNamespaceDecl  A pointer to the namespace declaration object which shall be parsed for function declarations.
     *  \remarks  This method also parses all child namespace declarations of the specified namespace declaration. */
    FunctionDeclarationVectorType   GetNamespaceFunctionDeclarations(::clang::NamespaceDecl *pNamespaceDecl);

    /** \brief  Checks whether a statement tree has only one branch (i.e. none of its nodes has more than one child).
     *  \param  pStatement  A pointer to the root of the statement tree. */
    static bool                     IsSingleBranchStatement(::clang::Stmt *pStatement);

    /** \brief  Replaces <b>all</b> instances of a declaration reference in a statement tree by a new value declaration.
     *  \param  pStatement        A pointer to the root of the statement tree which shall be parsed for the specified declaration references.
     *  \param  crstrDeclRefName  The name of the declaration reference which shall be replaced.
     *  \param  pNewDecl          A pointer to the value declaration to which all reference will be updated. */
    static void                     ReplaceDeclarationReferences(::clang::Stmt* pStatement, const std::string &crstrDeclRefName, ::clang::ValueDecl *pNewDecl);
  };


  // Template function specializations
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(bool         TValue)   { return CreateBoolLiteral(TValue);     }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(int8_t       TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(uint8_t      TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(int16_t      TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(uint16_t     TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(int32_t      TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(uint32_t     TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(int64_t      TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(uint64_t     TValue)   { return CreateIntegerLiteral(TValue);  }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(float        TValue)   { return CreateFloatingLiteral(TValue); }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(double       TValue)   { return CreateFloatingLiteral(TValue); }
  template<> inline ::clang::Expr* ClangASTHelper::CreateLiteral(std::string  TValue)   { return CreateStringLiteral(TValue);   }

} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_CLANG_AST_HELPER_H_

// vim: set ts=2 sw=2 sts=2 et ai:

