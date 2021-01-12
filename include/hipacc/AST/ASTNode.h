//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// Copyright (c) 2010, ARM Limited
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

//===--- ASTNode.h - Creating AST Nodes without Sema ----------------------===//
//
// This file implements the creation of AST nodes without Sema context.
//
//===----------------------------------------------------------------------===//

#ifndef _HIPACC_AST_ASTNODE_H_
#define _HIPACC_AST_ASTNODE_H_

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>


namespace clang {
namespace hipacc {
namespace ASTNode {
// creates a function declaration AST node
FunctionDecl *createFunctionDecl(ASTContext &Ctx, DeclContext *DC, StringRef
    Name, QualType RT, ArrayRef<QualType> ArgTypes, ArrayRef<std::string>
    ArgNames, bool isVariadic=false);

// creates a function call AST node
CallExpr *createFunctionCall(ASTContext &Ctx, FunctionDecl *FD,
    ArrayRef<Expr *> Expr);
DeclRefExpr *createDeclRefExpr(ASTContext &Ctx, ValueDecl *decl);


// creates a variable declaration AST node
DeclStmt *createDeclStmt(ASTContext &Ctx, Decl *VD);
VarDecl *createVarDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name,
    QualType T, Expr *init=nullptr);
RecordDecl *createRecordDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name,
    TagDecl::TagKind TK, ArrayRef<QualType> declTypes, ArrayRef<StringRef>
    declNames);

// create a member expression AST node
MemberExpr *createMemberExpr(ASTContext &Ctx, Expr *base, bool isArrow,
    ValueDecl *memberdecl, QualType T);

// creates a compound statement AST node representing multiple statements
CompoundStmt *createCompoundStmt(ASTContext &Ctx, ArrayRef<Stmt *> Stmts);

// creates a return statement AST node
ReturnStmt *createReturnStmt(ASTContext &Ctx, Expr *E);


// creates a if/then/else control flow AST node
IfStmt *createIfStmt(ASTContext &Ctx, Expr *cond, Stmt *then, Stmt
        *elsev=nullptr, Decl *decl=nullptr);

ForStmt *createForStmt(ASTContext &Ctx, Stmt *Init, Expr *Cond, Expr *Inc, Stmt
    *Body, VarDecl *condVar=nullptr);

WhileStmt *createWhileStmt(ASTContext &Ctx, VarDecl *Var, Expr *Cond, Stmt
    *Body);

// creates an AST node for binary operators
UnaryOperator *createUnaryOperator(ASTContext &Ctx, Expr *input,
    UnaryOperator::Opcode opc, QualType ResTy);
BinaryOperator *createBinaryOperator(ASTContext &Ctx, Expr *lhs, Expr *rhs,
    BinaryOperator::Opcode opc, QualType ResTy);
CompoundAssignOperator *createCompoundAssignOperator(ASTContext &Ctx, Expr *lhs,
    Expr *rhs, BinaryOperator::Opcode opc, QualType ResTy);

// creates an AST node for paren expressions
ParenExpr *createParenExpr(ASTContext &Ctx, Expr *val);

// creates an AST node for vector access
ExtVectorElementExpr *createExtVectorElementExpr(ASTContext &Ctx, QualType ty,
    Expr *base, StringRef Name);

// create an explicit/implicit cast required internally for correct AST
CStyleCastExpr *createCStyleCastExpr(ASTContext &Ctx, QualType T, CastKind Kind,
    Expr *Operand, CXXCastPath *BasePath, TypeSourceInfo *WrittenTy);
ImplicitCastExpr *createImplicitCastExpr( ASTContext &Ctx, QualType T, CastKind
    Kind, Expr *Operand, CXXCastPath *BasePath, ExprValueKind Cat);

IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int32_t val);
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint32_t val);
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int64_t val);
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint64_t val);
FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, float val);
FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, double val);


typedef struct {
  std::string elementType;
  size_t elementCount;
  size_t elementWidth;
} VectorTypeInfo;

size_t getBuiltinTypeSize(const BuiltinType *BT);
VectorTypeInfo createVectorTypeInfo(const VectorType *VT);


// create label/goto statements
LabelDecl *createLabelDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name);
LabelStmt *createLabelStmt(ASTContext &Ctx, LabelDecl *LD, Stmt *Stmt);
GotoStmt *createGotoStmt(ASTContext &Ctx, LabelDecl *LD);

} // namespace ASTNode
} // namespace hipacc
} // namespace clang

#endif /* _HIPACC_AST_ASTNODE_H_ */

// vim: set ts=2 sw=2 sts=2 et ai:

