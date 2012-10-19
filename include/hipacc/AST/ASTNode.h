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

#ifndef _ASTNODE_H_
#define _ASTNODE_H_

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>


namespace clang {
namespace hipacc {
namespace ASTNode {
// creates a function declaration AST node
FunctionDecl *createFunctionDecl(ASTContext &Ctx, DeclContext *DC,
    llvm::StringRef Name, QualType RT, unsigned int numArgs, QualType *ArgTypes,
    llvm::StringRef *ArgNames, bool isVariadic=false);

// creates a function call AST node
CallExpr *createFunctionCall(ASTContext &Ctx, FunctionDecl *FD,
    llvm::SmallVector<Expr *, 16> Expr);
DeclRefExpr *createDeclRefExpr(ASTContext &Ctx, ValueDecl *decl);


// creates a variable declaration AST node
DeclStmt *createDeclStmt(ASTContext &Ctx, Decl *VD);
VarDecl *createVarDecl(ASTContext &Ctx, DeclContext *DC, llvm::StringRef Name,
    QualType T, Expr *init=NULL);
RecordDecl *createRecordDecl(ASTContext &Ctx, DeclContext *DC, llvm::StringRef
    Name, TagDecl::TagKind TK, unsigned int numDecls, QualType *declTypes,
    llvm::StringRef *declNames);

// create a member expression AST node
MemberExpr *createMemberExpr(ASTContext &Ctx, Expr *base, bool isArrow,
    ValueDecl *memberdecl, QualType T);

// creates a compound statement AST node representing multiple statements
CompoundStmt *createCompoundStmt(ASTContext &Ctx, llvm::SmallVector<Stmt *, 16>
    Stmts);

// creates a return statement AST node
ReturnStmt *createReturnStmt(ASTContext &Ctx, Expr *E);


// creates a if/then/else control flow AST node
IfStmt *createIfStmt(ASTContext &Ctx, Expr *cond, Stmt *then, Stmt *elsev=NULL,
    Decl *decl=NULL);

ForStmt *createForStmt(ASTContext &Ctx, Stmt *Init, Expr *Cond, Expr *Inc, Stmt
    *Body, VarDecl *condVar=NULL);

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
    Expr *base, llvm::StringRef Name);

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
StringLiteral *createStringLiteral(ASTContext &Ctx, llvm::StringRef Name);

// create label/goto statements
LabelDecl *createLabelDecl(ASTContext &Ctx, DeclContext *DC, llvm::StringRef
    Name);
LabelStmt *createLabelStmt(ASTContext &Ctx, LabelDecl *LD, Stmt *Stmt);
GotoStmt *createGotoStmt(ASTContext &Ctx, LabelDecl *LD);

} // end namespace ASTNode 
} // end namespace hipacc
} // end namespace clang

#endif /* _ASTNODE_H_ */

// vim: set ts=2 sw=2 sts=2 et ai:

