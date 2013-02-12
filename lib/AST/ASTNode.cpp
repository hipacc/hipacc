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

//===--- ASTNode.cpp - Creating AST Nodes without Sema --------------------===//
//
// This file implements the creation of AST nodes without Sema context.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTNode.h"


namespace clang {
namespace hipacc {
namespace ASTNode {

FunctionDecl *createFunctionDecl(ASTContext &Ctx, DeclContext *DC, StringRef
    Name, QualType RT, unsigned int numArgs, QualType *ArgTypes, std::string
    *ArgNames, bool isVariadic) {
  bool hasWrittenPrototype = true;
  QualType T;
  SmallVector<QualType, 16> ArgTys;
  SmallVector<ParmVarDecl*, 16> Params;
  Params.reserve(numArgs);
  ArgTys.reserve(numArgs);

  // first create QualType for parameters
  if (numArgs == 0) {
    // simple void foo(), where the incoming RT is the result type.
    T = Ctx.getFunctionNoProtoType(RT);
    hasWrittenPrototype = false;
  } else {
    // otherwise, we have a function with an argument list
    for (unsigned int i=0; i<numArgs; ++i) {
      ParmVarDecl *Param = ParmVarDecl::Create(Ctx, DC, SourceLocation(),
          SourceLocation(), &Ctx.Idents.get(ArgNames[i]), ArgTypes[i], NULL,
          SC_None, SC_None, 0);
      QualType ArgTy = Param->getType();

      assert(!ArgTy.isNull() && "Couldn't parse type?");
      if (ArgTy->isPromotableIntegerType()) {
        ArgTy = Ctx.getPromotedIntegerType(ArgTy);
      } else if (const BuiltinType* BTy = ArgTy->getAs<BuiltinType>()) {
        if (BTy->getKind() == BuiltinType::Float) {
          ArgTy = Ctx.DoubleTy;
        }
      }
      ArgTys.push_back(ArgTy);
      Params.push_back(Param);
    }

    FunctionProtoType::ExtProtoInfo FPT = FunctionProtoType::ExtProtoInfo();
    FPT.Variadic = isVariadic;
    T = Ctx.getFunctionType(RT, ArgTys.data(), ArgTys.size(), FPT);
  }

  // create function name identifier
  IdentifierInfo &info = Ctx.Idents.get(Name);
  DeclarationName DecName(&info);

  // create function declaration
  assert(T.getTypePtr()->isFunctionType());
  FunctionDecl *FD = FunctionDecl::Create(Ctx, DC, SourceLocation(),
      SourceLocation(), DecName, T, NULL, SC_None, SC_None, false,
      hasWrittenPrototype);

  // add Decl objects for each parameter to the FunctionDecl
  DeclContext *DCF = FunctionDecl::castToDeclContext(FD);
  for (unsigned int i=0; i<numArgs; ++i) {
    Params.data()[i]->setDeclContext(FD);
    DCF->addDecl(Params.data()[i]);
  }
  FD->setParams(Params);

  // add Declaration
  DC->addDecl(FD);

  // Note: the body is not added here!
  return FD;
}


CallExpr *createFunctionCall(ASTContext &Ctx, FunctionDecl *FD, SmallVector<Expr
    *, 16> Expr) {
  // get reference to FD
  DeclRefExpr *FDRef = createDeclRefExpr(Ctx, FD);
  // now, we cast the reference to a pointer to the function type.
  QualType pToFunc = Ctx.getPointerType(FD->getType());

  ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx, pToFunc,
      CK_FunctionToPointerDecay, FDRef, NULL, VK_RValue);

  const FunctionType *FT = FD->getType()->getAs<FunctionType>();

  // create call expression
  CallExpr *E = new (Ctx) CallExpr(Ctx, Stmt::CallExprClass,
      Stmt::EmptyShell());
  E->setNumArgs(Ctx, Expr.size());
  E->setRParenLoc(SourceLocation());
  E->setCallee(ICE);
  for (unsigned I=0, N=Expr.size(); I!=N; ++I) {
    E->setArg(I, Expr.data()[I]);
  }
  E->setType(FT->getCallResultType(Ctx));

  return E;
}


CStyleCastExpr *createCStyleCastExpr(ASTContext &Ctx, QualType T, CastKind Kind,
    Expr *Operand, CXXCastPath *BasePath, TypeSourceInfo *WrittenTy) {
  CStyleCastExpr *E = CStyleCastExpr::Create(Ctx, T, VK_RValue, Kind, Operand,
      BasePath, WrittenTy, SourceLocation(), SourceLocation());

  return E;
}


ImplicitCastExpr *createImplicitCastExpr( ASTContext &Ctx, QualType T, CastKind
    Kind, Expr *Operand, CXXCastPath *BasePath, ExprValueKind Cat) {
  ImplicitCastExpr *E = ImplicitCastExpr::Create(Ctx, T, Kind, Operand,
      BasePath, Cat);

  return E;
}


IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int32_t val) {
  IntegerLiteral *E = IntegerLiteral::Create(Ctx, llvm::APInt(32, val),
      Ctx.IntTy, SourceLocation());

  return E;
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint32_t val) {
  IntegerLiteral *E = IntegerLiteral::Create(Ctx, llvm::APInt(32, val),
      Ctx.UnsignedIntTy, SourceLocation());

  return E;
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int64_t val) {
  IntegerLiteral *E = IntegerLiteral::Create(Ctx, llvm::APInt(64, val),
      Ctx.LongTy, SourceLocation());

  return E;
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint64_t val) {
  IntegerLiteral *E = IntegerLiteral::Create(Ctx, llvm::APInt(64, val),
      Ctx.UnsignedLongTy, SourceLocation());

  return E;
}


FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, float val) {
  FloatingLiteral *E = FloatingLiteral::Create(Ctx, llvm::APFloat(val), false,
      Ctx.FloatTy, SourceLocation());

  return E;
}
FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, double val) {
  FloatingLiteral *E = FloatingLiteral::Create(Ctx, llvm::APFloat(val), false,
      Ctx.DoubleTy, SourceLocation());

  return E;
}


StringLiteral *createStringLiteral(ASTContext &Ctx, StringRef Name) {
  StringLiteral *E = StringLiteral::CreateEmpty(Ctx, 1);
  QualType StrTy = Ctx.CharTy;

  if (Name.size() > 1) {
    StrTy = Ctx.getConstantArrayType(StrTy, llvm::APInt(32, Name.size()+1),
        ArrayType::Normal, 0);
  }

  E->setString(Ctx, Name, StringLiteral::UTF8, false);
  E->setType(StrTy);

  return E;
}


DeclStmt *createDeclStmt(ASTContext &Ctx, Decl *VD) {
  DeclStmt *S = new (Ctx) DeclStmt(Stmt::EmptyShell());

  S->setStartLoc(SourceLocation());
  S->setEndLoc(SourceLocation());
  S->setDeclGroup(DeclGroupRef(VD));

  return S;
}


VarDecl *createVarDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name,
    QualType T, Expr *init) {
  VarDecl *VD = VarDecl::Create(Ctx, NULL, SourceLocation(), SourceLocation(),
      NULL, QualType(), NULL, SC_None, SC_None);

  VD->setDeclContext(DC);
  VD->setLocation(SourceLocation());
  VD->setDeclName(&Ctx.Idents.get(Name));
  VD->setType(T);
  VD->setInit(init);
  // set VarDecl as being used - required for CodeGen
  VD->setUsed(true);

  return VD;
}


RecordDecl *createRecordDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name,
    TagDecl::TagKind TK, unsigned int numDecls, QualType *declTypes, StringRef
    *declNames) {
  RecordDecl *RD = RecordDecl::Create(Ctx, TK, DC, SourceLocation(),
      SourceLocation(), &Ctx.Idents.get(Name));

  for (unsigned int i=0; i<numDecls; i++) {
    RD->addDecl(createVarDecl(Ctx, RD, declNames[i], declTypes[i], NULL));
  }
  if (numDecls) {
    RD->setCompleteDefinition(true);
  }

  return RD;
}


MemberExpr *createMemberExpr(ASTContext &Ctx, Expr *base, bool isArrow,
    ValueDecl *memberdecl, QualType T) {
  MemberExpr *ME = new (Ctx) MemberExpr(base, isArrow, memberdecl,
      SourceLocation(), T, VK_LValue, OK_Ordinary);

  return ME;
}


CompoundStmt *createCompoundStmt(ASTContext &Ctx, SmallVector<Stmt *, 16> Stmts)
{
  CompoundStmt *S = new (Ctx) CompoundStmt(Stmt::EmptyShell());

  S->setStmts(Ctx, Stmts.data(), Stmts.size());
  S->setLBracLoc(SourceLocation());
  S->setRBracLoc(SourceLocation());

  return S;
}


ReturnStmt *createReturnStmt(ASTContext &Ctx, Expr *E) {
  ReturnStmt *S = new (Ctx) ReturnStmt(Stmt::EmptyShell());

  S->setRetValue(E);
  S->setReturnLoc(SourceLocation());
  S->setNRVOCandidate(NULL);

  return S;
}


IfStmt *createIfStmt(ASTContext &Ctx, Expr *cond, Stmt *then, Stmt *elsev, Decl
    *decl) {
  IfStmt *S = new (Ctx) IfStmt(Stmt::EmptyShell());

  S->setConditionVariable(Ctx, cast_or_null<VarDecl>(decl));
  S->setCond(cond);
  S->setThen(then);
  S->setElse(elsev);
  S->setIfLoc(SourceLocation());
  S->setElseLoc(SourceLocation());

  return S;
}


ForStmt *createForStmt(ASTContext &Ctx, Stmt *Init, Expr *Cond, Expr *Inc, Stmt
    *Body, VarDecl *condVar) {
  ForStmt *S = new (Ctx) ForStmt(Stmt::EmptyShell());

  S->setInit(Init);
  S->setCond(Cond);
  S->setConditionVariable(Ctx, cast_or_null<VarDecl>(condVar));
  S->setInc(Inc);
  S->setBody(Body);
  S->setForLoc(SourceLocation());
  S->setLParenLoc(SourceLocation());
  S->setRParenLoc(SourceLocation());

  return S;
}


WhileStmt *createWhileStmt(ASTContext &Ctx, VarDecl *Var, Expr *Cond, Stmt
    *Body) {
  WhileStmt *S = new (Ctx) WhileStmt(Stmt::EmptyShell());

  S->setConditionVariable(Ctx, cast_or_null<VarDecl>(Var));
  S->setCond(Cond);
  S->setBody(Body);
  S->setWhileLoc(SourceLocation());

  return S;
}


UnaryOperator *createUnaryOperator(ASTContext &Ctx, Expr *input,
    UnaryOperator::Opcode opc, QualType type) {
  return new (Ctx) UnaryOperator(input, opc, type, VK_RValue, OK_Ordinary,
      SourceLocation());
}


BinaryOperator *createBinaryOperator(ASTContext &Ctx, Expr *lhs, Expr *rhs,
    BinaryOperator::Opcode opc, QualType ResTy) {
  // check if this is an CompoundAssignOperator
  if (opc > BO_Assign && opc <= BO_OrAssign)
    return createCompoundAssignOperator(Ctx, lhs, rhs, opc, ResTy);

  BinaryOperator *E = new (Ctx) BinaryOperator(Stmt::EmptyShell());

  E->setLHS(lhs);
  E->setRHS(rhs);
  E->setOpcode(opc);
  E->setOperatorLoc(SourceLocation());
  E->setFPContractable(false);
  E->setType(ResTy);

  return E;
}


CompoundAssignOperator *createCompoundAssignOperator(ASTContext &Ctx, Expr *lhs,
    Expr *rhs, BinaryOperator::Opcode opc, QualType ResTy) {
  CompoundAssignOperator *E = new (Ctx) CompoundAssignOperator(Stmt::EmptyShell());

  E->setLHS(lhs);
  E->setRHS(rhs);
  E->setOpcode(opc);
  E->setOperatorLoc(SourceLocation());
  E->setFPContractable(false);
  E->setType(ResTy);
  E->setComputationLHSType(ResTy);
  E->setComputationResultType(ResTy);

  return E;
}


ParenExpr *createParenExpr(ASTContext &Ctx, Expr *val) {
  return new (Ctx) ParenExpr(SourceLocation(), SourceLocation(), val);
}


ExtVectorElementExpr *createExtVectorElementExpr(ASTContext &Ctx, QualType ty,
    Expr *base, StringRef name) {
  return new (Ctx) ExtVectorElementExpr(ty, VK_RValue, base,
      Ctx.Idents.get(name), SourceLocation());
}


DeclRefExpr *createDeclRefExpr(ASTContext &Ctx, ValueDecl *VD) {
  DeclRefExpr *E = DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc(),
      SourceLocation(), VD, false, VD->getLocation(), VD->getType(), VK_LValue,
      0, 0);

  return E;
}


LabelDecl *createLabelDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name) {
  LabelDecl *LD = LabelDecl::Create(Ctx, DC, SourceLocation(),
      &Ctx.Idents.get(Name));

  return LD;
}

LabelStmt *createLabelStmt(ASTContext &Ctx, LabelDecl *LD, Stmt *Stmt) {
  return new (Ctx) LabelStmt(SourceLocation(), LD, Stmt ? Stmt : new (Ctx)
      NullStmt(SourceLocation()));
}


GotoStmt *createGotoStmt(ASTContext &Ctx, LabelDecl *LD) {
  return new (Ctx) GotoStmt(LD, SourceLocation(), SourceLocation());
}

} // end ASTNode namespace
} // end hipacc namespace
} // end clang namespace

// vim: set ts=2 sw=2 sts=2 et ai:

