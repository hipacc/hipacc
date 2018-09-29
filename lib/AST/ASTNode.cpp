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
    Name, QualType RT, ArrayRef<QualType> ArgTypes, ArrayRef<std::string>
    ArgNames, bool isVariadic) {
  QualType QT;
  SmallVector<ParmVarDecl*, 16> Params;
  Params.reserve(ArgTypes.size());

  // first create QualType for parameters
  if (!ArgTypes.size()) {
    // simple void foo(), where the incoming RT is the result type.
    QT = Ctx.getFunctionNoProtoType(RT);
  } else {
    // otherwise, we have a function with an argument list
    for (size_t i=0; i<ArgTypes.size(); ++i) {
      ParmVarDecl *Param = ParmVarDecl::Create(Ctx, DC, SourceLocation(),
          SourceLocation(), &Ctx.Idents.get(ArgNames[i]), ArgTypes[i], nullptr,
          SC_None, 0);
      Params.push_back(Param);
    }

    FunctionProtoType::ExtProtoInfo FPT = FunctionProtoType::ExtProtoInfo();
    FPT.Variadic = isVariadic;
    QT = Ctx.getFunctionType(RT, ArgTypes, FPT);
  }

  // create function name identifier
  IdentifierInfo &info = Ctx.Idents.get(Name);
  DeclarationName DecName(&info);

  // create function declaration
  assert(QT->isFunctionType());
  FunctionDecl *FD = FunctionDecl::Create(Ctx, DC, SourceLocation(),
      SourceLocation(), DecName, QT, nullptr, SC_None);

  // add Decl objects for each parameter to the FunctionDecl
  DeclContext *DCF = FunctionDecl::castToDeclContext(FD);
  for (auto param : Params) {
    param->setDeclContext(FD);
    DCF->addDecl(param);
  }
  FD->setParams(Params);

  // add Declaration
  DC->addDecl(FD);

  // Note: the body is not added here!
  return FD;
}


CallExpr *createFunctionCall(ASTContext &Ctx, FunctionDecl *FD, ArrayRef<Expr *>
    Expr) {
  // get reference to FD
  DeclRefExpr *FDRef = createDeclRefExpr(Ctx, FD);
  // now, we cast the reference to a pointer to the function type.
  QualType pToFunc = Ctx.getPointerType(FD->getType());

  ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx, pToFunc,
      CK_FunctionToPointerDecay, FDRef, nullptr, VK_RValue);

  const FunctionType *FT = FD->getType()->getAs<FunctionType>();

  // create call expression
  CallExpr *E = new (Ctx) CallExpr(Ctx, Stmt::CallExprClass,
      Stmt::EmptyShell());
  E->setNumArgs(Ctx, Expr.size());
  E->setRParenLoc(SourceLocation());
  E->setCallee(ICE);
  for (size_t I=0, N=Expr.size(); I!=N; ++I) {
    E->setArg(I, Expr[I]);
  }
  E->setType(FT->getCallResultType(Ctx));

  return E;
}


CStyleCastExpr *createCStyleCastExpr(ASTContext &Ctx, QualType T, CastKind Kind,
    Expr *Operand, CXXCastPath *BasePath, TypeSourceInfo *WrittenTy) {
  return CStyleCastExpr::Create(Ctx, T, VK_RValue, Kind, Operand, BasePath,
      WrittenTy, SourceLocation(), SourceLocation());
}


ImplicitCastExpr *createImplicitCastExpr( ASTContext &Ctx, QualType T, CastKind
    Kind, Expr *Operand, CXXCastPath *BasePath, ExprValueKind Cat) {
  return ImplicitCastExpr::Create(Ctx, T, Kind, Operand, BasePath, Cat);
}


IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int32_t val) {
  return IntegerLiteral::Create(Ctx, llvm::APInt(32, val), Ctx.IntTy,
      SourceLocation());
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint32_t val) {
  return IntegerLiteral::Create(Ctx, llvm::APInt(32, val), Ctx.UnsignedIntTy,
      SourceLocation());
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, int64_t val) {
  return IntegerLiteral::Create(Ctx, llvm::APInt(64, val), Ctx.LongTy,
      SourceLocation());
}
IntegerLiteral *createIntegerLiteral(ASTContext &Ctx, uint64_t val) {
  return IntegerLiteral::Create(Ctx, llvm::APInt(64, val), Ctx.UnsignedLongTy,
      SourceLocation());
}


size_t getBuiltinTypeSize(const BuiltinType *BT) {
  size_t size = 1;

  switch (BT->getKind()) {
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      size = 8;
      break;
    case BuiltinType::Char16:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Half:
      size = 16;
      break;
    case BuiltinType::Char32:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Float:
      size = 32;
      break;
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::Double:
      size = 64;
      break;
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::Int128:
    case BuiltinType::UInt128:
    //case BuiltinType::LongDouble: //???
      size = 128;
      break;
    default:
      assert(false && "Type not supported");
      break;
  }

  return size;
}


VectorTypeInfo createVectorTypeInfo(const VectorType *VT) {
  VectorTypeInfo info;
  info.elementType = VT->getElementType().getAsString();
  info.elementCount = VT->getNumElements();
  info.elementWidth = 0;

  if (isa<BuiltinType>(VT->getElementType())) {
    info.elementWidth =
      getBuiltinTypeSize(dyn_cast<BuiltinType>(VT->getElementType()));
  }

  return info;
}


FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, float val) {
  return FloatingLiteral::Create(Ctx, llvm::APFloat(val), false, Ctx.FloatTy,
      SourceLocation());
}


FloatingLiteral *createFloatingLiteral(ASTContext &Ctx, double val) {
  return FloatingLiteral::Create(Ctx, llvm::APFloat(val), false, Ctx.DoubleTy,
      SourceLocation());
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
  VarDecl *VD = VarDecl::Create(Ctx, nullptr, SourceLocation(),
          SourceLocation(), nullptr, QualType(), nullptr, SC_None);

  VD->setDeclContext(DC);
  VD->setLocation(SourceLocation());
  VD->setDeclName(&Ctx.Idents.get(Name));
  VD->setType(T);
  VD->setInit(init);
  VD->setIsUsed(); // set VarDecl as being used - required for CodeGen

  return VD;
}


RecordDecl *createRecordDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name,
    TagDecl::TagKind TK, ArrayRef<QualType> declTypes, ArrayRef<StringRef>
    declNames) {
  RecordDecl *RD = RecordDecl::Create(Ctx, TK, DC, SourceLocation(),
      SourceLocation(), &Ctx.Idents.get(Name));

  for (size_t i=0; i<declTypes.size(); ++i) {
    RD->addDecl(createVarDecl(Ctx, RD, declNames[i], declTypes[i], nullptr));
  }
  if (declTypes.size()) {
    RD->setCompleteDefinition(true);
  }

  return RD;
}


MemberExpr *createMemberExpr(ASTContext &Ctx, Expr *base, bool isArrow,
    ValueDecl *memberdecl, QualType T) {
  return new (Ctx) MemberExpr(base, isArrow, SourceLocation(), memberdecl,
      SourceLocation(), T, VK_LValue, OK_Ordinary);
}


CompoundStmt *createCompoundStmt(ASTContext &Ctx, ArrayRef<Stmt *> Stmts) {
  return CompoundStmt::Create(Ctx, Stmts, SourceLocation(), SourceLocation());
}


ReturnStmt *createReturnStmt(ASTContext &Ctx, Expr *E) {
  ReturnStmt *S = new (Ctx) ReturnStmt(Stmt::EmptyShell());

  S->setRetValue(E);
  S->setReturnLoc(SourceLocation());
  S->setNRVOCandidate(nullptr);

  return S;
}


IfStmt *createIfStmt(ASTContext &Ctx, Expr *cond, Stmt *then, Stmt *elsev, Decl
    *decl) {
  IfStmt *S = new (Ctx) IfStmt(Stmt::EmptyShell());

  S->setInit(nullptr);
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
  E->setFPFeatures(FPOptions());
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
  E->setFPFeatures(FPOptions());
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
  return DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc(), SourceLocation(),
      VD, false, VD->getLocation(), VD->getType(), VK_LValue, 0, 0);
}


LabelDecl *createLabelDecl(ASTContext &Ctx, DeclContext *DC, StringRef Name) {
  return LabelDecl::Create(Ctx, DC, SourceLocation(), &Ctx.Idents.get(Name));
}

LabelStmt *createLabelStmt(ASTContext &Ctx, LabelDecl *LD, Stmt *Stmt) {
  return new (Ctx) LabelStmt(SourceLocation(), LD, Stmt ? Stmt : new (Ctx)
      NullStmt(SourceLocation()));
}


GotoStmt *createGotoStmt(ASTContext &Ctx, LabelDecl *LD) {
  return new (Ctx) GotoStmt(LD, SourceLocation(), SourceLocation());
}

} // namespace ASTNode
} // namespace hipacc
} // namespace clang

// vim: set ts=2 sw=2 sts=2 et ai:

