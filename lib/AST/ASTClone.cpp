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

//===--- ASTClone.cpp - Clone single AST Nodes ----------------------------===//
//
// This file implements the cloning of single AST nodes.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;


//===----------------------------------------------------------------------===//
// Statement/expression cloning
//===----------------------------------------------------------------------===//

void ASTTranslate::setExprProps(Expr *orig, Expr *clone) {
  clone->setTypeDependent(orig->isTypeDependent());
  clone->setValueDependent(orig->isValueDependent());
  clone->setInstantiationDependent(orig->isInstantiationDependent());
  clone->setContainsUnexpandedParameterPack(orig->containsUnexpandedParameterPack());
  clone->setValueKind(orig->getValueKind());
  clone->setObjectKind(orig->getObjectKind());
}


void ASTTranslate::setExprPropsClone(Expr *orig, Expr *clone) {
  clone->setType(orig->getType());
  setExprProps(orig, clone);
}


void ASTTranslate::setCastPath(CastExpr *orig, CXXCastPath &castPath) {
  for (CastExpr::path_iterator PI = orig->path_begin(), PE = orig->path_end();
      PI != PE; ++PI) {
    castPath.push_back(*PI);
  }
}


// Statements
Stmt *ASTTranslate::VisitStmt(Stmt *) {
  HIPACC_BASE_CLASS(Stmt);
  return NULL;
}

Stmt *ASTTranslate::VisitNullStmt(NullStmt *S) {
  return new (Ctx) NullStmt(S->getSemiLoc());
}

#ifdef NO_TRANSLATION
Stmt *ASTTranslate::VisitCompoundStmt(CompoundStmt *S) {
  CompoundStmt *result = new (Ctx) CompoundStmt(Ctx, MultiStmtArg(),
      S->getLBracLoc(), S->getLBracLoc());

  SmallVector<Stmt *, 16> body;
  for (CompoundStmt::const_body_iterator I=S->body_begin(), E=S->body_end();
      I!=E; ++I) {
    body.push_back(Clone(*I));
  }
  result->setStmts(Ctx, body.data(), body.size());

  return result;
}
#endif

Stmt *ASTTranslate::VisitLabelStmt(LabelStmt *S) {
  return new (Ctx) LabelStmt(S->getIdentLoc(), S->getDecl(),
      Clone(S->getSubStmt()));
}

Stmt *ASTTranslate::VisitAttributedStmt(AttributedStmt *S) {
  return AttributedStmt::Create(Ctx, S->getAttrLoc(), S->getAttrs(),
      Clone(S->getSubStmt()));
}

Stmt *ASTTranslate::VisitIfStmt(IfStmt *S) {
  return new (Ctx) IfStmt(Ctx, S->getIfLoc(),
      CloneDecl(S->getConditionVariable()), Clone(S->getCond()),
      Clone(S->getThen()), S->getElseLoc(), Clone(S->getElse()));
}

Stmt *ASTTranslate::VisitSwitchStmt(SwitchStmt *S) {
  SwitchStmt *result = new (Ctx) SwitchStmt(Ctx,
      CloneDecl(S->getConditionVariable()), Clone(S->getCond()));

  result->setBody(Clone(S->getBody()));
  result->setSwitchLoc(S->getSwitchLoc());

  return result;
}

Stmt *ASTTranslate::VisitWhileStmt(WhileStmt *S) {
  return new (Ctx) WhileStmt(Ctx, CloneDecl(S->getConditionVariable()),
      Clone(S->getCond()), Clone(S->getBody()), S->getWhileLoc());
}

Stmt *ASTTranslate::VisitDoStmt(DoStmt *S) {
  return new (Ctx) DoStmt(Clone(S->getBody()), Clone(S->getCond()),
      S->getDoLoc(), S->getWhileLoc(), S->getRParenLoc());
}

Stmt *ASTTranslate::VisitForStmt(ForStmt *S) {
  return new (Ctx) ForStmt(Ctx, Clone(S->getInit()), Clone(S->getCond()),
      CloneDecl(S->getConditionVariable()), Clone(S->getInc()),
      Clone(S->getBody()), S->getForLoc(), S->getLParenLoc(),
      S->getRParenLoc());
}

Stmt *ASTTranslate::VisitGotoStmt(GotoStmt *S) {
  return new (Ctx) GotoStmt(S->getLabel(), S->getGotoLoc(), S->getLabelLoc());
}

Stmt *ASTTranslate::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  return new (Ctx) IndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
      Clone(S->getTarget()));
}

Stmt *ASTTranslate::VisitContinueStmt(ContinueStmt *S) {
  return new (Ctx) ContinueStmt(S->getContinueLoc());
}

Stmt *ASTTranslate::VisitBreakStmt(BreakStmt *S) {
  return new (Ctx) BreakStmt(S->getBreakLoc());
}

#ifdef NO_TRANSLATION
Stmt *ASTTranslate::VisitReturnStmt(ReturnStmt *S) {
  return new (Ctx) ReturnStmt(S->getReturnLoc(), Clone(S->getRetValue()), 0);
}
#endif

Stmt *ASTTranslate::VisitDeclStmt(DeclStmt *S) {
  DeclGroupRef clonedDecls;

  if (S->isSingleDecl()) {
    clonedDecls = DeclGroupRef(CloneDecl(S->getSingleDecl()));
  } else if (S->getDeclGroup().isDeclGroup()) {
    SmallVector<Decl *, 16> clonedDeclGroup;
    const DeclGroupRef& DG = S->getDeclGroup();

    for (DeclGroupRef::const_iterator I=DG.begin(), E=DG.end(); I!=E; ++I) {
      clonedDeclGroup.push_back(CloneDecl(*I));
    }

    clonedDecls = DeclGroupRef(DeclGroup::Create(Ctx,
          clonedDeclGroup.data(), clonedDeclGroup.size()));
  }

  return new (Ctx) DeclStmt(clonedDecls, S->getStartLoc(), S->getEndLoc());
}

Stmt *ASTTranslate::VisitSwitchCase(SwitchCase *S) {
  HIPACC_BASE_CLASS(SwitchCase);
  return NULL;
}

Stmt *ASTTranslate::VisitCaseStmt(CaseStmt *S) {
  CaseStmt *result = new (Ctx) CaseStmt(Clone(S->getLHS()), Clone(S->getRHS()),
      S->getCaseLoc(), S->getEllipsisLoc(), S->getColonLoc());

  result->setSubStmt(Clone(S->getSubStmt()));

  return result;
}

Stmt *ASTTranslate::VisitDefaultStmt(DefaultStmt *S) {
  return new (Ctx) DefaultStmt(S->getDefaultLoc(), S->getColonLoc(),
      Clone(S->getSubStmt()));
}

Stmt *ASTTranslate::VisitCapturedStmt(CapturedStmt *S) {
  SmallVector<CapturedStmt::Capture, 16> Captures;
  SmallVector<Expr *, 16> CaptureInits;

  // Capture inits
  for (CapturedStmt::capture_init_iterator I = S->capture_init_begin(), E =
      S->capture_init_end(); I != E; ++I) {
    CaptureInits.push_back(Clone(*I));
  }

  // Captures
  for (CapturedStmt::capture_iterator I = S->capture_begin(), E =
      S->capture_end(); I != E; ++I) {
    Captures.push_back(CapturedStmt::Capture(I->getLocation(),
          I->getCaptureKind(), CloneDecl(I->getCapturedVar())));
  }

  return CapturedStmt::Create(Ctx, Clone(S->getCapturedStmt()),
      S->getCapturedRegionKind(), llvm::makeArrayRef(Captures.data(),
        Captures.size()), llvm::makeArrayRef(CaptureInits.data(),
        CaptureInits.size()), S->getCapturedDecl(), (RecordDecl
        *)S->getCapturedRecordDecl());
}


// Asm Statements
Stmt *ASTTranslate::VisitGCCAsmStmt(GCCAsmStmt *S) {
  SmallVector<IdentifierInfo *, 16> names;
  SmallVector<StringLiteral *, 16> constraints;
  SmallVector<Expr *, 16> exprs;

  // outputs
  for (unsigned int I=0, N=S->getNumOutputs(); I!=N; ++I) {
    names.push_back(S->getOutputIdentifier(I));
    constraints.push_back(S->getOutputConstraintLiteral(I));
    exprs.push_back(Clone(S->getOutputExpr(I)));
  }

  // inputs
  for (unsigned int I=0, N=S->getNumInputs(); I!=N; ++I) {
    names.push_back(S->getInputIdentifier(I));
    constraints.push_back(S->getInputConstraintLiteral(I));
    exprs.push_back(Clone(S->getInputExpr(I)));
  }

  // constraints
  SmallVector<StringLiteral *, 16> clobbers;
  for (unsigned int I=0, N=S->getNumClobbers(); I!=N; ++I) {
    clobbers.push_back(S->getClobberStringLiteral(I));
  }

  return new (Ctx) GCCAsmStmt(Ctx, S->getAsmLoc(), S->isSimple(),
      S->isVolatile(), S->getNumOutputs(), S->getNumInputs(), names.data(),
      constraints.data(), exprs.data(), S->getAsmString(), S->getNumClobbers(),
      clobbers.data(), S->getRParenLoc());
}


// C++ statements
Stmt *ASTTranslate::VisitCXXCatchStmt(CXXCatchStmt *S) {
  return new (Ctx) CXXCatchStmt(S->getCatchLoc(),
      static_cast<VarDecl*>(CloneDecl(S->getExceptionDecl())),
      Clone(S->getHandlerBlock()));
}

Stmt *ASTTranslate::VisitCXXTryStmt(CXXTryStmt *S) {
  SmallVector<Stmt *, 16> handlers;

  for (unsigned int I=0, N=S->getNumHandlers(); I!=N; ++I) {
    handlers.push_back((Stmt *)Clone(S->getHandler(I)));
  }

  return CXXTryStmt::Create(Ctx, S->getTryLoc(), Clone(S->getTryBlock()),
      llvm::makeArrayRef(handlers.data(), handlers.size()));
}

Stmt *ASTTranslate::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  HIPACC_NOT_SUPPORTED(CXXForRangeStmt);
  return NULL;
}


// Expressions
Expr *ASTTranslate::VisitExpr(Expr *E) {
  HIPACC_BASE_CLASS(Expr);
  return NULL;
}

Expr *ASTTranslate::VisitPredefinedExpr(PredefinedExpr *E) {
  Expr *result = new (Ctx) PredefinedExpr(E->getLocation(), E->getType(),
      E->getIdentType());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitDeclRefExpr(DeclRefExpr *E) {
  TemplateArgumentListInfo templateArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned int I=0, N=E->getNumTemplateArgs(); I!=N; ++I) {
    templateArgs.addArgument(E->getTemplateArgs()[I]);
  }

  ValueDecl *VD = CloneDecl(E->getDecl());

  DeclRefExpr *result = DeclRefExpr::Create(Ctx, E->getQualifierLoc(),
      E->getTemplateKeywordLoc(), VD, E->refersToEnclosingLocal(),
      E->getLocation(), VD->getType(), E->getValueKind(), E->getFoundDecl(),
      E->getNumTemplateArgs()?&templateArgs:0);

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitIntegerLiteral(IntegerLiteral *E) {
  Expr *result = IntegerLiteral::Create(Ctx, E->getValue(), E->getType(),
      E->getLocation());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitFloatingLiteral(FloatingLiteral *E) {
  Expr *result = FloatingLiteral::Create(Ctx, E->getValue(), E->isExact(),
      E->getType(), E->getLocation());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  Expr *result = new (Ctx) ImaginaryLiteral(Clone(E->getSubExpr()),
      E->getType());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitStringLiteral(StringLiteral *E) {
  SmallVector<SourceLocation, 16> concatLocations(E->tokloc_begin(),
      E->tokloc_end());

  Expr *result = StringLiteral::Create(Ctx, E->getString(), E->getKind(),
      E->isPascal(), E->getType(), concatLocations.data(),
      concatLocations.size());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCharacterLiteral(CharacterLiteral *E) {
  Expr *result = new (Ctx) CharacterLiteral(E->getValue(), E->getKind(),
      E->getType(), E->getLocation());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitParenExpr(ParenExpr *E) {
  Expr *result = new (Ctx) ParenExpr(E->getLParen(), E->getRParen(),
      Clone(E->getSubExpr()));

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitUnaryOperator(UnaryOperator *E) {
  Expr *result = new (Ctx) UnaryOperator(Clone(E->getSubExpr()), E->getOpcode(),
      E->getType(), E->getValueKind(), E->getObjectKind(), E->getOperatorLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitOffsetOfExpr(OffsetOfExpr *E) {
  OffsetOfExpr *result = OffsetOfExpr::CreateEmpty(Ctx, E->getNumExpressions(),
      E->getNumComponents());

  result->setOperatorLoc(E->getOperatorLoc());
  result->setTypeSourceInfo(E->getTypeSourceInfo());
  result->setRParenLoc(E->getRParenLoc());

  for (unsigned int I=0, N=E->getNumComponents(); I!=N; ++I) {
    result->setComponent(I, E->getComponent(I));
  }

  for (unsigned int I=0, N=E->getNumExpressions(); I!=N; ++I) {
    result->setIndexExpr(I, Clone(E->getIndexExpr(I)));
  }

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  Expr *result;

  if (E->isArgumentType()) {
    result = new (Ctx) UnaryExprOrTypeTraitExpr(E->getKind(),
        E->getArgumentTypeInfo(), E->getType(), E->getOperatorLoc(),
        E->getRParenLoc());
  } else {
    result = new (Ctx) UnaryExprOrTypeTraitExpr(E->getKind(),
        Clone(E->getArgumentExpr()), E->getType(), E->getOperatorLoc(),
        E->getRParenLoc());
  }

  setExprPropsClone(E, result);

  return result;
}


Expr *ASTTranslate::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  Expr *result = new (Ctx) ArraySubscriptExpr(Clone(E->getLHS()),
      Clone(E->getRHS()), E->getType(), E->getValueKind(), E->getObjectKind(),
      E->getRBracketLoc());

  setExprPropsClone(E, result);

  return result;
}

#ifdef NO_TRANSLATION
Expr *ASTTranslate::VisitCallExpr(CallExpr *E) {
  SmallVector<Expr *, 16> args;

  for (unsigned int I=0, N=E->getNumArgs(); I!=N; ++I) {
    args.push_back(Clone(E->getArg(I)));
  }

  CallExpr *result = new (Ctx) CallExpr(Ctx, Clone(E->getCallee()),
      llvm::makeArrayRef(args.data(), args.size()), E->getType(),
      E->getValueKind(), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitMemberExpr(MemberExpr *E) {
  MemberExpr *result = new (Ctx) MemberExpr(Clone(E->getBase()), E->isArrow(),
      CloneDecl(E->getMemberDecl()), E->getMemberNameInfo(), E->getType(),
      E->getValueKind(), E->getObjectKind());

  setExprPropsClone(E, result);

  return result;
}
#endif

Expr *ASTTranslate::VisitCastExpr(CastExpr *E) {
  HIPACC_BASE_CLASS(CastExpr);
  return NULL;
}

#ifdef NO_TRANSLATION
Expr *ASTTranslate::VisitBinaryOperator(BinaryOperator *E) {
  BinaryOperator *result = new (Ctx) BinaryOperator(Clone(E->getLHS()),
      Clone(E->getRHS()), E->getOpcode(), E->getType(), E->getValueKind(),
      E->getObjectKind(), E->getOperatorLoc(), E->isFPContractable());

  setExprPropsClone(E, result);

  return result;
}
#endif

Expr *ASTTranslate::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  Expr *result = new (Ctx) CompoundAssignOperator(Clone(E->getLHS()),
      Clone(E->getRHS()), E->getOpcode(), E->getType(), E->getValueKind(),
      E->getObjectKind(), E->getComputationLHSType(),
      E->getComputationResultType(), E->getOperatorLoc(),
      E->isFPContractable());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitAbstractConditionalOperator(AbstractConditionalOperator *E) {
  HIPACC_NOT_SUPPORTED(AbstractConditionalOperator);
  return NULL;
}

Expr *ASTTranslate::VisitConditionalOperator(ConditionalOperator *E) {
  Expr *result = new (Ctx) ConditionalOperator(Clone(E->getCond()),
      E->getQuestionLoc(), Clone(E->getLHS()), E->getColonLoc(),
      Clone(E->getRHS()), E->getType(), E->getValueKind(), E->getObjectKind());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitBinaryConditionalOperator(BinaryConditionalOperator *E)
{
  Expr *result = new (Ctx) BinaryConditionalOperator(Clone(E->getCommon()),
      Clone(E->getOpaqueValue()), Clone(E->getCond()), Clone(E->getTrueExpr()),
      Clone(E->getFalseExpr()), E->getQuestionLoc(), E->getColonLoc(),
      E->getType(), E->getValueKind(), E->getObjectKind());

  setExprPropsClone(E, result);

  return result;
}

#ifdef NO_TRANSLATION
Expr *ASTTranslate::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  ImplicitCastExpr *result = ImplicitCastExpr::Create(Ctx, E->getType(),
      E->getCastKind(), Clone(E->getSubExpr()), &castPath, E->getValueKind());

  setExprPropsClone(E, result);

  return result;
}
#endif

Expr *ASTTranslate::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  HIPACC_BASE_CLASS(ExplicitCastExpr);
  return NULL;
}

#ifdef NO_TRANSLATION
Expr *ASTTranslate::VisitCStyleCastExpr(CStyleCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  CStyleCastExpr *result = CStyleCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()), &castPath,
      E->getTypeInfoAsWritten(), E->getLParenLoc(), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}
#endif

Expr *ASTTranslate::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  Expr *result = new (Ctx) CompoundLiteralExpr(E->getLParenLoc(),
      E->getTypeSourceInfo(), E->getType(), E->getValueKind(),
      Clone(E->getInitializer()), E->isFileScope());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  Expr *result = new (Ctx) ExtVectorElementExpr(E->getType(), E->getValueKind(),
      Clone(E->getBase()), E->getAccessor(), E->getAccessorLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitInitListExpr(InitListExpr *E) {
  SmallVector<Expr *, 16> initExprs;

  for (unsigned int I=0, N=E->getNumInits(); I!=N; ++I) {
    initExprs.push_back(Clone(E->getInit(I)));
  }

  InitListExpr *result = new (Ctx) InitListExpr(Ctx, E->getLBraceLoc(),
      llvm::makeArrayRef(initExprs.data(), initExprs.size()),
      E->getRBraceLoc());

  result->setInitializedFieldInUnion(E->getInitializedFieldInUnion());
  if (E->hasArrayFiller()) result->setArrayFiller(Clone(E->getArrayFiller()));

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  SmallVector<Expr *, 16> indexExprs;

  unsigned numIndexExprs = E->getNumSubExprs() - 1;
  for (unsigned int I=0 ; I<numIndexExprs; ++I) {
    indexExprs.push_back(Clone(E->getSubExpr(I+1)));
  }

  Expr *result = DesignatedInitExpr::Create(Ctx, E->getDesignator(0), E->size(),
      indexExprs, E->getEqualOrColonLoc(), E->usesGNUSyntax(),
      Clone(E->getInit()));

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  Expr *result = new (Ctx) ImplicitValueInitExpr(E->getType());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitParenListExpr(ParenListExpr *E) {
  SmallVector<Expr *, 16> Exprs;

  for (unsigned int I=0, N=E->getNumExprs(); I!=N; ++I) {
    Exprs.push_back(Clone(E->getExpr(I)));
  }

  Expr *result = new (Ctx) ParenListExpr(Ctx, E->getLParenLoc(),
      llvm::makeArrayRef(Exprs.data(), Exprs.size()), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitVAArgExpr(VAArgExpr *E) {
  Expr *result = new (Ctx) VAArgExpr(E->getBuiltinLoc(), Clone(E->getSubExpr()),
      E->getWrittenTypeInfo(), E->getRParenLoc(), E->getType());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitGenericSelectionExpr(GenericSelectionExpr *E) {
  HIPACC_NOT_SUPPORTED(GenericSelectionExpr);
  return NULL;
}

Expr *ASTTranslate::VisitPseudoObjectExpr(PseudoObjectExpr *E) {
  HIPACC_NOT_SUPPORTED(PseudoObjectExpr);
  return NULL;
}


// Atomic Expressions
Expr *ASTTranslate::VisitAtomicExpr(AtomicExpr *E) {
  SmallVector<Expr *, 16> Args;

  for (unsigned int I=0, N=E->getNumSubExprs(); I!=N; ++I) {
    Args.push_back(Clone(E->getSubExprs()[I]));
  }

  Expr *result = new (Ctx) AtomicExpr(E->getBuiltinLoc(), Args, E->getType(),
      E->getOp(), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}


// GNU Extensions
Expr *ASTTranslate::VisitAddrLabelExpr(AddrLabelExpr *E) {
  Expr *result = new (Ctx) AddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
      E->getLabel(), E->getType());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitStmtExpr(StmtExpr *E) {
  Expr *result = new (Ctx) StmtExpr(Clone(E->getSubStmt()), E->getType(),
      E->getLParenLoc(), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitChooseExpr(ChooseExpr *E) {
  Expr *result = new (Ctx) ChooseExpr(E->getBuiltinLoc(), Clone(E->getCond()),
      Clone(E->getLHS()), Clone(E->getRHS()), E->getType(), E->getValueKind(),
      E->getObjectKind(), E->getRParenLoc(), E->isTypeDependent(),
      E->isValueDependent());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitGNUNullExpr(GNUNullExpr *E) {
  Expr *result = new (Ctx) GNUNullExpr(E->getType(), E->getTokenLocation());

  setExprPropsClone(E, result);

  return result;
}


// C++ Expressions
#ifdef NO_TRANSLATION
Expr *ASTTranslate::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  SmallVector<Expr *, 16> args;

  for (unsigned int I=0, N=E->getNumArgs(); I!=N; ++I) {
    args.push_back(Clone(E->getArg(I)));
  }

  CXXOperatorCallExpr *result = new (Ctx) CXXOperatorCallExpr(Ctx,
      E->getOperator(), Clone(E->getCallee()), llvm::makeArrayRef(args.data(),
        args.size()), E->getType(), E->getValueKind(), E->getRParenLoc(),
      E->isFPContractable());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  CXXMemberCallExpr *result = new (Ctx) CXXMemberCallExpr(Ctx,
      Clone(E->getCallee()), MultiExprArg(), E->getType(), E->getValueKind(),
      E->getRParenLoc());

  result->setNumArgs(Ctx, E->getNumArgs());

  for (unsigned int I=0, N=E->getNumArgs(); I!=N; ++I) {
    result->setArg(I, Clone(E->getArg(I)));
  }

  setExprPropsClone(E, result);

  return result;
}
#endif

Expr *ASTTranslate::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNamedCastExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  CXXStaticCastExpr *result = CXXStaticCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()), &castPath,
      E->getTypeInfoAsWritten(), E->getOperatorLoc(), E->getRParenLoc(),
      E->getAngleBrackets());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  CXXDynamicCastExpr *result = CXXDynamicCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()), &castPath,
      E->getTypeInfoAsWritten(), E->getOperatorLoc(), E->getRParenLoc(),
      E->getAngleBrackets());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  CXXReinterpretCastExpr *result = CXXReinterpretCastExpr::Create(Ctx,
      E->getType(), E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()),
      &castPath, E->getTypeInfoAsWritten(), E->getOperatorLoc(),
      E->getRParenLoc(), E->getAngleBrackets());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  CXXConstCastExpr *result = CXXConstCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), Clone(E->getSubExpr()), E->getTypeInfoAsWritten(),
      E->getOperatorLoc(), E->getRParenLoc(), E->getAngleBrackets());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  CXXCastPath castPath;
  setCastPath(E, castPath);

  CXXFunctionalCastExpr *result = CXXFunctionalCastExpr::Create(Ctx,
      E->getType(), E->getValueKind(), E->getTypeInfoAsWritten(),
      E->getTypeBeginLoc(), E->getCastKind(), Clone(E->getSubExpr()), &castPath,
      E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXTypeidExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUserDefinedLiteral(UserDefinedLiteral *E) {
  HIPACC_NOT_SUPPORTED(UserDefinedLiteral);
  return NULL;
}

Expr *ASTTranslate::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  Expr *result = new (Ctx) CXXBoolLiteralExpr(E->getValue(), E->getType(),
      E->getSourceRange().getBegin());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  Expr *result = new (Ctx) CXXNullPtrLiteralExpr(E->getType(),
      E->getSourceRange().getBegin());

  setExprPropsClone(E, result);

  return result;
}


Expr *ASTTranslate::VisitCXXThisExpr(CXXThisExpr *E) {
  Expr *result = new (Ctx) CXXThisExpr(E->getSourceRange().getBegin(),
      E->getType(), E->isImplicit());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXThrowExpr(CXXThrowExpr *E) {
  Expr *result = new (Ctx) CXXThrowExpr(Clone(E->getSubExpr()), E->getType(),
      E->getThrowLoc(), E->isThrownVariableInScope());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXDefaultArgExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXDefaultInitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXScalarValueInitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXNewExpr(CXXNewExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNewExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXDeleteExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXPseudoDestructorExpr);
  return NULL;
}

Expr *ASTTranslate::VisitTypeTraitExpr(TypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(TypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(UnaryTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(BinaryTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(ArrayTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(ExpressionTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E)
{
  HIPACC_NOT_SUPPORTED(DependentScopeDeclRefExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXConstructExpr(CXXConstructExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXConstructExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXBindTemporaryExpr);
  return NULL;
}

Expr *ASTTranslate::VisitExprWithCleanups(ExprWithCleanups *E) {
  HIPACC_NOT_SUPPORTED(ExprWithCleanups);
  return NULL;
}

Expr *ASTTranslate::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXTemporaryObjectExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr
    *E) {
  HIPACC_NOT_SUPPORTED(CXXUnresolvedConstructExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr
    *E) {
  HIPACC_NOT_SUPPORTED(CXXDependentScopeMemberExpr);
  return NULL;
}

Expr *ASTTranslate::VisitOverloadExpr(OverloadExpr *E) {
  HIPACC_NOT_SUPPORTED(OverloadExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  HIPACC_NOT_SUPPORTED(UnresolvedLookupExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  HIPACC_NOT_SUPPORTED(UnresolvedMemberExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNoexceptExpr);
  return NULL;
}

Expr *ASTTranslate::VisitPackExpansionExpr(PackExpansionExpr *E) {
  HIPACC_NOT_SUPPORTED(PackExpansionExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  HIPACC_NOT_SUPPORTED(SizeOfPackExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *E) {
  HIPACC_NOT_SUPPORTED(SubstNonTypeTemplateParmExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSubstNonTypeTemplateParmPackExpr(
    SubstNonTypeTemplateParmPackExpr *E) {
  HIPACC_NOT_SUPPORTED(SubstNonTypeTemplateParmPackExpr);
  return NULL;
}

Expr *ASTTranslate::VisitFunctionParmPackExpr(FunctionParmPackExpr *E) {
  HIPACC_NOT_SUPPORTED(FunctionParmPackExpr);
  return NULL;
}

Expr *ASTTranslate::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  MaterializeTemporaryExpr *result = new (Ctx) MaterializeTemporaryExpr(E->getType(),
      Clone(E->GetTemporaryExpr()), E->isBoundToLvalueReference());

  setExprPropsClone(E, result);

  return result;
}

Expr *ASTTranslate::VisitLambdaExpr(LambdaExpr *E) {
  SmallVector<LambdaExpr::Capture, 16> captures;
  SmallVector<Expr *, 16> captureInits;

  SmallVector<VarDecl *, 4> arrayIndexVars;
  SmallVector<unsigned int, 4> arrayIndexStarts;

  // Captures
  for (LambdaExpr::capture_iterator CI=E->capture_begin(), CE=E->capture_end();
      CI!=CE; ++CI) {
    captures.push_back(*CI);
  }
  // CaptureInits
  CXXRecordDecl::field_iterator curField = E->getLambdaClass()->field_begin();
  for (LambdaExpr::capture_init_iterator CI=E->capture_init_begin(),
      CE=E->capture_init_end(); CI!=CE; ++CI, ++curField) {
    captureInits.push_back(*CI);

    // ArrayIndex[Vars|Starts]
    if (curField->getType()->isArrayType()) {
      arrayIndexStarts.push_back(arrayIndexVars.size());

      ArrayRef<VarDecl *> curArrayIndexVars = E->getCaptureInitIndexVars(CI);
      unsigned int num_var = 0;
      QualType BaseType = curField->getType();
      while (const ConstantArrayType *Array =
          Ctx.getAsConstantArrayType(BaseType)) {
        arrayIndexVars.push_back(curArrayIndexVars[num_var]);
        BaseType = Array->getElementType();
        num_var++;
      }
    }
  }

  LambdaExpr *result = LambdaExpr::Create(Ctx, E->getLambdaClass(),
      E->getIntroducerRange(), E->getCaptureDefault(), captures,
      E->hasExplicitParameters(), E->hasExplicitResultType(), captureInits,
      arrayIndexVars, arrayIndexStarts, E->getBody()->getLocEnd(),
      E->containsUnexpandedParameterPack());

  setExprPropsClone(E, result);

  return result;
}


// CUDA Expressions
Expr *ASTTranslate::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  SmallVector<Expr *, 16> Args;

  for (unsigned int I=0, N=E->getNumArgs(); I!=N; ++I) {
    Args.push_back(Clone(E->getArg(I)));
  }

  Expr *result = new (Ctx) CUDAKernelCallExpr(Ctx, Clone(E->getCallee()),
      Clone(E->getConfig()), Args, E->getType(), E->getValueKind(),
      E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}


// Clang Extensions
Expr *ASTTranslate::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  SmallVector<Expr *, 16> body;

  for (unsigned int I=0, N=E->getNumSubExprs(); I!=N; ++I) {
    body.push_back(Clone(E->getExpr(I)));
  }

  Expr *result = new (Ctx) ShuffleVectorExpr(Ctx, body, E->getType(),
      E->getBuiltinLoc(), E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}


// OpenCL Expressions
Expr *ASTTranslate::VisitAsTypeExpr(AsTypeExpr *E) {
  Expr *result = new (Ctx) AsTypeExpr(Clone(E->getSrcExpr()), E->getType(),
      E->getValueKind(), E->getObjectKind(), E->getBuiltinLoc(),
      E->getRParenLoc());

  setExprPropsClone(E, result);

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

