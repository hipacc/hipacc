//
// Copyright (c) 2013, University of Erlangen-Nuremberg
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

//===--- Convolution.cpp - Add Interpolation Calls to the AST -------------===//
//
// This file implements the translation of lambda-functions for local operators.
//
//===----------------------------------------------------------------------===//

// includes for FLT_MAX, INT_MAX, etc.
#include <limits.h>
#include <float.h>

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;


// create expression for convolutions
Stmt *ASTTranslate::getConvolutionStmt(ConvolutionMode mode, DeclRefExpr
    *tmp_var, Expr *ret_val) {
  Stmt *result = NULL;
  FunctionDecl *fun;
  SmallVector<Expr *, 16> funArgs;

  switch (mode) {
    case HipaccSUM:
      // red += val;
      result = createCompoundAssignOperator(Ctx, tmp_var, ret_val, BO_AddAssign,
          tmp_var->getType());
      break;
    case HipaccMIN:
      // red = min(red, val);
      fun = lookup<FunctionDecl>(std::string("min"), tmp_var->getType(),
          hipaccMathNS);
      assert(fun && "could not lookup 'min'");
      funArgs.push_back(createImplicitCastExpr(Ctx, tmp_var->getType(),
            CK_LValueToRValue, tmp_var, NULL, VK_RValue));
      funArgs.push_back(ret_val);
      result = createBinaryOperator(Ctx, tmp_var, createFunctionCall(Ctx, fun,
            funArgs), BO_Assign, tmp_var->getType());
      break;
    case HipaccMAX:
      // red = max(red, val);
      fun = lookup<FunctionDecl>(std::string("max"), tmp_var->getType(),
          hipaccMathNS);
      assert(fun && "could not lookup 'max'");
      funArgs.push_back(createImplicitCastExpr(Ctx, tmp_var->getType(),
            CK_LValueToRValue, tmp_var, NULL, VK_RValue));
      funArgs.push_back(ret_val);
      result = createBinaryOperator(Ctx, tmp_var, createFunctionCall(Ctx, fun,
            funArgs), BO_Assign, tmp_var->getType());
      break;
    case HipaccPROD:
      // red *= val;
      result = createCompoundAssignOperator(Ctx, tmp_var, ret_val, BO_MulAssign,
          tmp_var->getType());
      break;
    case HipaccMEDIAN:
      assert(0 && "Unsupported convolution mode.");
  }

  return result;
}


// create init expression for given aggregation mode and type
Expr *ASTTranslate::getInitExpr(ConvolutionMode mode, QualType QT) {
  Expr *result = NULL, *initExpr = NULL;

  QualType EQT = QT;
  bool isVecType = QT->isVectorType();

  if (isVecType) {
    EQT = QT->getAs<VectorType>()->getElementType();
  }
  const BuiltinType *BT = EQT->getAs<BuiltinType>();

  assert(mode!=HipaccMEDIAN && "Median currently not supported.");

  switch (BT->getKind()) {
    case BuiltinType::WChar_U:
    case BuiltinType::WChar_S:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::LongDouble:
    case BuiltinType::Void:
    case BuiltinType::Bool:
    default:
      assert(0 && "BuiltinType for reduce function not supported.");

    #define GET_INIT_CONSTANT(MODE, SUM, MIN, MAX, PROD) \
      (MODE == HipaccSUM ? (SUM) : \
        (MODE == HipaccMIN ? (MIN) : \
          (MODE == HipaccMAX ? (MAX) : (PROD) )))

    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      initExpr = new (Ctx) CharacterLiteral(GET_INIT_CONSTANT(mode, 0,
            SCHAR_MAX, SCHAR_MIN, 1), CharacterLiteral::Ascii, QT,
          SourceLocation());
      break;
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      initExpr = new (Ctx) CharacterLiteral(GET_INIT_CONSTANT(mode, 0,
            UCHAR_MAX, 0, 1), CharacterLiteral::Ascii, QT, SourceLocation());
      break;
    case BuiltinType::Short: {
      llvm::APInt init(16, GET_INIT_CONSTANT(mode, 0, SHRT_MAX, SHRT_MIN, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Char16:
    case BuiltinType::UShort: {
      llvm::APInt init(16, GET_INIT_CONSTANT(mode, 0, USHRT_MAX, 0, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Int: {
      llvm::APInt init(32, GET_INIT_CONSTANT(mode, 0, INT_MAX, INT_MIN, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Char32:
    case BuiltinType::UInt: {
      llvm::APInt init(32, GET_INIT_CONSTANT(mode, 0, UINT_MAX, 0, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Long: {
      llvm::APInt init(64, GET_INIT_CONSTANT(mode, 0, LONG_MAX, LONG_MIN, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::ULong: {
      llvm::APInt init(64, GET_INIT_CONSTANT(mode, 0, ULONG_MAX, 0, 1));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Float: {
      llvm::APFloat init(GET_INIT_CONSTANT(mode, 0, FLT_MAX, FLT_MIN, 1));
      initExpr = FloatingLiteral::Create(Ctx, init, false, EQT, SourceLocation());
      break; }
    case BuiltinType::Double: {
      llvm::APFloat init(GET_INIT_CONSTANT(mode, 0, DBL_MAX, DBL_MIN, 1));
      initExpr = FloatingLiteral::Create(Ctx, init, false, EQT, SourceLocation());
      break; }
    #undef GET_INIT_CONSTANT
  }

  if (isVecType) {
    SmallVector<Expr *, 16> initExprs;
    size_t lanes = QT->getAs<VectorType>()->getNumElements();

    for (size_t I=0, N=lanes; I!=N; ++I) {
      initExprs.push_back(initExpr);
    }

    result = new (Ctx) InitListExpr(Ctx, SourceLocation(),
        llvm::makeArrayRef(initExprs.data(), initExprs.size()),
        SourceLocation());
    result->setType(QT);
  } else {
    result = initExpr;
  }

  return result;
}


// check if the current index of the domain space should be processed
Stmt *ASTTranslate::addDomainCheck(HipaccMask *Domain, DeclRefExpr *domain_var,
    Stmt *stmt) {
  assert(domain_var && "Domain.");

  Expr *dom_acc = NULL;
  switch (compilerOptions.getTargetCode()) {
    case TARGET_C:
    case TARGET_CUDA:
      // array subscript: Domain[y][x]
      dom_acc = accessMem2DAt(domain_var, createIntegerLiteral(Ctx,
            redIdxX.back()), createIntegerLiteral(Ctx, redIdxY.back()));
      break;
    case TARGET_OpenCLACC:
    case TARGET_OpenCLCPU:
    case TARGET_OpenCLGPU:
      // array subscript: Domain[y*width + x]
      dom_acc = accessMemArrAt(domain_var, createIntegerLiteral(Ctx,
            (int)Domain->getSizeX()), createIntegerLiteral(Ctx, redIdxX.back()),
          createIntegerLiteral(Ctx, redIdxY.back()));
      break;
    case TARGET_Renderscript:
    case TARGET_Filterscript:
      // allocation access: rsGetElementAt(Domain, x, y)
      dom_acc = accessMemAllocAt(domain_var, READ_ONLY, createIntegerLiteral(Ctx,
            redIdxX.back()), createIntegerLiteral(Ctx, redIdxY.back()));
      break;
  }
  // if (dom(x, y) > 0)
  BinaryOperator *check_dom = createBinaryOperator(Ctx, dom_acc, new (Ctx)
      CharacterLiteral(0, CharacterLiteral::Ascii, Ctx.UnsignedCharTy,
        SourceLocation()), BO_GT, Ctx.BoolTy);

  return createIfStmt(Ctx, check_dom, stmt);
}


// check if we have a convolve/reduce/iterate method and convert it
Expr *ASTTranslate::convertConvolution(CXXMemberCallExpr *E) {
  // check if this is a convolve function call
  ConvolveMethod method = Convolve;
  if (E->getDirectCallee()->getName().equals("convolve")) {
    method = Convolve;
    assert(convMask == NULL && "Nested convolution calls are not supported.");
  } else if (E->getDirectCallee()->getName().equals("reduce")) {
    method = Reduce;
  } else if (E->getDirectCallee()->getName().equals("iterate")) {
    method = Iterate;
  } else {
    assert(false && "unsupported convolution method.");
  }

  switch (method) {
    case Convolve:
      // convolve(mask, mode, [&] () { lambda-function; });
      assert(E->getNumArgs() == 3 && "Expected 3 arguments to 'convolve' call.");
      break;
    case Reduce:
      // reduce(domain, mode, [&] () { lambda-function; });
      assert(E->getNumArgs() == 3 && "Expected 3 arguments to 'reduce' call.");
      break;
    case Iterate:
      // reduce(domain, [&] () { lambda-function; });
      assert(E->getNumArgs() == 2 && "Expected 2 arguments to 'iterate' call.");
      break;
  }

  // first parameter: Mask<type> or Domain reference
  HipaccMask *Mask = NULL;
  if (method==Convolve)
    assert(isa<MemberExpr>(E->getArg(0)->IgnoreImpCasts()) &&
        isa<FieldDecl>(dyn_cast<MemberExpr>(
            E->getArg(0)->IgnoreImpCasts())->getMemberDecl()) &&
           "First parameter to 'convolve' call must be a Mask.");
  else
    assert(isa<MemberExpr>(E->getArg(0)->IgnoreImpCasts()) &&
        isa<FieldDecl>(dyn_cast<MemberExpr>(
            E->getArg(0)->IgnoreImpCasts())->getMemberDecl()) &&
           "First parameter to 'reduce'/'iterate' call must be a Domain.");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0)->IgnoreImpCasts());
  FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

  // look for Mask/Domain user class member variable
  if (Kernel->getMaskFromMapping(FD)) {
    Mask = Kernel->getMaskFromMapping(FD);
  }
  switch (method) {
    case Convolve:
      assert(Mask && !Mask->isDomain() &&
          "Could not find Mask Field Decl.");
      break;
    case Reduce:
    case Iterate:
      assert(Mask && Mask->isDomain() &&
          "Could not find Domain Field Decl.");
      break;
  }

  // second parameter: convolution/reduction mode
  if (method==Convolve || method==Reduce) {
    if (method==Convolve)
      assert(isa<DeclRefExpr>(E->getArg(1)) &&
          "Second parameter to 'convolve' call must be the convolution mode.");
    if (method==Reduce)
      assert(isa<DeclRefExpr>(E->getArg(1)) &&
          "Second parameter to 'reduce' call must be the reduction mode.");
    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getArg(1));

    if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
        DRE->getDecl()->getType().getAsString() ==
        "enum hipacc::HipaccConvolutionMode") {
      int64_t mode = E->getArg(1)->EvaluateKnownConstInt(Ctx).getSExtValue();
      switch (mode) {
        case HipaccSUM:
          if (method==Convolve) convMode = HipaccSUM;
          else redModes.push_back(HipaccSUM);
          break;
        case HipaccMIN:
          if (method==Convolve) convMode = HipaccMIN;
          else redModes.push_back(HipaccMIN);
          break;
        case HipaccMAX:
          if (method==Convolve) convMode = HipaccMAX;
          else redModes.push_back(HipaccMAX);
          break;
        case HipaccPROD:
          if (method==Convolve) convMode = HipaccPROD;
          else redModes.push_back(HipaccPROD);
          break;
        case HipaccMEDIAN:
          if (method==Convolve) convMode = HipaccMEDIAN;
          else redModes.push_back(HipaccMEDIAN);
        default:
          unsigned int DiagIDConvMode =
            Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "%0 mode not supported, allowed modes are: "
                "HipaccSUM, HipaccMIN, HipaccMAX, and HipaccPROD.");
          Diags.Report(E->getArg(1)->getExprLoc(), DiagIDConvMode)
            << (const char *)(Mask->isDomain()?"reduction":"convolution");
          exit(EXIT_FAILURE);
      }
    } else {
      unsigned int DiagIDConvMode =
        Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unknown %0 mode detected.");
      Diags.Report(E->getArg(1)->getExprLoc(), DiagIDConvMode)
        << (const char *)(Mask->isDomain()?"reduction":"convolution");
      exit(EXIT_FAILURE);
    }
  }

  // third parameter: lambda-function
  int li = 2;
  if (method==Iterate) li = 1;

  assert(isa<MaterializeTemporaryExpr>(E->getArg(li)) &&
         isa<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(
             E->getArg(li))->GetTemporaryExpr()->IgnoreImpCasts()) &&
         "Third parameter to 'reduce' or 'iterate' call must be a"
         "lambda-function.");
  LambdaExpr *LE = dyn_cast<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(
                       E->getArg(li))->GetTemporaryExpr()->IgnoreImpCasts());

  // check default capture kind
  if (LE->getCaptureDefault()==LCD_ByCopy) {
    unsigned int DiagIDCapture =
      Diags.getCustomDiagID(DiagnosticsEngine::Error,
          "Capture by copy [=] is not supported for '%0' lambda-function. "
          "Use capture by reference [&] instead.");
    Diags.Report(LE->getCaptureDefaultLoc(), DiagIDCapture)
      << (const char *)(method==Convolve ? "convolve" : method==Reduce ?
          "reduce" : "iterate");
    exit(EXIT_FAILURE);
  }
  // check capture kind of variables
  for (LambdaExpr::capture_iterator II=LE->capture_begin(),
                                    EE=LE->capture_end(); II!=EE; ++II) {
    LambdaExpr::Capture cap = *II;

    if (cap.capturesVariable() && cap.getCaptureKind()!=LCK_ByRef) {
      unsigned int DiagIDCapture =
        Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unsupported capture kind for variable '%0' in '%1' "
            "lambda-function. Use capture by reference instead: [&%0].");
      Diags.Report(cap.getLocation(), DiagIDCapture)
        << cap.getCapturedVar()->getNameAsString()
        << (const char *)(method==Convolve ? "convolve" : method==Reduce ?
            "reduce" : "iterate");
      exit(EXIT_FAILURE);
    }
  }

  // introduce temporary for holding the convolution/reduction result
  CompoundStmt *outerCompountStmt = curCStmt;
  std::stringstream LSST;
  LSST << "_tmp" << literalCount++;
  Expr *init = NULL;
  if (method==Reduce) {
    // init temporary variable depending on aggregation mode
    init = getInitExpr(redModes.back(),
        LE->getCallOperator()->getResultType());
  }
  VarDecl *tmp_decl = createVarDecl(Ctx, kernelDecl, LSST.str(),
      LE->getCallOperator()->getResultType(), init);
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(tmp_decl);
  DeclRefExpr *tmp_dre = createDeclRefExpr(Ctx, tmp_decl);

  switch (method) {
    case Convolve:
      convTmp = tmp_dre;
      convMask = Mask;
      preStmts.push_back(createDeclStmt(Ctx, tmp_decl));
      preCStmt.push_back(outerCompountStmt);
      break;
    case Reduce:
      redTmps.push_back(tmp_dre);
      redDomains.push_back(Mask);
      preStmts.push_back(createDeclStmt(Ctx, tmp_decl));
      preCStmt.push_back(outerCompountStmt);
      break;
    case Iterate:
      redDomains.push_back(Mask);
      break;
  }

  // unroll Mask/Domain
  for (size_t y=0; y<Mask->getSizeY(); ++y) {
    for (size_t x=0; x<Mask->getSizeX(); ++x) {
      bool doIterate = true;

      if (Mask->isDomain() && Mask->isConstant() && Mask->getInitList()) {
        Expr *E = Mask->getInitList()
                      ->getInit(Mask->getSizeY() * x + y)
                      ->IgnoreParenCasts();
        if (isa<IntegerLiteral>(E) &&
            dyn_cast<IntegerLiteral>(E)->getValue() == 0) {
          doIterate = false;
        }
      }

      if (doIterate) {
        Stmt *iteration = NULL;
        switch (method) {
          case Convolve:
            convIdxX = x;
            convIdxY = y;
            iteration = Clone(LE->getBody());
            break;
          case Reduce:
          case Iterate:
            redIdxX.push_back(x);
            redIdxY.push_back(y);
            iteration = Clone(LE->getBody());
            // add check if this iteration point should be processed - the
            // DeclRefExpr for the Domain is retrieved when visiting the
            // MemberExpr
            if (!Mask->isConstant()) {
              iteration = addDomainCheck(Mask,
                  dyn_cast_or_null<DeclRefExpr>(VisitMemberExpr(ME)),
                  iteration);
            }
            redIdxX.pop_back();
            redIdxY.pop_back();
            break;
        }
        preStmts.push_back(iteration);
        preCStmt.push_back(outerCompountStmt);
        // clear decls added while cloning last iteration
        LambdaDeclMap.clear();
      }
    }
  }

  // reset global variables
  switch (method) {
    case Convolve:
      convMask = NULL;
      convTmp = NULL;
      convIdxX = convIdxY = 0;
      break;
    case Reduce:
      redDomains.pop_back();
      redModes.pop_back();
      redTmps.pop_back();
      break;
    case Iterate:
      redDomains.pop_back();
      redTmps.pop_back();
      break;
  }

  // result of convolution
  switch (method) {
    case Convolve:
    case Reduce:
      // add ICE for CodeGen
      return createImplicitCastExpr(Ctx, LE->getCallOperator()->getResultType(),
          CK_LValueToRValue, tmp_dre, NULL, VK_RValue);
    case Iterate:
      return NULL;
  }
}

// vim: set ts=2 sw=2 sts=2 et ai:

