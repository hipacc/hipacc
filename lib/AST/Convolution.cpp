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

#include "hipacc/AST/ASTTranslate.h"

// includes for numeric_limits
#include <limits>

using namespace clang;
using namespace hipacc;
using namespace ASTNode;


// create expression for convolutions
Stmt *ASTTranslate::getConvolutionStmt(Reduce mode, DeclRefExpr *tmp_var,
    Expr *ret_val) {
  Stmt *result = nullptr;
  FunctionDecl *fun;
  SmallVector<Expr *, 16> funArgs;

  switch (mode) {
    case Reduce::SUM:
      // red += val;
      result = createCompoundAssignOperator(Ctx, tmp_var, ret_val, BO_AddAssign,
          tmp_var->getType());
      break;
    case Reduce::MIN:
      // red = min(red, val);
      fun = lookup<FunctionDecl>(std::string("min"), tmp_var->getType(),
          hipacc_math_ns);
      hipacc_require(fun,"could not lookup 'min'");
      funArgs.push_back(createImplicitCastExpr(Ctx, tmp_var->getType(),
            CK_LValueToRValue, tmp_var, nullptr, VK_RValue));
      funArgs.push_back(ret_val);
      result = createBinaryOperator(Ctx, tmp_var, createFunctionCall(Ctx, fun,
            funArgs), BO_Assign, tmp_var->getType());
      break;
    case Reduce::MAX:
      // red = max(red, val);
      fun = lookup<FunctionDecl>(std::string("max"), tmp_var->getType(),
          hipacc_math_ns);
      hipacc_require(fun,"could not lookup 'max'");
      funArgs.push_back(createImplicitCastExpr(Ctx, tmp_var->getType(),
            CK_LValueToRValue, tmp_var, nullptr, VK_RValue));
      funArgs.push_back(ret_val);
      result = createBinaryOperator(Ctx, tmp_var, createFunctionCall(Ctx, fun,
            funArgs), BO_Assign, tmp_var->getType());
      break;
    case Reduce::PROD:
      // red *= val;
      result = createCompoundAssignOperator(Ctx, tmp_var, ret_val, BO_MulAssign,
          tmp_var->getType());
      break;
    case Reduce::MEDIAN:
      hipacc_require(0,"Unsupported reduction mode.");
  }

  return result;
}


template<typename T> T get_init(Reduce mode) {
  switch (mode) {
    case Reduce::SUM:    return 0;
    case Reduce::MIN:    return std::numeric_limits<T>::max();
    case Reduce::MAX:    return std::numeric_limits<T>::min();
    case Reduce::PROD:   return 1;
    case Reduce::MEDIAN: hipacc_require(false,"Median not yet supported");
    default:             hipacc_require(false,"Unsupported reduction mode");
  }
  return 0;
}

// create init expression for given aggregation mode and type
Expr *ASTTranslate::getInitExpr(Reduce mode, QualType QT) {
  Expr *result = nullptr, *initExpr = nullptr;

  QualType EQT = QT;
  bool isVecType = QT->isVectorType();

  if (isVecType) {
    EQT = QT->getAs<VectorType>()->getElementType();
  }
  const BuiltinType *BT = EQT->getAs<BuiltinType>();

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
      hipacc_require(0,"BuiltinType for reduce function not supported.");

    // FIXME: Clang adds weird suffixes to integer literals with less then 32
    // bits (e.g. "i8", "Ui16", @see StmtPrinter::VisitIntegerLiteral). As a
    // workaround, we fall through for integer types smaller 32 bits and create
    // the accordingly signed 32 bit literal instead.
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      //initExpr = new (Ctx) CharacterLiteral(get_init<signed char>(mode),
      //    CharacterLiteral::Ascii, QT, SourceLocation());
      //break;
    case BuiltinType::Short: //{
      //llvm::APInt init(16, get_init<short>(mode));
      //initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      //break; }
    case BuiltinType::Int: //{
      //llvm::APInt init(32, get_init<int>(mode));
      //initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      //break; }
      initExpr = createIntegerLiteral(Ctx, get_init<int>(mode));
      break;
    case BuiltinType::Long: {
      llvm::APInt init(64, get_init<long long>(mode));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      //initExpr = new (Ctx) CharacterLiteral(get_init<unsigned char>(mode),
      //    CharacterLiteral::Ascii, QT, SourceLocation());
      //break;
    case BuiltinType::Char16:
    case BuiltinType::UShort: //{
      //llvm::APInt init(16, get_init<unsigned short>(mode));
      //initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      //break; }
    case BuiltinType::Char32:
    case BuiltinType::UInt: //{
      //llvm::APInt init(32, get_init<unsigned>(mode));
      //initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      //break; }
      initExpr = createIntegerLiteral(Ctx, get_init<unsigned>(mode));
      break;
    case BuiltinType::ULong: {
      llvm::APInt init(64, get_init<unsigned long long>(mode));
      initExpr = new (Ctx) IntegerLiteral(Ctx, init, EQT, SourceLocation());
      break; }
    case BuiltinType::Float: {
      llvm::APFloat init(get_init<float>(mode));
      initExpr = FloatingLiteral::Create(Ctx, init, false, EQT, SourceLocation());
      break; }
    case BuiltinType::Double: {
      llvm::APFloat init(get_init<double>(mode));
      initExpr = FloatingLiteral::Create(Ctx, init, false, EQT, SourceLocation());
      break; }
  }

  if (isVecType) {
    auto lanes = QT->getAs<VectorType>()->getNumElements();
    SmallVector<Expr *, 16> init_exprs(lanes, initExpr);

    result = new (Ctx) InitListExpr(Ctx, SourceLocation(), init_exprs,
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
  hipacc_require(domain_var,"Domain.");

  Expr *dom_acc = nullptr;
  switch (compilerOptions.getTargetLang()) {
    case Language::CUDA:
      // array subscript: Domain[y][x]
      dom_acc = accessMem2DAt(domain_var, createIntegerLiteral(Ctx,
            redIdxX.back()), createIntegerLiteral(Ctx, redIdxY.back()));
      break;
    case Language::C99:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      // array subscript: Domain[y*width + x]
      dom_acc = accessMemArrAt(domain_var, createIntegerLiteral(Ctx,
            static_cast<int>(Domain->getSizeX())), createIntegerLiteral(Ctx,
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
  enum class Method : uint8_t {
    Convolve,
    Reduce,
    Iterate
  };
  // check if this is a convolve function call
  Method method = Method::Convolve;
  if (E->getDirectCallee()->getName().equals("convolve")) {
    method = Method::Convolve;
    hipacc_require(convMask == nullptr,
            "Nested convolution calls are not supported.");
  } else if (E->getDirectCallee()->getName().equals("reduce")) {
    method = Method::Reduce;
  } else if (E->getDirectCallee()->getName().equals("iterate")) {
    method = Method::Iterate;
  } else {
    hipacc_require(false,"Unsupported convolution method.");
  }

  switch (method) {
    case Method::Convolve:
      // convolve(mask, mode, [&] () { lambda-function; });
      hipacc_require(E->getNumArgs() == 3,"Expected 3 arguments to 'convolve' call.");
      break;
    case Method::Reduce:
      // reduce(domain, mode, [&] () { lambda-function; });
      hipacc_require(E->getNumArgs() == 3,"Expected 3 arguments to 'reduce' call.");
      break;
    case Method::Iterate:
      // iterate(domain, [&] () { lambda-function; });
      hipacc_require(E->getNumArgs() == 2,"Expected 2 arguments to 'iterate' call.");
      break;
  }

  // first parameter: Mask<type> or Domain reference
  HipaccMask *Mask = nullptr;
  if (method==Method::Convolve)
    hipacc_require(isa<MemberExpr>(E->getArg(0)->IgnoreImpCasts()) &&
        isa<FieldDecl>(dyn_cast<MemberExpr>(E->getArg(0)->IgnoreImpCasts())->getMemberDecl()),
           "First parameter to 'convolve' call must be a Mask.");
  else
    hipacc_require(isa<MemberExpr>(E->getArg(0)->IgnoreImpCasts()) &&
        isa<FieldDecl>(dyn_cast<MemberExpr>(E->getArg(0)->IgnoreImpCasts())->getMemberDecl()),
           "First parameter to 'reduce'/'iterate' call must be a Domain.");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0)->IgnoreImpCasts());
  FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

  // look for Mask/Domain user class member variable
  if (Kernel->getMaskFromMapping(FD)) {
    Mask = Kernel->getMaskFromMapping(FD);
  }
  switch (method) {
    case Method::Convolve:
      hipacc_require(Mask && !Mask->isDomain(), "Could not find Mask Field Decl.");
      break;
    case Method::Reduce:
    case Method::Iterate:
      hipacc_require(Mask && Mask->isDomain(), "Could not find Domain Field Decl.");
      break;
  }

  // second parameter: convolution/reduction mode
  if (method==Method::Convolve || method==Method::Reduce) {
    if (method==Method::Convolve)
      hipacc_require(isa<DeclRefExpr>(E->getArg(1)),
          "Second parameter to 'convolve' call must be the convolution mode.");
    if (method==Method::Reduce)
      hipacc_require(isa<DeclRefExpr>(E->getArg(1)),
          "Second parameter to 'reduce' call must be the reduction mode.");
    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getArg(1));

    if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
        DRE->getDecl()->getType().getAsString() ==
        "enum hipacc::Reduce") {
      auto lval = E->getArg(1)->EvaluateKnownConstInt(Ctx);
      auto mval = static_cast<std::underlying_type<Reduce>::type>(Reduce::MEDIAN);
      auto mode = static_cast<Reduce>(lval.getZExtValue());
      hipacc_require(lval.isNonNegative() && lval.getZExtValue() <= mval,
             "invalid Reduce mode");
      if (method==Method::Convolve)
        convMode = mode;
      else
        redModes.push_back(mode);
    } else {
      unsigned DiagIDConvMode = Diags.getCustomDiagID(DiagnosticsEngine::Error,
          "Unknown Reduce mode detected.");
      Diags.Report(E->getArg(1)->getExprLoc(), DiagIDConvMode);
      exit(EXIT_FAILURE);
    }
  }

  // third parameter: lambda-function
  int li = 2;
  if (method==Method::Iterate)
    li = 1;

  hipacc_require(isa<MaterializeTemporaryExpr>(E->getArg(li)) &&
         isa<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(E->getArg(li))->getSubExpr()->IgnoreImpCasts()),
         "Third parameter to 'reduce' or 'iterate' call must be a"
         "lambda-function.");
         
  LambdaExpr *LE = dyn_cast<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(
                       E->getArg(li))->getSubExpr()->IgnoreImpCasts());

  // check default capture kind
  if (LE->getCaptureDefault()==LCD_ByCopy) {
    unsigned DiagIDCapture = Diags.getCustomDiagID(DiagnosticsEngine::Error,
        "Capture by copy [=] is not supported for '%0' lambda-function. "
        "Use capture by reference [&] instead.");
    Diags.Report(LE->getCaptureDefaultLoc(), DiagIDCapture)
      << (const char *)(method==Method::Convolve ? "convolve" : method==Method::Reduce ?
          "reduce" : "iterate");
    exit(EXIT_FAILURE);
  }
  // check capture kind of variables
  for (auto capture : LE->captures()) {
    if (capture.capturesVariable() && capture.getCaptureKind()!=LCK_ByRef) {
      unsigned DiagIDCapture = Diags.getCustomDiagID(DiagnosticsEngine::Error,
          "Unsupported capture kind for variable '%0' in '%1' "
          "lambda-function. Use capture by reference instead: [&%0].");
      Diags.Report(capture.getLocation(), DiagIDCapture)
        << capture.getCapturedVar()->getNameAsString()
        << (const char *)(method==Method::Convolve ? "convolve" : method==Method::Reduce ?
            "reduce" : "iterate");
      exit(EXIT_FAILURE);
    }
  }

  // init temporary variable depending on aggregation mode
  Expr *init = nullptr;
  switch (method) {
    case Method::Convolve:
      init = getInitExpr(convMode, LE->getCallOperator()->getReturnType());
      break;
    case Method::Reduce:
      init = getInitExpr(redModes.back(),
          LE->getCallOperator()->getReturnType());
      break;
    case Method::Iterate: break;
  }
  std::string tmp_lit("_tmp" + std::to_string(literalCount++));
  VarDecl *tmp_decl = createVarDecl(Ctx, kernelDecl, tmp_lit,
      LE->getCallOperator()->getReturnType(), init);
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(tmp_decl);
  DeclRefExpr *tmp_dre = createDeclRefExpr(Ctx, tmp_decl);

  // introduce temporary for holding the convolution/reduction result
  CompoundStmt *outerCompountStmt = curCStmt;

  switch (method) {
    case Method::Convolve:
      convTmp = tmp_dre;
      convMask = Mask;
      preStmts.push_back(createDeclStmt(Ctx, tmp_decl));
      preCStmt.push_back(outerCompountStmt);
      break;
    case Method::Reduce:
      redTmps.push_back(tmp_dre);
      redDomains.push_back(Mask);
      preStmts.push_back(createDeclStmt(Ctx, tmp_decl));
      preCStmt.push_back(outerCompountStmt);
      break;
    case Method::Iterate:
      redDomains.push_back(Mask);
      break;
  }

  // unroll Mask/Domain
  for (size_t y=0; y<Mask->getSizeY(); ++y) {
    for (size_t x=0; x<Mask->getSizeX(); ++x) {
      if (Mask->isDomain() && Mask->isConstant() &&
          !Mask->isDomainDefined(x, y))
        continue;

      Stmt *iteration = nullptr;
      switch (method) {
        case Method::Convolve:
          convIdxX = x;
          convIdxY = y;
          iteration = Clone(LE->getBody());
          break;
        case Method::Reduce:
        case Method::Iterate:
          redIdxX.push_back(x);
          redIdxY.push_back(y);
          iteration = Clone(LE->getBody());
          // add check if this iteration point should be processed - the
          // DeclRefExpr for the Domain is retrieved when visiting the
          // MemberExpr
          if (!Mask->isConstant()) {
            // set Domain as being used within Kernel
            Kernel->setUsed(FD->getNameAsString());
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

  // reset global variables
  switch (method) {
    case Method::Convolve:
      convMask = nullptr;
      convTmp = nullptr;
      convIdxX = convIdxY = 0;
      break;
    case Method::Reduce:
      redDomains.pop_back();
      redModes.pop_back();
      redTmps.pop_back();
      break;
    case Method::Iterate:
      redDomains.pop_back();
      break;
  }

  // result of convolution
  switch (method) {
    case Method::Convolve:
    case Method::Reduce:
      // add ICE for CodeGen
      return createImplicitCastExpr(Ctx, LE->getCallOperator()->getReturnType(),
          CK_LValueToRValue, tmp_dre, nullptr, VK_RValue);
    case Method::Iterate:
      return nullptr;
    default:
      hipacc_require(false,"Unsupported convolution method.");
      return nullptr;
  }
}

// vim: set ts=2 sw=2 sts=2 et ai:

