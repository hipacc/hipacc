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

using namespace clang;
using namespace hipacc;
using namespace ASTNode;


// create expression for convolutions
Stmt *ASTTranslate::getConvolutionStmt(ConvolutionMode mode, DeclRefExpr
    *tmp_var, Expr *ret_val) {
  Stmt *result;
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
      fun = getConvolutionFunction("min", tmp_var->getType());
      funArgs.push_back(createImplicitCastExpr(Ctx, tmp_var->getType(),
            CK_LValueToRValue, tmp_var, NULL, VK_RValue));
      funArgs.push_back(ret_val);
      result = createBinaryOperator(Ctx, tmp_var, createFunctionCall(Ctx, fun,
            funArgs), BO_Assign, tmp_var->getType());
      break;
    case HipaccMAX:
      // red = max(red, val);
      fun = getConvolutionFunction("max", tmp_var->getType());
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
      break;
  }

  return result;
}


// create function for given type
FunctionDecl *ASTTranslate::getConvolutionFunction(std::string name, QualType
    QT) {
  FunctionDecl *result = NULL;

  // lookup aggregation function
  for (DeclContext::lookup_result Lookup =
      Ctx.getTranslationUnitDecl()->lookup(DeclarationName(&Ctx.Idents.get(name)));
      !Lookup.empty(); Lookup=Lookup.slice(1)) {
    FunctionDecl *Decl = cast_or_null<FunctionDecl>(Lookup.front());

    if (Decl && Decl->getResultType() == QT.getDesugaredType(Ctx)) {
      return Decl;
    }
  }

  // create function declaration
  std::string vecTypeSpecifier = "";
  QualType VT = QT;
  if (QT->isVectorType()) {
    int lanes = VT->getAs<VectorType>()->getNumElements();
    std::stringstream LSS;
    LSS << "E" << lanes;
    vecTypeSpecifier = LSS.str();
    QT = QT->getAs<VectorType>()->getElementType();
  }

  std::string typeSpecifier = builtins.EncodeTypeIntoStr(QT, Ctx);
  std::string funcTypeSpecifier = vecTypeSpecifier + typeSpecifier;
  funcTypeSpecifier += vecTypeSpecifier + typeSpecifier;
  funcTypeSpecifier += vecTypeSpecifier + typeSpecifier;

  QualType FT = builtins.getBuiltinType(funcTypeSpecifier.c_str());
  result = builtins.CreateBuiltin(FT, name.c_str());

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

