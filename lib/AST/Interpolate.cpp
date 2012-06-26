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

//===--- Interpolate.cpp - Add Interpolation Calls to the AST -------------===//
//
// This file implements the translation of memory accesses while adding
// interpolation calls.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;
using namespace hipacc::Builtin;


// calculate index using nearest neighbor interpolation
Expr *ASTTranslate::addNNInterpolationX(HipaccAccessor *Acc, Expr *idx_x) {
    // acc_scale_x * (gid_x - is_offset_x)
    if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
      idx_x = createBinaryOperator(Ctx, idx_x,
          Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(),
          BO_Sub, Ctx.IntTy);
    }
    return createBinaryOperator(Ctx, Acc->getScaleXDecl(), createParenExpr(Ctx,
          idx_x), BO_Mul, Ctx.FloatTy);
}
Expr *ASTTranslate::addNNInterpolationY(HipaccAccessor *Acc, Expr *idx_y) {
    // acc_scale_y * (gid_y)
    return createBinaryOperator(Ctx, Acc->getScaleYDecl(), createParenExpr(Ctx,
          idx_y), BO_Mul, Ctx.FloatTy);
}


// create interpolation function declaration
FunctionDecl *ASTTranslate::getInterpolationFunction(HipaccAccessor *Acc) {
  // interpolation function is constructed as follows:
  // interpolate_ + <interpolation_mode> + _ + <boundary_handling_mode> + _ +
  // <memory_type>
  std::string name = "interpolate_";
  FunctionDecl *interpolateDecl = NULL;
  QualType QT = Acc->getImage()->getPixelQualType();
  std::string typeSpecifier = builtins.EncodeTypeIntoStr(QT, Ctx);

  switch (Acc->getInterpolation()) {
    case InterpolateNO:
    case InterpolateNN:
    default:
      break;
    case InterpolateLF:
      name += "lf_";
      break;
    case InterpolateCF:
      name += "cf_";
      break;
    case InterpolateL3:
      name += "l3_";
      break;
  }

  // only add boundary handling mode string if required
  if (imgBorderVal) {
    switch (Acc->getBoundaryHandling()) {
      case BOUNDARY_UNDEFINED:
      default:
        break;
      case BOUNDARY_CLAMP:
        name += "clamp_";
        break;
      case BOUNDARY_REPEAT:
        name += "repeat_";
        break;
      case BOUNDARY_MIRROR:
        name += "mirror_";
        break;
      case BOUNDARY_CONSTANT:
        name += "constant_";
        break;
    }

    if (imgBorder.top) name += "t";
    if (imgBorder.bottom) name += "b";
    if (imgBorder.left) name += "l";
    if (imgBorder.right) name += "r";
    name += "_";
  }

  if (Kernel->useTextureMemory(Acc)) {
    if (compilerOptions.emitCUDA()) {
      name += "tex";
    } else {
      name += "img";
    }
  } else {
    name += "gmem";
  }

  if (!compilerOptions.emitCUDA()) {
    // no function overloading supported in OpenCL -> add type specifier to function name
    name += "_" + typeSpecifier;
  }

  // lookup interpolation function
  for (DeclContext::lookup_result Lookup =
      Ctx.getTranslationUnitDecl()->lookup(DeclarationName(&Ctx.Idents.get(name)));
      Lookup.first!=Lookup.second; ++Lookup.first) {
    FunctionDecl *Decl = cast_or_null<FunctionDecl>(*Lookup.first);

    if (Decl && Decl->getResultType() == Acc->getImage()->getPixelQualType()) {
      interpolateDecl = Decl;
      break;
    }
  }

  // create function declaration
  if (!interpolateDecl) {
    std::string funcTypeSpecifier = typeSpecifier + typeSpecifier + "*C"
      + "iCfCfCiCiCiCiC";
    if (imgBorderVal && Acc->getBoundaryHandling()==BOUNDARY_CONSTANT) {
      funcTypeSpecifier += typeSpecifier + "C";
    }

    QualType FT = builtins.getBuiltinType(funcTypeSpecifier.c_str());
    interpolateDecl = builtins.CreateBuiltin(FT, name.c_str());
  }

  return interpolateDecl;
}


// calculate interpolated value using external function
Expr *ASTTranslate::addInterpolationCall(DeclRefExpr *LHS, HipaccAccessor
    *Acc, Expr *idx_x, Expr *idx_y) {
  idx_x = addNNInterpolationX(Acc, idx_x);
  idx_y = addNNInterpolationY(Acc, idx_y);

  FunctionDecl *interpolation = getInterpolationFunction(Acc);

  // parameters for interpolate function call
  llvm::SmallVector<Expr *, 16> args;
  if (compilerOptions.emitCUDA() && Kernel->useTextureMemory(Acc)) {
    assert(isa<ParmVarDecl>(LHS->getDecl()) && "texture variable must be a ParmVarDecl!");
    ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(LHS->getDecl());
    args.push_back(createDeclRefExpr(Ctx, CloneDeclTex(PVD)));
  } else {
    args.push_back(LHS);
  }
  args.push_back(Acc->getStrideDecl());
  args.push_back(idx_x);
  args.push_back(idx_y);
  args.push_back(Acc->getWidthDecl());
  args.push_back(Acc->getHeightDecl());
  // global offset_[x|y]
  if (Acc->getOffsetXDecl()) {
    args.push_back(Acc->getOffsetXDecl());
  } else {
    args.push_back(createIntegerLiteral(Ctx, 0));
  }
  if (Acc->getOffsetYDecl()) {
    args.push_back(Acc->getOffsetYDecl());
  } else {
    args.push_back(createIntegerLiteral(Ctx, 0));
  }
  // const val
  if (Acc->getBoundaryHandling() == BOUNDARY_CONSTANT && imgBorderVal!=0) {
    args.push_back(Acc->getConstExpr());
  }

  return createFunctionCall(Ctx, interpolation, args); 
}

// vim: set ts=2 sw=2 sts=2 et ai:

