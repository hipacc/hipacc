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


// create interpolation function name
std::string ASTTranslate::getInterpolationName(CompilerOptions &compilerOptions,
    HipaccKernel *Kernel, HipaccAccessor *Acc) {
  std::string name = "interpolate_";

  switch (Acc->getInterpolationMode()) {
    case Interpolate::NO:
    case Interpolate::NN:                break;
    case Interpolate::B5: name += "b5_"; break;
    case Interpolate::LF: name += "lf_"; break;
    case Interpolate::CF: name += "cf_"; break;
    case Interpolate::L3: name += "l3_"; break;
  }

  switch (compilerOptions.getTargetLang()) {
    case Language::C99: name += "gmem";  break;
    case Language::CUDA:
      switch (Kernel->useTextureMemory(Acc)) {
        case Texture::None:
        case Texture::Ldg:       name += "gmem";  break;
        case Texture::Linear1D:  name += "tex1D"; break;
        case Texture::Linear2D:
        case Texture::Array2D:   name += "tex2D"; break;
      }
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      switch (Kernel->useTextureMemory(Acc)) {
        case Texture::None:
        case Texture::Linear1D:
        case Texture::Linear2D:
        case Texture::Ldg:       name += "gmem";  break;
        case Texture::Array2D:   name += "img";   break;
      }
      break;
  }

  return name;
}


// calculate index using nearest neighbor interpolation
Expr *ASTTranslate::addNNInterpolationX(HipaccAccessor *Acc, Expr *idx_x) {
  // acc_scale_x * (gid_x - is_offset_x)
  idx_x = removeISOffsetX(idx_x);

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
  QualType QT = Acc->getImage()->getType();
  std::string typeSpecifier = builtins.EncodeTypeIntoStr(QT, Ctx);

  std::string name = getInterpolationName(compilerOptions, Kernel, Acc);

  // only add boundary handling mode string if required
  // for local operators only add support if the code variant requires this
  // for point, global, and user operators add boundary handling for all borders
  if (KernelClass->getKernelType() != LocalOperator || bh_variant.borderVal) {
    switch (Acc->getBoundaryMode()) {
      case Boundary::UNDEFINED:                      break;
      case Boundary::CLAMP:    name += "_clamp_";    break;
      case Boundary::REPEAT:   name += "_repeat_";   break;
      case Boundary::MIRROR:   name += "_mirror_";   break;
      case Boundary::CONSTANT: name += "_constant_"; break;
    }

    if (Acc->getBoundaryMode() != Boundary::UNDEFINED) {
      if (KernelClass->getKernelType() != LocalOperator) {
        name += "tblr";
      } else {
        if (bh_variant.borders.top)    name += "t";
        if (bh_variant.borders.bottom) name += "b";
        if (bh_variant.borders.left)   name += "l";
        if (bh_variant.borders.right)  name += "r";
      }
    }
  }

  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      // no function overloading supported in OpenCL -> add type specifier to function name
      name += "_" + builtins.EncodeTypeIntoStr(Acc->getImage()->getType(), Ctx);
      break;
  }

  // lookup interpolation function
  FunctionDecl *interpolateDecl = lookup<FunctionDecl>(name,
      Acc->getImage()->getType());

  // create function declaration
  if (!interpolateDecl) {
    std::string funcTypeSpecifier = typeSpecifier + typeSpecifier + "*C"
      + "iCfCfCiCiCiCiC";
    if (bh_variant.borderVal && Acc->getBoundaryMode() == Boundary::CONSTANT) {
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

  // mark image as being used within the kernel
  Kernel->setUsed(LHS->getNameInfo().getAsString());

  FunctionDecl *interpolation = getInterpolationFunction(Acc);

  // parameters for interpolate function call
  SmallVector<Expr *, 16> args;
  if (compilerOptions.emitCUDA() &&
      Kernel->useTextureMemory(Acc) != Texture::None) {
    hipacc_require(isa<ParmVarDecl>(LHS->getDecl()), "texture variable must be a ParmVarDecl!");
    ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(LHS->getDecl());
    args.push_back(createDeclRefExpr(Ctx, CloneDeclTex(PVD, "_tex")));
  } else {
    args.push_back(LHS);
  }
  args.push_back(getStrideDecl(Acc));
  args.push_back(idx_x);
  args.push_back(idx_y);
  args.push_back(getWidthDecl(Acc));
  args.push_back(getHeightDecl(Acc));
  // global offset_[x|y]
  if (Acc->getOffsetXDecl()) {
    args.push_back(getOffsetXDecl(Acc));
  } else {
    args.push_back(createIntegerLiteral(Ctx, 0));
  }
  if (Acc->getOffsetYDecl()) {
    args.push_back(getOffsetYDecl(Acc));
  } else {
    args.push_back(createIntegerLiteral(Ctx, 0));
  }
  // const val
  if (Acc->getBoundaryMode() == Boundary::CONSTANT && bh_variant.borderVal!=0) {
    args.push_back(Acc->getConstExpr());
  }

  return createFunctionCall(Ctx, interpolation, args);
}

// vim: set ts=2 sw=2 sts=2 et ai:

