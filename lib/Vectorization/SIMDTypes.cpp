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

//===--- SIMDTypes.cpp - Information about SIMD Vector Types --------------===//
//
// This file implements mapping of scalar types to vector types.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Vectorization/SIMDTypes.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;
using namespace hipacc::Builtin;


QualType SIMDTypes::createSIMDType(QualType QT, StringRef base, SIMDWidth
    simd_width) {
  int lanes;
  std::string tname(base);

  switch (simd_width) {
    default:
    case SIMD1:  lanes = 1;                 break;
    case SIMD2:  lanes = 2;  tname += "2";  break;
    case SIMD3:  lanes = 3;  tname += "3";  break;
    case SIMD4:  lanes = 4;  tname += "4";  break;
    case SIMD8:  lanes = 8;  tname += "8";  break;
    case SIMD16: lanes = 16; tname += "16"; break;
  }

  // use ext_vector_type for SIMD types
  QualType SIMDType = Ctx.getExtVectorType(QT, lanes);

  // create typedef and return this as result
  TypeSourceInfo *TInfo = Ctx.getTrivialTypeSourceInfo(SIMDType);
  TypedefDecl *TD = TypedefDecl::Create(Ctx, Ctx.getTranslationUnitDecl(),
      SourceLocation(), SourceLocation(), &Ctx.Idents.get(tname), TInfo);

  return Ctx.getTypeDeclType(TD);
}


QualType SIMDTypes::getSIMDType(QualType QT, StringRef base, SIMDWidth
    simd_width) {
  const BuiltinType *BT = QT->getAs<BuiltinType>();

  if (typeToVectorType[simd_width].count(BT)) {
    return typeToVectorType[simd_width][BT];
  }

  QualType SIMDType = createSIMDType(QT, base, simd_width);
  typeToVectorType[simd_width][BT] = SIMDType;

  return SIMDType;
}


QualType SIMDTypes::getSIMDTypeFromBT(const BuiltinType *BT, VarDecl *VD,
    SIMDWidth simd_width) {
  QualType SIMDType;

  switch (BT->getKind()) {
    case BuiltinType::WChar_U:
    case BuiltinType::WChar_S:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::LongDouble:
    default:
      Ctx.getDiagnostics().Report(VD->getLocation(), DiagIDType) <<
        BT->getName(PrintingPolicy(Ctx.getLangOpts())) << VD->getName();
      assert(0 && "BuiltinType not supported");
    case BuiltinType::Void:
      SIMDType = Ctx.VoidTy;
      break;
    case BuiltinType::Bool:
      SIMDType = Ctx.BoolTy;
      break;
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      SIMDType = createSIMDType(Ctx.CharTy, "char", simd_width);
      break;
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      SIMDType = createSIMDType(Ctx.UnsignedCharTy, "uchar", simd_width);
      break;
    case BuiltinType::Char16:
    case BuiltinType::Short:
      SIMDType = createSIMDType(Ctx.ShortTy, "short", simd_width);
      break;
    case BuiltinType::UShort:
      SIMDType = createSIMDType(Ctx.UnsignedShortTy, "ushort", simd_width);
      break;
    case BuiltinType::Char32:
    case BuiltinType::Int:
      SIMDType = createSIMDType(Ctx.IntTy, "int", simd_width);
      break;
    case BuiltinType::UInt:
      SIMDType = createSIMDType(Ctx.UnsignedIntTy, "uint", simd_width);
      break;
    case BuiltinType::Long:
      SIMDType = createSIMDType(Ctx.LongTy, "long", simd_width);
      break;
    case BuiltinType::ULong:
      SIMDType = createSIMDType(Ctx.UnsignedLongTy, "ulong", simd_width);
      break;
    case BuiltinType::Float:
      SIMDType = createSIMDType(Ctx.FloatTy, "float", simd_width);
      break;
    case BuiltinType::Double:
      SIMDType = createSIMDType(Ctx.DoubleTy, "double", simd_width);
      break;
  }

  return SIMDType;
}


QualType SIMDTypes::getSIMDType(ParmVarDecl *PVD, SIMDWidth simd_width) {
  QualType SIMDType;

  if (imgsToVectorType[simd_width].count(PVD)) {
    return imgsToVectorType[simd_width][PVD];
  }

  // get raw type, discarding pointer
  const BuiltinType *BT;
  if (PVD->getType()->isPointerType()) {
    BT = PVD->getType()->getPointeeType()->getAs<BuiltinType>();
  } else {
    BT = PVD->getType()->getAs<BuiltinType>();
  }

  if (typeToVectorType[simd_width].count(BT)) {
    SIMDType = typeToVectorType[simd_width][BT];
  } else {
    SIMDType = getSIMDTypeFromBT(BT, PVD, simd_width);
    typeToVectorType[simd_width][BT] = SIMDType;
  }

  // add __global address qualifier for OpenCL back end
  if (options.emitOpenCL()) {
    SIMDType = Ctx.getAddrSpaceQualType(SIMDType, LangAS::opencl_global);
  }

  // add pointer to type
  SIMDType = Ctx.getPointerType(SIMDType);
  imgsToVectorType[simd_width][PVD] = SIMDType;

  return SIMDType;
}


QualType SIMDTypes::getSIMDType(VarDecl *VD, SIMDWidth simd_width) {
  if (declsToVectorType[simd_width].count(VD)) {
    return declsToVectorType[simd_width][VD];
  }

  const BuiltinType *BT = VD->getType()->getAs<BuiltinType>();
  if (typeToVectorType[simd_width].count(BT)) {
    declsToVectorType[simd_width][VD] = typeToVectorType[simd_width][BT];

    return typeToVectorType[simd_width][BT];
  }

  QualType SIMDType = getSIMDTypeFromBT(BT, VD, simd_width);

  declsToVectorType[simd_width][VD] = SIMDType;
  typeToVectorType[simd_width][BT] = SIMDType;

  return SIMDType;
}

Expr *SIMDTypes::propagate(VarDecl *VD, Expr *E) {
  // vector types do not need further widening
  if (E->getType()->isVectorType() || E->getType()->isExtVectorType())
    return E;

  // only propagate scalars for CUDA back end
  if (!options.emitCUDA()) return E;

  QualType QT = VD->getType();
  const BuiltinType *BT = QT->getAs<BuiltinType>();

  FunctionDecl *make_vec = nullptr;
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
      Ctx.getDiagnostics().Report(E->getExprLoc(), DiagIDType) <<
        BT->getName(PrintingPolicy(Ctx.getLangOpts())) << VD->getName();
      assert(0 && "BuiltinType not supported");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      make_vec = builtins.getBuiltinFunction(CUDABImake_char4);
      break;
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      make_vec = builtins.getBuiltinFunction(CUDABImake_uchar4);
      break;
    case BuiltinType::Char16:
    case BuiltinType::Short:
      make_vec = builtins.getBuiltinFunction(CUDABImake_short4);
      break;
    case BuiltinType::UShort:
      make_vec = builtins.getBuiltinFunction(CUDABImake_ushort4);
      break;
    case BuiltinType::Char32:
    case BuiltinType::Int:
      make_vec = builtins.getBuiltinFunction(CUDABImake_int4);
      break;
    case BuiltinType::UInt:
      make_vec = builtins.getBuiltinFunction(CUDABImake_uint4);
      break;
    case BuiltinType::Long:
      make_vec = builtins.getBuiltinFunction(CUDABImake_long4);
      break;
    case BuiltinType::ULong:
      make_vec = builtins.getBuiltinFunction(CUDABImake_ulong4);
      break;
    case BuiltinType::Float:
      make_vec = builtins.getBuiltinFunction(CUDABImake_float4);
      break;
    case BuiltinType::Double:
      make_vec = builtins.getBuiltinFunction(CUDABImake_double4);
      break;
  }

  // parameters for make_vec
  SmallVector<Expr *, 16> args;
  args.push_back(E);

  return createFunctionCall(Ctx, make_vec, args);
}

// vim: set ts=2 sw=2 sts=2 et ai:

