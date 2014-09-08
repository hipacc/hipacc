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

//===--- SIMDTypes.h - Information about SIMD Vector Types ----------------===//
//
// This file implements mapping of scalar types to vector types.
//
//===----------------------------------------------------------------------===//

#ifndef _SIMDTYPES_H_
#define _SIMDTYPES_H_

#include <clang/AST/Type.h>

#include <sstream>

#include "hipacc/AST/ASTNode.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/Builtins.h"

namespace clang {
namespace hipacc {

enum SIMDWidth {
  SIMD1   = 0x0,
  SIMD2   = 0x1,
  SIMD3   = 0x2,
  SIMD4   = 0x3,
  SIMD8   = 0x4,
  SIMD16  = 0x5,
  SIMDEND
};

class SIMDTypes {
  private:
    ASTContext &Ctx;
    hipacc::Builtin::Context &builtins;
    CompilerOptions &options;
    llvm::DenseMap<const VarDecl *, QualType> declsToVectorType[SIMDEND];
    llvm::DenseMap<const ParmVarDecl *, QualType> imgsToVectorType[SIMDEND];
    llvm::DenseMap<const BuiltinType *, QualType> typeToVectorType[SIMDEND];
    unsigned DiagIDType;

    QualType getSIMDTypeFromBT(const BuiltinType *BT, VarDecl *VD, SIMDWidth
        simd_width);

  public:
    SIMDTypes(ASTContext &Ctx, hipacc::Builtin::Context &builtins,
        CompilerOptions &options) :
      Ctx(Ctx),
      builtins(builtins),
      options(options)
    {
      DiagIDType =
        Ctx.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
            "BuiltinType %0 for Variable %1 not supported.");
    }

    ~SIMDTypes() {}

    QualType getSIMDType(ParmVarDecl *PVD, SIMDWidth simd_width);
    QualType getSIMDType(VarDecl *VD, SIMDWidth simd_width);
    QualType getSIMDType(QualType QT, StringRef base, SIMDWidth simd_width);
    QualType createSIMDType(QualType QT, StringRef base, SIMDWidth simd_width);
    Expr *propagate(VarDecl *VD, Expr *E);
};
} // end namespace hipacc
} // end namespace clang

#endif  // _SIMDTYPES_H_

// vim: set ts=2 sw=2 sts=2 et ai:

