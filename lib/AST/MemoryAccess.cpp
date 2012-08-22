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

//===--- MemoryAccess.cpp - Rewrite Memory Accesses in the AST ------------===//
//
// This file implements the translation of memory accesses to different memory
// regions.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;
using namespace hipacc::Builtin;


// add local offset to index
Expr *ASTTranslate::addLocalOffset(Expr *idx, Expr *local_offset) {
  if (local_offset) {
    idx = createBinaryOperator(Ctx, idx, Clone(local_offset), BO_Add,
        Ctx.IntTy);
  }

  return idx;
}


// add global offset to index
Expr *ASTTranslate::addGlobalOffsetY(Expr *idx_y, HipaccAccessor *Acc) {
  if (Acc->getOffsetYDecl()) {
    idx_y = createBinaryOperator(Ctx, idx_y, Acc->getOffsetYDecl(), BO_Add,
        Ctx.IntTy);
  }

  return idx_y;
}
Expr *ASTTranslate::addGlobalOffsetX(Expr *idx_x, HipaccAccessor *Acc) {
  if (Acc->getOffsetXDecl()) {
    idx_x = createBinaryOperator(Ctx, idx_x, Acc->getOffsetXDecl(), BO_Add,
        Ctx.IntTy);
  }

  return idx_x;
}


// remove iteration space offset from index
Expr *ASTTranslate::removeISOffsetX(Expr *idx_x, HipaccAccessor *Acc) {
  if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
      idx_x = createBinaryOperator(Ctx, idx_x,
          Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(), BO_Sub,
          Ctx.IntTy);
  }

  return idx_x;
}


// access 1D memory array
Expr *ASTTranslate::accessMem(DeclRefExpr *LHS, HipaccAccessor *Acc,
    MemoryAccess memAcc, Expr *local_offset_x, Expr *local_offset_y) {
  Expr *idx_x = gidXRef;
  Expr *idx_y = gidYRef;

  // step 0: add local offset: gid_[x|y] + local_offset_[x|y]
  idx_x = addLocalOffset(idx_x, local_offset_x);
  idx_y = addLocalOffset(idx_y, local_offset_y);

  // step 1: remove is_offset and add interpolation & boundary handling
  switch (Acc->getInterpolation()) {
    case InterpolateNO:
      if (Acc!=Kernel->getIterationSpace()->getAccessor()) {
        idx_x = removeISOffsetX(idx_x, Acc);
      }
      break;
    case InterpolateNN:
      idx_x = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), NULL, NULL);
      idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), NULL, NULL);
      break;
    case InterpolateLF:
    case InterpolateCF:
    case InterpolateL3:
      return addInterpolationCall(LHS, Acc, idx_x, idx_y);
      break;
  }

  // step 2: add global Accessor/Iteration Space offset
  if (Acc!=Kernel->getIterationSpace()->getAccessor()) {
    idx_x = addGlobalOffsetX(idx_x, Acc);
  }
  idx_y = addGlobalOffsetY(idx_y, Acc);

  // step 3: access the appropriate memory
  switch (memAcc) {
    case WRITE_ONLY:
    case READ_ONLY:
      if (Kernel->useTextureMemory(Acc)) {
        if (compilerOptions.emitCUDA()) {
          return accessMemTexAt(LHS, Acc, memAcc, idx_x, idx_y);
        } else {
          return accessMemImgAt(LHS, Acc, memAcc, idx_x, idx_y);
        }
      } else {
        return accessMemArrAt(LHS, Acc->getStrideDecl(), idx_x, idx_y);
      }
      break;
    case UNDEFINED:
    case READ_WRITE:
    default:
      assert(0 && "Unsupported memory access with offset specification!\n");
      break;
  }
}


// access 1D memory array at given index
Expr *ASTTranslate::accessMemArrAt(DeclRefExpr *LHS, Expr *stride, Expr *idx_x,
    Expr *idx_y) {
  Expr *result = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
        createParenExpr(Ctx, idx_y), stride, BO_Mul, Ctx.IntTy), idx_x, BO_Add,
      Ctx.IntTy);

  result = new (Ctx) ArraySubscriptExpr(LHS, result,
      LHS->getType().getTypePtr()->getPointeeType(), VK_LValue, OK_Ordinary,
      SourceLocation());

  return result;
}


// access 2D memory array
Expr *ASTTranslate::accessMemPolly(DeclRefExpr *LHS, HipaccAccessor *Acc,
    MemoryAccess memAcc, Expr *local_offset_x, Expr *local_offset_y) {
  Expr *idx_x = gidXRef;
  Expr *idx_y = gidYRef;

  // step 0: add local offset: gid_[x|y] + local_offset_[x|y]
  idx_x = addLocalOffset(idx_x, local_offset_x);
  idx_y = addLocalOffset(idx_y, local_offset_y);

  // step 1: remove is_offset and add interpolation & boundary handling
  // no interpolation & boundary handling for Polly
  if (Acc!=Kernel->getIterationSpace()->getAccessor()) {
    idx_x = removeISOffsetX(idx_x, Acc);
  }

  // step 2: add global Accessor/Iteration Space offset
  if (Acc!=Kernel->getIterationSpace()->getAccessor()) {
    idx_x = addGlobalOffsetX(idx_x, Acc);
  }
  idx_y = addGlobalOffsetY(idx_y, Acc);

  // step 3: access the appropriate memory
  return accessMem2DAt(LHS, idx_x, idx_y);
}


// access 2D memory array at given index
Expr *ASTTranslate::accessMem2DAt(DeclRefExpr *LHS, Expr *idx_x, Expr *idx_y) {
  QualType QT = LHS->getType();
  QualType QT2 =
    QT.getTypePtr()->getPointeeType()->getAsArrayTypeUnsafe()->getElementType();

  Expr *result = new (Ctx) ArraySubscriptExpr(createImplicitCastExpr(Ctx, QT,
        CK_LValueToRValue, LHS, NULL, VK_RValue), idx_y,
      QT.getTypePtr()->getPointeeType(), VK_LValue, OK_Ordinary,
      SourceLocation());

  result = new (Ctx) ArraySubscriptExpr(createImplicitCastExpr(Ctx,
        Ctx.getPointerType(QT2), CK_ArrayToPointerDecay, result, NULL,
        VK_RValue), idx_x, QT2, VK_LValue, OK_Ordinary, SourceLocation());

  return result;
}


// get tex1Dfetch function for given Accessor
FunctionDecl *ASTTranslate::getTextureFunction(HipaccAccessor *Acc, MemoryAccess
    memAcc) {
  const BuiltinType *BT =
    Acc->getImage()->getPixelQualType()->getAs<BuiltinType>();

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
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::Double:
    default:
      assert(0 && "BuiltinType for CUDA texture not supported.");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1DfetchSc);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2DSc);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2DwriteSc);
      }
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1DfetchUc);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2DUc);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2DwriteUc);
      }
    case BuiltinType::Short:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1Dfetchs);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2Ds);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2Dwrites);
      }
    case BuiltinType::Char16:
    case BuiltinType::UShort:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1DfetchUs);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2DUs);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2DwriteUs);
      }
    case BuiltinType::Int:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1Dfetchi);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2Di);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2Dwritei);
      }
    case BuiltinType::Char32:
    case BuiltinType::UInt:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1DfetchUi);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2DUi);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2DwriteUi);
      }
    case BuiltinType::Float:
      if (memAcc==READ_ONLY) {
        switch (Kernel->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            return builtins.getBuiltinFunction(CUDABItex1Dfetchf);
          case Linear2D:
          case Array2D:
            return builtins.getBuiltinFunction(CUDABItex2Df);
        }
      } else {
        return builtins.getBuiltinFunction(CUDABIsurf2Dwritef);
      }
  }
}


// get read_image function for given Accessor
FunctionDecl *ASTTranslate::getImageFunction(HipaccAccessor *Acc, MemoryAccess
    memAcc) {
  const BuiltinType *BT =
    Acc->getImage()->getPixelQualType()->getAs<BuiltinType>();

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
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::Double:
    default:
      assert(0 && "BuiltinType for OpenCL Image not supported.");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::Int:
      if (memAcc==READ_ONLY) {
        return builtins.getBuiltinFunction(OPENCLBIread_imagei);
      } else {
        return builtins.getBuiltinFunction(OPENCLBIwrite_imagei);
      }
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::Char16:
    case BuiltinType::UShort:
    case BuiltinType::Char32:
    case BuiltinType::UInt:
      if (memAcc==READ_ONLY) {
        return builtins.getBuiltinFunction(OPENCLBIread_imageui);
      } else {
        return builtins.getBuiltinFunction(OPENCLBIwrite_imageui);
      }
    case BuiltinType::Float:
      if (memAcc==READ_ONLY) {
        return builtins.getBuiltinFunction(OPENCLBIread_imagef);
      } else {
        return builtins.getBuiltinFunction(OPENCLBIwrite_imagef);
      }
  }
}


// access linear texture memory at given index
Expr *ASTTranslate::accessMemTexAt(DeclRefExpr *LHS, HipaccAccessor *Acc,
    MemoryAccess memAcc, Expr *idx_x, Expr *idx_y) {

  FunctionDecl *texture_function = getTextureFunction(Acc, memAcc);

  // clone Decl
  TemplateArgumentListInfo templateArgs(LHS->getLAngleLoc(),
      LHS->getRAngleLoc());
  for (unsigned int i=0, e=LHS->getNumTemplateArgs(); i!=e; ++i) {
    templateArgs.addArgument(LHS->getTemplateArgs()[i]);
  }

  assert(isa<ParmVarDecl>(LHS->getDecl()) && "texture variable must be a ParmVarDecl!");
  ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(LHS->getDecl());
  DeclRefExpr *LHStex = DeclRefExpr::Create(Ctx,
      LHS->getQualifierLoc(),
      LHS->getTemplateKeywordLoc(),
      CloneDeclTex(PVD, (memAcc==READ_ONLY)?"_tex":"_surf"),
      LHS->refersToEnclosingLocal(),
      LHS->getLocation(),
      LHS->getType(), LHS->getValueKind(),
      LHS->getFoundDecl(),
      LHS->getNumTemplateArgs()?&templateArgs:0);

  LHStex->setObjectKind(LHS->getObjectKind());
  LHStex->setValueDependent(LHS->isValueDependent());
  LHStex->setTypeDependent(LHS->isTypeDependent());

  // parameters for tex1Dfetch, tex2D, or surf2Dwrite
  llvm::SmallVector<Expr *, 16> args;

  if (memAcc == READ_ONLY) {
    args.push_back(LHStex);
    switch (Kernel->useTextureMemory(Acc)) {
      default:
      case Linear1D:
        args.push_back(createBinaryOperator(Ctx, createBinaryOperator(Ctx,
                createParenExpr(Ctx, idx_y), Acc->getStrideDecl(), BO_Mul,
                Ctx.IntTy), idx_x, BO_Add, Ctx.IntTy));
        break;
      case Linear2D:
      case Array2D:
        args.push_back(idx_x);
        args.push_back(idx_y);
        break;
    }
  } else {
    // writeImageRHS is set by VisitBinaryOperator - side effect
    writeImageRHS = createParenExpr(Ctx, writeImageRHS);
    args.push_back(writeImageRHS);
    args.push_back(LHStex);
    // byte addressing required for surf2Dwrite
    args.push_back(createBinaryOperator(Ctx, idx_x, createIntegerLiteral(Ctx,
            (int)Acc->getImage()->getPixelSize()), BO_Mul, Ctx.IntTy));
    args.push_back(idx_y);
  }

  return createFunctionCall(Ctx, texture_function, args); 
}


// access image memory at given index
Expr *ASTTranslate::accessMemImgAt(DeclRefExpr *LHS, HipaccAccessor *Acc,
    MemoryAccess memAcc, Expr *idx_x, Expr *idx_y) {
  Expr *result, *coord;

  // construct coordinate: (int2)(gid_x, gid_y)
  coord = createBinaryOperator(Ctx, idx_x, idx_y, BO_Comma, Ctx.IntTy);
  coord = createParenExpr(Ctx, coord);
  coord = createCStyleCastExpr(Ctx, simdTypes.getSIMDType(Ctx.IntTy, "int",
        SIMD2), CK_VectorSplat, coord, NULL, NULL);

  FunctionDecl *image_function = getImageFunction(Acc, memAcc);

  // create function call for image objects in OpenCL
  if (memAcc == READ_ONLY) {
    // parameters for read_image
    llvm::SmallVector<Expr *, 16> args;
    args.push_back(LHS);
    args.push_back(kernelSamplerRef);
    args.push_back(coord);

    result = createFunctionCall(Ctx, image_function, args); 
    result = createExtVectorElementExpr(Ctx,
        Acc->getImage()->getPixelQualType(), result, "x");
  } else {
    QualType QT;
    // determine cast type for write_image functions
    if (image_function == builtins.getBuiltinFunction(OPENCLBIwrite_imagei)) {
      QT = simdTypes.getSIMDType(Ctx.IntTy, "int", SIMD4);
    } else if (image_function ==
        builtins.getBuiltinFunction(OPENCLBIwrite_imageui)) { 
      QT = simdTypes.getSIMDType(Ctx.UnsignedIntTy, "uint", SIMD4);
    } else {
      QT = simdTypes.getSIMDType(Ctx.FloatTy, "float", SIMD4);
    }

    // writeImageRHS is set by VisitBinaryOperator - side effect
    writeImageRHS = createParenExpr(Ctx, writeImageRHS);
    writeImageRHS = createCStyleCastExpr(Ctx, QT, CK_VectorSplat, writeImageRHS,
        NULL, NULL);
    // parameters for write_image
    llvm::SmallVector<Expr *, 16> args;
    args.push_back(LHS);
    args.push_back(coord);
    args.push_back(writeImageRHS);

    result = createFunctionCall(Ctx, image_function, args); 
  }

  return result;
}


// access shared memory
Expr *ASTTranslate::accessMemShared(DeclRefExpr *LHS, Expr *local_offset_x, Expr
    *local_offset_y) {
  Expr *idx_x = lidXRef;
  Expr *idx_y = lidYRef;

  // step 0: add local offset: lid_[x|y] + local_offset_[x|y]
  idx_x = addLocalOffset(idx_x, local_offset_x);
  idx_y = addLocalOffset(idx_y, local_offset_y);

  return accessMemSharedAt(LHS, idx_x, idx_y);
}


// access shared memory at given index
Expr *ASTTranslate::accessMemSharedAt(DeclRefExpr *LHS, Expr *idx_x, Expr
    *idx_y) {
  Expr *result;

  QualType QT =
    LHS->getType()->castAsArrayTypeUnsafe()->getElementType()->castAsArrayTypeUnsafe()->getElementType();
  QualType QT2 = LHS->getType()->castAsArrayTypeUnsafe()->getElementType();

  // calculate index: [idx_y][idx_x]
  result = new (Ctx) ArraySubscriptExpr(createImplicitCastExpr(Ctx, QT2,
        CK_LValueToRValue, LHS, NULL, VK_RValue), idx_y,
      QT2.getTypePtr()->getPointeeType(), VK_LValue, OK_Ordinary,
      SourceLocation());

  result = new (Ctx) ArraySubscriptExpr(createImplicitCastExpr(Ctx,
        Ctx.getPointerType(QT), CK_ArrayToPointerDecay, result, NULL,
        VK_RValue), idx_x, QT, VK_LValue, OK_Ordinary, SourceLocation());

  return result;
}


// stage single image line (warp size) to shared memory
void ASTTranslate::stageLineToSharedMemory(ParmVarDecl *PVD,
    llvm::SmallVector<Stmt *, 16> &stageBody, Expr *local_offset_x, Expr
    *local_offset_y, Expr *global_offset_x, Expr *global_offset_y) {
  VarDecl *VD = KernelDeclMapShared[PVD];
  HipaccAccessor *Acc = KernelDeclMapAcc[PVD];
  DeclRefExpr *paramDRE = createDeclRefExpr(Ctx, PVD);

  Expr *LHS = accessMemShared(createDeclRefExpr(Ctx, VD), local_offset_x,
      local_offset_y);

  Expr *RHS;
  if (Acc->getBoundaryHandling()!=BOUNDARY_UNDEFINED && bh_variant.borderVal) {
    llvm::SmallVector<Stmt *, 16> bhStmts;
    llvm::SmallVector<CompoundStmt *, 16> bhCStmts;
    RHS = addBorderHandling(paramDRE, global_offset_x, global_offset_y, Acc,
        bhStmts, bhCStmts);

    // add border handling statements to ifBody/stageBody
    for (unsigned int i=0, e=bhStmts.size(); i!=e; ++i) {
      stageBody.push_back(bhStmts.data()[i]);
    }
  } else {
    RHS = accessMem(paramDRE, Acc, READ_ONLY, global_offset_x, global_offset_y);
  }

  stageBody.push_back(createBinaryOperator(Ctx, LHS, RHS, BO_Assign,
        Acc->getImage()->getPixelQualType()));
}


// stage iteration p to shared memory
void ASTTranslate::stageIterationToSharedMemory(llvm::SmallVector<Stmt *, 16>
    &stageBody, int p) {
  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      N=kernelDecl->param_end(); I!=N; ++I) {
    ParmVarDecl *PVD = *I;

    if (KernelDeclMapShared[PVD]) {
      HipaccAccessor *Acc = KernelDeclMapAcc[PVD];

      // check if the bottom apron has to be fetched
      if (p==(int)Kernel->getPixelsPerThread() && Acc->getSizeY() <= 1)
        continue;

      Expr *global_offset_x, *global_offset_y;
      IntegerLiteral *SX, *SY, *SX2, *SY2;
      llvm::SmallVector<Stmt *, 16> ifBody;

      if (Acc->getSizeX() > 1) {
        SX = createIntegerLiteral(Ctx, (int)Kernel->max_threads_per_warp);
        SX2 = createIntegerLiteral(Ctx, (int)Kernel->max_threads_per_warp);
        global_offset_x = createParenExpr(Ctx, createUnaryOperator(Ctx, SX2,
              UO_Minus, Ctx.IntTy));
      } else {
        SX = createIntegerLiteral(Ctx, 0);
        SX2 = createIntegerLiteral(Ctx, 0);
        global_offset_x = NULL;
      }
      if (Acc->getSizeY() > 1) {
        SY = createIntegerLiteral(Ctx, (int)Acc->getSizeY());
        SY2 = createIntegerLiteral(Ctx, (int)Acc->getSizeY()/2);
        global_offset_y = createParenExpr(Ctx, createUnaryOperator(Ctx, SY2,
              UO_Minus, Ctx.IntTy));
      } else {
        SY = createIntegerLiteral(Ctx, 0);
        SY2 = createIntegerLiteral(Ctx, 0);
        global_offset_y = NULL;
      }


      // check if we need to stage right apron
      int num_stages_x = 0;
      if (Acc->getSizeX() > 1) {
          num_stages_x = 2;
      }

      // load row (line)
      for (int i=0; i<=num_stages_x; i++) {
        // _smem[lidYRef][lidXRef + i*blockDim.x] =
        //        Image[-SX/2 + i*blockDim.x, -SY/2];
        Expr *local_offset_x = NULL, *global_offset_x = NULL;
        if (Acc->getSizeX() > 1) {
          local_offset_x = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                i), local_size_x, BO_Mul, Ctx.IntTy);
          global_offset_x = createBinaryOperator(Ctx, local_offset_x, SX2,
              BO_Sub, Ctx.IntTy);
        }

        stageLineToSharedMemory(PVD, stageBody, local_offset_x, NULL,
            global_offset_x, global_offset_y);
      }
    }
  }
}

// vim: set ts=2 sw=2 sts=2 et ai:

