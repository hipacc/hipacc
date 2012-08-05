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
      if (Kernel->useTextureMemory(Acc) && !compilerOptions.emitCUDA()) {
        return accessMemImgAt(LHS, Acc, memAcc, idx_x, idx_y);
      } else {
        return accessMemArrAt(LHS, Acc->getStrideDecl(), idx_x, idx_y);
      }
      break;
    case READ_ONLY:
      if (Kernel->useTextureMemory(Acc)) {
        if (compilerOptions.emitCUDA()) {
          return accessMemTexAt(LHS, Acc, idx_x, idx_y);
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
FunctionDecl *ASTTranslate::getTex1DFetchFunction(HipaccAccessor *Acc) {
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
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1DfetchSc);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2DSc);
      }
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1DfetchUc);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2DUc);
      }
    case BuiltinType::Short:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1Dfetchs);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2Ds);
      }
    case BuiltinType::Char16:
    case BuiltinType::UShort:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1DfetchUs);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2DUs);
      }
    case BuiltinType::Int:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
        case HipaccKernelFeatures::Linear2D:
          return builtins.getBuiltinFunction(CUDABItex1Dfetchi);
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2Di);
      }
    case BuiltinType::Char32:
    case BuiltinType::UInt:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1DfetchUi);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2DUi);
      }
    case BuiltinType::Float:
      switch (Kernel->useTextureMemory(Acc)) {
        default:
        case HipaccKernelFeatures::Linear1D:
          return builtins.getBuiltinFunction(CUDABItex1Dfetchf);
        case HipaccKernelFeatures::Linear2D:
        case HipaccKernelFeatures::Array2D:
          return builtins.getBuiltinFunction(CUDABItex2Df);
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
Expr *ASTTranslate::accessMemTexAt(DeclRefExpr *LHS, HipaccAccessor *Acc, Expr
    *idx_x, Expr *idx_y) {
  Expr *result = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
        createParenExpr(Ctx, idx_y), Acc->getStrideDecl(), BO_Mul, Ctx.IntTy),
      idx_x, BO_Add, Ctx.IntTy);

  FunctionDecl *tex1Dfetch = getTex1DFetchFunction(Acc);

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
      CloneDeclTex(PVD),
      LHS->refersToEnclosingLocal(),
      LHS->getLocation(),
      LHS->getType(), LHS->getValueKind(),
      LHS->getFoundDecl(),
      LHS->getNumTemplateArgs()?&templateArgs:0);

  LHStex->setObjectKind(LHS->getObjectKind());
  LHStex->setValueDependent(LHS->isValueDependent());
  LHStex->setTypeDependent(LHS->isTypeDependent());

  // parameters for tex1Dfetch
  llvm::SmallVector<Expr *, 16> args;
  args.push_back(LHStex);
  switch (Kernel->useTextureMemory(Acc)) {
    default:
    case HipaccKernelFeatures::Linear1D:
      args.push_back(result);
      break;
    case HipaccKernelFeatures::Linear2D:
    case HipaccKernelFeatures::Array2D:
      args.push_back(idx_x);
      args.push_back(idx_y);
      break;
  }

  result = createFunctionCall(Ctx, tex1Dfetch, args); 

  return result;
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


// stage single image line to shared memory
void ASTTranslate::stage_line_to_shared_memory(ParmVarDecl *PVD,
    llvm::SmallVector<Stmt *, 16> &stageBody, Expr *local_offset_x, Expr
    *local_offset_y, Expr *global_offset_x, Expr *global_offset_y) {
  VarDecl *VD = KernelDeclMapShared[PVD];
  HipaccAccessor *Acc = KernelDeclMapAcc[PVD];
  DeclRefExpr *paramDRE = createDeclRefExpr(Ctx, PVD);

  Expr *LHS = accessMemShared(createDeclRefExpr(Ctx, VD), local_offset_x,
      local_offset_y);

  Expr *RHS;
  if (Acc->getBoundaryHandling() != BOUNDARY_UNDEFINED) {
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


// stage first iteration to shared memory
bool ASTTranslate::stage_first_iteration_to_shared_memory(llvm::SmallVector<Stmt
    *, 16> &stageBody) {
  bool found = false;

  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      N=kernelDecl->param_end(); I!=N; ++I) {
    ParmVarDecl *PVD = *I;

    if (KernelDeclMapShared[PVD]) {
      found = true;
      HipaccAccessor *Acc = KernelDeclMapAcc[PVD];

      Expr *global_offset_x, *global_offset_y;
      IntegerLiteral *SX, *SY, *SX2, *SY2;
      llvm::SmallVector<Stmt *, 16> ifBody;
      BinaryOperator *cond_x = NULL, *cond_y = NULL;

      if (Acc->getSizeX() > 1) {
        SX = createIntegerLiteral(Ctx, (int)Acc->getSizeX());
        SX2 = createIntegerLiteral(Ctx, (int)Acc->getSizeX()/2);
        global_offset_x = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
              0), SX2, BO_Sub, Ctx.IntTy);
      } else {
        SX = createIntegerLiteral(Ctx, 0);
        SX2 = createIntegerLiteral(Ctx, 0);
        global_offset_x = NULL;
      }
      if (Acc->getSizeY() > 1) {
        SY = createIntegerLiteral(Ctx, (int)Acc->getSizeY());
        SY2 = createIntegerLiteral(Ctx, (int)Acc->getSizeY()/2);
        global_offset_y = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
              0), SY2, BO_Sub, Ctx.IntTy);
      } else {
        SY = createIntegerLiteral(Ctx, 0);
        SY2 = createIntegerLiteral(Ctx, 0);
        global_offset_y = NULL;
      }


      // load first tile
      if (Kernel->getNumThreadsY() > 1) {
        // _smem[lidYRef][lidXRef] = Image[-SX/2, -SY/2];
        stage_line_to_shared_memory(PVD, ifBody, NULL, NULL, global_offset_x,
            global_offset_y);

        // check if the index is still within the iteration space in case we
        // have a tiling with multiple threads in the y-dimension
        cond_y = createBinaryOperator(Ctx, gidYRef, isHeight, BO_LT,
            Ctx.BoolTy);
        stageBody.push_back(createIfStmt(Ctx, cond_y, createCompoundStmt(Ctx,
                ifBody)));
        ifBody.clear();
      } else {
        // _smem[lidYRef][lidXRef] = Image[-SX/2, -SY/2];
        stage_line_to_shared_memory(PVD, stageBody, NULL, NULL, global_offset_x,
            global_offset_y);
      }

      // check if we need to stage right apron
      if (Acc->getSizeX() > 1) {
        int num_stages =
          (int)ceil((float)(Acc->getSizeX()-1)/Kernel->getNumThreadsX());
        for (int i=1; i<=num_stages; i++) {
          // if (lidx + i*blockDim.x < blockDim.x + SX-1)
          //      _smem[lidYRef][lidXRef + i*blockDim.x] =
          //          Image[-SX/2 + i*blockDim.x, -SY/2];
          Expr *local_offset_x = createBinaryOperator(Ctx,
              createIntegerLiteral(Ctx, i), local_size_x, BO_Mul, Ctx.IntTy);
          Expr *global_offset_x = createBinaryOperator(Ctx, local_offset_x, SX2,
              BO_Sub, Ctx.IntTy);

          stage_line_to_shared_memory(PVD, ifBody, local_offset_x, NULL,
              global_offset_x, global_offset_y);

          // lidx + offset_x < blockDim.x + SX-1
          cond_x = createBinaryOperator(Ctx, createBinaryOperator(Ctx, lidXRef,
                local_offset_x, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx,
                  createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1),
                    BO_Sub, Ctx.IntTy), local_size_x, BO_Add, Ctx.IntTy), BO_LT,
              Ctx.BoolTy);

          stageBody.push_back(createIfStmt(Ctx, cond_x, createCompoundStmt(Ctx,
                  ifBody)));
          ifBody.clear();
        }
      }

      // check if we need to stage bottom apron
      if (Acc->getSizeY() > 1) {
        int num_stages =
          (int)ceil((float)(Acc->getSizeY()-1)/Kernel->getNumThreadsY());
        for (int i=1; i<=num_stages; i++) {
          // if (lidy + i*blockDim.y < blockDim.y + SY-1)
          //      _smem[lidYRef + i*blockDim.y][lidXRef] =
          //          Image(-SX/2, i*blockDim.y - SY/2);
          Expr *local_offset_y = createBinaryOperator(Ctx,
              createIntegerLiteral(Ctx, i), local_size_y, BO_Mul, Ctx.IntTy);
          Expr *global_offset_y = createBinaryOperator(Ctx, local_offset_y, SY2,
              BO_Sub, Ctx.IntTy);

          stage_line_to_shared_memory(PVD, ifBody, NULL, local_offset_y,
              global_offset_x, global_offset_y);

          // lidy + offset_y < blockDim.y + SY-1
          cond_y = createBinaryOperator(Ctx, createBinaryOperator(Ctx, lidYRef,
                local_offset_y, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx,
                  createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx, 1),
                    BO_Sub, Ctx.IntTy), local_size_y, BO_Add, Ctx.IntTy), BO_LT,
              Ctx.BoolTy);

          // check if the index is still within the iteration space in case we
          // have a tiling with multiple threads in the y-dimension
          if (Kernel->getNumThreadsY() > 1) {
            cond_y = createBinaryOperator(Ctx, cond_y, createBinaryOperator(Ctx,
                  createBinaryOperator(Ctx, gidYRef, global_offset_y, BO_Add,
                    Ctx.IntTy), isHeight, BO_LT, Ctx.BoolTy), BO_LAnd,
                Ctx.BoolTy);
          }

          stageBody.push_back(createIfStmt(Ctx, cond_y, createCompoundStmt(Ctx,
                  ifBody)));
          ifBody.clear();
        }
      }

      // check if we need to stage bottom/right apron
      if (Acc->getSizeX() > 1 && Acc->getSizeY() > 1) {
        int num_stages_x =
          (int)ceil((float)(Acc->getSizeX()-1)/Kernel->getNumThreadsX());
        int num_stages_y =
          (int)ceil((float)(Acc->getSizeY()-1)/Kernel->getNumThreadsY());
        for (int j=1; j<=num_stages_y; j++) {
          Expr *local_offset_y = createBinaryOperator(Ctx,
              createIntegerLiteral(Ctx, j), local_size_y, BO_Mul, Ctx.IntTy);
          Expr *global_offset_y = createBinaryOperator(Ctx, local_offset_y, SY2,
              BO_Sub, Ctx.IntTy);
          for (int i=1; i<=num_stages_x; i++) {
            // if (lidx + i*blockDim.x < blockDim.x + SX-1 &&
            //     lidy + j*blockDim.y < blockDim.y + SY-1)
            //      _smem[lidYRef + j*blockDim.y][lidXRef + i*blockDim.x] =
            //          Image(i*blockDim.x - SX/2, j*blockDim.y - SY/2);
            Expr *local_offset_x = createBinaryOperator(Ctx,
                createIntegerLiteral(Ctx, i), local_size_x, BO_Mul, Ctx.IntTy);
            Expr *global_offset_x = createBinaryOperator(Ctx, local_offset_x,
                SX2, BO_Sub, Ctx.IntTy);

            stage_line_to_shared_memory(PVD, ifBody, local_offset_x,
                local_offset_y, global_offset_x, global_offset_y);

            // lidx + offset_x < blockDim.x + SX-1
            cond_x = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
                  lidXRef, local_offset_x, BO_Add, Ctx.IntTy),
                createBinaryOperator(Ctx, createBinaryOperator(Ctx, SX,
                    createIntegerLiteral(Ctx, 1), BO_Sub, Ctx.IntTy),
                  local_size_x, BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);

            // lidy + offset_y < blockDim.y + SY-1
            cond_y = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
                  lidYRef, local_offset_y, BO_Add, Ctx.IntTy),
                createBinaryOperator(Ctx, createBinaryOperator(Ctx, SY,
                    createIntegerLiteral(Ctx, 1), BO_Sub, Ctx.IntTy),
                  local_size_y, BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);

            // check if the index is still within the iteration space in case we
            // have a tiling with multiple threads in the y-dimension
            if (Kernel->getNumThreadsY() > 1) {
              cond_y = createBinaryOperator(Ctx, cond_y,
                  createBinaryOperator(Ctx, createBinaryOperator(Ctx, gidYRef,
                      global_offset_y, BO_Add, Ctx.IntTy), isHeight, BO_LT,
                    Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
            }

            // cond_x && cond_y
            stageBody.push_back(createIfStmt(Ctx, createBinaryOperator(Ctx,
                    cond_x, cond_y, BO_LAnd, Ctx.BoolTy),
                  createCompoundStmt(Ctx, ifBody)));
            ifBody.clear();
          }
        }
      }
    }
  }

  return found;
}


// stage next iteration to shared memory
bool ASTTranslate::stage_next_iteration_to_shared_memory(llvm::SmallVector<Stmt
    *, 16> &stageBody) {
  bool found = false;

  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      N=kernelDecl->param_end(); I!=N; ++I) {
    ParmVarDecl *PVD = *I;

    if (KernelDeclMapShared[PVD]) {
      found = true;
      HipaccAccessor *Acc = KernelDeclMapAcc[PVD];

      IntegerLiteral *SX, *SY, *SX2, *SY2;
      if (Acc->getSizeX() > 1) {
        SX = createIntegerLiteral(Ctx, (int)Acc->getSizeX());
        SX2 = createIntegerLiteral(Ctx, (int)Acc->getSizeX()/2);
      } else {
        SX = createIntegerLiteral(Ctx, 0);
        SX2 = createIntegerLiteral(Ctx, 0);
      }
      if (Acc->getSizeY() > 1) {
        SY = createIntegerLiteral(Ctx, (int)Acc->getSizeY());
        SY2 = createIntegerLiteral(Ctx, (int)Acc->getSizeY()/2);
      } else {
        SY = createIntegerLiteral(Ctx, 0);
        SY2 = createIntegerLiteral(Ctx, 0);
      }

      // load next line to shared memory
      // if (lidx + i*blockDim.x < blockDim.x + SX-1)
      //      _smem[lidYRef + 2*SY/2 + (1+PPT)*blockDim.y]
      //           [lidXRef + i*blockDim.x] =
      //          Image(i*blockDim.x - SX/2,(1+PPT)*blockDim.y + SY/2);
      int num_stages_x = 0;
      if (Acc->getSizeX() > 1) {
        num_stages_x =
          (int)ceil((float)(Acc->getSizeX()-1)/Kernel->getNumThreadsX());
      }

      // 1*blockDim.y
      Expr *local_offset_y = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
            1), local_size_y, BO_Mul, Ctx.IntTy);
      // 1*blockDim.y + SY2
      Expr *global_offset_y = createBinaryOperator(Ctx, local_offset_y, SY2,
          BO_Add, Ctx.IntTy);
      // += 2*SY2
      local_offset_y = createBinaryOperator(Ctx, local_offset_y,
          createBinaryOperator(Ctx, SY2, SY2, BO_Add, Ctx.IntTy), BO_Add,
          Ctx.IntTy);

      // check if the index is still within the iteration space in case we
      // have a tiling with multiple threads in the y-dimension
      BinaryOperator *cond_y = NULL;
      if (Kernel->getNumThreadsY() > 1 || Kernel->getPixelsPerThread() > 1) {
        cond_y = createBinaryOperator(Ctx, createBinaryOperator(Ctx, gidYRef,
              global_offset_y, BO_Add, Ctx.IntTy), isHeight, BO_LT, Ctx.BoolTy);
      }

      for (int i=0; i<=num_stages_x; i++) {
        Expr *local_offset_x = NULL;
        Expr *global_offset_x = NULL;
        BinaryOperator *cond = NULL;
        llvm::SmallVector<Stmt *, 16> ifBody;

        if (i!=1) {
          local_offset_x = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                i), local_size_x, BO_Mul, Ctx.IntTy);
          global_offset_x = createBinaryOperator(Ctx, local_offset_x, SX2,
              BO_Sub, Ctx.IntTy);

          // lidx + offset_x < blockDim.x + SX-1
          cond = createBinaryOperator(Ctx, createBinaryOperator(Ctx, lidXRef,
                local_offset_x, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx,
                  createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1),
                    BO_Sub, Ctx.IntTy), local_size_x, BO_Add, Ctx.IntTy), BO_LT,
              Ctx.BoolTy);
          if (cond_y) {
            cond = createBinaryOperator(Ctx, cond, cond_y, BO_LAnd, Ctx.BoolTy);
          }

          stage_line_to_shared_memory(PVD, ifBody, local_offset_x,
              local_offset_y, global_offset_x, global_offset_y);
          stageBody.push_back(createIfStmt(Ctx, cond, createCompoundStmt(Ctx,
                  ifBody)));
        } else {
          stage_line_to_shared_memory(PVD, stageBody, local_offset_x,
              local_offset_y, global_offset_x, global_offset_y);
        }
      }
    }
  }

  return found;
}


// update shared memory for next iteration
bool ASTTranslate::update_shared_memory_for_next_iteration(
    llvm::SmallVector<Stmt *, 16> &stageBody) {
  bool found = false;

  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      N=kernelDecl->param_end(); I!=N; ++I) {
    ParmVarDecl *PVD = *I;

    if (KernelDeclMapShared[PVD]) {
      found = true;
      HipaccAccessor *Acc = KernelDeclMapAcc[PVD];
      VarDecl *VD = KernelDeclMapShared[PVD];

      IntegerLiteral *SX, *SX2;
      if (Acc->getSizeX() > 1) {
        SX = createIntegerLiteral(Ctx, (int)Acc->getSizeX());
        SX2 = createIntegerLiteral(Ctx, (int)Acc->getSizeX()/2);
      } else {
        SX = createIntegerLiteral(Ctx, 0);
        SX2 = createIntegerLiteral(Ctx, 0);
      }

      // update shared memory locations for next iteration
      // if (lidx + i*blockDim.x < blockDim.x + SX-1)
      //      _smem[lidYRef + j*blockDim.y] [lidXRef + i*blockDim.x] =
      //      _smem[lidYRef + (j+1)*blockDim.y] [lidXRef + i*blockDim.x];
      int num_stages_x = 0, num_stages_y = 0;
      if (Acc->getSizeX() > 1) {
        num_stages_x =
          (int)ceil((float)(Acc->getSizeX()-1)/Kernel->getNumThreadsX());
      }
      if (Acc->getSizeY() > 1) {
        num_stages_y =
          (int)ceil((float)(Acc->getSizeY()-1)/Kernel->getNumThreadsY());
      }

      for (int j=0; j<=num_stages_y; j++) {
        Expr *local_offset_y_left = createBinaryOperator(Ctx,
            createIntegerLiteral(Ctx, j), local_size_y, BO_Mul, Ctx.IntTy);
        Expr *local_offset_y_right = createBinaryOperator(Ctx,
            createIntegerLiteral(Ctx, j+1), local_size_y, BO_Mul, Ctx.IntTy);
        for (int i=0; i<=num_stages_x; i++) {
          Expr *local_offset_x = NULL;
          BinaryOperator *cond = NULL;

          if (i!=0) {
            local_offset_x = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                  i), local_size_x, BO_Mul, Ctx.IntTy);

            // lidx + offset_x < blockDim.x + SX-1
            cond = createBinaryOperator(Ctx, createBinaryOperator(Ctx, lidXRef,
                  local_offset_x, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx,
                    createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1),
                      BO_Sub, Ctx.IntTy), local_size_x, BO_Add, Ctx.IntTy),
                BO_LT, Ctx.BoolTy);

            Expr *LHS = accessMemShared(createDeclRefExpr(Ctx, VD),
                local_offset_x, local_offset_y_left);
            Expr *RHS = accessMemShared(createDeclRefExpr(Ctx, VD),
                local_offset_x, local_offset_y_right);
            stageBody.push_back(createIfStmt(Ctx, cond,
                  createBinaryOperator(Ctx, LHS, RHS, BO_Assign,
                    Acc->getImage()->getPixelQualType())));
          } else {
            Expr *LHS = accessMemShared(createDeclRefExpr(Ctx, VD),
                local_offset_x, local_offset_y_left);
            Expr *RHS = accessMemShared(createDeclRefExpr(Ctx, VD),
                local_offset_x, local_offset_y_right);
            stageBody.push_back(createBinaryOperator(Ctx, LHS, RHS, BO_Assign,
                  Acc->getImage()->getPixelQualType()));
          }
        }
      }
    }
  }

  return found;
}

// vim: set ts=2 sw=2 sts=2 et ai:

