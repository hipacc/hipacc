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

//===--- ASTTranslate.cpp - C to CL Translation of the AST ----------------===//
//
// This file implements translation of statements and expressions.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;
using namespace hipacc::Builtin;


//===----------------------------------------------------------------------===//
// Statement/expression transformations
//===----------------------------------------------------------------------===//


// C/Polly initialization
void ASTTranslate::initC(SmallVector<Stmt *, 16> &kernelBody, Stmt *S) {
  VarDecl *gid_x = NULL, *gid_y = NULL;

  // Polly: int gid_x = offset_x;
  if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
        getOffsetXDecl(Kernel->getIterationSpace()->getAccessor()));
  } else {
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
        createIntegerLiteral(Ctx, 0));
  }

  // Polly: int gid_y = offset_y;
  if (Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl()) {
    gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.IntTy,
        getOffsetYDecl(Kernel->getIterationSpace()->getAccessor()));
  } else {
    gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.IntTy,
        createIntegerLiteral(Ctx, 0));
  }

  // add gid_x and gid_y statements
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);
  DeclStmt *gid_x_stmt = createDeclStmt(Ctx, gid_x);
  DeclStmt *gid_y_stmt = createDeclStmt(Ctx, gid_y);

  tileVars.global_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.global_id_y = createDeclRefExpr(Ctx, gid_y);
  gidXRef = tileVars.global_id_x;
  gidYRef = tileVars.global_id_y;

  // convert the function body to kernel syntax
  Stmt *clonedStmt = Clone(S);
  assert(isa<CompoundStmt>(clonedStmt) && "CompoundStmt for kernel function body expected!");

  //
  // for (int gid_y=offset_y; gid_y<2048; gid_y++) {
  //     for (int gid_x=offset_x; gid_x<4096; gid_x++) {
  //         body
  //     }
  // }
  //
  ForStmt *innerLoop = createForStmt(Ctx, gid_x_stmt, createBinaryOperator(Ctx,
        tileVars.global_id_x, createIntegerLiteral(Ctx, 4096), BO_LT,
        Ctx.BoolTy), createUnaryOperator(Ctx, tileVars.global_id_x, UO_PostInc,
          tileVars.global_id_x->getType()), clonedStmt);
  ForStmt *outerLoop = createForStmt(Ctx, gid_y_stmt, createBinaryOperator(Ctx,
        tileVars.global_id_y, createIntegerLiteral(Ctx, 2048), BO_LT,
        Ctx.BoolTy), createUnaryOperator(Ctx, tileVars.global_id_y, UO_PostInc,
          tileVars.global_id_y->getType()), innerLoop);

  kernelBody.push_back(outerLoop);
}


// CUDA initialization
void ASTTranslate::initCUDA(SmallVector<Stmt *, 16> &kernelBody) {
  VarDecl *gid_x = NULL, *gid_y = NULL;
  SmallVector<QualType, 16> uintDeclTypes;
  SmallVector<StringRef, 16> uintDeclNames;
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclNames.push_back("x");
  uintDeclNames.push_back("y");
  uintDeclNames.push_back("z");

  // CUDA
  /*DEVICE_BUILTIN*/
  //struct uint3
  //{
  //  unsigned int x, y, z;
  //};
  RecordDecl *uint3RD = createRecordDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "uint3", TTK_Struct, uintDeclTypes.size(), uintDeclTypes.data(),
      uintDeclNames.data());

  /*DEVICE_BUILTIN*/
  //typedef struct uint3 uint3;

  /*DEVICE_BUILTIN*/
  //struct dim3
  //{
  //    unsigned int x, y, z;
  //};

  /*DEVICE_BUILTIN*/
  //typedef struct dim3 dim3;

  //uint3 threadIdx;
  VarDecl *threadIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "threadIdx", Ctx.getTypeDeclType(uint3RD), NULL);
  //uint3 blockIdx;
  VarDecl *blockIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockIdx", Ctx.getTypeDeclType(uint3RD), NULL);
  //dim3 blockDim;
  VarDecl *blockDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockDim", Ctx.getTypeDeclType(uint3RD), NULL);
  //dim3 gridDim;
  //VarDecl *gridDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "gridDim", Ctx.getTypeDeclType(uint3RD), NULL);
  //int warpSize;
  //VarDecl *warpSize = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "warpSize", Ctx.IntTy, NULL);

  DeclRefExpr *TIRef = createDeclRefExpr(Ctx, threadIdx);
  DeclRefExpr *BIRef = createDeclRefExpr(Ctx, blockIdx);
  DeclRefExpr *BDRef = createDeclRefExpr(Ctx, blockDim);
  VarDecl *xVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "x",
      Ctx.IntTy, NULL);
  VarDecl *yVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "y",
      Ctx.IntTy, NULL);

  tileVars.local_id_x = createMemberExpr(Ctx, TIRef, false, xVD,
      xVD->getType());
  tileVars.local_id_y = createMemberExpr(Ctx, TIRef, false, yVD,
      yVD->getType());
  tileVars.block_id_x = createMemberExpr(Ctx, BIRef, false, xVD,
      xVD->getType());
  tileVars.block_id_y = createMemberExpr(Ctx, BIRef, false, yVD,
      yVD->getType());
  tileVars.local_size_x = createMemberExpr(Ctx, BDRef, false, xVD,
      xVD->getType());
  tileVars.local_size_y = createMemberExpr(Ctx, BDRef, false, yVD,
      yVD->getType());
  //DeclRefExpr *GDRef = createDeclRefExpr(Ctx, gridDim);
  //tileVars.grid_size_x = createMemberExpr(Ctx, GDRef, false, xVD,
  //    xVD->getType());
  //tileVars.grid_size_y = createMemberExpr(Ctx, GDRef, false, yVD,
  //    yVD->getType());

  // CUDA: const int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
  gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, createBinaryOperator(Ctx, tileVars.local_size_x,
          tileVars.block_id_x, BO_Mul, Ctx.IntTy), tileVars.local_id_x, BO_Add,
        Ctx.IntTy));

  // CUDA: const int gid_y = blockDim.y*PPT*blockIdx.y + threadIdx.y;
  Expr *YE = createBinaryOperator(Ctx, tileVars.local_size_y,
      tileVars.block_id_y, BO_Mul, Ctx.IntTy);
  if (Kernel->getPixelsPerThread() > 1) {
    YE = createBinaryOperator(Ctx, YE, createIntegerLiteral(Ctx,
          (int)Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy);
  }
  gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, YE, tileVars.local_id_y, BO_Add, Ctx.IntTy));

  // add gid_x and gid_y statements
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);
  kernelBody.push_back(createDeclStmt(Ctx, gid_x));
  kernelBody.push_back(createDeclStmt(Ctx, gid_y));

  tileVars.global_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.global_id_y = createDeclRefExpr(Ctx, gid_y);
}


// OpenCL initialization
void ASTTranslate::initOpenCL(SmallVector<Stmt *, 16> &kernelBody) {
  VarDecl *gid_x = NULL, *gid_y = NULL;
  // uint get_work_dim();
  //FunctionDecl *get_work_dim =
  //  builtins.getBuiltinFunction(OPENCLBIget_work_dim);
  // size_t get_global_size(uint dimindx);
  //FunctionDecl *get_global_size =
  //  builtins.getBuiltinFunction(OPENCLBIget_global_size);
  //size_t get_global_id(uint dimindx);
  FunctionDecl *get_global_id =
    builtins.getBuiltinFunction(OPENCLBIget_global_id);
  //size_t get_local_size(uint dimindx);
  FunctionDecl *get_local_size =
    builtins.getBuiltinFunction(OPENCLBIget_local_size);
  //size_t get_local_id(uint dimindx);
  FunctionDecl *get_local_id =
    builtins.getBuiltinFunction(OPENCLBIget_local_id);
  //size_t get_num_groups(uint dimindx);
  //FunctionDecl *get_num_groups =
  //  builtins.getBuiltinFunction(OPENCLBIget_num_groups);
  //size_t get_group_id(uint dimindx);
  FunctionDecl *get_group_id =
    builtins.getBuiltinFunction(OPENCLBIget_group_id);

  // .(0) .(1)
  SmallVector<Expr *, 16> tmpArg0;
  SmallVector<Expr *, 16> tmpArg1;
  tmpArg0.push_back(createIntegerLiteral(Ctx, 0));
  tmpArg1.push_back(createIntegerLiteral(Ctx, 1));
  //ImplicitCastExpr *get_global_size0, *get_global_size1;
  //get_global_size0 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg0), NULL,
  //    VK_RValue);
  //get_global_size1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg1), NULL,
  //    VK_RValue);
  ImplicitCastExpr *get_global_id0, *get_global_id1;
  get_global_id0 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg0), NULL,
      VK_RValue);
  get_global_id1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg1), NULL,
      VK_RValue);
  tileVars.local_size_x = createImplicitCastExpr(Ctx,
      Ctx.getConstType(Ctx.IntTy), CK_IntegralCast, createFunctionCall(Ctx,
        get_local_size, tmpArg0), NULL, VK_RValue);
  tileVars.local_size_y = createImplicitCastExpr(Ctx,
      Ctx.getConstType(Ctx.IntTy), CK_IntegralCast, createFunctionCall(Ctx,
        get_local_size, tmpArg1), NULL, VK_RValue);
  tileVars.local_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg0), NULL,
      VK_RValue);
  tileVars.local_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg1), NULL,
      VK_RValue);
  tileVars.block_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg0), NULL,
      VK_RValue);
  tileVars.block_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg1), NULL,
      VK_RValue);
  //grid_size_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg0), NULL,
  //    VK_RValue);
  //grid_size_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg1), NULL,
  //    VK_RValue);

  // OpenCL: const int gid_x = get_global_id(0);
  gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      get_global_id0);

  Expr *YE;
  if (Kernel->getPixelsPerThread() > 1) {
    // OpenCL: const int gid_y = get_local_size(1) * get_group_id(1)*PPT +
    //                           get_local_id(1);
    YE = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
          createBinaryOperator(Ctx, tileVars.local_size_y, tileVars.block_id_y,
            BO_Mul, Ctx.IntTy), createIntegerLiteral(Ctx,
              (int)Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy),
        tileVars.local_id_y, BO_Add, Ctx.IntTy);
  } else {
    // OpenCL: const int gid_y = get_global_id(1)*PPT;
    YE = get_global_id1;
  }
  gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      YE);

  // add gid_x and gid_y statements
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);
  kernelBody.push_back(createDeclStmt(Ctx, gid_x));
  kernelBody.push_back(createDeclStmt(Ctx, gid_y));

  tileVars.global_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.global_id_y = createDeclRefExpr(Ctx, gid_y);
}


// Renderscript initialization
void ASTTranslate::initRenderscript(SmallVector<Stmt *, 16> &kernelBody) {
  VarDecl *gid_x = NULL, *gid_y = NULL;
  VarDecl *xDecl = createVarDecl(Ctx, kernelDecl, "x", Ctx.UnsignedIntTy);
  VarDecl *yDecl = createVarDecl(Ctx, kernelDecl, "y", Ctx.UnsignedIntTy);

  // const int gid_x = x;
  gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      createDeclRefExpr(Ctx, xDecl));

  Expr *YE;
  if (Kernel->getPixelsPerThread() > 1) {
    // const int gid_y = y*PPT;
    YE = createBinaryOperator(Ctx, createDeclRefExpr(Ctx, yDecl),
        createIntegerLiteral(Ctx, (int)Kernel->getPixelsPerThread()), BO_Mul,
        Ctx.IntTy);
  } else {
    // OpenCL: const int gid_y = get_global_id(1)*PPT;
    YE = createDeclRefExpr(Ctx, yDecl);
  }
  gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      YE);

  // add gid_x and gid_y statements
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);
  kernelBody.push_back(createDeclStmt(Ctx, gid_x));
  kernelBody.push_back(createDeclStmt(Ctx, gid_y));

  tileVars.global_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.global_id_y = createDeclRefExpr(Ctx, gid_y);
  tileVars.local_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.local_id_y = createDeclRefExpr(Ctx, gid_y);
  tileVars.block_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.block_id_y = createDeclRefExpr(Ctx, gid_y);
  tileVars.local_size_x = getStrideDecl(Kernel->getIterationSpace()->getAccessor());
  tileVars.local_size_y = createIntegerLiteral(Ctx, 0);

  if (compilerOptions.emitFilterscript()) {
    VarDecl *output = createVarDecl(Ctx, kernelDecl, "OutputVal",
                          Kernel->getIterationSpace()->getImage()->getPixelQualType());
    DC->addDecl(output);
    kernelBody.push_back(createDeclStmt(Ctx, output));
    retValRef = createDeclRefExpr(Ctx, output);
  }
}


Stmt *ASTTranslate::Hipacc(Stmt *S) {
  if (S == NULL) return NULL;

  // search for image width and height parameters
  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      E=kernelDecl->param_end(); I!=E; ++I) {
    ParmVarDecl *PVD = *I;

    // the first parameter is the output image; create association between them.
    if (I==kernelDecl->param_begin()) {
      outputImage = createDeclRefExpr(Ctx, PVD);
      continue;
    }

    // search for iteration space parameters
    if (PVD->getName().equals("is_width")) {
      Kernel->getIterationSpace()->getAccessor()->setWidthDecl(createDeclRefExpr(Ctx,
            PVD));
      continue;
    }
    if (PVD->getName().equals("is_height")) {
      Kernel->getIterationSpace()->getAccessor()->setHeightDecl(createDeclRefExpr(Ctx,
            PVD));
      continue;
    }
    if (PVD->getName().equals("is_stride")) {
      Kernel->getIterationSpace()->getAccessor()->setStrideDecl(createDeclRefExpr(Ctx,
            PVD));
      continue;
    }
    if (PVD->getName().equals("is_offset_x")) {
      Kernel->getIterationSpace()->getAccessor()->setOffsetXDecl(createDeclRefExpr(Ctx,
            PVD));
      continue;
    }
    if (PVD->getName().equals("is_offset_y")) {
      Kernel->getIterationSpace()->getAccessor()->setOffsetYDecl(createDeclRefExpr(Ctx,
            PVD));
      continue;
    }
    if (PVD->getName().equals("bh_start_left")) {
      bh_start_left = createDeclRefExpr(Ctx, PVD);
      continue;
    }
    if (PVD->getName().equals("bh_start_right")) {
      bh_start_right = createDeclRefExpr(Ctx, PVD);
      continue;
    }
    if (PVD->getName().equals("bh_start_top")) {
      bh_start_top = createDeclRefExpr(Ctx, PVD);
      continue;
    }
    if (PVD->getName().equals("bh_start_bottom")) {
      bh_start_bottom = createDeclRefExpr(Ctx, PVD);
      continue;
    }
    if (PVD->getName().equals("bh_fall_back")) {
      bh_fall_back = createDeclRefExpr(Ctx, PVD);
      continue;
    }

    if (compilerOptions.emitRenderscript() ||
        compilerOptions.emitRenderscriptGPU() ||
        compilerOptions.emitFilterscript()) {
      // search for uint32_t x, uint32_t y parameters
      if (PVD->getName().equals("x")) {
        // TODO: scan for uint32_t x
        continue;
      }
      if (PVD->getName().equals("y")) {
        // TODO: scan for uint32_t y
        continue;
      }
    }


    // search for image width, height and stride parameters
    for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
      FieldDecl *FD = KernelClass->getImgFields().data()[i];
      HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

      if (PVD->getName().equals(FD->getNameAsString() + "_width")) {
        Acc->setWidthDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(FD->getNameAsString() + "_height")) {
        Acc->setHeightDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(FD->getNameAsString() + "_stride")) {
        Acc->setStrideDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(FD->getNameAsString() + "_offset_x")) {
        Acc->setOffsetXDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(FD->getNameAsString() + "_offset_y")) {
        Acc->setOffsetYDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
    }
  }

  // in case no stride was found, use image width as fallback
  for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
    FieldDecl *FD = KernelClass->getImgFields().data()[i];
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

    if (Acc->getStrideDecl() == NULL) {
      Acc->setStrideDecl(Acc->getWidthDecl());
    }
  }

  // initialize target-specific variables and add gid_x and gid_y declarations
  // to kernel body
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  SmallVector<Stmt *, 16> kernelBody;
  FunctionDecl *barrier;
  switch (compilerOptions.getTargetCode()) {
    default:
    case TARGET_C:
      initC(kernelBody, S);
      return createCompoundStmt(Ctx, kernelBody);
      break;
    case TARGET_CUDA:
      initCUDA(kernelBody);
      // void __syncthreads();
      barrier = builtins.getBuiltinFunction(CUDABI__syncthreads);
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      initOpenCL(kernelBody);
      // void barrier(cl_mem_fence_flags);
      barrier = builtins.getBuiltinFunction(OPENCLBIbarrier);
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      initRenderscript(kernelBody);
      break;
  }
  lidXRef = tileVars.local_id_x;
  lidYRef = tileVars.local_id_y;
  gidXRef = tileVars.global_id_x;
  gidYRef = tileVars.global_id_y;

  for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
    FieldDecl *FD = KernelClass->getImgFields().data()[i];
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

    // add scale factor calculations for interpolation:
    // float acc_scale_x = (float)acc_width/is_width;
    // float acc_scale_y = (float)acc_height/is_height;
    if (Acc->getInterpolation()!=InterpolateNO) {
      Expr *scaleExprX = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getWidthDecl(Acc), NULL,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getWidthDecl(Kernel->getIterationSpace()->getAccessor()), BO_Div,
          Ctx.FloatTy);
      Expr *scaleExprY = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getHeightDecl(Acc), NULL,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getHeightDecl(Kernel->getIterationSpace()->getAccessor()), BO_Div,
          Ctx.FloatTy);
      VarDecl *scaleDeclX = createVarDecl(Ctx, kernelDecl, Acc->getName() +
          "scale_x", Ctx.FloatTy, scaleExprX);
      VarDecl *scaleDeclY = createVarDecl(Ctx, kernelDecl, Acc->getName() +
          "scale_y", Ctx.FloatTy, scaleExprY);
      DC->addDecl(scaleDeclX);
      DC->addDecl(scaleDeclY);
      kernelBody.push_back(createDeclStmt(Ctx, scaleDeclX));
      kernelBody.push_back(createDeclStmt(Ctx, scaleDeclY));
      Acc->setScaleXDecl(createDeclRefExpr(Ctx, scaleDeclX));
      Acc->setScaleYDecl(createDeclRefExpr(Ctx, scaleDeclY));
    }
  }

  KernelDeclMapShared.clear();
  KernelDeclMapVector.clear();
  KernelDeclMapAcc.clear();

  // add vector pointer declarations for images
  if (Kernel->vectorize() && !compilerOptions.emitC()) {
    for (unsigned int i=0; i<=KernelClass->getNumImages(); i++) {
      FieldDecl *FD = KernelClass->getImgFields().data()[i];
      // output image - iteration space
      StringRef name = "Output";
      if (i<KernelClass->getNumImages()) {
        // normal parameter
        name = FD->getName();
      }

      // search for member name in kernel parameter list
      for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
          N=kernelDecl->param_end(); I!=N; ++I) {
        ParmVarDecl *PVD = *I;

        // parameter name matches
        if (PVD->getName().equals(name)) {
          // <type>4 *Input4 = (<type>4 *) Input;
          VarDecl *VD = CloneDecl(PVD);

          // update output Image reference
          if (name.equals("Output")) outputImage = createDeclRefExpr(Ctx, VD);

          VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
                createDeclRefExpr(Ctx, PVD), NULL,
                Ctx.getTrivialTypeSourceInfo(VD->getType())));

          kernelBody.push_back(createDeclStmt(Ctx, VD));
        }
      }
    }
  }

  // add shared/local memory declarations
  bool use_shared = false;
  bool border_handling = false;
  bool kernel_x = false;
  bool kernel_y = false;
  for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
    FieldDecl *FD = KernelClass->getImgFields().data()[i];
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);
    MemoryAccess memAcc = KernelClass->getImgAccess(FD);

    // bail out for user defined kernels
    if (KernelClass->getKernelType()==UserOperator) break;

    // check if we need border handling
    if (Acc->getBoundaryHandling() != BOUNDARY_UNDEFINED) {
      if (Acc->getSizeX() > 1 || Acc->getSizeY() > 1) border_handling = true;
      if (Acc->getSizeX() > 1) kernel_x = true;
      if (Acc->getSizeY() > 1) kernel_y = true;
    }

    // check if we need shared memory
    if (memAcc == READ_ONLY && Kernel->useLocalMemory(Acc)) {
      std::string sharedName = "_smem";
      sharedName += FD->getNameAsString();
      use_shared = true;

      VarDecl *VD;
      QualType QT;
      // __shared__ T _smemIn[SY-1 + BSY*PPT][3 * BSX];
      // for left and right halo, add 2*BSX
      if (!emitEstimation && compilerOptions.exploreConfig()) {
        Expr *SX, *SY;

        SX = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSX_EXPLORE", Ctx.IntTy, NULL));
        if (Acc->getSizeX() > 1) {
          // 3*BSX
          SX = createBinaryOperator(Ctx, createIntegerLiteral(Ctx, 3), SX,
              BO_Mul, Ctx.IntTy);
        }
        // add padding to avoid bank conflicts
        SX = createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1), BO_Add,
            Ctx.IntTy);

        // TODO: set the same as below at runtime
        SY = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSY_EXPLORE", Ctx.IntTy, NULL));

        if (Kernel->getPixelsPerThread() > 1) {
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                (int)Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy);
        }

        if (Acc->getSizeY() > 1) {
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                (int)Acc->getSizeY()-1), BO_Add, Ctx.IntTy);
        }

        QT = Acc->getImage()->getPixelQualType();
        QT = Ctx.getVariableArrayType(QT, SX, ArrayType::Normal,
            QT.getQualifiers(), SourceLocation());
        QT = Ctx.getVariableArrayType(QT, SY, ArrayType::Normal,
            QT.getQualifiers(), SourceLocation());
      } else {
        llvm::APInt SX, SY;
        SX = llvm::APInt(32, Kernel->getNumThreadsX());
        if (Acc->getSizeX() > 1) {
          // 3*BSX
          SX *= llvm::APInt(32, 3);
        }
        // add padding to avoid bank conflicts
        SX += llvm::APInt(32, 1);

        // size_y = ceil((PPT*BSY+SX-1)/BSY)
        int smem_size_y =
          (int)ceilf((float)(Kernel->getPixelsPerThread()*Kernel->getNumThreadsY()
                + Acc->getSizeY()-1)/(float)Kernel->getNumThreadsY());
        SY = llvm::APInt(32, smem_size_y *Kernel->getNumThreadsY());

        QT = Acc->getImage()->getPixelQualType();
        QT = Ctx.getConstantArrayType(QT, SX, ArrayType::Normal,
            QT.getQualifiers());
        QT = Ctx.getConstantArrayType(QT, SY, ArrayType::Normal,
            QT.getQualifiers());
      }

      switch (compilerOptions.getTargetCode()) {
        case TARGET_C:
        case TARGET_Renderscript:
        case TARGET_RenderscriptGPU:
        case TARGET_Filterscript:
          break;
        case TARGET_CUDA:
          VD = createVarDecl(Ctx, DC, sharedName, QT, NULL);
          VD->addAttr(new (Ctx) CUDASharedAttr(SourceLocation(), Ctx));
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLCPU:
          VD = createVarDecl(Ctx, DC, sharedName, Ctx.getAddrSpaceQualType(QT,
                LangAS::opencl_local), NULL);
          break;
      }

      // search for member name in kernel parameter list
      for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
          N=kernelDecl->param_end(); I!=N; ++I) {
        ParmVarDecl *PVD = *I;

        // parameter name matches
        if (PVD->getName().equals(FD->getName())) {
          // store mapping between ParmVarDecl and shared memory VarDecl
          KernelDeclMapShared[PVD] = VD;
          KernelDeclMapAcc[PVD] = Acc;

          break;
        }
      }

      // add VarDecl to current kernel DeclContext
      DC->addDecl(VD);
      kernelBody.push_back(createDeclStmt(Ctx, VD));
    }
  }

  // activate boundary handling for exploration
  if (compilerOptions.exploreConfig()) {
    border_handling = true;
    kernel_x = true;
    kernel_y = true;
  }


  SmallVector<LabelDecl *, 16> LDS;
  LabelDecl *LDExit = createLabelDecl(Ctx, kernelDecl, "BH_EXIT");
  LabelStmt *LSExit = createLabelStmt(Ctx, LDExit, NULL);
  GotoStmt *GSExit = createGotoStmt(Ctx, LDExit);


  // only create labels if we need border handling
  for (int i=0; i<=9 && border_handling; i++) {
    LabelDecl *LD;
    Expr *if_goto = NULL;

    switch (i) {
      default:
      case 0:
        // fall back: in case the image is too small, use code variant with
        // boundary handling for all borders
        LD = createLabelDecl(Ctx, kernelDecl, "BH_FB");
        if_goto = getBHFallBack();
        break;
      case 1:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.x < bh_start_left &&
        //              blockIdx.y < bh_start_top) goto BO_TL;
        // OpenCL:  if (get_group_id(0) < bh_start_left &&
        //              get_group_id(1) < bh_start_top) goto BO_TL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_TL");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartLeft(), BO_LT, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_y, getBHStartTop(), BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 2:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.x >= bh_start_right &&
        //              blockIdx.y < bh_start_top) goto BO_TR;
        // OpenCL:  if (get_group_id(0) >= bh_start_right &&
        //              get_group_id(1) < bh_start_top) goto BO_TR;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_TR");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartRight(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_y, getBHStartTop(), BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 3:
        // check if we have only a row filter
        if (!kernel_y) continue;

        // CUDA:    if (blockIdx.y < bh_start_top) goto BO_T;
        // OpenCL:  if (get_group_id(1) < bh_start_top) goto BO_T;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_T");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartTop(), BO_LT, Ctx.BoolTy);
        break;
      case 4:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom &&
        //              blockIdx.x < bh_start_left) goto BO_BL;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom &&
        //              get_group_id(0) < bh_start_left) goto BO_BL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_BL");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_x, getBHStartLeft(), BO_LT, Ctx.BoolTy),
            BO_LAnd, Ctx.BoolTy);
        break;
      case 5:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom &&
        //              blockIdx.x >= bh_start_right) goto BO_BR;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom &&
        //              get_group_id(0) >= bh_start_right) goto BO_BL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_BR");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_x, getBHStartRight(), BO_GE, Ctx.BoolTy),
            BO_LAnd, Ctx.BoolTy);
        break;
      case 6:
        // this is not required for row filter, but for kernels where the
        // iteration space is not a multiple of the block size
        if (Kernel->getNumThreadsY()<=1 && Kernel->getPixelsPerThread()<=1 &&
            !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom) goto BO_B;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom) goto BO_B;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_B");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        break;
      case 7:
        // this is not required for column filters, but for kernels where the
        // iteration space is not a multiple of the block size

        // CUDA:    if (blockIdx.x >= bh_start_right) goto BO_R;
        // OpenCL:  if (get_group_id(0) >= bh_start_right) goto BO_R;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_R");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartRight(), BO_GE, Ctx.BoolTy);
        break;
      case 8:
        // check if we have only a column filter
        if (!kernel_x) continue;

        // CUDA:    if (blockIdx.x < bh_start_left) goto BO_L;
        // OpenCL:  if (get_group_id(0) < bh_start_left) goto BO_L;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_L");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartLeft(), BO_LT, Ctx.BoolTy);
        break;
      case 9:
        LD = createLabelDecl(Ctx, kernelDecl, "BH_NO");

        // Note: this dummy check, which is always true is required for
        // the nvcc compiler to generate good code - otherwise nvcc
        // tries to optimize the code of different blocks and register
        // usage will increase significantly and will eventually spill
        // registers to local memory. The OpenCL compiler does not show
        // this behavior.
        // CUDA: if (blockDim.x >= 16) goto BH_NO;
        // OpenCL: if (get_local_size(0) >= 16) goto BH_NO;
        if_goto = createBinaryOperator(Ctx, tileVars.local_size_x,
            createIntegerLiteral(Ctx, 16), BO_GE, Ctx.BoolTy);
        break;
    }
    LDS.push_back(LD);
    GotoStmt *GS = createGotoStmt(Ctx, LD);
    IfStmt *goto_check = createIfStmt(Ctx, if_goto, GS);
    kernelBody.push_back(goto_check);
  }

  // clear all stored decls before cloning, otherwise existing VarDecls will
  // be reused and we will miss declarations
  KernelDeclMapTex.clear();

  int ld_count = 0;
  for (int i=border_handling?0:9; i<=9; i++) {
    // set border handling mode
    switch (i) {
      case 0:
        if (kernel_y) {
          bh_variant.borders.top = 1;
          bh_variant.borders.bottom = 1;
        }
        if (kernel_x) {
          bh_variant.borders.left = 1;
          bh_variant.borders.right = 1;
        }
        break;
      case 1:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.top = 1;
        bh_variant.borders.left = 1;
        break;
      case 2:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.top = 1;
        bh_variant.borders.right = 1;
        break;
      case 3:
        if (kernel_y) bh_variant.borders.top = 1;
        else continue;
        break;
      case 4:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.bottom = 1;
        bh_variant.borders.left = 1;
        break;
      case 5:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.bottom = 1;
        bh_variant.borders.right = 1;
        break;
      case 6:
        // this is not required for row filter, but for kernels where the
        // iteration space is not a multiple of the block size
        if (Kernel->getNumThreadsY()>1 || Kernel->getPixelsPerThread()>1 ||
            kernel_y) bh_variant.borders.bottom = 1;
        else continue;
        break;
      case 7:
        // this is not required for column filters, but for kernels where the
        // iteration space is not a multiple of the block size
        bh_variant.borders.right = 1;
        break;
      case 8:
        if (kernel_x) bh_variant.borders.left = 1;
        else continue;
        break;
      case 9:
        break;
      default:
        break;
    }

    // if (gid_x >= is_offset_x && gid_x < is_width+is_offset_x)
    BinaryOperator *check_bop = NULL;
    if (border_handling) {
      // if (gid_x >= is_offset_x)
      if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl() &&
          !(kernel_x && !bh_variant.borders.left) && bh_variant.borderVal) {
        check_bop = createBinaryOperator(Ctx, gidXRef,
            getOffsetXDecl(Kernel->getIterationSpace()->getAccessor()), BO_GE,
            Ctx.BoolTy);
      }
      // if (gid_x < is_width+is_offset_x)
      if (!(kernel_x && !bh_variant.borders.right) && bh_variant.borderVal) {
        BinaryOperator *check_tmp = NULL;
        if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
          check_tmp = createBinaryOperator(Ctx, gidXRef,
              createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()->getAccessor()),
                getOffsetXDecl(Kernel->getIterationSpace()->getAccessor()),
                BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, gidXRef,
              getWidthDecl(Kernel->getIterationSpace()->getAccessor()), BO_LT,
              Ctx.BoolTy);
        }
        if (check_bop) {
          check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
              Ctx.BoolTy);
        } else {
          check_bop = check_tmp;
        }
      }
      // Filterscript iteration space is always the whole image, so we need to
      // check the y-dimension as well:
      // if (gid_y >= is_offset_y && gid_y < is_height+is_offset_y)
      if (compilerOptions.emitFilterscript()) {
        // if (gid_y >= is_offset_y)
        if (Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl() &&
            !(kernel_y && !bh_variant.borders.left) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = NULL;
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()->getAccessor()), BO_GE,
              Ctx.BoolTy);
          if (check_bop) {
            check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
                Ctx.BoolTy);
          } else {
            check_bop = check_tmp;
          }
        }
        // if (gid_y < is_height+is_offset_y)
        if (!(kernel_y && !bh_variant.borders.right) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = NULL;
          if (Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl()) {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()->getAccessor()),
                  getOffsetYDecl(Kernel->getIterationSpace()->getAccessor()),
                  BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);
          } else {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                getWidthDecl(Kernel->getIterationSpace()->getAccessor()), BO_LT,
                Ctx.BoolTy);
          }
          if (check_bop) {
            check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
                Ctx.BoolTy);
          } else {
            check_bop = check_tmp;
          }
        }
      }
    } else {
      // if (gid_x < is_width+is_offset_x)
      if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
        check_bop = createBinaryOperator(Ctx, gidXRef,
            getOffsetXDecl(Kernel->getIterationSpace()->getAccessor()), BO_GE,
            Ctx.BoolTy);
        check_bop = createBinaryOperator(Ctx, check_bop,
            createBinaryOperator(Ctx, gidXRef, createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()->getAccessor()),
                getOffsetXDecl(Kernel->getIterationSpace()->getAccessor()),
                BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
      } else {
        check_bop = createBinaryOperator(Ctx, gidXRef,
            getWidthDecl(Kernel->getIterationSpace()->getAccessor()), BO_LT,
            Ctx.BoolTy);
      }
      // Filterscript iteration space is always the whole image, so we need to
      // check the y-dimension as well.
      if (compilerOptions.emitFilterscript()) {
        // if (gid_y < is_height+is_offset_y)
        BinaryOperator *check_tmp = NULL;
        if (Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl()) {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()->getAccessor()), BO_GE,
              Ctx.BoolTy);
          check_tmp = createBinaryOperator(Ctx, check_tmp,
              createBinaryOperator(Ctx, gidYRef, createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()->getAccessor()),
                  getOffsetYDecl(Kernel->getIterationSpace()->getAccessor()),
                  BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getWidthDecl(Kernel->getIterationSpace()->getAccessor()), BO_LT,
              Ctx.BoolTy);
        }
        if (check_bop) {
          check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
              Ctx.BoolTy);
        } else {
          check_bop = check_tmp;
        }
      }
    }


    // stage pixels into shared memory
    // ppt + ceil((size_y-1)/sy) iterations
    int p_add = 0;
    if (Kernel->getMaxSizeYUndef()) {
      p_add = (int)ceilf(2*Kernel->getMaxSizeYUndef() /
          (float)Kernel->getNumThreadsY());
    }
    SmallVector<Stmt *, 16> labelBody;
    for (int p=0; use_shared && p<(int)Kernel->getPixelsPerThread()+p_add; p++) {
      if (p==0) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
        // first iteration
        stageIterationToSharedMemory(labelBody, p);
      } else {
        // update lid_y to lid_y + p*local_size_y
        // update gid_y to gid_y + p*local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        // load next iteration to shared memory
        stageIterationToSharedMemory(labelBody, p);
      }
    }
    // synchronize shared memory
    if (use_shared) {
      // add memory barrier synchronization
      SmallVector<Expr *, 16> args;
      switch (compilerOptions.getTargetCode()) {
        case TARGET_C:
        case TARGET_Renderscript:
        case TARGET_RenderscriptGPU:
        case TARGET_Filterscript:
          break;
        case TARGET_CUDA:
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLCPU:
          // TODO: pass CLK_LOCAL_MEM_FENCE argument to barrier()
          args.push_back(createIntegerLiteral(Ctx, 0));
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
      }
    }

    for (int p=0; p<(int)Kernel->getPixelsPerThread(); p++) {
      // clear all stored decls before cloning, otherwise existing
      // VarDecls will be reused and we will miss declarations
      KernelDeclMap.clear();

      // calculate multiple pixels per thread
      SmallVector<Stmt *, 16> pptBody;

      if (p==0) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
      } else {
        // update lid_y to lid_y + p*local_size_y
        // update gid_y to gid_y + p*local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
      }

      // convert kernel function body to CUDA/OpenCL kernel syntax
      Stmt *clonedStmt = Clone(S);
      assert(isa<CompoundStmt>(clonedStmt) && "CompoundStmt for kernel function body expected!");

      // add iteration space check when calculating multiple pixels per thread,
      // having a tiling with multiple threads in the y-dimension, or in case
      // exploration is done
      bool require_is_check = true;
      if (border_handling) {
        // code variant for column filter not processing the bottom
        if (kernel_y && !bh_variant.borders.bottom) require_is_check = false;
        // code variant without border handling
        if (!bh_variant.borderVal) require_is_check = false;
        // number of threads is 1 and no exploration
        if (Kernel->getNumThreadsY()==1 && Kernel->getPixelsPerThread()==1 &&
            !compilerOptions.exploreConfig())
          require_is_check = false;
      } else {
        // exploration
        if (Kernel->getNumThreadsY()==1 && Kernel->getPixelsPerThread()==1 &&
            !compilerOptions.exploreConfig()) require_is_check = false;
      }
      if (require_is_check &&
          // Not necessary for Filterscript, gid_y has already been checked
          !compilerOptions.emitFilterscript()) {
        // if (gid_y + p < is_height)
        BinaryOperator *inner_check_bop = createBinaryOperator(Ctx, gidYRef,
            getHeightDecl(Kernel->getIterationSpace()->getAccessor()), BO_LT,
            Ctx.BoolTy);
        IfStmt *inner_ispace_check = createIfStmt(Ctx, inner_check_bop,
            clonedStmt);
        pptBody.push_back(inner_ispace_check);
      } else {
        pptBody.push_back(clonedStmt);
      }


      // add iteration space checking in case we have padded images and/or
      // padded block/grid configurations
      if (check_bop) {
        IfStmt *ispace_check = createIfStmt(Ctx, check_bop,
            createCompoundStmt(Ctx, pptBody));
        labelBody.push_back(ispace_check);
      } else {
        for (unsigned int i=0, e=pptBody.size(); i!=e; ++i) {
          labelBody.push_back(pptBody.data()[i]);
        }
      }
    }

    // add label statement if needed (boundary handling), else add body
    if (border_handling) {
      LabelStmt *LS = createLabelStmt(Ctx, LDS.data()[ld_count++],
          createCompoundStmt(Ctx, labelBody));
      kernelBody.push_back(LS);
      kernelBody.push_back(GSExit);
    } else {
      kernelBody.push_back(createCompoundStmt(Ctx, labelBody));
    }

    // reset image border configuration
    bh_variant.borderVal = 0;
    // reset lid_y and gid_y
    lidYRef = tileVars.local_id_y;
    gidYRef = tileVars.global_id_y;
  }

  if (border_handling) {
    kernelBody.push_back(LSExit);
  }

  if (compilerOptions.emitFilterscript()) {
      Expr *result = accessMem(outputImage,
          Kernel->getIterationSpace()->getAccessor(), READ_ONLY);
      setExprProps(outputImage, result);
      kernelBody.push_back(createReturnStmt(Ctx, result));
  }

  CompoundStmt *CS = createCompoundStmt(Ctx, kernelBody);

  return CS;
}


VarDecl *ASTTranslate::CloneVarDecl(VarDecl *D) {
  VarDecl *result = NULL;
  ParmVarDecl *PVD = NULL;
  VarDecl *VD = NULL;
  std::string name;
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  QualType QT;

  switch (D->getKind()) {
    default:
      assert(0 && "Only VarDecls supported!");
      break;
    case Decl::ParmVar:
      PVD = static_cast<ParmVarDecl *>(D);
      VD = static_cast<VarDecl *>(D);

      result = KernelDeclMapVector[PVD];

      if (!result) {
        name = PVD->getName();
        QT = PVD->getType();

        // only vectorize image PVDs
        if (Kernel->vectorize() && !compilerOptions.emitC()) {
          for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
            FieldDecl *FD = KernelClass->getImgFields().data()[i];

            // parameter name matches
            if (PVD->getName().equals(FD->getName()) ||
                PVD->getName().equals("Output")) {
              // mark original variable as being used
              Kernel->setUsed(name);

              // add suffix to vectorized variable
              name += "4";

              // get SIMD4 type
              QT = simdTypes.getSIMDType(PVD, SIMD4);
              break;
            }
          }
        }
      }

      break;
    case Decl::Var:
      VD = static_cast<VarDecl *>(D);

      result = KernelDeclMap[VD];
      if (!result && convMask) result = LambdaDeclMap[VD];

      if (!result) {
        QT = VD->getType();
        name  = VD->getName();

        if (Kernel->vectorize() && !compilerOptions.emitC()) {
          VectorInfo VI = KernelClass->getVectorizeInfo(VD);

          if (VI == VECTORIZE) {
            QT = simdTypes.getSIMDType(VD, SIMD4);
          }
        }
      }

      break;
  }

  if (!result) {
    result = VarDecl::Create(Ctx, DC, VD->getInnerLocStart(), VD->getLocation(),
        &Ctx.Idents.get(name), QT, VD->getTypeSourceInfo(),
        VD->getStorageClass());
    // set VarDecl as being used - required for CodeGen
    result->setUsed(true);

    if (!PVD && Kernel->vectorize() && compilerOptions.emitCUDA() &&
        KernelClass->getVectorizeInfo(VD) == VECTORIZE) {
      result->setInit(simdTypes.propagate(VD, Clone(VD->getInit())));
    } else {
      result->setInit(Clone(VD->getInit()));
    }
    result->setTLSKind(VD->getTLSKind());
    result->setInitStyle(VD->getInitStyle());

    // store mapping between original VarDecl and cloned VarDecl
    if (PVD) {
      KernelDeclMapVector[PVD] = result;
    } else {
      if (convMask) {
        LambdaDeclMap[VD] = result;
        LambdaDeclMap[result] = result;
      } else {
        KernelDeclMap[VD] = result;
        KernelDeclMap[result] = result;
      }
    }

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


VarDecl *ASTTranslate::CloneDeclTex(ParmVarDecl *D, std::string prefix) {
  if (D == NULL) {
    return NULL;
  }

  VarDecl *result = NULL;
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);

  result = KernelDeclMapTex[D];

  if (!result) {
    std::string texName = prefix;
    texName += D->getName();
    texName += Kernel->getName();
    result = VarDecl::Create(Ctx, DC, D->getInnerLocStart(), D->getLocation(),
        &Ctx.Idents.get(texName), D->getType(), D->getTypeSourceInfo(),
        D->getStorageClass());

    result->setInit(Clone(D->getInit()));
    result->setTLSKind(D->getTLSKind());
    result->setInitStyle(D->getInitStyle());

    // store mapping between original VarDecl and cloned VarDecl
    KernelDeclMapTex[D] = result;
    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


#ifdef NO_TRANSLATION
#else
Stmt *ASTTranslate::VisitCompoundStmt(CompoundStmt *S) {
  CompoundStmt *result = new (Ctx) CompoundStmt(Ctx, MultiStmtArg(),
      S->getLBracLoc(), S->getLBracLoc());

  SmallVector<Stmt *, 16> body;
  for (CompoundStmt::const_body_iterator I=S->body_begin(), E=S->body_end();
      I!=E; ++I) {
    curCStmt = S;
    Stmt *newS = Clone(*I);
    curCStmt = S;

    if (preStmts.size()) {
      unsigned int num_stmts = 0;
      for (unsigned int i=0, e=preStmts.size(); i!=e; ++i) {
        if (preCStmt.data()[i]==S) {
          body.push_back(preStmts.data()[i]);
          num_stmts++;
        }
      }
      for (unsigned int i=0; i<num_stmts; i++) {
        preStmts.pop_back();
        preCStmt.pop_back();
      }
    }

    body.push_back(newS);

    if (postStmts.size()) {
      unsigned int num_stmts = 0;
      for (unsigned int i=0, e=postStmts.size(); i!=e; ++i) {
        if (postCStmt.data()[i]==S) {
          body.push_back(postStmts.data()[i]);
          num_stmts++;
        }
      }
      for (unsigned int i=0; i<num_stmts; i++) {
        postStmts.pop_back();
        postCStmt.pop_back();
      }
    }
  }

  result->setStmts(Ctx, body.data(), body.size());

  return result;
}


Stmt *ASTTranslate::VisitReturnStmt(ReturnStmt *S) {
  // within convolve lambda-functions, return statements are replaced by
  // reductions
  if (convMask && convRed) {
    Stmt *convInitExpr, *convRedExpr;
    Expr *retVal = Clone(S->getRetValue());

    convInitExpr = createBinaryOperator(Ctx, convRed, retVal, BO_Assign,
        convRed->getType());
    switch (convMode) {
      case HipaccSUM:
        // red += val;
        convRedExpr = createCompoundAssignOperator(Ctx, convRed, retVal,
            BO_AddAssign, convRed->getType());
        break;
      case HipaccMIN:
        // if (val < red) red = val;
        convRedExpr = createIfStmt(Ctx, createBinaryOperator(Ctx, retVal,
              convRed, BO_LT, Ctx.BoolTy), createBinaryOperator(Ctx, convRed,
                retVal, BO_Assign, convRed->getType()));
        break;
      case HipaccMAX:
        // if (val > red) red = val;
        convRedExpr = createIfStmt(Ctx, createBinaryOperator(Ctx, retVal,
              convRed, BO_GT, Ctx.BoolTy), createBinaryOperator(Ctx, convRed,
                retVal, BO_Assign, convRed->getType()));
        break;
      case HipaccPROD:
        // red *= val;
        convRedExpr = createCompoundAssignOperator(Ctx, convRed, retVal,
            BO_MulAssign, convRed->getType());
        break;
      case HipaccMEDIAN:
        assert(0 && "Unsupported convolution mode.");
        break;
    }

    if (convIdxX + convIdxY == 0) {
      // conv_tmp = ...
      return convInitExpr;
    } else {
      // conv_tmp += ...
      return convRedExpr;
    }
  } else {
    return new (Ctx) ReturnStmt(S->getReturnLoc(), Clone(S->getRetValue()), 0);
  }
}


Expr *ASTTranslate::VisitCallExpr(CallExpr *E) {
  if (E->getDirectCallee()) {
    QualType QT = E->getCallReturnType();
    if (Kernel->vectorize() && !compilerOptions.emitC()) {
      QT = simdTypes.getSIMDType(QT, QT.getAsString(), SIMD4);
      // cast ExtVectorType to VectorType - Builtin functions are created
      // using VectorTypes
      const VectorType *VT = QT.getCanonicalType()->getAs<VectorType>();
      QT = QualType(VT, QT.getQualifiers());
    }

    // check if this is a convolve function call
    if (E->getDirectCallee()->getName().equals("convolve")) {
      DiagnosticsEngine &Diags = Ctx.getDiagnostics();

      // convolve(mask, mode, [&] () { lambda-function; });
      assert(E->getNumArgs() == 3 && "Expected 3 arguments to 'convolve' call.");

      // first parameter: Mask<type> reference
      HipaccMask *Mask = NULL;
      assert(isa<MemberExpr>(E->getArg(0)->IgnoreImpCasts()) && "First parameter to 'convolve' call must be a Mask.");
      MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0)->IgnoreImpCasts());

      // get FieldDecl of the MemberExpr
      assert(isa<FieldDecl>(ME->getMemberDecl()) && "Mask must be a C++-class member.");
      FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

      // look for Mask user class member variable
      if (Kernel->getMaskFromMapping(FD)) {
        Mask = Kernel->getMaskFromMapping(FD);
      }
      assert(Mask && "Could not find Mask Field Decl.");

      // second parameter: convolution mode
      assert(isa<DeclRefExpr>(E->getArg(1)) && "Second parameter to 'convolve' call must be the convolution mode.");
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getArg(1));

      if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
          DRE->getDecl()->getType().getAsString() ==
          "enum hipacc::HipaccConvolutionMode") {
        int64_t mode = E->getArg(1)->EvaluateKnownConstInt(Ctx).getSExtValue();
        switch (mode) {
          case HipaccSUM:
            convMode = HipaccSUM;
            break;
          case HipaccMIN:
            convMode = HipaccMIN;
            break;
          case HipaccMAX:
            convMode = HipaccMAX;
            break;
          case HipaccPROD:
            convMode = HipaccPROD;
            break;
          case HipaccMEDIAN:
            convMode = HipaccMEDIAN;
          default:
            unsigned int DiagIDConvMode =
              Diags.getCustomDiagID(DiagnosticsEngine::Error,
                  "Convolution mode not supported, allowed modes are: HipaccSUM, HipaccMIN, HipaccMAX, and HipaccPROD.");
            Diags.Report(E->getArg(1)->getExprLoc(), DiagIDConvMode);
            break;
        }
      } else {
        unsigned int DiagIDConvMode =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Unknown convolution mode detected.");
        Diags.Report(E->getArg(1)->getExprLoc(), DiagIDConvMode);
      }

      // third parameter: lambda-function
      assert(isa<MaterializeTemporaryExpr>(E->getArg(2)) && "Third parameter to 'convolve' call must be a lambda-function.");
      assert(isa<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(E->getArg(2))->GetTemporaryExpr()->IgnoreImpCasts()) &&
          "Third parameter to 'convolve' call must be a lambda-function.");
      LambdaExpr *LE = dyn_cast<LambdaExpr>(dyn_cast<MaterializeTemporaryExpr>(E->getArg(2))->GetTemporaryExpr()->IgnoreImpCasts());

      // check default capture kind
      for (LambdaExpr::capture_iterator II=LE->capture_begin(), EE=LE->capture_end(); II!=EE; ++II) {
        LambdaExpr::Capture cap = *II;

        if (cap.capturesVariable() && cap.getCaptureKind()==LCK_ByCopy) {
          unsigned int DiagIDCapture =
            Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "Capture by copy [=] is not supported for convolve lambda-function (variable %0), use capture by reference [&] instead.");
          Diags.Report(LE->getExprLoc(), DiagIDCapture) << cap.getCapturedVar();
        }
      }

      // introduce temporary for holding the convolution result
      CompoundStmt *outerCompountStmt = curCStmt;
      std::stringstream LSST;
      LSST << "_conv_tmp" << literalCount++;
      VarDecl *conv_tmp = createVarDecl(Ctx, kernelDecl, LSST.str(),
          LE->getCallOperator()->getResultType(), NULL);
      DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
      DC->addDecl(conv_tmp);
      DeclRefExpr *conv_red = createDeclRefExpr(Ctx, conv_tmp);
      convRed = conv_red;
      convMask = Mask;
      preStmts.push_back(createDeclStmt(Ctx, conv_tmp));
      preCStmt.push_back(outerCompountStmt);

      // unroll convolution
      for (unsigned int y=0; y<Mask->getSizeY(); y++) {
        for (unsigned int x=0; x<Mask->getSizeX(); x++) {
          convIdxX = x;
          convIdxY = y;
          Stmt *convIteration = Clone(LE->getBody());
          preStmts.push_back(convIteration);
          preCStmt.push_back(outerCompountStmt);
          // clear decls added while cloning last iteration
          LambdaDeclMap.clear();
        }
      }

      convMask = NULL;
      convRed = NULL;
      convIdxX = convIdxY = 0;

      // add ICE for CodeGen
      return createImplicitCastExpr(Ctx, conv_red->getType(), CK_LValueToRValue,
          conv_red, NULL, VK_RValue);
    }

    // lookup if this function call is supported and choose appropriate
    // function, e.g. exp() instead of expf() in case of OpenCL
    FunctionDecl *targetFD = NULL;
    if (compilerOptions.emitC()) {
      targetFD = E->getDirectCallee();
    } else {
      DeclContext *DC = E->getDirectCallee()->getEnclosingNamespaceContext();
      if (DC->isNamespace()) {
        NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC);
        if (NS->getNameAsString() == "math") {
          DC = DC->getParent();
          if (DC) NS = dyn_cast<NamespaceDecl>(DC);

          if (NS->getNameAsString() == "hipacc") {
            // namespace hipacc::math
            targetFD = E->getDirectCallee();
            bool doUpdate = false;

            if (!compilerOptions.emitCUDA()) {
              std::string name = E->getDirectCallee()->getNameAsString();
              if (name.at(name.length()-1)=='f') {
                if (name!="modf" && name!="erf") {
                  // remove trailing f
                  name.resize(name.size() - 1);
                  doUpdate = true;
                }
              } else if (name.at(0)=='l' && name=="labs") {
                // remove leading l
                name.erase(0, 1);
                doUpdate = true;
              }

              if (doUpdate) {
                SmallVector<QualType, 16> argTypes;
                SmallVector<std::string, 16> argNames;

                for (FunctionDecl::param_iterator P=targetFD->param_begin(),
                    PEnd=targetFD->param_end(); P!=PEnd; ++P) {
                  argTypes.push_back((*P)->getType());
                  argNames.push_back((*P)->getName());
                }

                targetFD = createFunctionDecl(Ctx,
                    Ctx.getTranslationUnitDecl(), name,
                    targetFD->getResultType(), makeArrayRef(argTypes),
                    makeArrayRef(argNames));
              }
            }
          }
        }
      }

      if (!targetFD) {
        targetFD = builtins.getBuiltinFunction(E->getDirectCallee()->getName(),
            QT, compilerOptions.getTargetCode());
      }
    }

    if (!targetFD) {
      DiagnosticsEngine &Diags = Ctx.getDiagnostics();
      unsigned int DiagIDCallExpr =
        Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Found unsupported function call '%0' in kernel.");
      SmallVector<const char *, 16> builtinNames;
      builtins.getBuiltinNames(compilerOptions.getTargetCode(), builtinNames);
      Diags.Report(E->getExprLoc(), DiagIDCallExpr) << E->getDirectCallee()->getName();

      llvm::errs() << "Supported functions are: ";
      for (unsigned int i=0, e=builtinNames.size(); i!=e; ++i) {
        llvm::errs() << builtinNames.data()[i];
        llvm::errs() << ((i==e-1)?".\n":", ");
      }
      exit(EXIT_FAILURE);
    }

    // add ICE for CodeGen
    ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx,
        Ctx.getPointerType(targetFD->getType()), CK_FunctionToPointerDecay,
        createDeclRefExpr(Ctx, targetFD), NULL, VK_RValue);

    // create CallExpr
    CallExpr *result = new (Ctx) CallExpr(Ctx, ICE, MultiExprArg(),
        E->getType(), E->getValueKind(), E->getRParenLoc());

    result->setNumArgs(Ctx, E->getNumArgs());

    for (unsigned int I=0, N=E->getNumArgs(); I<N; ++I) {
      result->setArg(I, Clone(E->getArg(I)));
    }

    setExprProps(E, result);

    return result;
  } else {
    assert(0 && "CallExpr without FunctionDecl as Callee!");
  }
}


Expr *ASTTranslate::VisitMemberExpr(MemberExpr *E) {
  // TODO: create a map with all expressions not to be cloned ..
  if (E==tileVars.local_size_x || E==tileVars.local_size_y) return E;
  if (E==tileVars.local_id_x || E==tileVars.local_id_y) return E;
  if (E==tileVars.block_id_x || E==tileVars.block_id_y) return E;

  // replace member class variables by kernel parameter references
  // (MemberExpr 0x4bd4af0 'int' ->d 0x4bd2330
  //  (CXXThisExpr 0x4bd4ac8 'class hipacc::VerticalMeanFilter *' this))
  // -->
  // (DeclRefExpr 0x4bda540 'int' ParmVar='d' 0x4bd8010)
  ValueDecl *VD = E->getMemberDecl();
  ValueDecl *paramDecl = NULL;

  // search for member name in kernel parameter list
  for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
      N=kernelDecl->param_end(); I!=N; ++I) {
    ParmVarDecl *PVD = *I;

    // parameter name matches
    if (PVD->getName().equals(VD->getName())) {
      paramDecl = PVD;

      // mark parameter as being used within the kernel
      Kernel->setUsed(VD->getName());

      // get vector declaration
      if (Kernel->vectorize() && !compilerOptions.emitC()) {
        if (KernelDeclMapVector.count(PVD)) {
          paramDecl = KernelDeclMapVector[PVD];
          llvm::errs() << "Vectorize: \n";
          paramDecl->dump();
          llvm::errs() << "\n";
        }
      }

      break;
    }
  }

  if (!paramDecl) {
    DiagnosticsEngine &Diags = Ctx.getDiagnostics();
    unsigned int DiagIDParameter =
      Diags.getCustomDiagID(DiagnosticsEngine::Error,
          "Couldn't find initialization of kernel member variable '%0' in class constructor.");
    Diags.Report(E->getExprLoc(), DiagIDParameter) << VD->getName();
    exit(EXIT_FAILURE);
  }

  // check if the parameter is a Mask and replace it by a global VarDecl
  if (!compilerOptions.emitC()) {
    for (unsigned int i=0; i<KernelClass->getNumMasks(); i++) {
      FieldDecl *FD = KernelClass->getMaskFields().data()[i];

      if (paramDecl->getName().equals(FD->getName())) {
        HipaccMask *Mask = Kernel->getMaskFromMapping(FD);

        if (Mask && (Mask->isConstant() || compilerOptions.emitCUDA())) {
          VarDecl *maskVar = NULL;
          // get Mask reference
          for (DeclContext::lookup_result Lookup =
              Ctx.getTranslationUnitDecl()->lookup(DeclarationName(&Ctx.Idents.get(Mask->getName()+Kernel->getName())));
              !Lookup.empty(); Lookup=Lookup.slice(1)) {
            maskVar = cast_or_null<VarDecl>(Lookup.front());

            if (maskVar) break;
          }

          if (!maskVar) {
            maskVar = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
                Mask->getName()+Kernel->getName(), paramDecl->getType());

            DeclContext *DC =
              TranslationUnitDecl::castToDeclContext(Ctx.getTranslationUnitDecl());
            DC->addDecl(maskVar);
          }
          paramDecl = maskVar;
        }
      }
    }
  }

  Expr *result = createDeclRefExpr(Ctx, paramDecl);
  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitBinaryOperator(BinaryOperator *E) {
  Expr *result;
  Expr *RHS = Clone(E->getRHS());

  // check if we have a binary assignment and an Image object on the left-hand
  // side. In case we need built-in function to write to the Image (e.g.
  // write_imagef in OpenCL), we have to replace the BinaryOperator by one
  // function call.
  if (E->getOpcode() == BO_Assign) writeImageRHS = RHS;
  Expr *LHS = Clone(E->getLHS());

  QualType QT;
  // use the type of LHS in case of vectorization
  if (!E->getType()->isExtVectorType() && LHS->getType()->isVectorType()) {
    QT = LHS->getType();
  } else {
    QT = E->getType();
  }

  // writeImageRHS has changed, use LHS
  if (E->getOpcode() == BO_Assign && writeImageRHS && writeImageRHS!=RHS) {
    // TODO: insert checks +=, -=, /=, and *= are not supported on Image objects
    result = LHS;
  } else {
    // normal case: clone binary operator
    result = new (Ctx) BinaryOperator(LHS, RHS, E->getOpcode(), QT,
        E->getValueKind(), E->getObjectKind(), E->getOperatorLoc(),
        E->isFPContractable());
  }
  if (E->getOpcode() == BO_Assign) writeImageRHS = NULL;

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  Expr *subExpr = Clone(E->getSubExpr());

  QualType QT = E->getType();
  CastKind CK = E->getCastKind();

  CXXCastPath castPath;
  setCastPath(E, castPath);

  Expr *litExpr = subExpr->IgnoreImpCasts();
  if (isa<UnaryOperator>(litExpr)) {
    litExpr = ((UnaryOperator *)litExpr)->getSubExpr();
  }

  if (E->getCastKind() == CK_LValueToRValue &&
      (isa<IntegerLiteral>(litExpr->IgnoreParenCasts()) ||
       isa<FloatingLiteral>(litExpr->IgnoreParenCasts()) ||
       isa<CharacterLiteral>(litExpr->IgnoreParenCasts()))) {
    // in case of constant propagation, lvalue-to-rvalue casts are invalid
    if (subExpr->getType() == E->getType()) return subExpr;

    if (isa<FloatingLiteral>(litExpr->IgnoreParenCasts())) {
      if (E->getType()->isFloatingType()) {
        CK = CK_FloatingCast;
      } else {
        CK = CK_FloatingToIntegral;
      }
    } else {
      if (E->getType()->isFloatingType()) {
        CK = CK_IntegralToFloating;
      } else {
        CK = CK_IntegralCast;
      }
    }
  } else {
    // in case of vectorization, the cast type may change for the cloned subExpr
    switch (CK) {
      default: break;
      case CK_LValueToRValue:
      case CK_NoOp:
        QT = subExpr->getType();
        break;
    }
  }

  Expr *result = ImplicitCastExpr::Create(Ctx, QT, CK, subExpr, &castPath,
      E->getValueKind());

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitCStyleCastExpr(CStyleCastExpr *E) {
  Expr *subExpr = Clone(E->getSubExpr());
  QualType QT;

  // in case of vectorization, the cast type may change for the cloned subExpr
  switch (E->getCastKind()) {
    default:
      QT = E->getType();
      break;
    case CK_LValueToRValue:
    case CK_NoOp:
      QT = subExpr->getType();
      break;
  }

  CXXCastPath castPath;
  setCastPath(E, castPath);

  CStyleCastExpr *result = CStyleCastExpr::Create(Ctx, QT, E->getValueKind(),
      E->getCastKind(), subExpr, &castPath, E->getTypeInfoAsWritten(),
      E->getLParenLoc(), E->getRParenLoc());

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  Expr *result = NULL;

  // assume that all CXXOperatorCallExpr are memory access functions, since we
  // don't support function calls
  assert(isa<MemberExpr>(E->getArg(0)) && "Memory access function assumed.");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0));

  // get FieldDecl of the MemberExpr
  assert(isa<FieldDecl>(ME->getMemberDecl()) && "Image must be a C++-class member.");
  FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

  // MemberExpr is converted to DeclRefExpr when cloning
  DeclRefExpr *LHS = dyn_cast<DeclRefExpr>(Clone(E->getArg(0)));


  // look for Mask user class member variable
  if (Kernel->getMaskFromMapping(FD)) {
    HipaccMask *Mask = Kernel->getMaskFromMapping(FD);
    MemoryAccess memAcc = KernelClass->getImgAccess(FD);
    assert(memAcc==READ_ONLY && "mask &readonle supp");

    switch (E->getNumArgs()) {
      default:
        assert(0 && "0 or 2 arguments for Mask operator() expected!");
        break;
      case 1:
        assert(convMask && convMask==Mask && "0 arguments for Mask operator() only allowed within convolution lambda-function.");
        // within convolute lambda-function
        if (Mask->isConstant()) {
          // propagate constants
          result = Clone(Mask->getInitList()->getInit(Mask->getSizeY() *
                convIdxX + convIdxY)->IgnoreParenCasts());
        } else {
          // access mask elements
          Expr *midx_x = createIntegerLiteral(Ctx, convIdxX);
          Expr *midx_y = createIntegerLiteral(Ctx, convIdxY);

          switch (compilerOptions.getTargetCode()) {
            case TARGET_C:
            case TARGET_CUDA:
              // array subscript: Mask[conv_y][conv_x]
              result = accessMem2DAt(LHS, midx_x, midx_y);
              break;
            case TARGET_OpenCL:
            case TARGET_OpenCLCPU:
            case TARGET_Renderscript:
              if (Mask->isConstant()) {
                // array subscript: Mask[conv_y][conv_x]
                result = accessMem2DAt(LHS, midx_x, midx_y);
              } else {
                // array subscript: Mask[(conv_y)*width + conv_x]
                result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                      (int)Mask->getSizeX()), midx_x, midx_y);
              }
              break;
            case TARGET_RenderscriptGPU:
            case TARGET_Filterscript:
              // allocation access: rsGetElementAt(Mask, conv_x, conv_y)
              result = accessMemAllocAt(LHS, memAcc, midx_x, midx_y);
              break;
          }
        }
        break;
      case 3:
        // 0: -> (this *) Mask class
        // 1: -> x
        // 2: -> y
        switch (compilerOptions.getTargetCode()) {
          case TARGET_C:
          case TARGET_CUDA:
            // array subscript: Mask[y+size_y/2][x+size_x/2]
            result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                  Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                    (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                createBinaryOperator(Ctx, Clone(E->getArg(2)),
                  createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                  Ctx.IntTy));
            break;
          case TARGET_OpenCL:
          case TARGET_OpenCLCPU:
          case TARGET_Renderscript:
            if (Mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            } else {
              // array subscript: Mask[(y+size_y/2)*width + x+size_x/2]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    (int)Mask->getSizeX()), createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            }
            break;
          case TARGET_RenderscriptGPU:
          case TARGET_Filterscript:
            if (Mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            } else {
              // allocation access: rsGetElementAt(Mask, x+size_x/2, y+size_y/2)
              result = accessMemAllocAt(LHS, memAcc, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            }
            break;
        }
        break;
    }
  }


  // look for Image user class member variable
  if (Kernel->getImgFromMapping(FD)) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);
    MemoryAccess memAcc = KernelClass->getImgAccess(FD);

    // Images are ParmVarDecls
    bool use_shared = false;
    DeclRefExpr *DRE = NULL;
    if (!Kernel->vectorize()) { // Images are replaced by local pointers
      ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(LHS->getDecl());
      assert(PVD && "Image variable must be a ParmVarDecl!");

      if (KernelDeclMapShared[PVD]) {
        // shared/local memory
        use_shared = true;
        VarDecl *VD = KernelDeclMapShared[PVD];
        DRE = createDeclRefExpr(Ctx, VD);
      }
    }

    Expr *SY, *TX;
    if (Acc->getSizeX() > 1) {
      if (compilerOptions.exploreConfig()) {
        TX = tileVars.local_size_x;
      } else {
        TX = createIntegerLiteral(Ctx, (int)Kernel->getNumThreadsX());
      }
    } else {
      TX = createIntegerLiteral(Ctx, 0);
    }
    if (Acc->getSizeY() > 1) {
      SY = createIntegerLiteral(Ctx, (int)Acc->getSizeY()/2);
    } else {
      SY = createIntegerLiteral(Ctx, 0);
    }

    switch (E->getNumArgs()) {
      default:
        assert(0 && "0 or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // 0: -> (this *) Image Class
        if (compilerOptions.emitC()) {
          // no padding is considered, data is accessed as a 2D-array
          result = accessMemPolly(LHS, Acc, memAcc, NULL, NULL);
        } else {
          if (use_shared) {
            result = accessMemShared(DRE, TX, SY);
          } else {
            result = accessMem(LHS, Acc, memAcc);
          }
        }
        break;
      case 2:
        // 0: -> (this *) Image Class
        // 1: -> Mask
        assert(isa<MemberExpr>(E->getArg(1)->IgnoreImpCasts()) && "Accessor operator() with 1 argument requires a convolution Mask.");
        assert(convMask && convMask==Kernel->getMaskFromMapping(dyn_cast<FieldDecl>(dyn_cast<MemberExpr>(E->getArg(1)->IgnoreImpCasts())->getMemberDecl())) &&
            "Accessor operator() with 1 argument requires a convolution Mask.");
        assert(convMask && "0 or 2 arguments for Accessor operator() expected!\n");
      case 3:
        // 0: -> (this *) Image Class
        // 1: -> offset x
        // 2: -> offset y
        Expr *offset_x, *offset_y;
        if (E->getNumArgs()==2) {
          offset_x = createIntegerLiteral(Ctx,
              convIdxX-(int)convMask->getSizeX()/2);
          offset_y = createIntegerLiteral(Ctx,
              convIdxY-(int)convMask->getSizeY()/2);
        } else {
          offset_x = Clone(E->getArg(1));
          offset_y = Clone(E->getArg(2));
        }

        if (compilerOptions.emitC()) {
          // no padding is considered, data is accessed as a 2D-array
          result = accessMemPolly(LHS, Acc, memAcc, offset_x, offset_y);
        } else {
          if (use_shared) {
            result = accessMemShared(DRE, createBinaryOperator(Ctx, offset_x,
                  TX, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx, offset_y,
                    SY, BO_Add, Ctx.IntTy));
          } else {
            switch (memAcc) {
              case READ_ONLY:
                if (bh_variant.borderVal) {
                  return addBorderHandling(LHS, offset_x, offset_y, Acc);
                }
                // fall through
              case WRITE_ONLY:
              case READ_WRITE:
              case UNDEFINED:
                result = accessMem(LHS, Acc, memAcc, offset_x, offset_y);
                break;
            }
          }
        }
        break;
    }
  }

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  assert(isa<MemberExpr>(E->getCallee()) &&
      "Hipacc: Stumbled upon unsupported expression or statement: CXXMemberCallExpr");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getCallee());

  DeclRefExpr *LHS;
  HipaccAccessor *Acc = NULL;
  MemoryAccess memAcc = UNDEFINED;
  Expr *result;

  if (isa<CXXThisExpr>(ME->getBase()->IgnoreImpCasts())) {
    // Kernel context -> use Iteration Space output Accessor
    LHS = outputImage;
    Acc = Kernel->getIterationSpace()->getAccessor();
    memAcc = WRITE_ONLY;

    // getX() method -> gid_x - is_offset_x
    if (ME->getMemberNameInfo().getAsString() == "getX") {
      return createParenExpr(Ctx, removeISOffsetX(gidXRef, Acc));
    }

    // getY() method -> gid_y
    if (ME->getMemberNameInfo().getAsString() == "getY") {
      if (compilerOptions.emitFilterscript()) {
        return createParenExpr(Ctx, removeISOffsetY(gidYRef, Acc));
      } else {
        return gidYRef;
      }
    }

    // output() method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output") {
      assert(E->getNumArgs()==0 && "no arguments for output() method supported!");

      switch (compilerOptions.getTargetCode()) {
        case TARGET_C:
          // no padding is considered, data is accessed as a 2D-array
          result = accessMemPolly(LHS, Acc, memAcc, NULL, NULL);
          break;
        case TARGET_CUDA:
        case TARGET_OpenCL:
        case TARGET_OpenCLCPU:
        case TARGET_Renderscript:
        case TARGET_RenderscriptGPU:
          result = accessMem(LHS, Acc, memAcc);
          break;
        case TARGET_Filterscript:
          postStmts.push_back(createReturnStmt(Ctx, retValRef));
          postCStmt.push_back(curCStmt);
          result = retValRef;
          break;
      }

      setExprProps(E, result);

      return result;
    }
  } else if (isa<MemberExpr>(ME->getBase()->IgnoreImpCasts())) {
    // Accessor context -> use Accessor
    // MemberExpr is converted to DeclRefExpr when cloning
    LHS = dyn_cast<DeclRefExpr>(Clone(ME->getBase()->IgnoreImpCasts()));

    // find corresponding Image user class member variable
    MemberExpr *ImgAcc = dyn_cast<MemberExpr>(ME->getBase()->IgnoreImpCasts());
    FieldDecl *FD = dyn_cast<FieldDecl>(ImgAcc->getMemberDecl());

    Acc = Kernel->getImgFromMapping(FD);
    memAcc = KernelClass->getImgAccess(FD);
    assert(Acc && "Could not find Image/Accessor Field Decl.");

    // Acc.getX() method -> acc_scale_x * (gid_x - is_offset_x)
    if (ME->getMemberNameInfo().getAsString() == "getX") {
      Expr *idx_x = gidXRef;
      // remove is_offset_x and scale index to Accessor size
      if (Acc->getInterpolation()!=InterpolateNO) {
        idx_x = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), NULL,
            Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      } else {
        idx_x = createParenExpr(Ctx, removeISOffsetX(gidXRef, Acc));
      }

      return idx_x;
    }

    // Acc.getY() method -> acc_scale_y * gid_y
    if (ME->getMemberNameInfo().getAsString() == "getY") {
      Expr *idx_y = gidYRef;
      // scale index to Accessor size
      if (Acc->getInterpolation()!=InterpolateNO) {
        idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), NULL,
            Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      } else if (compilerOptions.emitFilterscript()) {
        idx_y = createParenExpr(Ctx, removeISOffsetY(gidYRef, Acc));
      }

      return idx_y;
    }
  }

  // Acc.getPixel(x, y) method -> img[y][x]
  // outputAtPixel(x, y) method -> img[y][x]
  if (ME->getMemberNameInfo().getAsString() == "getPixel" ||
      ME->getMemberNameInfo().getAsString() == "outputAtPixel") {
    assert(Acc && E->getNumArgs()==2 && "x and y argument for getPixel() or outputAtPixel() required!");
    Expr *idx_x = addGlobalOffsetX(Clone(E->getArg(0)), Acc);
    Expr *idx_y = addGlobalOffsetY(Clone(E->getArg(1)), Acc);

    switch (compilerOptions.getTargetCode()) {
      case TARGET_C:
        // no padding is considered, data is accessed as a 2D-array
        result = accessMem2DAt(LHS, idx_x, idx_y);
        break;
      case TARGET_CUDA:
        if (Kernel->useTextureMemory(Acc)) {
          result = accessMemTexAt(LHS, Acc, memAcc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        }
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLCPU:
        if (Kernel->useTextureMemory(Acc)) {
          result = accessMemImgAt(LHS, Acc, memAcc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        }
        break;
      case TARGET_Renderscript:
        result = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        break;
      case TARGET_RenderscriptGPU:
      case TARGET_Filterscript:
        if (ME->getMemberNameInfo().getAsString() == "outputAtPixel" &&
            compilerOptions.emitFilterscript()) {
            assert(0 && "Filterscript does not support outputAtPixel().");
        }
        result = accessMemAllocAt(LHS, memAcc, idx_x, idx_y);
        break;
    }

    setExprProps(E, result);

    return result;
  }

  HIPACC_NOT_SUPPORTED(CXXMemberCallExpr);
  return NULL;
}
#endif

// vim: set ts=2 sw=2 sts=2 et ai:

