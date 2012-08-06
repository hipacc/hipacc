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

//#define TEST_UPDATE_SMEM

//===----------------------------------------------------------------------===//
// Statement/expression transformations
//===----------------------------------------------------------------------===//


Stmt* ASTTranslate::Hipacc(Stmt *S) {
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
      isWidth = createDeclRefExpr(Ctx, PVD);
      Kernel->getIterationSpace()->getAccessor()->setWidthDecl(isWidth);
      continue;
    }
    if (PVD->getName().equals("is_height")) {
      isHeight = createDeclRefExpr(Ctx, PVD);
      Kernel->getIterationSpace()->getAccessor()->setHeightDecl(isHeight);
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

    // search for image width, height and stride parameters
    for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
      FieldDecl *FD = KernelClass->getImgFields().data()[i];
      HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

      if (PVD->getName().equals(Acc->getWidthParm())) {
        Acc->setWidthDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(Acc->getHeightParm())) {
        Acc->setHeightDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(Acc->getStrideParm())) {
        Acc->setStrideDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(Acc->getOffsetXParm())) {
        Acc->setOffsetXDecl(createDeclRefExpr(Ctx, PVD));
        continue;
      }
      if (PVD->getName().equals(Acc->getOffsetYParm())) {
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


  // create variable declarations for global id variables that are generated
  // when the kernel function is pretty printed
  VarDecl *gid_x = NULL, *gid_y = NULL;
  DeclRefExpr *gid_x_ref, *gid_y_ref = NULL;
  DeclStmt *gid_x_stmt, *gid_y_stmt = NULL;

  // OpenCL function calls to built-in functions
  FunctionDecl *barrier;
  ImplicitCastExpr *get_global_size0, *get_global_size1;
  ImplicitCastExpr *get_global_id0, *get_global_id1;
  llvm::SmallVector<Expr *, 16> tmpArg0;
  llvm::SmallVector<Expr *, 16> tmpArg1;
  tmpArg0.push_back(createIntegerLiteral(Ctx, 0));
  tmpArg1.push_back(createIntegerLiteral(Ctx, 1));

  if (emitPolly) {
    // Polly: int gid_x = offset_x;
    if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
      gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
          Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl());
    } else {
      gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
          createIntegerLiteral(Ctx, 0));
    }
    gid_x_stmt = createDeclStmt(Ctx, gid_x);
    gid_x_ref = createDeclRefExpr(Ctx, gid_x);
    gidXRef = gid_x_ref;

    // Polly: int gid_y = offset_y;
    if (Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl()) {
      gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.IntTy,
          Kernel->getIterationSpace()->getAccessor()->getOffsetYDecl());
    } else {
      gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.IntTy,
          createIntegerLiteral(Ctx, 0));
    }
    gid_y_stmt = createDeclStmt(Ctx, gid_y);
    gid_y_ref = createDeclRefExpr(Ctx, gid_y);
    gidYRef = gid_y_ref;

    // convert the function body to kernel syntax
    Stmt* clonedStmt = Clone(S);
    assert(isa<CompoundStmt>(clonedStmt) && "CompoundStmt for kernel function body expected!");

    //
    // for (int gid_y=offset_y; gid_y<2048; gid_y++) {
    //     for (int gid_x=offset_x; gid_x<4096; gid_x++) {
    //         body
    //     }
    // }
    //
    ForStmt *innerLoop = createForStmt(Ctx, gid_x_stmt,
        createBinaryOperator(Ctx, gidXRef, createIntegerLiteral(Ctx, 4096),
          BO_LT, Ctx.BoolTy), createUnaryOperator(Ctx, gidXRef, UO_PostInc,
            gidXRef->getType()), clonedStmt);
    ForStmt *outerLoop = createForStmt(Ctx, gid_y_stmt,
        createBinaryOperator(Ctx, gidYRef, createIntegerLiteral(Ctx, 2048),
          BO_LT, Ctx.BoolTy), createUnaryOperator(Ctx, gidYRef, UO_PostInc,
            gidYRef->getType()), innerLoop);

    llvm::SmallVector<Stmt *, 16> kernelBody;
    kernelBody.push_back(outerLoop);
    CompoundStmt *CS = createCompoundStmt(Ctx, kernelBody);

    return CS;
  } else if (compilerOptions.emitCUDA()) {
    // CUDA
    /*DEVICE_BUILTIN*/
    //struct uint3
    //{
    //  unsigned int x, y, z;
    //};
    llvm::SmallVector<QualType, 16> uintDeclTypes;
    llvm::SmallVector<llvm::StringRef, 16> uintDeclNames;
    uintDeclTypes.push_back(Ctx.UnsignedIntTy);
    uintDeclTypes.push_back(Ctx.UnsignedIntTy);
    uintDeclTypes.push_back(Ctx.UnsignedIntTy);
    uintDeclNames.push_back("x");
    uintDeclNames.push_back("y");
    uintDeclNames.push_back("z");
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
    VarDecl *gridDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
        "gridDim", Ctx.getTypeDeclType(uint3RD), NULL);
    //int warpSize;
    VarDecl *warpSize = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
        "warpSize", Ctx.IntTy, NULL);

    DeclRefExpr *TIRef = createDeclRefExpr(Ctx, threadIdx);
    DeclRefExpr *BIRef = createDeclRefExpr(Ctx, blockIdx);
    DeclRefExpr *BDRef = createDeclRefExpr(Ctx, blockDim);
    DeclRefExpr *GDRef = createDeclRefExpr(Ctx, gridDim);
    VarDecl *xVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "x",
        Ctx.IntTy, NULL);
    VarDecl *yVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "y",
        Ctx.IntTy, NULL);

    local_id_x = createMemberExpr(Ctx, TIRef, false, xVD, xVD->getType());
    local_id_y = createMemberExpr(Ctx, TIRef, false, yVD, yVD->getType());
    block_id_x = createMemberExpr(Ctx, BIRef, false, xVD, xVD->getType());
    block_id_y = createMemberExpr(Ctx, BIRef, false, yVD, yVD->getType());
    local_size_x = createMemberExpr(Ctx, BDRef, false, xVD, xVD->getType());
    local_size_y = createMemberExpr(Ctx, BDRef, false, yVD, yVD->getType());
    grid_size_x = createMemberExpr(Ctx, GDRef, false, xVD, xVD->getType());
    grid_size_y = createMemberExpr(Ctx, GDRef, false, yVD, yVD->getType());

    // CUDA: const int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
        createBinaryOperator(Ctx, createBinaryOperator(Ctx, local_size_x,
            block_id_x, BO_Mul, Ctx.IntTy), local_id_x, BO_Add, Ctx.IntTy));

    // CUDA: const int gid_y = blockDim.y*PPT*blockIdx.y + threadIdx.y;
    Expr *YE = createBinaryOperator(Ctx, local_size_y, block_id_y, BO_Mul,
        Ctx.IntTy);
    if (Kernel->getPixelsPerThread() > 1) {
      YE = createBinaryOperator(Ctx, YE, createIntegerLiteral(Ctx,
            Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy);
    }
    gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
        createBinaryOperator(Ctx, YE, local_id_y, BO_Add, Ctx.IntTy));

    // void __syncthreads();
    barrier = builtins.getBuiltinFunction(CUDABI__syncthreads);
  } else {
    // uint get_work_dim();
    FunctionDecl *get_work_dim =
      builtins.getBuiltinFunction(OPENCLBIget_work_dim);
    // size_t get_global_size(uint dimindx);
    FunctionDecl *get_global_size =
      builtins.getBuiltinFunction(OPENCLBIget_global_size);
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
    FunctionDecl *get_num_groups =
      builtins.getBuiltinFunction(OPENCLBIget_num_groups);
    //size_t get_group_id(uint dimindx);
    FunctionDecl *get_group_id =
      builtins.getBuiltinFunction(OPENCLBIget_group_id);

    // void barrier(cl_mem_fence_flags);
    barrier = builtins.getBuiltinFunction(OPENCLBIbarrier);

    // .(0) .(1)
    get_global_size0 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg0),
        NULL, VK_RValue);
    get_global_size1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg1),
        NULL, VK_RValue);
    get_global_id0 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg0), NULL,
        VK_RValue);
    get_global_id1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg1), NULL,
        VK_RValue);
    local_size_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_local_size, tmpArg0), NULL,
        VK_RValue);
    local_size_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_local_size, tmpArg1), NULL,
        VK_RValue);
    local_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg0), NULL,
        VK_RValue);
    local_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg1), NULL,
        VK_RValue);
    grid_size_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg0), NULL,
        VK_RValue);
    grid_size_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg1), NULL,
        VK_RValue);
    block_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg0), NULL,
        VK_RValue);
    block_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
        CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg1), NULL,
        VK_RValue);


    // OpenCL: const int gid_x = get_global_id(0);
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
        get_global_id0);

    Expr *YE;
    if (Kernel->getPixelsPerThread() > 1) {
      // OpenCL: const int gid_y = get_local_size(1) * get_group_id(1)*PPT +
      //                           get_local_id(1);
      YE = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
            createBinaryOperator(Ctx, local_size_y, block_id_y, BO_Mul,
              Ctx.IntTy), createIntegerLiteral(Ctx,
                Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy), local_id_y,
          BO_Add, Ctx.IntTy);
    } else {
      // OpenCL: const int gid_y = get_global_id(1)*PPT;
      YE = get_global_id1;
    }
    gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
        YE);
  }
  lidXRef = local_id_x;
  lidYRef = local_id_y;

  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);
  gid_x_stmt = createDeclStmt(Ctx, gid_x);
  gid_x_ref = createDeclRefExpr(Ctx, gid_x);
  gidXRef = gid_x_ref;

  gid_y_stmt = createDeclStmt(Ctx, gid_y);
  gid_y_ref = createDeclRefExpr(Ctx, gid_y);
  gidYRef = gid_y_ref;

  // add gid_x and gid_y declarations to kernel body
  llvm::SmallVector<Stmt *, 16> kernelBody;
  kernelBody.push_back(gid_x_stmt);
  kernelBody.push_back(gid_y_stmt);

  for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
    FieldDecl *FD = KernelClass->getImgFields().data()[i];
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

    // add scale factor calculations for interpolation:
    // float acc_scale_x = (float)acc_width/is_width;
    // float acc_scale_y = (float)acc_height/is_height;
    if (Acc->getInterpolation()!=InterpolateNO) {
      Expr *scaleExprX = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, Acc->getWidthDecl(), NULL,
            NULL), Kernel->getIterationSpace()->getAccessor()->getWidthDecl(),
          BO_Div, Ctx.FloatTy);
      Expr *scaleExprY = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, Acc->getHeightDecl(), NULL,
            NULL), Kernel->getIterationSpace()->getAccessor()->getHeightDecl(),
          BO_Div, Ctx.FloatTy);
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
  if (Kernel->vectorize()) {
    for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
      FieldDecl *FD = KernelClass->getImgFields().data()[i];

      // search for member name in kernel parameter list
      for (FunctionDecl::param_iterator I=kernelDecl->param_begin(),
          N=kernelDecl->param_end(); I!=N; ++I) {
        ParmVarDecl *PVD = *I;

        // parameter name matches
        if (PVD->getName().equals(FD->getName())) {
          // <type>4 *Input4 = (<type>4 *) Input;
          VarDecl *VD = CloneDecl(PVD);

          VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
                createDeclRefExpr(Ctx, PVD), NULL, NULL));

          kernelBody.push_back(createDeclStmt(Ctx, VD));
        }
      }
    }
  }

  // add shared/local memory declarations
  bool useShared = false;
  bool border_handling = false;
  bool kernel_x = false;
  bool kernel_y = false;
  for (unsigned int i=0; i<KernelClass->getNumImages(); i++) {
    FieldDecl *FD = KernelClass->getImgFields().data()[i];
    HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);
    MemoryAccess memAcc = KernelClass->getImgAccess(FD);

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
      useShared = true;

      VarDecl *VD;
      QualType QT;
#ifdef TEST_UPDATE_SMEM
      // __shared__ T _smemIn[SY-1 + 2*BSY][SX-1 + BSX + 1];
#else
      // __shared__ T _smemIn[SY-1 + BSY*PPT][SX-1 + BSX + 1];
#endif
      if (!emitEstimation && compilerOptions.exploreConfig()) {
        Expr *SX, *SY;

        SX = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSX_EXPLORE", Ctx.IntTy, NULL));
        SY = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSY_EXPLORE", Ctx.IntTy, NULL));
        if (Kernel->getPixelsPerThread() > 1) {
#ifdef TEST_UPDATE_SMEM
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx, 2),
              BO_Mul, Ctx.IntTy);
#else
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy);
#endif
        }

        if (Acc->getSizeX() > 1) {
          SX = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                Acc->getSizeX()), SX, BO_Add, Ctx.IntTy);
        }
        if (Acc->getSizeY() > 1) {
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                Acc->getSizeY()-1), BO_Add, Ctx.IntTy);
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
          SX += llvm::APInt(32, Acc->getSizeX()) - llvm::APInt(32, 1);
        }
        // add padding to avoid bank conflicts
        if ((Acc->getSizeX()-1) % 16 == 0) {
          SX += llvm::APInt(32, 1);
        }

        SY = llvm::APInt(32, Kernel->getNumThreadsY());
        if (Kernel->getPixelsPerThread() > 1) {
#ifdef TEST_UPDATE_SMEM
          SY *= llvm::APInt(32, 2);
#else
          SY *= llvm::APInt(32, Kernel->getPixelsPerThread());
#endif
        }
        if (Acc->getSizeY() > 1) {
          SY += llvm::APInt(32, Acc->getSizeY()) - llvm::APInt(32, 1);
        }

        QT = Acc->getImage()->getPixelQualType();
        QT = Ctx.getConstantArrayType(QT, SX, ArrayType::Normal,
            QT.getQualifiers());
        QT = Ctx.getConstantArrayType(QT, SY, ArrayType::Normal,
            QT.getQualifiers());
      }

      if (compilerOptions.emitCUDA()) {
        VD = createVarDecl(Ctx, DC, sharedName, QT, NULL);
        VD->addAttr(new (Ctx) CUDASharedAttr(SourceLocation(), Ctx));
      } else {
        VD = createVarDecl(Ctx, DC, sharedName, Ctx.getAddrSpaceQualType(QT,
              LangAS::opencl_local), NULL);
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


  llvm::SmallVector<LabelDecl *, 16> LDS;
  LabelDecl *LDExit = createLabelDecl(Ctx, kernelDecl, "BH_EXIT");
  LabelStmt *LSExit = createLabelStmt(Ctx, LDExit, NULL);
  GotoStmt *GSExit = createGotoStmt(Ctx, LDExit);

  // calculate the number of additional blocks required for border handling
  // due to grids bigger than the image.
  Expr *numBlocksBHBottom, *numBlocksBHRight, *numBlocksBHLeft, *numBlocksBHTop;
  if (!emitEstimation && compilerOptions.exploreConfig()) {
    numBlocksBHLeft = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
          "BHX_EXPLORE", Ctx.IntTy, NULL));
    numBlocksBHTop = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
          "BHY_EXPLORE", Ctx.IntTy, NULL));
  } else {
    numBlocksBHLeft = createIntegerLiteral(Ctx,
        emitEstimation?23:Kernel->getNumBlocksBHL());
    numBlocksBHTop = createIntegerLiteral(Ctx,
        emitEstimation?23:Kernel->getNumBlocksBHY());
  }

  FunctionDecl *ceilFD = NULL;
  if (compilerOptions.emitCUDA()) {
    ceilFD = builtins.getBuiltinFunction(HIPACCBIceilf);
  } else {
    ceilFD = builtins.getBuiltinFunction(OPENCLBIceilf);
  }

  llvm::SmallVector<Expr *, 16> args;
  // CUDA: ceilf((blockDim.x*gridDim.x - (is_width + size_x)) / (float)(blockDim.x))
  // OpenCL: ceil(get_local_size(0)*get_num_groups(0) - (is_width + size_x)) /
  //      (float)(get_local_size(0))
  numBlocksBHRight = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
        local_size_x, grid_size_x, BO_Mul, Ctx.IntTy), createParenExpr(Ctx,
          createBinaryOperator(Ctx, isWidth, createIntegerLiteral(Ctx,
              Kernel->getMaxSizeX()), BO_Sub, Ctx.IntTy)), BO_Sub, Ctx.IntTy);
  numBlocksBHRight = createBinaryOperator(Ctx, createParenExpr(Ctx,
        numBlocksBHRight), createCStyleCastExpr(Ctx, ceilFD->getResultType(),
          CK_IntegralToFloating, local_size_x, NULL, NULL), BO_Div, Ctx.IntTy);

  args.push_back(numBlocksBHRight);
  numBlocksBHRight = createFunctionCall(Ctx, ceilFD, args); 

  // CUDA: ceilf((blockDim.y*PPT*gridDim.y - (is_height + size_y)) /
  //      (float)(blockDim.y*PPT))
  // OpenCL: ceilf((get_local_size(1)*PPT*get_num_groups(1) - (is_height + size_y)) /
  //      (float)(get_local_size(1)*PPT))
  numBlocksBHBottom = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
        createBinaryOperator(Ctx, local_size_y, createIntegerLiteral(Ctx,
            Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy), grid_size_y,
        BO_Mul, Ctx.IntTy), createParenExpr(Ctx, createBinaryOperator(Ctx,
            isHeight, createIntegerLiteral(Ctx, Kernel->getMaxSizeY()), BO_Sub,
            Ctx.IntTy)), BO_Sub, Ctx.IntTy); 
  numBlocksBHBottom = createBinaryOperator(Ctx, createParenExpr(Ctx,
        numBlocksBHBottom), createCStyleCastExpr(Ctx, ceilFD->getResultType(),
          CK_IntegralToFloating, createParenExpr(Ctx, createBinaryOperator(Ctx,
              local_size_y, createIntegerLiteral(Ctx,
                Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy)), NULL, NULL),
      BO_Div, Ctx.IntTy);

  args.clear();
  args.push_back(numBlocksBHBottom);
  numBlocksBHBottom = createFunctionCall(Ctx, ceilFD, args); 

  // only create labels if we need border handling
  for (int i=0; i<=9 && border_handling; i++) {
    LabelDecl *LD;
    BinaryOperator *if_goto = NULL;

    switch (i) {
      case 0:
        // fall back: in case the image is too small, use code variant with
        // boundary handling for all borders
        LD = createLabelDecl(Ctx, kernelDecl, "BH_FB");
        // CUDA: if (gridDim.x < numBlocksBHLeft + numBlocksBHRight ||
        //          gridDim.y < numBlocksBHTop + numBlocksBHBottom) goto BO_TL;
        // OpenCL: if (get_num_groups(0) < numBlocksBHLeft + numBlocksBHRight ||
        //            get_num_groups(1) < numBlocksBHTop + numBlocksBHBottom)
        //            goto BO_TL;
        if (kernel_x) {
          if_goto = createBinaryOperator(Ctx, grid_size_x,
              createBinaryOperator(Ctx, numBlocksBHLeft, numBlocksBHRight,
                BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);
        }
        if (kernel_y) {
          if (kernel_x) {
            if_goto = createBinaryOperator(Ctx, createBinaryOperator(Ctx,
                  grid_size_y, createBinaryOperator(Ctx, numBlocksBHTop,
                    numBlocksBHBottom, BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy),
                if_goto, BO_LOr, Ctx.BoolTy);
          } else {
            if_goto = createBinaryOperator(Ctx, grid_size_y,
                createBinaryOperator(Ctx, numBlocksBHTop, numBlocksBHBottom,
                  BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);
          }
        }
        break;
      case 1:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_TL");
        // CUDA: if (blockIdx.x < numBlocksBHLeft && blockIdx.y <
        //      numBlocksBHTop) goto BO_TL;
        // OpenCL: if (get_group_id(0) < numBlocksBHLeft &&
        //            get_group_id(1) < numBlocksBHTop) goto BO_TL;
        if_goto = createBinaryOperator(Ctx, block_id_x, numBlocksBHLeft, BO_LT,
            Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              block_id_y, numBlocksBHTop, BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 2:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_TR");
        // CUDA: if (blockIdx.x >= gridDim.x-numBlocksBHRight
        //        && blockIdx.y < numBlocksBHTop) goto BO_TR;
        // OpenCL: if (get_group_id(0) >= get_num_groups(0) - numBlocksBHRight
        //        && get_group_id(1) < numBlocksBHTop) goto BO_TR;
        if_goto = createBinaryOperator(Ctx, block_id_x,
            createBinaryOperator(Ctx, grid_size_x, numBlocksBHRight, BO_Sub,
              Ctx.IntTy), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              block_id_y, numBlocksBHTop, BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 3:
        // check if we have only a row filter
        if (!kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_T");
        // CUDA: if (blockIdx.y < numBlocksBHTop) goto BO_T;
        // OpenCL: if (get_group_id(1) < numBlocksBHTop) goto BO_T;
        if_goto = createBinaryOperator(Ctx, block_id_y, numBlocksBHTop, BO_LT,
            Ctx.BoolTy);
        break;
      case 4:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_BL");
        // CUDA: if (blockIdx.y >= gridDim.y-numBlocksBHBottom
        //        && blockIdx.x < numBlocksBHLeft) goto BO_BL;
        // OpenCL: if (get_group_id(1) >= get_num_groups(1) - numBlocksBHBottom
        //          && get_group_id(0) < numBlocksBHLeft) goto BO_BL;
        if_goto = createBinaryOperator(Ctx, block_id_y,
            createBinaryOperator(Ctx, grid_size_y, numBlocksBHBottom, BO_Sub,
              Ctx.IntTy), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              block_id_x, numBlocksBHLeft, BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 5:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_BR");
        // CUDA: if (blockIdx.y >= gridDim.y-numBlocksBHBottom
        //        && blockIdx.x >= gridDim.x-numBlocksBHRight) goto BO_BR;
        // OpenCL: if (get_group_id(1) >= get_num_groups(1) - numBlocksBHBottom
        //          && get_group_id(0) >= get_num_groups(0) - numBlocksBHRight)
        //          goto BO_BL;
        if_goto = createBinaryOperator(Ctx, block_id_y,
            createBinaryOperator(Ctx, grid_size_y, numBlocksBHBottom, BO_Sub,
              Ctx.IntTy), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              block_id_x, createBinaryOperator(Ctx, grid_size_x,
                numBlocksBHRight, BO_Sub, Ctx.IntTy), BO_GE, Ctx.BoolTy),
            BO_LAnd, Ctx.BoolTy);
        break;
      case 6:
        // check if we have only a row filter
        if (!kernel_y) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_B");
        // CUDA: if (blockIdx.y >= gridDim.y-numBlocksBHBottom) goto BO_B;
        // OpenCL: if (get_group_id(1) >= get_num_groups(1) - numBlocksBHBottom)
        //            goto BO_B;
        if_goto = createBinaryOperator(Ctx, block_id_y,
            createBinaryOperator(Ctx, grid_size_y, numBlocksBHBottom, BO_Sub,
              Ctx.IntTy), BO_GE, Ctx.BoolTy);
        break;
      case 7:
        // check if we have only a column filter
        if (!kernel_x) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_R");
        // CUDA: if (blockIdx.x >= gridDim.x-numBlocksBHRight) goto BO_R;
        // OpenCL: if (get_group_id(0) >= get_num_groups(0) - numBlocksBHRight)
        //            goto BO_R;
        if_goto = createBinaryOperator(Ctx, block_id_x,
            createBinaryOperator(Ctx, grid_size_x, numBlocksBHRight, BO_Sub,
              Ctx.IntTy), BO_GE, Ctx.BoolTy);
        break;
      case 8:
        // check if we have only a column filter
        if (!kernel_x) continue;

        LD = createLabelDecl(Ctx, kernelDecl, "BH_L");
        // CUDA: if (blockIdx.x < numBlocksBHLeft) goto BO_L;
        // OpenCL: if (get_group_id(0) < numBlocksBHLeft) goto BO_L;
        if_goto = createBinaryOperator(Ctx, block_id_x, numBlocksBHLeft, BO_LT,
            Ctx.BoolTy);
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
        if_goto = createBinaryOperator(Ctx, local_size_x,
            createIntegerLiteral(Ctx, 16), BO_GE, Ctx.BoolTy);
        break;
      default:
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
          imgBorder.top = 1;
          imgBorder.bottom = 1;
        }
        if (kernel_x) {
          imgBorder.left = 1;
          imgBorder.right = 1;
        }
        break;
      case 1:
        if (!kernel_x || !kernel_y) continue;
        imgBorder.top = 1;
        imgBorder.left = 1;
        break;
      case 2:
        if (!kernel_x || !kernel_y) continue;
        imgBorder.top = 1;
        imgBorder.right = 1;
        break;
      case 3:
        if (kernel_y) imgBorder.top = 1;
        else continue;
        break;
      case 4:
        if (!kernel_x || !kernel_y) continue;
        imgBorder.bottom = 1;
        imgBorder.left = 1;
        break;
      case 5:
        if (!kernel_x || !kernel_y) continue;
        imgBorder.bottom = 1;
        imgBorder.right = 1;
        break;
      case 6:
        if (kernel_y) imgBorder.bottom = 1;
        else continue;
        break;
      case 7:
        if (kernel_x) imgBorder.right = 1;
        else continue;
        break;
      case 8:
        if (kernel_x) imgBorder.left = 1;
        else continue;
        break;
      case 9:
        break;
      default:
        break;
    }

    // stage pixels into shared memory
    llvm::SmallVector<Stmt *, 16> labelBody;
    bool barrier_required = false;
    if (useShared) {
      barrier_required = stage_first_iteration_to_shared_memory(labelBody);
    }
    if (barrier_required) {
      // add memory barrier loading data
      llvm::SmallVector<Expr *, 16> args;
      if (compilerOptions.emitCUDA()) {
        labelBody.push_back(createFunctionCall(Ctx, barrier, args));
      } else {
        // TODO: pass CLK_LOCAL_MEM_FENCE argument to barrier()
        args.push_back(createIntegerLiteral(Ctx, 0));
        labelBody.push_back(createFunctionCall(Ctx, barrier, args));
      }
    }

    // if (gid_x >= is_offset_x && gid_x < is_width+is_offset_x)
    BinaryOperator *check_bop = NULL;
    if (!border_handling) {
      if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
        check_bop = createBinaryOperator(Ctx, gidXRef,
            Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(), BO_GE,
            Ctx.BoolTy);
        check_bop = createBinaryOperator(Ctx, check_bop,
            createBinaryOperator(Ctx, gidXRef, createBinaryOperator(Ctx,
                isWidth,
                Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(),
                BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
      } else {
        check_bop = createBinaryOperator(Ctx, gidXRef, isWidth, BO_LT,
            Ctx.BoolTy);
      }
    } else {
      // if (gid_x >= is_offset_x)
      if (border_handling && imgBorder.left &&
          Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
        check_bop = createBinaryOperator(Ctx, gidXRef,
            Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(), BO_GE,
            Ctx.BoolTy);
      }
      // if (gid_x < is_width+is_offset_x)
      if (border_handling && imgBorder.right) {
        if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
          check_bop = createBinaryOperator(Ctx, gidXRef,
              createBinaryOperator(Ctx, isWidth,
                Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(),
                BO_Add, Ctx.IntTy), BO_LT, Ctx.BoolTy);
        } else {
          check_bop = createBinaryOperator(Ctx, gidXRef, isWidth, BO_LT,
              Ctx.BoolTy);
        }
      }
    }

    for (unsigned int p=0; p<Kernel->getPixelsPerThread(); p++) {
      // calculate multiple pixels per thread
      llvm::SmallVector<Stmt *, 16> pptBody;

      // clear all stored decls before cloning, otherwise existing
      // VarDecls will be reused and we will miss declarations
      KernelDeclMap.clear();

      // update gid_y to gid_y + p*local_size_y
      gidYRef = createBinaryOperator(Ctx, gid_y_ref, createBinaryOperator(Ctx,
            createIntegerLiteral(Ctx, p), local_size_y, BO_Mul, Ctx.IntTy),
          BO_Add, Ctx.IntTy);
#ifdef TEST_UPDATE_SMEM
#else
      // update lid_y to lid_y + p*local_size_y
      lidYRef = createBinaryOperator(Ctx, local_id_y, createBinaryOperator(Ctx,
            createIntegerLiteral(Ctx, p), local_size_y, BO_Mul, Ctx.IntTy),
          BO_Add, Ctx.IntTy);
#endif

      // load next iteration to shared memory
      if (barrier_required && p<Kernel->getPixelsPerThread()-1) {
        stage_next_iteration_to_shared_memory(labelBody);
      }

      // convert kernel function body to CUDA/OpenCL kernel syntax
      Stmt* clonedStmt = Clone(S);
      assert(isa<CompoundStmt>(clonedStmt) && "CompoundStmt for kernel function body expected!");

      // add iteration space check when calculating multiple pixels per thread,
      // having a tiling with multiple threads in the y-dimension, or in case
      // exploration is done
      if ((border_handling && imgBorder.bottom) ||
          Kernel->getPixelsPerThread()>1 || Kernel->getNumThreadsY()>1 ||
          compilerOptions.exploreConfig()) {
        // if (gid_y + p < is_height)
        BinaryOperator *inner_check_bop = createBinaryOperator(Ctx, gidYRef,
            isHeight, BO_LT, Ctx.BoolTy);
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


      // add memory barrier loading data
      // reorder shared memory
      if (barrier_required && p<Kernel->getPixelsPerThread()-1) {
#ifdef TEST_UPDATE_SMEM
        update_shared_memory_for_next_iteration(labelBody);
#endif
        llvm::SmallVector<Expr *, 16> args;
        if (compilerOptions.emitCUDA()) {
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
        } else {
          // TODO: pass CLK_LOCAL_MEM_FENCE argument to barrier()
          args.push_back(createIntegerLiteral(Ctx, 0));
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
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
    imgBorderVal = 0;
    // reset gid_y
    gidYRef = gid_y_ref;
    // reset lid_y
    lidYRef = local_id_y;
  }

  if (border_handling) {
    kernelBody.push_back(LSExit);
  }
  CompoundStmt *CS = createCompoundStmt(Ctx, kernelBody);

  return CS;
}


template<class T> T *ASTTranslate::CloneDecl(T *D) {
  if (D == NULL) {
    return NULL;
  }

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
        name += "4";
        QT = PVD->getType();

        if (Kernel->vectorize()) {
          QT = simdTypes.getSIMDType(PVD, SIMD4);
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

        if (Kernel->vectorize()) {
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
        VD->getStorageClass(), VD->getStorageClassAsWritten());
    // set VarDecl as being used - required for CodeGen
    result->setUsed(true);

    if (!PVD && Kernel->vectorize() &&
        KernelClass->getVectorizeInfo(VD) == VECTORIZE) {
      result->setInit(simdTypes.propagate(VD, Clone(VD->getInit())));
    } else {
      result->setInit(Clone(VD->getInit()));
    }
    result->setThreadSpecified(VD->isThreadSpecified());
    result->setInitStyle(VD->getInitStyle());

    // store mapping between original VarDecl and cloned VarDecl
    if (PVD) {
      KernelDeclMapVector[PVD] = result;
    } else {
      if (convMask) LambdaDeclMap[VD] = result;
      else KernelDeclMap[VD] = result;
    }

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return static_cast<T *>(result);
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
        D->getStorageClass(), D->getStorageClassAsWritten());

    result->setInit(Clone(D->getInit()));
    result->setThreadSpecified(D->isThreadSpecified());
    result->setInitStyle(D->getInitStyle());

    // store mapping between original VarDecl and cloned VarDecl
    KernelDeclMapTex[D] = result;
    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


template<class T> T *ASTTranslate::Clone(T *S) {
  if (S == NULL) {
    return NULL;
  }

  T *clonedStmt = static_cast<T *>(Visit(S));

  return clonedStmt;
}


// normal statements
Stmt *ASTTranslate::VisitStmt(Stmt *) {
  assert(0 && "Hipacc: Stumbled upon 'Stmt', implementation of any derived class missing?");

  return NULL;
}

Stmt *ASTTranslate::VisitNullStmt(NullStmt *S) {
  return new (Ctx) NullStmt(S->getSemiLoc());
}

Stmt *ASTTranslate::VisitCompoundStmt(CompoundStmt *S) {
  unsigned int numStmts = S->size();
  CompoundStmt* result = new (Ctx) CompoundStmt(Ctx, NULL, 0, S->getLBracLoc(),
      S->getLBracLoc());

  if (numStmts > 0) {
    llvm::SmallVector<Stmt *, 16> body;
    for (CompoundStmt::const_body_iterator I=S->body_begin(), E=S->body_end();
        I!=E; ++I) {
      curCompoundStmtVistor = S;
      Stmt *newS = Clone(*I);
      curCompoundStmtVistor = S;

      if (bhStmtsVistor.size()) {
        unsigned int num_stmts = 0;
        for (unsigned int i=0, e=bhStmtsVistor.size(); i!=e; ++i) {
          if (bhCStmtsVistor.data()[i]==S) {
            body.push_back(bhStmtsVistor.data()[i]);
            num_stmts++;
          }
        }
        for (unsigned int i=0; i<num_stmts; i++) {
          bhStmtsVistor.pop_back();
          bhCStmtsVistor.pop_back();
        }
      }

      body.push_back(newS);
    }
    result->setStmts(Ctx, body.data(), body.size());
  }

  return result;
}

Stmt *ASTTranslate::VisitSwitchCase(SwitchCase *S) {
  assert(0 && "Hipacc: Stumbled upon 'SwitchCase', implementation of any derived class missing?");

  return NULL;
}

Stmt *ASTTranslate::VisitCaseStmt(CaseStmt *S) {
  CaseStmt* result = new (Ctx) CaseStmt(Clone(S->getLHS()), Clone(S->getRHS()),
      S->getCaseLoc(), S->getEllipsisLoc(), S->getColonLoc());

  result->setSubStmt(Clone(S->getSubStmt()));

  return result;
}

Stmt *ASTTranslate::VisitDefaultStmt(DefaultStmt *S) {
  return new (Ctx) DefaultStmt(S->getDefaultLoc(), S->getColonLoc(),
      Clone(S->getSubStmt()));
}

Stmt *ASTTranslate::VisitLabelStmt(LabelStmt *S) {
  return new (Ctx) LabelStmt(S->getIdentLoc(), S->getDecl(),
      Clone(S->getSubStmt()));
}

Stmt *ASTTranslate::VisitIfStmt(IfStmt *S) {
  return new (Ctx) IfStmt(Ctx, S->getIfLoc(),
      CloneDecl(S->getConditionVariable()), Clone(S->getCond()),
      Clone(S->getThen()), S->getElseLoc(), Clone(S->getElse()));
}

Stmt *ASTTranslate::VisitSwitchStmt(SwitchStmt *S) {
  SwitchStmt* result = new (Ctx) SwitchStmt(Ctx,
      CloneDecl(S->getConditionVariable()), Clone(S->getCond()));

  result->setBody(Clone(S->getBody()));
  result->setSwitchLoc(S->getSwitchLoc());

  return result;
}

Stmt *ASTTranslate::VisitWhileStmt(WhileStmt *S) {
  return new (Ctx) WhileStmt(Ctx, CloneDecl(S->getConditionVariable()),
      Clone(S->getCond()), Clone(S->getBody()), S->getWhileLoc());
}

Stmt *ASTTranslate::VisitDoStmt(DoStmt *S) {
  return new (Ctx) DoStmt(Clone(S->getBody()), Clone(S->getCond()),
      S->getDoLoc(), S->getWhileLoc(), S->getRParenLoc());
}

Stmt *ASTTranslate::VisitForStmt(ForStmt *S) {
  return new (Ctx) ForStmt(Ctx, Clone(S->getInit()), Clone(S->getCond()),
      CloneDecl(S->getConditionVariable()), Clone(S->getInc()),
      Clone(S->getBody()), S->getForLoc(), S->getLParenLoc(),
      S->getRParenLoc());
}

Stmt *ASTTranslate::VisitGotoStmt(GotoStmt *S) {
  return new (Ctx) GotoStmt(S->getLabel(), S->getGotoLoc(), S->getLabelLoc());
}

Stmt *ASTTranslate::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  return new (Ctx) IndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
      Clone(S->getTarget()));
}

Stmt *ASTTranslate::VisitContinueStmt(ContinueStmt *S) {
  return new (Ctx) ContinueStmt(S->getContinueLoc());
}

Stmt *ASTTranslate::VisitBreakStmt(BreakStmt *S) {
  return new (Ctx) BreakStmt(S->getBreakLoc());
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
        convRedExpr = createBinaryOperator(Ctx, convRed, retVal, BO_AddAssign,
            convRed->getType());
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
        convRedExpr = createBinaryOperator(Ctx, convRed, retVal, BO_MulAssign,
            convRed->getType());
        break;
      case HipaccMEDIAN:
      default:
        assert(0 && "Unsupported convolution mode.");
        break;
    }

    if (convMask->isConstant()) {
      if (convIdxX + convIdxY == 0) {
        // conv_tmp = ...
        return convInitExpr;
      } else {
        // conv_tmp += ...
        return convRedExpr;
      }
    } else {
      // if (conv_x == 0 && conv_y == 0) {
      //   conv_tmp = ...
      // } else {
      //   conv_tmp += ...
      // }
      return createIfStmt(Ctx, createBinaryOperator(Ctx,
            createBinaryOperator(Ctx, convExprX, createIntegerLiteral(Ctx, 0),
              BO_EQ, Ctx.BoolTy), createBinaryOperator(Ctx, convExprY,
                createIntegerLiteral(Ctx, 0), BO_EQ, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy), convInitExpr, convRedExpr);
    }
  } else {
    return new (Ctx) ReturnStmt(S->getReturnLoc(), Clone(S->getRetValue()), 0);
  }
}

Stmt *ASTTranslate::VisitDeclStmt(DeclStmt *S) {
  DeclGroupRef clonedDecls;

  if (S->isSingleDecl()) {
    clonedDecls = DeclGroupRef(CloneDecl(S->getSingleDecl()));
  } else if (S->getDeclGroup().isDeclGroup()) {
    llvm::SmallVector<Decl *, 16> clonedDeclGroup;
    const DeclGroupRef& DG = S->getDeclGroup();

    for (DeclGroupRef::const_iterator I=DG.begin(), E=DG.end(); I!=E; ++I) {
      clonedDeclGroup.push_back(CloneDecl(*I));
    }

    clonedDecls = DeclGroupRef(DeclGroup::Create(Ctx,
          clonedDeclGroup.data(), clonedDeclGroup.size()));
  }

  return new (Ctx) DeclStmt(clonedDecls, S->getStartLoc(), S->getEndLoc());
}

Stmt *ASTTranslate::VisitAsmStmt(AsmStmt *S) {
  llvm::SmallVector<IdentifierInfo *, 16> names;
  llvm::SmallVector<StringLiteral *, 16> constraints;
  llvm::SmallVector<Expr *, 16> exprs;

  // outputs
  for (unsigned int I=0, N=S->getNumOutputs(); I!=N; ++I) {
    names.push_back(S->getOutputIdentifier(I));
    constraints.push_back(S->getOutputConstraintLiteral(I));
    exprs.push_back(Clone(S->getOutputExpr(I)));
  }

  // inputs
  for (unsigned int I=0, N=S->getNumInputs(); I!=N; ++I) {
    names.push_back(S->getInputIdentifier(I));
    constraints.push_back(S->getInputConstraintLiteral(I));
    exprs.push_back(Clone(S->getInputExpr(I)));
  }

  // constraints
  llvm::SmallVector<StringLiteral *, 16> clobbers;
  for (unsigned int I=0; I!=S->getNumClobbers(); ++I) {
    clobbers.push_back(S->getClobber(I));
  }

  return new (Ctx) AsmStmt(Ctx, S->getAsmLoc(), S->isSimple(), S->isVolatile(),
      S->isMSAsm(), S->getNumOutputs(), S->getNumInputs(), names.data(),
      constraints.data(), exprs.data(), S->getAsmString(), S->getNumClobbers(),
      clobbers.data(), S->getRParenLoc());
}

Expr *ASTTranslate::VisitExpr(Expr *E) {
  assert(0 && "Hipacc: Stumbled upon 'Expr', implementation of any derived class missing?");

  return NULL;
}

Expr *ASTTranslate::VisitPredefinedExpr(PredefinedExpr *E) {
  Expr* result = new (Ctx) PredefinedExpr(E->getLocation(), E->getType(),
      E->getIdentType());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitDeclRefExpr(DeclRefExpr *E) {
  TemplateArgumentListInfo templateArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned int i=0, e=E->getNumTemplateArgs(); i!=e; ++i) {
    templateArgs.addArgument(E->getTemplateArgs()[i]);
  }

  ValueDecl *VD = CloneDecl(E->getDecl());

  DeclRefExpr *result = DeclRefExpr::Create(Ctx, E->getQualifierLoc(),
      E->getTemplateKeywordLoc(), VD, E->refersToEnclosingLocal(),
      E->getLocation(), VD->getType(), E->getValueKind(), E->getFoundDecl(),
      E->getNumTemplateArgs()?&templateArgs:0);

  result->setObjectKind(E->getObjectKind());
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitLambdaExpr(LambdaExpr *E) {
  llvm::errs() << "Found LambdaExpr:\n";
  E->dump();
  llvm::errs() << "\n";

  return E;
}

Expr *ASTTranslate::VisitIntegerLiteral(IntegerLiteral *E) {
  Expr *result = IntegerLiteral::Create(Ctx, E->getValue(), E->getType(),
      E->getLocation());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitFloatingLiteral(FloatingLiteral *E) {
  Expr *result = FloatingLiteral::Create(Ctx, E->getValue(), E->isExact(),
      E->getType(), E->getLocation());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  Expr *result = new (Ctx) ImaginaryLiteral(Clone(E->getSubExpr()),
      E->getType());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitStringLiteral(StringLiteral *E) {
  llvm::SmallVector<SourceLocation, 16> concatLocations(E->tokloc_begin(),
      E->tokloc_end());

  Expr *result = StringLiteral::Create(Ctx, E->getString(), E->getKind(),
      E->isPascal(), E->getType(), concatLocations.data(),
      concatLocations.size());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCharacterLiteral(CharacterLiteral *E) {
  Expr *result = new (Ctx) CharacterLiteral(E->getValue(), E->getKind(),
      E->getType(), E->getLocation());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitParenExpr(ParenExpr *E) {
  Expr *result = new (Ctx) ParenExpr(E->getLParen(), E->getRParen(),
      Clone(E->getSubExpr()));

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitParenListExpr(ParenListExpr *E) {
  llvm::SmallVector<Expr *, 16> Exprs;

  for (unsigned int i=0, e=E->getNumExprs(); i<e; ++i) {
    Exprs.push_back(Clone(E->getExpr(i)));
  }

  Expr *result = new (Ctx) ParenListExpr(Ctx, E->getLParenLoc(), Exprs.data(),
      Exprs.size(), E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitUnaryOperator(UnaryOperator *E) {
  Expr *result = new (Ctx) UnaryOperator(Clone(E->getSubExpr()), E->getOpcode(),
      E->getType(), E->getValueKind(), E->getObjectKind(), E->getOperatorLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitOffsetOfExpr(OffsetOfExpr *E) {
  OffsetOfExpr *result = OffsetOfExpr::CreateEmpty(Ctx, E->getNumExpressions(),
      E->getNumComponents());

  result->setType(E->getType());
  result->setOperatorLoc(E->getOperatorLoc());
  result->setTypeSourceInfo(E->getTypeSourceInfo());
  result->setRParenLoc(E->getRParenLoc());

  for (unsigned int I=0, N=E->getNumComponents(); I!=N; ++I) {
    result->setComponent(I, E->getComponent(I));
  }

  for (unsigned int I=0, N=E->getNumExpressions(); I<N; ++I) {
    result->setIndexExpr(I, Clone(E->getIndexExpr(I)));
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) { 
  Expr *result;

  if (E->isArgumentType()) {
    result = new (Ctx) UnaryExprOrTypeTraitExpr(E->getKind(),
        E->getArgumentTypeInfo(), E->getType(), E->getOperatorLoc(),
        E->getRParenLoc());
  } else {
    result = new (Ctx) UnaryExprOrTypeTraitExpr(E->getKind(),
        Clone(E->getArgumentExpr()), E->getType(), E->getOperatorLoc(),
        E->getRParenLoc());
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}


Expr *ASTTranslate::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  Expr *result = new (Ctx) ArraySubscriptExpr(Clone(E->getLHS()),
      Clone(E->getRHS()), E->getType(), E->getValueKind(), E->getObjectKind(),
      E->getRBracketLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCallExpr(CallExpr *E) {
  FunctionDecl *targetFD;

  if (E->getDirectCallee()) {
    QualType QT = E->getCallReturnType();
    if (Kernel->vectorize()) {
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
      
      // check if Mask is initialized with constants
      if (!Mask->isConstant()) {
        unsigned int DiagIDConstMask =
          Diags.getCustomDiagID(DiagnosticsEngine::Warning,
              "Unable to unroll convolution loop and propagate constants: Mask '%0' for 'convolve' call needs to be initialized using constants.");
        Diags.Report(E->getArg(0)->getExprLoc(), DiagIDConstMask) << FD->getName();
        unsigned int DiagIDConstMaskInit =
          Diags.getCustomDiagID(DiagnosticsEngine::Warning,
              "With the following Mask declaration:");
        Diags.Report(Mask->getDecl()->getLocation(), DiagIDConstMaskInit);
      }

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
      CompoundStmt *outerCompountStmt = curCompoundStmtVistor;
      std::stringstream LSST;
      LSST << "_conv_tmp" << literalCount++;
      VarDecl *conv_tmp = createVarDecl(Ctx, kernelDecl, LSST.str(),
          E->getCallReturnType(), NULL);
      DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
      DC->addDecl(conv_tmp);
      DeclRefExpr *conv_red = createDeclRefExpr(Ctx, conv_tmp);
      convRed = conv_red;
      bhStmtsVistor.push_back(createDeclStmt(Ctx, conv_tmp));
      bhCStmtsVistor.push_back(outerCompountStmt);

      // unroll convolution
      convMask = Mask;

      if (!Mask->isConstant()) {
        // int _conv_x = 0;
        VarDecl *conv_x = createVarDecl(Ctx, kernelDecl, "_conv_x", Ctx.IntTy,
            createIntegerLiteral(Ctx, 0));
        DeclStmt *conv_x_stmt = createDeclStmt(Ctx, conv_x);
        convExprX  = createDeclRefExpr(Ctx, conv_x);

        // int _conv_y = 0;
        VarDecl *conv_y = createVarDecl(Ctx, kernelDecl, "_conv_y", Ctx.IntTy,
            createIntegerLiteral(Ctx, 0));
        DeclStmt *conv_y_stmt = createDeclStmt(Ctx, conv_y);
        convExprY = createDeclRefExpr(Ctx, conv_y);

        // convert the lambda-function body to kernel syntax
        Stmt* convIterations = Clone(LE->getBody());

        //
        // for (int _conv_y=0; _conv_y<size_y; _conv_y++) {
        //     for (int _conv_x=0; _conv_x<size_x; _conv_x++) {
        //         lambda-function body / iteration
        //     }
        // }
        //
        ForStmt *innerLoop = createForStmt(Ctx, conv_x_stmt,
            createBinaryOperator(Ctx, convExprX, createIntegerLiteral(Ctx,
                Mask->getSizeX()), BO_LT, Ctx.BoolTy), createUnaryOperator(Ctx,
              convExprX, UO_PostInc, convExprX->getType()), convIterations);
        ForStmt *outerLoop = createForStmt(Ctx, conv_y_stmt,
            createBinaryOperator(Ctx, convExprY, createIntegerLiteral(Ctx,
                Mask->getSizeY()), BO_LT, Ctx.BoolTy), createUnaryOperator(Ctx,
              convExprY, UO_PostInc, convExprY->getType()), innerLoop);

        bhStmtsVistor.push_back(outerLoop);
        bhCStmtsVistor.push_back(outerCompountStmt);
      } else {
        for (unsigned int y=0; y<Mask->getSizeY(); y++) {
          for (unsigned int x=0; x<Mask->getSizeX(); x++) {
            convIdxX = x;
            convIdxY = y;
            Stmt *convIteration = Clone(LE->getBody());
            bhStmtsVistor.push_back(convIteration);
            bhCStmtsVistor.push_back(outerCompountStmt);
            // clear decls added while cloning last iteration
            LambdaDeclMap.clear();
          }
        }
      }

      convMask = NULL;
      convRed = NULL;
      convIdxX = convIdxY = 0;
      convExprX = convExprY = NULL;

      return conv_red;
    }

    // lookup if this function call is supported and choose appropriate
    // function, e.g. exp() instead of expf() in case of OpenCL
    targetFD = builtins.getBuiltinFunction(E->getDirectCallee()->getName(), QT,
        compilerOptions.emitCUDA() ? hipacc::CUDA_TARGET :
        hipacc::OPENCL_TARGET);

    if (!targetFD) {
      DiagnosticsEngine &Diags = Ctx.getDiagnostics();
      unsigned int DiagIDCallExpr =
        Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Found unsupported function call '%0' in kernel.");
      llvm::SmallVector<const char *, 16> builtinNames;
      builtins.getBuiltinNames(hipacc::C_TARGET, builtinNames);
      Diags.Report(E->getExprLoc(), DiagIDCallExpr) << E->getDirectCallee()->getName();

      llvm::errs() << "Supported functions are: ";
      for (unsigned int i=0, e=builtinNames.size(); i!=e; ++i) {
        llvm::errs() << builtinNames.data()[i];
        llvm::errs() << ((i==e-1)?".\n":", ");
      }
      exit(EXIT_FAILURE);
    }
  } else {
    assert(0 && "CallExpr without FunctionDecl as Callee!");
  }

  // add ICE for CodeGen
  ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx,
      Ctx.getPointerType(targetFD->getType()), CK_FunctionToPointerDecay,
      createDeclRefExpr(Ctx, targetFD), NULL, VK_RValue);

  // create CallExpr
  CallExpr *result = new (Ctx) CallExpr(Ctx, ICE, NULL, 0, E->getType(),
      E->getValueKind(), E->getRParenLoc());

  result->setNumArgs(Ctx, E->getNumArgs());

  for (unsigned int I=0, N=E->getNumArgs(); I<N; ++I) {
    result->setArg(I, Clone(E->getArg(I)));
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitMemberExpr(MemberExpr *E) {
  // TODO: create a map with all expressions not to be cloned ..
  if (E==local_size_x) return E;
  if (E==local_size_y) return E;

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

      // get vector declaration
      if (Kernel->vectorize()) {
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
  if (!emitPolly) {
    for (unsigned int i=0; i<KernelClass->getNumMasks(); i++) {
      FieldDecl *FD = KernelClass->getMaskFields().data()[i];

      if (paramDecl->getName().equals(FD->getName())) {
        HipaccMask *Mask = Kernel->getMaskFromMapping(FD);

        if (Mask && (Mask->isConstant() || compilerOptions.emitCUDA())) {
          VarDecl *maskVar = NULL;
          // get Mask reference
          for (DeclContext::lookup_result Lookup =
              Ctx.getTranslationUnitDecl()->lookup(DeclarationName(&Ctx.Idents.get(Mask->getName()+Kernel->getName())));
              Lookup.first!=Lookup.second; ++Lookup.first) {
            maskVar = cast_or_null<VarDecl>(*Lookup.first);

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

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCastExpr(CastExpr *E) {
  assert(0 && "Hipacc: Stumbled upon 'CastExpr', implementation of any derived class missing?");

  return NULL;
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

  // writeImageRHS has changed, use LHS
  if (E->getOpcode() == BO_Assign && writeImageRHS && writeImageRHS!=RHS) {
    // TODO: insert checks +=, -=, /=, and *= are not supported on Image objects
    result = LHS;
  } else {
    // normal case: clone binary operator
    // use type of LHS, so that the widened type is used in case of
    // vectorization
    result = new (Ctx) BinaryOperator(LHS, RHS, E->getOpcode(), LHS->getType(),
        E->getValueKind(), E->getObjectKind(), E->getOperatorLoc());
  }
  if (E->getOpcode() == BO_Assign) writeImageRHS = NULL;

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  Expr *result = new (Ctx) CompoundAssignOperator(Clone(E->getLHS()),
      Clone(E->getRHS()), E->getOpcode(), E->getType(), E->getValueKind(),
      E->getObjectKind(), E->getComputationLHSType(),
      E->getComputationResultType(), E->getOperatorLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

// abstract base class
Expr *ASTTranslate::VisitAbstractConditionalOperator(AbstractConditionalOperator *E) {
  HIPACC_NOT_SUPPORTED(AbstractConditionalOperator);
  return NULL;
}

Expr *ASTTranslate::VisitConditionalOperator(ConditionalOperator *E) {
  Expr *result = new (Ctx) ConditionalOperator(Clone(E->getCond()),
      E->getQuestionLoc(), Clone(E->getLHS()), E->getColonLoc(),
      Clone(E->getRHS()), E->getType(), E->getValueKind(), E->getObjectKind());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitBinaryConditionalOperator(BinaryConditionalOperator *E)
{
  Expr *result = new (Ctx) BinaryConditionalOperator(Clone(E->getCommon()),
      Clone(E->getOpaqueValue()), Clone(E->getCond()), Clone(E->getTrueExpr()),
      Clone(E->getFalseExpr()), E->getQuestionLoc(), E->getColonLoc(),
      E->getType(), E->getValueKind(), E->getObjectKind());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitImplicitCastExpr(ImplicitCastExpr *E) {
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

  Expr *result = ImplicitCastExpr::Create(Ctx, QT, E->getCastKind(), subExpr,
      NULL, E->getValueKind());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  assert(0 && "Hipacc: Stumbled upon 'ExplicitCastExpr', implementation of any derived class missing?");

  return NULL;
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

  Expr *result = CStyleCastExpr::Create(Ctx, QT, E->getValueKind(),
      E->getCastKind(), subExpr, NULL, E->getTypeInfoAsWritten(),
      E->getLParenLoc(), E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  Expr *result = new (Ctx) CompoundLiteralExpr(E->getLParenLoc(),
      E->getTypeSourceInfo(), E->getType(), E->getValueKind(),
      Clone(E->getInitializer()), E->isFileScope());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  Expr *result = new (Ctx) ExtVectorElementExpr(E->getType(), E->getValueKind(),
      Clone(E->getBase()), E->getAccessor(), E->getAccessorLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitInitListExpr(InitListExpr *E) {
  llvm::SmallVector<Expr *, 16> initExprs;

  for (unsigned int I=0, N=E->getNumInits(); I<N; ++I) {
    initExprs.push_back(Clone(E->getInit(I)));
  }

  InitListExpr* result = new (Ctx) InitListExpr(Ctx, E->getLBraceLoc(),
      initExprs.data(), initExprs.size(), E->getRBraceLoc());

  result->setInitializedFieldInUnion(E->getInitializedFieldInUnion());

  // FIXME: clone the syntactic form, can this become recursive?
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  llvm::SmallVector<Expr *, 16> indexExprs;

  for (int I=0, N=indexExprs.size(); I<N; ++I) {
    indexExprs.push_back(Clone(E->getSubExpr(I)));
  }

  Expr *result = DesignatedInitExpr::Create(Ctx, E->getDesignator(0),
      E->size(), &indexExprs[0] + 1, indexExprs.size() - 1,    // no &indexExprs[1] 
      E->getEqualOrColonLoc(), E->usesGNUSyntax(), indexExprs[0]);

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  Expr *result = new (Ctx) ImplicitValueInitExpr(E->getType());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitVAArgExpr(VAArgExpr *E) {
  Expr *result = new (Ctx) VAArgExpr(E->getBuiltinLoc(), Clone(E->getSubExpr()),
      E->getWrittenTypeInfo(), E->getRParenLoc(), E->getType());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitAddrLabelExpr(AddrLabelExpr *E) {
  Expr *result = new (Ctx) AddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
      E->getLabel(), E->getType());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitStmtExpr(StmtExpr *E) {
  Expr *result = new (Ctx) StmtExpr(Clone(E->getSubStmt()), E->getType(),
      E->getLParenLoc(), E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitChooseExpr(ChooseExpr *E) {
  Expr *result = new (Ctx) ChooseExpr(E->getBuiltinLoc(), Clone(E->getCond()),
      Clone(E->getLHS()), Clone(E->getRHS()), E->getType(), E->getValueKind(),
      E->getObjectKind(), E->getRParenLoc(), E->isTypeDependent(),
      E->isValueDependent());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitGNUNullExpr(GNUNullExpr *E) {
  Expr *result = new (Ctx) GNUNullExpr(E->getType(), E->getTokenLocation());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  llvm::SmallVector<Expr *, 16> body;

  for (unsigned int I=0, N=E->getNumSubExprs(); I<N; ++I) {
    body.push_back(Clone(E->getExpr(I)));
  }

  Expr *result = new (Ctx) ShuffleVectorExpr(Ctx, body.data(),
      E->getNumSubExprs(), E->getType(), E->getBuiltinLoc(), E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitBlockExpr(BlockExpr *E) {
  HIPACC_NOT_SUPPORTED(BlockExpr);
  return NULL;
}

Expr *ASTTranslate::VisitGenericSelectionExpr(GenericSelectionExpr *E) {
  HIPACC_NOT_SUPPORTED(GenericSelectionExpr);
  return NULL;
}


// C++ statements
Stmt *ASTTranslate::VisitCXXCatchStmt(CXXCatchStmt *S) {
  return new (Ctx) CXXCatchStmt(S->getCatchLoc(),
      static_cast<VarDecl*>(CloneDecl(S->getExceptionDecl())),
      Clone(S->getHandlerBlock()));
}

Stmt *ASTTranslate::VisitCXXTryStmt(CXXTryStmt *S) {
  llvm::SmallVector<Stmt *, 16> handlers;

  for (unsigned int I=0, N=S->getNumHandlers(); I<N; ++I) {
    handlers.push_back((Stmt *)Clone(S->getHandler(I)));
  }

  return CXXTryStmt::Create(Ctx, S->getTryLoc(), Clone(S->getTryBlock()),
      handlers.data(), handlers.size());
}

Stmt *ASTTranslate::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  HIPACC_NOT_SUPPORTED(CXXForRangeStmt);
  return NULL;
}

Expr *ASTTranslate::VisitUserDefinedLiteral(UserDefinedLiteral *E) {
  HIPACC_NOT_SUPPORTED(UserDefinedLiteral);
  return NULL;
}

#if 0
Expr *ASTTranslate::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  CXXOperatorCallExpr *result = new (Ctx) CXXOperatorCallExpr(Ctx,
      E->getOperator(), Clone(E->getCallee()), NULL, 0, E->getType(),
      E->getRParenLoc());

  result->setNumArgs(Ctx, E->getNumArgs());

  for (unsigned int I=0, N=E->getNumArgs(); I<N; ++I) {
    result->setArg(I, Clone(E->getArg(I)));
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}
#else
Expr *ASTTranslate::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  bool found_mask = false;
  Expr *result = NULL;
  HipaccMask *Mask = NULL;

  // assume that all CXXOperatorCallExpr are memory access functions, since we
  // don't support function calls
  assert(isa<MemberExpr>(E->getArg(0)) && "Memory access function assumed.");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0));

  // get FieldDecl of the MemberExpr
  assert(isa<FieldDecl>(ME->getMemberDecl()) && "Image must be a C++-class member.");
  FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

  // find corresponding Image user class member variable
  MemoryAccess memAcc = UNDEFINED;
  if (Kernel->getImgFromMapping(FD)) {
    memAcc = KernelClass->getImgAccess(FD);
  }

  // look for Mask user class member variable
  if (memAcc == UNDEFINED) {
    if (Kernel->getMaskFromMapping(FD)) {
      Mask = Kernel->getMaskFromMapping(FD);
      memAcc = READ_ONLY;
      found_mask = true;
    }
  }
  assert(memAcc!=UNDEFINED && "Could not find Image/Accessor/Mask Field Decl.");

  // MemberExpr is converted to DeclRefExpr when cloning
  DeclRefExpr *LHS = dyn_cast<DeclRefExpr>(Clone(E->getArg(0)));

  if (found_mask) {
    if (emitPolly || compilerOptions.emitCUDA() ||
        Mask->isConstant()) {
      switch (E->getNumArgs()) {
        default:
          assert(0 && "0 or 2 arguments for Mask operator() expected!");
          break;
        case 1:
          assert(convMask && convMask==Mask && "0 arguments for Mask operator() only allowed within convolution lambda-function.");
          if (Mask->isConstant()) {
            // within convolute lambda-function propagate constant
            result =
              Clone(Mask->getInitList()->getInit(Mask->getSizeY()*convIdxX +
                    convIdxY)->IgnoreParenCasts());
            // in case CUDA code is generated, cast single-precision floating
            // point constants explicitly - implicit conversions are extensive
            // on older hardware (CC < 2.0)
            if (compilerOptions.emitCUDA() && Mask->getType() == Ctx.FloatTy) {
              result = createCStyleCastExpr(Ctx, Ctx.FloatTy, CK_FloatingCast,
                  result, NULL, NULL);
            }
          } else {
            // array subscript: Mask[conv_y][conv_x]
            result = accessMem2DAt(LHS, convExprX, convExprY);
          }
          break;
        case 3:
          // array subscript: Mask[y+size_y/2][x+size_x/2]
          // 0: -> (this *) Mask class
          // 1: -> x
          // 2: -> y
          result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                  (int)Mask->getSizeX()/2), BO_Add, Ctx.IntTy),
              createBinaryOperator(Ctx, Clone(E->getArg(2)),
                createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
                Ctx.IntTy));
          break;
      }
    } else {
      // array subscript: Mask[(y+size_y/2)*width + x+size_x/2]
      // 0: -> (this *) Mask class
      // 1: -> x
      // 2: -> y
      result = accessMemArrAt(LHS, createIntegerLiteral(Ctx, Mask->getSizeX()),
          createBinaryOperator(Ctx, Clone(E->getArg(1)),
            createIntegerLiteral(Ctx, (int)Mask->getSizeX()/2), BO_Add,
            Ctx.IntTy), createBinaryOperator(Ctx, Clone(E->getArg(2)),
              createIntegerLiteral(Ctx, (int)Mask->getSizeY()/2), BO_Add,
              Ctx.IntTy));
    }
    result->setValueDependent(E->isValueDependent());
    result->setTypeDependent(E->isTypeDependent());

    return result;
  }

  // Masks are handled before - Images are ParmVarDecls
  ParmVarDecl *PVD = NULL;
  if (!Kernel->vectorize()) {
    assert(isa<ParmVarDecl>(LHS->getDecl()) && "Image variable must be a ParmVarDecl!");
    PVD = dyn_cast<ParmVarDecl>(LHS->getDecl());
  }

  // get Image class for the current image
  HipaccAccessor *Acc = Kernel->getImgFromMapping(FD);

  // replace Image accesses by global memory access
  // Output(EI) = ...
  // ->
  // Output[gid_y * width + gid_x] = ...
  if (emitPolly) {
    switch (E->getNumArgs()) {
      default:
        assert(0 && "0 or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // no padding is considered, data is accessed as a 2D-array
        // 0: -> (this *) Image Class
        result = accessMemPolly(LHS, Acc, memAcc, NULL, NULL);
        break;
      case 3:
        // no padding is considered, data is accessed as a 2D-array
        // 0: -> (this *) Image Class
        // 1: -> offset x
        // 2: -> offset y
        result = accessMemPolly(LHS, Acc, memAcc, E->getArg(1), E->getArg(2));
        break;
    }
  } else if (KernelDeclMapShared[PVD]) {
    // shared/local memory
    VarDecl *VD = KernelDeclMapShared[PVD];
    DeclRefExpr *DRE = createDeclRefExpr(Ctx, VD);

    IntegerLiteral *SX, *SY;
    if (Acc->getSizeX() > 1) {
      SX = createIntegerLiteral(Ctx, Acc->getSizeX()/2);
    } else {
      SX = createIntegerLiteral(Ctx, 0);
    }
    if (Acc->getSizeY() > 1) {
      SY = createIntegerLiteral(Ctx, Acc->getSizeY()/2);
    } else {
      SY = createIntegerLiteral(Ctx, 0);
    }

    switch (E->getNumArgs()) {
      default:
        assert(0 && "0 or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // 0: -> (this *) Image Class
        result = accessMemShared(DRE, SX, SY);
        break;
      case 3:
        // 0: -> (this *) Image Class
        // 1: -> offset x
        // 2: -> offset y
        result = accessMemShared(DRE, createBinaryOperator(Ctx, E->getArg(1),
              SX, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx, E->getArg(2),
                SY, BO_Add, Ctx.IntTy));
        break;
    }
  } else {
    switch (E->getNumArgs()) {
      default:
        assert(0 && "0 or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // 0: -> (this *) Image Class
        result = accessMem(LHS, Acc, memAcc);
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
          if (convMask->isConstant()) {
            offset_x = createIntegerLiteral(Ctx,
                convIdxX-(int)convMask->getSizeX()/2);
            offset_y = createIntegerLiteral(Ctx,
                convIdxY-(int)convMask->getSizeY()/2);
          } else {
            offset_x = createBinaryOperator(Ctx, convExprX,
                createIntegerLiteral(Ctx, (int)convMask->getSizeX()/2), BO_Sub,
                Ctx.IntTy);
            offset_y = createBinaryOperator(Ctx, convExprY,
                createIntegerLiteral(Ctx, (int)convMask->getSizeY()/2), BO_Sub,
                Ctx.IntTy);
          }
        } else {
          offset_x = E->getArg(1);
          offset_y = E->getArg(2);
        }

        switch (memAcc) {
          case READ_ONLY:
            if (Acc->getBoundaryHandling() != BOUNDARY_UNDEFINED) {
              return addBorderHandling(LHS, offset_x, offset_y, Acc);
            }
            // fall through
          case WRITE_ONLY:
            result = accessMem(LHS, Acc, memAcc, offset_x, offset_y);
            break;
          case UNDEFINED:
          case READ_WRITE:
          default:
            assert(0 && "Unsupported memory access with offset specification!\n");
            break;
        }
        break;
    }
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}
#endif


#if 0
Expr *ASTTranslate::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  CXXMemberCallExpr *result = new (Ctx) CXXMemberCallExpr(Ctx,
      Clone(E->getCallee()), NULL, 0, E->getType(), E->getRParenLoc());

  result->setNumArgs(Ctx, E->getNumArgs());

  for (unsigned int I=0, N=E->getNumArgs(); I<N; ++I) {
    result->setArg(I, Clone(E->getArg(I)));
  }

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}
#else
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
      return removeISOffsetX(gidXRef, Acc);
    }

    // getY() method -> gid_y
    if (ME->getMemberNameInfo().getAsString() == "getY") {
      return gidYRef;
    }

    // output() method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output") {
      assert(E->getNumArgs()==0 && "no arguments for output() method supported!");

      if (emitPolly) {
        // no padding is considered, data is accessed as a 2D-array
        result = accessMemPolly(LHS, Acc, memAcc, NULL, NULL);
      } else {
        result = accessMem(LHS, Acc, memAcc);
      }

      result->setValueDependent(E->isValueDependent());
      result->setTypeDependent(E->isTypeDependent());

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
    if (Kernel->getImgFromMapping(FD)) {
      memAcc = KernelClass->getImgAccess(FD);
    }
    assert(memAcc!=UNDEFINED && "Could not find Image/Accessor Field Decl.");

    // Acc.getX() method -> acc_scale_x * (gid_x - is_offset_x)
    if (ME->getMemberNameInfo().getAsString() == "getX") {
      Expr *idx_x = gidXRef;
      // remove is_offset_x and scale index to Accessor size
      if (Acc->getInterpolation()!=InterpolateNO) {
        idx_x = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), NULL, NULL);
      } else {
        idx_x = removeISOffsetX(gidXRef, Acc);
      }

      return idx_x;
    }

    // Acc.getY() method -> acc_scale_y * gid_y
    if (ME->getMemberNameInfo().getAsString() == "getY") {
      Expr *idx_y = gidYRef;
      // scale index to Accessor size
      if (Acc->getInterpolation()!=InterpolateNO) {
        idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), NULL, NULL);
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

    if (emitPolly) {
      result = accessMem2DAt(LHS, idx_x, idx_y);
    } else {
      if (Kernel->useTextureMemory(Acc)) {
        if (compilerOptions.emitCUDA()) {
          result = accessMemTexAt(LHS, Acc, memAcc, idx_x, idx_y);
        } else {
          result = accessMemImgAt(LHS, Acc, memAcc, idx_x, idx_y);
        }
      } else {
        result = accessMemArrAt(LHS, Acc->getStrideDecl(), idx_x, idx_y);
      }
    }

    result->setValueDependent(E->isValueDependent());
    result->setTypeDependent(E->isTypeDependent());

    return result;
  }

  HIPACC_NOT_SUPPORTED(CXXMemberCallExpr);
  return NULL;
}
#endif

Expr *ASTTranslate::VisitCXXConstructExpr(CXXConstructExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXConstructExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXTemporaryObjectExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNamedCastExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  CXXStaticCastExpr *result = CXXStaticCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()), NULL,
      E->getTypeInfoAsWritten(), E->getOperatorLoc(), E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  CXXDynamicCastExpr *result = CXXDynamicCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()), NULL,
      E->getTypeInfoAsWritten(), E->getOperatorLoc(), E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  CXXReinterpretCastExpr *result = CXXReinterpretCastExpr::Create(Ctx,
      E->getType(), E->getValueKind(), E->getCastKind(), Clone(E->getSubExpr()),
      NULL, E->getTypeInfoAsWritten(), E->getOperatorLoc(), E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  CXXConstCastExpr *result = CXXConstCastExpr::Create(Ctx, E->getType(),
      E->getValueKind(), Clone(E->getSubExpr()), E->getTypeInfoAsWritten(),
      E->getOperatorLoc(), E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  CXXFunctionalCastExpr *result = CXXFunctionalCastExpr::Create(Ctx,
      E->getType(), E->getValueKind(), E->getTypeInfoAsWritten(),
      E->getTypeBeginLoc(), E->getCastKind(), Clone(E->getSubExpr()), NULL,
      E->getRParenLoc());

  // FIXME: set CXXCastPath
  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  Expr *result = new (Ctx) CXXBoolLiteralExpr(E->getValue(), E->getType(),
      E->getSourceRange().getBegin());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  Expr *result = new (Ctx) CXXNullPtrLiteralExpr(E->getType(),
      E->getSourceRange().getBegin());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}


// unsupported CXX Statements and Expressions
Expr *ASTTranslate::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXTypeidExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXThisExpr(CXXThisExpr *E) {
  Expr *result = new (Ctx) CXXThisExpr(E->getSourceRange().getBegin(),
      E->getType(), E->isImplicit());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXThrowExpr(CXXThrowExpr *E) {
  Expr *result = new (Ctx) CXXThrowExpr(Clone(E->getSubExpr()), E->getType(),
      E->getThrowLoc(), E->isThrownVariableInScope());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

Expr *ASTTranslate::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXDefaultArgExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXBindTemporaryExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXScalarValueInitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXNewExpr(CXXNewExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNewExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXDeleteExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXPseudoDestructorExpr);
  return NULL;
}

Expr *ASTTranslate::VisitExprWithCleanups(ExprWithCleanups *E) {
  HIPACC_NOT_SUPPORTED(ExprWithCleanups);
  return NULL;
}

Expr *ASTTranslate::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr
    *E) {
  HIPACC_NOT_SUPPORTED(CXXDependentScopeMemberExpr);
  return NULL;
}

Expr *ASTTranslate::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E)
{
  HIPACC_NOT_SUPPORTED(DependentScopeDeclRefExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr
    *E) {
  HIPACC_NOT_SUPPORTED(CXXUnresolvedConstructExpr);
  return NULL;
}

Expr *ASTTranslate::VisitOverloadExpr(OverloadExpr *E) {
  HIPACC_NOT_SUPPORTED(OverloadExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  HIPACC_NOT_SUPPORTED(UnresolvedMemberExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  HIPACC_NOT_SUPPORTED(UnresolvedLookupExpr);
  return NULL;
}

Expr *ASTTranslate::VisitTypeTraitExpr(TypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(TypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(UnaryTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(BinaryTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(ArrayTypeTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  HIPACC_NOT_SUPPORTED(ExpressionTraitExpr);
  return NULL;
}

Expr *ASTTranslate::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  HIPACC_NOT_SUPPORTED(CXXNoexceptExpr);
  return NULL;
}

Expr *ASTTranslate::VisitPackExpansionExpr(PackExpansionExpr *E) {
  HIPACC_NOT_SUPPORTED(PackExpansionExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  HIPACC_NOT_SUPPORTED(SizeOfPackExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *E) {
  HIPACC_NOT_SUPPORTED(SubstNonTypeTemplateParmExpr);
  return NULL;
}

Expr *ASTTranslate::VisitSubstNonTypeTemplateParmPackExpr(
    SubstNonTypeTemplateParmPackExpr *E) {
  HIPACC_NOT_SUPPORTED(SubstNonTypeTemplateParmPackExpr);
  return NULL;
}

Expr *ASTTranslate::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  HIPACC_NOT_SUPPORTED(MaterializeTemporaryExpr);
  return NULL;
}

Expr *ASTTranslate::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  HIPACC_NOT_SUPPORTED(OpaqueValueExpr);
  return NULL;
}

Expr *ASTTranslate::VisitAtomicExpr(AtomicExpr *E) {
  llvm::SmallVector<Expr *, 16> Args;

  for (unsigned int i=0, e=E->getNumSubExprs(); i<e; ++i) {
    Args.push_back(Clone(E->getSubExprs()[i]));
  }

  Expr *result = new (Ctx) AtomicExpr(E->getBuiltinLoc(), Args.data(),
      Args.size(), E->getType(), E->getOp(), E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

// CUDA Expressions
Expr *ASTTranslate::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  llvm::SmallVector<Expr *, 16> Args;

  for (unsigned int i=0, e=E->getNumArgs(); i<e; ++i) {
    Args.push_back(Clone(E->getArg(i)));
  }

  Expr *result = new (Ctx) CUDAKernelCallExpr(Ctx, Clone(E->getCallee()),
      Clone(E->getConfig()), Args.data(), Args.size(), E->getType(),
      E->getValueKind(), E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

// OpenCL Expressions
Expr *ASTTranslate::VisitAsTypeExpr(AsTypeExpr *E) {
  Expr *result = new (Ctx) AsTypeExpr(Clone(E->getSrcExpr()), E->getType(),
      E->getValueKind(), E->getObjectKind(), E->getBuiltinLoc(),
      E->getRParenLoc());

  result->setValueDependent(E->isValueDependent());
  result->setTypeDependent(E->isTypeDependent());

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

