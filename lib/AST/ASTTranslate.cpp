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

//===--- ASTTranslate.cpp - Translation of the AST ------------------------===//
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



// add cast to unsigned variables if needed
Expr *ASTTranslate::addCastToInt(Expr *E) {
  return createCStyleCastExpr(Ctx, Ctx.IntTy, CK_IntegralCast, E, nullptr,
      Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
}


// clone function for execution on device
FunctionDecl *ASTTranslate::cloneFunction(FunctionDecl *FD) {
  FunctionDecl *result = KernelFunctionMap[FD];

  // clone function
  if (!result) {
    cloneFuns.push_back(FD);
    TranslationMode oldMode = astMode;
    astMode = CloneAST;

    llvm::errs() << "Cloning function '"
                 << FD->getNameAsString() << "' "
                 << "for execution on device.\n";

    // check return type
    QualType retType = FD->getReturnType();
    if (!retType->isStandardLayoutType() && !retType->isVoidType()) {
      unsigned DiagIDRetType = Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Cannot convert function '%0' for execution on device. "
            "Return type is no not supported: ");
      Diags.Report(FD->getLocStart(), DiagIDRetType) << FD->getNameAsString();
      exit(EXIT_FAILURE);
    }

    // check parameters
    SmallVector<QualType, 16> argTypes;
    SmallVector<std::string, 16> argNames;

    for (auto param : FD->params()) {
      QualType QT = param->getType();

      // allow reference types for CUDA only
      if (compilerOptions.emitCUDA()) {
        QT = QT.getNonReferenceType();
      }

      if (!QT->isStandardLayoutType()) {
        unsigned DiagIDParmType =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Cannot convert function '%0' for execution on device. "
              "Argument type is no not supported: ");
        Diags.Report(param->getLocation(), DiagIDParmType) << FD->getNameAsString();
        exit(EXIT_FAILURE);
      }

      argTypes.push_back(param->getType());
      argNames.push_back(param->getNameAsString());
    }

    // create signature
    result = createFunctionDecl(Ctx, Ctx.getTranslationUnitDecl(),
        FD->getNameAsString() + Kernel->getName(), retType, argTypes, argNames);

    // first store function declaration
    KernelFunctionMap[FD] = result;

    // then clone body (recursive functions)
    result->setBody(Clone(FD->getBody()));
    Kernel->addFunctionCall(result);

    astMode = oldMode;
    cloneFuns.pop_back();
  } else {
    if (std::find(cloneFuns.begin(), cloneFuns.end(), FD)!=cloneFuns.end()) {
      unsigned DiagIDRecursion =
        Diags.getCustomDiagID(DiagnosticsEngine::Warning,
            "recursive call to function '%0' detected: this is only supported on some devices and may cause segmentation faults!");
      Diags.Report(FD->getLocation(), DiagIDRecursion) << FD->getNameAsString();
    }
  }

  return result;
}


template <typename T>
T *ASTTranslate::lookup(std::string name, QualType QT, NamespaceDecl *NS) {
  DeclContext *DC = Ctx.getTranslationUnitDecl();
  if (NS) DC = Decl::castToDeclContext(NS);

  for (auto *decl : DC->lookup(&Ctx.Idents.get(name))) {
    if (auto result = cast_or_null<T>(decl)) {
      if (auto fun = dyn_cast<FunctionDecl>(result)) {
        if (fun->getReturnType().getDesugaredType(Ctx) ==
            QT.getDesugaredType(Ctx)) return result;
        continue;
      }
      if (auto var = dyn_cast<VarDecl>(result)) {
        if (var->getType().getDesugaredType(Ctx) == QT.getDesugaredType(Ctx))
          return result;
        continue;
      }

      // default case
      return result;
    }
  }

  return nullptr;
}


// C/C++ initialization
void ASTTranslate::initCPU(SmallVector<Stmt *, 16> &kernelBody, Stmt *S) {
  VarDecl *gid_x = nullptr, *gid_y = nullptr;

  // C/C++: int gid_x = offset_x;
  if (Kernel->getIterationSpace()->getOffsetXDecl()) {
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
        getOffsetXDecl(Kernel->getIterationSpace()));
  } else {
    gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.IntTy,
        createIntegerLiteral(Ctx, 0));
  }

  // C/C++: int gid_y = offset_y;
  if (Kernel->getIterationSpace()->getOffsetYDecl()) {
    gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.IntTy,
        getOffsetYDecl(Kernel->getIterationSpace()));
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
  gidYRef = tileVars.global_id_y;

  // set also other variables not used by C back end
  tileVars.local_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.local_id_y = createDeclRefExpr(Ctx, gid_y);
  tileVars.block_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.block_id_y = createDeclRefExpr(Ctx, gid_y);
  tileVars.local_size_x = getStrideDecl(Kernel->getIterationSpace());
  tileVars.local_size_y = createIntegerLiteral(Ctx, 0);

  // check if we need border handling
  if (KernelClass->getKernelType() != UserOperator) {
    for (auto img : KernelClass->getImgFields()) {
      HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

      // check if we need border handling
      if (Acc->getBoundaryMode() != Boundary::UNDEFINED) {
        if (Acc->getSizeX() > 1) {
            bh_variant.borders.left = 1;
            bh_variant.borders.right = 1;
        }
        if (Acc->getSizeY() > 1) {
            bh_variant.borders.top = 1;
            bh_variant.borders.bottom = 1;
        }
      }
    }
  }

  // convert the function body to kernel syntax
  Stmt *clonedStmt = Clone(S);
  assert(isa<CompoundStmt>(clonedStmt) && "CompoundStmt for kernel function body expected!");

  //
  // for (int gid_y=offset_y; gid_y<is_height+offset_y; gid_y++) {
  //     for (int gid_x=offset_x; gid_x<is_width+offset_x; gid_x++) {
  //         body
  //     }
  // }
  //
  Expr *upper_x = getWidthDecl(Kernel->getIterationSpace());
  Expr *upper_y = getHeightDecl(Kernel->getIterationSpace());
  if (Kernel->getIterationSpace()->getOffsetXDecl()) {
    upper_x = createBinaryOperator(Ctx, upper_x,
        getOffsetXDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy);
  }
  if (Kernel->getIterationSpace()->getOffsetYDecl()) {
    upper_y = createBinaryOperator(Ctx, upper_y,
        getOffsetYDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy);
  }
  ForStmt *innerLoop = createForStmt(Ctx, gid_x_stmt, createBinaryOperator(Ctx,
        tileVars.global_id_x, upper_x, BO_LT, Ctx.BoolTy),
      createUnaryOperator(Ctx, tileVars.global_id_x, UO_PostInc,
        tileVars.global_id_x->getType()), clonedStmt);
  ForStmt *outerLoop = createForStmt(Ctx, gid_y_stmt, createBinaryOperator(Ctx,
        tileVars.global_id_y, upper_y, BO_LT, Ctx.BoolTy),
      createUnaryOperator(Ctx, tileVars.global_id_y, UO_PostInc,
        tileVars.global_id_y->getType()), innerLoop);

  kernelBody.push_back(outerLoop);
}


// CUDA initialization
void ASTTranslate::initCUDA(SmallVector<Stmt *, 16> &kernelBody) {
  VarDecl *gid_x = nullptr, *gid_y = nullptr;
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
  //  unsigned x, y, z;
  //};
  RecordDecl *uint3RD = createRecordDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "uint3", TTK_Struct, uintDeclTypes, uintDeclNames);

  /*DEVICE_BUILTIN*/
  //typedef struct uint3 uint3;

  /*DEVICE_BUILTIN*/
  //struct dim3
  //{
  //    unsigned x, y, z;
  //};

  /*DEVICE_BUILTIN*/
  //typedef struct dim3 dim3;

  //uint3 threadIdx;
  VarDecl *threadIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "threadIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  //uint3 blockIdx;
  VarDecl *blockIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  //dim3 blockDim;
  VarDecl *blockDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockDim", Ctx.getTypeDeclType(uint3RD), nullptr);
  //dim3 gridDim;
  //VarDecl *gridDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "gridDim", Ctx.getTypeDeclType(uint3RD), nullptr);
  //int warpSize;
  //VarDecl *warpSize = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "warpSize", Ctx.IntTy, nullptr);

  DeclRefExpr *TIRef = createDeclRefExpr(Ctx, threadIdx);
  DeclRefExpr *BIRef = createDeclRefExpr(Ctx, blockIdx);
  DeclRefExpr *BDRef = createDeclRefExpr(Ctx, blockDim);
  VarDecl *xVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "x",
      Ctx.IntTy, nullptr);
  VarDecl *yVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "y",
      Ctx.IntTy, nullptr);

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
  VarDecl *gid_x = nullptr, *gid_y = nullptr;
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
  //    CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg0),
  //    nullptr, VK_RValue);
  //get_global_size1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_global_size, tmpArg1),
  //    nullptr, VK_RValue);
  ImplicitCastExpr *get_global_id0, *get_global_id1;
  get_global_id0 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg0), nullptr,
      VK_RValue);
  get_global_id1 = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_global_id, tmpArg1), nullptr,
      VK_RValue);
  tileVars.local_size_x = createImplicitCastExpr(Ctx,
      Ctx.getConstType(Ctx.IntTy), CK_IntegralCast, createFunctionCall(Ctx,
        get_local_size, tmpArg0), nullptr, VK_RValue);
  tileVars.local_size_y = createImplicitCastExpr(Ctx,
      Ctx.getConstType(Ctx.IntTy), CK_IntegralCast, createFunctionCall(Ctx,
        get_local_size, tmpArg1), nullptr, VK_RValue);
  tileVars.local_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg0), nullptr,
      VK_RValue);
  tileVars.local_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_local_id, tmpArg1), nullptr,
      VK_RValue);
  tileVars.block_id_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg0), nullptr,
      VK_RValue);
  tileVars.block_id_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
      CK_IntegralCast, createFunctionCall(Ctx, get_group_id, tmpArg1), nullptr,
      VK_RValue);
  //grid_size_x = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg0),
  //    nullptr, VK_RValue);
  //grid_size_y = createImplicitCastExpr(Ctx, Ctx.getConstType(Ctx.IntTy),
  //    CK_IntegralCast, createFunctionCall(Ctx, get_num_groups, tmpArg1),
  //    nullptr, VK_RValue);

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
  VarDecl *gid_x = nullptr, *gid_y = nullptr;
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
  tileVars.local_size_x = getStrideDecl(Kernel->getIterationSpace());
  tileVars.local_size_y = createIntegerLiteral(Ctx, 1);

  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::Renderscript: {
        // retValRef: Kernel parameter pointing to current output pixel
        VarDecl *output = createVarDecl(Ctx, kernelDecl, "_iter",
            Kernel->getIterationSpace()->getImage()->getType());
        retValRef = createDeclRefExpr(Ctx, output);
      }
      break;
    case Language::Filterscript: {
        // retValRef: Variable storing output value to return from kernel
        VarDecl *output = createVarDecl(Ctx, kernelDecl, "OutputVal",
            Kernel->getIterationSpace()->getImage()->getType());
        DC->addDecl(output);
        kernelBody.push_back(createDeclStmt(Ctx, output));
        retValRef = createDeclRefExpr(Ctx, output);
      }
      break;
  }
}


// update tileVars to constants if required
void ASTTranslate::updateTileVars() {
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      tileVars.local_id_x = addCastToInt(tileVars.local_id_x);
      tileVars.local_id_y = addCastToInt(tileVars.local_id_y);
      tileVars.block_id_x = addCastToInt(tileVars.block_id_x);
      tileVars.block_id_y = addCastToInt(tileVars.block_id_y);
      // select fastest method for accessing blockDim.[x|y]
      // TODO: define this in HipaccDeviceOptions
      if (compilerOptions.getTargetDevice()>=Device::Fermi_20 &&
          compilerOptions.getTargetDevice()<=Device::Kepler_30 &&
          compilerOptions.getTargetLang()==Language::CUDA) {
        if (compilerOptions.exploreConfig() && !emitEstimation) {
          tileVars.local_size_x = createDeclRefExpr(Ctx, createVarDecl(Ctx,
                kernelDecl, "BSX_EXPLORE", Ctx.IntTy, nullptr));
          tileVars.local_size_y = createDeclRefExpr(Ctx, createVarDecl(Ctx,
                kernelDecl, "BSY_EXPLORE", Ctx.IntTy, nullptr));
        } else {
          // use constant for final kernel configuration
          tileVars.local_size_x = createIntegerLiteral(Ctx,
              (int)Kernel->getNumThreadsX());
          tileVars.local_size_y = createIntegerLiteral(Ctx,
              (int)Kernel->getNumThreadsY());
        }
      } else {
        // cast blockDim.[x|y] to signed integer
        tileVars.local_size_x = addCastToInt(tileVars.local_size_x);
        tileVars.local_size_y = addCastToInt(tileVars.local_size_y);
      }
      break;
  }
}


Stmt *ASTTranslate::Hipacc(Stmt *S) {
  if (S==nullptr) return nullptr;

  // search for image width and height parameters
  for (auto param : kernelDecl->params()) {
    auto parm_ref = createDeclRefExpr(Ctx, param);
    // the first parameter is the output image; create association between them.
    if (param == *kernelDecl->param_begin()) {
      outputImage = parm_ref;
      continue;
    }

    // search for boundary handling parameters
    if (param->getName().equals("bh_start_left")) {
      bh_start_left = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_right")) {
      bh_start_right = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_top")) {
      bh_start_top = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_bottom")) {
      bh_start_bottom = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_fall_back")) {
      bh_fall_back = parm_ref;
      continue;
    }

    if (compilerOptions.emitRenderscript() ||
        compilerOptions.emitFilterscript()) {
      // search for uint32_t x, uint32_t y parameters
      if (param->getName().equals("x")) {
        // TODO: scan for uint32_t x
        continue;
      }
      if (param->getName().equals("y")) {
        // TODO: scan for uint32_t y
        continue;
      }
    }


    // search for image width, height and stride parameters
    for (auto img : KernelClass->getImgFields()) {
      HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

      if (param->getName().equals(img->getNameAsString() + "_width")) {
        Acc->setWidthDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_height")) {
        Acc->setHeightDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_stride")) {
        Acc->setStrideDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_offset_x")) {
        Acc->setOffsetXDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_offset_y")) {
        Acc->setOffsetYDecl(parm_ref);
        continue;
      }
    }
  }

  // in case no stride was found, use image width as fallback
  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

    if (Acc->getStrideDecl() == nullptr) {
      Acc->setStrideDecl(Acc->getWidthDecl());
    }
  }

  // initialize target-specific variables and add gid_x and gid_y declarations
  // to kernel body
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  SmallVector<Stmt *, 16> kernelBody;
  FunctionDecl *barrier;
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
      initCPU(kernelBody, S);
      return createCompoundStmt(Ctx, kernelBody);
      break;
    case Language::CUDA:
      initCUDA(kernelBody);
      // void __syncthreads();
      barrier = builtins.getBuiltinFunction(CUDABI__syncthreads);
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      initOpenCL(kernelBody);
      // void barrier(cl_mem_fence_flags);
      barrier = builtins.getBuiltinFunction(OPENCLBIbarrier);
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      initRenderscript(kernelBody);
      break;
  }
  lidYRef = tileVars.local_id_y;
  gidYRef = tileVars.global_id_y;

  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

    // add scale factor calculations for interpolation:
    // float acc_scale_x = (float)acc_width/is_width;
    // float acc_scale_y = (float)acc_height/is_height;
    if (Acc->getInterpolationMode() != Interpolate::NO) {
      Expr *scaleExprX = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getWidthDecl(Acc), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getWidthDecl(Kernel->getIterationSpace()), BO_Div, Ctx.FloatTy);
      Expr *scaleExprY = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getHeightDecl(Acc), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getHeightDecl(Kernel->getIterationSpace()), BO_Div, Ctx.FloatTy);
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

  // clear all stored decls before cloning, otherwise existing VarDecls will
  // be reused and we will miss declarations
  KernelDeclMapTex.clear();
  KernelDeclMapShared.clear();
  KernelDeclMapVector.clear();
  KernelDeclMapAcc.clear();
  KernelFunctionMap.clear();

  // add vector pointer declarations for images
  if (Kernel->vectorize() && !compilerOptions.emitC99()) {
    // search for member name in kernel parameter list
    for (auto param : kernelDecl->params()) {
      // output image - iteration space
      if (param->getName().equals((*kernelDecl->param_begin())->getName())) {
        // <type>4 *Output4 = (<type>4 *) Output;
        VarDecl *VD = CloneParmVarDecl(param);

        VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
              createDeclRefExpr(Ctx, param), nullptr,
              Ctx.getTrivialTypeSourceInfo(VD->getType())));

        kernelBody.push_back(createDeclStmt(Ctx, VD));

        // update output Image reference
        outputImage = createDeclRefExpr(Ctx, VD);
      }
    }

    for (auto img : KernelClass->getImgFields()) {
      StringRef name = img->getName();

      // search for member name in kernel parameter list
      for (auto param : kernelDecl->params()) {
        // parameter name matches
        if (param->getName().equals(name)) {
          // <type>4 *Input4 = (<type>4 *) Input;
          VarDecl *VD = CloneParmVarDecl(param);

          VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
                createDeclRefExpr(Ctx, param), nullptr,
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
  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);
    MemoryAccess mem_acc = KernelClass->getMemAccess(img);

    // bail out for user defined kernels
    if (KernelClass->getKernelType()==UserOperator) break;

    // check if we need border handling
    if (Acc->getBoundaryMode() != Boundary::UNDEFINED) {
      if (Acc->getSizeX() > 1 || Acc->getSizeY() > 1) border_handling = true;
      if (Acc->getSizeX() > 1) kernel_x = true;
      if (Acc->getSizeY() > 1) kernel_y = true;
    }

    // check if we need shared memory
    if (mem_acc == READ_ONLY && Kernel->useLocalMemory(Acc)) {
      std::string sharedName = "_smem";
      sharedName += img->getNameAsString();
      use_shared = true;

      VarDecl *VD;
      QualType QT;
      // __shared__ T _smemIn[SY-1 + BSY*PPT][3 * BSX];
      // for left and right halo, add 2*BSX
      if (!emitEstimation && compilerOptions.exploreConfig()) {
        Expr *SX = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSX_EXPLORE", Ctx.IntTy, nullptr));
        Expr *BSY = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSY_EXPLORE", Ctx.IntTy, nullptr));
        Expr *SY = BSY;

        if (Acc->getSizeX() > 1) {
          // 3*BSX
          SX = createBinaryOperator(Ctx, createIntegerLiteral(Ctx, 3), SX,
              BO_Mul, Ctx.IntTy);
        }
        // add padding to avoid bank conflicts
        SX = createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1), BO_Add,
            Ctx.IntTy);

        // size_y = ceil((PPT*BSY+SY-1)/BSY)
        // -> PPT*BSY + ((SY-2)/BSY + 1) * BSY
        if (Kernel->getPixelsPerThread() > 1) {
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                (int)Kernel->getPixelsPerThread()), BO_Mul, Ctx.IntTy);
        }

        if (Acc->getSizeY() > 1) {
          SY = createBinaryOperator(Ctx, SY, createBinaryOperator(Ctx,
                createParenExpr(Ctx, createBinaryOperator(Ctx,
                    createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                        (int)Acc->getSizeY()-2), BSY, BO_Div, Ctx.IntTy),
                    createIntegerLiteral(Ctx, 1), BO_Add, Ctx.IntTy)), BSY,
                BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        }

        QT = Acc->getImage()->getType();
        QT = Ctx.getVariableArrayType(QT, SX, ArrayType::Normal, 0,
                SourceLocation());
        QT = Ctx.getVariableArrayType(QT, SY, ArrayType::Normal, 0,
                SourceLocation());
      } else {
        llvm::APInt SX, SY;
        SX = llvm::APInt(32, Kernel->getNumThreadsX());
        if (Acc->getSizeX() > 1) {
          // 3*BSX
          SX *= llvm::APInt(32, 3);
        }
        // add padding to avoid bank conflicts
        SX += llvm::APInt(32, 1);

        // size_y = ceil((PPT*BSY+SY-1)/BSY)
        int smem_size_y =
          (int)ceilf((float)(Kernel->getPixelsPerThread()*Kernel->getNumThreadsY()
                + Acc->getSizeY()-1)/(float)Kernel->getNumThreadsY());
        SY = llvm::APInt(32, smem_size_y*Kernel->getNumThreadsY());

        QT = Acc->getImage()->getType();
        QT = Ctx.getConstantArrayType(QT, SX, ArrayType::Normal, 0);
        QT = Ctx.getConstantArrayType(QT, SY, ArrayType::Normal, 0);
      }

      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          VD = createVarDecl(Ctx, DC, sharedName, QT, nullptr);
          VD->addAttr(CUDASharedAttr::CreateImplicit(Ctx));
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          VD = createVarDecl(Ctx, DC, sharedName, Ctx.getAddrSpaceQualType(QT,
                LangAS::opencl_local), nullptr);
          break;
      }

      // search for member name in kernel parameter list
      for (auto param : kernelDecl->params()) {
        // parameter name matches
        if (param->getName().equals(img->getName())) {
          // store mapping between ParmVarDecl and shared memory VarDecl
          KernelDeclMapShared[param] = VD;
          KernelDeclMapAcc[param] = Acc;

          break;
        }
      }

      // add VarDecl to current kernel DeclContext
      DC->addDecl(VD);
      kernelBody.push_back(createDeclStmt(Ctx, VD));
    }
  }

  // activate boundary handling for exploration
  if (compilerOptions.exploreConfig() && use_shared) {
    border_handling = true;
    kernel_x = true;
    kernel_y = true;
  }


  SmallVector<LabelDecl *, 16> LDS;
  LabelDecl *LDExit = createLabelDecl(Ctx, kernelDecl, "BH_EXIT");
  LabelStmt *LSExit = createLabelStmt(Ctx, LDExit, nullptr);
  GotoStmt *GSExit = createGotoStmt(Ctx, LDExit);


  // only create labels if we need border handling
  for (size_t i=0; i<=9 && border_handling; ++i) {
    LabelDecl *LD;
    Expr *if_goto = nullptr;

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

        if (compilerOptions.getTargetDevice()<=Device::Tesla_13 &&
            compilerOptions.getTargetLang()==Language::CUDA) {
          // TODO: remove this once CUDA 7 is widespread and Tesla architecture
          // is discontinued
          // CUDA: if (blockDim.x >= 16) goto BH_NO;
          if_goto = createBinaryOperator(Ctx, tileVars.local_size_x,
              createIntegerLiteral(Ctx, 16), BO_GE, Ctx.BoolTy);
        }
        break;
    }
    LDS.push_back(LD);
    Stmt *GS = createGotoStmt(Ctx, LD);
    if (if_goto)
      GS = createIfStmt(Ctx, if_goto, GS);
    kernelBody.push_back(GS);
  }

  // add casts to tileVars if required
  updateTileVars();

  int ld_count = 0;
  for (size_t i=border_handling?0:9; i<=9; ++i) {
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
    BinaryOperator *check_bop = nullptr;
    if (border_handling) {
      // if (gid_x >= is_offset_x)
      if (Kernel->getIterationSpace()->getOffsetXDecl() &&
          !(kernel_x && !bh_variant.borders.left) && bh_variant.borderVal) {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getOffsetXDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
      }
      // if (gid_x < is_width+is_offset_x)
      if (!(kernel_x && !bh_variant.borders.right) && bh_variant.borderVal) {
        BinaryOperator *check_tmp = nullptr;
        if (Kernel->getIterationSpace()->getOffsetXDecl()) {
          check_tmp = createBinaryOperator(Ctx, tileVars.global_id_x,
              createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()),
                getOffsetXDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy),
              BO_LT, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, tileVars.global_id_x,
              getWidthDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
        }
        if (check_bop) {
          check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
              Ctx.BoolTy);
        } else {
          check_bop = check_tmp;
        }
      }
      // Renderscript iteration space is always the whole image, so we need to
      // check the y-dimension as well:
      // if (gid_y >= is_offset_y && gid_y < is_height+is_offset_y)
      if (compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        // if (gid_y >= is_offset_y)
        if (Kernel->getIterationSpace()->getOffsetYDecl() &&
            !(kernel_y && !bh_variant.borders.left) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = nullptr;
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
          if (check_bop) {
            check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
                Ctx.BoolTy);
          } else {
            check_bop = check_tmp;
          }
        }
        // if (gid_y < is_height+is_offset_y)
        if (!(kernel_y && !bh_variant.borders.right) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = nullptr;
          if (Kernel->getIterationSpace()->getOffsetYDecl()) {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()),
                  getOffsetYDecl(Kernel->getIterationSpace()), BO_Add,
                  Ctx.IntTy), BO_LT, Ctx.BoolTy);
          } else {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
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
      if (Kernel->getIterationSpace()->getOffsetXDecl()) {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getOffsetXDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
        check_bop = createBinaryOperator(Ctx, check_bop,
            createBinaryOperator(Ctx, tileVars.global_id_x,
              createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()),
                getOffsetXDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy),
              BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
      } else {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getWidthDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
      }
      // if (gid_y >= is_offset_y && gid_y < is_height+is_offset_y)
      // Renderscript iteration space is always the whole image, so we need to
      // check the y-dimension as well.
      if (compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        // if (gid_y < is_height+is_offset_y)
        BinaryOperator *check_tmp = nullptr;
        if (Kernel->getIterationSpace()->getOffsetYDecl()) {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
          check_tmp = createBinaryOperator(Ctx, check_tmp,
              createBinaryOperator(Ctx, gidYRef, createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()),
                  getOffsetYDecl(Kernel->getIterationSpace()), BO_Add,
                  Ctx.IntTy), BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
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
    for (size_t p=0; use_shared && p<Kernel->getPixelsPerThread()+p_add; ++p) {
      if (compilerOptions.exploreConfig()) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
        // all iterations
        stageIterationToSharedMemoryExploration(labelBody);

        break;
      }
      if (p==0) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
        // first iteration
        stageIterationToSharedMemory(labelBody, p);
      } else {
        // update lid_y to lid_y + p*(int)local_size_y
        // update gid_y to gid_y + p*(int)local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, (int32_t)p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, (int32_t)p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        // load next iteration to shared memory
        stageIterationToSharedMemory(labelBody, p);
      }
    }
    // synchronize shared memory
    if (use_shared) {
      // add memory barrier synchronization
      SmallVector<Expr *, 16> args;
      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          // CLK_LOCAL_MEM_FENCE -> 1
          // CLK_GLOBAL_MEM_FENCE -> 2
          args.push_back(createIntegerLiteral(Ctx, 1));
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
      }
    }

    for (size_t p=0; p<Kernel->getPixelsPerThread(); ++p) {
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
        // update lid_y to lid_y + p*(int)local_size_y
        // update gid_y to gid_y + p*(int)local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, (int32_t)p),
              tileVars.local_size_y, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx, (int32_t)p),
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
        if (!bh_variant.borderVal && !compilerOptions.exploreConfig())
          require_is_check = false;
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
            getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
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
        for (auto stmt : pptBody)
          labelBody.push_back(stmt);
      }
    }

    // add label statement if needed (boundary handling), else add body
    if (border_handling) {
      LabelStmt *LS = createLabelStmt(Ctx, LDS[ld_count++],
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
    // in case no value was written, return the value of the iteration space
    Expr *result = accessMem(outputImage, Kernel->getIterationSpace(),
        READ_ONLY);
    setExprProps(outputImage, result);
    kernelBody.push_back(createReturnStmt(Ctx, result));
  }

  CompoundStmt *CS = createCompoundStmt(Ctx, kernelBody);

  return CS;
}


VarDecl *ASTTranslate::CloneVarDecl(VarDecl *VD) {
  VarDecl *result = KernelDeclMap[VD];

  if (!result && (convMask || !redDomains.empty()))
    result = LambdaDeclMap[VD];

  if (!result) {
    QualType QT = VD->getType();
    TypeSourceInfo *TInfo = VD->getTypeSourceInfo();
    std::string name = VD->getName();

    if (Kernel->vectorize() && KernelClass->getVectorizeInfo(VD) == VECTORIZE &&
        !compilerOptions.emitC99()) {
      QT = simdTypes.getSIMDType(VD, SIMD4);
      TInfo = Ctx.getTrivialTypeSourceInfo(QT);
    }

    DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
    result = VarDecl::Create(Ctx, DC, VD->getInnerLocStart(), VD->getLocation(),
        &Ctx.Idents.get(name), QT, TInfo, VD->getStorageClass());
    result->setIsUsed(); // set VarDecl as being used - required for CodeGen
    if (Kernel->vectorize() && KernelClass->getVectorizeInfo(VD) == VECTORIZE &&
        !compilerOptions.emitC99() ) {
      result->setInit(simdTypes.propagate(VD, Clone(VD->getInit())));
    } else {
      result->setInit(Clone(VD->getInit()));
    }
    result->setInitStyle(VD->getInitStyle());
    result->setTSCSpec(VD->getTSCSpec());

    // store mapping between original VarDecl and cloned VarDecl
    if (convMask || !redDomains.empty()) {
      LambdaDeclMap[VD] = result;
      LambdaDeclMap[result] = result;
    } else {
      KernelDeclMap[VD] = result;
      KernelDeclMap[result] = result;
    }

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


VarDecl *ASTTranslate::CloneParmVarDecl(ParmVarDecl *PVD) {
  VarDecl *result = KernelDeclMapVector[PVD];

  if (!result) {
    std::string name = PVD->getName();
    QualType QT = PVD->getType();
    TypeSourceInfo *TInfo = PVD->getTypeSourceInfo();

    // only vectorize image PVDs
    if (Kernel->vectorize() && !compilerOptions.emitC99()) {
      for (auto img : KernelClass->getImgFields()) {
        // parameter name matches
        if (PVD->getName().equals(img->getName()) ||
            PVD->getName().equals((*kernelDecl->param_begin())->getName())) {
          // mark original variable as being used
          Kernel->setUsed(name);

          // add suffix to vectorized variable
          name += "4";

          // get SIMD4 type
          QT = simdTypes.getSIMDType(PVD, SIMD4);
          TInfo = Ctx.getTrivialTypeSourceInfo(QT);
          break;
        }
      }
    }

    DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
    result = VarDecl::Create(Ctx, DC, PVD->getInnerLocStart(),
        PVD->getLocation(), &Ctx.Idents.get(name), QT, TInfo,
        PVD->getStorageClass());
    result->setIsUsed(); // set VarDecl as being used - required for CodeGen
    result->setInit(Clone(PVD->getInit()));
    result->setInitStyle(PVD->getInitStyle());
    result->setTSCSpec(PVD->getTSCSpec());

    // store mapping between original VarDecl and cloned VarDecl
    KernelDeclMapVector[PVD] = result;

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


VarDecl *ASTTranslate::CloneDeclTex(ParmVarDecl *PVD, std::string prefix) {
  if (PVD==nullptr) return nullptr;

  VarDecl *result = KernelDeclMapTex[PVD];

  if (!result) {
    std::string texName = prefix;
    texName += PVD->getName();
    texName += Kernel->getName();

    DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
    result = VarDecl::Create(Ctx, DC, PVD->getInnerLocStart(),
        PVD->getLocation(), &Ctx.Idents.get(texName), PVD->getType(),
        PVD->getTypeSourceInfo(), PVD->getStorageClass());
    result->setIsUsed(); // set VarDecl as being used - required for CodeGen
    result->setInit(Clone(PVD->getInit()));
    result->setInitStyle(PVD->getInitStyle());
    result->setTSCSpec(PVD->getTSCSpec());

    // store mapping between original VarDecl and cloned VarDecl
    KernelDeclMapTex[PVD] = result;

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}


Stmt *ASTTranslate::VisitCompoundStmtTranslate(CompoundStmt *S) {
  CompoundStmt *result = new (Ctx) CompoundStmt(Ctx, MultiStmtArg(),
      S->getLBracLoc(), S->getLBracLoc());

  SmallVector<Stmt *, 16> body;
  for (auto stmt : S->body()) {
    curCStmt = S;
    Stmt *newS = Clone(stmt);
    curCStmt = S;

    if (preStmts.size()) {
      size_t num_stmts = 0;
      for (size_t i=0, e=preStmts.size(); i!=e; ++i) {
        if (preCStmt[i]==S) {
          body.push_back(preStmts[i]);
          num_stmts++;
        }
      }
      for (size_t i=0; i<num_stmts; ++i) {
        preStmts.pop_back();
        preCStmt.pop_back();
      }
    }

    if (newS) body.push_back(newS);

    if (postStmts.size()) {
      size_t num_stmts = 0;
      for (size_t i=0, e=postStmts.size(); i!=e; ++i) {
        if (postCStmt[i]==S) {
          body.push_back(postStmts[i]);
          num_stmts++;
        }
      }
      for (size_t i=0; i<num_stmts; ++i) {
        postStmts.pop_back();
        postCStmt.pop_back();
      }
    }
  }

  result->setStmts(Ctx, body.data(), body.size());

  return result;
}


Stmt *ASTTranslate::VisitReturnStmtTranslate(ReturnStmt *S) {
  // replace return statements within convolve lambda-functions
  if (convMask && convTmp) {
    return getConvolutionStmt(convMode, convTmp, Clone(S->getRetValue()));
  } else if (!redDomains.empty() && !redTmps.empty()) {
    return getConvolutionStmt(redModes.back(), redTmps.back(),
                              Clone(S->getRetValue()));
  } else {
    return new (Ctx) ReturnStmt(S->getReturnLoc(), Clone(S->getRetValue()), 0);
  }
}


Expr *ASTTranslate::VisitCallExprTranslate(CallExpr *E) {
  if (E->getDirectCallee()) {
    // lookup if this function call is supported and choose appropriate
    // function, e.g. exp() instead of expf() in case of OpenCL
    FunctionDecl *targetFD = nullptr;
    FunctionDecl *convert = nullptr;
    if (compilerOptions.emitC99()) {
      targetFD = E->getDirectCallee();
    } else {
      DeclContext *DC = E->getDirectCallee()->getEnclosingNamespaceContext();
      if (DC->isNamespace()) {
        NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC);
        if (NS->getNameAsString() == "math") {
          DC = DC->getParent();
          if (DC) NS = dyn_cast<NamespaceDecl>(DC);
        }

        if (NS->getNameAsString() == "hipacc") {
          // namespace hipacc::math
          targetFD = E->getDirectCallee();

          if (!compilerOptions.emitCUDA()) {
            bool doUpdate = false;
            std::string name = E->getDirectCallee()->getNameAsString();
            if (name.at(name.length()-1)=='f') {
              if (name!="modf" && name!="erf") {
                // remove trailing f
                name.resize(name.size() - 1);
                doUpdate = true;
              }
            } else if (name=="labs") {
              // remove leading l
              name.erase(0, 1);
              doUpdate = true;
              // require convert function ulong -> long
              convert = lookup<FunctionDecl>(std::string("convert_long4"),
                  simdTypes.getSIMDType(Ctx.LongTy, std::string("long"), SIMD4),
                  hipaccNS);
              assert(convert && "could not lookup 'convert_long4'");
            } else if (name=="abs") {
              // require convert function uint -> int
              convert = lookup<FunctionDecl>(std::string("convert_int4"),
                  simdTypes.getSIMDType(Ctx.IntTy, std::string("int"), SIMD4),
                  hipaccNS);
              assert(convert && "could not lookup 'convert_int4'");
            }

            if (doUpdate) {
              SmallVector<QualType, 16> argTypes;
              SmallVector<std::string, 16> argNames;

              for (auto param : targetFD->params()) {
                argTypes.push_back(param->getType());
                argNames.push_back(param->getName());
              }

              targetFD = createFunctionDecl(Ctx, Ctx.getTranslationUnitDecl(),
                  name, targetFD->getReturnType(), argTypes, argNames);
            }
          }
        }
      }

      // check if we have an intrinsic (math) function
      if (!targetFD) {
        QualType QT = E->getCallReturnType();
        if (Kernel->vectorize() && !compilerOptions.emitC99()) {
          QT = simdTypes.getSIMDType(QT, QT.getAsString(), SIMD4);
          assert(false && "widening of intrinsic functions not supported currently");
        }

        targetFD = builtins.getBuiltinFunction(E->getDirectCallee()->getName(),
            QT, compilerOptions.getTargetLang());
      }

      // check if this function is allowed for device execution
      if (!targetFD) {
        targetFD = cloneFunction(E->getDirectCallee());
      }
    }

    if (!targetFD) {
      unsigned DiagIDCallExpr = Diags.getCustomDiagID(DiagnosticsEngine::Error,
          "Found unsupported function call '%0' in kernel.");
      SmallVector<const char *, 16> builtinNames;
      builtins.getBuiltinNames(compilerOptions.getTargetLang(), builtinNames);
      Diags.Report(E->getExprLoc(), DiagIDCallExpr) << E->getDirectCallee()->getName();

      llvm::errs() << "Supported functions are: ";
      for (auto name : builtinNames) {
        llvm::errs() << name;
        llvm::errs() << ((name==builtinNames.back())?".\n":", ");
      }
      exit(EXIT_FAILURE);
    }

    // add ICE for CodeGen
    ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx,
        Ctx.getPointerType(targetFD->getType()), CK_FunctionToPointerDecay,
        createDeclRefExpr(Ctx, targetFD), nullptr, VK_RValue);

    // create CallExpr
    CallExpr *result = new (Ctx) CallExpr(Ctx, ICE, MultiExprArg(),
        E->getType(), E->getValueKind(), E->getRParenLoc());

    result->setNumArgs(Ctx, E->getNumArgs());
    size_t num_arg = 0;
    for (auto arg : E->arguments())
      result->setArg(num_arg++, Clone(arg));

    setExprProps(E, result);

    if (convert) {
      // add ICE for CodeGen
      ImplicitCastExpr *ICE = createImplicitCastExpr(Ctx,
          Ctx.getPointerType(convert->getType()), CK_FunctionToPointerDecay,
          createDeclRefExpr(Ctx, convert), nullptr, VK_RValue);

      // create CallExpr
      CallExpr *conv_result = new (Ctx) CallExpr(Ctx, ICE, MultiExprArg(),
          E->getType(), E->getValueKind(), E->getRParenLoc());
      conv_result->setNumArgs(Ctx, 1);
      conv_result->setArg(0, result);
      result = conv_result;
      setExprProps(E, result);
    }

    return result;
  } else {
    assert(0 && "CallExpr without FunctionDecl as Callee!");
  }
}


Expr *ASTTranslate::VisitMemberExprTranslate(MemberExpr *E) {
  // TODO: create a map with all expressions not to be cloned ..
  if (E==tileVars.local_size_x->IgnoreParenCasts() ||
      E==tileVars.local_size_y->IgnoreParenCasts() ||
      E==tileVars.local_id_x->IgnoreParenCasts() ||
      E==tileVars.local_id_y->IgnoreParenCasts() ||
      E==tileVars.block_id_x->IgnoreParenCasts() ||
      E==tileVars.block_id_y->IgnoreParenCasts()) return E;

  // replace member class variables by kernel parameter references
  // (MemberExpr 0x4bd4af0 'int' ->d 0x4bd2330
  //  (CXXThisExpr 0x4bd4ac8 'class hipacc::VerticalMeanFilter *' this))
  // -->
  // (DeclRefExpr 0x4bda540 'int' ParmVar='d' 0x4bd8010)
  ValueDecl *VD = E->getMemberDecl();
  ValueDecl *paramDecl = nullptr;

  // search for member name in kernel parameter list
  for (auto param : kernelDecl->params()) {
    // parameter name matches
    if (param->getName().equals(VD->getName())) {
      paramDecl = param;

      // get vector declaration
      if (Kernel->vectorize() && !compilerOptions.emitC99()) {
        if (KernelDeclMapVector.count(param)) {
          paramDecl = KernelDeclMapVector[param];
          llvm::errs() << "Vectorize: \n";
          paramDecl->dump();
          llvm::errs() << "\n";
        }
      }

      break;
    }
  }

  if (!paramDecl) {
    unsigned DiagIDParameter = Diags.getCustomDiagID(DiagnosticsEngine::Error,
        "Couldn't find initialization of kernel member variable '%0' in class constructor.");
    Diags.Report(E->getExprLoc(), DiagIDParameter) << VD->getName();
    exit(EXIT_FAILURE);
  }

  // check if the parameter is a Mask and replace it by a global VarDecl
  bool isMask = false;
  for (auto mask : KernelClass->getMaskFields()) {
    if (paramDecl->getName().equals(mask->getName())) {
      HipaccMask *Mask = Kernel->getMaskFromMapping(mask);

      if (Mask) {
        isMask = true;
        if (Mask->isConstant() || compilerOptions.emitC99() ||
            compilerOptions.emitCUDA()) {
          // get Mask/Domain reference
          VarDecl *maskVar = lookup<VarDecl>(Mask->getName() +
              Kernel->getName(), Mask->getType());

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

  if (!isMask) {
      // mark parameter as being used within the kernel unless for Masks and
      // Domains
      Kernel->setUsed(VD->getName());
  }

  Expr *result = createDeclRefExpr(Ctx, paramDecl);
  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitBinaryOperatorTranslate(BinaryOperator *E) {
  Expr *result;

  // remember the current CompoundStmt, which has to be the same for the LHS and
  // the RHS (the current CompoundStmt might change during cloning LHS or RHS)
  CompoundStmt *CStmt = curCStmt;
  Expr *RHS = Clone(E->getRHS());
  curCStmt = CStmt;

  // check if we have a binary assignment and an Image object on the left-hand
  // side. In case we need built-in function to write to the Image (e.g.
  // write_imagef in OpenCL), we have to replace the BinaryOperator by a
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
  if (E->getOpcode() == BO_Assign) writeImageRHS = nullptr;

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitImplicitCastExprTranslate(ImplicitCastExpr *E) {
  Expr *subExpr = Clone(E->getSubExpr());

  QualType QT = E->getType();
  CastKind CK = E->getCastKind();

  CXXCastPath castPath;
  setCastPath(E, castPath);

  Expr *litExpr = subExpr->IgnoreImpCasts();
  if (auto uo = dyn_cast<UnaryOperator>(litExpr))
    litExpr = uo->getSubExpr();

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


Expr *ASTTranslate::VisitCStyleCastExprTranslate(CStyleCastExpr *E) {
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


Expr *ASTTranslate::VisitCXXOperatorCallExprTranslate(CXXOperatorCallExpr *E) {
  Expr *result = nullptr;

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
  if (auto mask = Kernel->getMaskFromMapping(FD)) {
    MemoryAccess mem_acc = KernelClass->getMemAccess(FD);
    assert(mem_acc==READ_ONLY &&
        "only read-only memory access to Mask supported");

    switch (E->getNumArgs()) {
      default:
        assert(0 && "0, 1, or 2 arguments for Mask operator() expected!");
        break;
      case 1:
        assert(convMask && convMask==mask &&
            "0 arguments for Mask operator() only allowed within"
            "convolution lambda-function.");
        // within convolute lambda-function
        if (mask->isConstant()) {
          // propagate constants
          result = Clone(mask->getInitExpr(convIdxX, convIdxY));
        } else {
          // access mask elements
          Expr *midx_x = createIntegerLiteral(Ctx, convIdxX);
          Expr *midx_y = createIntegerLiteral(Ctx, convIdxY);

          // set Mask as being used within Kernel
          Kernel->setUsed(FD->getNameAsString());
          switch (compilerOptions.getTargetLang()) {
            case Language::C99:
            case Language::CUDA:
              // array subscript: Mask[conv_y][conv_x]
              result = accessMem2DAt(LHS, midx_x, midx_y);
              break;
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              // array subscript: Mask[(conv_y)*width + conv_x]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    (int)mask->getSizeX()), midx_x, midx_y);
              break;
            case Language::Renderscript:
            case Language::Filterscript:
              // allocation access: rsGetElementAt(Mask, conv_x, conv_y)
              result = accessMemAllocAt(LHS, mem_acc, midx_x, midx_y);
              break;
          }
        }
        break;
      case 2:
        // 0: -> (this *) Mask class
        // 1: -> (dom) Domain class
        {
        assert(isa<MemberExpr>(E->getArg(1)) && "Memory access function assumed.");
        MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(1));
        assert(isa<FieldDecl>(ME->getMemberDecl()) && "Domain must be a C++-class member.");
        FieldDecl *domFD = dyn_cast<FieldDecl>(ME->getMemberDecl());

        // look for Domain user class member variable
        assert(Kernel->getMaskFromMapping(domFD) && "Could not find Domain variable.");
        HipaccMask *Domain = Kernel->getMaskFromMapping(domFD);
        (void)Domain; // silent compiler warning
        assert(Domain->isDomain() && "Domain required.");

        assert(mask->getSizeX()==Domain->getSizeX() &&
               mask->getSizeY()==Domain->getSizeY() &&
               "Mask and Domain size must be equal.");

        // within reduce/iterate lambda-function
        if (mask->isConstant()) {
          // propagate constants
          result = Clone(mask->getInitExpr(redIdxX.back(), redIdxY.back()));
        } else {
          // access mask elements
          Expr *midx_x = createIntegerLiteral(Ctx, redIdxX.back());
          Expr *midx_y = createIntegerLiteral(Ctx, redIdxY.back());

          // set Mask as being used within Kernel
          Kernel->setUsed(FD->getNameAsString());
          switch (compilerOptions.getTargetLang()) {
            case Language::C99:
            case Language::CUDA:
              // array subscript: Mask[conv_y][conv_x]
              result = accessMem2DAt(LHS, midx_x, midx_y);
              break;
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              // array subscript: Mask[(conv_y)*width + conv_x]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    (int)mask->getSizeX()), midx_x, midx_y);
              break;
            case Language::Renderscript:
            case Language::Filterscript:
              // allocation access: rsGetElementAt(Mask, conv_x, conv_y)
              result = accessMemAllocAt(LHS, mem_acc, midx_x, midx_y);
              break;
          }
        }
        }
        break;
      case 3:
        // 0: -> (this *) Mask class
        // 1: -> x
        // 2: -> y

        // set Mask as being used within Kernel
        Kernel->setUsed(FD->getNameAsString());
        switch (compilerOptions.getTargetLang()) {
          case Language::C99:
          case Language::CUDA:
            // array subscript: Mask[y+size_y/2][x+size_x/2]
            result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                  Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                    (int)mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                createBinaryOperator(Ctx, Clone(E->getArg(2)),
                  createIntegerLiteral(Ctx, (int)mask->getSizeY()/2), BO_Add,
                  Ctx.IntTy));
            break;
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            if (mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            } else {
              // array subscript: Mask[(y+size_y/2)*width + x+size_x/2]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    (int)mask->getSizeX()), createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            }
            break;
          case Language::Renderscript:
          case Language::Filterscript:
            if (mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            } else {
              // allocation access: rsGetElementAt(Mask, x+size_x/2, y+size_y/2)
              result = accessMemAllocAt(LHS, mem_acc, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      (int)mask->getSizeX()/2), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx, (int)mask->getSizeY()/2), BO_Add,
                    Ctx.IntTy));
            }
            break;
        }
        break;
    }
  }


  // look for Image user class member variable
  if (auto acc = Kernel->getImgFromMapping(FD)) {
    MemoryAccess mem_acc = KernelClass->getMemAccess(FD);

    // Images are ParmVarDecls
    bool use_shared = false;
    DeclRefExpr *DRE = nullptr;
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
    if (acc->getSizeX() > 1) {
      if (compilerOptions.exploreConfig()) {
        TX = tileVars.local_size_x;
      } else {
        TX = createIntegerLiteral(Ctx, (int)Kernel->getNumThreadsX());
      }
    } else {
      TX = createIntegerLiteral(Ctx, 0);
    }
    if (acc->getSizeY() > 1) {
      SY = createIntegerLiteral(Ctx, (int)acc->getSizeY()/2);
    } else {
      SY = createIntegerLiteral(Ctx, 0);
    }

    HipaccMask *Mask = nullptr;
    int mask_idx_x = 0, mask_idx_y = 0;
    switch (E->getNumArgs()) {
      default:
        assert(0 && "0, 1, or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // 0: -> (this *) Image Class
        if (use_shared) {
          result = accessMemShared(DRE, TX, SY);
        } else {
          result = accessMem(LHS, acc, mem_acc);
        }
        break;
      case 2:
        // 0: -> (this *) Image Class
        // 1: -> Mask | Domain
        {
        assert(isa<MemberExpr>(E->getArg(1)->IgnoreImpCasts()) &&
            "Accessor operator() with 1 argument requires a"
            "convolution Mask or Domain as parameter.");
        MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(1)->IgnoreImpCasts());
        FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
        Mask = Kernel->getMaskFromMapping(FD);
        }
        if (convMask) {
          assert(convMask==Mask &&
              "the Mask parameter for Accessor operator(Mask) has to be"
              "the Mask parameter of the convolve method.");
          mask_idx_x = convIdxX;
          mask_idx_y = convIdxY;
        } else {
          bool found = false;
          for (unsigned int i = 0; i < redDomains.size(); ++i) {
            if (redDomains[i]==Mask) {
              mask_idx_x = redIdxX[i];
              mask_idx_y = redIdxY[i];
              found = true;
              break;
            }
          }
          assert(found &&
              "the Domain parameter for Accessor operator(Domain) has to be"
              "the Domain parameter of the reduce method.");
        }
      case 3:
        // 0: -> (this *) Image Class
        // 1: -> offset x
        // 2: -> offset y
        Expr *offset_x, *offset_y;
        if (E->getNumArgs()==2) {
          offset_x = createIntegerLiteral(Ctx,
              mask_idx_x-(int)Mask->getSizeX()/2);
          offset_y = createIntegerLiteral(Ctx,
              mask_idx_y-(int)Mask->getSizeY()/2);
        } else {
          offset_x = Clone(E->getArg(1));
          offset_y = Clone(E->getArg(2));
        }

        if (use_shared) {
          result = accessMemShared(DRE, createBinaryOperator(Ctx, offset_x,
                TX, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx, offset_y,
                  SY, BO_Add, Ctx.IntTy));
        } else {
          switch (mem_acc) {
            case READ_ONLY:
              if (bh_variant.borderVal) {
                return addBorderHandling(LHS, offset_x, offset_y, acc);
              }
              // fall through
            case WRITE_ONLY:
            case READ_WRITE:
            case UNDEFINED:
              result = accessMem(LHS, acc, mem_acc, offset_x, offset_y);
              break;
          }
        }
        break;
    }
  }

  setExprProps(E, result);

  return result;
}


Expr *ASTTranslate::VisitCXXMemberCallExprTranslate(CXXMemberCallExpr *E) {
  assert(isa<MemberExpr>(E->getCallee()) &&
      "Hipacc: Stumbled upon unsupported expression or statement: CXXMemberCallExpr");
  MemberExpr *ME = cast<MemberExpr>(E->getCallee());

  auto mem_at_fun = [&] (HipaccAccessor *acc, DeclRefExpr *LHS,
                         MemoryAccess mem_acc) -> Expr * {
    assert(E->getNumArgs()==2 &&
           "x and y argument for pixel_at() or output_at() required!");
    Expr *idx_x = addGlobalOffsetX(Clone(E->getArg(0)), acc);
    Expr *idx_y = addGlobalOffsetY(Clone(E->getArg(1)), acc);
    Expr *result = nullptr;

    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
        result = accessMem2DAt(LHS, idx_x, idx_y);
        break;
      case Language::CUDA:
        if (Kernel->useTextureMemory(acc)!=Texture::None) {
          result = accessMemTexAt(LHS, acc, mem_acc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(acc), idx_x, idx_y);
        }
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (Kernel->useTextureMemory(acc)!=Texture::None) {
          result = accessMemImgAt(LHS, acc, mem_acc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(acc), idx_x, idx_y);
        }
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        if (ME->getMemberNameInfo().getAsString() == "output_at" &&
            compilerOptions.emitFilterscript()) {
            assert(0 && "Filterscript does not support output_at().");
        }
        result = accessMemAllocAt(LHS, mem_acc, idx_x, idx_y);
        break;
    }

    setExprProps(E, result);

    return result;
  };

  if (isa<CXXThisExpr>(ME->getBase()->IgnoreImpCasts())) {
    // check if this is a convolve function call
    if (E->getDirectCallee() && (
          E->getDirectCallee()->getName().equals("convolve") ||
          E->getDirectCallee()->getName().equals("reduce") ||
          E->getDirectCallee()->getName().equals("iterate"))) {
      return convertConvolution(E);
    }

    // Kernel context -> use Iteration Space output Accessor
    auto LHS = outputImage;
    HipaccAccessor *acc = Kernel->getIterationSpace();
    MemoryAccess mem_acc = KernelClass->getMemAccess(KernelClass->getOutField());

    // x() method -> gid_x - is_offset_x
    if (ME->getMemberNameInfo().getAsString() == "x") {
      return createParenExpr(Ctx, removeISOffsetX(tileVars.global_id_x));
    }

    // y() method -> gid_y
    if (ME->getMemberNameInfo().getAsString() == "y") {
      if (compilerOptions.emitC99() ||
          compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        return createParenExpr(Ctx, removeISOffsetY(gidYRef));
      } else {
        return gidYRef;
      }
    }

    // output() method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output") {
      assert(E->getNumArgs()==0 && "no arguments for output() method supported!");
      Expr *result = nullptr;

      switch (compilerOptions.getTargetLang()) {
        case Language::Renderscript:
          if (Kernel->getPixelsPerThread() <= 1) {
            // write to output pixel pointed to by kernel parameter
            LHS = retValRef;
          }
          // fall through
        case Language::C99:
        case Language::CUDA:
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          result = accessMem(LHS, acc, mem_acc);
          break;
        case Language::Filterscript:
          postStmts.push_back(createReturnStmt(Ctx, retValRef));
          postCStmt.push_back(curCStmt);
          result = retValRef;
          break;
      }

      setExprProps(E, result);

      return result;
    }

    // output_at(x, y) method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output_at") {
      return mem_at_fun(acc, LHS, mem_acc);
    }
  }

  if (auto base = dyn_cast<MemberExpr>(ME->getBase()->IgnoreImpCasts())) {
    FieldDecl *FD = dyn_cast<FieldDecl>(base->getMemberDecl());

    if (auto acc = Kernel->getImgFromMapping(FD)) {
      MemoryAccess mem_acc = KernelClass->getMemAccess(FD);

      // Acc.x() method -> acc_scale_x * (gid_x - is_offset_x)
      if (ME->getMemberNameInfo().getAsString() == "x") {
        // remove is_offset_x and scale index to Accessor size
        if (acc->getInterpolationMode() != Interpolate::NO) {
          return createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
              createParenExpr(Ctx, addNNInterpolationX(acc,
                  tileVars.global_id_x)), nullptr,
              Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
        } else {
          return createParenExpr(Ctx, removeISOffsetX(tileVars.global_id_x));
        }
      }

      // Acc.y() method -> acc_scale_y * gid_y
      if (ME->getMemberNameInfo().getAsString() == "y") {
        Expr *idx_y = gidYRef;
        // scale index to Accessor size
        if (acc->getInterpolationMode() != Interpolate::NO) {
          idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
              createParenExpr(Ctx, addNNInterpolationY(acc, idx_y)), nullptr,
              Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
        } else if (compilerOptions.emitRenderscript() ||
            compilerOptions.emitFilterscript()) {
          idx_y = createParenExpr(Ctx, removeISOffsetY(gidYRef));
        }

        return idx_y;
      }

      // Acc.pixel_at(x, y) method -> img[y][x]
      if (ME->getMemberNameInfo().getAsString() == "pixel_at") {
        // MemberExpr is converted to DeclRefExpr when cloning
        auto LHS = cast<DeclRefExpr>(Clone(ME->getBase()->IgnoreImpCasts()));
        return mem_at_fun(acc, LHS, mem_acc);
      }
    }

    if (auto mask = Kernel->getMaskFromMapping(FD)) {
      if (mask->isDomain()) {
        bool isDomainValid = false;
        int redDepth = 0;

        // search corresponding domain
        for (size_t i=0, e=redDomains.size(); i!=e; ++i) {
          if (mask == redDomains[i]) {
            isDomainValid = true;
            redDepth = i;
            break;
          }
        }

        assert(isDomainValid && "Getting Domain reduction IDs is only allowed "
                                "within reduction lambda-function.");
        // within convolute lambda-function
        if (ME->getMemberNameInfo().getAsString() == "x") {
          return createIntegerLiteral(Ctx,
              redIdxX[redDepth] - (int)redDomains[redDepth]->getSizeX()/2);
        }
        if (ME->getMemberNameInfo().getAsString() == "y") {
          return createIntegerLiteral(Ctx,
              redIdxY[redDepth] - (int)redDomains[redDepth]->getSizeY()/2);
        }
      } else {
        assert(mask==convMask && "Getting Mask convolution IDs is only allowed "
                                 "allowed within convolution lambda-function.");
        // within convolute lambda-function
        if (ME->getMemberNameInfo().getAsString() == "x") {
          return createIntegerLiteral(Ctx, convIdxX - (int)mask->getSizeX()/2);
        }
        if (ME->getMemberNameInfo().getAsString() == "y") {
          return createIntegerLiteral(Ctx, convIdxY - (int)mask->getSizeY()/2);
        }
      }
    }
  }

  assert(0 && "Hipacc: Stumbled upon unsupported expression: CXXMemberCallExpr");
  return nullptr;
}

// vim: set ts=2 sw=2 sts=2 et ai:

