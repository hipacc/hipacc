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

//===--- BorderHandling.cpp - Add Border Handling Support in the AST ------===//
//
// This file implements the extension of memory accesses for border handling.
//
//===----------------------------------------------------------------------===//

#include "hipacc/AST/ASTTranslate.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;


// add border handling: CLAMP
Stmt *ASTTranslate::addClampUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper,
    bool) {
  // if (idx >= upper) idx = upper-1;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createIntegerLiteral(Ctx, 1), BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr, nullptr);
}
Stmt *ASTTranslate::addClampLower(HipaccAccessor *Acc, Expr *idx, Expr *lower,
    bool) {
  // if (idx < lower) idx = lower;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx, lower,
        BO_Assign, Ctx.IntTy), nullptr, nullptr);
}


// add border handling: REPEAT
Stmt *ASTTranslate::addRepeatUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper,
    bool is_x) {
  // while (idx >= upper) idx -= is_width | is_height;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);
  Expr *stride = is_x ? getWidthDecl(Acc) : getHeightDecl(Acc);

  return createWhileStmt(Ctx, nullptr, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, stride, BO_Sub, Ctx.IntTy), BO_Assign,
        Ctx.IntTy));
}
Stmt *ASTTranslate::addRepeatLower(HipaccAccessor *Acc, Expr *idx, Expr *lower,
    bool is_x) {
  // while (idx < lower) idx += is_width | is_height;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);
  Expr *stride = is_x ? getWidthDecl(Acc) : getHeightDecl(Acc);

  return createWhileStmt(Ctx, nullptr, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, stride, BO_Add, Ctx.IntTy), BO_Assign,
        Ctx.IntTy));
}


// add border handling: MIRROR
Stmt *ASTTranslate::addMirrorUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper,
    bool) {
  // if (idx >= upper) idx = upper - (idx+1 - upper);
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createParenExpr(Ctx,
            createBinaryOperator(Ctx, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Add, Ctx.IntTy),
              createParenExpr(Ctx, upper), BO_Sub, Ctx.IntTy)) , BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr, nullptr);
}
Stmt *ASTTranslate::addMirrorLower(HipaccAccessor *Acc, Expr *idx, Expr *lower,
    bool) {
  // if (idx < lower) idx = lower + (lower - idx-1);
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, lower, createParenExpr(Ctx,
            createBinaryOperator(Ctx, lower, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Sub, Ctx.IntTy), BO_Sub,
              Ctx.IntTy)) , BO_Add, Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr,
      nullptr);
}


// add border handling: CONSTANT
Expr *ASTTranslate::addConstantUpper(HipaccAccessor *Acc, Expr *idx, Expr
    *upper, Expr *cond) {
  // (idx < upper)
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_LT, Ctx.BoolTy);

  if (cond) {
    return createBinaryOperator(Ctx, bo_upper, cond, BO_LAnd, Ctx.BoolTy);
  } else {
    return bo_upper;
  }
}
Expr *ASTTranslate::addConstantLower(HipaccAccessor *Acc, Expr *idx, Expr
    *lower, Expr *cond) {
  // (idx >= lower)
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_GE, Ctx.BoolTy);

  if (cond) {
    return createBinaryOperator(Ctx, bo_lower, cond, BO_LAnd, Ctx.BoolTy);
  } else {
    return bo_lower;
  }
}


// add border handling statements to the AST
Expr *ASTTranslate::addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x,
    Expr *local_offset_y, HipaccAccessor *Acc) {
  return addBorderHandling(LHS, local_offset_x, local_offset_y, Acc, preStmts,
      preCStmt);
}
Expr *ASTTranslate::addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x,
    Expr *local_offset_y, HipaccAccessor *Acc, SmallVector<Stmt *, 16> &bhStmts,
    SmallVector<CompoundStmt *, 16> &bhCStmt) {
  Expr *result = nullptr;
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);

  std::stringstream LSSX, LSSY, LSST;
  LSSX << "_gid_x" << literalCount;
  LSSY << "_gid_y" << literalCount;
  LSST << "_tmp" << literalCount;
  literalCount++;

  Expr *lowerX, *upperX, *lowerY, *upperY;
  if (Acc->getOffsetXDecl()) {
    lowerX = getOffsetXDecl(Acc);
    upperX = createBinaryOperator(Ctx, getOffsetXDecl(Acc), getWidthDecl(Acc),
        BO_Add, Ctx.IntTy);
  } else {
    lowerX = createIntegerLiteral(Ctx, 0);
    upperX = getWidthDecl(Acc);
  }
  if (Acc->getOffsetYDecl()) {
    lowerY = getOffsetYDecl(Acc);
    upperY = createBinaryOperator(Ctx, getOffsetYDecl(Acc), getHeightDecl(Acc),
        BO_Add, Ctx.IntTy);
  } else {
    lowerY = createIntegerLiteral(Ctx, 0);
    upperY = getHeightDecl(Acc);
  }

  Expr *idx_x = tileVars.global_id_x;
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
      if ((compilerOptions.emitC() ||
           compilerOptions.emitRenderscript() ||
           compilerOptions.emitFilterscript()) &&
          Acc!=Kernel->getIterationSpace()->getAccessor()) {
        idx_y = removeISOffsetY(idx_y, Acc);
      }
      break;
    case InterpolateNN:
      idx_x = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), nullptr,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), nullptr,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      break;
    case InterpolateLF:
    case InterpolateCF:
    case InterpolateL3:
      return addInterpolationCall(LHS, Acc, idx_x, idx_y);
  }

  // step 2: add global Accessor/Iteration Space offset
  if (Acc!=Kernel->getIterationSpace()->getAccessor()) {
    idx_x = addGlobalOffsetX(idx_x, Acc);
    idx_y = addGlobalOffsetY(idx_y, Acc);
  } else {
    if (!(compilerOptions.emitC() ||
          compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript())) {
      idx_y = addGlobalOffsetY(idx_y, Acc);
    }
  }

  // add temporary variables for updated idx_x and idx_y
  if (local_offset_x) {
    VarDecl *tmp_x = createVarDecl(Ctx, kernelDecl, LSSX.str(), Ctx.IntTy,
        idx_x);
    DC->addDecl(tmp_x);
    idx_x = createDeclRefExpr(Ctx, tmp_x);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_x));
    bhCStmt.push_back(curCStmt);
  }

  if (local_offset_y) {
    VarDecl *tmp_y = createVarDecl(Ctx, kernelDecl, LSSY.str(), Ctx.IntTy,
        idx_y);
    DC->addDecl(tmp_y);
    idx_y = createDeclRefExpr(Ctx, tmp_y);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_y));
    bhCStmt.push_back(curCStmt);
  }

  if (Acc->getBoundaryHandling() == BOUNDARY_CONSTANT) {
    // <type> _tmp<0> = const_val;
    Expr *RHS = nullptr;
    Expr *const_val = Acc->getConstExpr();
    VarDecl *tmp_t = createVarDecl(Ctx, kernelDecl, LSST.str(),
        const_val->getType(), const_val);

    DC->addDecl(tmp_t);
    DeclRefExpr *tmp_t_ref = createDeclRefExpr(Ctx, tmp_t);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_t));
    bhCStmt.push_back(curCStmt);

    Expr *bo_constant = nullptr;
    if (bh_variant.borders.right && local_offset_x) {
      // < _gid_x<0> >= offset_x+width >
      bo_constant = addConstantUpper(Acc, idx_x, upperX, bo_constant);
    }
    if (bh_variant.borders.bottom && local_offset_y) {
      // if (_gid_y<0> >= offset_y+height)
      bo_constant = addConstantUpper(Acc, idx_y, upperY, bo_constant);
    }
    if (bh_variant.borders.left && local_offset_x) {
      // if (_gid_x<0> < offset_x)
      bo_constant = addConstantLower(Acc, idx_x, lowerX, bo_constant);
    }
    if (bh_variant.borders.top && local_offset_y) {
      // if (_gid_y<0> < offset_y)
      bo_constant = addConstantLower(Acc, idx_y, lowerY, bo_constant);
    }

    switch (compilerOptions.getTargetCode()) {
      case TARGET_C:
          RHS = accessMem2DAt(LHS, idx_x, idx_y);
          break;
      case TARGET_CUDA:
        if (Kernel->useTextureMemory(Acc)) {
          RHS = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        // fall through
      case TARGET_OpenCLACC:
      case TARGET_OpenCLCPU:
      case TARGET_OpenCLGPU:
        if (Kernel->useTextureMemory(Acc)) {
          RHS = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        RHS = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        break;
      case TARGET_Renderscript:
      case TARGET_Filterscript:
        RHS = accessMemAllocAt(LHS, READ_ONLY, idx_x, idx_y);
        break;
    }
    setExprProps(LHS, RHS);

    // tmp<0> = RHS;
    if (bo_constant) {
      bhStmts.push_back(createIfStmt(Ctx, bo_constant, createBinaryOperator(Ctx,
                      tmp_t_ref, RHS, BO_Assign, tmp_t_ref->getType()), nullptr,
                  nullptr));
      bhCStmt.push_back(curCStmt);
    } else {
      bhStmts.push_back(createBinaryOperator(Ctx, tmp_t_ref, RHS, BO_Assign,
            tmp_t_ref->getType()));
      bhCStmt.push_back(curCStmt);
    }
    result = tmp_t_ref;
  } else {
    Stmt *(clang::hipacc::ASTTranslate::*lowerFun)
      (HipaccAccessor *Acc, Expr *idx, Expr *lower, bool) = nullptr;
    Stmt *(clang::hipacc::ASTTranslate::*upperFun)
      (HipaccAccessor *Acc, Expr *idx, Expr *upper, bool) = nullptr;
    switch (Acc->getBoundaryHandling()) {
      case BOUNDARY_CLAMP:
        lowerFun = &clang::hipacc::ASTTranslate::addClampLower;
        upperFun = &clang::hipacc::ASTTranslate::addClampUpper;
        break;
      case BOUNDARY_REPEAT:
        lowerFun = &clang::hipacc::ASTTranslate::addRepeatLower;
        upperFun = &clang::hipacc::ASTTranslate::addRepeatUpper;
        break;
      case BOUNDARY_MIRROR:
        lowerFun = &clang::hipacc::ASTTranslate::addMirrorLower;
        upperFun = &clang::hipacc::ASTTranslate::addMirrorUpper;
        break;
      case BOUNDARY_UNDEFINED:
        // in case of exploration boundary handling variants are required
        if (!compilerOptions.exploreConfig()) {
          assert(0 && "addBorderHandling && BOUNDARY_UNDEFINED!");
        }
        break;
      case BOUNDARY_CONSTANT:
        assert(0 && "addBorderHandling && BOUNDARY_CONSTANT!");
        break;
    }

    if (upperFun) {
      if (bh_variant.borders.right && local_offset_x) {
        bhStmts.push_back((*this.*upperFun)(Acc, idx_x, upperX, true));
        bhCStmt.push_back(curCStmt);
      }
      if (bh_variant.borders.bottom && local_offset_y) {
        bhStmts.push_back((*this.*upperFun)(Acc, idx_y, upperY, false));
        bhCStmt.push_back(curCStmt);
      }
    }
    if (lowerFun) {
      if (bh_variant.borders.left && local_offset_x) {
        bhStmts.push_back((*this.*lowerFun)(Acc, idx_x, lowerX, true));
        bhCStmt.push_back(curCStmt);
      }
      if (bh_variant.borders.top && local_offset_y) {
        bhStmts.push_back((*this.*lowerFun)(Acc, idx_y, lowerY, false));
        bhCStmt.push_back(curCStmt);
      }
    }

    // get data
    switch (compilerOptions.getTargetCode()) {
      case TARGET_C:
          result = accessMem2DAt(LHS, idx_x, idx_y);
          break;
      case TARGET_CUDA:
        if (Kernel->useTextureMemory(Acc)) {
          result = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        // fall through
      case TARGET_OpenCLACC:
      case TARGET_OpenCLCPU:
      case TARGET_OpenCLGPU:
        if (Kernel->useTextureMemory(Acc)) {
          result = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        result = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        break;
      case TARGET_Renderscript:
      case TARGET_Filterscript:
        result = accessMemAllocAt(LHS, READ_ONLY, idx_x, idx_y);
        break;
    }
    setExprProps(LHS, result);
  }

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

