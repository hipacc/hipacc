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
Stmt *ASTTranslate::addClampUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper) {
  // if (idx >= upper) idx = upper-1;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createIntegerLiteral(Ctx, 1), BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), NULL, NULL);
}
Stmt *ASTTranslate::addClampLower(HipaccAccessor *Acc, Expr *idx, Expr *lower) {
  // if (idx < lower) idx = lower;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx, lower,
        BO_Assign, Ctx.IntTy), NULL, NULL);
}


// add border handling: REPEAT
Stmt *ASTTranslate::addRepeatUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper) {
  // while (idx >= upper) idx -= stride;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createWhileStmt(Ctx, NULL, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, Acc->getWidthDecl(), BO_Sub, Ctx.IntTy),
        BO_Assign, Ctx.IntTy));
}
Stmt *ASTTranslate::addRepeatLower(HipaccAccessor *Acc, Expr *idx, Expr *lower) {
  // while (idx < lower) idx += stride;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createWhileStmt(Ctx, NULL, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, Acc->getWidthDecl(), BO_Add, Ctx.IntTy),
        BO_Assign, Ctx.IntTy));
}


// add border handling: MIRROR
Stmt *ASTTranslate::addMirrorUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper) {
  // if (idx >= upper) idx = upper - (idx+1 - upper);
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createParenExpr(Ctx,
            createBinaryOperator(Ctx, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Add, Ctx.IntTy),
              createParenExpr(Ctx, upper), BO_Sub, Ctx.IntTy)) , BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), NULL, NULL);
}
Stmt *ASTTranslate::addMirrorLower(HipaccAccessor *Acc, Expr *idx, Expr *lower) {
  // if (idx < lower) idx = lower + (lower - idx-1);
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, lower, createParenExpr(Ctx,
            createBinaryOperator(Ctx, lower, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Sub, Ctx.IntTy), BO_Sub,
              Ctx.IntTy)) , BO_Add, Ctx.IntTy), BO_Assign, Ctx.IntTy), NULL,
      NULL);
}


// add border handling: CONSTANT
Expr *ASTTranslate::addConstantUpper(HipaccAccessor *Acc, Expr *idx, Expr
    *upper, Expr* cond) {
  // (idx < upper)
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_LT, Ctx.BoolTy);

  if (cond) {
    return createBinaryOperator(Ctx, bo_upper, cond, BO_LAnd, Ctx.BoolTy);
  } else {
    return bo_upper;
  }
}
Expr *ASTTranslate::addConstantLower(HipaccAccessor *Acc, Expr *idx, Expr
    *lower, Expr* cond) {
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
  return addBorderHandling(LHS, local_offset_x, local_offset_y, Acc,
      bhStmtsVistor, bhCStmtsVistor);
}
Expr *ASTTranslate::addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x,
    Expr *local_offset_y, HipaccAccessor *Acc, SmallVector<Stmt *, 16> &bhStmts,
    SmallVector<CompoundStmt *, 16> &bhCStmts) {
  Expr *RHS, *result;
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);

  std::stringstream LSSX, LSSY, LSST;
  LSSX << "_gid_x" << literalCount;
  LSSY << "_gid_y" << literalCount;
  LSST << "_tmp" << literalCount;
  literalCount++;

  Expr *lowerX, *upperX, *lowerY, *upperY;
  if (Acc->getOffsetXDecl()) {
    lowerX = Acc->getOffsetXDecl();
    upperX = createBinaryOperator(Ctx, Acc->getOffsetXDecl(),
        Acc->getWidthDecl(), BO_Add, Ctx.IntTy);
  } else {
    lowerX = createIntegerLiteral(Ctx, 0);
    upperX = Acc->getWidthDecl();
  }
  if (Acc->getOffsetYDecl()) {
    lowerY = Acc->getOffsetYDecl();
    upperY = createBinaryOperator(Ctx, Acc->getOffsetYDecl(),
        Acc->getHeightDecl(), BO_Add, Ctx.IntTy);
  } else {
    lowerY = createIntegerLiteral(Ctx, 0);
    upperY = Acc->getHeightDecl();
  }

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
          createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), NULL,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), NULL,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
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

  // add temporary variables for updated idx_x and idx_y
  if (local_offset_x) {
    VarDecl *tmp_x = createVarDecl(Ctx, kernelDecl, LSSX.str(), Ctx.IntTy,
        idx_x);
    DC->addDecl(tmp_x);
    idx_x = createDeclRefExpr(Ctx, tmp_x);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_x));
    bhCStmts.push_back(curCompoundStmtVistor);
  }

  if (local_offset_y) {
    VarDecl *tmp_y = createVarDecl(Ctx, kernelDecl, LSSY.str(), Ctx.IntTy,
        idx_y);
    DC->addDecl(tmp_y);
    idx_y = createDeclRefExpr(Ctx, tmp_y);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_y));
    bhCStmts.push_back(curCompoundStmtVistor);
  }

  if (Acc->getBoundaryHandling() == BOUNDARY_CONSTANT) {
    // <type> _tmp<0> = const_val;
    Expr *const_val = Acc->getConstExpr();
    VarDecl *tmp_t = createVarDecl(Ctx, kernelDecl, LSST.str(),
        const_val->getType(), const_val);

    DC->addDecl(tmp_t);
    DeclRefExpr *tmp_t_ref = createDeclRefExpr(Ctx, tmp_t);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_t));
    bhCStmts.push_back(curCompoundStmtVistor);

    Expr *bo_constant = NULL;
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

    if (Kernel->useTextureMemory(Acc)) {
      if (compilerOptions.emitCUDA()) {
        RHS = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
      } else {
        RHS = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
      }
    } else {
      RHS = accessMemArrAt(LHS, Acc->getStrideDecl(), idx_x, idx_y);
    }
    setExprProps(LHS, RHS);

    // tmp<0> = RHS;
    if (bo_constant) {
      bhStmts.push_back(createIfStmt(Ctx, bo_constant, createBinaryOperator(Ctx,
              tmp_t_ref, RHS, BO_Assign, tmp_t_ref->getType()), NULL, NULL));
    } else {
      bhStmts.push_back(createBinaryOperator(Ctx, tmp_t_ref, RHS, BO_Assign,
            tmp_t_ref->getType()));
    }
    bhCStmts.push_back(curCompoundStmtVistor);
    result = tmp_t_ref;
  } else {
    Stmt *(clang::hipacc::ASTTranslate::*lowerFun)
      (HipaccAccessor *Acc, Expr *idx, Expr *lower) = NULL;
    Stmt *(clang::hipacc::ASTTranslate::*upperFun)
      (HipaccAccessor *Acc, Expr *idx, Expr *upper) = NULL;
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
      case BOUNDARY_CONSTANT:
      case BOUNDARY_UNDEFINED:
        assert(0 && "addBorderHandling && BOUNDARY_UNDEFINED!");
        break;
    }

    if (bh_variant.borders.right && local_offset_x) {
      bhStmts.push_back((*this.*upperFun)(Acc, idx_x, upperX));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.bottom && local_offset_y) {
      bhStmts.push_back((*this.*upperFun)(Acc, idx_y, upperY));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.left && local_offset_x) {
      bhStmts.push_back((*this.*lowerFun)(Acc, idx_x, lowerX));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.top && local_offset_y) {
      bhStmts.push_back((*this.*lowerFun)(Acc, idx_y, lowerY));
      bhCStmts.push_back(curCompoundStmtVistor);
    }

    // get data
    if (Kernel->useTextureMemory(Acc)) {
      if (compilerOptions.emitCUDA()) {
        result = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
      } else {
        result = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
      }
    } else {
      result = accessMemArrAt(LHS, Acc->getStrideDecl(), idx_x, idx_y);
    }
    setExprProps(LHS, result);
  }

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

