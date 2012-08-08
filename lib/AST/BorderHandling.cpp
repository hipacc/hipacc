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
Expr *ASTTranslate::addBorderHandling(DeclRefExpr *LHS, Expr *EX, Expr *EY,
    HipaccAccessor *Acc) {
  return addBorderHandling(LHS, EX, EY, Acc, bhStmtsVistor, bhCStmtsVistor);
}
Expr *ASTTranslate::addBorderHandling(DeclRefExpr *LHS, Expr *EX, Expr *EY,
    HipaccAccessor *Acc, llvm::SmallVector<Stmt *, 16> &bhStmts,
    llvm::SmallVector<CompoundStmt *, 16> &bhCStmts) {
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

  // gid_x = gid_x - is_offset_x + offset_x + xf
  Expr *tmp_x_ref = gidXRef;
  if (Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl()) {
    tmp_x_ref = createBinaryOperator(Ctx, tmp_x_ref,
        Kernel->getIterationSpace()->getAccessor()->getOffsetXDecl(), BO_Sub,
        Ctx.IntTy);
  }
  if (Acc->getOffsetXDecl()) {
    tmp_x_ref = createBinaryOperator(Ctx, tmp_x_ref, Acc->getOffsetXDecl(),
        BO_Add, Ctx.IntTy);
  }
  if (EX) {
    VarDecl *tmp_x = createVarDecl(Ctx, kernelDecl, LSSX.str(), Ctx.IntTy,
        createBinaryOperator(Ctx, tmp_x_ref, Clone(EX), BO_Add, Ctx.IntTy));
    DC->addDecl(tmp_x);
    tmp_x_ref = createDeclRefExpr(Ctx, tmp_x);
    bhStmts.push_back(createDeclStmt(Ctx, tmp_x));
    bhCStmts.push_back(curCompoundStmtVistor);
  }

  // gid_y = gid_y + offset_y + yf
  Expr *tmp_y_ref = gidYRef;
  if (Acc->getOffsetYDecl()) {
    tmp_y_ref = createBinaryOperator(Ctx, tmp_y_ref, Acc->getOffsetYDecl(),
        BO_Add, Ctx.IntTy);
  }
  if (EY) {
    VarDecl *tmp_y = createVarDecl(Ctx, kernelDecl, LSSY.str(), Ctx.IntTy,
        createBinaryOperator(Ctx, tmp_y_ref, Clone(EY), BO_Add, Ctx.IntTy));
    DC->addDecl(tmp_y);
    tmp_y_ref = createDeclRefExpr(Ctx, tmp_y);
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
    if (bh_variant.borders.right && EX) {
      // < _gid_x<0> >= offset_x+width >
      bo_constant = addConstantUpper(Acc, tmp_x_ref, upperX, bo_constant);
    }
    if (bh_variant.borders.bottom && EY) {
      // if (_gid_y<0> >= offset_y+height)
      bo_constant = addConstantUpper(Acc, tmp_y_ref, upperY, bo_constant);
    }
    if (bh_variant.borders.left && EX) {
      // if (_gid_x<0> < offset_x)
      bo_constant = addConstantLower(Acc, tmp_x_ref, lowerX, bo_constant);
    }
    if (bh_variant.borders.top && EY) {
      // if (_gid_y<0> < offset_y)
      bo_constant = addConstantLower(Acc, tmp_y_ref, lowerY, bo_constant);
    }

    if (Kernel->useTextureMemory(Acc)) {
      if (compilerOptions.emitCUDA()) {
        RHS = accessMemTexAt(LHS, Acc, READ_ONLY, tmp_x_ref, tmp_y_ref);
      } else {
        RHS = accessMemImgAt(LHS, Acc, READ_ONLY, tmp_x_ref, tmp_y_ref);
      }
    } else {
      RHS = accessMemArrAt(LHS, Acc->getStrideDecl(), tmp_x_ref, tmp_y_ref);
    }
    RHS->setValueDependent(LHS->isValueDependent());
    RHS->setTypeDependent(LHS->isTypeDependent());

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
      default:
        assert(0 && "addBorderHandling && BOUNDARY_UNDEFINED!");
        break;
    }

    if (bh_variant.borders.right && EX) {
      bhStmts.push_back((*this.*upperFun)(Acc, tmp_x_ref, upperX));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.bottom && EY) {
      bhStmts.push_back((*this.*upperFun)(Acc, tmp_y_ref, upperY));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.left && EX) {
      bhStmts.push_back((*this.*lowerFun)(Acc, tmp_x_ref, lowerX));
      bhCStmts.push_back(curCompoundStmtVistor);
    }
    if (bh_variant.borders.top && EY) {
      bhStmts.push_back((*this.*lowerFun)(Acc, tmp_y_ref, lowerY));
      bhCStmts.push_back(curCompoundStmtVistor);
    }

    // get data
    if (Kernel->useTextureMemory(Acc)) {
      if (compilerOptions.emitCUDA()) {
        result = accessMemTexAt(LHS, Acc, READ_ONLY, tmp_x_ref, tmp_y_ref);
      } else {
        result = accessMemImgAt(LHS, Acc, READ_ONLY, tmp_x_ref, tmp_y_ref);
      }
    } else {
      result = accessMemArrAt(LHS, Acc->getStrideDecl(), tmp_x_ref, tmp_y_ref);
    }
    result->setValueDependent(LHS->isValueDependent());
    result->setTypeDependent(LHS->isTypeDependent());
  }

  return result;
}

// vim: set ts=2 sw=2 sts=2 et ai:

