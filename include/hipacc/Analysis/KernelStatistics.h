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

//===--- KernelStatistics.h - Statistics and Analysis of Kernels ----------===//
//
// This file implements various statistics and analysis for source-level CFGs of
// kernel functions.
// Statistics include number of instructions (ALU/SPU) and memory operations
// (global memory, constant memory).
// Analysis include use-def analysis for vectorization.
//
//===----------------------------------------------------------------------===//

#ifndef _KERNELSTATISTICS_H_
#define _KERNELSTATISTICS_H_

#include <clang/AST/ASTContext.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Analysis/AnalysisContext.h>
#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <clang/Basic/Diagnostic.h>

#include "hipacc/Device/TargetDescription.h"
#include "hipacc/DSL/CompilerKnownClasses.h"

namespace clang {
namespace hipacc {
// read/write analysis of image accesses
enum MemoryAccess {
  UNDEFINED   = 0x0,
  READ_ONLY   = 0x1,
  WRITE_ONLY  = 0x2,
  READ_WRITE  = (READ_ONLY | WRITE_ONLY)
};

// stride analysis of image accesses
enum MemoryAccessDetail {
  NO_STRIDE   = 0x1,
  STRIDE_X    = 0x2,
  STRIDE_Y    = 0x4,
  STRIDE_XY   = 0x8,
  USER_XY     = 0x10
};

// vectorization information for variables
enum VectorInfo {
  SCALAR      = 0x0,
  VECTORIZE   = 0x1,
  PROPAGATE   = 0x2
};

class KernelStatistics : public ManagedAnalysis {
  private:
    KernelStatistics(void *impl);
    void *impl;

  public:
    MemoryAccess getMemAccess(const FieldDecl *FD);
    MemoryAccessDetail getMemAccessDetail(const FieldDecl *FD);
    MemoryAccessDetail getOutAccessDetail();
    VectorInfo getVectorizeInfo(const VarDecl *VD);
    KernelType getKernelType();

    virtual ~KernelStatistics();

    static KernelStatistics *computeKernelStatistics(AnalysisDeclContext
        &analysisContext, StringRef name, CompilerKnownClasses
        &compilerClasses);

    static KernelStatistics *create(AnalysisDeclContext &analysisContext,
        StringRef name, CompilerKnownClasses &compilerClasses) {
      return computeKernelStatistics(analysisContext, name, compilerClasses);
    }

    static void setAnalysisOptions(AnalysisDeclContext &AC) {
      // specify which statement nodes should be part of the CFG
      AC.getCFGBuildOptions().PruneTriviallyFalseEdges = false;
      AC.getCFGBuildOptions().AddEHEdges = false;
      AC.getCFGBuildOptions().AddInitializers = false;
      AC.getCFGBuildOptions().AddImplicitDtors = false;

      //AC.getCFGBuildOptions().setAllAlwaysAdd();
      AC.getCFGBuildOptions()
        .setAlwaysAdd(Stmt::DeclStmtClass)
        .setAlwaysAdd(Stmt::DeclRefExprClass)
        .setAlwaysAdd(Stmt::DoStmtClass)
        .setAlwaysAdd(Stmt::BinaryConditionalOperatorClass)
        .setAlwaysAdd(Stmt::ConditionalOperatorClass)
        .setAlwaysAdd(Stmt::ArraySubscriptExprClass)
        .setAlwaysAdd(Stmt::BinaryOperatorClass)
        .setAlwaysAdd(Stmt::CallExprClass)
        .setAlwaysAdd(Stmt::CStyleCastExprClass)
        .setAlwaysAdd(Stmt::ChooseExprClass)
        .setAlwaysAdd(Stmt::CompoundLiteralExprClass)
        .setAlwaysAdd(Stmt::StmtExprClass)
        .setAlwaysAdd(Stmt::LambdaExprClass)
        .setAlwaysAdd(Stmt::UnaryOperatorClass)
        .setAlwaysAdd(Stmt::ForStmtClass)
        .setAlwaysAdd(Stmt::IfStmtClass)
        .setAlwaysAdd(Stmt::CaseStmtClass)
        .setAlwaysAdd(Stmt::DefaultStmtClass)
        .setAlwaysAdd(Stmt::SwitchStmtClass);
      //.setAlwaysAdd(Stmt::CXXMemberCallExprClass)
      //.setAlwaysAdd(Stmt::CXXOperatorCallExprClass)
      //.setAlwaysAdd(Stmt::MemberExprClass)
      //.setAlwaysAdd(Stmt::CXXThisExprClass)
      //.setAlwaysAdd(Stmt::ImplicitCastExprClass)
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _KERNELSTATISTICS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

