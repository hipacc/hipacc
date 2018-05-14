//
// Copyright (c) 2018, University of Erlangen-Nuremberg
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

//===--- ASTFuse.h - Kernel Fusion for the AST --------------------------===//
//
// This file implements the fusion and printing of the translated kernels.
//
//===--------------------------------------------------------------------===//

#ifndef _ASTFUSE_H_
#define _ASTFUSE_H_

#include <clang/AST/Attr.h>
#include <clang/AST/Type.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Ownership.h>
#include <llvm/ADT/SmallVector.h>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/Analysis/HostDataDeps.h"
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/Builtins.h"
#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/Vectorization/SIMDTypes.h"

#include <functional>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace clang {
namespace hipacc {

class ASTTranslateFusion;

class ASTFuse {
  private:
    ASTContext &Ctx;
    hipacc::Builtin::Context &builtins;
    CompilerOptions &compilerOptions;
    PrintingPolicy Policy;
    HostDataDeps *dataDeps;

    // fusible kernel lists
    SmallVector<std::list<HipaccKernel *>, 16> vecFusibleKernelLists;
    FunctionDecl *curFusedKernelDecl;
    SmallVector<Stmt *, 16> curFusedKernelBody;
    std::string fusedKernelName;
    std::string fusedFileName;
    std::map<std::string, HipaccKernel *> FuncDeclParamKernelMap;
    std::map<std::string, FieldDecl *> FuncDeclParamDeclMap;
    std::map<HipaccKernel *, std::string> fusedKernelNameMap;
    SmallVector<std::string, 16> fusedFileNamesAll;

    enum SubListPosition {
      Source,
      Intermediate,
      Destination,
      Undefined
    };
    struct FusionTypeTags {
      SubListPosition Point2PointLoc = Undefined;
      SubListPosition Local2PointLoc = Undefined;
      SubListPosition Point2LocalLoc = Undefined;
      SubListPosition Local2LocalLoc = Undefined;
    };
    std::map<HipaccKernel *, FusionTypeTags *> FusibleKernelSubListPosMap;

    // "global variables"
    unsigned fusionRegVarCount;
    SmallVector<VarDecl *, 16> fusionRegVarDecls;
    unsigned fusionIdxVarCount;
    SmallVector<VarDecl *, 16> fusionIdxVarDecls;

    // member functions
    void printFusedKernelFunction(std::list<HipaccKernel *>& l);
    void HipaccFusion(std::list<HipaccKernel *>& l);
    void initKernelFusion();
    FunctionDecl *createFusedKernelDecl(std::list<HipaccKernel *> &l);
    void insertPrologFusedKernel(); 
    void insertEpilogFusedKernel(); 
    void createReg4FusionVarDecl(QualType QT); 
    void createIdx4FusionVarDecl();
    void createGidVarDecl(); 
    void markKernelPositionSublist(std::list<HipaccKernel *> &l);

  public:
    ASTFuse(ASTContext& Ctx, hipacc::Builtin::Context &builtins, 
        CompilerOptions &options, PrintingPolicy Policy, HostDataDeps *dataDeps) : 
      Ctx(Ctx), 
      builtins(builtins), 
      compilerOptions(options), 
      Policy(Policy),
      dataDeps(dataDeps),
      curFusedKernelDecl(nullptr),
      fusedKernelName(""),
      fusedFileName(""),
      fusionRegVarCount(0),
      fusionIdxVarCount(0)
      {
        // prepare the fusible kernel lists
        for (unsigned i=0; i<dataDeps->getNumberOfFusibleKernelList(); ++i) {
          std::list<HipaccKernel *> list;
          vecFusibleKernelLists.push_back(list);
        }
      }

    // Called by Rewriter
    void parseFusibleKernel(HipaccKernel *K);
    SmallVector<std::string, 16> getFusedFileNamesAll() const; 
    bool isSrcKernel(HipaccKernel *K) const;
    bool isDestKernel(HipaccKernel *K) const;
    HipaccKernel *getProducerKernel(HipaccKernel *K);
    std::string getFusedKernelName(HipaccKernel *K);
};


class ASTTranslateFusion: public ASTTranslate {
  private:
    bool bSkipGidDecl;
    // point-to-point replacements
    bool bReplaceOutputExpr;
    Expr *exprOutputFusion;
    bool bReplaceInputExpr;
    Expr *exprInputFusion;
    // point-to-local replacements
    bool bReplaceInputIdxExpr;
    Expr *exprInputIdxFusion;
    bool bReplaceInputLocalExprs;
    Stmt *stmtProducerBodyP2L;
    std::string kernelParamNameSuffix;

    // onverload functions for ASTTranslate
    void initCUDA(SmallVector<Stmt *, 16> &kernelBody);
    Expr *VisitCXXMemberCallExprTranslate(CXXMemberCallExpr *E);
    Expr *VisitMemberExprTranslate(MemberExpr *E);
    Expr *VisitCXXOperatorCallExprTranslate(CXXOperatorCallExpr *E);
    void stageLineToSharedMemory(ParmVarDecl *PVD, SmallVector<Stmt *, 16>
        &stageBody, Expr *local_offset_x, Expr *local_offset_y, Expr
        *global_offset_x, Expr *global_offset_y);

  public:
    Stmt *Hipacc(Stmt *S);

  public:
    ASTTranslateFusion(ASTContext& Ctx, FunctionDecl *kernelDecl, HipaccKernel 
        *kernel, HipaccKernelClass *kernelClass, hipacc::Builtin::Context
        &builtins, CompilerOptions &options, bool emitEstimation=false) :
      ASTTranslate(Ctx, kernelDecl, kernel, kernelClass, builtins, options, 
          emitEstimation),
      bSkipGidDecl(true),
      bReplaceOutputExpr(false),
      exprOutputFusion(nullptr),
      bReplaceInputExpr(false),
      exprInputFusion(nullptr),
      bReplaceInputIdxExpr(false),
      exprInputIdxFusion(nullptr),
      bReplaceInputLocalExprs(false),
      kernelParamNameSuffix("_"+Kernel->getKernelName())
    {}

    // getters and setters
    void setSkipGidDecl(bool b) { bSkipGidDecl = b; } 

    // domain specific fusion methods
    void configSrcOperatorP2P(VarDecl *VD);
    void configDestOperatorP2P(VarDecl *VD);
    void configIntermOperatorP2P(VarDecl *VDIn, VarDecl *VDOut);
    void configSrcOperatorP2L(VarDecl *VDReg, VarDecl *VDIdx);
    void configDestOperatorP2L(VarDecl *VDReg, VarDecl *VDIdx, Stmt *S);
};

} // namespace hipacc
} // namespace clang

#endif  // _ASTFUSE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

