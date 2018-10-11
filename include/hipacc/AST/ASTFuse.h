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
#include <queue>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace clang {
namespace hipacc {

class ASTFuse {
  private:
    ASTContext &Ctx;
    DiagnosticsEngine &Diags;
    hipacc::Builtin::Context &builtins;
    CompilerOptions &compilerOptions;
    HipaccDevice targetDevice;
    PrintingPolicy Policy;
    HostDataDeps *dataDeps;

    // variables for all fusible kernel lists
    SmallVector<std::list<HipaccKernel *>, 16> vecFusibleKernelLists;
    std::map<HipaccKernel *, std::string> fusedKernelNameMap;
    std::map<HipaccKernel *, std::tuple<unsigned, unsigned>> fusedLocalKernelMemorySizeMap;
    SmallVector<std::string, 16> fusedFileNamesAll;

    // variables per fusible kernel lists
    FunctionDecl *curFusedKernelDecl;
    SmallVector<Stmt *, 16> curFusedKernelBody;
    std::string fusedKernelName;
    std::string fusedFileName;
    std::map<std::string, HipaccKernel *> FuncDeclParamKernelMap;
    std::map<std::string, FieldDecl *> FuncDeclParamDeclMap;
    std::map<HipaccKernel *, std::tuple<unsigned, unsigned>> localKernelMemorySizeMap;
    std::tuple<unsigned, unsigned> localKernelMaxAccSizeUpdated;
    unsigned fusionRegVarCount;
    SmallVector<VarDecl *, 16> fusionRegVarDecls;
    SmallVector<Stmt *, 16> fusionRegSharedStmts;
    unsigned fusionIdxVarCount;
    SmallVector<VarDecl *, 16> fusionIdxVarDecls;

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

    // new
    std::map<std::string, std::tuple<unsigned, unsigned, unsigned>> FusibleKernelBlockLocation;
    std::set<std::vector<std::list<std::string>>> fusibleSetNames;
    std::vector<std::vector<std::list<HipaccKernel*>>> fusibleKernelSet;

    // member functions
    void setFusedKernelConfiguration(std::list<HipaccKernel *>& l);
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
    void recomputeMemorySizeLocalFusion(std::list<HipaccKernel *> &l);

  public:
    ASTFuse(ASTContext& Ctx, DiagnosticsEngine &Diags, hipacc::Builtin::Context &builtins,
        CompilerOptions &options, PrintingPolicy Policy, HostDataDeps *dataDeps) :
      Ctx(Ctx),
      Diags(Diags),
      builtins(builtins),
      compilerOptions(options),
      targetDevice(options),
      Policy(Policy),
      dataDeps(dataDeps),
      curFusedKernelDecl(nullptr),
      fusedKernelName(""),
      fusedFileName(""),
      fusionRegVarCount(0),
      fusionIdxVarCount(0)
      {
        fusibleSetNames = dataDeps->getFusibleSetNames();
        unsigned blockCnt, vectorCnt, listCnt;
        blockCnt = 0;
        for (auto PBN : fusibleSetNames) { // block level
          vectorCnt = 0;
          std::vector<std::list<HipaccKernel*>> kv;
          for (auto sL : PBN) {              // vector level 
            listCnt = 0;
            std::list<HipaccKernel*> kl;
            for (auto nam : sL) {              // list level
              auto pos = std::make_tuple(blockCnt, vectorCnt, listCnt);
              FusibleKernelBlockLocation[nam] = pos;
              listCnt++;
            }
            kv.push_back(kl);
            vectorCnt++;
          }
          fusibleKernelSet.push_back(kv); 
          blockCnt++;
        }
      }

    // Called by Rewriter
    bool parseFusibleKernel(HipaccKernel *K);
    SmallVector<std::string, 16> getFusedFileNamesAll() const;
    bool isSrcKernel(HipaccKernel *K);
    bool isDestKernel(HipaccKernel *K);
    HipaccKernel *getProducerKernel(HipaccKernel *K);
    std::string getFusedKernelName(HipaccKernel *K);
    unsigned getNewYSizeLocalKernel(HipaccKernel *K);
};

} // namespace hipacc
} // namespace clang

#endif  // _ASTFUSE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

