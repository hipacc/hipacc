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

//===--- Polly.cpp - Polyhedral Analysis for Kernels using Polly ----------===//
//
// This file implements the interface to Polly for kernel transformations like
// loop fusion.
//
//===----------------------------------------------------------------------===//

#include <polly/LinkAllPasses.h>
#include <polly/ScopDetection.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/PassManager.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Target/TargetLibraryInfo.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>

#include "hipacc/Analysis/Polly.h"

namespace polly {
void initializePollyPasses(llvm::PassRegistry &Registry);
}

using namespace clang;
using namespace hipacc;


void Polly::analyzeKernel() {
  // enable statistics for LLVM passes
  llvm::EnableStatistics();

  // code generation for Polly
  llvm::LLVMContext *LLVMCtx = new llvm::LLVMContext;
  clang::CodeGenerator *llvm_ir_cg =
    clang::CreateLLVMCodeGen(Clang.getDiagnostics(), func->getNameAsString(),
        Clang.getCodeGenOpts(), Clang.getTargetOpts(), *LLVMCtx);

  DeclGroupRef DG = DeclGroupRef(func);
  llvm_ir_cg->Initialize(Ctx);
  llvm_ir_cg->HandleTopLevelDecl(DG);

  // get module
  llvm::Module *irModule = llvm_ir_cg->GetModule();
  assert(irModule && "Module is unavailable");
  llvm_ir_cg->ReleaseModule();
  delete llvm_ir_cg;

  // initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeIPA(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

  polly::initializePollyPasses(Registry);

  // create a PassManager to hold and optimize the collection of passes we are
  // about to build
  llvm::PassManager Passes;

  // add an appropriate TargetLibraryInfo pass for the module's triple
  Passes.add(new llvm::TargetLibraryInfo(Triple(irModule->getTargetTriple())));

  // add an appropriate DataLayout instance for this module.
  if (irModule->getDataLayout())
    Passes.add(new DataLayoutPass());

  Passes.add(llvm::createPromoteMemoryToRegisterPass());
  Passes.add(llvm::createFunctionInliningPass());
  Passes.add(llvm::createCFGSimplificationPass());
  Passes.add(llvm::createInstructionCombiningPass());
  Passes.add(llvm::createTailCallEliminationPass());
  Passes.add(llvm::createLoopSimplifyPass());
  Passes.add(llvm::createLCSSAPass());
  Passes.add(llvm::createLoopRotatePass());
  Passes.add(llvm::createLCSSAPass());
  Passes.add(llvm::createLoopUnswitchPass());
  Passes.add(llvm::createInstructionCombiningPass());
  Passes.add(llvm::createLoopSimplifyPass());
  Passes.add(llvm::createLCSSAPass());
  Passes.add(llvm::createIndVarSimplifyPass());
  Passes.add(llvm::createLoopDeletionPass());
  Passes.add(llvm::createInstructionCombiningPass());
  Passes.add(llvm::createBasicAliasAnalysisPass());

  Passes.add(polly::createCodePreparationPass());
  Passes.add(polly::createScopInfoPass());
  Passes.add(new polly::ScopDetection());
  Passes.add(polly::createJSONExporterPass());

  // run optimization passes
  Passes.run(*irModule);

  // print Stats
  llvm::PrintStatistics();
}

// vim: set ts=2 sw=2 sts=2 et ai:

