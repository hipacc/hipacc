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

#include <clang/CodeGen/ModuleBuilder.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <polly/Canonicalization.h>
#include <polly/RegisterPasses.h>

#include "hipacc/Analysis/Polly.h"

using namespace clang;
using namespace hipacc;

void Polly::analyzeKernel() {
  // enable statistics for LLVM passes
  llvm::EnableStatistics();

  // code generation for Polly
  llvm::LLVMContext llvm_context;
  CodeGenerator *llvm_ir_cg = CreateLLVMCodeGen(Clang.getDiagnostics(),
      func->getNameAsString(), Clang.getHeaderSearchOpts(),
      Clang.getPreprocessorOpts(), Clang.getCodeGenOpts(), llvm_context);

  DeclGroupRef DG = DeclGroupRef(func);
  llvm_ir_cg->Initialize(Ctx);
  llvm_ir_cg->HandleTopLevelDecl(DG);

  // get module
  llvm::Module *ir_module = llvm_ir_cg->GetModule();
  assert(ir_module && "Module is unavailable");
  llvm_ir_cg->ReleaseModule();
  delete llvm_ir_cg;

  // initialize passes
  llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
  polly::initializePollyPasses(Registry);

  // run optimization passes
  llvm::legacy::PassManager Passes;
  polly::registerCanonicalicationPasses(Passes);
  polly::registerPollyPasses(Passes);
  Passes.run(*ir_module);

  // print stats
  llvm::PrintStatistics();
}

// vim: set ts=2 sw=2 sts=2 et ai:

