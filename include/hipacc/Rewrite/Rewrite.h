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

//===--- Rewrite.h - OpenCL/CUDA rewriter for the AST ---------------------===//
//
// This file implements functionality for rewriting OpenCL/CUDA kernels.
//
//===----------------------------------------------------------------------===//

#ifndef _REWRITE_H_
#define _REWRITE_H_

#include <clang/Analysis/CFG.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Casting.h>

#include "hipacc/Config/config.h"
#include "hipacc/Analysis/KernelStatistics.h"
#ifdef USE_POLLY
#include "hipacc/Analysis/Polly.h"
#endif
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/DSL/CompilerKnownClasses.h"
#include "hipacc/Rewrite/CreateHostStrings.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <unistd.h>

namespace clang {
namespace hipacc {
class HipaccRewriteAction : public ASTFrontendAction {
  CompilerOptions &options;

  public:
  HipaccRewriteAction(CompilerOptions &options) :
    options(options)
  {}

  protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI, StringRef
      InFile);
};
ASTConsumer *CreateHipaccRewriteAction(CompilerInstance &CI, CompilerOptions
    &options, llvm::raw_ostream *out);
} // end namespace hipacc
} // end namespace clang

#endif  // _REWRITE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

