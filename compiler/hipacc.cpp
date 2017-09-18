//
// Copyright (c) 2014, Saarland University
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

//===--- hipacc.cpp - DSL Source-to-Source Compiler -----------------------===//
//
// This file implements the DSL Source-to-Source Compiler.
//
//===----------------------------------------------------------------------===//

#include "hipacc.h"
#include "hipacc/Backend/BackendConfigurationManager.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/Rewrite/Rewrite.h"

#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Signals.h>

#include <sstream>
#include <memory>

using namespace clang;
using namespace hipacc;


static void LLVMErrorHandler(void *userData, const std::string &message, bool
    genCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine *>(userData);

  Diags.Report(diag::err_fe_error_backend) << message;

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  exit(genCrashDiag ? 70 : EXIT_FAILURE);
}


std::string getExecutablePath(const char *argv0) {
  // this just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, mainAddr);
}


void printCopyright() {
  llvm::errs() << "\n"
    << "Copyright (c) 2014, Saarland University\n"
    << "Copyright (c) 2012, University of Erlangen-Nuremberg\n"
    << "Copyright (c) 2012, Siemens AG\n"
    << "Copyright (c) 2010, ARM Limited\n"
    << "All rights reserved.\n\n";
}


void printUsage() {
  llvm::errs() << "OVERVIEW: Hipacc - Heterogeneous Image Processing Acceleration framework\n\n"
    << "USAGE:  hipacc [options] <input>\n\n"
    << "OPTIONS:\n\n"
    << "  -emit-cpu               Emit C++ code\n"
    << "  -emit-cuda              Emit CUDA code for GPU devices\n"
    << "  -emit-opencl-acc        Emit OpenCL code for Accelerator devices\n"
    << "  -emit-opencl-cpu        Emit OpenCL code for CPU devices\n"
    << "  -emit-opencl-gpu        Emit OpenCL code for GPU devices\n"
    << "  -emit-renderscript      Emit Renderscript code for Android\n"
    << "  -emit-filterscript      Emit Filterscript code for Android\n"
    << "  -emit-padding <n>       Emit CUDA/OpenCL/Renderscript image padding, using alignment of <n> bytes for GPU devices\n"
    << "  -target <n>             Generate code for GPUs with code name <n>.\n"
    << "                          Code names for CUDA/OpenCL on NVIDIA devices are:\n"
    << "                            'Fermi-20' and 'Fermi-21' for Fermi architecture.\n"
    << "                            'Kepler-30', 'Kepler-32', 'Kepler-35', and 'Kepler-37' for Kepler architecture.\n"
    << "                            'Maxwell-50', 'Maxwell-52', and 'Maxwell-53' for Maxwell architecture.\n"
    << "                          Code names for for OpenCL on AMD devices are:\n"
    << "                            'Evergreen'      for Evergreen architecture (Radeon HD5xxx).\n"
    << "                            'NorthernIsland' for Northern Island architecture (Radeon HD6xxx).\n"
    << "                          Code names for for OpenCL on ARM devices are:\n"
    << "                            'Midgard' for Mali-T6xx' for Mali.\n"
    << "                          Code names for for OpenCL on Intel Xeon Phi devices are:\n"
    << "                            'KnightsCorner' for Knights Corner Many Integrated Cores architecture.\n"
    << "  -explore-config         Emit code that explores all possible kernel configuration and print its performance\n"
    << "  -use-config <nxm>       Emit code that uses a configuration of nxm threads, e.g. 128x1\n"
    << "  -time-kernels           Emit code that executes each kernel multiple times to get accurate timings\n"
    << "  -use-textures <o>       Enable/disable usage of textures (cached) in CUDA/OpenCL to read/write image pixels - for GPU devices only\n"
    << "                          Valid values for CUDA on NVIDIA devices: 'off', 'Linear1D', 'Linear2D', 'Array2D', and 'Ldg'\n"
    << "                          Valid values for OpenCL: 'off' and 'Array2D'\n"
    << "  -use-local <o>          Enable/disable usage of shared/local memory in CUDA/OpenCL to stage image pixels to scratchpad\n"
    << "                          Valid values: 'on' and 'off'\n"
    << "  -vectorize <o>          Enable/disable vectorization of generated CUDA/OpenCL code\n"
    << "                          Valid values: 'on' and 'off'\n"
    << "  -pixels-per-thread <n>  Specify how many pixels should be calculated per thread\n"
    << "  -rs-package <string>    Specify Renderscript package name. (default: \"org.hipacc.rs\")\n"
    << "  -o <file>               Write output to <file>\n"
    << "  --help                  Display available options\n"
    << "  --version               Display version information\n";
}


void printVersion() {
  llvm::errs() << "hipacc version " << HIPACC_VERSION
    << " (" << GIT_REPOSITORY " " << GIT_VERSION << ")\n";
}


/// entry to our framework
int main(int argc, char *argv[]) {
  // first, print the Copyright notice
  printCopyright();

  // get stack trace on SegFaults
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);

  // argument list for CompilerInvocation after removing our compiler flags
  SmallVector<const char *, 16> Args;
  CompilerOptions compilerOptions = CompilerOptions();
  std::string out;

  // Initialize the backends
  Backend::BackendConfigurationManager BackendConfigManager(&compilerOptions);
  try
  {
    // Convert the command line arguments into the backend argument vector type
    Backend::CommonDefines::ArgumentVectorType vecArguments;
    for (int i = 1; i < argc; ++i)
    {
      vecArguments.push_back(argv[i]);
    }

    // Let the backend configuration manager parse the command line arguments
    BackendConfigManager.Configure(vecArguments);
    out = BackendConfigManager.GetOutputFile();

    // Fetch the commands vector the clang invocation and convert it to clang's format
    vecArguments = BackendConfigManager.GetClangArguments();
    for (auto itArgument : vecArguments)
    {
      char *pcArgument = (char*) calloc(itArgument.size() + 1, sizeof(char));
      if (pcArgument == NULL)
      {
        throw std::runtime_error("Cannot allocate memory for clang command argument!");
      }

      strcpy(pcArgument, itArgument.c_str());

      Args.push_back(pcArgument);
    }
  }
  catch (std::exception &e)
  {
    llvm::errs() << "ERROR: " << e.what();
    return EXIT_FAILURE;
  }


  // create target device description from compiler options
  HipaccDevice targetDevice(compilerOptions);


  // Sanity check - Invalid specification for kernel configuration
  if (compilerOptions.useKernelConfig(USER_ON)) {
    if (compilerOptions.getKernelConfigX()*compilerOptions.getKernelConfigY() >
        (int)targetDevice.max_threads_per_block) {
      llvm::errs() << "ERROR: Invalid kernel configuration: maximum threads for target device are "
                   << targetDevice.max_threads_per_block << "!\n\n";
      printUsage();
      return EXIT_FAILURE;
    }
  }

  // print summary of compiler options
  compilerOptions.printSummary(targetDevice.getTargetDeviceName());


  // setup and initialize compiler instance
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  std::string Path = getExecutablePath(argv[0]);

  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // initialize a compiler invocation object from the arguments
  bool success;
  success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
      const_cast<const char **>(Args.data()),
      const_cast<const char **>(Args.data()) + Args.size(), Diags);

  // infer the builtin include path if unspecified
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty()) {
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(argv[0], mainAddr);
  }

  // create the actual diagnostics engine
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics()) return EXIT_FAILURE;
  // print diagnostics in color
  Clang->getDiagnosticOpts().ShowColors = 1;
  // print statistics
  //Clang->getFrontendOpts().ShowStats = 1;

  // set an error handler, so that any LLVM back end diagnostics go through
  // our error handler
  llvm::install_fatal_error_handler(LLVMErrorHandler,
      static_cast<void *>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!success) return EXIT_FAILURE;

  // create and execute the frontend action
  std::unique_ptr<ASTFrontendAction> Act(new HipaccRewriteAction(compilerOptions, out));

  if (!Clang->ExecuteAction(*Act)) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

// vim: set ts=2 sw=2 sts=2 et ai:

