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

//===--- hipacc.cpp - CUDA/OpenCL Source-to-Source Compiler ---------------===//
//
// This file implements the CUDA/OpenCL Source-to-Source Compiler.
//
//===----------------------------------------------------------------------===//

#include "hipacc.h"

using namespace clang;
using namespace hipacc;


static void LLVMErrorHandler(void *userData, const std::string &message) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine *>(userData);

  Diags.Report(diag::err_fe_error_backend) << message;

  // we cannot recover from llvm errors.
  exit(EXIT_FAILURE);
}


llvm::sys::Path getExecutablePath(const char *argv0) {
  // this just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  return llvm::sys::Path::GetMainExecutable(argv0, mainAddr);
}


void printCopyright() {
  llvm::errs() << "\n"
    << "Copyright (c) 2012, University of Erlangen-Nuremberg\n"
    << "Copyright (c) 2012, Siemens AG\n"
    << "Copyright (c) 2010, ARM Limited\n"
    << "All rights reserved.\n\n";
}


void printUsage() {
  llvm::errs() << "OVERVIEW: HIPACC - Heterogeneous Image Processing Acceleration framework\n\n"
    << "USAGE:  hipacc [options] <input>\n\n"
    << "OPTIONS:\n\n"
    << "  -emit-cuda              Emit CUDA code; default is OpenCL code\n"
    << "  -emit-opencl-x86        Emit OpenCL code for x86 devices, no padding supported\n"
    << "  -emit-padding <n>       Emit CUDA/OpenCL image padding, using alignment of <n> bytes for GPU devices\n"
    << "  -compute-capability <n> Generate code for GPUs with compute capability <n>.\n"
    << "                          Valid values for CUDA/OpenCL on NVIDIA devices are 10, 11, 12, 13, 20, 21, 30, and 35\n"
    << "                          Valid values for OpenCL on AMD devices is 58 and 69\n"
    << "  -explore-config         Emit code that explores all possible kernel configuration and print its performance\n"
    << "  -time-kernels           Emit code that executes each kernel multiple times to get accurate timings\n"
    << "  -use-textures           Use textures (cached) in CUDA/OpenCL to read image pixels - for GPU devices only\n"
    << "  -use-local              Use shared/local memory in CUDA/OpenCL to stage image pixels to scratchpad\n"
    << "  -vectorize              Vectorize generated CUDA/OpenCL\n"
    << "  -pixels-per-thread <n>  Specify how many pixels should be calculated per thread\n"
    << "  -o <file>               Write output to <file>\n"
    << "  --help                  Display available options\n"
    << "  --version               Display version information\n";
}


void printVersion() {
  llvm::errs() << "This is hipacc version " << HIPACC_VERSION << "\n";
}


/// entry to our framework
int main(int argc, char *argv[]) {
  // first, print the Copyright notice
  printCopyright();

  // get stack trace on SigFaults
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);

  // argument list for CompilerInvocation after removing our compiler flags
  llvm::SmallVector<const char *, 16> Args;
  CompilerOptions compilerOptions = CompilerOptions();

  // support exceptions
  Args.push_back("-fexceptions");

  // parse command line options
  for (int i=1; i<argc; ++i) {
    if (llvm::StringRef(argv[i]) == "-emit-cuda") {
      compilerOptions.setTargetCode(TARGET_CUDA);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-emit-opencl-x86") {
      compilerOptions.setTargetCode(TARGET_OpenCLx86);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-emit-padding") {
      assert(i<(argc-1) && "Mandatory alignment parameter for -emit-padding switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "Expected alignment in bytes for -emit-padding switch.\n";
        exit(EXIT_FAILURE);
      }
      compilerOptions.setPadding(val);
      ++i;
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-compute-capability") {
      assert(i<(argc-1) && "Mandatory version parameter for -compute-capability switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "Expected version parameter for -compute-capability switch.\n";
        exit(EXIT_FAILURE);
      }
      compilerOptions.setTargetDevice((hipaccTargetDevice)val);
      ++i;
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-explore-config") {
      compilerOptions.setExploreConfig(USER_ON);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-time-kernels") {
      compilerOptions.setTimeKernels(USER_ON);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-use-textures") {
      compilerOptions.setTextureMemory(USER_ON);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-use-local") {
      compilerOptions.setLocalMemory(USER_ON);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-vectorize") {
      compilerOptions.setVectorizeKernels(USER_ON);
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-pixels-per-thread") {
      assert(i<(argc-1) && "Mandatory integer parameter for -pixels-per-thread switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "Expected integer parameter for -pixels-per-thread switch.\n";
        exit(EXIT_FAILURE);
      }
      compilerOptions.setPixelsPerThread(val);
      ++i;
      continue;
    }
    if (llvm::StringRef(argv[i]) == "-help" || llvm::StringRef(argv[i]) ==
        "--help") {
      printUsage();
      return EXIT_SUCCESS;
    }
    if (llvm::StringRef(argv[i]) == "-version" || llvm::StringRef(argv[i]) ==
        "--version") {
      printVersion();
      return EXIT_SUCCESS;
    }

    Args.push_back(argv[i]);
  }

  // sanity checks
  HipaccDevice targetDevice(compilerOptions);
  if (!targetDevice.isAMDGPU() && !targetDevice.isNVIDIAGPU()) {
    llvm::errs() << "Wrong compute capability specified: "
      << compilerOptions.getTargetDevice() << "\n"
      << "  Supported for NVIDIA devices are 10, 11, 12, 13, 20, 21, 30, and 35.\n"
      << "  Supported for AMD devices is 58 and 69.\n";
    exit(EXIT_FAILURE);
  }
  if (compilerOptions.useTextureMemory() && compilerOptions.emitOpenCLx86()) {
      compilerOptions.setTextureMemory(OFF);
      llvm::errs() << "Warning: texture support disabled! x86 devices do not support textures!\n";
  }
  if (compilerOptions.emitCUDA() && !targetDevice.isNVIDIAGPU()) {
    llvm::errs() << "CUDA code generation selected, but no CUDA-capable target device specified!\n"
      << "  Please select correct compute capability/code generation backend combination.\n";
    return EXIT_FAILURE;
  }

  // print summary of compiler options
  compilerOptions.printSummary(targetDevice.getTargetDeviceName());


  // setup and initialize compiler instance
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  llvm::sys::Path Path = getExecutablePath(argv[0]);

  llvm::OwningPtr<CompilerInstance> Clang(new CompilerInstance());
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, DiagsBuffer);

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

  // create the compilers actual diagnostics engine
  Clang->createDiagnostics(Args.size(), const_cast<char **>(Args.data()));
  if (!Clang->hasDiagnostics()) return EXIT_FAILURE;
  // print diagnostics in color
  Clang->getDiagnosticOpts().ShowColors = 1;
  // print statistics
  //Clang->getFrontendOpts().ShowStats = 1;

  // set an error handler, so that any LLVM backend diagnostics go through
  // our error handler
  llvm::install_fatal_error_handler(LLVMErrorHandler,
      static_cast<void *>(&Clang->getDiagnostics()));
  
  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!success) return EXIT_FAILURE;

  // create and execute the frontend action
  llvm::OwningPtr<ASTFrontendAction> Act(new HipaccRewriteAction(compilerOptions));

  if (!Clang->ExecuteAction(*Act)) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

// vim: set ts=2 sw=2 sts=2 et ai:

