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
    << "Copyright (c) 2012, University of Erlangen-Nuremberg\n"
    << "Copyright (c) 2012, Siemens AG\n"
    << "Copyright (c) 2010, ARM Limited\n"
    << "All rights reserved.\n\n";
}


void printUsage() {
  llvm::errs() << "OVERVIEW: HIPAcc - Heterogeneous Image Processing Acceleration framework\n\n"
    << "USAGE:  hipacc [options] <input>\n\n"
    << "OPTIONS:\n\n"
    << "  -emit-cuda              Emit CUDA code; default is OpenCL code\n"
    << "  -emit-opencl-acc        Emit OpenCL code for Accelerator devices\n"
    << "  -emit-opencl-cpu        Emit OpenCL code for CPU devices\n"
    << "  -emit-opencl-gpu        Emit OpenCL code for GPU devices\n"
    << "  -emit-renderscript      Emit Renderscript code for Android\n"
    << "  -emit-filterscript      Emit Filterscript code for Android\n"
    << "  -emit-padding <n>       Emit CUDA/OpenCL/Renderscript image padding, using alignment of <n> bytes for GPU devices\n"
    << "  -target <n>             Generate code for GPUs with code name <n>.\n"
    << "                          Code names for CUDA/OpenCL on NVIDIA devices are:\n"
    << "                            'Tesla-10', 'Tesla-11', 'Tesla-12', and 'Tesla-13' for Tesla architecture.\n"
    << "                            'Fermi-20' and 'Fermi-21' for Fermi architecture.\n"
    << "                            'Kepler-30' and 'Kepler-35' for Kepler architecture.\n"
    << "                          Code names for for OpenCL on AMD devices are:\n"
    << "                            'Evergreen'      for Evergreen architecture (Radeon HD5xxx).\n"
    << "                            'NorthernIsland' for Northern Island architecture (Radeon HD6xxx).\n"
    << "                          Code names for for OpenCL on ARM devices are:\n"
    << "                            'Midgard' for Mali-T6xx' for Mali.\n"
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

  // support exceptions
  Args.push_back("-fexceptions");

  // parse command line options
  for (int i=1; i<argc; ++i) {
    if (StringRef(argv[i]) == "-emit-cuda") {
      compilerOptions.setTargetCode(TARGET_CUDA);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-acc") {
      compilerOptions.setTargetCode(TARGET_OpenCLACC);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-cpu") {
      compilerOptions.setTargetCode(TARGET_OpenCLCPU);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-gpu") {
      compilerOptions.setTargetCode(TARGET_OpenCLGPU);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-renderscript") {
      compilerOptions.setTargetCode(TARGET_Renderscript);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-filterscript") {
      compilerOptions.setTargetCode(TARGET_Filterscript);
      compilerOptions.setPixelsPerThread(1);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-padding") {
      assert(i<(argc-1) && "Mandatory alignment parameter for -emit-padding switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "ERROR: Expected alignment in bytes for -emit-padding switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setPadding(val);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-target") {
      assert(i<(argc-1) && "Mandatory code name parameter for -target switch missing.");
      if (StringRef(argv[i+1]) == "Tesla-10") {
        compilerOptions.setTargetDevice(TESLA_10);
      } else if (StringRef(argv[i+1]) == "Tesla-11") {
        compilerOptions.setTargetDevice(TESLA_11);
      } else if (StringRef(argv[i+1]) == "Tesla-12") {
        compilerOptions.setTargetDevice(TESLA_12);
      } else if (StringRef(argv[i+1]) == "Tesla-13") {
        compilerOptions.setTargetDevice(TESLA_13);
      } else if (StringRef(argv[i+1]) == "Fermi-20") {
        compilerOptions.setTargetDevice(FERMI_20);
      } else if (StringRef(argv[i+1]) == "Fermi-21") {
        compilerOptions.setTargetDevice(FERMI_21);
      } else if (StringRef(argv[i+1]) == "Kepler-30") {
        compilerOptions.setTargetDevice(KEPLER_30);
      } else if (StringRef(argv[i+1]) == "Kepler-35") {
        compilerOptions.setTargetDevice(KEPLER_35);
      } else if (StringRef(argv[i+1]) == "Evergreen") {
        compilerOptions.setTargetDevice(EVERGREEN);
      } else if (StringRef(argv[i+1]) == "NorthernIsland") {
        compilerOptions.setTargetDevice(NORTHERN_ISLAND);
      } else if (StringRef(argv[i+1]) == "Midgard") {
        compilerOptions.setTargetDevice(MIDGARD);
      } else {
        llvm::errs() << "ERROR: Expected valid code name specification for -target switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-explore-config") {
      compilerOptions.setExploreConfig(USER_ON);
      continue;
    }
    if (StringRef(argv[i]) == "-use-config") {
      assert(i<(argc-1) && "Mandatory configuration specification for -use-config switch missing.");
      int x=0, y=0, ret=0;
      ret = sscanf(argv[i+1], "%dx%d", &x, &y);
      if (ret!=2) {
        llvm::errs() << "ERROR: Expected valid configuration specification for -use-config switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setKernelConfig(x, y);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-time-kernels") {
      compilerOptions.setTimeKernels(USER_ON);
      continue;
    }
    if (StringRef(argv[i]) == "-use-textures") {
      assert(i<(argc-1) && "Mandatory texture memory specification for -use-textures switch missing.");
      if (StringRef(argv[i+1]) == "off") {
        compilerOptions.setTextureMemory(NoTexture);
      } else if (StringRef(argv[i+1]) == "Linear1D") {
        compilerOptions.setTextureMemory(Linear1D);
      } else if (StringRef(argv[i+1]) == "Linear2D") {
        compilerOptions.setTextureMemory(Linear2D);
      } else if (StringRef(argv[i+1]) == "Array2D") {
        compilerOptions.setTextureMemory(Array2D);
      } else if (StringRef(argv[i+1]) == "Ldg") {
        compilerOptions.setTextureMemory(Ldg);
      } else {
        llvm::errs() << "ERROR: Expected valid texture memory specification for -use-textures switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-use-local") {
      assert(i<(argc-1) && "Mandatory local memory specification for -use-local switch missing.");
      if (StringRef(argv[i+1]) == "off") {
        compilerOptions.setLocalMemory(USER_OFF);
      } else if (StringRef(argv[i+1]) == "on") {
        compilerOptions.setLocalMemory(USER_ON);
      } else {
        llvm::errs() << "ERROR: Expected valid local memory specification for -use-local switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-vectorize") {
      assert(i<(argc-1) && "Mandatory vectorization specification for -vectorize switch missing.");
      if (StringRef(argv[i+1]) == "off") {
        compilerOptions.setVectorizeKernels(USER_OFF);
      } else if (StringRef(argv[i+1]) == "on") {
        compilerOptions.setVectorizeKernels(USER_ON);
      } else {
        llvm::errs() << "ERROR: Expected valid vectorization specification for -use-vectorize switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-pixels-per-thread") {
      assert(i<(argc-1) && "Mandatory integer parameter for -pixels-per-thread switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "ERROR: Expected integer parameter for -pixels-per-thread switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setPixelsPerThread(val);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-rs-package") {
      assert(i<(argc-1) && "Mandatory package name string for -rs-package switch missing.");
      compilerOptions.setRSPackageName(argv[i+1]);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-help" || StringRef(argv[i]) == "--help") {
      printUsage();
      return EXIT_SUCCESS;
    }
    if (StringRef(argv[i]) == "-version" || StringRef(argv[i]) == "--version") {
      printVersion();
      return EXIT_SUCCESS;
    }

    Args.push_back(argv[i]);
  }

  // create target device description from compiler options
  HipaccDevice targetDevice(compilerOptions);

  //
  // sanity checks
  //

  // CUDA supported only on NVIDIA devices
  if (compilerOptions.emitCUDA() && !targetDevice.isNVIDIAGPU()) {
    llvm::errs() << "ERROR: CUDA code generation selected, but no CUDA-capable target device specified!\n"
                 << "  Please select correct target device/code generation backend combination.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  // OpenCL (GPU) only supported on GPU devices
  if (compilerOptions.emitOpenCLGPU() &&
      !(targetDevice.isAMDGPU() || targetDevice.isARMGPU() ||
        targetDevice.isNVIDIAGPU())) {
    llvm::errs() << "ERROR: OpenCL (GPU) code generation selected, but no OpenCL-capable target device specified!\n"
                 << "  Please select correct target device/code generation backend combination.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  // Textures in CUDA - writing to Array2D textures introduced with Fermi
  if (compilerOptions.emitCUDA() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()==Array2D &&
        compilerOptions.getTargetDevice() < FERMI_20) {
      llvm::errs() << "Warning: 'Array2D' texture memory only supported for Fermi and later on (CC >= 2.0)!"
                   << "  Using 'Linear2D' instead!\n";
      compilerOptions.setTextureMemory(Linear2D);
    }
  }
  // Textures in CUDA - Ldg (load via texture cache) was introduced with Kepler
  if (compilerOptions.emitCUDA() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()==Ldg &&
        compilerOptions.getTargetDevice() < KEPLER_35) {
      llvm::errs() << "Warning: 'Ldg' texture memory only supported for Kepler and later on (CC >= 3.5)!"
                   << "  Using 'Linear1D' instead!\n";
      compilerOptions.setTextureMemory(Linear1D);
    }
  }
  // Textures in OpenCL - only supported on some CPU platforms
  if (compilerOptions.emitOpenCLCPU() && compilerOptions.useTextureMemory(USER_ON)) {
      llvm::errs() << "\nWarning: image support is only available on some CPU devices!\n\n";
  }
  // Textures in OpenCL - only Array2D textures supported
  if (compilerOptions.emitOpenCLGPU() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()!=Array2D) {
      llvm::errs() << "Warning: 'Linear1D', 'Linear2D', and 'Ldg' texture memory not supported by OpenCL!"
                   << "  Using 'Array2D' instead!\n";
      compilerOptions.setTextureMemory(Array2D);
    }
  }
  // Invalid specification for kernel configuration
  if (compilerOptions.useKernelConfig(USER_ON)) {
    if (compilerOptions.getKernelConfigX()*compilerOptions.getKernelConfigY() >
        (int)targetDevice.max_threads_per_block) {
      llvm::errs() << "ERROR: Invalid kernel configuration: maximum threads for target device are "
                   << targetDevice.max_threads_per_block << "!\n\n";
      printUsage();
      return EXIT_FAILURE;
    }
  }
  // Pixels per thread > 1 not supported on Filterscript
  if (compilerOptions.emitFilterscript() &&
      compilerOptions.getPixelsPerThread() > 1) {
    llvm::errs() << "ERROR: Calculating multiple pixels per thread selected, which is not supported for Filterscript!\n"
                 << "  Please disable multiple pixels per thread oder switch target code generation backend.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  if (compilerOptions.timeKernels(USER_ON) &&
      compilerOptions.exploreConfig(USER_ON)) {
    // kernels are timed internally by the runtime in case of exploration
    compilerOptions.setTimeKernels(OFF);
  }

  // print summary of compiler options
  compilerOptions.printSummary(targetDevice.getTargetDeviceName());


  // setup and initialize compiler instance
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  std::string Path = getExecutablePath(argv[0]);

  OwningPtr<CompilerInstance> Clang(new CompilerInstance());
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

  // create the compilers actual diagnostics engine
  Clang->createDiagnostics();
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
  OwningPtr<ASTFrontendAction> Act(new HipaccRewriteAction(compilerOptions));

  if (!Clang->ExecuteAction(*Act)) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

// vim: set ts=2 sw=2 sts=2 et ai:

