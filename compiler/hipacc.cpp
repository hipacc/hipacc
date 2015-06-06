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
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/Rewrite/Rewrite.h"

#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/Host.h>

#include <sstream>

using namespace clang;
using namespace hipacc;


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
    << "                            'Tesla-10', 'Tesla-11', 'Tesla-12', and 'Tesla-13' for Tesla architecture.\n"
    << "                            'Fermi-20' and 'Fermi-21' for Fermi architecture.\n"
    << "                            'Kepler-30' and 'Kepler-35' for Kepler architecture.\n"
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

  // argument list for Driver after removing our compiler flags
  SmallVector<const char *, 16> args;
  CompilerOptions compilerOptions = CompilerOptions();
  std::string out;

  // parse command line options
  for (int i=0; i<argc; ++i) {
    if (StringRef(argv[i]) == "-emit-cpu") {
      compilerOptions.setTargetLang(Language::C99);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-cuda") {
      compilerOptions.setTargetLang(Language::CUDA);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-acc") {
      compilerOptions.setTargetLang(Language::OpenCLACC);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-cpu") {
      compilerOptions.setTargetLang(Language::OpenCLCPU);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-opencl-gpu") {
      compilerOptions.setTargetLang(Language::OpenCLGPU);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-renderscript") {
      compilerOptions.setTargetLang(Language::Renderscript);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-filterscript") {
      compilerOptions.setTargetLang(Language::Filterscript);
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
        compilerOptions.setTargetDevice(Device::Tesla_10);
      } else if (StringRef(argv[i+1]) == "Tesla-11") {
        compilerOptions.setTargetDevice(Device::Tesla_11);
      } else if (StringRef(argv[i+1]) == "Tesla-12") {
        compilerOptions.setTargetDevice(Device::Tesla_12);
      } else if (StringRef(argv[i+1]) == "Tesla-13") {
        compilerOptions.setTargetDevice(Device::Tesla_13);
      } else if (StringRef(argv[i+1]) == "Fermi-20") {
        compilerOptions.setTargetDevice(Device::Fermi_20);
      } else if (StringRef(argv[i+1]) == "Fermi-21") {
        compilerOptions.setTargetDevice(Device::Fermi_21);
      } else if (StringRef(argv[i+1]) == "Kepler-30") {
        compilerOptions.setTargetDevice(Device::Kepler_30);
      } else if (StringRef(argv[i+1]) == "Kepler-35") {
        compilerOptions.setTargetDevice(Device::Kepler_35);
      } else if (StringRef(argv[i+1]) == "Evergreen") {
        compilerOptions.setTargetDevice(Device::Evergreen);
      } else if (StringRef(argv[i+1]) == "NorthernIsland") {
        compilerOptions.setTargetDevice(Device::NorthernIsland);
      } else if (StringRef(argv[i+1]) == "Midgard") {
        compilerOptions.setTargetDevice(Device::Midgard);
      } else if (StringRef(argv[i+1]) == "KnightsCorner") {
        compilerOptions.setTargetDevice(Device::KnightsCorner);
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
        compilerOptions.setTextureMemory(Texture::None);
      } else if (StringRef(argv[i+1]) == "Linear1D") {
        compilerOptions.setTextureMemory(Texture::Linear1D);
      } else if (StringRef(argv[i+1]) == "Linear2D") {
        compilerOptions.setTextureMemory(Texture::Linear2D);
      } else if (StringRef(argv[i+1]) == "Array2D") {
        compilerOptions.setTextureMemory(Texture::Array2D);
      } else if (StringRef(argv[i+1]) == "Ldg") {
        compilerOptions.setTextureMemory(Texture::Ldg);
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
    if (StringRef(argv[i]) == "-o") {
      assert(i<(argc-1) && "Mandatory output file name for -o switch missing.");
      args.push_back(argv[i]);
      args.push_back(argv[++i]);
      out = argv[i];
      continue;
    }

    args.push_back(argv[i]);
  }

  // create target device description from compiler options
  HipaccDevice targetDevice(compilerOptions);

  //
  // sanity checks
  //

  // CUDA supported only on NVIDIA devices
  if (compilerOptions.emitCUDA() && !targetDevice.isNVIDIAGPU()) {
    llvm::errs() << "ERROR: CUDA code generation selected, but no CUDA-capable target device specified!\n"
                 << "  Please select correct target device/code generation back end combination.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  // OpenCL (GPU) only supported on GPU devices
  if (compilerOptions.emitOpenCLGPU() &&
      !(targetDevice.isAMDGPU() || targetDevice.isARMGPU() ||
        targetDevice.isNVIDIAGPU())) {
    llvm::errs() << "ERROR: OpenCL (GPU) code generation selected, but no OpenCL-capable GPU target device specified!\n"
                 << "  Please select correct target device/code generation back end combination.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  // OpenCL (ACC) only supported on accelerator devices
  if (compilerOptions.emitOpenCLACC() && !targetDevice.isINTELACC()) {
    llvm::errs() << "ERROR: OpenCL (ACC) code generation selected, but no OpenCL-capable accelerator device specified!\n"
                 << "  Please select correct target device/code generation back end combination.\n\n";
    printUsage();
    return EXIT_FAILURE;
  }
  // Textures in CUDA - writing to Array2D textures introduced with Fermi
  if (compilerOptions.emitCUDA() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()==Texture::Array2D &&
        compilerOptions.getTargetDevice() < Device::Fermi_20) {
      llvm::errs() << "Warning: 'Array2D' texture memory only supported for Fermi and later on (CC >= 2.0)!"
                   << "  Using 'Linear2D' instead!\n";
      compilerOptions.setTextureMemory(Texture::Linear2D);
    }
  }
  // Textures in CUDA - Ldg (load via texture cache) was introduced with Kepler
  if (compilerOptions.emitCUDA() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()==Texture::Ldg &&
        compilerOptions.getTargetDevice() < Device::Kepler_35) {
      llvm::errs() << "Warning: 'Ldg' texture memory only supported for Kepler and later on (CC >= 3.5)!"
                   << "  Using 'Linear1D' instead!\n";
      compilerOptions.setTextureMemory(Texture::Linear1D);
    }
  }
  // Textures in OpenCL - only supported on some CPU platforms
  if (compilerOptions.emitOpenCLCPU() && compilerOptions.useTextureMemory(USER_ON)) {
      llvm::errs() << "Warning: image support is only available on some CPU devices!\n";
  }
  // Textures in OpenCL - only supported on some CPU platforms
  if (compilerOptions.emitOpenCLACC() && compilerOptions.useTextureMemory(USER_ON)) {
      llvm::errs() << "ERROR: image support is not available on ACC devices!\n\n";
      printUsage();
      return EXIT_FAILURE;
  }
  // Textures in OpenCL - only Array2D textures supported
  if (compilerOptions.emitOpenCLGPU() && compilerOptions.useTextureMemory(USER_ON)) {
    if (compilerOptions.getTextureType()!=Texture::Array2D) {
      llvm::errs() << "Warning: 'Linear1D', 'Linear2D', and 'Ldg' texture memory not supported by OpenCL!\n"
                   << "  Using 'Array2D' instead!\n";
      compilerOptions.setTextureMemory(Texture::Array2D);
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
    llvm::errs() << "Warning: computing multiple pixels per thread is not supported by Filterscript!\n"
                 << "  Computing only a single pixel per thread instead!\n";
    compilerOptions.setPixelsPerThread(1);
  }
  // No scratchpad memory support in Renderscript/Filterscript
  if ((compilerOptions.emitFilterscript() ||compilerOptions.emitRenderscript())
      && compilerOptions.useLocalMemory(USER_ON)) {
    llvm::errs() << "Warning: local memory support is not available in Renderscript and Filterscript!\n"
                 << "  Local memory disabled!\n";
    compilerOptions.setLocalMemory(USER_OFF);
  }
  if (compilerOptions.timeKernels(USER_ON) &&
      compilerOptions.exploreConfig(USER_ON)) {
    // kernels are timed internally by the runtime in case of exploration
    compilerOptions.setTimeKernels(OFF);
  }

  // print summary of compiler options
  compilerOptions.printSummary(targetDevice.getTargetDeviceName());


  // use the Driver
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      &*DiagOpts, &DiagnosticPrinter, false);

  driver::Driver Driver(args[0], llvm::sys::getDefaultTargetTriple(),
      Diagnostics);
  Driver.setCheckInputsExist(false);
  Driver.setTitle("hipacc");

  std::unique_ptr<driver::Compilation> Compilation(
      Driver.BuildCompilation(args));

  // use the flags from the first job
  const driver::JobList &Jobs = Compilation->getJobs();
  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  const llvm::opt::ArgStringList *const cc1_args = &Cmd.getArguments();

  std::unique_ptr<CompilerInvocation> Invocation(new CompilerInvocation());
  CompilerInvocation::CreateFromArgs(*Invocation,
      cc1_args->data() + 1, cc1_args->data() + cc1_args->size(), Diagnostics);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  Invocation->getDependencyOutputOpts() = DependencyOutputOptions();

  // create a compiler instance to handle the actual work
  CompilerInstance Compiler;
  Compiler.setInvocation(Invocation.release());

  // create the action for Hipacc
  std::unique_ptr<ASTFrontendAction> HipaccAction(
      new HipaccRewriteAction(compilerOptions, out));

  // create the compiler's actual diagnostics engine.
  Compiler.createDiagnostics();
  if (!Compiler.hasDiagnostics())
    return EXIT_FAILURE;

  // run the action
  return !Compiler.ExecuteAction(*HipaccAction);
}

// vim: set ts=2 sw=2 sts=2 et ai:

