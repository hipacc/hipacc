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
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Signals.h>

#include <sstream>
<<<<<<< HEAD
#include <fstream>
||||||| 035cfd9
=======
#include <memory>
>>>>>>> vectorization

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
  std::string execPath = llvm::sys::fs::getMainExecutable(argv0, mainAddr);

  // search for separator of dirname and basename
  size_t pos = execPath.rfind('/', execPath.length());
  if (pos == std::string::npos) {
    // try backslash on windows
    pos = execPath.rfind('\\', execPath.length());

    if (pos == std::string::npos) {
      llvm::errs() << "ERROR: Could not determine path to Hipacc executable.";
      exit(EXIT_FAILURE);
    }
  }

  // strip basename from path
  return execPath.substr(0, pos);
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
    << "  -nvcc-path              Path for the NVCC compier for JIT compilation\n"
    << "  -cl-compiler-path       Path for the OpenCL compier for JIT compilation\n"
    << "  -ccbin-path             Path host compiler binary directory (Windows only)\n"
    << "  -rt-includes-path       Path for the runtime libraries\n"
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
    << "  -reduce-config <nxm>    Emit code that uses a multi-dimensional reduction configuration of\n"
    << "                            n warps per block    (affects block size and shared memory size)\n"
    << "                            m partial histograms (affects number of blocks)\n"
    << "  -time-kernels           Emit code that executes each kernel multiple times to get accurate timings\n"
    << "  -use-textures <o>       Enable/disable usage of textures (cached) in CUDA/OpenCL to read/write image pixels - for GPU devices only\n"
    << "                          Valid values for CUDA on NVIDIA devices: 'off', 'Linear1D', 'Linear2D', 'Array2D', and 'Ldg'\n"
    << "                          Valid values for OpenCL: 'off' and 'Array2D'\n"
    << "  -use-local <o>          Enable/disable usage of shared/local memory in CUDA/OpenCL to stage image pixels to scratchpad\n"
    << "                          Valid values: 'on' and 'off'\n"
    << "  -vectorize <o>          Enable/disable vectorization of generated CUDA/OpenCL code\n"
    << "                          Valid values: 'on' and 'off'\n"
    << "  -pixels-per-thread <n>  Specify how many pixels should be calculated per thread (GPU)\n"
    << "  -rows-per-thread <n>    Specify how many rows should be calculated per thread (CPU)\n"
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
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

  // setup and initialize compiler instance
  void *mainAddr = (void *) (intptr_t) getExecutablePath;
  std::string Path = getExecutablePath(argv[0]);

  // argument list for CompilerInvocation after removing our compiler flags
  SmallVector<const char *, 16> Args;
  CompilerOptions compilerOptions = CompilerOptions();
  std::string out;

<<<<<<< HEAD
  // parse command line options
  for (int i=0; i<argc; ++i) {
    if (StringRef(argv[i]) == "-emit-cpu") {
      compilerOptions.setTargetLang(Language::C99);
      compilerOptions.setTargetDevice(Device::CPU);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-cuda") {
      compilerOptions.setTargetLang(Language::CUDA);
      continue;
    }
    if (StringRef(argv[i]) == "-nvcc-path") {
      hipacc_require(i<(argc-1), "Mandatory path parameter for nvcc missing.");
      std::string nvcc_compiler = StringRef(argv[i+1]);
      std::ifstream compiler(nvcc_compiler);
      hipacc_require(compiler.good(), "cannot find NVCC for JIT compilation.");
      compilerOptions.setNvccPath(nvcc_compiler);
      continue;
    }
    if (StringRef(argv[i]) == "-ccbin-path") {
      hipacc_require(i<(argc-1), "Mandatory path parameter for ccbin-path missing.");
      std::string ccbin_path = StringRef(argv[i+1]);
      compilerOptions.setCCBinPath(ccbin_path);
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
    if (StringRef(argv[i]) == "-cl-compiler-path") {
      hipacc_require(i<(argc-1), "Mandatory path parameter for nvcc missing.");
      std::string cl_compiler = StringRef(argv[i+1]);
      std::ifstream compiler(cl_compiler);
      hipacc_require(compiler.good(), "cannot find OpenCL compiler for JIT compilation.");
      compilerOptions.setClCompilerPath(cl_compiler);
      continue;
    }
    if (StringRef(argv[i]) == "-rt-includes-path") {
      hipacc_require(i<(argc-1), "Mandatory path parameter for runtime missing.");
      std::string rt_path = StringRef(argv[i+1]);
      compilerOptions.setRTIncPath(rt_path);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-renderscript") {
      compilerOptions.setTargetLang(Language::Renderscript);
      continue;
||||||| 035cfd9
  // parse command line options
  for (int i=0; i<argc; ++i) {
    if (StringRef(argv[i]) == "-emit-cpu") {
      compilerOptions.setTargetLang(Language::C99);
      compilerOptions.setTargetDevice(Device::CPU);
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
=======
  // Initialize the backends
  Backend::BackendConfigurationManager BackendConfigManager(&compilerOptions);
  try
  {
    // Convert the command line arguments into the backend argument vector type
    Backend::CommonDefines::ArgumentVectorType vecArguments;
    for (int i = 1; i < argc; ++i)
    {
      vecArguments.push_back(argv[i]);
>>>>>>> vectorization
    }
<<<<<<< HEAD
    if (StringRef(argv[i]) == "-emit-filterscript") {
      compilerOptions.setTargetLang(Language::Filterscript);
      compilerOptions.setPixelsPerThread(1);
      continue;
    }
    if (StringRef(argv[i]) == "-emit-padding") {
      hipacc_require(i<(argc-1), "Mandatory alignment parameter for -emit-padding switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory code name parameter for -target switch missing.");

      if (!compilerOptions.emitCUDA() &&
          !compilerOptions.emitOpenCLACC() &&
          !compilerOptions.emitOpenCLGPU() &&
          !compilerOptions.emitRenderscript() &&
          !compilerOptions.emitFilterscript()) {
        llvm::errs() << "WARNING: Setting target is only supported for GPU code generation.\n\n";
        continue;
      }
||||||| 035cfd9
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

      if (!compilerOptions.emitCUDA() &&
          !compilerOptions.emitOpenCLACC() &&
          !compilerOptions.emitOpenCLGPU() &&
          !compilerOptions.emitRenderscript() &&
          !compilerOptions.emitFilterscript()) {
        llvm::errs() << "WARNING: Setting target is only supported for GPU code generation.\n\n";
        continue;
      }
=======
>>>>>>> vectorization

<<<<<<< HEAD
      if (StringRef(argv[i+1]) == "Fermi-20") {
        compilerOptions.setTargetDevice(Device::Fermi_20);
      } else if (StringRef(argv[i+1]) == "Fermi-21") {
        compilerOptions.setTargetDevice(Device::Fermi_21);
      } else if (StringRef(argv[i+1]) == "Kepler-30") {
        compilerOptions.setTargetDevice(Device::Kepler_30);
      } else if (StringRef(argv[i+1]) == "Kepler-32") {
        compilerOptions.setTargetDevice(Device::Kepler_32);
      } else if (StringRef(argv[i+1]) == "Kepler-35") {
        compilerOptions.setTargetDevice(Device::Kepler_35);
      } else if (StringRef(argv[i+1]) == "Kepler-37") {
        compilerOptions.setTargetDevice(Device::Kepler_37);
      } else if (StringRef(argv[i+1]) == "Maxwell-50") {
        compilerOptions.setTargetDevice(Device::Maxwell_50);
      } else if (StringRef(argv[i+1]) == "Maxwell-52") {
        compilerOptions.setTargetDevice(Device::Maxwell_52);
      } else if (StringRef(argv[i+1]) == "Maxwell-53") {
        compilerOptions.setTargetDevice(Device::Maxwell_53);
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
      hipacc_require(i<(argc-1), "Mandatory configuration specification for -use-config switch missing.");
      int x=0, y=0, ret=0;
      ret = sscanf(argv[i+1], "%dx%d", &x, &y);
      if (ret!=2) {
        llvm::errs() << "ERROR: Expected valid configuration specification for -use-config switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
||||||| 035cfd9
      if (StringRef(argv[i+1]) == "Fermi-20") {
        compilerOptions.setTargetDevice(Device::Fermi_20);
      } else if (StringRef(argv[i+1]) == "Fermi-21") {
        compilerOptions.setTargetDevice(Device::Fermi_21);
      } else if (StringRef(argv[i+1]) == "Kepler-30") {
        compilerOptions.setTargetDevice(Device::Kepler_30);
      } else if (StringRef(argv[i+1]) == "Kepler-32") {
        compilerOptions.setTargetDevice(Device::Kepler_32);
      } else if (StringRef(argv[i+1]) == "Kepler-35") {
        compilerOptions.setTargetDevice(Device::Kepler_35);
      } else if (StringRef(argv[i+1]) == "Kepler-37") {
        compilerOptions.setTargetDevice(Device::Kepler_37);
      } else if (StringRef(argv[i+1]) == "Maxwell-50") {
        compilerOptions.setTargetDevice(Device::Maxwell_50);
      } else if (StringRef(argv[i+1]) == "Maxwell-52") {
        compilerOptions.setTargetDevice(Device::Maxwell_52);
      } else if (StringRef(argv[i+1]) == "Maxwell-53") {
        compilerOptions.setTargetDevice(Device::Maxwell_53);
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
=======
    // Let the backend configuration manager parse the command line arguments
    BackendConfigManager.Configure(vecArguments);
    out = BackendConfigManager.GetOutputFile();

    // Fetch the commands vector the clang invocation and convert it to clang's format
    vecArguments = BackendConfigManager.GetClangArguments(Path);
    for (auto itArgument : vecArguments)
    {
      char *pcArgument = (char*) calloc(itArgument.size() + 1, sizeof(char));
      if (pcArgument == NULL)
      {
        throw std::runtime_error("Cannot allocate memory for clang command argument!");
>>>>>>> vectorization
      }
<<<<<<< HEAD
      compilerOptions.setKernelConfig(x, y);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-reduce-config") {
      hipacc_require(i<(argc-1), "Mandatory configuration specification for -reduce-config switch missing.");
      int num_warps=0, num_hists=0, ret=0;
      ret = sscanf(argv[i+1], "%dx%d", &num_warps, &num_hists);
      if (ret!=2) {
        llvm::errs() << "ERROR: Expected valid configuration specification for -use-config switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setReduceConfig(num_warps, num_hists);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-time-kernels") {
      compilerOptions.setTimeKernels(USER_ON);
      continue;
    }
    if (StringRef(argv[i]) == "-use-textures") {
      hipacc_require(i<(argc-1), "Mandatory texture memory specification for -use-textures switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory local memory specification for -use-local switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory vectorization specification for -vectorize switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory integer parameter for -pixels-per-thread switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory package name string for -rs-package switch missing.");
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
      hipacc_require(i<(argc-1), "Mandatory output file name for -o switch missing.");
      args.push_back(argv[i]);
      args.push_back(argv[++i]);
      out = argv[i];
      continue;
    }
    if (StringRef(argv[i]) == "-rows-per-thread") {
      hipacc_require(i<(argc-1), "Mandatory integer parameter for -rows-per-thread switch missing.");
      std::istringstream buffer(argv[i+1]);
      int val;
      buffer >> val;
      if (buffer.fail()) {
        llvm::errs() << "ERROR: Expected integer parameter for -rows-per-thread switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setPixelsPerThread(val);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-multi-threading") {
      llvm::errs() << "Warning: OpenMP multi-threading not yet supported.\n";
      if (StringRef(argv[i + 1]) != "off"
          && StringRef(argv[i + 1]) != "on") {
        llvm::errs() << "ERROR: Expected valid specification for -multi-threading switch.\n\n";
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-instruction-set") {
      llvm::errs() << "Warning: Instruction set specification not yet supported.\n";
      if (StringRef(argv[i + 1]) != "sse4.2"
          && StringRef(argv[i + 1]) != "avx"
          && StringRef(argv[i + 1]) != "avx2") {
        llvm::errs() << "ERROR: Expected valid specification for -instruction-set switch.\n\n";
        return EXIT_FAILURE;
      }
      ++i;
      continue;
    }

    if(strcmp("-I", argv[i]) == 0 && !args.empty() && strcmp("-I", args.back()) == 0)
      continue;
||||||| 035cfd9
      compilerOptions.setKernelConfig(x, y);
      ++i;
      continue;
    }
    if (StringRef(argv[i]) == "-reduce-config") {
      assert(i<(argc-1) && "Mandatory configuration specification for -reduce-config switch missing.");
      int num_warps=0, num_hists=0, ret=0;
      ret = sscanf(argv[i+1], "%dx%d", &num_warps, &num_hists);
      if (ret!=2) {
        llvm::errs() << "ERROR: Expected valid configuration specification for -use-config switch.\n\n";
        printUsage();
        return EXIT_FAILURE;
      }
      compilerOptions.setReduceConfig(num_warps, num_hists);
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
=======
>>>>>>> vectorization

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


  if (compilerOptions.useKernelConfig(USER_ON) && !compilerOptions.emitC99()) {
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


  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // use the Driver (from Tooling.cpp)
  driver::Driver Driver(Args[0], llvm::sys::getDefaultTargetTriple(),
      Diags);

  const std::unique_ptr<driver::Compilation> Compilation(
      Driver.BuildCompilation(Args));

  // use the flags from the first job
  const driver::JobList &Jobs = Compilation->getJobs();
  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
<<<<<<< HEAD
  const llvm::opt::ArgStringList *const cc1_args = &Cmd.getArguments();

  std::unique_ptr<CompilerInvocation> Invocation(new CompilerInvocation());
  CompilerInvocation::CreateFromArgs(*Invocation,
      cc1_args->data() + 1, cc1_args->data() + cc1_args->size(), Diagnostics);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  Invocation->getDependencyOutputOpts() = DependencyOutputOptions();
  Invocation->getPreprocessorOpts().addMacroDef("__HIPACC__");

  // create a compiler instance to handle the actual work
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));

  // create the action for Hipacc
  std::unique_ptr<ASTFrontendAction> HipaccAction(
      new HipaccRewriteAction(compilerOptions, out));

  // create the compiler's actual diagnostics engine.
  Compiler.createDiagnostics();
  if (!Compiler.hasDiagnostics())
    return EXIT_FAILURE;
||||||| 035cfd9
  const llvm::opt::ArgStringList *const cc1_args = &Cmd.getArguments();

  std::unique_ptr<CompilerInvocation> Invocation(new CompilerInvocation());
  CompilerInvocation::CreateFromArgs(*Invocation,
      cc1_args->data() + 1, cc1_args->data() + cc1_args->size(), Diagnostics);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  Invocation->getDependencyOutputOpts() = DependencyOutputOptions();

  // create a compiler instance to handle the actual work
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));

  // create the action for Hipacc
  std::unique_ptr<ASTFrontendAction> HipaccAction(
      new HipaccRewriteAction(compilerOptions, out));

  // create the compiler's actual diagnostics engine.
  Compiler.createDiagnostics();
  if (!Compiler.hasDiagnostics())
    return EXIT_FAILURE;
=======
  const llvm::opt::ArgStringList cc1_args = Cmd.getArguments();

  // initialize a compiler invocation object from the arguments
  bool success;
  success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
      const_cast<const char **>(cc1_args.data()),
      const_cast<const char **>(cc1_args.data()) + cc1_args.size(), Diags);

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
>>>>>>> vectorization

  return EXIT_SUCCESS;
}

// vim: set ts=2 sw=2 sts=2 et ai:

