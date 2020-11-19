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

//===--- CompilerOptions.h - List of compiler options for code generation -===//
//
// This provides compiler options that drive the code generation.
//
//===----------------------------------------------------------------------===//

#ifndef _COMPILER_OPTIONS_H_
#define _COMPILER_OPTIONS_H_

#include "ErrorReporting.h"

#include "hipacc/Config/config.h"
#include "hipacc/Device/TargetDevices.h"

#include <clang/Basic/Version.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <memory>


#if CLANG_VERSION_MAJOR != 10
#error "Clang Version 10.x required!"
#endif

namespace clang {
namespace hipacc {
namespace Backend {
  class ICodeGenerator;
  typedef std::shared_ptr<ICodeGenerator> ICodeGeneratorPtr;
}

// compiler option possibilities
enum CompilerOption {
  AUTO                = 0x1,
  ON                  = 0x2,
  OFF                 = 0x4,
  USER_ON             = 0x8,
  USER_OFF            = 0x10
};

// target language specification
enum class Language : uint8_t {
  C99,
  CUDA,
  OpenCLACC,
  OpenCLCPU,
  OpenCLGPU
};

class CompilerOptions {
  private:
    // target code and device specification
    Language target_lang;
    Device target_device;
    CompilerOption print_verbose;
    // target code features
    CompilerOption time_kernels;
    // target code features - may be selected by the framework
    CompilerOption kernel_config;
    CompilerOption reduce_config;
    CompilerOption align_memory;
    CompilerOption texture_memory;
    CompilerOption local_memory;
    CompilerOption multiple_pixels;
    CompilerOption vectorize_kernels;
    CompilerOption fuse_kernels;
    CompilerOption use_graph;
    // user defined values for target code features
    int kernel_config_x, kernel_config_y;
    int reduce_config_num_warps, reduce_config_num_units;
    int align_bytes;
    int pixels_per_thread;
    Texture texture_type;
    bool use_openmp;
    std::string nvcc_path, cl_compiler_path, ccbin_path, rt_includes_path;

    // The selected code generator
    Backend::ICodeGeneratorPtr  _spCodeGenerator;

    void getOptionAsString(CompilerOption option, int val=-1) {
      switch (option) {
        case USER_ON:
          llvm::errs() << "USER - ENABLED";
          if (val!=-1) llvm::errs() << " with value '" << val << "'";
          break;
        case USER_OFF:
          llvm::errs() << "USER - DISABLED";
          break;
        case AUTO:
          llvm::errs() << "AUTO - determined by the framework";
          break;
        case ON:
          llvm::errs() << "ENABLED";
          break;
        case OFF:
          llvm::errs() << "DISABLED";
          break;
      }
    }

  public:
    CompilerOptions() :
      target_lang(Language::OpenCLGPU),
      target_device(Device::Kepler_30),
      print_verbose(OFF),
      time_kernels(OFF),
      kernel_config(AUTO),
      reduce_config(AUTO),
      align_memory(AUTO),
      texture_memory(AUTO),
      local_memory(AUTO),
      multiple_pixels(AUTO),
      vectorize_kernels(OFF),
      fuse_kernels(OFF),
      use_graph(OFF),
      kernel_config_x(128),
      kernel_config_y(1),
      reduce_config_num_warps(16),
      reduce_config_num_units(16),
      align_bytes(0),
      pixels_per_thread(1),
      texture_type(Texture::None),
      use_openmp(false),
      nvcc_path("nvcc"),
      cl_compiler_path(""),
      ccbin_path(""),
      rt_includes_path(""),
      _spCodeGenerator(nullptr)
    {}
    bool emitC99() { return target_lang == Language::C99; }
    bool emitCUDA() { return target_lang == Language::CUDA; }
    bool emitOpenCL() {
      return target_lang == Language::OpenCLACC ||
             target_lang == Language::OpenCLCPU ||
             target_lang == Language::OpenCLGPU;
    }
    bool emitOpenCLACC() { return target_lang == Language::OpenCLACC; }
    bool emitOpenCLCPU() { return target_lang == Language::OpenCLCPU; }
    bool emitOpenCLGPU() { return target_lang == Language::OpenCLGPU; }

    Language getTargetLang() { return target_lang; }
    Device getTargetDevice() { return target_device; }

    static const auto option_ou  = static_cast<CompilerOption>(     ON|USER_ON);
    static const auto option_aou = static_cast<CompilerOption>(AUTO|ON|USER_ON);

    bool printVerbose(CompilerOption option=option_ou) {
      return print_verbose & option;
    }
    bool timeKernels(CompilerOption option=option_ou) {
      return time_kernels & option;
    }
    bool useKernelConfig(CompilerOption option=option_ou) {
      return kernel_config & option;
    }
    int getKernelConfigX() { return kernel_config_x; }
    int getKernelConfigY() { return kernel_config_y; }

    bool useReduceConfig(CompilerOption option=option_ou) {
      return reduce_config & option;
    }
    int getReduceConfigNumWarps() { return reduce_config_num_warps; }
    int getReduceConfigNumUnits() { return reduce_config_num_units; }

    bool emitPadding(CompilerOption option=option_aou) {
      return align_memory & option;
    }
    int getAlignment() { return align_bytes; }

    bool useTextureMemory(CompilerOption option=option_ou) {
      return texture_memory & option;
    }
    Texture getTextureType() { return texture_type; }

    bool useLocalMemory(CompilerOption option=option_ou) {
      return local_memory & option;
    }
    bool vectorizeKernels(CompilerOption option=option_ou) {
      return vectorize_kernels & option;
    }
    bool fuseKernels(CompilerOption option=option_ou) {
      return fuse_kernels & option;
    }
    bool useGraph(CompilerOption option=option_ou) {
      return use_graph & option;
    }
    bool multiplePixelsPerThread(CompilerOption option=option_ou) {
      return multiple_pixels & option;
    }
    bool allowMisAlignedAccess() {
      return (target_device >= Device::Kepler_30 && target_device <= Device::Maxwell_53);
    }

    int getPixelsPerThread() { return pixels_per_thread; }
    bool useOpenMP() { return use_openmp; }
    std::string getNvccPath() { return nvcc_path; }
    std::string getClCompilerPath() { return cl_compiler_path; }
    std::string getCCBinPath() { return ccbin_path; }
    std::string getRTIncPath() { return rt_includes_path; }

    void setTargetLang(Language lang) { target_lang = lang; }
    void setTargetDevice(Device td) { target_device = td; }
    void setPrintVerbose(CompilerOption o) { print_verbose = o; }
    void setTimeKernels(CompilerOption o) { time_kernels = o; }
    void setLocalMemory(CompilerOption o) { local_memory = o; }
    void setVectorizeKernels(CompilerOption o) { vectorize_kernels = o; }
    void setFuseKernels(CompilerOption o) { fuse_kernels = o; }
    void setUseGraph(CompilerOption o) { use_graph = o; }

    void setTextureMemory(Texture type) {
      texture_type = type;
      if (type == Texture::None) texture_memory = USER_OFF;
      else texture_memory = USER_ON;
    }

    void setKernelConfig(int x, int y) {
      kernel_config = USER_ON;
      kernel_config_x = x;
      kernel_config_y = y;
    }

    void setReduceConfig(int num_warps, int num_units) {
      reduce_config = USER_ON;
      reduce_config_num_warps = num_warps;
      reduce_config_num_units = num_units;
    }

    void setPadding(int bytes) {
      align_bytes = bytes;
      if (bytes > 1) align_memory = USER_ON;
      else align_memory = USER_OFF;
    }

    void setPixelsPerThread(int pixels) {
      pixels_per_thread = pixels;
      if (pixels > 1) multiple_pixels = USER_ON;
      else multiple_pixels = USER_OFF;
    }

    void setOpenMP(bool flag) {
      use_openmp = flag;
    }

    void setNvccPath(const std::string &path) {
      nvcc_path = path;
    }

    void setClCompilerPath(const std::string &path) {
      cl_compiler_path = path;
    }

    void setCCBinPath(const std::string &path) {
      ccbin_path = path;
    }

    void setRTIncPath(const std::string &path) {
      rt_includes_path = path;
    }

    std::string getTargetPrefix() {
      switch (target_lang) {
        case Language::C99:          return "cc";
        case Language::CUDA:         return "cu";
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:    return "cl";
        default: hipacc_require(false , "Unsupported target language"); return "";
      }
    }

    Backend::ICodeGeneratorPtr  getCodeGenerator()                                            { return _spCodeGenerator; }
    void                        setCodeGenerator(Backend::ICodeGeneratorPtr& spCodeGenerator)  { _spCodeGenerator = spCodeGenerator; }

    void printSummary(const std::string &target_device) {
      llvm::errs() << "HIPACC compiler configuration summary: \n";
      llvm::errs() << "  Runtime include path: " << rt_includes_path << "\n";
      llvm::errs() << "  Generating target code for '";
      switch (target_lang) {
        case Language::C99:          llvm::errs() << "C/C++";        break;
        case Language::CUDA:         llvm::errs() << "CUDA";         break;
        case Language::OpenCLACC:    llvm::errs() << "OpenCL (ACC)"; break;
        case Language::OpenCLCPU:    llvm::errs() << "OpenCL (CPU)"; break;
        case Language::OpenCLGPU:    llvm::errs() << "OpenCL (GPU)"; break;
      }
      llvm::errs() << "' language.\n";
      llvm::errs() << "  Target device is '" << target_device << "'";

      llvm::errs() << "\n  Automatic timing of kernel executions: ";
      getOptionAsString(time_kernels);

      llvm::errs() << "\n  Kernel execution configuration: ";
      getOptionAsString(kernel_config);
      if (useKernelConfig()) {
        llvm::errs() << ": " << kernel_config_x << "x" << kernel_config_y;
      }
      llvm::errs() << "\n  Multi-dimension reduction configuration: ";
      getOptionAsString(kernel_config);
      if (useReduceConfig()) {
        llvm::errs() << ": " << reduce_config_num_warps << " warps"
                     << ", " << reduce_config_num_units << " histograms";
      }
      llvm::errs() << "\n  Alignment of image memory: ";
      getOptionAsString(align_memory, align_bytes);
      llvm::errs() << "\n  Usage of texture memory for images: ";
      getOptionAsString(texture_memory);
      switch (texture_type) {
        case Texture::None:                                    break;
        case Texture::Linear1D:  llvm::errs() << ": Linear1D"; break;
        case Texture::Linear2D:  llvm::errs() << ": Linear2D"; break;
        case Texture::Array2D:   llvm::errs() << ": Array2D";  break;
        case Texture::Ldg:       llvm::errs() << ": Ldg";      break;
      }
      llvm::errs() << "\n  Usage of local memory reading from images: ";
      getOptionAsString(local_memory);
      llvm::errs() << "\n  Mapping multiple pixels to one thread: ";
      getOptionAsString(multiple_pixels, pixels_per_thread);
      llvm::errs() << "\n  Vectorization of kernels: ";
      getOptionAsString(vectorize_kernels);
      llvm::errs() << "\n  Usage of kernel fusion: ";
      getOptionAsString(fuse_kernels);
      llvm::errs() << "\n  Usage of cuda graph: ";
      getOptionAsString(use_graph);
      llvm::errs() << "\n\n";
    }
};
} // namespace hipacc
} // namespace clang

#endif  // _COMPILER_OPTIONS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

