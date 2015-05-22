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

#include "hipacc/Config/config.h"
#include "hipacc/Device/TargetDevices.h"

#include <llvm/Support/raw_ostream.h>

#include <string>

namespace clang {
namespace hipacc {

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
  OpenCLGPU,
  Renderscript,
  Filterscript
};

class CompilerOptions {
  private:
    // target code and device specification
    Language target_lang;
    Device target_device;
    // target code features
    CompilerOption explore_config;
    CompilerOption time_kernels;
    // target code features - may be selected by the framework
    CompilerOption kernel_config;
    CompilerOption align_memory;
    CompilerOption texture_memory;
    CompilerOption local_memory;
    CompilerOption multiple_pixels;
    CompilerOption vectorize_kernels;
    // user defined values for target code features
    int kernel_config_x, kernel_config_y;
    int align_bytes;
    int pixels_per_thread;
    Texture texture_type;
    std::string rs_package_name, rs_directory;

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
      target_device(Device::Fermi_20),
      explore_config(OFF),
      time_kernels(OFF),
      kernel_config(AUTO),
      align_memory(AUTO),
      texture_memory(AUTO),
      local_memory(AUTO),
      multiple_pixels(AUTO),
      vectorize_kernels(OFF),
      kernel_config_x(128),
      kernel_config_y(1),
      align_bytes(0),
      pixels_per_thread(1),
      texture_type(Texture::None),
      rs_package_name("org.hipacc.rs"),
      rs_directory("/data/local/tmp")
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
    bool emitRenderscript() { return target_lang == Language::Renderscript; }
    bool emitFilterscript() { return target_lang == Language::Filterscript; }

    Language getTargetLang() { return target_lang; }
    Device getTargetDevice() { return target_device; }

    bool exploreConfig(CompilerOption option=(CompilerOption)(ON|USER_ON)) {
      if (explore_config & option) return true;
      return false;
    }
    bool timeKernels(CompilerOption option=(CompilerOption)(ON|USER_ON)) {
      if (time_kernels & option) return true;
      return false;
    }
    bool useKernelConfig(CompilerOption option=(CompilerOption)(ON|USER_ON)) {
      if (kernel_config & option) return true;
      return false;
    }
    int getKernelConfigX() { return kernel_config_x; }
    int getKernelConfigY() { return kernel_config_y; }

    bool emitPadding(CompilerOption option=(CompilerOption)(AUTO|ON|USER_ON)) {
      if (align_memory & option) return true;
      return false;
    }
    int getAlignment() { return align_bytes; }

    bool useTextureMemory(CompilerOption option=(CompilerOption)(ON|USER_ON))
    {
      if (texture_memory & option) return true;
      return false;
    }
    Texture getTextureType() { return texture_type; }

    bool useLocalMemory(CompilerOption option=(CompilerOption)(ON|USER_ON)) {
      if (local_memory & option) return true;
      return false;
    }
    bool vectorizeKernels(CompilerOption option=(CompilerOption)(ON|USER_ON))
    {
      if (vectorize_kernels & option) return true;
      return false;
    }
    bool multiplePixelsPerThread(CompilerOption
        option=(CompilerOption)(ON|USER_ON)) {
      if (multiple_pixels & option) return true;
      return false;
    }
    int getPixelsPerThread() { return pixels_per_thread; }
    std::string getRSPackageName() { return rs_package_name; }
    std::string getRSDirectory() { return rs_directory; }

    void setTargetLang(Language lang) { target_lang = lang; }
    void setTargetDevice(Device td) { target_device = td; }
    void setExploreConfig(CompilerOption o) { explore_config = o; }
    void setTimeKernels(CompilerOption o) { time_kernels = o; }
    void setLocalMemory(CompilerOption o) { local_memory = o; }
    void setVectorizeKernels(CompilerOption o) { vectorize_kernels = o; }

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

    void setRSPackageName(std::string name) {
      rs_package_name = name;
      rs_directory = "/data/data/" + name;
    }

    std::string getTargetPrefix() {
      switch (target_lang) {
        case Language::C99:          return "cc";
        case Language::CUDA:         return "cu";
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:    return "cl";
        case Language::Renderscript: return "rs";
        case Language::Filterscript: return "fs";
      }
    }

    void printSummary(std::string target_device) {
      llvm::errs() << "HIPACC compiler configuration summary: \n";
      llvm::errs() << "  Generating target code for '";
      switch (target_lang) {
        case Language::C99:          llvm::errs() << "C/C++";        break;
        case Language::CUDA:         llvm::errs() << "CUDA";         break;
        case Language::OpenCLACC:    llvm::errs() << "OpenCL (ACC)"; break;
        case Language::OpenCLCPU:    llvm::errs() << "OpenCL (CPU)"; break;
        case Language::OpenCLGPU:    llvm::errs() << "OpenCL (GPU)"; break;
        case Language::Renderscript: llvm::errs() << "Renderscript"; break;
        case Language::Filterscript: llvm::errs() << "Filterscript"; break;
      }
      llvm::errs() << "' language.\n";
      llvm::errs() << "  Target device is '" << target_device << "'";

      llvm::errs() << "\n  Exploration of kernel configurations: ";
      getOptionAsString(explore_config);
      llvm::errs() << "\n  Automatic timing of kernel executions: ";
      getOptionAsString(time_kernels);

      llvm::errs() << "\n  Kernel execution configuration: ";
      getOptionAsString(kernel_config);
      if (useKernelConfig()) {
        llvm::errs() << ": " << kernel_config_x << "x" << kernel_config_y;
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
      llvm::errs() << "\n\n";
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _COMPILER_OPTIONS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

