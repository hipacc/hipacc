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

//===--- TargetDescription.h - Target hardware feature description -==========//
//
// This provides an abstract description of target hardware features.
//
//===----------------------------------------------------------------------===//

#ifndef _TARGET_DESCRIPTION_H
#define _TARGET_DESCRIPTION_H

#include <sstream>

#include "hipacc/Config/config.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDevices.h"

namespace clang {
namespace hipacc {
// kernel type categorization
enum KernelType {
  PointOperator   = 0x0,
  LocalOperator   = 0x1,
  GlobalOperator  = 0x2,
  UserOperator    = 0x4,
  NumOperatorTypes
};

class HipaccDeviceOptions {
  public:
    unsigned alignment;
    unsigned local_memory_threshold;
    unsigned default_num_threads_x;
    unsigned default_num_threads_y;
    unsigned pixels_per_thread[NumOperatorTypes];
    Texture require_textures[NumOperatorTypes];
    bool vectorization;

  public:
    HipaccDeviceOptions(CompilerOptions &options) :
      default_num_threads_x(128),
      default_num_threads_y(1)
    {
      switch (options.getTargetDevice()) {
        case Device::Tesla_10:
        case Device::Tesla_11:
          alignment = 512;  // Quadro FX 1800
          alignment = 1024; // GeForce GTS 8800
          local_memory_threshold = 9999,
          pixels_per_thread[PointOperator] = 8;
          if (options.emitCUDA()) {
            pixels_per_thread[LocalOperator] = 8;
          } else {
            pixels_per_thread[LocalOperator] = 1;
          }
          pixels_per_thread[GlobalOperator] = 31;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::Linear2D;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::Linear2D;
          vectorization = false;
          break;
        case Device::Tesla_12:
        case Device::Tesla_13:
          alignment = 512;
          local_memory_threshold = 9999;
          pixels_per_thread[PointOperator] = 8;
          if (options.emitCUDA()) {
            pixels_per_thread[LocalOperator] = 4;
          } else {
            pixels_per_thread[LocalOperator] = 1;
          }
          pixels_per_thread[GlobalOperator] = 31;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::Linear2D;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::Linear2D;
          vectorization = false;
          break;
        case Device::Fermi_20:
        case Device::Fermi_21:
        case Device::Kepler_30:
        case Device::Kepler_35:
          alignment = 256;
          if (options.emitCUDA()) local_memory_threshold = 6;
          else local_memory_threshold = 11;
          default_num_threads_x = 256;
          pixels_per_thread[PointOperator] = 1;
          if (options.emitCUDA()) {
            pixels_per_thread[LocalOperator] = 8;
          } else {
            pixels_per_thread[LocalOperator] = 16;
          }
          pixels_per_thread[GlobalOperator] = 15;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::Linear1D;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::Linear1D;
          vectorization = false;
          break;
        case Device::Evergreen:
          alignment = 1024;
          local_memory_threshold = 17;
          pixels_per_thread[PointOperator] = 4;
          pixels_per_thread[LocalOperator] = 8;
          pixels_per_thread[GlobalOperator] = 32;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::None;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::None;
          vectorization = true;
          break;
        case Device::NorthernIsland:
          alignment = 512;
          local_memory_threshold = 21;
          pixels_per_thread[PointOperator] = 4;
          pixels_per_thread[LocalOperator] = 4;
          pixels_per_thread[GlobalOperator] = 32;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::None;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::None;
          vectorization = true;
          break;
        case Device::Midgard:
          alignment = 512;
          local_memory_threshold = 9999;
          default_num_threads_x = 4;
          pixels_per_thread[PointOperator] = 1;
          pixels_per_thread[LocalOperator] = 1;
          pixels_per_thread[GlobalOperator] = 32;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::None;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::None;
          vectorization = true;
          break;
        case Device::KnightsCorner:
          alignment = 64;
          local_memory_threshold = 9999;
          default_num_threads_x = 512;
          pixels_per_thread[PointOperator] = 2;
          pixels_per_thread[LocalOperator] = 2;
          pixels_per_thread[GlobalOperator] = 2;
          require_textures[PointOperator] = Texture::None;
          require_textures[LocalOperator] = Texture::None;
          require_textures[GlobalOperator] = Texture::None;
          require_textures[UserOperator] = Texture::None;
          vectorization = true;
          break;
      }

      // deactivate for custom operators
      pixels_per_thread[UserOperator] = 1;

      // use default provided by user as compiler option
      if (options.multiplePixelsPerThread((CompilerOption)(USER_ON|USER_OFF))) {
        pixels_per_thread[PointOperator] = options.getPixelsPerThread();
        pixels_per_thread[LocalOperator] = options.getPixelsPerThread();
        pixels_per_thread[GlobalOperator] = options.getPixelsPerThread();
        pixels_per_thread[UserOperator] = options.getPixelsPerThread();
      }

      if (options.useKernelConfig(USER_ON)) {
        default_num_threads_x = options.getKernelConfigX();
        default_num_threads_y = options.getKernelConfigY();
      }

      if (options.emitPadding(USER_ON)) {
        alignment = options.getAlignment();
      }

      if (options.useLocalMemory(USER_ON)) {
        local_memory_threshold = 2;
      }
      if (options.useLocalMemory(USER_OFF)) {
        local_memory_threshold = 9999;
      }

      if (options.useTextureMemory(USER_ON) ||
          options.useTextureMemory(USER_OFF)) {
        require_textures[PointOperator] = options.getTextureType();
        require_textures[LocalOperator] = options.getTextureType();
        require_textures[GlobalOperator] = options.getTextureType();
        require_textures[UserOperator] = options.getTextureType();
      }

      if (options.vectorizeKernels(USER_ON)) {
        vectorization = true;
      } else if (options.vectorizeKernels((CompilerOption)(USER_OFF|OFF))) {
        vectorization = false;
      }
    }
};


class HipaccDevice : public HipaccDeviceOptions {
  public:
    Device target_device;
    unsigned max_threads_per_warp;
    unsigned max_threads_per_block;
    unsigned max_blocks_per_multiprocessor;
    unsigned max_warps_per_multiprocessor;
    unsigned max_threads_per_multiprocessor;
    unsigned max_total_registers;
    unsigned max_total_shared_memory;
    unsigned max_register_per_thread;

    // NVIDIA only device properties
    unsigned num_alus;
    unsigned num_sfus;

  public:
    HipaccDevice(CompilerOptions &options) :
      HipaccDeviceOptions(options),
      target_device(options.getTargetDevice()),
      max_threads_per_warp(32),
      max_blocks_per_multiprocessor(8),
      num_alus(0),
      num_sfus(0)
    {
      switch (target_device) {
        case Device::Tesla_10:
        case Device::Tesla_11:
          max_threads_per_block = 512;
          max_warps_per_multiprocessor = 24;
          max_threads_per_multiprocessor = 768;
          max_total_registers = 8192;
          max_total_shared_memory = 16384;
          max_register_per_thread = 124;
          num_alus = 8;
          num_sfus = 2;
          break;
        case Device::Tesla_12:
        case Device::Tesla_13:
          max_threads_per_block = 512;
          max_warps_per_multiprocessor = 32;
          max_threads_per_multiprocessor = 1024;
          max_total_registers = 16384;
          max_total_shared_memory = 16384;
          max_register_per_thread = 124;
          num_alus = 8;
          num_sfus = 2;
          break;
        case Device::Fermi_20:
          max_threads_per_block = 1024;
          max_warps_per_multiprocessor = 48;
          max_threads_per_multiprocessor = 1536;
          max_total_registers = 32768;
          max_total_shared_memory = 49152;
          max_register_per_thread = 63;
          num_alus = 32;
          num_sfus = 4;
          break;
        case Device::Fermi_21:
          max_threads_per_block = 1024;
          max_warps_per_multiprocessor = 48;
          max_threads_per_multiprocessor = 1536;
          max_total_registers = 32768;
          max_total_shared_memory = 49152;
          max_register_per_thread = 63;
          num_alus = 48;
          num_sfus = 8;
          break;
        case Device::Kepler_30:
        case Device::Kepler_35:
          max_blocks_per_multiprocessor = 16;
          max_threads_per_block = 1024;
          max_warps_per_multiprocessor = 64;
          max_threads_per_multiprocessor = 2048;
          max_total_registers = 65536;
          max_total_shared_memory = 49152;
          if (target_device==Device::Kepler_30) max_register_per_thread = 63;
          else max_register_per_thread = 255;
          num_alus = 192;
          num_sfus = 32;
          // plus 8 CUDA FP64 cores according to andatech
          break;
        case Device::Evergreen:
        case Device::NorthernIsland:
          max_threads_per_warp = 64;
          max_blocks_per_multiprocessor = 8;
          max_threads_per_block = 256;
          max_warps_per_multiprocessor = 32;
          // 58: max wavefronts per GPU = 496
          // 69: max wavefronts per GPU = 512
          // 58: average active wavefronts per CU = 24.8
          // 69: average active wavefronts per CU = 21.3
          max_threads_per_multiprocessor = 1024;
          max_total_registers = 16384;    // each 4x32bit => 256kB
          max_total_shared_memory = 32768;
          num_alus = 4; // 5 on 58; 4 on 69
          num_sfus = 1; // 1 sfu -> 1 alu
          break;
        case Device::Midgard:
          max_threads_per_warp = 4,
          // max_blocks_per_multiprocessor - unknown
          max_threads_per_block = 256;
          max_warps_per_multiprocessor = 64; // unknown
          max_threads_per_multiprocessor = 256;
          max_total_registers = 32768; // unknown
          max_total_shared_memory = 32768;
          num_alus = 4; // vector 4
          num_sfus = 1; // just a guess
          break;
        case Device::KnightsCorner:
          max_threads_per_warp = 4,
          // max_blocks_per_multiprocessor - unknown
          max_threads_per_block = 8192;
          max_warps_per_multiprocessor = 64; // unknown
          max_threads_per_multiprocessor = 256; // unknown
          max_total_registers = 32; // each 512bit => 2kB
          max_total_shared_memory = 32768;
          num_alus = 16; // 512 bit vector units - for single precision
          num_sfus = 0;
          break;
      }
    }

    bool isAMDGPU() {
      switch (target_device) {
        default:                     return false;
        case Device::Evergreen:
        case Device::NorthernIsland: return true;
      }
    }

    bool isARMGPU() {
      switch (target_device) {
        default:              return false;
        case Device::Midgard: return true;
      }
    }

    bool isINTELACC() {
      switch (target_device) {
        default:                    return false;
        case Device::KnightsCorner: return true;
      }
    }

    bool isNVIDIAGPU() {
      switch (target_device) {
        default:                return false;
        case Device::Tesla_10:
        case Device::Tesla_11:
        case Device::Tesla_12:
        case Device::Tesla_13:
        case Device::Fermi_20:
        case Device::Fermi_21:
        case Device::Kepler_30:
        case Device::Kepler_35: return true;
      }
    }

    std::string getTargetDeviceName() {
      switch (target_device) {
        //case Device::CPU:             return "x86_64 CPU";
        case Device::Tesla_10:        return "NVIDIA Tesla (10)";
        case Device::Tesla_11:        return "NVIDIA Tesla (11)";
        case Device::Tesla_12:        return "NVIDIA Tesla (12)";
        case Device::Tesla_13:        return "NVIDIA Tesla (13)";
        case Device::Fermi_20:        return "NVIDIA Fermi (20)";
        case Device::Fermi_21:        return "NVIDIA Fermi (21)";
        case Device::Kepler_30:       return "NVIDIA Kepler (30)";
        case Device::Kepler_35:       return "NVIDIA Kepler (35)";
        case Device::Evergreen:       return "AMD Evergreen";
        case Device::NorthernIsland:  return "AMD Northern Island";
        //case Device::SouthernIsland:  return "AMD Southern Island";
        case Device::Midgard:         return "ARM Midgard: Mali-T6xx";
        case Device::KnightsCorner:   return "Intel MIC: Knights Corner";
      }
    }

    std::string getCompileCommand(bool emitCUDA) {
      if (emitCUDA)
        return CUDA_COMPILER;
      return CL_COMPILER;
    }

    std::string getCLIncludes() {
      if (isARMGPU())
        return EMBEDDED_RUNTIME_INCLUDES;
      return RUNTIME_INCLUDES;
    }

    unsigned getTargetCC() {
      assert(isNVIDIAGPU() && "compute capability only valid for NVIDIA");
      return static_cast<std::underlying_type<Device>::type>(target_device);
    }

    std::string getCompileOptions(std::string kernel, std::string file, bool
        emitCUDA) {
      if (emitCUDA) {
        return " -I " + std::string(RUNTIME_INCLUDES) + " -arch=sm_" +
          std::to_string(getTargetCC()) + " -cubin -Xptxas -v " + file +
          ".cu 2>&1";
      } else {
        if (isAMDGPU()) {
          return " -i " + std::string(RUNTIME_INCLUDES) + " -k " + kernel +
            " -f " + file + ".cl -p AMD -d GPU 2>&1";
        } else {
          return " -i " + std::string(RUNTIME_INCLUDES) + " -k " + kernel +
            " -f " + file + ".cl -p NVIDIA -d GPU 2>&1";
        }
      }
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _TARGET_DESCRIPTION_H

// vim: set ts=2 sw=2 sts=2 et ai:

