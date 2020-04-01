//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef __HIPACC_CU_HPP__
#define __HIPACC_CU_HPP__

#include <cuda.h>

#if CUDA_VERSION < 7000
#error "CUDA 7.0 or higher required!"
#endif

#ifdef __has_include
#if __has_include(<nvml.h>) && defined _MSC_VER
#define NVML_FOUND
#include <nvml.h>
#pragma comment(lib, "nvml.lib")
#endif
#endif

#include <nvrtc.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "hipacc_base.hpp"

// error checking device driver

std::string getCUDAErrorCodeStrDrv(CUresult errorCode);
std::string getCUDAErrorCodeStr(cudaError_t errorCode);

inline void checkErrDrv(CUresult err, const std::string &name) {
  if (err != CUDA_SUCCESS) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                "): " + getCUDAErrorCodeStrDrv(err));
  }
}

inline void checkErr(cudaError_t err, const std::string &name) {
  if (err != cudaSuccess) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                "): " + getCUDAErrorCodeStr(err));
  }
}

#ifdef NVML_FOUND
// error checking NVML
inline void checkErrNVML(nvmlReturn_t err, const std::string &name) {
  if (err != NVML_SUCCESS) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                "): " + nvmlErrorString(err));
  }
}
#endif

inline void checkErrNVRTC(nvrtcResult err, const std::string &name) {
  if (err != NVRTC_SUCCESS) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                "): " + nvrtcGetErrorString(err));
  }
}

extern HipaccKernelTimingBase hipaccCudaTiming;
inline float hipacc_last_kernel_timing() {
  return hipaccCudaTiming.get_last_kernel_timing();
}

// TODO: what is the purpose of this empty class?
class HipaccContext : public HipaccContextBase {
public:
  static HipaccContext &getInstance();
};

class HipaccImageCudaBase : public HipaccImageBase {
public:
  virtual ~HipaccImageCudaBase() = 0;
  virtual void *get_device_memory() const = 0;
  virtual void *get_host_memory() const = 0;
  virtual size_t get_width() const = 0;
  virtual size_t get_height() const = 0;
  virtual size_t get_stride() const = 0;
  virtual size_t get_alignment() const = 0;
  virtual size_t get_pixel_size() const = 0;
  virtual hipaccMemoryType get_mem_type() const = 0;
};
typedef std::shared_ptr<HipaccImageCudaBase> HipaccImageCuda;

class HipaccImageCudaRaw : public HipaccImageCudaBase {
private:
  size_t width, height;
  size_t stride, alignment;
  size_t pixel_size;
  hipaccMemoryType mem_type;
  void *device_mem{};
  std::unique_ptr<char[]> host_mem;

public:
  HipaccImageCudaRaw(size_t width, size_t height, size_t stride,
                     size_t alignment, size_t pixel_size, void *mem,
                     hipaccMemoryType mem_type = hipaccMemoryType::Global)
                   : width(width), height(height), stride(stride),
                   alignment(alignment), pixel_size(pixel_size),
                   mem_type(mem_type), device_mem(mem),
                   host_mem(new char[width * height * pixel_size]) {
    std::fill(host_mem.get(), host_mem.get() + width * height * pixel_size, 0);
  }

  ~HipaccImageCudaRaw() {
    cudaError_t err = cudaFree(device_mem);
    checkErr(err, "cudaFree()");
  }

  bool operator==(const HipaccImageCudaRaw &other) const {
    return device_mem == other.device_mem;
  }

  void *get_device_memory() const final { return device_mem; }
  void *get_host_memory() const final { return host_mem.get(); }
  size_t get_width() const final { return width; }
  size_t get_height() const final { return height; }
  size_t get_stride() const final { return stride; }
  size_t get_alignment() const final { return alignment; }
  size_t get_pixel_size() const final { return pixel_size; }
  hipaccMemoryType get_mem_type() const final { return mem_type; }
};

class HipaccAccessor : public HipaccAccessorBase {
public:
  HipaccImageCuda img;

public:
  HipaccAccessor(HipaccImageCuda img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
      : HipaccAccessorBase(width, height, offset_x, offset_y), img(img) {}
  HipaccAccessor(HipaccImageCuda img)
      : HipaccAccessorBase(img->get_width(), img->get_height(), 0, 0), img(img) {}
};

struct hipacc_const_info { // TODO: VOID
  hipacc_const_info(std::string name, void *memory, int size)
      : name(name), memory(memory), size(size) {}
  std::string name;
  void *memory;
  int size;
};

struct hipacc_tex_info { // TODO: VOID
  hipacc_tex_info(std::string name, CUarray_format format,
                  const HipaccImageCuda &image, hipaccMemoryType tex_type)
      : name(name), format(format), image(image), tex_type(tex_type) {}
  std::string name;
  CUarray_format format;
  const HipaccImageCuda &image;
  hipaccMemoryType tex_type;
};

class HipaccPyramidCuda : public HipaccPyramid {
private:
  std::vector<HipaccImageCuda> imgs_;

public:
  HipaccPyramidCuda(const int depth) : HipaccPyramid(depth) {}

  void add(const HipaccImageCuda &img) { imgs_.push_back(img); }
  HipaccImageCuda operator()(int relative) {
    assert(level_ + relative >= 0 && level_ + relative < (int)imgs_.size() &&
           "Accessed pyramid stage is out of bounds.");
    return imgs_.at(level_ + relative);
  }
  int depth() const { return depth_; }
  int level() const { return level_; }
  void levelInc() { ++level_; }
  void levelDec() { --level_; }
  bool is_top_level() const { return level_ == 0; }
  bool is_bottom_level() const { return level_ == depth_ - 1; }
  void swap(HipaccPyramidCuda &other) { imgs_.swap(other.imgs_); }
  bool bind() {
    if (!bound_) {
      bound_ = true;
      level_ = 0;
      return true;
    } else {
      return false;
    }
  }
  void unbind() { bound_ = false; }
};

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 &block);
dim3 hipaccCalcGridFromBlock(hipacc_launch_info &info, dim3 &block);
void hipaccInitCUDA();
void hipaccCopyMemory(const HipaccImageCuda &src, HipaccImageCuda &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst);
void hipaccLaunchKernel(const void *kernel, std::string kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing = true);
void hipaccLaunchKernelBenchmark(const void *kernel, std::string kernel_name,
                                 dim3 &grid, dim3 &block,
                                 std::vector<void *> &args,
                                 bool print_timing = true);
void hipaccLaunchKernelExploration(std::string filename, std::string kernel,
                                   std::vector<void *> &args,
                                   std::vector<hipacc_smem_info> &smems,
                                   std::vector<hipacc_const_info> &consts,
                                   std::vector<hipacc_tex_info *> &texs,
                                   hipacc_launch_info &info, size_t warp_size,
                                   size_t max_threads_per_block,
                                   size_t max_threads_for_kernel,
                                   size_t max_smem_per_block, size_t heu_tx,
                                   size_t heu_ty, int cc);

//
// TEMPLATES
//

template <typename T>
HipaccImageCuda
createImage(T *host_mem, void *mem, size_t width, size_t height, size_t stride,
            size_t alignment,
            hipaccMemoryType mem_type = hipaccMemoryType::Global);
template <typename T> T *createMemory(size_t stride, size_t height);
template <typename T>
HipaccImageCuda hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                   size_t alignment);
template <typename T>
HipaccImageCuda hipaccCreateMemory(T *host_mem, size_t width, size_t height);
template <typename T>
HipaccImageCuda hipaccCreateArray2D(T *host_mem, size_t width, size_t height);
template <typename T, typename TI>
TI hipaccCreatePyramidImage(const TI &base, size_t width, size_t height);
template <typename T> void hipaccWriteMemory(HipaccImageCuda &img, T *host_mem);
template <typename T> T *hipaccReadMemory(const HipaccImageCuda &img);
template <typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex,
                       const HipaccImageCuda &img);
template <typename T>
void hipaccBindSurface(hipaccMemoryType mem_type, const surfaceReference *surf,
                       const HipaccImageCuda &img);
void hipaccUnbindTexture(const textureReference *tex);
template <typename T>
void hipaccWriteSymbol(const void *symbol, T *host_mem, size_t width,
                       size_t height);
template <typename T>
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string symbol_name,
                      size_t width, size_t height);
template <typename T>
void hipaccWriteDomainFromMask(const void *symbol, T *host_mem, size_t width,
                               size_t height);

//
// DRIVER API
//

void hipaccCreateModule(CUmodule &module, const void *ptx, int cc);
void hipaccCompileCUDAToModule(CUmodule &module, std::string file_name, int cc,
                               std::vector<std::string> &build_options);
void hipaccGetKernel(CUfunction &function, CUmodule &module,
                     std::string kernel_name);
size_t blockSizeToSmemSize(int blockSize);
void hipaccPrintKernelOccupancy(CUfunction fun, int tile_size_x,
                                int tile_size_y);
void hipaccLaunchKernel(CUfunction &kernel, std::string kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing = true);
void hipaccLaunchKernelBenchmark(CUfunction &kernel, std::string kernel_name,
                                 dim3 &grid, dim3 &block, void **args,
                                 bool print_timing = true);
void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module,
                     std::string global_name);
void hipaccGetTexRef(CUtexref &tex, CUmodule &module, std::string texture_name);
void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module,
                      std::string surface_name);
void hipaccBindTextureDrv(CUtexref &texture, const HipaccImageCuda &img,
                          CUarray_format format, hipaccMemoryType tex_type);
void hipaccBindSurfaceDrv(CUsurfref &surface, const HipaccImageCuda &img);

//
// REDUCTIONS AND BINNING
//

template <typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name,
                       const void *kernel1D, std::string kernel1D_name,
                       const HipaccAccessor &acc, unsigned int max_threads,
                       unsigned int pixels_per_thread,
                       const textureReference *tex);
template <typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name,
                       const void *kernel1D, std::string kernel1D_name,
                       const HipaccImageCuda &img, unsigned int max_threads,
                       unsigned int pixels_per_thread,
                       const textureReference *tex);
template <typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D,
                                  std::string kernel2D_name,
                                  const HipaccAccessor &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  const textureReference *tex);
template <typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D,
                                  std::string kernel2D_name,
                                  const HipaccImageCuda &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  const textureReference *tex);
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccAccessor &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  hipacc_tex_info tex_info, int cc);
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccImageCuda &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  hipacc_tex_info tex_info, int cc);

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 128
#define MAX_SEGMENTS 512 // equals 65k bins (MAX_SEGMENTS*SEGMENT_SIZE)
#endif
template <typename T, typename T2>
T *hipaccApplyBinningSegmented(const void *kernel2D, std::string kernel2D_name,
                               HipaccAccessor &acc, unsigned int num_hists,
                               unsigned int num_warps, unsigned int num_bins,
                               const textureReference *tex);

//
// PYRAMID
//

template <typename T>
HipaccImageCuda hipaccCreatePyramidImage(const HipaccImageCuda &base,
                                         size_t width, size_t height);

template <typename T>
HipaccPyramidCuda hipaccCreatePyramid(const HipaccImageCuda &img, size_t depth);

#include "hipacc_cu.tpp"

#endif // __HIPACC_CU_HPP__
