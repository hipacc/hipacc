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

#if CUDA_VERSION < 10000
#error "CUDA 10.0 or higher required!"
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

inline void checkErrNVRTC(nvrtcResult err, const std::string &name) {
  if (err != NVRTC_SUCCESS) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                "): " + nvrtcGetErrorString(err));
  }
}

// TODO: what is the purpose of this empty class?
class HipaccContext {
public:
  static HipaccContext &getInstance();
};

template<typename T>
class HipaccImageCudaBase {
public:

  using pixel_type = T;

  virtual ~HipaccImageCudaBase() = default;
  virtual pixel_type const*get_device_memory() const = 0;
  virtual pixel_type *get_device_memory() = 0;
  virtual pixel_type const*get_host_memory() const = 0;
  virtual pixel_type *get_host_memory() = 0;
  virtual int get_width() const = 0;
  virtual int get_height() const = 0;
  virtual int get_stride() const = 0;
  virtual int get_alignment() const = 0;
  virtual int get_pixel_size() const = 0;
  virtual hipaccMemoryType get_mem_type() const = 0;
};

template<typename T>
using HipaccImageCuda = std::shared_ptr<HipaccImageCudaBase<T>>;

template<typename T>
class HipaccImageCudaRaw : public HipaccImageCudaBase<T> {
private:

  static constexpr int pixel_size = sizeof(T);

  const int width{};
  const int height{};
  const int stride{};
  const int alignment{};
  const hipaccMemoryType mem_type;
  T *const device_mem{};
  std::unique_ptr<T[]> host_mem;

public:

  using pixel_type = T;

  HipaccImageCudaRaw(int width, int height, int stride,
                     int alignment, pixel_type *mem,
                     hipaccMemoryType mem_type = hipaccMemoryType::Global)
                   : width(width), height(height), stride(stride),
                   alignment(alignment),
                   mem_type(mem_type), device_mem(mem),
                   host_mem(new pixel_type[width * height]) {
    std::fill(host_mem.get(), host_mem.get() + width * height, T{});
  }

  ~HipaccImageCudaRaw() {
    cudaError_t err = cudaFree(device_mem);
    checkErr(err, "cudaFree()");
  }

  bool operator==(const HipaccImageCudaRaw &other) const {
    return device_mem == other.device_mem;
  }

  pixel_type const*get_device_memory() const override final { return device_mem; }
  pixel_type *get_device_memory() final { return device_mem; }
  pixel_type const* get_host_memory() const override final { return host_mem.get(); }
  pixel_type *get_host_memory() final { return host_mem.get(); }
  int get_width() const override final { return width; }
  int get_height() const override final { return height; }
  int get_stride() const override final { return stride; }
  int get_alignment() const override final { return alignment; }
  int get_pixel_size() const override final { return pixel_size; }
  hipaccMemoryType get_mem_type() const override final { return mem_type; }
};

template<typename T>
struct HipaccAccessor : public HipaccAccessorBase {

  HipaccImageCuda<T> img;

  HipaccAccessor(HipaccImageCuda<T> const& img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
      : HipaccAccessorBase(width, height, offset_x, offset_y), img(img) {}

  HipaccAccessor(HipaccImageCuda<T> const& img)
      : HipaccAccessorBase(img->get_width(), img->get_height(), 0, 0), img(img) {}
};

template<typename T>
inline HipaccAccessor<T> hipaccMakeAccessor(HipaccImageCuda<T> const& img)
{
  return HipaccAccessor<T>{ img };
}

template<typename T>
inline HipaccAccessor<T> hipaccMakeAccessor(HipaccImageCuda<T> const& img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
{
  return HipaccAccessor<T>{ img, width, height, offset_x, offset_y };
}

struct hipacc_const_info { // TODO: VOID
  hipacc_const_info(std::string const& name, void *memory, int size)
      : name(name), memory(memory), size(size) {}
  std::string name;
  void *memory;
  int size;
};

template<typename T>
struct hipacc_tex_info {
  hipacc_tex_info(std::string const &name, CUarray_format format,
                  const HipaccImageCuda<T> &image, hipaccMemoryType tex_type)
      : name(name), format(format), image(image), tex_type(tex_type) {}
  std::string name;
  CUarray_format format;
  const HipaccImageCuda<T> &image;
  hipaccMemoryType tex_type;
};

template<typename T>
hipacc_tex_info<T> hipacc_make_tex_info(std::string const &name, CUarray_format format,
                                        const HipaccImageCuda<T> &image, hipaccMemoryType tex_type)
{
  return hipacc_tex_info<T>{ name, format, image, tex_type };
}

template<typename T>
class HipaccPyramidCuda final : public HipaccPyramid {
private:
  std::vector<HipaccImageCuda<T>> imgs_;

public:
  explicit HipaccPyramidCuda(const int depth) : HipaccPyramid(depth) {}

  void add(const HipaccImageCuda<T> &img) { imgs_.push_back(img); }

  HipaccImageCuda<T> operator()(int relative) {
    assert(level_ + relative >= 0 && level_ + relative < (int)imgs_.size() &&
           "Accessed pyramid stage is out of bounds.");
    return imgs_.at(level_ + relative);
  }

  void swap(HipaccPyramidCuda &other) { imgs_.swap(other.imgs_); }
};


class HipaccExecutionParameterCudaBase  {
private:
  cudaStream_t stream_{};
protected:
  void set_stream(cudaStream_t s) { stream_ = s; }
public:
  cudaStream_t get_stream() const { return stream_; }
  virtual void pre_kernel() = 0;
  virtual void post_kernel() = 0;
};

using HipaccExecutionParameterCuda = std::shared_ptr<HipaccExecutionParameterCudaBase>;

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 const&block);
dim3 hipaccCalcGridFromBlock(hipacc_launch_info const&info, dim3 const&block);
void hipaccInitCUDA();
template<typename T>
void hipaccCopyMemory(const HipaccImageCuda<T> &src, HipaccImageCuda<T> &dst);
template<typename T>
void hipaccCopyMemoryRegion(const HipaccAccessor<T> &src,
                            const HipaccAccessor<T> &dst);
void hipaccLaunchKernel(const void *kernel, std::string const& kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing = false, const int shared_memory_size = 0);
template <typename KernelFunction, typename... KernelParameters>
void hipaccLaunchKernel(KernelFunction const &kernel_function, dim3 const &gridDim, dim3 const &blockDim, HipaccExecutionParameterCuda const& ep, bool print_timing, size_t shared_memory, KernelParameters &&... parameters);
template <typename KernelFunction, typename... KernelParameters>
void hipaccLaunchKernelCudaGraph(KernelFunction const &kernel_function, dim3 const &gridDim, dim3 const &blockDim, HipaccExecutionParameterCuda const& ep, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaKernelNodeParams &kernelNodeArgs, size_t shared_memory, KernelParameters &&... parameters);
inline HipaccExecutionParameterCuda hipaccMapExecutionParameter(HipaccExecutionParameterCuda ep) { return ep; }

//
// TEMPLATES
//

template <typename T>
HipaccImageCuda<T>
createImage(T *host_mem, T *mem, size_t width, size_t height, size_t stride,
            size_t alignment,
            hipaccMemoryType mem_type = hipaccMemoryType::Global);
template <typename T> T *createMemory(size_t stride, size_t height);
template <typename T>
HipaccImageCuda<T> hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                   size_t alignment);
template <typename T>
HipaccImageCuda<T> hipaccCreateMemory(T *host_mem, size_t width, size_t height);
template <typename T>
HipaccImageCuda<T> hipaccCreateArray2D(T *host_mem, size_t width, size_t height);
template <typename T, typename TI>
TI hipaccCreatePyramidImage(const TI &base, size_t width, size_t height);
template <typename T> void hipaccWriteMemory(HipaccImageCuda<T> &img, T *host_mem);
template <typename T> void hipaccWriteMemoryCudaGraph(HipaccImageCuda<T> &img, T *host_mem, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaMemcpy3DParms &memcpyArgs);
template <typename T> T *hipaccReadMemory(const HipaccImageCuda<T> &img);
template <typename T> T *hipaccReadMemoryCudaGraph(const HipaccImageCuda<T> &img, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaMemcpy3DParms &memcpyArgs);
template <typename T> HipaccImageCuda<T> hipaccMapMemory(HipaccImageCuda<T> img) { return img; }
template <typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex,
                       const HipaccImageCuda<T> &img);
template <typename T>
void hipaccBindSurface(hipaccMemoryType mem_type, const surfaceReference *surf,
                       const HipaccImageCuda<T> &img);
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
                        dim3 &block, void **args, bool print_timing = true, const int shared_memory_size = 0);
void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module,
                     std::string global_name);
void hipaccGetTexRef(CUtexref &tex, CUmodule &module, std::string texture_name);
void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module,
                      std::string surface_name);

template<typename T>
void hipaccBindTextureDrv(CUtexref &texture, const HipaccImageCuda<T> &img,
                          CUarray_format format, hipaccMemoryType tex_type);

template<typename T>
void hipaccBindSurfaceDrv(CUsurfref &surface, const HipaccImageCuda<T> &img);

//
// REDUCTIONS AND BINNING
//

template <typename T, class KernelFunc>
T hipaccApplyReductionShared(const KernelFunc &kernel2D,
                                  const HipaccAccessor<T> &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  const textureReference *tex);
template <typename T, class KernelFunc>
T hipaccApplyReductionShared(const KernelFunc &kernel2D,
                                  const HipaccImageCuda<T> &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  HipaccExecutionParameterCuda const &ep,
                                  const textureReference *tex);

#ifndef DEFAULT_SEGMENT_SIZE
# define DEFAULT_SEGMENT_SIZE 128
#endif

template <typename T, typename T2, class KernelFunc, int SEGMENT_SIZE=DEFAULT_SEGMENT_SIZE>
T *hipaccApplyBinningSegmented(KernelFunc const &kernel2D,
                                                const HipaccAccessor<T2> &acc,
                                                unsigned int num_warps,
                                                unsigned int num_units,
                                                unsigned int num_bins,
                                                HipaccExecutionParameterCuda const &ep,
                                                const textureReference *tex,
                                                bool print_timing);

//
// PYRAMID
//

template <typename T>
HipaccImageCuda<T> hipaccCreatePyramidImage(const HipaccImageCuda<T> &base,
                                         size_t width, size_t height);

template <typename T>
HipaccPyramidCuda<T> hipaccCreatePyramid(const HipaccImageCuda<T> &img, size_t depth);

#include "hipacc_cu.tpp"

#endif // __HIPACC_CU_HPP__
