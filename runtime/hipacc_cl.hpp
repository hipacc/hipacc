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

#ifndef __HIPACC_CL_HPP__
#define __HIPACC_CL_HPP__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "hipacc_base.hpp"

#define EVENT_TIMING

enum cl_platform_name {
  AMD = 0x1,
  APPLE = 0x2,
  ARM = 0x4,
  INTEL = 0x8,
  NVIDIA = 0x10,
  ALL = (AMD | APPLE | ARM | INTEL | NVIDIA)
};

std::string getOpenCLErrorCodeStr(int error);

#define checkErr(err, name) __checkOpenCLErrors(err, name, __FILE__, __LINE__)

inline void __checkOpenCLErrors(cl_int err, const std::string &name,
                                const std::string &file, const int line);

extern HipaccKernelTimingBase hipaccClTiming;
inline float hipacc_last_kernel_timing() {
  return hipaccClTiming.get_last_kernel_timing();
}

class HipaccContext {
private:
  std::vector<cl_platform_id> platforms;
  std::vector<cl_platform_name> platform_names;
  std::vector<cl_device_id> devices, devices_all;
  std::vector<cl_context> contexts;
  std::vector<cl_command_queue> queues;

public:
  static HipaccContext &getInstance();
  void add_platform(cl_platform_id id, cl_platform_name name);
  void add_device(cl_device_id id);
  void add_device_all(cl_device_id id);
  void add_context(cl_context id);
  void add_command_queue(cl_command_queue id);
  std::vector<cl_platform_id> get_platforms();
  std::vector<cl_platform_name> get_platform_names();
  std::vector<cl_device_id> get_devices();
  std::vector<cl_device_id> get_devices_all();
  std::vector<cl_context> get_contexts();
  std::vector<cl_command_queue> get_command_queues();
};

class HipaccImageOpenCLBase {
public:
  virtual ~HipaccImageOpenCLBase() = 0;
  virtual void *get_device_memory() const = 0;
  virtual void *get_host_memory() const = 0;
  virtual size_t get_width() const = 0;
  virtual size_t get_height() const = 0;
  virtual size_t get_stride() const = 0;
  virtual size_t get_alignment() const = 0;
  virtual size_t get_pixel_size() const = 0;
  virtual hipaccMemoryType get_mem_type() const = 0;
};
typedef std::shared_ptr<HipaccImageOpenCLBase> HipaccImageOpenCL;

class HipaccImageOpenCLRaw : public HipaccImageOpenCLBase {
private:
  size_t width, height;
  size_t stride, alignment;
  size_t pixel_size;
  hipaccMemoryType mem_type;
  cl_mem device_mem;
  std::unique_ptr<char[]> host_mem;

public:
  HipaccImageOpenCLRaw(size_t width, size_t height, size_t stride,
                     size_t alignment, size_t pixel_size, cl_mem mem,
                     hipaccMemoryType mem_type = hipaccMemoryType::Global)
                   : width(width), height(height), stride(stride),
                   alignment(alignment), pixel_size(pixel_size),
                   mem_type(mem_type), device_mem(mem),
                   host_mem(new char[width * height * pixel_size]) {
    std::fill(host_mem.get(), host_mem.get() + width * height * pixel_size, 0);
  }

  ~HipaccImageOpenCLRaw() {
    cl_int err = clReleaseMemObject(device_mem);
    checkErr(err, "clReleaseMemObject()");
  }

  bool operator==(const HipaccImageOpenCLRaw &other) const {
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
  HipaccImageOpenCL img;

public:
  HipaccAccessor(HipaccImageOpenCL img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
      : HipaccAccessorBase(width, height, offset_x, offset_y), img(img) {}
  HipaccAccessor(HipaccImageOpenCL img)
      : HipaccAccessorBase(img->get_width(), img->get_height(), 0, 0), img(img) {}
};

template<typename T>
HipaccAccessor hipaccMakeAccessor(HipaccImageOpenCL const& img)
{
  return HipaccAccessor{ img };
}

template<typename T>
HipaccAccessor hipaccMakeAccessor(HipaccImageOpenCL const& img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
{
  return HipaccAccessor{ img, width, height, offset_x, offset_y };
}

class HipaccPyramidOpenCL : public HipaccPyramid {
private:
  std::vector<HipaccImageOpenCL> imgs_;

public:
  HipaccPyramidOpenCL(const int depth) : HipaccPyramid(depth) {}

  void add(const HipaccImageOpenCL &img) { imgs_.push_back(img); }
  HipaccImageOpenCL operator()(int relative) {
    assert(level_ + relative >= 0 && level_ + relative < (int)imgs_.size() &&
           "Accessed pyramid stage is out of bounds.");
    return imgs_.at(level_ + relative);
  }
  void swap(HipaccPyramidOpenCL &other) { imgs_.swap(other.imgs_); }
};

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block);
void hipaccCalcGridFromBlock(hipacc_launch_info &info, size_t *block,
                             size_t *grid);
void hipaccInitPlatformsAndDevices(cl_device_type dev_type,
                                   cl_platform_name platform_name = ALL);
std::vector<cl_device_id> hipaccGetAllDevices();
void hipaccCreateContextsAndCommandQueues(bool all_devies = false);
void hipaccDumpBinary(cl_program program, cl_device_id device);
cl_kernel
hipaccBuildProgramAndKernel(std::string file_name, std::string kernel_name,
                            bool print_progress = true,
                            bool dump_binary = false, bool print_log = false,
                            std::string build_options = std::string(),
                            std::string build_includes = std::string());
cl_sampler hipaccCreateSampler(cl_bool normalized_coords,
                               cl_addressing_mode addressing_mode,
                               cl_filter_mode filter_mode);
void hipaccCopyMemory(const HipaccImageOpenCL &src, HipaccImageOpenCL &dst,
                      int num_device = 0);
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst, int num_device = 0);
double hipaccCopyBufferBenchmark(const HipaccImageOpenCL &src,
                                 HipaccImageOpenCL &dst, int num_device = 0,
                                 bool print_timing = false);
void hipaccLaunchKernel(cl_kernel kernel, size_t *global_work_size,
                        size_t *local_work_size, bool print_timing = true);
void hipaccLaunchKernelBenchmark(cl_kernel kernel, size_t *global_work_size,
                                 size_t *local_work_size,
                                 std::vector<std::pair<size_t, void *>> &args,
                                 bool print_timing = true);
void hipaccLaunchKernelExploration(std::string filename, std::string kernel,
                                   std::vector<std::pair<size_t, void *>> &args,
                                   std::vector<hipacc_smem_info> &smems,
                                   hipacc_launch_info &info, int warp_size,
                                   int max_threads_per_block,
                                   int max_threads_for_kernel,
                                   int max_smem_per_block, int heu_tx,
                                   int heu_ty);

template <typename T>
HipaccImageOpenCL
createImage(T *host_mem, cl_mem mem, size_t width, size_t height, size_t stride,
            size_t alignment,
            hipaccMemoryType mem_type = hipaccMemoryType::Global);
template <typename T>
cl_mem createBuffer(size_t stride, size_t height, cl_mem_flags flags);
template <typename T>
HipaccImageOpenCL hipaccCreateBuffer(T *host_mem, size_t width, size_t height,
                                     size_t alignment);
template <typename T>
HipaccImageOpenCL hipaccCreateBuffer(T *host_mem, size_t width, size_t height);
template <typename T>
HipaccImageOpenCL hipaccCreateBufferConstant(T *host_mem, size_t width,
                                             size_t height);
template <typename T>
HipaccImageOpenCL hipaccCreateImage(T *host_mem, size_t width, size_t height,
                                    cl_channel_type channel_type,
                                    cl_channel_order channel_order);
template <typename T>
HipaccImageOpenCL hipaccCreateImage(T *host_mem, size_t width, size_t height);
template <typename T>
void hipaccWriteMemory(HipaccImageOpenCL &img, T *host_mem, int num_device = 0);
template <typename T>
T *hipaccReadMemory(const HipaccImageOpenCL &img, int num_device = 0);
template <typename T>
void hipaccWriteDomainFromMask(HipaccImageOpenCL &dom, T *host_mem);
template <typename T>
void hipaccSetKernelArg(cl_kernel kernel, unsigned int num, size_t size,
                        T *param);
template <typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D,
                       const HipaccAccessor &acc, unsigned int max_threads,
                       unsigned int pixels_per_thread);
template <typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D,
                       const HipaccImageOpenCL &img, unsigned int max_threads,
                       unsigned int pixels_per_thread);
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccAccessor &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread);
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccImageOpenCL &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread);

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 128
#endif
template <typename T, typename T2>
T *hipaccApplyBinningSegmented(cl_kernel kernel2D, cl_kernel kernel1D,
                               const HipaccAccessor &acc,
                               unsigned int num_hists, unsigned int num_warps,
                               unsigned int num_bins);
//
// PYRAMID
//

template <typename T>
HipaccImageOpenCL hipaccCreatePyramidImage(const HipaccImageOpenCL &base,
                                           size_t width, size_t height);

template <typename T>
HipaccPyramidOpenCL hipaccCreatePyramid(const HipaccImageOpenCL &img,
                                        size_t depth);

#include "hipacc_cl.tpp"

#endif // __HIPACC_CL_HPP__
