//
// Copyright (c) 2014, Saarland University
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

#ifndef __HIPACC_CPU_HPP__
#define __HIPACC_CPU_HPP__

#include <cmath>
#include <cstring>
#include <iostream>

#include "hipacc_base.hpp"


#if (!defined(_WIN32) || !defined(_MSC_VER)) && (!defined(__GNUC__) || __GNUC__ < 9)
#  define _mm256_set_m128(va, vb) _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
#  define _mm256_set_m128i(va, vb) _mm256_insertf128_si256(_mm256_castsi128_si256(vb), va, 1)
#endif

class HipaccContext {
public:
  static HipaccContext &getInstance();
};

class HipaccImageCpuBase {
public:
  virtual ~HipaccImageCpuBase() = 0;
  virtual void *get_aligned_host_memory() const = 0;
  virtual void *get_host_memory() const = 0;
  virtual size_t get_width() const = 0;
  virtual size_t get_height() const = 0;
  virtual size_t get_stride() const = 0;
  virtual size_t get_alignment() const = 0;
  virtual size_t get_pixel_size() const = 0;
  virtual hipaccMemoryType get_mem_type() const = 0;
};
typedef std::shared_ptr<HipaccImageCpuBase> HipaccImageCpu;

class HipaccImageCpuRaw : public HipaccImageCpuBase {
private:
  size_t width, height;
  size_t stride, alignment;
  size_t pixel_size;
  hipaccMemoryType mem_type;
  char *mem{};
  std::unique_ptr<char[]> host_mem;

public:
  HipaccImageCpuRaw(size_t width, size_t height, size_t stride,
                     size_t alignment, size_t pixel_size, void *mem,
                     hipaccMemoryType mem_type = hipaccMemoryType::Global)
                   : width(width), height(height), stride(stride),
                   alignment(alignment), pixel_size(pixel_size),
                   mem_type(mem_type), mem((char *)mem),
                   host_mem(new char[width * height * pixel_size]) {
    std::fill(host_mem.get(), host_mem.get() + width * height * pixel_size, 0);
  }

  ~HipaccImageCpuRaw() { delete[] mem; }

  bool operator==(const HipaccImageCpuRaw &other) const {
    return mem == other.mem;
  }

  void *get_aligned_host_memory() const final { return (void *)mem; }
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
  HipaccImageCpu img;

public:
  HipaccAccessor(HipaccImageCpu img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
      : HipaccAccessorBase(static_cast<int>(width), static_cast<int>(height), offset_x, offset_y), img(img) {}
  HipaccAccessor(HipaccImageCpu img)
      : HipaccAccessorBase(static_cast<int>(img->get_width()), static_cast<int>(img->get_height()), 0, 0), img(img) {}
};

template<typename T>
HipaccAccessor hipaccMakeAccessor(HipaccImageCpu const& img)
{
  return HipaccAccessor{ img };
}

template<typename T>
HipaccAccessor hipaccMakeAccessor(HipaccImageCpu const& img, size_t width, size_t height,
                 int32_t offset_x = 0, int32_t offset_y = 0)
{
  return HipaccAccessor{ img, width, height, offset_x, offset_y };
}

class HipaccPyramidCpu : public HipaccPyramid {
private:
  std::vector<HipaccImageCpu> imgs_;

public:
  HipaccPyramidCpu(const int depth) : HipaccPyramid(depth) {}

  void add(const HipaccImageCpu &img) { imgs_.push_back(img); }
  HipaccImageCpu operator()(int relative) {
    assert(level_ + relative >= 0 && level_ + relative < (int)imgs_.size() &&
           "Accessed pyramid stage is out of bounds.");
    return imgs_.at(level_ + relative);
  }
  void swap(HipaccPyramidCpu &other) { imgs_.swap(other.imgs_); }
};

void hipaccStartTiming();
void hipaccStopTiming();
void hipaccCopyMemory(const HipaccImageCpu &src, HipaccImageCpu &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst);

template <typename T>
HipaccImageCpu
createImage(T *host_mem, void *mem, size_t width, size_t height, size_t stride,
            size_t alignment,
            hipaccMemoryType mem_type = hipaccMemoryType::Global);
template <typename T>
HipaccImageCpu hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                  size_t alignment);
template <typename T>
HipaccImageCpu hipaccCreateMemory(T *host_mem, size_t width, size_t height);
template <typename T> void hipaccWriteMemory(HipaccImageCpu &img, T *host_mem);
template <typename T> T *hipaccReadMemory(const HipaccImageCpu &img);
template <typename T> HipaccImageCpu hipaccMapMemory(HipaccImageCpu img) { return img; }
template <typename T>
void hipaccWriteDomainFromMask(HipaccImageCpu &dom, T *host_mem);

//
// PYRAMID
//

template <typename T>
HipaccImageCpu hipaccCreatePyramidImage(const HipaccImageCpu &base,
                                        size_t width, size_t height);

template <typename T>
HipaccPyramidCpu hipaccCreatePyramid(const HipaccImageCpu &img, size_t depth);


class HipaccExecutionParameterCpuBase  {
public:
  virtual void run_kernel(std::function<void()>) = 0;
  virtual void wait() = 0;
};

using HipaccExecutionParameterCpu = std::shared_ptr<HipaccExecutionParameterCpuBase>;

inline HipaccExecutionParameterCpu hipaccMapExecutionParameter(HipaccExecutionParameterCpu ep) { return ep; }

void hipaccLaunchKernel(std::function<void()> kernel_function, HipaccExecutionParameterCpu const& ep,
                        bool print_timing);

template <typename T>
T hipaccApplyReduction(std::function<T()> reduction_function, HipaccExecutionParameterCpu const& ep,
                       bool print_timing);


#include "hipacc_cpu.tpp"

#endif // __HIPACC_CPU_HPP__
