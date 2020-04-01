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

// This is the standalone (header-only) Hipacc CPU runtime

#include "hipacc_cpu.hpp"

#ifndef __HIPACC_CPU_STANDALONE_HPP__
#define __HIPACC_CPU_STANDALONE_HPP__

#include "hipacc_base_standalone.hpp"

HipaccKernelTimingBase hipaccCpuTiming;

HipaccContext &HipaccContext::getInstance() {
  static HipaccContext instance;

  return instance;
}

HipaccImageCpuBase::~HipaccImageCpuBase() {}

int64_t start_time = 0;
int64_t end_time = 0;

void hipaccStartTiming() { start_time = hipacc_time_micro(); }

void hipaccStopTiming() {
  end_time = hipacc_time_micro();
  hipaccCpuTiming.set_gpu_timing((end_time - start_time) * 1.0e-3f);

  hipaccRuntimeLogTrivial(
      hipaccRuntimeLogLevel::INFO,
      "<HIPACC:> Kernel timing: " +
          std::to_string(hipaccCpuTiming.get_last_kernel_timing()) + "(ms)");
}

// Copy from memory to memory
void hipaccCopyMemory(const HipaccImageCpu &src, HipaccImageCpu &dst) {
  size_t height = src->get_height();
  size_t stride = src->get_stride();
  std::memcpy(dst->get_aligned_host_memory(), src->get_aligned_host_memory(),
              src->get_pixel_size() * stride * height);
}

// Copy from memory region to memory region
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst) {
  for (size_t i = 0; i < dst.height; ++i) {
    std::memcpy(
        &((uchar *)dst.img->get_aligned_host_memory())
            [dst.offset_x * dst.img->get_pixel_size() +
             (dst.offset_y + i) * dst.img->get_stride() * dst.img->get_pixel_size()],
        &((uchar *)src.img->get_aligned_host_memory())
            [src.offset_x * src.img->get_pixel_size() +
             (src.offset_y + i) * src.img->get_stride() * src.img->get_pixel_size()],
        src.width * src.img->get_pixel_size());
  }
}

#endif // __HIPACC_CPU_STANDALONE_HPP__
