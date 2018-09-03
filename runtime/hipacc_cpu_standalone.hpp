//
// Copyright (c) 2014, Saarland University
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


// This is the standalone (header-only) Hipacc CPU runtime


#include "hipacc_cpu.hpp"


#ifndef __HIPACC_CPU_STANDALONE_HPP__
#define __HIPACC_CPU_STANDALONE_HPP__


#include "hipacc_base_standalone.hpp"


HipaccContext& HipaccContext::getInstance() {
    static HipaccContext instance;

    return instance;
}

HipaccImageCPU::HipaccImageCPU(size_t width, size_t height, size_t stride,
               size_t alignment, size_t pixel_size, void* mem,
               hipaccMemoryType mem_type)
    : HipaccImageBase(width, height, stride, alignment, pixel_size, mem,
        mem_type), mem((char*)mem) {
}

HipaccImageCPU::~HipaccImageCPU() {
    delete[] mem;
}

long start_time = 0L;
long end_time = 0L;

void hipaccStartTiming() {
    start_time = hipacc_time_micro();
}

void hipaccStopTiming() {
    end_time = hipacc_time_micro();
    last_gpu_timing = (end_time - start_time) * 1.0e-3f;

    std::cerr << "<HIPACC:> Kernel timing: "
              << last_gpu_timing << "(ms)" << std::endl;
}


// Copy from memory to memory
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst) {
    size_t height = src->height;
    size_t stride = src->stride;
    std::memcpy(dst->mem, src->mem, src->pixel_size*stride*height);
}


// Copy from memory region to memory region
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst) {
    for (size_t i=0; i<dst.height; ++i) {
        std::memcpy(&((uchar*)dst.img->mem)[dst.offset_x*dst.img->pixel_size + (dst.offset_y + i)*dst.img->stride*dst.img->pixel_size],
                    &((uchar*)src.img->mem)[src.offset_x*src.img->pixel_size + (src.offset_y + i)*src.img->stride*src.img->pixel_size],
                    src.width*src.img->pixel_size);
    }
}


#endif  // __HIPACC_CPU_STANDALONE_HPP__

