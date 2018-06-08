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

#ifndef __HIPACC_CPU_HPP__
#define __HIPACC_CPU_HPP__

#include <cmath>
#include <cstring>
#include <iostream>

#include "hipacc_base.hpp"

class HipaccContext : public HipaccContextBase {
    public:
        static HipaccContext &getInstance();
};

class HipaccImageCPU : public HipaccImageBase {
    private:
        char *mem;
    public:
        HipaccImageCPU(size_t width, size_t height, size_t stride,
                       size_t alignment, size_t pixel_size, void* mem,
                       hipaccMemoryType mem_type=Global);
        ~HipaccImageCPU();
};

extern long start_time;
extern long end_time;


void hipaccStartTiming();
void hipaccStopTiming();
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst);


template<typename T>
HipaccImage createImage(T *host_mem, void *mem, size_t width, size_t height, size_t stride, size_t alignment, hipaccMemoryType mem_type=Global);
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height, size_t alignment);
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height);
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem);
template<typename T>
T *hipaccReadMemory(const HipaccImage &img);
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem);


#include "hipacc_cpu.tpp"


#endif  // __HIPACC_CPU_HPP__

