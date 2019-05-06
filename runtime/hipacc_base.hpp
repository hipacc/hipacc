//
// Copyright (c) 2013, University of Erlangen-Nuremberg
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

#ifndef __HIPACC_BASE_HPP__
#define __HIPACC_BASE_HPP__

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "hipacc_math_functions.hpp"

#define HIPACC_NUM_ITERATIONS 10

#ifdef _WIN32
# define setenv(a,b,c) _putenv_s(a,b)
#endif

extern float hipacc_last_timing;
float hipacc_last_kernel_timing();
int64_t hipacc_time_micro();

enum hipaccMemoryType {
    Global,
    Linear1D,
    Linear2D,
    Array2D,
    Surface
};

class HipaccImageBase {
    public:
        size_t width, height;
        size_t stride, alignment;
        size_t pixel_size;
        void *mem;
        hipaccMemoryType mem_type;
        char *host;

    public:
        HipaccImageBase(size_t width, size_t height, size_t stride,
                    size_t alignment, size_t pixel_size, void *mem,
                    hipaccMemoryType mem_type=Global);

        ~HipaccImageBase();

        bool operator==(const HipaccImageBase &other) const;
};

typedef std::shared_ptr<HipaccImageBase> HipaccImage;

class HipaccAccessor {
    public:
        HipaccImage img;
        size_t width, height;
        int32_t offset_x, offset_y;

    public:
        HipaccAccessor(HipaccImage img, size_t width, size_t height, int32_t offset_x=0, int32_t offset_y=0);

        HipaccAccessor(HipaccImage img);
};


class HipaccContextBase {
    protected:
        HipaccContextBase();
        HipaccContextBase(HipaccContextBase const &);
        void operator=(HipaccContextBase const &);
};


typedef struct hipacc_launch_info {
    hipacc_launch_info(int size_x, int size_y, int is_width, int is_height, int
            offset_x, int offset_y, int pixels_per_thread, int simd_width);
    hipacc_launch_info(int size_x, int size_y, HipaccAccessor &Acc, int pixels_per_thread, int simd_width);
    int size_x, size_y;
    int is_width, is_height;
    int offset_x, offset_y;
    int pixels_per_thread, simd_width;
    // calculated later on
    int bh_start_left, bh_start_right;
    int bh_start_top, bh_start_bottom;
    int bh_fall_back;
} hipacc_launch_info;


typedef struct hipacc_smem_info {
    hipacc_smem_info(int size_x, int size_y, int pixel_size);
    int size_x, size_y;
    int pixel_size;
} hipacc_smem_info;


class HipaccPyramid {
  public:
    const int depth_;
    int level_;
    std::vector<HipaccImage> imgs_;
    bool bound_;

  public:
    HipaccPyramid(const int depth);
    void add(const HipaccImage &img);
    HipaccImage &operator()(int relative);
    int depth() const;
    int level() const;
    bool is_top_level() const;
    bool is_bottom_level() const;
    void swap(HipaccPyramid &other);
    bool bind();
    void unbind();
};


void hipaccTraverse(HipaccPyramid &p0, const std::function<void()> func);
void hipaccTraverse(HipaccPyramid &p0, HipaccPyramid &p1,
                    const std::function<void()> func);
void hipaccTraverse(HipaccPyramid &p0, HipaccPyramid &p1, HipaccPyramid &p2,
                    const std::function<void()> func);
void hipaccTraverse(HipaccPyramid &p0, HipaccPyramid &p1, HipaccPyramid &p2,
                    HipaccPyramid &p3, const std::function<void()> func);
void hipaccTraverse(HipaccPyramid &p0, HipaccPyramid &p1, HipaccPyramid &p2,
                    HipaccPyramid &p3, HipaccPyramid &p4,
                    const std::function<void()> func);
void hipaccTraverse(std::vector<HipaccPyramid*> pyrs,
                    const std::function<void()> func);
void hipaccTraverse(unsigned int loop=1,
                    const std::function<void()> func=[]{});


// templates
template<typename data_t>
HipaccPyramid hipaccCreatePyramid(const HipaccImage &img, size_t depth);


// forward declarations
template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height);


#include "hipacc_base.tpp"


#endif // __HIPACC_BASE_HPP__

