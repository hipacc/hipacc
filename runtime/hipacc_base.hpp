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

#include "hipacc_base.hpp"
#include "hipacc_math_functions.hpp"

#define HIPACC_NUM_ITERATIONS 10

static float total_time = 0.0f;
static float last_gpu_timing = 0.0f;

// get GPU timing of last executed Kernel in ms
float hipaccGetLastKernelTiming() {
    return last_gpu_timing;
}


enum hipaccMemoryType {
    Global,
    Linear1D,
    Linear2D,
    Array2D,
    Surface
};


class HipaccImage {
    public:
        int32_t width, height;
        int32_t stride, alignment;
        int32_t pixel_size;
        void *mem;
        hipaccMemoryType mem_type;

    public:
        HipaccImage(int32_t width, int32_t height, int32_t stride, int32_t
                alignment, int32_t pixel_size, void *mem, hipaccMemoryType
                mem_type=Global) :
            width(width),
            height(height),
            stride(stride),
            alignment(alignment),
            pixel_size(pixel_size),
            mem(mem),
            mem_type(mem_type)
            {}

        bool operator==(HipaccImage other) const {
            return mem==other.mem;
        }
};

class HipaccAccessor {
    public:
        HipaccImage &img;
        int32_t width, height;
        int32_t offset_x, offset_y;

    public:
        HipaccAccessor(HipaccImage &img, int32_t width, int32_t height, int32_t offset_x, int32_t offset_y) :
            img(img),
            width(width),
            height(height),
            offset_x(offset_x),
            offset_y(offset_y) {}

        HipaccAccessor(HipaccImage &img) :
            img(img),
            width(img.width),
            height(img.height),
            offset_x(0),
            offset_y(0) {}
};


class HipaccContextBase {
    protected:
        std::vector<HipaccImage> imgs;

        HipaccContextBase() {};
        HipaccContextBase(HipaccContextBase const &);
        void operator=(HipaccContextBase const &);

    public:
        void add_image(HipaccImage &img) { imgs.push_back(img); }
        void del_image(HipaccImage &img) {
            imgs.erase(std::remove(imgs.begin(), imgs.end(), img), imgs.end());
        }
};


typedef struct hipacc_launch_info {
    hipacc_launch_info(int size_x, int size_y, int is_width, int is_height, int
            offset_x, int offset_y, int pixels_per_thread, int simd_width) :
        size_x(size_x), size_y(size_y), is_width(is_width),
        is_height(is_height), offset_x(offset_x), offset_y(offset_y), pixels_per_thread(pixels_per_thread), simd_width(simd_width),
        bh_start_left(0), bh_start_right(0), bh_start_top(0),
        bh_start_bottom(0), bh_fall_back(0) {}
    hipacc_launch_info(int size_x, int size_y, HipaccAccessor &Acc, int pixels_per_thread, int simd_width) :
        size_x(size_x), size_y(size_y), is_width(Acc.width),
        is_height(Acc.height), offset_x(Acc.offset_x), offset_y(Acc.offset_y),
        pixels_per_thread(pixels_per_thread), simd_width(simd_width),
        bh_start_left(0), bh_start_right(0), bh_start_top(0),
        bh_start_bottom(0), bh_fall_back(0) {}
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
    hipacc_smem_info(int size_x, int size_y, int pixel_size) :
        size_x(size_x), size_y(size_y), pixel_size(pixel_size) {}
    int size_x, size_y;
    int pixel_size;
} hipacc_smem_info;



unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    ++x;

    // get at least the warp size
    if (x < 32) x = 32;

    return x;
}


#endif  // __HIPACC_BASE_HPP__

