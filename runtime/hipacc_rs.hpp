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

#ifndef __HIPACC_RS_HPP__
#define __HIPACC_RS_HPP__

#include <RenderScript.h>
#include <Type.h>
#include <Allocation.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <algorithm>

#include "hipacc_types.hpp"
#include "hipacc_math_functions.hpp"

using namespace android;
using namespace android::renderscriptCpp;

#define HIPACC_NUM_ITERATIONS 10

static float total_time = 0.0f;
static float last_gpu_timing = 0.0f;

enum hipaccBoundaryMode {
    BOUNDARY_UNDEFINED,
    BOUNDARY_CLAMP,
    BOUNDARY_REPEAT,
    BOUNDARY_MIRROR,
    BOUNDARY_CONSTANT
};

typedef struct hipacc_smem_info {
    hipacc_smem_info(int size_x, int size_y, int pixel_size) :
        size_x(size_x), size_y(size_y), pixel_size(pixel_size) {}
    int size_x, size_y;
    int pixel_size;
} hipacc_smem_info;

typedef struct hipacc_launch_info {
    hipacc_launch_info(int size_x, int size_y, int is_width, int is_height, int
            offset_x, int offset_y, int pixels_per_thread, int simd_width) :
        size_x(size_x), size_y(size_y), is_width(is_width),
        is_height(is_height), offset_x(offset_x), offset_y(offset_y),
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


template<typename F>
class hipacc_script_arg {
  private:
    int id;
    void* valptr;
    void(F::*memptr)();

  public:
    int getId() { return id; }

#define CREATE_SCRIPT_ARG(ID, T) \
    hipacc_script_arg(void(F::*setter)(T), T const* arg) \
        : id(ID), memptr((void(F::*)())setter), valptr((void*)arg) {} \
    std::pair<void(F::*)(T), T*> get ## ID() { \
        return std::make_pair((void(F::*)(T))memptr, (T*)valptr); \
    }

    CREATE_SCRIPT_ARG(0, uint8_t)
    CREATE_SCRIPT_ARG(1, uint16_t)
    CREATE_SCRIPT_ARG(2, uint32_t)
    CREATE_SCRIPT_ARG(3, uint64_t)

    CREATE_SCRIPT_ARG(4, int8_t)
    CREATE_SCRIPT_ARG(5, int16_t)
    CREATE_SCRIPT_ARG(6, int32_t)
    CREATE_SCRIPT_ARG(7, int64_t)

    CREATE_SCRIPT_ARG(8, bool)
    CREATE_SCRIPT_ARG(9, char)
    CREATE_SCRIPT_ARG(10, float)
    CREATE_SCRIPT_ARG(11, double)

    CREATE_SCRIPT_ARG(12, uchar4)
    CREATE_SCRIPT_ARG(13, ushort4)
    CREATE_SCRIPT_ARG(14, uint4)

    CREATE_SCRIPT_ARG(15, char4)
    CREATE_SCRIPT_ARG(16, short4)
    CREATE_SCRIPT_ARG(17, int4)

    CREATE_SCRIPT_ARG(18, float4)
    CREATE_SCRIPT_ARG(19, double4)

    CREATE_SCRIPT_ARG(20, sp<Allocation>)
};


#define SET_SCRIPT_ARG_ID(SCRIPT, ARG, ID) \
    hipaccSetScriptArg(SCRIPT, \
                       ARG.get ## ID().first, \
                       *(ARG.get ## ID().second));

#define SET_SCRIPT_ARG(SCRIPT, ARG) \
    switch (ARG.getId()) { \
       case 0: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 0) break; \
       case 1: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 1) break; \
       case 2: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 2) break; \
       case 3: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 3) break; \
       case 4: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 4) break; \
       case 5: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 5) break; \
       case 6: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 6) break; \
       case 7: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 7) break; \
       case 8: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 8) break; \
       case 9: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 9) break; \
       case 10: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 10) break; \
       case 11: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 11) break; \
       case 12: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 12) break; \
       case 13: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 13) break; \
       case 14: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 14) break; \
       case 15: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 15) break; \
       case 16: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 16) break; \
       case 17: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 17) break; \
       case 18: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 18) break; \
       case 19: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 19) break; \
       case 20: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 20) break; \
    }


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block) {
    // calculate item id of a) first work item that requires no border handling
    // (left, top) and b) first work item that requires border handling (right,
    // bottom)
    if (info.size_x > 0) {
        info.bh_start_left = (int)ceil((float)(info.offset_x + info.size_x)
                                       / (block[0] * info.simd_width))
                                  * block[0];
        info.bh_start_right =
            (int)floor((float)(info.offset_x + info.is_width - info.size_x)
                       / (block[0] * info.simd_width)) * block[0];
    } else {
        info.bh_start_left = 0;
        info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width)
                                         / (block[0] * info.simd_width))
                                   * block[0];
    }
    if (info.size_y > 0) {
        info.bh_start_top = (int)ceil((float)(info.size_y)
                                      / (info.pixels_per_thread * block[1]))
                                 * block[1];
        info.bh_start_bottom = (int)floor((float)(info.is_height - info.size_y)
                                          / (block[1] * info.pixels_per_thread))
                                    * block[1];
    } else {
        info.bh_start_top = 0;
        info.bh_start_bottom = (int)floor((float)(info.is_height)
                                         / (block[1] * info.pixels_per_thread))
                                    * block[1];
    }

    if ((info.bh_start_right - info.bh_start_left) > 1 &&
        (info.bh_start_bottom - info.bh_start_top) > 1) {
        info.bh_fall_back = 0;
    } else {
        info.bh_fall_back = 1;
    }
}


long getNanoTime() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec*1000000000LL + now.tv_nsec;
}


const char *getRSErrorCodeStr(int errorNum) {
    switch (errorNum) {
        case RS_ERROR_NONE:
            return "RS_ERROR_NONE";
        case RS_ERROR_BAD_SHADER:
            return "RS_ERROR_BAD_SHADER";
        case RS_ERROR_BAD_SCRIPT:
            return "RS_ERROR_BAD_SCRIPT";
        case RS_ERROR_BAD_VALUE:
            return "RS_ERROR_BAD_VALUE";
        case RS_ERROR_OUT_OF_MEMORY:
            return "RS_ERROR_OUT_OF_MEMORY";
        case RS_ERROR_DRIVER:
            return "RS_ERROR_DRIVER";
        case RS_ERROR_FATAL_UNKNOWN:
            return "RS_ERROR_FATAL_UNKNOWN";
        case RS_ERROR_FATAL_DRIVER:
            return "RS_ERROR_FATAL_DRIVER";
        case RS_ERROR_FATAL_PROGRAM_LINK:
            return "RS_ERROR_FATAL_PROGRAM_LINK";
        default:
            return "unknown error code";
    }
}


RenderScript::ErrorHandlerFunc_t errorHandler(uint32_t errorNum,
                                              const char *errorText) {
    std::cerr << "ERROR: " << getRSErrorCodeStr(errorNum)
              << " (" << errorNum << ")" << std::endl
              << "    " << errorText << std::endl;
}


class HipaccContext {
    public:
        typedef struct {
            int width;
            int height;
            int stride;
            int alignment;
            int pixel_size;
        } rs_dims;

    private:
        RenderScript context;
        std::vector<std::pair<sp<Allocation>, rs_dims> > mems;

        HipaccContext() {};
        HipaccContext(HipaccContext const &);
        void operator=(HipaccContext const &);

    public:
        static HipaccContext &getInstance() {
            static HipaccContext instance;

            return instance;
        }
        void add_memory(sp<Allocation> id, rs_dims dim) {
            mems.push_back(std::make_pair(id, dim));
        }
        void del_memory(sp<Allocation> id) {
            unsigned int num=0;
            std::vector<std::pair<sp<Allocation>, rs_dims> >::const_iterator i;
            for (i=mems.begin(); i!=mems.end(); ++i, ++num) {
                if (i->first == id) {
                    mems.erase(mems.begin() + num);
                    return;
                }
            }

            std::cerr << "ERROR: Unknown Allocation requested: "
                      << id.get() << std::endl;
            exit(EXIT_FAILURE);
        }
        RenderScript* get_context() { return &context; }
        rs_dims get_mem_dims(sp<Allocation> id) {
            std::vector<std::pair<sp<Allocation>, rs_dims> >::const_iterator i;
            for (i=mems.begin(); i!=mems.end(); ++i) {
                if (i->first == id) return i->second;
            }

            std::cerr << "ERROR: Unknown Allocation requested: "
                      << id.get() << std::endl;
            exit(EXIT_FAILURE);
        }
};


// Get GPU timing of last executed Kernel in ms
float hipaccGetLastKernelTiming() {
    return last_gpu_timing;
}


// Create RenderScript context
void hipaccInitRenderScript(int targetAPI) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    RenderScript* rs = Ctx.get_context();

    rs->setErrorHandler((RenderScript::ErrorHandlerFunc_t)&errorHandler);

    // Create context
    if (!rs->init(targetAPI)) {
        std::cerr << "ERROR: RenderScript initialization failed for targetAPI: "
                  << targetAPI << std::endl;
    }
}

template<typename T>
T hipaccInitScript() {
    std::string cache_path = "/sdcard";
    return T(HipaccContext::getInstance().get_context(), cache_path.c_str(),
            cache_path.length());
}


// Write to allocation
template<typename T>
void hipaccWriteAllocation(sp<Allocation> allocation, T *host_mem) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::rs_dims dim = Ctx.get_mem_dims(allocation);

    int width = dim.width;
    int height = dim.height;
    int stride = dim.stride;

    if (stride > width) {
        T* buff = new T[stride * height];
        for (int i = 0; i < height; i++) {
            memcpy(buff + (i * stride), host_mem + (i * width),
                   sizeof(T) * width);
        }
        allocation->copyFromUnchecked(buff, sizeof(T) * stride * height);
        delete[] buff;
    } else {
        allocation->copyFromUnchecked(host_mem, sizeof(T) * width * height);
    }
}


// Read from allocation
template<typename T>
void hipaccReadAllocation(T *host_mem, sp<Allocation> allocation) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::rs_dims dim = Ctx.get_mem_dims(allocation);

    int width = dim.width;
    int height = dim.height;
    int stride = dim.stride;

    if (stride > width) {
        T* buff = new T[stride * height];
        allocation->copyToUnchecked(buff, sizeof(T) * stride * height);
        for (int i = 0; i < height; i++) {
            memcpy(host_mem + (i * width), buff + (i * stride),
                   sizeof(T) * width);
        }
        delete[] buff;
    } else {
        allocation->copyToUnchecked(host_mem, sizeof(T) * width * height);
    }
}


// Copy from allocation to allocation
void hipaccCopyAllocation(sp<Allocation> src, sp<Allocation> dst) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::rs_dims src_dim = Ctx.get_mem_dims(src);
    HipaccContext::rs_dims dst_dim = Ctx.get_mem_dims(dst);

    assert(src_dim.width == dst_dim.width && src_dim.height == dst_dim.height &&
           src_dim.pixel_size == dst_dim.pixel_size && "Invalid CopyAllocation!");

    int height = src_dim.height;
    int stride = src_dim.stride;

    dst->copy1DRangeFrom(0, stride*height, src, 0);
}


// Copy from allocation region to allocation region
void hipaccCopyAllocationRegion(sp<Allocation> src, sp<Allocation> dst, int src_offset_x, int src_offset_y, int dst_offset_x, int dst_offset_y, int roi_width, int roi_height) {
    HipaccContext &Ctx = HipaccContext::getInstance();

    dst->copy2DRangeFrom(dst_offset_x, dst_offset_y, roi_width, roi_height, src, roi_width*roi_height, src_offset_x, src_offset_y);
}


#define CREATE_ALLOCATION(T, E) \
/* Allocate memory with alignment specified */ \
sp<Allocation> hipaccCreateAllocation(T *host_mem, int width, int height, \
                                      int *stride, int alignment) { \
    HipaccContext &Ctx = HipaccContext::getInstance(); \
    RenderScript* rs = Ctx.get_context(); \
\
    *stride = (int)ceil((float)(width) / (alignment / sizeof(T))) \
                   * (alignment / sizeof(T)); \
\
    Type::Builder type(rs, E); \
    type.setX(*stride); \
    type.setY(height); \
\
    sp<Allocation> allocation = Allocation::createTyped(rs, type.create()); \
    HipaccContext::rs_dims dim = { width, height, *stride, \
                                   alignment, sizeof(T) }; \
    Ctx.add_memory(allocation, dim); \
\
    if (host_mem) { \
        hipaccWriteAllocation(allocation, host_mem); \
    } \
\
    return allocation; \
} \
\
/* Allocate memory without any alignment considerations */ \
sp<Allocation> hipaccCreateAllocation(T *host_mem, int width, int height, \
                                      int *stride) { \
    HipaccContext &Ctx = HipaccContext::getInstance(); \
    RenderScript* rs = Ctx.get_context(); \
\
    *stride = width; \
\
    Type::Builder type(rs, E); \
    type.setX(*stride); \
    type.setY(height); \
\
    sp<Allocation> allocation = Allocation::createTyped(rs, type.create()); \
    HipaccContext::rs_dims dim = { width, height, width, 0, sizeof(T) }; \
    Ctx.add_memory(allocation, dim); \
\
    if (host_mem) { \
        hipaccWriteAllocation(allocation, host_mem); \
    } \
\
    return allocation; \
} \
\
/* Allocate memory in GPU constant memory space */ \
sp<Allocation> hipaccCreateAllocationConstant(T *host_mem, \
                                              int width, int height) { \
    HipaccContext &Ctx = HipaccContext::getInstance(); \
    RenderScript* rs = Ctx.get_context(); \
\
    Type::Builder type(rs, E); \
    type.setX(width); \
    type.setY(height); \
\
    sp<Allocation> allocation = \
        Allocation::createTyped(rs, type.create(), \
                                RS_ALLOCATION_USAGE_GRAPHICS_CONSTANTS); \
    HipaccContext::rs_dims dim = { width, height, width, 0, sizeof(T) }; \
    Ctx.add_memory(allocation, dim); \
\
    if (host_mem) { \
        hipaccWriteAllocation(allocation, host_mem); \
    } \
\
    return allocation; \
}


CREATE_ALLOCATION(uint8_t,  Element::U8(rs))
CREATE_ALLOCATION(uint16_t, Element::U16(rs))
CREATE_ALLOCATION(uint32_t, Element::U32(rs))
CREATE_ALLOCATION(uint64_t, Element::U64(rs))

CREATE_ALLOCATION(int8_t,   Element::I8(rs))
CREATE_ALLOCATION(int16_t,  Element::I16(rs))
CREATE_ALLOCATION(int32_t,  Element::I32(rs))
CREATE_ALLOCATION(int64_t,  Element::I64(rs))

CREATE_ALLOCATION(bool,     Element::BOOLEAN(rs))
CREATE_ALLOCATION(char,     Element::U8(rs))
CREATE_ALLOCATION(float,    Element::F32(rs))
CREATE_ALLOCATION(double,   Element::F64(rs))

CREATE_ALLOCATION(uchar4,   Element::U8_4(rs))
CREATE_ALLOCATION(ushort4,  Element::U16_4(rs))
CREATE_ALLOCATION(uint4,    Element::U32_4(rs))
//CREATE_ALLOCATION(ulong4,   Element::U64_4(rs))

CREATE_ALLOCATION(char4,    Element::I8_4(rs))
CREATE_ALLOCATION(short4,   Element::I16_4(rs))
CREATE_ALLOCATION(int4,     Element::I32_4(rs))
//CREATE_ALLOCATION(long4,    Element::I64_4(rs))

CREATE_ALLOCATION(float4,   Element::F32_4(rs))
CREATE_ALLOCATION(double4,  Element::F64_4(rs))


// Destroy allocation
//void hipaccDestroyAllocation(sp<Allocation> mem) {
    // TODO: Clarify proper removal of allocations
//}


// Set a single argument of script
template<typename F, typename T>
void hipaccSetScriptArg(F* script, void(F::*setter)(T), T param) {
    (script->*setter)(param);
}


// Launch script kernel (one allocation)
template<typename F>
void hipaccLaunchScriptKernel(
    F* script,
    void(F::*kernel)(sp<const Allocation>) const,
    sp<Allocation>& out, size_t *work_size, bool print_timing=true
) {
    long end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();
    RenderScript* rs = Ctx.get_context();

    rs->finish();
    start = getNanoTime();
    (script->*kernel)(out);
    rs->finish();
    end = getNanoTime();

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << (end - start) * 1.0e-6f << "(ms)" << std::endl;
    }
    total_time += (end - start) * 1.0e-6f;
    last_gpu_timing = (end - start) * 1.0e-6f;
}


// Launch script kernel (two allocations)
template<typename F>
void hipaccLaunchScriptKernel(
    F* script,
    void(F::*kernel)(sp<const Allocation>, sp<const Allocation>) const,
    sp<Allocation>& in, sp<Allocation>& out, size_t *work_size,
    bool print_timing=true
) {
    long end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();
    RenderScript* rs = Ctx.get_context();

    rs->finish();
    start = getNanoTime();
    (script->*kernel)(in, out);
    rs->finish();
    end = getNanoTime();

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << (end - start) * 1.0e-6f << "(ms)" << std::endl;
    }
    total_time += (end - start) * 1.0e-6f;
    last_gpu_timing = (end - start) * 1.0e-6f;
}


// Benchmark timing for a kernel call
template<typename F>
void hipaccLaunchScriptKernelBenchmark(
    F* script,
    std::vector<hipacc_script_arg<F> > args,
    void(F::*kernel)(sp<const Allocation>) const,
    sp<Allocation>& out, size_t *work_size,
    bool print_timing=true
) {
    float med_dt;
    std::vector<float> times;

    for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
        // set kernel arguments
        for (unsigned int i=0; i<args.size(); i++) {
            SET_SCRIPT_ARG(script, args.data()[i]);
        }

        // launch kernel
        hipaccLaunchScriptKernel(script, kernel, out, work_size, print_timing);
        times.push_back(last_gpu_timing);
    }
    std::sort(times.begin(), times.end());
    med_dt = times.at(HIPACC_NUM_ITERATIONS/2);

    last_gpu_timing = med_dt;
}


// Perform configuration exploration for a kernel call
template<typename F, typename T>
void hipaccLaunchScriptKernelExploration(
    F* script,
    std::vector<hipacc_script_arg<F> > args,
    void(F::*kernel)(sp<const Allocation>) const,
    std::vector<hipacc_smem_info> smems, hipacc_launch_info &info,
    int warp_size, int max_threads_per_block, int max_threads_for_kernel,
    int max_smem_per_block, int opt_tx, int opt_ty,
    sp<Allocation> iter_space_fs
) {
    std::cerr << "<HIPACC:> Exploring configurations for kernel"
              << " '" << kernel << "':" << std::endl;

    for (int curr_warp_size = 1; curr_warp_size <= (int)ceil((float)info.is_width/3);
         curr_warp_size += (curr_warp_size < warp_size ? 1 : warp_size)) {
        // check if we exceed maximum number of threads
        if (curr_warp_size > max_threads_for_kernel) continue;

        size_t work_size[2];
        work_size[0] = curr_warp_size;
        work_size[1] = 1;
        size_t global_work_size[2];
        sp<Allocation> iter_space;

        hipaccPrepareKernelLaunch(info, work_size);

        float med_dt;
        std::vector<float> times;
        for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
            for (unsigned int i=0; i<args.size(); i++) {
                SET_SCRIPT_ARG(script, args.data()[i]);
            }

            // start timing
            total_time = 0.0f;

            // launch kernel
            hipaccLaunchScriptKernel(script, kernel,
                    (iter_space_fs.get() ? iter_space_fs : iter_space),
                    work_size, false);

            // stop timing
            times.push_back(total_time);
        }
        std::sort(times.begin(), times.end());
        med_dt = times.at(HIPACC_NUM_ITERATIONS/2);

        // print timing
        std::cerr << "<HIPACC:> Kernel config: "
                  << std::setw(4) << std::right << work_size[0] << "x"
                  << std::setw(2) << std::left << work_size[1]
                  << std::setw(5-floor(log10(work_size[0]*work_size[1])))
                  << std::right << "(" << work_size[0]*work_size[1] << "): "
                  << std::setw(8) << std::fixed << std::setprecision(4)
                  << med_dt << " ms" << std::endl;
    }
}


// Perform global reduction and return result
template<typename F, typename T>
T hipaccApplyReduction(
    F *script,
    void(F::*kernel2D)(sp<const Allocation>) const,
    void(F::*kernel1D)(sp<const Allocation>) const,
    void(F::*setter)(sp<Allocation>),
    std::vector<hipacc_script_arg<F> > args,
    sp<Allocation>& input, int is_width, bool print_timing=true
) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    RenderScript* rs = Ctx.get_context();
    long end, start;
    sp<Allocation> output;      // memory for reduction
    T result;                   // host result

    // allocate temporary memory
    int stride_output;
    output = hipaccCreateAllocation((T*)NULL, is_width, 1, &stride_output);
    (script->*setter)(output);

    // allocation for 1st reduction step
    sp<Allocation> is1 = hipaccCreateAllocation((T*)NULL, is_width, 1, &stride_output);
    // allocation for 2nd reduction step
    sp<Allocation> is2 = hipaccCreateAllocation((T*)NULL, 1, 1, &stride_output);

    // set arguments
    for (unsigned int i=0; i<args.size(); i++) {
        SET_SCRIPT_ARG(script, args.data()[i]);
    }

    rs->finish();
    start = getNanoTime();
    (script->*kernel2D)(is1);   // first step: reduce image (region) into linear memory
    (script->*kernel1D)(is2);   // second step: reduce linear memory
    rs->finish();
    end = getNanoTime();
    last_gpu_timing = (end - start) * 1.0e-6f;

    if (print_timing) {
        std::cerr << "<HIPACC:> Reduction timing: "
                  << (end - start) * 1.0e-6f << "(ms)" << std::endl;
    }

    // download result of reduction
    // FIXME: the current Renderscript implementation copies the whole array
    // regardless of the size specified by the user
    #if 0
    output->copyTo(&result, sizeof(T));
    #else
    T* buff = new T[is_width];
    output->copyTo(buff, sizeof(T)*is_width);
    result = buff[0];
    delete[] buff;
    #endif

    return result;
}

#endif  // __HIPACC_RS_HPP__

