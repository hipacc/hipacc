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

using namespace android;

#ifndef RS_TARGET_API
  // RS_TARGET_API now defines development kit platform version. If not
  // specified assume highest currently available version, which is also the
  // only one supporting Renderscript in Android NDK. In the future, this define
  // will be removed completely.
# define RS_TARGET_API 19
#endif

#ifdef RS_TARGET_API // RS_TARGET_API
# if RS_TARGET_API < 16
#   error Renderscript target API < 16 is not supported!
# elif RS_TARGET_API < 18
#   include <Type.h>
#   include <Allocation.h>
    using namespace android::renderscriptCpp;
    // Namespace for ErrorHandlerFunc_t
#   define EHF android::renderscriptCpp::RenderScript
    // Abstraction for type name
#   define RS RenderScript
    // Abstraction for type pointer
#   define PRS RS*
    // Abstraction for functions
#   define INIT(rs, target) rs->init(target)
#   define COPYTO(T, src, offset, count, buf) \
      (src)->copyToUnchecked(buf, sizeof(T) * count);
#   define COPYFROM(T, dst, offset, count, buf) \
      (dst)->copyFromUnchecked(buf, sizeof(T) * count);
#   define COPYFROM2D(dst, dx, dy, width, height, src, count, sx, sy) \
      (dst)->copy2DRangeFrom(dx, dy, width, height, src, count, sx, sy);
    // Function signature for forEach() kernel functions
#   define KERNEL1(type, name) \
      void(type::*name)(sp<const Allocation>) const
#   define KERNEL2(type, name) \
      void(type::*name)(sp<const Allocation>, sp<const Allocation>) const
# elif RS_TARGET_API < 20
    using namespace android::RSC;
    // Namespace for ErrorHandlerFunc_t
#   define EHF android::RSC
    // Abstraction for type encapsulated by strong pointer
#   define PRS sp<RS>
    // Abstraction for functions
#   define INIT(rs, target) rs->init()
#   define COPYTO(T, src, offset, count, buf) \
      (src)->copy1DRangeTo(offset, count, buf);
#   define COPYFROM(T, dst, offset, count, buf) \
      (dst)->copy1DRangeFrom(offset, count, buf);
#   define COPYFROM2D(dst, dx, dy, width, height, src, count, sx, sy) \
      (dst)->copy2DRangeFrom(dx, dy, width, height, src, sx, sy);
    // Function signature for forEach() kernel functions
#   define KERNEL1(type, name) \
      void(type::*name)(sp<const Allocation>)
#   define KERNEL2(type, name) \
      void(type::*name)(sp<const Allocation>, sp<const Allocation>)
# else // RS_TARGET_API > 19
#   error Renderscript target API > 19 is not supported!
# endif // RS_TARGET_API > 19
#endif // RS_TARGET_API

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

const sp<Allocation> *hipaccGetAllocation(HipaccImage &img);
void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block);
long getMicroTime();
std::string getRSErrorCodeStr(int errorNum);
EHF::ErrorHandlerFunc_t errorHandler(uint32_t errorNum, std::string errorText);
void hipaccInitRenderScript(int targetAPI);
void hipaccCopyMemory(HipaccImage &src, HipaccImage &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst);
void hipaccReleaseMemory(HipaccImage &img);
#define CREATE_ALLOCATION_DECL(T) \
  HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height, size_t alignment); \
  HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height);
CREATE_ALLOCATION_DECL(uint8_t)
CREATE_ALLOCATION_DECL(uint16_t)
CREATE_ALLOCATION_DECL(uint32_t)
CREATE_ALLOCATION_DECL(uint64_t)
CREATE_ALLOCATION_DECL(int8_t)
CREATE_ALLOCATION_DECL(int16_t)
CREATE_ALLOCATION_DECL(int32_t)
CREATE_ALLOCATION_DECL(int64_t)
CREATE_ALLOCATION_DECL(bool)
CREATE_ALLOCATION_DECL(char)
CREATE_ALLOCATION_DECL(float)
CREATE_ALLOCATION_DECL(double)
CREATE_ALLOCATION_DECL(uchar4)
CREATE_ALLOCATION_DECL(ushort4)
CREATE_ALLOCATION_DECL(uint4)
CREATE_ALLOCATION_DECL(char4)
CREATE_ALLOCATION_DECL(short4)
CREATE_ALLOCATION_DECL(int4)
CREATE_ALLOCATION_DECL(float4)
CREATE_ALLOCATION_DECL(double4)

class HipaccContext : public HipaccContextBase {
    private:
        PRS context;
        std::list<std::pair<sp<Allocation>, HipaccImage> > allocs;

        HipaccContext() {
          context = new RS();
        }

    public:
        static HipaccContext &getInstance() {
            static HipaccContext instance;

            return instance;
        }
        void add_image(HipaccImage &img, sp<Allocation> id) {
            HipaccContextBase::add_image(img);
            allocs.push_back(std::make_pair(id, img));
        }
        void del_image(HipaccImage &img) {
            size_t num=0;
            std::list<std::pair<sp<Allocation>, HipaccImage> >::iterator i;
            for (i=allocs.begin(); i!=allocs.end(); ++i, ++num) {
                if (i->second == img) {
                    allocs.erase(i);
                    HipaccContextBase::del_image(img);
                    return;
                }
            }

            std::cerr << "ERROR: Unknown Allocation requested: "
                      << img.mem << std::endl;
            exit(EXIT_FAILURE);
        }
        const sp<Allocation> *get_allocation(HipaccImage &img) {
            std::list<std::pair<sp<Allocation>, HipaccImage> >::const_iterator i;
            for (i=allocs.begin(); i!=allocs.end(); ++i) {
                if (i->second == img) {
                    return &i->first;
                }
            }
            exit(EXIT_FAILURE);
        }
        PRS get_context() { return context; }
};


template<typename F>
class hipacc_script_arg {
  private:
    int id;
    void *valptr;
    void(F::*memptr)();

  public:
    int getId() const { return id; }

#define CREATE_SCRIPT_ARG(ID, T) \
    hipacc_script_arg(void(F::*setter)(T), T const *arg) \
        : id(ID), memptr((void(F::*)())setter), valptr((void*)arg) {} \
    std::pair<void(F::*)(T), T*> get ## ID() const { \
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

    hipacc_script_arg(void(F::*setter)(sp<Allocation>),
                      sp<Allocation> const *arg)
        : id(20), memptr((void(F::*)())setter), valptr((void*)arg) {}
    std::pair<void(F::*)(sp<Allocation>), sp<Allocation>*> get20() const {
      return std::make_pair((void(F::*)(sp<Allocation>))memptr,
                            (sp<Allocation>*)valptr);
    }

#if RS_TARGET_API > 18
    // Set a single allocation of script. Extends previously declared ctors
    // by support for newly introduced setter type with 'const' specifier. This
    // can be removed by setting the 'const' specifier in caller by the
    // rewriter (some time in the future when pre-19 Renderscript support will
    // eventually be dropped). Same issue at function @see hipaccSetScriptArg.
    hipacc_script_arg(void(F::*setter)(sp<const Allocation>),
                      sp<Allocation> const *arg)
        : id(21), memptr((void(F::*)())setter), valptr((void*)arg) {}
    std::pair<void(F::*)(sp<const Allocation>), sp<Allocation>*> get21() const {
      return std::make_pair((void(F::*)(sp<const Allocation>))memptr,
                            (sp<Allocation>*)valptr);
    }
#else // RS_TARGET_API > 18
    // unused getter, just to make the compiler happy
    std::pair<void(F::*)(int),int*> get21() const {
      return std::make_pair((void(F::*)(int))NULL, (int*)NULL);
    }
#endif // RS_TARGET_API > 18
};


#define SET_SCRIPT_ARG_ID(SCRIPT, ARG, ID) \
    hipaccSetScriptArg(SCRIPT, \
                       (ARG).get ## ID().first, \
                       *((ARG).get ## ID().second));

#define SET_SCRIPT_ARG(SCRIPT, ARG) \
    switch ((ARG).getId()) { \
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
       case 21: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 21) break; \
    }

#ifndef EXCLUDE_IMPL

const sp<Allocation> *hipaccGetAllocation(HipaccImage &img) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    return Ctx.get_allocation(img);
}

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block) {
    // calculate item id of a) first work item that requires no border handling
    // (left, top) and b) first work item that requires border handling (right,
    // bottom)
    if (info.size_x > 0) {
        info.bh_start_left = (int)ceilf((float)(info.offset_x + info.size_x)
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
        info.bh_start_top = (int)ceilf((float)(info.size_y)
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


std::string getRSErrorCodeStr(int errorNum) {
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


EHF::ErrorHandlerFunc_t errorHandler(uint32_t errorNum, std::string errorText) {
    std::cerr << "ERROR: " << getRSErrorCodeStr(errorNum)
              << " (" << errorNum << ")" << std::endl
              << "    " << errorText << std::endl;
}


// Create RenderScript context
void hipaccInitRenderScript(int targetAPI) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    PRS rs = Ctx.get_context();

    rs->setErrorHandler((EHF::ErrorHandlerFunc_t)&errorHandler);

    // Create context
    if (!INIT(rs, targetAPI)) {
        std::cerr << "ERROR: RenderScript initialization failed for targetAPI: "
                  << targetAPI << std::endl;
    }
}

#endif // EXCLUDE_IMPL

template<typename T>
T hipaccInitScript() {
#if RS_TARGET_API < 19
    std::string cache_path = "/sdcard";
    return T(HipaccContext::getInstance().get_context(), cache_path.c_str(),
            cache_path.length());
#else
    return T(HipaccContext::getInstance().get_context());
#endif
}


// Write to allocation
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem) {
    if (host_mem == NULL) return;

    size_t width  = img.width;
    size_t height = img.height;
    size_t stride = img.stride;

    if ((char *)host_mem != img.host)
        std::copy(host_mem, host_mem + width*height, (T*)img.host);

    if (stride > width) {
        T* buff = new T[stride * height];
        for (size_t i=0; i<height; ++i) {
            std::memcpy(buff + (i * stride), host_mem + (i * width), sizeof(T) * width);
        }
        COPYFROM(T, (Allocation *)img.mem, 0, stride * height, buff);
        delete[] buff;
    } else {
        COPYFROM(T, (Allocation *)img.mem, 0, width * height, host_mem);
    }
}


// Read from allocation
template<typename T>
T *hipaccReadMemory(HipaccImage &img) {
    size_t width  = img.width;
    size_t height = img.height;
    size_t stride = img.stride;

    if (stride > width) {
        T* buff = new T[stride * height];
        COPYTO(T, (Allocation *)img.mem, 0, stride * height, buff);
        for (size_t i=0; i<height; ++i) {
            std::memcpy(&((T*)img.host)[i*width], buff + (i * stride), sizeof(T) * width);
        }
        delete[] buff;
    } else {
        COPYTO(T, (Allocation *)img.mem, 0, width * height, (T*)img.host);
    }

    return (T*)img.host;
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem) {
    size_t size = dom.width * dom.height;
    uchar *dom_mem = new uchar[size];

    for (size_t i=0; i<size; ++i) {
        dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
    }

    hipaccWriteMemory(dom, dom_mem);

    delete[] dom_mem;
}

#ifndef EXCLUDE_IMPL

// Copy from allocation to allocation
void hipaccCopyMemory(HipaccImage &src, HipaccImage &dst) {
    assert(src.width == dst.width && src.height == dst.height &&
           src.pixel_size == dst.pixel_size && "Invalid CopyAllocation!");

    ((Allocation *)dst.mem)->copy1DRangeFrom(0, src.stride*src.height, (Allocation *)src.mem, 0);
}


// Copy from allocation region to allocation region
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst) {
    COPYFROM2D((Allocation *)dst.img.mem, dst.offset_x, dst.offset_y,
        src.width, src.height, (Allocation *)src.img.mem, src.width*src.height,
        src.offset_x, src.offset_y);
}


#define CREATE_ALLOCATION(T, E) \
HipaccImage createImage(T *host_mem, size_t width, size_t height, size_t stride, size_t alignment) { \
    HipaccContext &Ctx = HipaccContext::getInstance(); \
    PRS rs = Ctx.get_context(); \
\
    Type::Builder type(rs, E); \
    type.setX(stride); \
    type.setY(height); \
\
    sp<Allocation> allocation = Allocation::createTyped(rs, type.create()); \
\
    HipaccImage img = HipaccImage(width, height, stride, alignment, sizeof(T), (void *)allocation.get()); \
    Ctx.add_image(img, allocation); \
    hipaccWriteMemory(img, host_mem ? host_mem : (T*)img.host); \
\
    return img; \
} \
\
/* Allocate memory with alignment specified */ \
HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height, size_t alignment) { \
    alignment = (size_t)ceilf((float)alignment/sizeof(T)) * sizeof(T); \
    size_t stride = (size_t)ceilf((float)(width) / (alignment / sizeof(T))) * (alignment / sizeof(T)); \
    return createImage(host_mem, width, height, stride, alignment); \
} \
\
/* Allocate memory without any alignment considerations */ \
HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height) { \
    return createImage(host_mem, width, height, width, 0); \
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


// Release memory
void hipaccReleaseMemory(HipaccImage &img) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    Ctx.del_image(img);
}

#endif // EXCLUDE_IMPL

// Set a single argument of script
template<typename F, typename T>
void hipaccSetScriptArg(F* script, void(F::*setter)(T), T param) {
    (script->*setter)(param);
}


#if RS_TARGET_API > 18
// Set a single allocation of script. Extends previously declared function by
// support for newly introduced setter type with 'const' specifier. This can be
// removed by setting the 'const' specifier in caller by the rewriter (some
// time in the future when pre-19 Renderscript support will eventually be
// dropped). Same issue at class @see hipacc_script_arg.
template<typename F>
void hipaccSetScriptArg(F* script, void(F::*setter)(sp<const Allocation>),
                        sp<Allocation> param) {
    (script->*setter)((sp<const Allocation>)param);
}
#endif // RS_TARGET_API > 18


// Launch script kernel (one allocation)
template<typename F>
void hipaccLaunchScriptKernel(
    F* script,
    KERNEL1(F, kernel),
    HipaccImage &out, size_t *work_size, bool print_timing=true
) {
    long end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();
    PRS rs = Ctx.get_context();

    rs->finish();
    start = getMicroTime();
    (script->*kernel)((Allocation *)out.mem);
    rs->finish();
    end = getMicroTime();

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << (end - start) * 1.0e-3f << "(ms)" << std::endl;
    }
    total_time += (end - start) * 1.0e-3f;
    last_gpu_timing = (end - start) * 1.0e-3f;
}


// Launch script kernel (two allocations)
template<typename F>
void hipaccLaunchScriptKernel(
    F* script,
    KERNEL2(F, kernel),
    HipaccImage &in, HipaccImage &out, size_t *work_size, bool print_timing=true
) {
    long end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();
    PRS rs = Ctx.get_context();

    rs->finish();
    start = getMicroTime();
    (script->*kernel)((Allocation *)in.mem, (Allocation *)out.mem);
    rs->finish();
    end = getMicroTime();

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << (end - start) * 1.0e-3f << "(ms)" << std::endl;
    }
    total_time += (end - start) * 1.0e-3f;
    last_gpu_timing = (end - start) * 1.0e-3f;
}


// Benchmark timing for a kernel call
template<typename F>
void hipaccLaunchScriptKernelBenchmark(
    F* script,
    std::vector<hipacc_script_arg<F> > args,
    KERNEL1(F, kernel),
    HipaccImage &out, size_t *work_size,
    bool print_timing=true
) {
    float med_dt;
    std::vector<float> times;

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        // set kernel arguments
        for (typename std::vector<hipacc_script_arg<F> >::const_iterator
                it = args.begin(); it != args.end(); ++it) {
            SET_SCRIPT_ARG(script, *it);
        }

        // launch kernel
        hipaccLaunchScriptKernel(script, kernel, out, work_size, print_timing);
        times.push_back(last_gpu_timing);
    }
    std::sort(times.begin(), times.end());
    med_dt = times.at(HIPACC_NUM_ITERATIONS/2);

    last_gpu_timing = med_dt;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark ("
                  << work_size[0] * work_size[1] << ": "
                  << work_size[0] << "x" << work_size[1] << "): "
                  << med_dt << "(ms)" << std::endl;
    }
}


// Perform configuration exploration for a kernel call
template<typename F, typename T>
void hipaccLaunchScriptKernelExploration(
    F* script,
    std::vector<hipacc_script_arg<F> > args,
    KERNEL1(F, kernel),
    std::vector<hipacc_smem_info> smems, hipacc_launch_info &info,
    int warp_size, int max_threads_per_block, int max_threads_for_kernel,
    int max_smem_per_block, int heu_tx, int heu_ty,
    HipaccImage &iter_space
) {
    int opt_ws=1;
    float opt_time = FLT_MAX;

    std::cerr << "<HIPACC:> Exploring configurations for kernel"
              << " '" << kernel << "':" << std::endl;

    for (size_t curr_warp_size = 1; curr_warp_size <= (int)ceilf((float)info.is_width/3);
         curr_warp_size += (curr_warp_size < warp_size ? 1 : warp_size)) {
        // check if we exceed maximum number of threads
        if (curr_warp_size > max_threads_for_kernel) continue;

        size_t work_size[2];
        work_size[0] = curr_warp_size;
        work_size[1] = 1;

        hipaccPrepareKernelLaunch(info, work_size);

        float med_dt;
        std::vector<float> times;
        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            for (typename std::vector<hipacc_script_arg<F> >::const_iterator
                    it = args.begin(); it != args.end(); ++it) {
                SET_SCRIPT_ARG(script, *it);
            }

            // start timing
            total_time = 0.0f;

            // launch kernel
            hipaccLaunchScriptKernel(script, kernel, iter_space , work_size,
                    false);

            // stop timing
            times.push_back(total_time);
        }
        std::sort(times.begin(), times.end());
        med_dt = times.at(HIPACC_NUM_ITERATIONS/2);

        if (med_dt < opt_time) {
            opt_time = med_dt;
            opt_ws = curr_warp_size;
        }

        // print timing
        std::cerr << "<HIPACC:> Kernel config: "
                  << std::setw(4) << std::right << work_size[0] << "x"
                  << std::setw(2) << std::left << work_size[1]
                  << std::setw(5-floor(log10f((float)(work_size[0]*work_size[1]))))
                  << std::right << "(" << work_size[0]*work_size[1] << "): "
                  << std::setw(8) << std::fixed << std::setprecision(4)
                  << med_dt << " ms" << std::endl;
    }
    std::cerr << "<HIPACC:> Best configurations for kernel '" << kernel << "': "
              << opt_ws << ": " << opt_time << " ms" << std::endl;
}


// Perform global reduction and return result
template<typename F, typename T>
T hipaccApplyReduction(
    F *script,
    KERNEL1(F, kernel2D),
    KERNEL1(F, kernel1D),
    void(F::*setter)(sp<Allocation>),
    std::vector<hipacc_script_arg<F> > args,
    int is_width, bool print_timing=true
) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    PRS rs = Ctx.get_context();
    long end, start;

    // allocate temporary memory
    HipaccImage out_img = hipaccCreateAllocation((T*)NULL, is_width, 1);

    // allocation for 1st reduction step
    HipaccImage is1_img = hipaccCreateAllocation((T*)NULL, is_width, 1);
    sp<Allocation> is1 = (Allocation *)is1_img.mem;
    (script->*setter)(is1);

    // allocation for 2nd reduction step
    HipaccImage is2_img = hipaccCreateAllocation((T*)NULL, 1, 1);
    sp<Allocation> is2 = (Allocation *)is2_img.mem;

    // set arguments
    for (typename std::vector<hipacc_script_arg<F> >::const_iterator
            it = args.begin(); it != args.end(); ++it) {
        SET_SCRIPT_ARG(script, *it);
    }

    rs->finish();
    start = getMicroTime();
    (script->*kernel2D)(is1);   // first step: reduce image (region) into linear memory
    (script->*kernel1D)(is2);   // second step: reduce linear memory
    rs->finish();
    end = getMicroTime();
    last_gpu_timing = (end - start) * 1.0e-3f;

    if (print_timing) {
        std::cerr << "<HIPACC:> Reduction timing: "
                  << (end - start) * 1.0e-3f << "(ms)" << std::endl;
    }

    // download result of reduction
    T result;
    COPYTO(T, is2, 0, 1, &result);

    return result;
}


template<typename T>
HipaccImage hipaccCreatePyramidImage(HipaccImage &base, size_t width, size_t height) {
    if (base.alignment > 0) {
        return hipaccCreateAllocation((T*)NULL, width, height, base.alignment);
    } else {
        return hipaccCreateAllocation((T*)NULL, width, height);
    }
}

#endif  // __HIPACC_RS_HPP__

