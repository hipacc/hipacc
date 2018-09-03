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


// This is the standalone (header-only) Hipacc Renderscript runtime


#include "hipacc_rs.hpp"


#ifndef __HIPACC_RS_STANDALONE_HPP__
#define __HIPACC_RS_STANDALONE_HPP__


#include "hipacc_base_standalone.hpp"


HipaccContext::HipaccContext() {
    context = new RS();
}

HipaccContext& HipaccContext::getInstance() {
    static HipaccContext instance;

    return instance;
}

sp<RS> HipaccContext::get_context() {
    return context;
}

HipaccImageAndroid::HipaccImageAndroid(size_t width, size_t height, size_t
    stride, size_t alignment, size_t pixel_size, sp<const Allocation> alloc,
    hipaccMemoryType mem_type)
      : HipaccImageBase(width, height, stride, alignment, pixel_size,
          (void*)alloc.get(), mem_type), alloc(alloc) {
}

template<typename F>
int hipacc_script_arg<F>::getId() const {
    return id;
}

template<typename F>
hipacc_script_arg<F>::hipacc_script_arg(void(F::*setter)(sp<const Allocation>), sp<const Allocation> const *arg)
      : id(22), memptr((void(F::*)())setter), valptr((void*)arg) {
}

template<typename F>
std::pair<void(F::*)(sp<const Allocation>), sp<const Allocation>*> hipacc_script_arg<F>::get22() const {
    return std::make_pair((void(F::*)(sp<const Allocation>))memptr, (sp<const Allocation>*)valptr);
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
        case RS_ERROR_NONE:                 return "RS_ERROR_NONE";
        case RS_ERROR_BAD_SHADER:           return "RS_ERROR_BAD_SHADER";
        case RS_ERROR_BAD_SCRIPT:           return "RS_ERROR_BAD_SCRIPT";
        case RS_ERROR_BAD_VALUE:            return "RS_ERROR_BAD_VALUE";
        case RS_ERROR_OUT_OF_MEMORY:        return "RS_ERROR_OUT_OF_MEMORY";
        case RS_ERROR_DRIVER:               return "RS_ERROR_DRIVER";
        case RS_ERROR_FATAL_DEBUG:          return "RS_ERROR_FATAL_DEBUG";
        case RS_ERROR_FATAL_UNKNOWN:        return "RS_ERROR_FATAL_UNKNOWN";
        case RS_ERROR_FATAL_DRIVER:         return "RS_ERROR_FATAL_DRIVER";
        case RS_ERROR_FATAL_PROGRAM_LINK:   return "RS_ERROR_FATAL_PROGRAM_LINK";
        default:                            return "unknown error code";
    }
}


void errorHandler(uint32_t errorNum, const char *errorText) {
    std::cerr << "ERROR: " << getRSErrorCodeStr(errorNum)
              << " (" << errorNum << ")" << std::endl
              << "    " << errorText << std::endl;
}


// Create RenderScript context
void hipaccInitRenderScript(std::string rs_directory) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    sp<RS> rs = Ctx.get_context();

    rs->setErrorHandler((ErrorHandlerFunc_t)&errorHandler);

    // Create context
    if (!rs->init(rs_directory.c_str())) {
        std::cerr << "ERROR: RenderScript initialization failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}


// Copy from allocation to allocation
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst) {
    assert(src->width == dst->width && src->height == dst->height &&
           src->pixel_size == dst->pixel_size && "Invalid CopyAllocation!");

    ((Allocation *)dst->mem)->copy1DRangeFrom(0, src->stride*src->height, (Allocation *)src->mem, 0);
}


// Copy from allocation region to allocation region
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst) {
    ((Allocation *)dst.img->mem)->copy2DRangeFrom(dst.offset_x, dst.offset_y, src.width, src.height,
                      (Allocation *)src.img->mem, src.offset_x, src.offset_y);
}


#define CREATE_ALLOCATION_IMPL(T, E) \
HipaccImage createImage(T *host_mem, size_t width, size_t height, size_t stride, size_t alignment) { \
    HipaccContext &Ctx = HipaccContext::getInstance(); \
    sp<RS> rs = Ctx.get_context(); \
\
    Type::Builder type(rs, E); \
    type.setX(stride); \
    type.setY(height); \
\
    sp<const Allocation> allocation = Allocation::createTyped(rs, type.create()); \
\
    HipaccImage img = std::make_shared<HipaccImageAndroid>(width, height, stride, alignment, sizeof(T), allocation); \
    hipaccWriteMemory(img, host_mem ? host_mem : (T*)img->host); \
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


CREATE_ALLOCATION_IMPL(uint8_t,  Element::U8(rs))
CREATE_ALLOCATION_IMPL(uint16_t, Element::U16(rs))
CREATE_ALLOCATION_IMPL(uint32_t, Element::U32(rs))
CREATE_ALLOCATION_IMPL(uint64_t, Element::U64(rs))

CREATE_ALLOCATION_IMPL(int8_t,   Element::I8(rs))
CREATE_ALLOCATION_IMPL(int16_t,  Element::I16(rs))
CREATE_ALLOCATION_IMPL(int32_t,  Element::I32(rs))
CREATE_ALLOCATION_IMPL(int64_t,  Element::I64(rs))

CREATE_ALLOCATION_IMPL(bool,     Element::BOOLEAN(rs))
CREATE_ALLOCATION_IMPL(char,     Element::U8(rs))
CREATE_ALLOCATION_IMPL(float,    Element::F32(rs))
CREATE_ALLOCATION_IMPL(double,   Element::F64(rs))

CREATE_ALLOCATION_IMPL(uchar4,   Element::U8_4(rs))
CREATE_ALLOCATION_IMPL(ushort4,  Element::U16_4(rs))
CREATE_ALLOCATION_IMPL(uint4,    Element::U32_4(rs))
CREATE_ALLOCATION_IMPL(ulong4,   Element::U64_4(rs))

CREATE_ALLOCATION_IMPL(char4,    Element::I8_4(rs))
CREATE_ALLOCATION_IMPL(short4,   Element::I16_4(rs))
CREATE_ALLOCATION_IMPL(int4,     Element::I32_4(rs))
CREATE_ALLOCATION_IMPL(long4,    Element::I64_4(rs))

CREATE_ALLOCATION_IMPL(float4,   Element::F32_4(rs))
CREATE_ALLOCATION_IMPL(double4,  Element::F64_4(rs))


#endif  // __HIPACC_RS_STANDALONE_HPP__
