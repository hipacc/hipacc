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


#define PATH_MAX 256
#include <RenderScript.h>

using namespace android;
using namespace android::RSC;

#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "hipacc_base.hpp"


class HipaccContext : public HipaccContextBase {
    private:
        sp<RS> context;

        HipaccContext();

    public:
        static HipaccContext &getInstance();
        sp<RS> get_context();
};

class HipaccImageAndroid : public HipaccImageBase {
    private:
        sp<const Allocation> alloc;
    public:
        HipaccImageAndroid(size_t width, size_t height, size_t stride,
                           size_t alignment, size_t pixel_size, sp<const Allocation> alloc,
                           hipaccMemoryType mem_type=Global);
};


template<typename F>
class hipacc_script_arg {
  private:
    int id;
    void(F::*memptr)();
    void *valptr;

  public:
    int getId() const;

#define CREATE_SCRIPT_ARG_DECL(ID, T) \
    hipacc_script_arg(void(F::*setter)(T), T const *arg); \
    std::pair<void(F::*)(T), T*> get ## ID() const;

    CREATE_SCRIPT_ARG_DECL(0, uint8_t)
    CREATE_SCRIPT_ARG_DECL(1, uint16_t)
    CREATE_SCRIPT_ARG_DECL(2, uint32_t)
    CREATE_SCRIPT_ARG_DECL(3, uint64_t)

    CREATE_SCRIPT_ARG_DECL(4, int8_t)
    CREATE_SCRIPT_ARG_DECL(5, int16_t)
    CREATE_SCRIPT_ARG_DECL(6, int32_t)
    CREATE_SCRIPT_ARG_DECL(7, int64_t)

    CREATE_SCRIPT_ARG_DECL(8, bool)
    CREATE_SCRIPT_ARG_DECL(9, char)
    CREATE_SCRIPT_ARG_DECL(10, float)
    CREATE_SCRIPT_ARG_DECL(11, double)

    CREATE_SCRIPT_ARG_DECL(12, uchar4)
    CREATE_SCRIPT_ARG_DECL(13, ushort4)
    CREATE_SCRIPT_ARG_DECL(14, uint4)
    CREATE_SCRIPT_ARG_DECL(15, ulong4)

    CREATE_SCRIPT_ARG_DECL(16, char4)
    CREATE_SCRIPT_ARG_DECL(17, short4)
    CREATE_SCRIPT_ARG_DECL(18, int4)
    CREATE_SCRIPT_ARG_DECL(19, long4)

    CREATE_SCRIPT_ARG_DECL(20, float4)
    CREATE_SCRIPT_ARG_DECL(21, double4)

    hipacc_script_arg(void(F::*setter)(sp<const Allocation>), sp<const Allocation> const *arg);
    std::pair<void(F::*)(sp<const Allocation>), sp<const Allocation>*> get22() const;
};


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block);
std::string getRSErrorCodeStr(int errorNum);
void errorHandler(uint32_t errorNum, const char *errorText);
void hipaccInitRenderScript(std::string rs_directory);
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst);


#define CREATE_ALLOCATION_DECL(T, E) \
HipaccImage createImage(T *host_mem, size_t width, size_t height, size_t stride, size_t alignment); \
HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height, size_t alignment); \
HipaccImage hipaccCreateAllocation(T *host_mem, size_t width, size_t height);

CREATE_ALLOCATION_DECL(uint8_t,  Element::U8(rs))
CREATE_ALLOCATION_DECL(uint16_t, Element::U16(rs))
CREATE_ALLOCATION_DECL(uint32_t, Element::U32(rs))
CREATE_ALLOCATION_DECL(uint64_t, Element::U64(rs))

CREATE_ALLOCATION_DECL(int8_t,   Element::I8(rs))
CREATE_ALLOCATION_DECL(int16_t,  Element::I16(rs))
CREATE_ALLOCATION_DECL(int32_t,  Element::I32(rs))
CREATE_ALLOCATION_DECL(int64_t,  Element::I64(rs))

CREATE_ALLOCATION_DECL(bool,     Element::BOOLEAN(rs))
CREATE_ALLOCATION_DECL(char,     Element::U8(rs))
CREATE_ALLOCATION_DECL(float,    Element::F32(rs))
CREATE_ALLOCATION_DECL(double,   Element::F64(rs))

CREATE_ALLOCATION_DECL(uchar4,   Element::U8_4(rs))
CREATE_ALLOCATION_DECL(ushort4,  Element::U16_4(rs))
CREATE_ALLOCATION_DECL(uint4,    Element::U32_4(rs))
CREATE_ALLOCATION_DECL(ulong4,   Element::U64_4(rs))

CREATE_ALLOCATION_DECL(char4,    Element::I8_4(rs))
CREATE_ALLOCATION_DECL(short4,   Element::I16_4(rs))
CREATE_ALLOCATION_DECL(int4,     Element::I32_4(rs))
CREATE_ALLOCATION_DECL(long4,    Element::I64_4(rs))

CREATE_ALLOCATION_DECL(float4,   Element::F32_4(rs))
CREATE_ALLOCATION_DECL(double4,  Element::F64_4(rs))


template<typename T>
T hipaccInitScript();
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem);
template<typename T>
T *hipaccReadMemory(const HipaccImage &img);
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem);
template<typename F, typename T>
void hipaccSetScriptArg(F* script, void(F::*setter)(T), T param);
template<typename F>
void hipaccLaunchKernel(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    HipaccImage &out, size_t *work_size, bool print_timing=true);
template<typename F>
void hipaccLaunchKernel(
    F* script,
    void(F::*kernel)(sp<Allocation>, sp<Allocation>),
    const HipaccImage &in, HipaccImage &out, size_t *work_size, bool print_timing=true);
template<typename F>
void hipaccLaunchKernelBenchmark(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    HipaccImage &out, size_t *work_size,
    std::vector<hipacc_script_arg<F>> args,
    bool print_timing=true);
template<typename F, typename T>
void hipaccLaunchKernelExploration(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    std::vector<hipacc_script_arg<F>> args,
    std::vector<hipacc_smem_info>, hipacc_launch_info &info,
    int warp_size, int, int max_threads_for_kernel,
    int, int, int,
    HipaccImage &out);
template<typename F, typename T>
T hipaccApplyReduction(
    F *script,
    void(F::*kernel2D)(sp<Allocation>),
    void(F::*kernel1D)(sp<Allocation>),
    void(F::*setter)(sp<const Allocation>),
    std::vector<hipacc_script_arg<F>> args,
    int is_width, bool print_timing=true);
template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height);


#include "hipacc_rs.tpp"


#endif  // __HIPACC_RS_HPP__
