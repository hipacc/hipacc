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

#ifndef __HIPACC_RS_TPP__
#define __HIPACC_RS_TPP__


#define CREATE_SCRIPT_ARG_IMPL(ID, T) \
    template<typename F> \
    hipacc_script_arg<F>::hipacc_script_arg(void(F::*setter)(T), T const *arg) \
        : id(ID), memptr((void(F::*)())setter), valptr((void*)arg) {} \
    template<typename F> \
    std::pair<void(F::*)(T), T*> hipacc_script_arg<F>::get ## ID() const { \
        return std::make_pair((void(F::*)(T))memptr, (T*)valptr); \
    }

CREATE_SCRIPT_ARG_IMPL(0, uint8_t)
CREATE_SCRIPT_ARG_IMPL(1, uint16_t)
CREATE_SCRIPT_ARG_IMPL(2, uint32_t)
CREATE_SCRIPT_ARG_IMPL(3, uint64_t)

CREATE_SCRIPT_ARG_IMPL(4, int8_t)
CREATE_SCRIPT_ARG_IMPL(5, int16_t)
CREATE_SCRIPT_ARG_IMPL(6, int32_t)
CREATE_SCRIPT_ARG_IMPL(7, int64_t)

CREATE_SCRIPT_ARG_IMPL(8, bool)
CREATE_SCRIPT_ARG_IMPL(9, char)
CREATE_SCRIPT_ARG_IMPL(10, float)
CREATE_SCRIPT_ARG_IMPL(11, double)

CREATE_SCRIPT_ARG_IMPL(12, uchar4)
CREATE_SCRIPT_ARG_IMPL(13, ushort4)
CREATE_SCRIPT_ARG_IMPL(14, uint4)
CREATE_SCRIPT_ARG_IMPL(15, ulong4)

CREATE_SCRIPT_ARG_IMPL(16, char4)
CREATE_SCRIPT_ARG_IMPL(17, short4)
CREATE_SCRIPT_ARG_IMPL(18, int4)
CREATE_SCRIPT_ARG_IMPL(19, long4)

CREATE_SCRIPT_ARG_IMPL(20, float4)
CREATE_SCRIPT_ARG_IMPL(21, double4)


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
       case 22: SET_SCRIPT_ARG_ID(SCRIPT, ARG, 22) break; \
    }


template<typename T>
T hipaccInitScript() {
    return T(HipaccContext::getInstance().get_context());
}


// Write to allocation
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem) {
    if (host_mem == NULL) return;

    size_t width  = img->width;
    size_t height = img->height;
    size_t stride = img->stride;

    if ((char *)host_mem != img->host)
        std::copy(host_mem, host_mem + width*height, (T*)img->host);

    if (stride > width) {
        T* buff = new T[stride * height];
        for (size_t i=0; i<height; ++i) {
            std::memcpy(buff + (i * stride), host_mem + (i * width), sizeof(T) * width);
        }
        ((Allocation *)img->mem)->copy1DRangeFrom(0, stride * height, buff);
        delete[] buff;
    } else {
        ((Allocation *)img->mem)->copy1DRangeFrom(0, width * height, host_mem);
    }
}


// Read from allocation
template<typename T>
T *hipaccReadMemory(const HipaccImage &img) {
    size_t width  = img->width;
    size_t height = img->height;
    size_t stride = img->stride;

    if (stride > width) {
        T* buff = new T[stride * height];
        ((Allocation *)img->mem)->copy1DRangeTo(0, stride * height, buff);
        for (size_t i=0; i<height; ++i) {
            std::memcpy(&((T*)img->host)[i*width], buff + (i * stride), sizeof(T) * width);
        }
        delete[] buff;
    } else {
        ((Allocation *)img->mem)->copy1DRangeTo(0, width * height, (T*)img->host);
    }

    return (T*)img->host;
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem) {
    size_t size = dom->width * dom->height;
    uchar *dom_mem = new uchar[size];

    for (size_t i=0; i<size; ++i) {
        dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
    }

    hipaccWriteMemory(dom, dom_mem);

    delete[] dom_mem;
}


// Set a single argument of script
template<typename F, typename T>
void hipaccSetScriptArg(F* script, void(F::*setter)(T), T param) {
    (script->*setter)(param);
}


// Launch script kernel (one allocation)
template<typename F>
void hipaccLaunchKernel(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    HipaccImage &out, size_t *work_size, bool print_timing
) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    sp<RS> rs = Ctx.get_context();

    rs->finish();
    auto start = hipacc_time_micro();
    (script->*kernel)((Allocation *)out->mem);
    rs->finish();
    auto end = hipacc_time_micro();
    last_gpu_timing = (end - start) * 1.0e-3f;

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << last_gpu_timing << "(ms)" << std::endl;
    }
}


// Launch script kernel (two allocations)
template<typename F>
void hipaccLaunchKernel(
    F* script,
    void(F::*kernel)(sp<Allocation>, sp<Allocation>),
    const HipaccImage &in, HipaccImage &out, size_t *work_size, bool print_timing
) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    sp<RS> rs = Ctx.get_context();

    rs->finish();
    auto start = hipacc_time_micro();
    (script->*kernel)((Allocation *)in->mem, (Allocation *)out->mem);
    rs->finish();
    auto end = hipacc_time_micro();
    last_gpu_timing = (end - start) * 1.0e-3f;

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << work_size[0] * work_size[1]
                  << ": " << work_size[0] << "x" << work_size[1] << "): "
                  << last_gpu_timing << "(ms)" << std::endl;
    }
}


// Benchmark timing for a kernel call
template<typename F>
void hipaccLaunchKernelBenchmark(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    HipaccImage &out, size_t *work_size,
    std::vector<hipacc_script_arg<F>> args,
    bool print_timing
) {
    std::vector<float> times;

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        for (auto &arg : args)
            SET_SCRIPT_ARG(script, arg);

        hipaccLaunchKernel(script, kernel, out, work_size, print_timing);
        times.push_back(last_gpu_timing);
    }

    std::sort(times.begin(), times.end());
    last_gpu_timing = times[times.size()/2];

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark ("
                  << work_size[0] * work_size[1] << ": "
                  << work_size[0] << "x" << work_size[1] << "): "
                  << last_gpu_timing << " | " << times.front() << " | " << times.back()
                  << " (median(" << HIPACC_NUM_ITERATIONS << ") | minimum | maximum) ms" << std::endl;
    }
}


// Perform configuration exploration for a kernel call
template<typename F, typename T>
void hipaccLaunchKernelExploration(
    F* script,
    void(F::*kernel)(sp<Allocation>),
    std::vector<hipacc_script_arg<F>> args,
    std::vector<hipacc_smem_info>, hipacc_launch_info &info,
    int warp_size, int, int max_threads_for_kernel,
    int, int, int,
    HipaccImage &out
) {
    int opt_ws=1;
    float opt_time = FLT_MAX;

    std::cerr << "<HIPACC:> Exploring configurations for kernel"
              << " '" << kernel << "':" << std::endl;

    for (int curr_warp_size = 1; curr_warp_size <= (int)ceilf((float)info.is_width/3);
         curr_warp_size += (curr_warp_size < warp_size ? 1 : warp_size)) {
        // check if we exceed maximum number of threads
        if (curr_warp_size > max_threads_for_kernel)
            continue;

        size_t work_size[2];
        work_size[0] = curr_warp_size;
        work_size[1] = 1;

        hipaccPrepareKernelLaunch(info, work_size);

        std::vector<float> times;
        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            for (auto &arg : args)
                SET_SCRIPT_ARG(script, arg);

            hipaccLaunchKernel(script, kernel, out, work_size, false);
            times.push_back(last_gpu_timing);
        }
        std::sort(times.begin(), times.end());
        last_gpu_timing = times[times.size()/2];

        if (last_gpu_timing < opt_time) {
            opt_time = last_gpu_timing;
            opt_ws = curr_warp_size;
        }

        // print timing
        std::cerr << "<HIPACC:> Kernel config: "
                  << std::setw(4) << std::right << work_size[0] << "x"
                  << std::setw(2) << std::left << work_size[1]
                  << std::setw(5-floor(log10f((float)(work_size[0]*work_size[1]))))
                  << std::right << "(" << work_size[0]*work_size[1] << "): "
                  << std::setw(8) << std::fixed << std::setprecision(4)
                  << last_gpu_timing << " | " << times.front() << " | " << times.back()
                  << " (median(" << HIPACC_NUM_ITERATIONS << ") | minimum | maximum) ms" << std::endl;
    }
    last_gpu_timing = opt_time;
    std::cerr << "<HIPACC:> Best configurations for kernel '" << kernel << "': "
              << opt_ws << ": " << opt_time << " ms" << std::endl;
}


// Perform global reduction and return result
template<typename F, typename T>
T hipaccApplyReduction(
    F *script,
    void(F::*kernel2D)(sp<Allocation>),
    void(F::*kernel1D)(sp<Allocation>),
    void(F::*setter)(sp<const Allocation>),
    std::vector<hipacc_script_arg<F>> args,
    int is_width, bool print_timing
) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    sp<RS> rs = Ctx.get_context();

    // allocate temporary memory
    HipaccImage out_img = hipaccCreateAllocation((T*)NULL, is_width, 1);

    // allocation for 1st reduction step
    HipaccImage is1_img = hipaccCreateAllocation((T*)NULL, is_width, 1);
    sp<Allocation> is1 = (Allocation *)is1_img->mem;
    (script->*setter)(is1);

    // allocation for 2nd reduction step
    HipaccImage is2_img = hipaccCreateAllocation((T*)NULL, 1, 1);
    sp<Allocation> is2 = (Allocation *)is2_img->mem;

    // set arguments
    for (auto &arg : args)
        SET_SCRIPT_ARG(script, arg);

    rs->finish();
    auto start = hipacc_time_micro();
    (script->*kernel2D)(is1);   // first step: reduce image (region) into linear memory
    (script->*kernel1D)(is2);   // second step: reduce linear memory
    rs->finish();
    auto end = hipacc_time_micro();
    last_gpu_timing = (end - start) * 1.0e-3f;

    if (print_timing) {
        std::cerr << "<HIPACC:> Reduction timing: "
                  << last_gpu_timing << "(ms)" << std::endl;
    }

    // download result of reduction
    T result;
    is2->copy1DRangeTo(0, 1, &result);

    return result;
}


template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height) {
    if (base->alignment > 0) {
        return hipaccCreateAllocation((T*)NULL, width, height, base->alignment);
    } else {
        return hipaccCreateAllocation((T*)NULL, width, height);
    }
}


#endif  // __HIPACC_RS_TPP__
