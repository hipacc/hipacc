//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
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

#ifndef __HIPACC_CU_HPP__
#define __HIPACC_CU_HPP__

#include <cuda.h>

#if CUDA_VERSION < 7000
    #error "CUDA 7.0 or higher required!"
#endif

/* #undef NVML_FOUND */
#ifdef NVML_FOUND
#include <nvml.h>
#endif
#define NVRTC_FOUND
#ifdef NVRTC_FOUND
#include <nvrtc.h>
#endif


#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "hipacc_base.hpp"


// Macro for error checking device driver
#if 1
#define checkErrDrv(err, name) \
    if (err != CUDA_SUCCESS) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" \
                  << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
        std::cerr << getCUDAErrorCodeStrDrv(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#else
inline void checkErrDrv(CUresult err, std::string name) {
    if (err != CUDA_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << (err) << "): ";
        std::cerr << getCUDAErrorCodeStrDrv(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif
// Macro for error checking
#if 1
#define checkErr(err, name) \
    if (err != cudaSuccess) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" \
                  << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#else
inline void checkErr(cudaError_t err, std::string name) {
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << name << " (" << err << "): ";
        std::cerr << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef NVML_FOUND
// Macro for error checking NVML
#if 1
#define checkErrNVML(err, name) \
    if (err != NVML_SUCCESS) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" \
                  << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
        std::cerr << nvmlErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#else
inline void checkErrNVML(nvmlReturn_t err, std::string name) {
    if (err != NVML_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << (err) << "): ";
        std::cerr << nvmlErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif
#endif

#ifdef NVRTC_FOUND
#if 1
#define checkErrNVRTC(err, name) \
    if (err != NVRTC_SUCCESS ) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" \
                  << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
        std::cerr << nvrtcGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#else
inline void checkErrNVRTC(nvrtcResult err, std::string name) {
    if (err != NVRTC_SUCCESS ) {
        std::cerr << "ERROR: " << name << " (" << (err) << "): ";
        std::cerr << nvrtcGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif
#endif


std::string getCUDAErrorCodeStrDrv(CUresult errorCode);


class HipaccContext : public HipaccContextBase {
    public:
        static HipaccContext &getInstance();
};


class HipaccImageCUDA : public HipaccImageBase {
    public:
        HipaccImageCUDA(size_t width, size_t height, size_t stride,
                        size_t alignment, size_t pixel_size, void *mem,
                        hipaccMemoryType mem_type=Global);
        ~HipaccImageCUDA();
};


typedef struct hipacc_const_info {
    hipacc_const_info(std::string name, void *memory, int size);
    std::string name;
    void *memory;
    int size;
} hipacc_const_info;


typedef struct hipacc_tex_info {
    hipacc_tex_info(std::string name, CUarray_format format,
        const HipaccImage &image, hipaccMemoryType tex_type);
    std::string name;
    CUarray_format format;
    const HipaccImage &image;
    hipaccMemoryType tex_type;
} hipacc_tex_info;


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 &block);
dim3 hipaccCalcGridFromBlock(hipacc_launch_info &info, dim3 &block);
void hipaccInitCUDA();
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst);
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst);
void hipaccLaunchKernel(const void *kernel, std::string kernel_name, dim3 grid, dim3 block, void **args, bool print_timing=true);
void hipaccLaunchKernelBenchmark(const void *kernel, std::string kernel_name, dim3 grid, dim3 block, std::vector<void *> args, bool print_timing=true);
void hipaccLaunchKernelExploration(std::string filename, std::string kernel, std::vector<void *> args,
                                   std::vector<hipacc_smem_info> smems, std::vector<hipacc_const_info> consts, std::vector<hipacc_tex_info*> texs,
                                   hipacc_launch_info &info, size_t warp_size, size_t max_threads_per_block, size_t max_threads_for_kernel, size_t max_smem_per_block, size_t heu_tx, size_t heu_ty, int cc);



//
// TEMPLATES
//

template<typename T>
HipaccImage createImage(T *host_mem, void *mem, size_t width, size_t height, size_t stride, size_t alignment, hipaccMemoryType mem_type=Global);
template<typename T>
T *createMemory(size_t stride, size_t height);
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height, size_t alignment);
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height);
template<typename T>
HipaccImage hipaccCreateArray2D(T *host_mem, size_t width, size_t height);
template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height);
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem);
template<typename T>
T *hipaccReadMemory(const HipaccImage &img);
template<typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex, const HipaccImage &img);
template<typename T>
void hipaccBindSurface(hipaccMemoryType mem_type, const surfaceReference *surf, const HipaccImage &img);
void hipaccUnbindTexture(const textureReference *tex);
template<typename T>
void hipaccWriteSymbol(const void *symbol, T *host_mem, size_t width, size_t height);
template<typename T>
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string symbol_name, size_t width, size_t height);
template<typename T>
void hipaccWriteDomainFromMask(const void *symbol, T *host_mem, size_t width, size_t height);



//
// DRIVER API
//

void hipaccCreateModule(CUmodule &module, const void *ptx, int cc);
void hipaccCompileCUDAToModule(CUmodule &module, std::string file_name, int cc, std::vector<std::string> &build_options);
void hipaccGetKernel(CUfunction &function, CUmodule &module, std::string kernel_name);
size_t blockSizeToSmemSize(int blockSize);
void hipaccPrintKernelOccupancy(CUfunction fun, int tile_size_x, int tile_size_y);
void hipaccLaunchKernel(CUfunction &kernel, std::string kernel_name, dim3 grid, dim3 block, void **args, bool print_timing=true);
void hipaccLaunchKernelBenchmark(CUfunction &kernel, std::string kernel_name, dim3 grid, dim3 block, void **args, bool print_timing=true);
void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module, std::string global_name);
void hipaccGetTexRef(CUtexref &tex, CUmodule &module, std::string texture_name);
void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module, std::string surface_name);
void hipaccBindTextureDrv(CUtexref &texture, const HipaccImage &img, CUarray_format
        format, hipaccMemoryType tex_type);
void hipaccBindSurfaceDrv(CUsurfref &surface, const HipaccImage &img);



//
// REDUCTIONS AND BINNING
//

template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const void *kernel1D, std::string kernel1D_name,
                       const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex);
template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const void *kernel1D, std::string kernel1D_name,
                       const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex);
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string kernel2D_name,
                                  const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex);
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string kernel2D_name,
                                  const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex);
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D,
                                  const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc);
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D,
                                  const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc);

#ifndef SEGMENT_SIZE
# define SEGMENT_SIZE 128
# define MAX_SEGMENTS 512 // equals 65k bins (MAX_SEGMENTS*SEGMENT_SIZE)
#endif
template<typename T, typename T2>
T* hipaccApplyBinningSegmented(const void *kernel2D, std::string kernel2D_name,
                               HipaccAccessor &acc, unsigned int num_hists, unsigned int num_warps, unsigned int num_bins, const textureReference *tex);


#include "hipacc_cu.tpp"


#endif  // __HIPACC_CU_HPP__

