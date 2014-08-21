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

#if CUDA_VERSION < 5000
    #error "CUDA 5.0 or higher required!"
#endif

//#define USE_NVML
#ifdef USE_NVML
#include <nvml.h>
#endif

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
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

class HipaccContext : public HipaccContextBase {
    public:
        static HipaccContext &getInstance() {
            static HipaccContext instance;

            return instance;
        }
};


typedef struct hipacc_const_info {
    hipacc_const_info(std::string name, void *memory, int size) :
        name(name), memory(memory), size(size) {}
    std::string name;
    void *memory;
    int size;
} hipacc_const_info;


typedef struct hipacc_tex_info {
    hipacc_tex_info(std::string name, CUarray_format format, HipaccImage &image,
            hipaccMemoryType tex_type) :
        name(name), format(format), image(image), tex_type(tex_type) {}
    std::string name;
    CUarray_format format;
    HipaccImage &image;
    hipaccMemoryType tex_type;
} hipacc_tex_info;


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 &block) {
    // calculate block id of a) first block that requires no border handling
    // (left, top) and b) first block that requires border handling (right,
    // bottom)
    if (info.size_x > 0) {
        info.bh_start_left = (int)ceilf((float)(info.offset_x + info.size_x) / (block.x * info.simd_width));
        info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width - info.size_x) / (block.x * info.simd_width));
    } else {
        info.bh_start_left = 0;
        info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width) / (block.x * info.simd_width));
    }
    if (info.size_y > 0) {
        // for shared memory calculate additional blocks to be staged - this is
        // only required if shared memory is used, otherwise, info.size_y would
        // be sufficient
        int p_add = (int)ceilf(2*info.size_y / (float)block.y);
        info.bh_start_top = (int)ceilf((float)(info.size_y) / (info.pixels_per_thread * block.y));
        info.bh_start_bottom = (int)floor((float)(info.is_height - p_add*block.y) / (block.y * info.pixels_per_thread));
    } else {
        info.bh_start_top = 0;
        info.bh_start_bottom = (int)floor((float)(info.is_height) / (block.y * info.pixels_per_thread));
    }

    if ((info.bh_start_right - info.bh_start_left) > 1 && (info.bh_start_bottom - info.bh_start_top) > 1) {
        info.bh_fall_back = 0;
    } else {
        info.bh_fall_back = 1;
    }
}


dim3 hipaccCalcGridFromBlock(hipacc_launch_info &info, dim3 &block) {
    return dim3(
            (int)ceilf((float)(info.is_width + info.offset_x)/(block.x*info.simd_width)),
            (int)ceilf((float)(info.is_height)/(block.y*info.pixels_per_thread))
            );
}


#if CUDA_VERSION >= 6000
std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
    const char *errorName;
    const char *errorString;
    cuGetErrorName(errorCode, &errorName);
    cuGetErrorString(errorCode, &errorString);
    return std::string(errorName) + ": " + std::string(errorString);
}
#else
std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
    switch (errorCode) {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";
        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";
        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";
        case CUDA_ERROR_DEINITIALIZED:
            return "CUDA_ERROR_DEINITIALIZED";
        case CUDA_ERROR_PROFILER_DISABLED:
            return "CUDA_ERROR_PROFILER_DISABLED";
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
        case CUDA_ERROR_NO_DEVICE:
            return "CUDA_ERROR_NO_DEVICE";
        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";
        case CUDA_ERROR_INVALID_IMAGE:
            return "CUDA_ERROR_INVALID_IMAGE";
        case CUDA_ERROR_INVALID_CONTEXT:
            return "CUDA_ERROR_INVALID_CONTEXT";
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
        case CUDA_ERROR_MAP_FAILED:
            return "CUDA_ERROR_MAP_FAILED";
        case CUDA_ERROR_UNMAP_FAILED:
            return "CUDA_ERROR_UNMAP_FAILED";
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";
        case CUDA_ERROR_ALREADY_MAPPED:
            return "CUDA_ERROR_ALREADY_MAPPED";
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "CUDA_ERROR_ALREADY_ACQUIRED";
        case CUDA_ERROR_NOT_MAPPED:
            return "CUDA_ERROR_NOT_MAPPED";
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
        case CUDA_ERROR_INVALID_SOURCE:
            return "CUDA_ERROR_INVALID_SOURCE";
        case CUDA_ERROR_FILE_NOT_FOUND:
            return "CUDA_ERROR_FILE_NOT_FOUND";
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
        case CUDA_ERROR_OPERATING_SYSTEM:
            return "CUDA_ERROR_OPERATING_SYSTEM";
        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";
        case CUDA_ERROR_NOT_FOUND:
            return "CUDA_ERROR_NOT_FOUND";
        case CUDA_ERROR_NOT_READY:
            return "CUDA_ERROR_NOT_READY";
        case CUDA_ERROR_LAUNCH_FAILED:
            return "CUDA_ERROR_LAUNCH_FAILED";
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
        case CUDA_ERROR_ASSERT:
            return "CUDA_ERROR_ASSERT";
        case CUDA_ERROR_TOO_MANY_PEERS:
            return "CUDA_ERROR_TOO_MANY_PEERS";
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
        case CUDA_ERROR_NOT_PERMITTED:
            return "CUDA_ERROR_NOT_PERMITTED";
        case CUDA_ERROR_NOT_SUPPORTED:
            return "CUDA_ERROR_NOT_SUPPORTED";
        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
        default:
            return "unknown error code";
    }
}
#endif
// Macro for error checking device driver
#if 1
#define checkErrDrv(err, name) \
    if (err != CUDA_SUCCESS) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
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
        std::cerr << "ERROR: " << name << " (" << (err) << ")" << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
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

#ifdef USE_NVML
// Macro for error checking NVML
#if 1
#define checkErrNVML(err, name) \
    if (err != NVML_SUCCESS) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
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


// Initialize CUDA devices
void hipaccInitCUDA() {
    cudaError_t err = cudaSuccess;
    int device_count, driver_version = 0, runtime_version = 0;

    setenv("CUDA_CACHE_DISABLE", "1", 1);

    err = cudaGetDeviceCount(&device_count);
    checkErr(err, "cudaGetDeviceCount()");
    err = cudaDriverGetVersion(&driver_version);
    checkErr(err, "cudaDriverGetVersion()");
    err = cudaRuntimeGetVersion(&runtime_version);
    checkErr(err, "cudaRuntimeGetVersion()");

    std::cerr << "CUDA Driver/Runtime Version " << driver_version/1000 << "." << (driver_version%100)/10
        << "/" << runtime_version/1000 << "." << (runtime_version%100)/10 << std::endl;

    for (size_t i=0; i<(size_t)device_count; ++i) {
        cudaDeviceProp device_prop;

        err = cudaSetDevice(i);
        checkErr(err, "cudaSetDevice()");
        err = cudaGetDeviceProperties(&device_prop, i);
        checkErr(err, "cudaGetDeviceProperties()");

        if (i==0) std::cerr << "  [*] ";
        else std::cerr << "  [ ] ";
        std::cerr << "Name: " << device_prop.name << std::endl;
        std::cerr << "      Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    }
    err = cudaSetDevice(0);
    checkErr(err, "cudaSetDevice()");
}


// Allocate memory with alignment specified
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, int width, int height, int alignment) {
    T *mem;
    HipaccContext &Ctx = HipaccContext::getInstance();

    // alignment has to be a multiple of sizeof(T)
    alignment = (int)ceilf((float)alignment/sizeof(T)) * sizeof(T);
    // compute stride
    int stride = (int)ceilf((float)(width)/(alignment/sizeof(T))) * (alignment/sizeof(T));
    cudaError_t err = cudaMalloc((void **) &mem, sizeof(T)*stride*height);
    //err = cudaMallocPitch((void **) &mem, &stride, stride*sizeof(float), height);
    checkErr(err, "cudaMalloc()");

    HipaccImage img = HipaccImage(width, height, stride, alignment, sizeof(T), (void *)mem);
    Ctx.add_image(img);

    return img;
}


// Allocate memory without any alignment considerations
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, int width, int height) {
    T *mem;
    HipaccContext &Ctx = HipaccContext::getInstance();

    cudaError_t err = cudaMalloc((void **) &mem, sizeof(T)*width*height);
    checkErr(err, "cudaMalloc()");

    HipaccImage img = HipaccImage(width, height, width, 0, sizeof(T), (void *)mem);
    Ctx.add_image(img);

    return img;
}


// Allocate 2D array
template<typename T>
HipaccImage hipaccCreateArray2D(T *host_mem, int width, int height) {
    cudaArray *array;
    int flags = cudaArraySurfaceLoadStore;
    HipaccContext &Ctx = HipaccContext::getInstance();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height, flags);
    checkErr(err, "cudaMallocArray()");

    HipaccImage img = HipaccImage(width, height, width, 0, sizeof(T), (void *)array, Array2D);
    Ctx.add_image(img);

    return img;
}


// Allocate memory for Pyramid image
template<typename T>
HipaccImage hipaccCreatePyramidImage(HipaccImage &base, int width, int height) {
    switch (base.mem_type) {
        default:
            if (base.alignment > 0) {
                return hipaccCreateMemory<T>(NULL, width, height, base.alignment);
            } else {
                return hipaccCreateMemory<T>(NULL, width, height);
            }
        case Array2D:
            return hipaccCreateArray2D<T>(NULL, width, height);
    }
}


// Release memory
void hipaccReleaseMemory(HipaccImage &img) {
    cudaError_t err = cudaSuccess;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (img.mem_type >= Array2D) {
        err = cudaFreeArray((cudaArray *)img.mem);
        checkErr(err, "cudaFreeArray()");
    } else {
        err = cudaFree(img.mem);
        checkErr(err, "cudaFree()");
    }

    Ctx.del_image(img);
}


// Write to memory
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem) {
    cudaError_t err = cudaSuccess;

    int width = img.width;
    int height = img.height;
    int stride = img.stride;

    std::copy(host_mem, host_mem + width*height, (T*)img.host);
    if (img.mem_type >= Array2D) {
        err = cudaMemcpyToArray((cudaArray *)img.mem, 0, 0, host_mem, sizeof(T)*width*height, cudaMemcpyHostToDevice);
        checkErr(err, "cudaMemcpyToArray()");
    } else {
        if (stride > width) {
            err = cudaMemcpy2D(img.mem, stride*sizeof(T), host_mem, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice);
            checkErr(err, "cudaMemcpy2D()");
        } else {
            err = cudaMemcpy(img.mem, host_mem, sizeof(T)*width*height, cudaMemcpyHostToDevice);
            checkErr(err, "cudaMemcpy()");
        }
    }
}


// Read from memory
template<typename T>
T *hipaccReadMemory(HipaccImage &img) {
    int width = img.width;
    int height = img.height;
    int stride = img.stride;

    cudaError_t err = cudaSuccess;
    if (img.mem_type >= Array2D) {
        err = cudaMemcpyFromArray((T*)img.host, (cudaArray *)img.mem, 0, 0, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
        checkErr(err, "cudaMemcpyFromArray()");
    } else {
        if (stride > width) {
            err = cudaMemcpy2D((T*)img.host, width*sizeof(T), img.mem, stride*sizeof(T), width*sizeof(T), height, cudaMemcpyDeviceToHost);
            checkErr(err, "cudaMemcpy2D()");
        } else {
            err = cudaMemcpy((T*)img.host, img.mem, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
            checkErr(err, "cudaMemcpy()");
        }
    }

    return (T*)img.host;
}


// Copy from memory to memory
void hipaccCopyMemory(HipaccImage &src, HipaccImage &dst) {
    cudaError_t err = cudaSuccess;

    int height = src.height;
    int stride = src.stride;

    if (src.mem_type >= Array2D) {
        err = cudaMemcpyArrayToArray((cudaArray *)dst.mem, 0, 0, (cudaArray *)src.mem, 0, 0, stride*height*src.pixel_size, cudaMemcpyDeviceToDevice);
        checkErr(err, "cudaMemcpyArrayToArray()");
    } else {
        err = cudaMemcpy(dst.mem, src.mem, src.pixel_size*stride*height, cudaMemcpyDeviceToDevice);
        checkErr(err, "cudaMemcpy()");
    }
}


// Copy from memory region to memory region
void hipaccCopyMemoryRegion(HipaccAccessor src, HipaccAccessor dst) {
    cudaError_t err = cudaSuccess;

    if (src.img.mem_type >= Array2D) {
        err = cudaMemcpy2DArrayToArray((cudaArray *)dst.img.mem,
                dst.offset_x*dst.img.pixel_size, dst.offset_y,
                (cudaArray *)src.img.mem, src.offset_x*src.img.pixel_size,
                src.offset_y, src.width*src.img.pixel_size, src.height,
                cudaMemcpyDeviceToDevice);
        checkErr(err, "cudaMemcpy2DArrayToArray()");
    } else {
        void *dst_start = (char *)dst.img.mem + dst.offset_x*dst.img.pixel_size + (dst.offset_y*dst.img.stride*dst.img.pixel_size);
        void *src_start = (char *)src.img.mem + src.offset_x*src.img.pixel_size + (src.offset_y*src.img.stride*src.img.pixel_size);

        err = cudaMemcpy2D(dst_start, dst.img.stride*dst.img.pixel_size,
                           src_start, src.img.stride*src.img.pixel_size,
                           src.width*src.img.pixel_size, src.height,
                           cudaMemcpyDeviceToDevice);
        checkErr(err, "cudaMemcpy2D()");
    }
}


// Bind memory to texture
template<typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const struct textureReference
        *tex, HipaccImage &img) {
    cudaError_t err = cudaSuccess;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

    switch (mem_type) {
        case Linear1D:
            assert(img.mem_type<=Linear2D && "expected linear memory");
            err = cudaBindTexture(NULL, tex, img.mem, &channelDesc,
                    sizeof(T)*img.stride*img.height);
            checkErr(err, "cudaBindTexture()");
            break;
        case Linear2D:
            assert(img.mem_type<=Linear2D && "expected linear memory");
            err = cudaBindTexture2D(NULL, tex, img.mem, &channelDesc, img.width,
                    img.height, img.stride*sizeof(T));
            checkErr(err, "cudaBindTexture2D()");
            break;
        case Array2D:
            assert(img.mem_type==Array2D && "expected Array2D memory");
            err = cudaBindTextureToArray(tex, (cudaArray *)img.mem, &channelDesc);
            checkErr(err, "cudaBindTextureToArray()");
            break;
        default:
            assert(false && "wrong texture type");
    }
}


// Bind 2D array to surface
template<typename T>
void hipaccBindSurface(const struct surfaceReference *surf, HipaccImage &img) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaError_t err = cudaBindSurfaceToArray(surf, (cudaArray *)img.mem, &channelDesc);
    checkErr(err, "cudaBindSurfaceToArray()");
}


// Unbind texture
void hipaccUnbindTexture(const struct textureReference *tex) {
    cudaError_t err = cudaUnbindTexture(tex);
    checkErr(err, "cudaUnbindTexture()");
}


// Write to symbol
template<typename T>
void hipaccWriteSymbol(const void *symbol, T *host_mem, int width, int height) {
    cudaError_t err = cudaSuccess;
    err = cudaMemcpyToSymbol(symbol, host_mem, sizeof(T)*width*height);
    checkErr(err, "cudaMemcpyToSymbol()");
}


// Read from symbol
template<typename T>
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string symbol_name, int width, int height) {
    cudaError_t err = cudaSuccess;
    err = cudaMemcpyFromSymbol(host_mem, symbol, sizeof(T)*width*height);
    checkErr(err, "cudaMemcpyFromSymbol()");
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(const void *symbol, T *host_mem, int width, int height) {
    size_t size = width * height;
    uchar *dom_mem = new uchar[size];

    for (size_t i=0; i<size; ++i) {
        dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
    }

    hipaccWriteSymbol<uchar>(symbol, dom_mem, width, height);

    delete[] dom_mem;
}


// Set the configuration for a kernel
void hipaccConfigureCall(dim3 grid, dim3 block) {
    cudaError_t err = cudaConfigureCall(grid, block, 0, 0);
    checkErr(err, "cudaConfigureCall()");
}


// Set a single argument of a kernel
void hipaccSetupArgument(const void *arg, size_t size, size_t &offset) {
    // GPU data has to be accessed aligned
    if (offset % size) {
        offset += offset % size;
    }
    cudaError_t err = cudaSetupArgument(arg, size, offset);
    checkErr(err, "clSetKernelArg()");
    offset += size;
}


// Launch kernel
void hipaccLaunchKernel(const void *kernel, std::string kernel_name, dim3 grid, dim3 block, bool print_timing=true) {
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, end;
    float time;
    std::string error_string = "cudaLaunch(" + kernel_name + ")";

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    err = cudaLaunch(kernel);
    checkErr(err, error_string.c_str());

    cudaThreadSynchronize();
    err = cudaGetLastError();
    checkErr(err, error_string.c_str());

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    last_gpu_timing = time;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing ("<< block.x*block.y << ": " << block.x << "x" << block.y << "): " << time << "(ms)" << std::endl;
    }
}


// Benchmark timing for a kernel call
void hipaccLaunchKernelBenchmark(const void *kernel, std::string kernel_name, std::vector<std::pair<size_t, void *> > args, dim3 grid, dim3 block, bool print_timing=true) {
    float min_dt=FLT_MAX;

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        // setup call
        hipaccConfigureCall(grid, block);

        // set kernel arguments
        size_t offset = 0;
        for (size_t j=0; j<args.size(); ++j) {
            hipaccSetupArgument(args[j].second, args[j].first, offset);
        }

        // launch kernel
        hipaccLaunchKernel(kernel, kernel_name, grid, block, print_timing);
        if (last_gpu_timing < min_dt) min_dt = last_gpu_timing;
    }

    last_gpu_timing = min_dt;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark ("<< block.x*block.y << ": " << block.x << "x" << block.y << "): " << min_dt << "(ms)" << std::endl;
    }
}


//
// DRIVER API
//

// Compile CUDA source file to ptx assembly using nvcc compiler
void hipaccCompileCUDAToPTX(std::string file_name, int cc, std::string build_options = std::string()) {
    char line[FILENAME_MAX];
    FILE *fpipe;

    std::stringstream ss;
    ss << cc;

    std::string command = "${NVCC_COMPILER} -ptx -arch=compute_" + ss.str() + " ";
    command += build_options + " " + file_name;
    command += " -o " + file_name + ".ptx 2>&1";

    if (!(fpipe = (FILE *)popen(command.c_str(), "r"))) {
        perror("Problems with pipe");
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
        std::cerr << line;
    }
    pclose(fpipe);
}


// Load ptx assembly, create a module and kernel
void hipaccCreateModuleKernel(CUfunction &function, CUmodule &module, std::string file_name, std::string kernel_name, int cc) {
    CUresult err = CUDA_SUCCESS;
    CUjit_target target_cc;

    #if CUDA_VERSION >= 6000
    target_cc = (CUjit_target) cc;
    #else
    switch (cc) {
        default:
            std::cerr << "ERROR: Specified compute capability '" << cc << "' is not supported!" << std::endl;
            exit(EXIT_FAILURE);
        case 10: target_cc = CU_TARGET_COMPUTE_10; break;
        case 11: target_cc = CU_TARGET_COMPUTE_11; break;
        case 12: target_cc = CU_TARGET_COMPUTE_12; break;
        case 13: target_cc = CU_TARGET_COMPUTE_13; break;
        case 20: target_cc = CU_TARGET_COMPUTE_20; break;
        case 21: target_cc = CU_TARGET_COMPUTE_21; break;
        case 30: target_cc = CU_TARGET_COMPUTE_30; break;
        case 35: target_cc = CU_TARGET_COMPUTE_35; break;
    }
    #endif


    std::ifstream srcFile(file_name.c_str());
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open PTX source file '" << file_name.c_str() << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string ptxString(std::istreambuf_iterator<char>(srcFile),
            (std::istreambuf_iterator<char>()));

    const int errorLogSize = 10240;
    char errorLogBuffer[errorLogSize] = {0};

    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET };
    void *optionValues[] = { (void *)errorLogBuffer, (void *)errorLogSize, (void *)target_cc };

    // Load PTX source
    err = cuModuleLoadDataEx(&module, ptxString.c_str(), 2, options, optionValues);
    if (err != CUDA_SUCCESS) {
        std::cerr << "Error log: " << errorLogBuffer << std::endl;
    }
    checkErrDrv(err, "cuModuleLoadDataEx()");

    // Get function entry point
    err = cuModuleGetFunction(&function, module, kernel_name.c_str());
    checkErrDrv(err, "cuModuleGetFunction()");
}


// Computes occupancy for kernel function
size_t blockSizeToSmemSize(int blockSize) { return 0; } // TODO: provide proper function to estimate smem usage
void hipaccPrintKernelOccupancy(CUfunction fun, int tile_size_x, int tile_size_y) {
    #if CUDA_VERSION >= 6050
    CUresult err = CUDA_SUCCESS;
    CUdevice dev = 0;

    int warp_size;
    err = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    checkErrDrv(err, "cuDeviceGetAttribute()");
    int block_size = tile_size_x*tile_size_y;
    size_t dynamic_smem_bytes = 0;
    int block_size_limit = 0;

    int active_blocks;
    int min_grid_size, opt_block_size;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, fun, block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");

    err = cuOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, fun, &blockSizeToSmemSize, dynamic_smem_bytes, block_size_limit);
    checkErrDrv(err, "cuOccupancyMaxPotentialBlockSize()");

    // re-compute with optimal block size
    int max_blocks;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, fun, opt_block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");
    int max_warps = max_blocks * (opt_block_size/warp_size);
    int active_warps = active_blocks * (block_size/warp_size);
    float occupancy = (float)active_warps/(float)max_warps;
    std::cerr << ";  occupancy: "
              << std::fixed << std::setprecision(2) << occupancy << " ("
              << active_warps << " out of " << max_warps << " warps"
              //<< "; optimal block size: " << opt_block_size
              << ")" << std::endl;
    #else
    std::cerr << std::endl;
    #endif
}


// Launch kernel
void hipaccLaunchKernel(CUfunction &kernel, std::string kernel_name, dim3 grid, dim3 block, void **args, bool print_timing=true) {
    CUresult err = CUDA_SUCCESS;
    cudaEvent_t start, end;
    float time;
    std::string error_string = "cuLaunchKernel(" + kernel_name + ")";

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    // Launch the kernel
    err = cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL);
    checkErrDrv(err, error_string.c_str());
    err = cuCtxSynchronize();
    checkErrDrv(err, error_string.c_str());

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    total_time += time;

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    last_gpu_timing = time;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << block.x*block.y << ": " << block.x << "x" << block.y << "): " << time << "(ms)" << std::endl;
    }
}
void hipaccLaunchKernelBenchmark(CUfunction &kernel, std::string kernel_name, dim3 grid, dim3 block, void **args, bool print_timing=true) {
    float min_dt=FLT_MAX;

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; i++) {
        hipaccLaunchKernel(kernel, kernel_name, grid, block, args, print_timing);
        if (last_gpu_timing < min_dt) min_dt = last_gpu_timing;
    }

    last_gpu_timing = min_dt;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark (" << block.x*block.y
                  << ": " << block.x << "x" << block.y << "): " << min_dt
                  << "(ms)" << std::endl;
    }
}


// Get global reference from module
void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module, std::string global_name) {
    size_t size;
    CUresult err = cuModuleGetGlobal(&global, &size, module, global_name.c_str());
    checkErrDrv(err, "cuModuleGetGlobal()");
}


// Get texture reference from module
void hipaccGetTexRef(CUtexref &tex, CUmodule &module, std::string texture_name) {
    CUresult err = cuModuleGetTexRef(&tex, module, texture_name.c_str());
    checkErrDrv(err, "cuModuleGetTexRef()");
}


// Get surface reference from module
void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module, std::string surface_name) {
    CUresult err = cuModuleGetSurfRef(&surf, module, surface_name.c_str());
    checkErrDrv(err, "cuModuleGetSurfRef()");
}


// Bind texture to linear memory
void hipaccBindTextureDrv(CUtexref &texture, HipaccImage &img, CUarray_format
        format, hipaccMemoryType tex_type) {
    checkErrDrv(cuTexRefSetFormat(texture, format, 1), "cuTexRefSetFormat()");
    checkErrDrv(cuTexRefSetFlags(texture, CU_TRSF_READ_AS_INTEGER), "cuTexRefSetFlags()");
    switch (tex_type) {
        case Linear1D:
            checkErrDrv(cuTexRefSetAddress(0, texture, (CUdeviceptr)img.mem,
                        img.pixel_size*img.stride*img.height),
                    "cuTexRefSetAddress()");
            break;
        case Linear2D:
            CUDA_ARRAY_DESCRIPTOR desc;
            desc.Format = format;
            desc.NumChannels = 1;
            desc.Width = img.width;
            desc.Height = img.height;
            checkErrDrv(cuTexRefSetAddress2D(texture, &desc, (CUdeviceptr)img.mem,
                        img.pixel_size*img.stride), "cuTexRefSetAddress2D()");
            break;
        case Array2D:
            checkErrDrv(cuTexRefSetArray(texture, (CUarray)img.mem,
                        CU_TRSA_OVERRIDE_FORMAT), "cuTexRefSetArray()");
            break;
        default:
            assert(false && "not a texture");
    }
}


// Bind surface to 2D array
void hipaccBindSurfaceDrv(CUsurfref &surface, HipaccImage &img) {
    checkErrDrv(cuSurfRefSetArray(surface, (CUarray)img.mem, 0), "cuSurfRefSetArray()");
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const
        void *kernel1D, std::string kernel1D_name, HipaccAccessor &acc, unsigned
        int max_threads, unsigned int pixels_per_thread, const struct
        textureReference *tex) {
    cudaError_t err = cudaSuccess;
    T *output;  // GPU memory for reduction
    T result;   // host result

    // first step: reduce image (region) into linear memory
    dim3 block(max_threads, 1);
    dim3 grid((int)ceilf((float)(acc.img.width)/(block.x*2)), (int)ceilf((float)(acc.height)/pixels_per_thread));
    unsigned int num_blocks = grid.x*grid.y;
    unsigned int idle_left = 0;

    err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
        // reduce iteration space by idle blocks
        idle_left = acc.offset_x / block.x;
        unsigned int idle_right = (acc.img.width - (acc.offset_x+acc.width)) / block.x;
        grid.x = (int)ceilf((float)
                (acc.img.width - (idle_left + idle_right) * block.x) /
                (block.x*2));

        // update number of blocks
        num_blocks = grid.x*grid.y;
        idle_left *= block.x;
    }

    size_t offset = 0;
    hipaccConfigureCall(grid, block);

    switch (acc.img.mem_type) {
        default:
        case Global:
            hipaccSetupArgument(&acc.img.mem, sizeof(T *), offset);
            break;
        case Array2D:
            hipaccBindTexture<T>(Array2D, tex, acc.img);
            break;
    }

    hipaccSetupArgument(&output, sizeof(T *), offset);
    hipaccSetupArgument(&acc.img.width, sizeof(unsigned int), offset);
    hipaccSetupArgument(&acc.img.height, sizeof(unsigned int), offset);
    hipaccSetupArgument(&acc.img.stride, sizeof(unsigned int), offset);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
        hipaccSetupArgument(&acc.offset_x, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.offset_y, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.width, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.height, sizeof(unsigned int), offset);
        hipaccSetupArgument(&idle_left, sizeof(unsigned int), offset);
    }

    hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block);


    // second step: reduce partial blocks on GPU
    // this is done in one shot, so no additional memory is required, i.e. the
    // same array can be used for the input and output array
    // block.x is fixed, either max_threads or power of two
    block.x = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) :
        max_threads;
    grid.x = 1;
    grid.y = 1;
    // calculate the number of pixels reduced per thread
    int num_steps = (num_blocks + (block.x - 1)) / (block.x);

    offset = 0;
    hipaccConfigureCall(grid, block);

    hipaccSetupArgument(&output, sizeof(T *), offset);
    hipaccSetupArgument(&output, sizeof(T *), offset);
    hipaccSetupArgument(&num_blocks, sizeof(unsigned int), offset);
    hipaccSetupArgument(&num_steps, sizeof(unsigned int), offset);

    hipaccLaunchKernel(kernel1D, kernel1D_name, grid, block);

    // get reduced value
    err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}
// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const
        void *kernel1D, std::string kernel1D_name, HipaccImage &img, unsigned
        int max_threads, unsigned int pixels_per_thread, const struct
        textureReference *tex) {
    HipaccAccessor acc(img);
    return hipaccApplyReduction<T>(kernel2D, kernel2D_name, kernel1D,
            kernel1D_name, acc, max_threads, pixels_per_thread, tex);
}


// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string
        kernel2D_name, HipaccAccessor &acc, unsigned int max_threads, unsigned
        int pixels_per_thread, const struct textureReference *tex) {
    cudaError_t err = cudaSuccess;
    T *output;  // GPU memory for reduction
    T result;   // host result

    // single step reduction: reduce image (region) into linear memory and
    // reduce the linear memory using memory fence operations
    dim3 block(max_threads, 1);
    dim3 grid((int)ceilf((float)(acc.img.width)/(block.x*2)), (int)ceilf((float)(acc.height)/pixels_per_thread));
    unsigned int num_blocks = grid.x*grid.y;
    unsigned int idle_left = 0;

    err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
        // reduce iteration space by idle blocks
        idle_left = acc.offset_x / block.x;
        unsigned int idle_right = (acc.img.width - (acc.offset_x+acc.width)) / block.x;
        grid.x = (int)ceilf((float)
                (acc.img.width - (idle_left + idle_right) * block.x) /
                (block.x*2));

        // update number of blocks
        num_blocks = grid.x*grid.y;
        idle_left *= block.x;
    }

    size_t offset = 0;
    hipaccConfigureCall(grid, block);

    switch (acc.img.mem_type) {
        default:
        case Global:
            hipaccSetupArgument(&acc.img.mem, sizeof(T *), offset);
            break;
        case Array2D:
            hipaccBindTexture<T>(Array2D, tex, acc.img);
            break;
    }

    hipaccSetupArgument(&output, sizeof(T *), offset);
    hipaccSetupArgument(&acc.img.width, sizeof(unsigned int), offset);
    hipaccSetupArgument(&acc.img.height, sizeof(unsigned int), offset);
    hipaccSetupArgument(&acc.img.stride, sizeof(unsigned int), offset);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
        hipaccSetupArgument(&acc.offset_x, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.offset_y, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.width, sizeof(unsigned int), offset);
        hipaccSetupArgument(&acc.height, sizeof(unsigned int), offset);
        hipaccSetupArgument(&idle_left, sizeof(unsigned int), offset);
    }

    hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block);

    err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}
// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string
        kernel2D_name, HipaccImage &img, unsigned int max_threads, unsigned int
        pixels_per_thread, const struct textureReference *tex) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionThreadFence<T>(kernel2D, kernel2D_name, acc,
            max_threads, pixels_per_thread, tex);
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
        std::string kernel1D, HipaccAccessor &acc, unsigned int max_threads,
        unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc) {
    cudaError_t err = cudaSuccess;
    T *output;  // GPU memory for reduction
    T result;   // host result

    unsigned int num_blocks = (int)ceilf((float)(acc.img.width)/(max_threads*2))*acc.height;
    unsigned int idle_left = 0;

    err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    void *argsReduction2D[] = {
        (void *)&acc.img.mem,
        (void *)&output,
        (void *)&acc.img.width,
        (void *)&acc.img.height,
        (void *)&acc.img.stride,
        (void *)&acc.offset_x,
        (void *)&acc.offset_y,
        (void *)&acc.width,
        (void *)&acc.height,
        (void *)&idle_left
    };
    void *argsReduction2DArray[] = {
        (void *)&output,
        (void *)&acc.img.width,
        (void *)&acc.img.height,
        (void *)&acc.img.stride,
        (void *)&acc.offset_x,
        (void *)&acc.offset_y,
        (void *)&acc.width,
        (void *)&acc.height,
        (void *)&idle_left
    };

    std::cerr << "<HIPACC:> Exploring pixels per thread for '" << kernel2D << ", " << kernel1D << "'" << std::endl;

    float opt_time = FLT_MAX;
    int opt_ppt = 1;
    for (size_t ppt=1; ppt<=acc.height; ++ppt) {
        std::stringstream num_ppt_ss;
        std::stringstream num_bs_ss;
        num_ppt_ss << ppt;
        num_bs_ss << max_threads;

        std::string compile_options = "-D PPT=" + num_ppt_ss.str() + " -D BS=" + num_bs_ss.str() + " -I./include ";
        compile_options += "-D BSX_EXPLORE=64 -D BSY_EXPLORE=1 ";
        hipaccCompileCUDAToPTX(filename, cc, compile_options);

        std::string ptx_filename = filename + ".ptx";
        CUmodule modReduction;
        CUfunction exploreReduction2D;
        CUfunction exploreReduction1D;
        hipaccCreateModuleKernel(exploreReduction2D, modReduction, ptx_filename, kernel2D, cc);
        hipaccCreateModuleKernel(exploreReduction1D, modReduction, ptx_filename, kernel1D, cc);

        float min_dt=FLT_MAX;
        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            dim3 block(max_threads, 1);
            dim3 grid((int)ceilf((float)(acc.img.width)/(block.x*2)), (int)ceilf((float)(acc.height)/ppt));
            num_blocks = grid.x*grid.y;

            // check if the reduction is applied to the whole image
            if ((acc.offset_x || acc.offset_y) &&
                (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
                // reduce iteration space by idle blocks
                idle_left = acc.offset_x / block.x;
                unsigned int idle_right = (acc.img.width - (acc.offset_x+acc.width)) / block.x;
                grid.x = (int)ceilf((float)
                        (acc.img.width - (idle_left + idle_right) * block.x) /
                        (block.x*2));

                // update number of blocks
                num_blocks = grid.x*grid.y;
                idle_left *= block.x;
            }

            // start timing
            total_time = 0.0f;

            // bind texture to CUDA array
            CUtexref texImage;
            if (tex_info.tex_type==Array2D) {
                hipaccGetTexRef(texImage, modReduction, tex_info.name);
                hipaccBindTextureDrv(texImage, tex_info.image, tex_info.format, tex_info.tex_type);
                hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block, argsReduction2DArray, false);
            } else {
                hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block, argsReduction2D, false);
            }


            // second step: reduce partial blocks on GPU
            grid.y = 1;
            while (num_blocks > 1) {
                block.x = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) :
                    max_threads;
                grid.x = (int)ceilf((float)(num_blocks)/(block.x*ppt));

                void *argsReduction1D[] = {
                    (void *)&output,
                    (void *)&output,
                    (void *)&num_blocks,
                    (void *)&ppt
                };

                hipaccLaunchKernel(exploreReduction1D, kernel1D, grid, block, argsReduction1D, false);

                num_blocks = grid.x;
            }
            // stop timing
            if (total_time < min_dt) min_dt = total_time;
        }
        if (total_time < opt_time) {
            opt_time = total_time;
            opt_ppt = ppt;
        }

        // print timing
        std::cerr << "<HIPACC:> PPT: " << std::setw(4) << std::right << ppt
                  << ", " << std::setw(8) << std::fixed << std::setprecision(4)
                  << min_dt << " ms" << std::endl;

        // cleanup
        CUresult err = cuModuleUnload(modReduction);
        checkErrDrv(err, "cuModuleUnload()");
    }
    std::cerr << "<HIPACC:> Best unroll factor for reduction kernel '"
              << kernel2D << "/" << kernel1D << "': "
              << opt_ppt << ": " << opt_time << " ms" << std::endl;

    // get reduced value
    err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
        std::string kernel1D, HipaccImage &img, unsigned int max_threads,
        unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, acc,
            max_threads, pixels_per_thread, tex_info, cc);
}


// Perform configuration exploration for a kernel call
void hipaccKernelExploration(std::string filename, std::string kernel,
        std::vector<void *> args, std::vector<hipacc_smem_info> smems,
        std::vector<hipacc_const_info> consts, std::vector<hipacc_tex_info*>
        texs, hipacc_launch_info &info, size_t warp_size, size_t
        max_threads_per_block, size_t max_threads_for_kernel, size_t
        max_smem_per_block, size_t heu_tx, size_t heu_ty, int cc) {
    CUresult err = CUDA_SUCCESS;
    std::string ptx_filename = filename + ".ptx";
    size_t opt_tx=warp_size, opt_ty=1;
    float opt_time = FLT_MAX;

    std::cerr << "<HIPACC:> Exploring configurations for kernel '" << kernel
              << "': configuration provided by heuristic " << heu_tx*heu_ty
              << " (" << heu_tx << "x" << heu_ty << "). " << std::endl;


    #ifdef USE_NVML
    nvmlReturn_t nvml_err = NVML_SUCCESS;
    nvmlDevice_t nvml_device;
    nvmlEnableState_t nvml_mode;
    unsigned int nvml_device_count, nvml_temperature, nvml_power;

    nvml_err = nvmlInit();
    checkErrNVML(nvml_err, "nvmlInit()");

    nvml_err = nvmlDeviceGetCount(&nvml_device_count);
    checkErrNVML(nvml_err, "nvmlDeviceGetCount()");
    assert(nvml_device_count>0 && "no device detected by NVML");

    nvml_err = nvmlDeviceGetHandleByIndex(0, &nvml_device);
    checkErrNVML(nvml_err, "nvmlDeviceGetHandleByIndex()");

    nvml_err = nvmlDeviceGetPowerManagementMode(nvml_device, &nvml_mode);
    if (nvml_mode == NVML_FEATURE_DISABLED || nvml_err == NVML_ERROR_NOT_SUPPORTED) {
        std::cerr << "NVML Warning: device does not support querying power usage!" << std::endl;
    } else {
        checkErrNVML(nvml_err, "nvmlDeviceGetPowerManagementMode()");
    }
    #endif


    for (size_t tile_size_x=warp_size; tile_size_x<=max_threads_per_block; tile_size_x+=warp_size) {
        for (size_t tile_size_y=1; tile_size_y<=max_threads_per_block; ++tile_size_y) {
            // check if we exceed maximum number of threads
            if (tile_size_x*tile_size_y > max_threads_for_kernel) continue;

            // check if we exceed size of shared memory
            size_t used_smem = 0;
            for (size_t i=0; i<smems.size(); ++i) {
                used_smem += (tile_size_x + smems[i].size_x)*(tile_size_y + smems[i].size_y - 1) * smems[i].pixel_size;
            }
            if (used_smem >= max_smem_per_block) continue;
            if (used_smem && tile_size_x > warp_size) continue;

            std::stringstream num_threads_x_ss, num_threads_y_ss;
            num_threads_x_ss << tile_size_x;
            num_threads_y_ss << tile_size_y;

            // compile kernel
            std::string compile_options = "-D BSX_EXPLORE=" +
                num_threads_x_ss.str() + " -D BSY_EXPLORE=" +
                num_threads_y_ss.str() + " -I./include ";
            hipaccCompileCUDAToPTX(filename, cc, compile_options);

            CUmodule modKernel;
            CUfunction exploreKernel;
            hipaccCreateModuleKernel(exploreKernel, modKernel, ptx_filename, kernel, cc);

            // load constant memory
            CUdeviceptr constMem;
            for (size_t i=0; i<consts.size(); ++i) {
                hipaccGetGlobal(constMem, modKernel, consts[i].name);
                err = cuMemcpyHtoD(constMem, consts[i].memory, consts[i].size);
                checkErrDrv(err, "cuMemcpyHtoD()");
            }

            CUtexref texImage;
            CUsurfref surfImage;
            for (size_t i=0; i<texs.size(); ++i) {
                if (texs[i]->tex_type==Surface) {
                    // bind surface memory
                    hipaccGetSurfRef(surfImage, modKernel, texs[i]->name);
                    hipaccBindSurfaceDrv(surfImage, texs[i]->image);
                } else {
                    // bind texture memory
                    hipaccGetTexRef(texImage, modKernel, texs[i]->name);
                    hipaccBindTextureDrv(texImage, texs[i]->image,
                            texs[i]->format, texs[i]->tex_type);
                }
            }

            dim3 block(tile_size_x, tile_size_y);
            dim3 grid(hipaccCalcGridFromBlock(info, block));
            hipaccPrepareKernelLaunch(info, block);

            float min_dt=FLT_MAX;
            for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
                // start timing
                total_time = 0.0f;

                hipaccLaunchKernel(exploreKernel, kernel, grid, block, args.data(), false);

                // stop timing
                if (total_time < min_dt) min_dt = total_time;
            }
            if (min_dt < opt_time) {
                opt_time = min_dt;
                opt_tx = tile_size_x;
                opt_ty = tile_size_y;
            }

            #ifdef USE_NVML
            nvml_err = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &nvml_temperature);
            checkErrNVML(nvml_err, "nvmlDeviceGetTemperature()");
            nvml_err = nvmlDeviceGetPowerUsage(nvml_device, &nvml_power);
            checkErrNVML(nvml_err, "nvmlDeviceGetPowerUsage()");
            #endif

            // print timing
            std::cerr << "<HIPACC:> Kernel config: "
                      << std::setw(4) << std::right << tile_size_x << "x"
                      << std::setw(2) << std::left << tile_size_y
                      << std::setw(5-floor(log10f((float)(tile_size_x*tile_size_y))))
                      << std::right << "(" << tile_size_x*tile_size_y << "): "
                      << std::setw(8) << std::fixed << std::setprecision(4)
                      << min_dt << " ms";
            #ifdef USE_NVML
            std::cerr << ";  temperature: " << nvml_temperature << " °C"
                      << ";  power usage: " << nvml_power/1000.f << " W";
            #endif
            hipaccPrintKernelOccupancy(exploreKernel, tile_size_x, tile_size_y);

            // cleanup
            err = cuModuleUnload(modKernel);
            checkErrDrv(err, "cuModuleUnload()");
        }
    }
    std::cerr << "<HIPACC:> Best configurations for kernel '" << kernel << "': "
              << opt_tx*opt_ty << " (" << opt_tx << "x" << opt_ty << "): "
              << opt_time << " ms" << std::endl;

    #ifdef USE_NVML
    nvml_err = nvmlShutdown();
    checkErrNVML(nvml_err, "nvmlShutdown()");
    #endif
}

#endif  // __HIPACC_CU_HPP__

