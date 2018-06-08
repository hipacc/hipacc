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

#ifndef __HIPACC_CU_TPP__
#define __HIPACC_CU_TPP__


template<typename T>
HipaccImage createImage(T *host_mem, void *mem, size_t width, size_t height, size_t stride, size_t alignment, hipaccMemoryType mem_type) {
    HipaccImage img = std::make_shared<HipaccImageCUDA>(width, height, stride, alignment, sizeof(T), mem, mem_type);
    hipaccWriteMemory(img, host_mem ? host_mem : (T*)img->host);

    return img;
}

template<typename T>
T *createMemory(size_t stride, size_t height) {
    T *mem;
    cudaError_t err = cudaMalloc((void **) &mem, sizeof(T)*stride*height);
    //err = cudaMallocPitch((void **) &mem, &stride, sizeof(T)*stride, height);
    checkErr(err, "cudaMalloc()");
    return mem;
}


// Allocate memory with alignment specified
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height, size_t alignment) {
    // alignment has to be a multiple of sizeof(T)
    alignment = (int)ceilf((float)alignment/sizeof(T)) * sizeof(T);
    size_t stride = (int)ceilf((float)(width)/(alignment/sizeof(T))) * (alignment/sizeof(T));

    T *mem = createMemory<T>(stride, height);
    return createImage(host_mem, (void *)mem, width, height, stride, alignment);
}


// Allocate memory without any alignment considerations
template<typename T>
HipaccImage hipaccCreateMemory(T *host_mem, size_t width, size_t height) {
    T *mem = createMemory<T>(width, height);
    return createImage(host_mem, (void *)mem, width, height, width, 0);
}


// Allocate 2D array
template<typename T>
HipaccImage hipaccCreateArray2D(T *host_mem, size_t width, size_t height) {
    cudaArray *array;
    int flags = cudaArraySurfaceLoadStore;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaError_t err = cudaMallocArray(&array, &channelDesc, width, height, flags);
    checkErr(err, "cudaMallocArray()");

    return createImage(host_mem, (void *)array, width, height, width, 0, Array2D);
}


// Allocate memory for Pyramid image
template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height) {
    switch (base->mem_type) {
        default:
            if (base->alignment > 0) {
                return hipaccCreateMemory<T>(NULL, width, height, base->alignment);
            } else {
                return hipaccCreateMemory<T>(NULL, width, height);
            }
        case Array2D:
            return hipaccCreateArray2D<T>(NULL, width, height);
    }
}


// Write to memory
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem) {
    if (host_mem == NULL) return;

    size_t width  = img->width;
    size_t height = img->height;
    size_t stride = img->stride;

    if ((char *)host_mem != img->host)
        std::copy(host_mem, host_mem + width*height, (T*)img->host);

    if (img->mem_type >= Array2D) {
        cudaError_t err = cudaMemcpyToArray((cudaArray *)img->mem, 0, 0, host_mem, sizeof(T)*width*height, cudaMemcpyHostToDevice);
        checkErr(err, "cudaMemcpyToArray()");
    } else {
        if (stride > width) {
            cudaError_t err = cudaMemcpy2D(img->mem, stride*sizeof(T), host_mem, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice);
            checkErr(err, "cudaMemcpy2D()");
        } else {
            cudaError_t err = cudaMemcpy(img->mem, host_mem, sizeof(T)*width*height, cudaMemcpyHostToDevice);
            checkErr(err, "cudaMemcpy()");
        }
    }
}


// Read from memory
template<typename T>
T *hipaccReadMemory(const HipaccImage &img) {
    size_t width  = img->width;
    size_t height = img->height;
    size_t stride = img->stride;

    if (img->mem_type >= Array2D) {
        cudaError_t err = cudaMemcpyFromArray((T*)img->host, (cudaArray *)img->mem, 0, 0, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
        checkErr(err, "cudaMemcpyFromArray()");
    } else {
        if (stride > width) {
            cudaError_t err = cudaMemcpy2D((T*)img->host, width*sizeof(T), img->mem, stride*sizeof(T), width*sizeof(T), height, cudaMemcpyDeviceToHost);
            checkErr(err, "cudaMemcpy2D()");
        } else {
            cudaError_t err = cudaMemcpy((T*)img->host, img->mem, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
            checkErr(err, "cudaMemcpy()");
        }
    }

    return (T*)img->host;
}


// Bind memory to texture
template<typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex, const HipaccImage &img) {
    cudaError_t err = cudaSuccess;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

    switch (mem_type) {
        case Linear1D:
            assert(img->mem_type<=Linear2D && "expected linear memory");
            err = cudaBindTexture(NULL, tex, img->mem, &channelDesc, sizeof(T)*img->stride*img->height);
            checkErr(err, "cudaBindTexture()");
            break;
        case Linear2D:
            assert(img->mem_type<=Linear2D && "expected linear memory");
            err = cudaBindTexture2D(NULL, tex, img->mem, &channelDesc, img->width, img->height, img->stride*sizeof(T));
            checkErr(err, "cudaBindTexture2D()");
            break;
        case Array2D:
            assert(img->mem_type==Array2D && "expected Array2D memory");
            err = cudaBindTextureToArray(tex, (cudaArray *)img->mem, &channelDesc);
            checkErr(err, "cudaBindTextureToArray()");
            break;
        default:
            assert(false && "wrong texture type");
    }
}


// Bind 2D array to surface
template<typename T>
void hipaccBindSurface(hipaccMemoryType mem_type, const surfaceReference *surf, const HipaccImage &img) {
    assert(mem_type==Surface && "wrong texture type");
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaError_t err = cudaBindSurfaceToArray(surf, (cudaArray *)img->mem, &channelDesc);
    checkErr(err, "cudaBindSurfaceToArray()");
}


// Write to symbol
template<typename T>
void hipaccWriteSymbol(const void *symbol, T *host_mem, size_t width, size_t height) {
    cudaError_t err = cudaMemcpyToSymbol(symbol, host_mem, sizeof(T)*width*height);
    checkErr(err, "cudaMemcpyToSymbol()");
}


// Read from symbol
template<typename T>
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string symbol_name, size_t width, size_t height) {
    cudaError_t err = cudaMemcpyFromSymbol(host_mem, symbol, sizeof(T)*width*height);
    checkErr(err, "cudaMemcpyFromSymbol()");
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(const void *symbol, T *host_mem, size_t width, size_t height) {
    size_t size = width * height;
    uchar *dom_mem = new uchar[size];

    for (size_t i=0; i<size; ++i) {
        dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
    }

    hipaccWriteSymbol<uchar>(symbol, dom_mem, width, height);

    delete[] dom_mem;
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const void *kernel1D, std::string kernel1D_name,
                       const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex) {
    T *output;  // GPU memory for reduction
    T result;   // host result

    // first step: reduce image (region) into linear memory
    dim3 block(max_threads, 1);
    dim3 grid((int)ceilf((float)(acc.img->width)/(block.x*2)), (int)ceilf((float)(acc.height)/pixels_per_thread));
    unsigned int num_blocks = grid.x*grid.y;
    unsigned int idle_left = 0;

    cudaError_t err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
        // reduce iteration space by idle blocks
        idle_left = acc.offset_x / block.x;
        unsigned int idle_right = (acc.img->width - (acc.offset_x+acc.width)) / block.x;
        grid.x = (int)ceilf((float)
                (acc.img->width - (idle_left + idle_right) * block.x) /
                (block.x*2));

        // update number of blocks
        num_blocks = grid.x*grid.y;
        idle_left *= block.x;
    }

    std::vector<void*> args_step1;
    switch (acc.img->mem_type) {
        default:
        case Global:
            args_step1.push_back((void*)&acc.img->mem);
            break;
        case Array2D:
            hipaccBindTexture<T>(Array2D, tex, acc.img);
            break;
    }

    args_step1.push_back((void*)&output);
    args_step1.push_back((void*)&acc.img->width);
    args_step1.push_back((void*)&acc.img->height);
    args_step1.push_back((void*)&acc.img->stride);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
        args_step1.push_back((void*)&acc.offset_x);
        args_step1.push_back((void*)&acc.offset_y);
        args_step1.push_back((void*)&acc.width);
        args_step1.push_back((void*)&acc.height);
        args_step1.push_back((void*)&idle_left);
    }

    hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args_step1.data());


    // second step: reduce partial blocks on GPU
    // this is done in one shot, so no additional memory is required, i.e. the
    // same array can be used for the input and output array
    // block.x is fixed, either max_threads or power of two
    block.x = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) : max_threads;
    grid.x = 1;
    grid.y = 1;
    // calculate the number of pixels reduced per thread
    int num_steps = (num_blocks + (block.x - 1)) / (block.x);

    std::vector<void*> args_step2;
    args_step2.push_back((void*)&output);
    args_step2.push_back((void*)&output);
    args_step2.push_back((void*)&num_blocks);
    args_step2.push_back((void*)&num_steps);

    hipaccLaunchKernel(kernel1D, kernel1D_name, grid, block, args_step2.data());

    // get reduced value
    err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}

// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name, const void *kernel1D, std::string kernel1D_name,
                       const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex) {
    HipaccAccessor acc(img);
    return hipaccApplyReduction<T>(kernel2D, kernel2D_name, kernel1D, kernel1D_name, acc, max_threads, pixels_per_thread, tex);
}


// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string kernel2D_name,
                                  const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex) {
    T *output;  // GPU memory for reduction
    T result;   // host result

    // single step reduction: reduce image (region) into linear memory and
    // reduce the linear memory using memory fence operations
    dim3 block(max_threads, 1);
    dim3 grid((int)ceilf((float)(acc.img->width)/(block.x*2)), (int)ceilf((float)(acc.height)/pixels_per_thread));
    unsigned int num_blocks = grid.x*grid.y;
    unsigned int idle_left = 0;

    cudaError_t err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
        // reduce iteration space by idle blocks
        idle_left = acc.offset_x / block.x;
        unsigned int idle_right = (acc.img->width - (acc.offset_x+acc.width)) / block.x;
        grid.x = (int)ceilf((float)
                (acc.img->width - (idle_left + idle_right) * block.x) /
                (block.x*2));

        // update number of blocks
        idle_left *= block.x;
    }

    std::vector<void*> args;
    switch (acc.img->mem_type) {
        default:
        case Global:
            args.push_back((void*)&acc.img->mem);
            break;
        case Array2D:
            hipaccBindTexture<T>(Array2D, tex, acc.img);
            break;
    }

    args.push_back((void*)&output);
    args.push_back((void*)&acc.img->width);
    args.push_back((void*)&acc.img->height);
    args.push_back((void*)&acc.img->stride);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
        args.push_back((void*)&acc.offset_x);
        args.push_back((void*)&acc.offset_y);
        args.push_back((void*)&acc.width);
        args.push_back((void*)&acc.height);
        args.push_back((void*)&idle_left);
    }

    hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args.data());

    err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}

// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D, std::string kernel2D_name,
                                  const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, const textureReference *tex) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionThreadFence<T>(kernel2D, kernel2D_name, acc, max_threads, pixels_per_thread, tex);
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D,
                                  const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc) {
    T *output;  // GPU memory for reduction
    T result;   // host result

    unsigned int num_blocks = (int)ceilf((float)(acc.img->width)/(max_threads*2))*acc.height;
    unsigned int idle_left = 0;

    cudaError_t err = cudaMalloc((void **) &output, sizeof(T)*num_blocks);
    checkErr(err, "cudaMalloc()");

    void *argsReduction2D[] = {
        (void *)&acc.img->mem,
        (void *)&output,
        (void *)&acc.img->width,
        (void *)&acc.img->height,
        (void *)&acc.img->stride,
        (void *)&acc.offset_x,
        (void *)&acc.offset_y,
        (void *)&acc.width,
        (void *)&acc.height,
        (void *)&idle_left
    };
    void *argsReduction2DArray[] = {
        (void *)&output,
        (void *)&acc.img->width,
        (void *)&acc.img->height,
        (void *)&acc.img->stride,
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
        std::vector<float> times;
        std::stringstream num_ppt_ss;
        std::stringstream num_bs_ss;
        num_ppt_ss << ppt;
        num_bs_ss << max_threads;

        std::vector<std::string> compile_options;
        compile_options.push_back("-I./include");
        compile_options.push_back("-D PPT=" + num_ppt_ss.str());
        compile_options.push_back("-D BS=" + num_bs_ss.str());
        compile_options.push_back("-D BSX_EXPLORE=64");
        compile_options.push_back("-D BSY_EXPLORE=1");

        CUmodule modReduction;
        hipaccCompileCUDAToModule(modReduction, filename, cc, compile_options);

        CUfunction exploreReduction2D;
        CUfunction exploreReduction1D;
        hipaccGetKernel(exploreReduction2D, modReduction, kernel2D);
        hipaccGetKernel(exploreReduction1D, modReduction, kernel1D);

        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            dim3 block(max_threads, 1);
            dim3 grid((int)ceilf((float)(acc.img->width)/(block.x*2)), (int)ceilf((float)(acc.height)/ppt));
            num_blocks = grid.x*grid.y;

            // check if the reduction is applied to the whole image
            if ((acc.offset_x || acc.offset_y) &&
                (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
                // reduce iteration space by idle blocks
                idle_left = acc.offset_x / block.x;
                unsigned int idle_right = (acc.img->width - (acc.offset_x+acc.width)) / block.x;
                grid.x = (int)ceilf((float)
                        (acc.img->width - (idle_left + idle_right) * block.x) /
                        (block.x*2));

                // update number of blocks
                num_blocks = grid.x*grid.y;
                idle_left *= block.x;
            }

            // bind texture to CUDA array
            CUtexref texImage;
            if (tex_info.tex_type==Array2D) {
                hipaccGetTexRef(texImage, modReduction, tex_info.name);
                hipaccBindTextureDrv(texImage, tex_info.image, tex_info.format, tex_info.tex_type);
                hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block, argsReduction2DArray, false);
            } else {
                hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block, argsReduction2D, false);
            }
            float total_time = last_gpu_timing;


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
                total_time += last_gpu_timing;

                num_blocks = grid.x;
            }
            times.push_back(total_time);
        }

        std::sort(times.begin(), times.end());
        last_gpu_timing = times[times.size()/2];

        if (last_gpu_timing < opt_time) {
            opt_time = last_gpu_timing;
            opt_ppt = ppt;
        }

        // print timing
        std::cerr << "<HIPACC:> PPT: " << std::setw(4) << std::right << ppt
                  << ", " << std::setw(8) << std::fixed << std::setprecision(4)
                  << last_gpu_timing << " | " << times.front() << " | " << times.back()
                  << " (median(" << HIPACC_NUM_ITERATIONS << ") | minimum | maximum) ms" << std::endl;

        // cleanup
        CUresult err = cuModuleUnload(modReduction);
        checkErrDrv(err, "cuModuleUnload()");
    }
    last_gpu_timing = opt_time;
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
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D,
                                  const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread, hipacc_tex_info tex_info, int cc) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, acc, max_threads, pixels_per_thread, tex_info, cc);
}


#ifndef SEGMENT_SIZE
# define SEGMENT_SIZE 128
# define MAX_SEGMENTS 512 // equals 65k bins (MAX_SEGMENTS*SEGMENT_SIZE)
#endif
template<typename T, typename T2>
T* hipaccApplyBinningSegmented(const void *kernel2D, std::string kernel2D_name,
                               HipaccAccessor &acc, unsigned int num_hists, unsigned int num_warps, unsigned int num_bins, const textureReference *tex) {
    T *output;  // GPU memory for reduction
    T *result = new T[num_bins];   // host result

    dim3 grid(num_hists, (num_bins+SEGMENT_SIZE-1)/SEGMENT_SIZE);
    dim3 block(32, num_warps);

    cudaError_t err = cudaMalloc((void **) &output, sizeof(T)*num_hists*num_bins);
    checkErr(err, "cudaMalloc()");

    std::vector<void*> args;
    switch (acc.img->mem_type) {
        default:
        case Global:
            args.push_back((void*)&acc.img->mem);
            break;
        case Array2D:
            hipaccBindTexture<T>(Array2D, tex, acc.img);
            break;
    }

    args.push_back((void*)&output);
    args.push_back((void*)&acc.width);
    args.push_back((void*)&acc.height);
    args.push_back((void*)&acc.img->stride);
    args.push_back((void*)&num_bins);
    args.push_back((void*)&acc.offset_x);
    args.push_back((void*)&acc.offset_y);

    hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args.data());

    err = cudaMemcpy(result, output, sizeof(T)*num_bins, cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy()");

    err = cudaFree(output);
    checkErr(err, "cudaFree()");

    return result;
}


#endif  // __HIPACC_CU_TPP__

