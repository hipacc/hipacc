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

#ifndef __HIPACC_CL_TPP__
#define __HIPACC_CL_TPP__


template<typename T>
HipaccImage createImage(T *host_mem, cl_mem mem, size_t width, size_t height, size_t stride, size_t alignment, hipaccMemoryType mem_type) {
    HipaccImage img = std::make_shared<HipaccImageOpenCL>(width, height, stride, alignment, sizeof(T), mem, mem_type);
    hipaccWriteMemory(img, host_mem ? host_mem : (T*)img->host);
    return img;
}

template<typename T>
cl_mem createBuffer(size_t stride, size_t height, cl_mem_flags flags) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*stride*height, NULL, &err);
    checkErr(err, "clCreateBuffer()");
    return buffer;
}


// Allocate memory with alignment specified
template<typename T>
HipaccImage hipaccCreateBuffer(T *host_mem, size_t width, size_t height, size_t alignment) {
    // alignment has to be a multiple of sizeof(T)
    alignment = (size_t)ceilf((float)alignment/sizeof(T)) * sizeof(T);
    size_t stride = (size_t)ceilf((float)(width)/(alignment/sizeof(T))) * (alignment/sizeof(T));

    cl_mem buffer = createBuffer<T>(stride, height, CL_MEM_READ_WRITE);
    return createImage(host_mem, buffer, width, height, stride, alignment);
}


// Allocate memory without any alignment considerations
template<typename T>
HipaccImage hipaccCreateBuffer(T *host_mem, size_t width, size_t height) {
    cl_mem buffer = createBuffer<T>(width, height, CL_MEM_READ_WRITE);
    return createImage(host_mem, buffer, width, height, width, 0);
}


// Allocate constant buffer
template<typename T>
HipaccImage hipaccCreateBufferConstant(T *host_mem, size_t width, size_t height) {
    cl_mem buffer = createBuffer<T>(width, height, CL_MEM_READ_ONLY);
    return createImage(host_mem, buffer, width, height, width, 0);
}


// Allocate image - no alignment can be specified
template<typename T>
HipaccImage hipaccCreateImage(T *host_mem, size_t width, size_t height,
        cl_channel_type channel_type, cl_channel_order channel_order) {
    cl_int err = CL_SUCCESS;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_image_format image_format;
    image_format.image_channel_order = channel_order;
    image_format.image_channel_data_type = channel_type;
    HipaccContext &Ctx = HipaccContext::getInstance();

    #ifdef CL_VERSION_1_2
    cl_image_desc image_desc;
    memset(&image_desc, '\0', sizeof(cl_image_desc));

    // CL_MEM_OBJECT_IMAGE1D
    // CL_MEM_OBJECT_IMAGE1D_BUFFER
    // CL_MEM_OBJECT_IMAGE2D
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;

    cl_mem image = clCreateImage(Ctx.get_contexts()[0], flags, &image_format, &image_desc, NULL, &err);
    checkErr(err, "clCreateImage()");
    #else
    cl_mem image = clCreateImage2D(Ctx.get_contexts()[0], flags, &image_format, width, height, 0, NULL, &err);
    checkErr(err, "clCreateImage2D()");
    #endif

    return createImage(host_mem, image, width, height, width, 0, Array2D);
}


// Write to memory
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem, int num_device) {
    if (host_mem == NULL) return;

    size_t width  = img->width;
    size_t height = img->height;
    size_t stride = img->stride;

    if ((char *)host_mem != img->host)
        std::copy(host_mem, host_mem + width*height, (T*)img->host);

    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_int err = CL_SUCCESS;
    if (img->mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { width, height, 1 };
        // no stride supported for images in OpenCL
        const size_t input_row_pitch = width*sizeof(T);
        const size_t input_slice_pitch = 0;

        err = clEnqueueWriteImage(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, origin, region, input_row_pitch, input_slice_pitch, host_mem, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueWriteImage()");
    } else {
        if (stride > width) {
            for (size_t i=0; i<height; ++i) {
                err |= clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &host_mem[i*width], 0, NULL, NULL);
            }
        } else {
            err = clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, 0, sizeof(T)*width*height, host_mem, 0, NULL, NULL);
        }
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueWriteBuffer()");
    }
}


// Read from memory
template<typename T>
T *hipaccReadMemory(const HipaccImage &img, int num_device) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (img->mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { img->width, img->height, 1 };
        // no stride supported for images in OpenCL
        const size_t row_pitch = img->width*sizeof(T);
        const size_t slice_pitch = 0;

        err = clEnqueueReadImage(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, origin, region, row_pitch, slice_pitch, (T*)img->host, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueReadImage()");
    } else {
        size_t width = img->width;
        size_t height = img->height;
        size_t stride = img->stride;

        if (stride > width) {
            for (size_t i=0; i<height; ++i) {
                err |= clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &((T*)img->host)[i*width], 0, NULL, NULL);
            }
        } else {
            err = clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img->mem, CL_FALSE, 0, sizeof(T)*width*height, (T*)img->host, 0, NULL, NULL);
        }
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueReadBuffer()");
    }

    return (T*)img->host;
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem) {
    size_t size = dom->width * dom->height;
    uchar *dom_mem = new uchar[size];

    for (size_t i=0; i < size; ++i) {
        dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
    }

    hipaccWriteMemory(dom, dom_mem);

    delete[] dom_mem;
}


// Set a single argument of a kernel
template<typename T>
void hipaccSetKernelArg(cl_kernel kernel, unsigned int num, size_t size, T* param) {
    cl_int err = clSetKernelArg(kernel, num, size, param);
    checkErr(err, "clSetKernelArg()");
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T result;       // host result

    // first step: reduce image (region) into linear memory
    size_t local_work_size[2];
    local_work_size[0] = max_threads;
    local_work_size[1] = 1;
    size_t global_work_size[2];
    global_work_size[0] = (int)ceilf((float)(acc.img->width)/(local_work_size[0]*2))*local_work_size[0];
    global_work_size[1] = (int)ceilf((float)(acc.height)/(local_work_size[1]*pixels_per_thread))*local_work_size[1];

    unsigned int num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    hipaccSetKernelArg(kernel2D, 0, sizeof(cl_mem), &acc.img->mem);
    hipaccSetKernelArg(kernel2D, 1, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel2D, 2, sizeof(unsigned int), &acc.img->width);
    hipaccSetKernelArg(kernel2D, 3, sizeof(unsigned int), &acc.img->height);
    hipaccSetKernelArg(kernel2D, 4, sizeof(unsigned int), &acc.img->stride);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
        hipaccSetKernelArg(kernel2D, 5, sizeof(unsigned int), &acc.offset_x);
        hipaccSetKernelArg(kernel2D, 6, sizeof(unsigned int), &acc.offset_y);
        hipaccSetKernelArg(kernel2D, 7, sizeof(unsigned int), &acc.width);
        hipaccSetKernelArg(kernel2D, 8, sizeof(unsigned int), &acc.height);

        // reduce iteration space by idle blocks
        unsigned int idle_left = acc.offset_x / local_work_size[0];
        unsigned int idle_right = (acc.img->width - (acc.offset_x+acc.width)) / local_work_size[0];
        global_work_size[0] = (int)ceilf((float)
                (acc.img->width - (idle_left + idle_right) * local_work_size[0])
                / (local_work_size[0]*2))*local_work_size[0];
        // update number of blocks
        num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

        // set last argument: block offset in pixels
        idle_left *= local_work_size[0];
        hipaccSetKernelArg(kernel2D, 9, sizeof(unsigned int), &idle_left);
    }

    hipaccLaunchKernel(kernel2D, global_work_size, local_work_size);


    // second step: reduce partial blocks on GPU
    // this is done in one shot, so no additional memory is required, i.e. the
    // same array can be used for the input and output array
    // block.x is fixed, either max_threads or power of two
    local_work_size[0] = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2)
        : max_threads;
    global_work_size[0] = local_work_size[0];
    global_work_size[1] = 1;
    // calculate the number of pixels reduced per thread
    int num_steps = (num_blocks + (local_work_size[0] - 1)) / (local_work_size[0]);

    hipaccSetKernelArg(kernel1D, 0, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel1D, 1, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel1D, 2, sizeof(unsigned int), &num_blocks);
    hipaccSetKernelArg(kernel1D, 3, sizeof(unsigned int), &num_steps);

    hipaccLaunchKernel(kernel1D, global_work_size, local_work_size);

    // get reduced value
    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T), &result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}
// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccAccessor acc(img);
    return hipaccApplyReduction<T>(kernel2D, kernel1D, acc, max_threads, pixels_per_thread);
}


// Perform exploration of global reduction and return result
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D, const HipaccAccessor &acc, unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T result;       // host result

    unsigned int num_blocks = (int)ceilf((float)(acc.img->width)/(max_threads*2))*acc.height;
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    std::cerr << "<HIPACC:> Exploring pixels per thread for '" << kernel2D << ", " << kernel1D << "'" << std::endl;

    float opt_time = FLT_MAX;
    int opt_ppt = 1;
    for (size_t ppt=1; ppt<=acc.height; ++ppt) {
        std::vector<float> times;
        std::stringstream num_ppt_ss;
        std::stringstream num_bs_ss;
        num_ppt_ss << ppt;
        num_bs_ss << max_threads;

        std::string compile_options = "-D PPT=" + num_ppt_ss.str() + " -D BS=" + num_bs_ss.str() + " -I./include ";
        compile_options += "-D BSX_EXPLORE=64 -D BSY_EXPLORE=1 ";
        cl_kernel exploreReduction2D = hipaccBuildProgramAndKernel(filename, kernel2D, false, false, false, compile_options);
        cl_kernel exploreReduction1D = hipaccBuildProgramAndKernel(filename, kernel1D, false, false, false, compile_options);

        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            // first step: reduce image (region) into linear memory
            size_t local_work_size[2];
            local_work_size[0] = max_threads;
            local_work_size[1] = 1;
            size_t global_work_size[2];
            global_work_size[0] = (int)ceilf((float)(acc.img->width)/(local_work_size[0]*2))*local_work_size[0];
            global_work_size[1] = (int)ceilf((float)(acc.height)/(local_work_size[1]*ppt))*local_work_size[1];
            num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

            hipaccSetKernelArg(exploreReduction2D, 0, sizeof(cl_mem), &acc.img->mem);
            hipaccSetKernelArg(exploreReduction2D, 1, sizeof(cl_mem), &output);
            hipaccSetKernelArg(exploreReduction2D, 2, sizeof(unsigned int), &acc.img->width);
            hipaccSetKernelArg(exploreReduction2D, 3, sizeof(unsigned int), &acc.img->height);
            hipaccSetKernelArg(exploreReduction2D, 4, sizeof(unsigned int), &acc.img->stride);
            // check if the reduction is applied to the whole image
            if ((acc.offset_x || acc.offset_y) &&
                (acc.width!=acc.img->width || acc.height!=acc.img->height)) {
                hipaccSetKernelArg(exploreReduction2D, 5, sizeof(unsigned int), &acc.offset_x);
                hipaccSetKernelArg(exploreReduction2D, 6, sizeof(unsigned int), &acc.offset_y);
                hipaccSetKernelArg(exploreReduction2D, 7, sizeof(unsigned int), &acc.width);
                hipaccSetKernelArg(exploreReduction2D, 8, sizeof(unsigned int), &acc.height);

                // reduce iteration space by idle blocks
                unsigned int idle_left = acc.offset_x / local_work_size[0];
                unsigned int idle_right = (acc.img->width - (acc.offset_x+acc.width)) / local_work_size[0];
                global_work_size[0] = (int)ceilf((float)
                        (acc.img->width - (idle_left + idle_right) * local_work_size[0])
                        / (local_work_size[0]*2))*local_work_size[0];
                // update number of blocks
                num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

                // set last argument: block offset in pixels
                idle_left *= local_work_size[0];
                hipaccSetKernelArg(exploreReduction2D, 9, sizeof(unsigned int), &idle_left);
            }

            hipaccLaunchKernel(exploreReduction2D, global_work_size, local_work_size, false);
            float total_time = last_gpu_timing;

            // second step: reduce partial blocks on GPU
            global_work_size[1] = 1;
            while (num_blocks > 1) {
                local_work_size[0] = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) :
                    max_threads;
                global_work_size[0] = (int)ceilf((float)(num_blocks)/(local_work_size[0]*ppt))*local_work_size[0];

                hipaccSetKernelArg(exploreReduction1D, 0, sizeof(cl_mem), &output);
                hipaccSetKernelArg(exploreReduction1D, 1, sizeof(cl_mem), &output);
                hipaccSetKernelArg(exploreReduction1D, 2, sizeof(unsigned int), &num_blocks);
                hipaccSetKernelArg(exploreReduction1D, 3, sizeof(unsigned int), &ppt);

                hipaccLaunchKernel(exploreReduction1D, global_work_size, local_work_size, false);
                total_time += last_gpu_timing;

                num_blocks = global_work_size[0]/local_work_size[0];
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

        // release kernels
        err = clReleaseKernel(exploreReduction2D);
        checkErr(err, "clReleaseKernel()");
        err = clReleaseKernel(exploreReduction1D);
        checkErr(err, "clReleaseKernel()");
    }
    last_gpu_timing = opt_time;
    std::cerr << "<HIPACC:> Best unroll factor for reduction kernel '"
              << kernel2D << "/" << kernel1D << "': "
              << opt_ppt << ": " << opt_time << " ms" << std::endl;

    // get reduced value
    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T), &result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D, std::string kernel1D, const HipaccImage &img, unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, acc, max_threads, pixels_per_thread);
}


#ifndef SEGMENT_SIZE
# define SEGMENT_SIZE 128
#endif
template<typename T, typename T2>
T *hipaccApplyBinningSegmented(cl_kernel kernel2D, cl_kernel kernel1D, const HipaccAccessor &acc, unsigned int num_hists, unsigned int num_warps, unsigned int num_bins) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T *result = new T[num_bins];   // host result
    size_t wavefront;

    switch (Ctx.get_platform_names()[0]) {
      case AMD:
        wavefront = 64;
        break;
      case NVIDIA:
        wavefront = 32;
        break;
      default:
        wavefront = 1;
        break;
    }

    size_t local_work_size[2];
    local_work_size[0] = wavefront;
    local_work_size[1] = num_warps;
    size_t global_work_size[2];
    global_work_size[0] = local_work_size[0]*num_hists;
    global_work_size[1] = local_work_size[1]*((num_bins+SEGMENT_SIZE-1)/SEGMENT_SIZE);

    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_hists*num_bins, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    int offset = 0;
    hipaccSetKernelArg(kernel2D, offset++, sizeof(cl_mem), &acc.img->mem);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.width);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.height);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.img->stride);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &num_bins);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.offset_x);
    hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.offset_y);

    hipaccLaunchKernel(kernel2D, global_work_size, local_work_size);

    local_work_size[0] = wavefront;
    local_work_size[1] = 1;
    global_work_size[0] = max(local_work_size[0], (size_t)num_bins);
    global_work_size[1] = 1;

    offset = 0;
    hipaccSetKernelArg(kernel1D, offset++, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel1D, offset++, sizeof(unsigned int), &num_bins);

    hipaccLaunchKernel(kernel1D, global_work_size, local_work_size);

    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T)*num_bins, result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}


template<typename T>
HipaccImage hipaccCreatePyramidImage(const HipaccImage &base, size_t width, size_t height) {
  switch (base->mem_type) {
    case Array2D:
      return hipaccCreateImage<T>(NULL, width, height);

    case Global:
      if (base->alignment > 0) {
        return hipaccCreateBuffer<T>(NULL, width, height, base->alignment);
      } else {
        return hipaccCreateBuffer<T>(NULL, width, height);
      }

    default:
      assert("Memory type is not supported for target OpenCL");
      return base;
  }
}


#endif  // __HIPACC_CL_TPP__

