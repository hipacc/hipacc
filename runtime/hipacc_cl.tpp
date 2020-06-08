//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef __HIPACC_CL_TPP__
#define __HIPACC_CL_TPP__

template <typename T>
HipaccImageOpenCL createImage(T *host_mem, cl_mem mem, size_t width,
                              size_t height, size_t stride, size_t alignment,
                              hipaccMemoryType mem_type) {
  HipaccImageOpenCL img = std::make_shared<HipaccImageOpenCLRaw>(
      width, height, stride, alignment, sizeof(T), mem, mem_type);

  hipaccWriteMemory(img, host_mem ? host_mem : (T *)img->get_host_memory());
  return img;
}

template <typename T>
cl_mem createBuffer(size_t stride, size_t height, cl_mem_flags flags) {
  HipaccContext &Ctx = HipaccContext::getInstance();
  cl_int err = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(Ctx.get_contexts()[0], flags,
                                 sizeof(T) * stride * height, NULL, &err);
  checkErr(err, "clCreateBuffer()");
  return buffer;
}

// Allocate memory with alignment specified
template <typename T>
HipaccImageOpenCL hipaccCreateBuffer(T *host_mem, size_t width, size_t height,
                                     size_t alignment) {
  // alignment has to be a multiple of sizeof(T)
  alignment = (size_t)ceilf((float)alignment / sizeof(T)) * sizeof(T);
  size_t stride = (size_t)ceilf((float)(width) / (alignment / sizeof(T))) *
                  (alignment / sizeof(T));

  cl_mem buffer = createBuffer<T>(stride, height, CL_MEM_READ_WRITE);
  return createImage(host_mem, buffer, width, height, stride, alignment);
}

// Allocate memory without any alignment considerations
template <typename T>
HipaccImageOpenCL hipaccCreateBuffer(T *host_mem, size_t width, size_t height) {
  cl_mem buffer = createBuffer<T>(width, height, CL_MEM_READ_WRITE);
  return createImage(host_mem, buffer, width, height, width, 0);
}

// Allocate constant buffer
template <typename T>
HipaccImageOpenCL hipaccCreateBufferConstant(T *host_mem, size_t width,
                                             size_t height) {
  cl_mem buffer = createBuffer<T>(width, height, CL_MEM_READ_ONLY);
  return createImage(host_mem, buffer, width, height, width, 0);
}

// Allocate image - no alignment can be specified
template <typename T>
HipaccImageOpenCL hipaccCreateImage(T *host_mem, size_t width, size_t height,
                                    cl_channel_type channel_type,
                                    cl_channel_order channel_order) {
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

  cl_mem image = clCreateImage(Ctx.get_contexts()[0], flags, &image_format,
                               &image_desc, NULL, &err);
  checkErr(err, "clCreateImage()");
#else
  cl_mem image = clCreateImage2D(Ctx.get_contexts()[0], flags, &image_format,
                                 width, height, 0, NULL, &err);
  checkErr(err, "clCreateImage2D()");
#endif

  return createImage(host_mem, image, width, height, width, 0,
                     hipaccMemoryType::Array2D);
}

// Write to memory
template <typename T>
void hipaccWriteMemory(HipaccImageOpenCL &img, T *host_mem, int num_device) {
  if (host_mem == NULL)
    return;

  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if ((char *)host_mem != img->get_host_memory())
    std::copy(host_mem, host_mem + width * height, (T *)img->get_host_memory());

  HipaccContext &Ctx = HipaccContext::getInstance();
  cl_int err = CL_SUCCESS;
  if (img->get_mem_type() >= hipaccMemoryType::Array2D) {
    const size_t origin[] = {0, 0, 0};
    const size_t region[] = {width, height, 1};
    // no stride supported for images in OpenCL
    const size_t input_row_pitch = width * sizeof(T);
    const size_t input_slice_pitch = 0;

    err = clEnqueueWriteImage(Ctx.get_command_queues()[num_device],
                              (cl_mem)img->get_device_memory(), CL_FALSE,
                              origin, region, input_row_pitch,
                              input_slice_pitch, host_mem, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueWriteImage()");
  } else {
    if (stride > width) {
      for (size_t i = 0; i < height; ++i) {
        err |= clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device],
                                    (cl_mem)img->get_device_memory(), CL_FALSE,
                                    i * sizeof(T) * stride, sizeof(T) * width,
                                    &host_mem[i * width], 0, NULL, NULL);
      }
    } else {
      err = clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device],
                                 (cl_mem)img->get_device_memory(), CL_FALSE, 0,
                                 sizeof(T) * width * height, host_mem, 0, NULL,
                                 NULL);
    }
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueWriteBuffer()");
  }
}

// Read from memory
template <typename T>
T *hipaccReadMemory(const HipaccImageOpenCL &img, int num_device) {
  cl_int err = CL_SUCCESS;
  HipaccContext &Ctx = HipaccContext::getInstance();

  if (img->get_mem_type() >= hipaccMemoryType::Array2D) {
    const size_t origin[] = {0, 0, 0};
    const size_t region[] = {img->get_width(), img->get_height(), 1};
    // no stride supported for images in OpenCL
    const size_t row_pitch = img->get_width() * sizeof(T);
    const size_t slice_pitch = 0;

    err = clEnqueueReadImage(Ctx.get_command_queues()[num_device],
                             (cl_mem)img->get_device_memory(), CL_FALSE, origin,
                             region, row_pitch, slice_pitch,
                             (T *)img->get_host_memory(), 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueReadImage()");
  } else {
    size_t width = img->get_width();
    size_t height = img->get_height();
    size_t stride = img->get_stride();

    if (stride > width) {
      for (size_t i = 0; i < height; ++i) {
        err |= clEnqueueReadBuffer(Ctx.get_command_queues()[num_device],
                                   (cl_mem)img->get_device_memory(), CL_FALSE,
                                   i * sizeof(T) * stride, sizeof(T) * width,
                                   &((T *)img->get_host_memory())[i * width], 0,
                                   NULL, NULL);
      }
    } else {
      err = clEnqueueReadBuffer(Ctx.get_command_queues()[num_device],
                                (cl_mem)img->get_device_memory(), CL_FALSE, 0,
                                sizeof(T) * width * height,
                                (T *)img->get_host_memory(), 0, NULL, NULL);
    }
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueReadBuffer()");
  }

  return (T *)img->get_host_memory();
}

// Infer non-const Domain from non-const Mask
template <typename T>
void hipaccWriteDomainFromMask(HipaccImageOpenCL &dom, T *host_mem) {
  size_t size = dom->get_width() * dom->get_height();
  uchar *dom_mem = new uchar[size];

  for (size_t i = 0; i < size; ++i) {
    dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
  }

  hipaccWriteMemory(dom, dom_mem);

  delete[] dom_mem;
}

// Set a single argument of a kernel
template <typename T>
void hipaccSetKernelArg(cl_kernel kernel, unsigned int num, size_t size,
                        T *param) {
  cl_int err = clSetKernelArg(kernel, num, size, param);
  checkErr(err, "clSetKernelArg()");
}

// Perform global reduction and return result
template <typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D,
                       const HipaccAccessor &acc, unsigned int max_threads,
                       unsigned int pixels_per_thread, HipaccExecutionParameterOpenCL ep,
                       bool print_timing) {
  HipaccContext &Ctx = HipaccContext::getInstance();
  cl_command_queue cg{ ep ? ep->get_command_queue() : Ctx.get_command_queues()[0] };
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  cl_int err = CL_SUCCESS;
  cl_mem output; // GPU memory for reduction
  T result;      // host result

  // first step: reduce image (region) into linear memory
  size_t local_work_size[2];
  local_work_size[0] = max_threads;
  local_work_size[1] = 1;
  size_t global_work_size[2];
  global_work_size[0] =
      (int)ceilf((float)(acc.img->get_width()) / (local_work_size[0] * 2)) *
      local_work_size[0];
  global_work_size[1] = (int)ceilf((float)(acc.height) /
                                   (local_work_size[1] * pixels_per_thread)) *
                        local_work_size[1];

  unsigned int num_blocks = (global_work_size[0] / local_work_size[0]) *
                            (global_work_size[1] / local_work_size[1]);
  output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T) * num_blocks,
                          NULL, &err);
  checkErr(err, "clCreateBuffer()");

  auto accImgDevMem = acc.img->get_device_memory();
  hipaccSetKernelArg(kernel2D, 0, sizeof(cl_mem), &accImgDevMem);
  hipaccSetKernelArg(kernel2D, 1, sizeof(cl_mem), &output);
  auto accImgWidth = acc.img->get_width();
  auto accImgHeight = acc.img->get_height();
  auto accImgStride = acc.img->get_stride();
  hipaccSetKernelArg(kernel2D, 2, sizeof(unsigned int), &accImgWidth);
  hipaccSetKernelArg(kernel2D, 3, sizeof(unsigned int), &accImgHeight);
  hipaccSetKernelArg(kernel2D, 4, sizeof(unsigned int), &accImgStride);
  // check if the reduction is applied to the whole image
  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    hipaccSetKernelArg(kernel2D, 5, sizeof(unsigned int), &acc.offset_x);
    hipaccSetKernelArg(kernel2D, 6, sizeof(unsigned int), &acc.offset_y);
    hipaccSetKernelArg(kernel2D, 7, sizeof(unsigned int), &acc.width);
    hipaccSetKernelArg(kernel2D, 8, sizeof(unsigned int), &acc.height);

    // reduce iteration space by idle blocks
    unsigned int idle_left = acc.offset_x / local_work_size[0];
    unsigned int idle_right =
        (acc.img->get_width() - (acc.offset_x + acc.width)) / local_work_size[0];
    global_work_size[0] =
        (int)ceilf((float)(acc.img->get_width() -
                           (idle_left + idle_right) * local_work_size[0]) /
                   (local_work_size[0] * 2)) *
        local_work_size[0];
    // update number of blocks
    num_blocks = (global_work_size[0] / local_work_size[0]) *
                 (global_work_size[1] / local_work_size[1]);

    // set last argument: block offset in pixels
    idle_left *= local_work_size[0];
    hipaccSetKernelArg(kernel2D, 9, sizeof(unsigned int), &idle_left);
  }

  cl_event event_start, event_end;
  if (print_timing) {
#ifdef CL_VERSION_1_2 // or later
    err = clEnqueueMarkerWithWaitList(cg, 0, nullptr, &event_start);
    checkErr(err, "clEnqueueMarkerWithWaitList()");
#else
    err = clEnqueueMarker(cg, &event_start);
    checkErr(err, "clEnqueueMarker()");
#endif
  }

  hipaccLaunchKernel(kernel2D, global_work_size, local_work_size, ep, false);

  // second step: reduce partial blocks on GPU
  // this is done in one shot, so no additional memory is required, i.e. the
  // same array can be used for the input and output array
  // block.x is fixed, either max_threads or next multiple of 32
  local_work_size[0] = (num_blocks < max_threads)
                           ? ((num_blocks + 32 - 1) / 32) * 32
                           : max_threads;
  global_work_size[0] = local_work_size[0];
  global_work_size[1] = 1;
  // calculate the number of pixels reduced per thread
  int num_steps =
      (num_blocks + (local_work_size[0] - 1)) / (local_work_size[0]);

  hipaccSetKernelArg(kernel1D, 0, sizeof(cl_mem), &output);
  hipaccSetKernelArg(kernel1D, 1, sizeof(cl_mem), &output);
  hipaccSetKernelArg(kernel1D, 2, sizeof(unsigned int), &num_blocks);
  hipaccSetKernelArg(kernel1D, 3, sizeof(unsigned int), &num_steps);

  hipaccLaunchKernel(kernel1D, global_work_size, local_work_size, ep, false);

  if (print_timing) {
#ifdef CL_VERSION_1_2 // or later
    err = clEnqueueMarkerWithWaitList(cg, 0, nullptr, &event_end);
    checkErr(err, "clEnqueueMarkerWithWaitList()");
#else
    err = clEnqueueMarker(cg, &event_end);
    checkErr(err, "clEnqueueMarker()");
#endif
  }

  // get reduced value
  err = clEnqueueReadBuffer(cg, output, CL_FALSE, 0,
                            sizeof(T), &result, 0, NULL, NULL);
  err |= clFinish(cg);
  checkErr(err, "clEnqueueReadBuffer()");

  err = clReleaseMemObject(output);
  checkErr(err, "clReleaseMemObject()");

  if (print_timing) {
    cl_ulong end, start;
    float last_gpu_timing;

    err = clGetEventProfilingInfo(event_end, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event_start, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start = (cl_ulong)(start * 1e-3);
    end = (cl_ulong)(end * 1e-3);

    err = clReleaseEvent(event_start);
    checkErr(err, "clReleaseEvent()");

    err = clReleaseEvent(event_end);
    checkErr(err, "clReleaseEvent()");

    last_gpu_timing = (end - start) * 1.0e-3f;
    HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);

    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (reduce): " + std::to_string(last_gpu_timing) + "(ms)");
  }

  return result;
}
// Perform global reduction using memory fence operations and return result
template <typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D,
                       const HipaccImageOpenCL &img, unsigned int max_threads,
                       unsigned int pixels_per_thread, HipaccExecutionParameterOpenCL ep,
                       bool print_timing) {
  HipaccAccessor acc(img);
  return hipaccApplyReduction<T>(kernel2D, kernel1D, acc, max_threads,
                                 pixels_per_thread, ep, print_timing);
}

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 128
#endif
template <typename T, typename T2>
T *hipaccApplyBinningSegmented(cl_kernel kernel2D, cl_kernel kernel1D,
                               const HipaccAccessor &acc,
                               unsigned int num_warps, unsigned int num_hists,
                               unsigned int num_bins, HipaccExecutionParameterOpenCL ep,
                               bool print_timing) {
  HipaccContext &Ctx = HipaccContext::getInstance();
  cl_command_queue cg{ ep ? ep->get_command_queue() : Ctx.get_command_queues()[0] };
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  cl_int err = CL_SUCCESS;
  cl_mem output;               // GPU memory for reduction
  T *result = new T[num_bins]; // host result
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
  global_work_size[0] = local_work_size[0] * num_hists;
  global_work_size[1] =
      local_work_size[1] * ((num_bins + SEGMENT_SIZE - 1) / SEGMENT_SIZE);

  output = clCreateBuffer(Ctx.get_contexts()[0], flags,
                          sizeof(T) * num_hists * num_bins, NULL, &err);
  checkErr(err, "clCreateBuffer()");

  int offset = 0;

  auto accImgDevMem = acc.img->get_device_memory();
  hipaccSetKernelArg(kernel2D, offset++, sizeof(cl_mem), &accImgDevMem);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(cl_mem), &output);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.width);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.height);
  auto accImgStride = acc.img->get_stride();
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int),
                     &accImgStride);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &num_bins);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.offset_x);
  hipaccSetKernelArg(kernel2D, offset++, sizeof(unsigned int), &acc.offset_y);

  cl_event event_start, event_end;
  if (print_timing) {
#ifdef CL_VERSION_1_2 // or later
    err = clEnqueueMarkerWithWaitList(cg, 0, nullptr, &event_start);
    checkErr(err, "clEnqueueMarkerWithWaitList()");
#else
    err = clEnqueueMarker(cg, &event_start);
    checkErr(err, "clEnqueueMarker()");
#endif
  }

  hipaccLaunchKernel(kernel2D, global_work_size, local_work_size, ep, false);

  local_work_size[0] = wavefront;
  local_work_size[1] = 1;
  global_work_size[0] = std::max(local_work_size[0], (size_t)num_bins);
  global_work_size[1] = 1;

  offset = 0;
  hipaccSetKernelArg(kernel1D, offset++, sizeof(cl_mem), &output);
  hipaccSetKernelArg(kernel1D, offset++, sizeof(unsigned int), &num_bins);

  hipaccLaunchKernel(kernel1D, global_work_size, local_work_size, ep, false);

  if (print_timing) {
#ifdef CL_VERSION_1_2 // or later
    err = clEnqueueMarkerWithWaitList(cg, 0, nullptr, &event_end);
    checkErr(err, "clEnqueueMarkerWithWaitList()");
#else
    err = clEnqueueMarker(cg, &event_end);
    checkErr(err, "clEnqueueMarker()");
#endif
  }

  err = clEnqueueReadBuffer(cg, output, CL_FALSE, 0,
                            sizeof(T) * num_bins, result, 0, NULL, NULL);
  err |= clFinish(cg);
  checkErr(err, "clEnqueueReadBuffer()");

  err = clReleaseMemObject(output);
  checkErr(err, "clReleaseMemObject()");

  if (print_timing) {
    cl_ulong end, start;
    float last_gpu_timing;

    err = clGetEventProfilingInfo(event_end, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event_start, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start = (cl_ulong)(start * 1e-3);
    end = (cl_ulong)(end * 1e-3);

    err = clReleaseEvent(event_start);
    checkErr(err, "clReleaseEvent()");

    err = clReleaseEvent(event_end);
    checkErr(err, "clReleaseEvent()");

    last_gpu_timing = (end - start) * 1.0e-3f;
    HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);

    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (binning reduce): " + std::to_string(last_gpu_timing) + "(ms)");
  }

  return result;
}

// Allocate memory for Pyramid image
template <typename T>
HipaccImageOpenCL hipaccCreatePyramidImage(const HipaccImageOpenCL &base,
                                           size_t width, size_t height) {
  switch (base->get_mem_type()) {
  default:
    if (base->get_alignment() > 0) {
      return hipaccCreateBuffer<T>(NULL, width, height, base->get_alignment());
    } else {
      return hipaccCreateBuffer<T>(NULL, width, height);
    }
  case hipaccMemoryType::Array2D:
    return hipaccCreateBuffer<T>(NULL, width, height);
  }
}

template <typename T>
HipaccPyramidOpenCL hipaccCreatePyramid(const HipaccImageOpenCL &img,
                                        size_t depth) {
  HipaccPyramidOpenCL p(depth);
  p.add(img);

  size_t width = img->get_width() / 2;
  size_t height = img->get_height() / 2;
  for (size_t i = 1; i < depth; ++i) {
    assert(width * height > 0 && "Pyramid stages too deep for image size");
    p.add(hipaccCreatePyramidImage<T>(img, width, height));
    width /= 2;
    height /= 2;
  }
  return p;
}

#endif // __HIPACC_CL_TPP__
