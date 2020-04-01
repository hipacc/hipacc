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

#ifndef __HIPACC_CU_TPP__
#define __HIPACC_CU_TPP__

template <typename T>
HipaccImageCuda createImage(T *host_mem, void *mem, size_t width, size_t height,
                            size_t stride, size_t alignment,
                            hipaccMemoryType mem_type) {
  HipaccImageCuda img = std::make_shared<HipaccImageCudaRaw>(
      width, height, stride, alignment, sizeof(T), mem, mem_type);

  hipaccWriteMemory(img, host_mem ? host_mem : (T *)img->get_host_memory());
  return img;
}

template <typename T> T *createMemory(size_t stride, size_t height) {
  T *mem;
  cudaError_t err = cudaMalloc((void **)&mem, sizeof(T) * stride * height);
  checkErr(err, "cudaMalloc()");
  return mem;
}

// Allocate memory with alignment specified
template <typename T>
HipaccImageCuda hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                   size_t alignment) {
  // alignment has to be a multiple of sizeof(T)
  alignment = (int)ceilf((float)alignment / sizeof(T)) * sizeof(T);
  size_t stride = (int)ceilf((float)(width) / (alignment / sizeof(T))) *
                  (alignment / sizeof(T));

  T *mem = createMemory<T>(stride, height);
  return createImage(host_mem, (void *)mem, width, height, stride, alignment);
}

// Allocate memory without any alignment considerations
template <typename T>
HipaccImageCuda hipaccCreateMemory(T *host_mem, size_t width, size_t height) {
  T *mem = createMemory<T>(width, height);
  return createImage(host_mem, (void *)mem, width, height, width, 0);
}

// Allocate 2D array
template <typename T>
HipaccImageCuda hipaccCreateArray2D(T *host_mem, size_t width, size_t height) {
  cudaArray *array;
  int flags = cudaArraySurfaceLoadStore;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaError_t err =
      cudaMallocArray(&array, &channelDesc, width, height, flags); // MEM
  checkErr(err, "cudaMallocArray()");

  return createImage(host_mem, (void *)array, width, height, width, 0,
                     hipaccMemoryType::Array2D);
}

// Write to memory
template <typename T>
void hipaccWriteMemory(HipaccImageCuda &img, T *host_mem) {
  if (host_mem == NULL)
    return;

  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if ((char *)host_mem !=
      img->get_host_memory()) // copy if user provides host data
    std::copy(host_mem, host_mem + width * height, (T *)img->get_host_memory());

  if (img->get_mem_type() == hipaccMemoryType::Array2D ||
      img->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DToArray(
        (cudaArray *)img->get_device_memory(), 0, 0, host_mem,
        stride * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice);
    checkErr(err, "cudaMemcpy2DToArray()");
  } else {
    if (stride > width) {
      cudaError_t err = cudaMemcpy2D(
          img->get_device_memory(), stride * sizeof(T), host_mem,
          width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice);
      checkErr(err, "cudaMemcpy2D()");
    } else {
      cudaError_t err =
          cudaMemcpy(img->get_device_memory(), host_mem,
                     sizeof(T) * width * height, cudaMemcpyHostToDevice);
      checkErr(err, "cudaMemcpy()");
    }
  }
}

// Read from memory
template <typename T> T *hipaccReadMemory(const HipaccImageCuda &img) {
  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if (img->get_mem_type() == hipaccMemoryType::Array2D ||
      img->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DFromArray(
        (T *)img->get_host_memory(), stride * sizeof(T),
        (cudaArray *)img->get_device_memory(), 0, 0, width * sizeof(T), height,
        cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy2DFromArray()");
  } else {
    if (stride > width) {
      cudaError_t err =
          cudaMemcpy2D((T *)img->get_host_memory(), width * sizeof(T),
                       img->get_device_memory(), stride * sizeof(T),
                       width * sizeof(T), height, cudaMemcpyDeviceToHost);
      checkErr(err, "cudaMemcpy2D()");
    } else {
      cudaError_t err =
          cudaMemcpy((T *)img->get_host_memory(), img->get_device_memory(),
                     sizeof(T) * width * height, cudaMemcpyDeviceToHost);
      checkErr(err, "cudaMemcpy()");
    }
  }
  return (T *)img->get_host_memory();
}

// Bind memory to texture
template <typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex,
                       const HipaccImageCuda &img) {
  cudaError_t err = cudaSuccess;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

  switch (mem_type) {
  case hipaccMemoryType::Linear1D:
    assert(img->get_mem_type() <= hipaccMemoryType::Linear2D &&
           "expected linear memory");
    err = cudaBindTexture(NULL, tex, img->get_device_memory(), &channelDesc,
                          sizeof(T) * img->get_stride() * img->get_height());
    checkErr(err, "cudaBindTexture()");
    break;
  case hipaccMemoryType::Linear2D:
    assert(img->get_mem_type() <= hipaccMemoryType::Linear2D &&
           "expected linear memory");
    err = cudaBindTexture2D(NULL, tex, img->get_device_memory(), &channelDesc,
                            img->get_width(), img->get_height(), img->get_stride() * sizeof(T));
    checkErr(err, "cudaBindTexture2D()");
    break;
  case hipaccMemoryType::Array2D:
    assert(img->get_mem_type() == hipaccMemoryType::Array2D &&
           "expected Array2D memory");
    err = cudaBindTextureToArray(tex, (cudaArray *)img->get_device_memory(),
                                 &channelDesc);
    checkErr(err, "cudaBindTextureToArray()");
    break;
  default:
    assert(false && "wrong texture type");
  }
}

// Bind 2D array to surface
template <typename T>
void hipaccBindSurface(hipaccMemoryType mem_type, const surfaceReference *surf,
                       const HipaccImageCuda &img) {
  assert(mem_type == hipaccMemoryType::Surface && "wrong texture type");
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaError_t err = cudaBindSurfaceToArray(
      surf, (cudaArray *)img->get_device_memory(), &channelDesc);
  checkErr(err, "cudaBindSurfaceToArray()");
}

// Write to symbol
template <typename T>
void hipaccWriteSymbol(const void *symbol, T *host_mem, size_t width,
                       size_t height) {
  cudaError_t err =
      cudaMemcpyToSymbol(symbol, host_mem, sizeof(T) * width * height);
  checkErr(err, "cudaMemcpyToSymbol()");
}

// Read from symbol
template <typename T>
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string symbol_name,
                      size_t width, size_t height) {
  cudaError_t err =
      cudaMemcpyFromSymbol(host_mem, symbol, sizeof(T) * width * height);
  checkErr(err, "cudaMemcpyFromSymbol()");
}

// Infer non-const Domain from non-const Mask
template <typename T>
void hipaccWriteDomainFromMask(const void *symbol, T *host_mem, size_t width,
                               size_t height) {
  size_t size = width * height;
  uchar *dom_mem = new uchar[size];

  for (size_t i = 0; i < size; ++i) {
    dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
  }

  hipaccWriteSymbol<uchar>(symbol, dom_mem, width, height);

  delete[] dom_mem;
}

// Perform global reduction and return result
template <typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name,
                       const void *kernel1D, std::string kernel1D_name,
                       const HipaccAccessor &acc, unsigned int max_threads,
                       unsigned int pixels_per_thread,
                       const textureReference *tex) {
  T *output; // GPU memory for reduction
  T result;  // host result

  // first step: reduce image (region) into linear memory
  dim3 block(max_threads, 1);
  dim3 grid((int)ceilf((float)(acc.img->get_width()) / (block.x * 2)),
            (int)ceilf((float)(acc.height) / pixels_per_thread));
  unsigned int num_blocks = grid.x * grid.y;
  unsigned int idle_left = 0;

  cudaError_t err = cudaMalloc((void **)&output, sizeof(T) * num_blocks);
  checkErr(err, "cudaMalloc()");

  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    // reduce iteration space by idle blocks
    idle_left = acc.offset_x / block.x;
    unsigned int idle_right =
        (acc.img->get_width() - (acc.offset_x + acc.width)) / block.x;
    grid.x = (int)ceilf(
        (float)(acc.img->get_width() - (idle_left + idle_right) * block.x) /
        (block.x * 2));

    // update number of blocks
    num_blocks = grid.x * grid.y;
    idle_left *= block.x;
  }

  std::vector<void *> args_step1;
  switch (acc.img->get_mem_type()) {
  default:
  case hipaccMemoryType::Global: {
    auto accImgDevMem = acc.img->get_device_memory();
    args_step1.push_back(&accImgDevMem);
    break;
  }
  case hipaccMemoryType::Array2D:
    hipaccBindTexture<T>(hipaccMemoryType::Array2D, tex, *acc.img);
    break;
  }

  args_step1.push_back((void *)&output);
  auto accImgWidth = acc.img->get_width();
  auto accImgHeight = acc.img->get_height();
  auto accImgStride = acc.img->get_stride();
  args_step1.push_back((void *)&accImgWidth);
  args_step1.push_back((void *)&accImgHeight);
  args_step1.push_back((void *)&accImgStride);
  // check if the reduction is applied to the whole image
  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    args_step1.push_back((void *)&acc.offset_x);
    args_step1.push_back((void *)&acc.offset_y);
    args_step1.push_back((void *)&acc.width);
    args_step1.push_back((void *)&acc.height);
    args_step1.push_back((void *)&idle_left);
  }

  hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args_step1.data());

  // second step: reduce partial blocks on GPU
  // this is done in one shot, so no additional memory is required, i.e. the
  // same array can be used for the input and output array
  // block.x is fixed, either max_threads or multiple of 32
  block.x = (num_blocks < max_threads) ? ((num_blocks + 32 - 1) / 32) * 32
                                       : max_threads;
  grid.x = 1;
  grid.y = 1;
  // calculate the number of pixels reduced per thread
  int num_steps = (num_blocks + (block.x - 1)) / (block.x);

  std::vector<void *> args_step2;
  args_step2.push_back((void *)&output);
  args_step2.push_back((void *)&output);
  args_step2.push_back((void *)&num_blocks);
  args_step2.push_back((void *)&num_steps);

  hipaccLaunchKernel(kernel1D, kernel1D_name, grid, block, args_step2.data());

  // get reduced value
  err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  return result;
}

// Perform global reduction and return result
template <typename T>
T hipaccApplyReduction(const void *kernel2D, std::string kernel2D_name,
                       const void *kernel1D, std::string kernel1D_name,
                       const HipaccImageCuda &img, unsigned int max_threads,
                       unsigned int pixels_per_thread,
                       const textureReference *tex) {
  HipaccAccessor acc(img);
  return hipaccApplyReduction<T>(kernel2D, kernel2D_name, kernel1D,
                                 kernel1D_name, acc, max_threads,
                                 pixels_per_thread, tex);
}

// Perform global reduction using memory fence operations and return result
template <typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D,
                                  std::string kernel2D_name,
                                  const HipaccAccessor &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  const textureReference *tex) {
  T *output; // GPU memory for reduction
  T result;  // host result

  // single step reduction: reduce image (region) into linear memory and
  // reduce the linear memory using memory fence operations
  dim3 block(max_threads, 1);
  dim3 grid((int)ceilf((float)(acc.img->get_width()) / (block.x * 2)),
            (int)ceilf((float)(acc.height) / pixels_per_thread));
  unsigned int num_blocks = grid.x * grid.y;
  unsigned int idle_left = 0;

  cudaError_t err = cudaMalloc((void **)&output, sizeof(T) * num_blocks);
  checkErr(err, "cudaMalloc()");

  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    // reduce iteration space by idle blocks
    idle_left = acc.offset_x / block.x;
    unsigned int idle_right =
        (acc.img->get_width() - (acc.offset_x + acc.width)) / block.x;
    grid.x = (int)ceilf(
        (float)(acc.img->get_width() - (idle_left + idle_right) * block.x) /
        (block.x * 2));

    // update number of blocks
    idle_left *= block.x;
  }

  std::vector<void *> args;
  switch (acc.img->get_mem_type()) {
  default:
  case hipaccMemoryType::Global: {
    auto accImgDevMem = acc.img->get_device_memory();
    args.push_back(&accImgDevMem);
    break;
  }
  case hipaccMemoryType::Array2D:
    hipaccBindTexture<T>(hipaccMemoryType::Array2D, tex, acc.img);
    break;
  }

  args.push_back((void *)&output);
  auto accImgWidth = acc.img->get_width();
  auto accImgHeight = acc.img->get_height();
  auto accImgStride = acc.img->get_stride();
  args.push_back((void *)&accImgWidth);
  args.push_back((void *)&accImgHeight);
  args.push_back((void *)&accImgStride);
  // check if the reduction is applied to the whole image
  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    args.push_back((void *)&acc.offset_x);
    args.push_back((void *)&acc.offset_y);
    args.push_back((void *)&acc.width);
    args.push_back((void *)&acc.height);
    args.push_back((void *)&idle_left);
  }

  hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args.data());

  err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  return result;
}

// Perform global reduction using memory fence operations and return result
template <typename T>
T hipaccApplyReductionThreadFence(const void *kernel2D,
                                  std::string kernel2D_name,
                                  const HipaccImageCuda &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  const textureReference *tex) {
  HipaccAccessor acc(img);
  return hipaccApplyReductionThreadFence<T>(
      kernel2D, kernel2D_name, acc, max_threads, pixels_per_thread, tex);
}

// Perform global reduction and return result
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccAccessor &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  hipacc_tex_info tex_info, int cc) {
  T *output; // GPU memory for reduction
  T result;  // host result

  unsigned int num_blocks =
      (int)ceilf((float)(acc.img->get_width()) / (max_threads * 2)) * acc.height;
  unsigned int idle_left = 0;

  cudaError_t err = cudaMalloc((void **)&output, sizeof(T) * num_blocks);
  checkErr(err, "cudaMalloc()");

  auto accImgDevMem = acc.img->get_device_memory();
  auto accImgWidth = acc.img->get_width();
  auto accImgHeight = acc.img->get_height();
  auto accImgStride = acc.img->get_stride();
  void *argsReduction2D[] = {(void *)&accImgDevMem,    (void *)&output,
                             (void *)&accImgWidth,  (void *)&accImgHeight,
                             (void *)&accImgStride, (void *)&acc.offset_x,
                             (void *)&acc.offset_y,    (void *)&acc.width,
                             (void *)&acc.height,      (void *)&idle_left};
  void *argsReduction2DArray[] = {
      (void *)&output,          (void *)&accImgWidth,
      (void *)&accImgHeight, (void *)&accImgStride,
      (void *)&acc.offset_x,    (void *)&acc.offset_y,
      (void *)&acc.width,       (void *)&acc.height,
      (void *)&idle_left};

  hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                          "<HIPACC:> Exploring pixels per thread for '" +
                              kernel2D + ", " + kernel1D + "'");

  float opt_time = FLT_MAX;
  int opt_ppt = 1;
  for (size_t ppt = 1; ppt <= acc.height; ++ppt) {
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

    for (size_t i = 0; i < HIPACC_NUM_ITERATIONS; ++i) {
      dim3 block(max_threads, 1);
      dim3 grid((int)ceilf((float)(acc.img->get_width()) / (block.x * 2)),
                (int)ceilf((float)(acc.height) / ppt));
      num_blocks = grid.x * grid.y;

      // check if the reduction is applied to the whole image
      if ((acc.offset_x || acc.offset_y) &&
          (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
        // reduce iteration space by idle blocks
        idle_left = acc.offset_x / block.x;
        unsigned int idle_right =
            (acc.img->get_width() - (acc.offset_x + acc.width)) / block.x;
        grid.x = (int)ceilf(
            (float)(acc.img->get_width() - (idle_left + idle_right) * block.x) /
            (block.x * 2));

        // update number of blocks
        num_blocks = grid.x * grid.y;
        idle_left *= block.x;
      }

      // bind texture to CUDA array
      CUtexref texImage;
      if (tex_info.tex_type == hipaccMemoryType::Array2D) {
        hipaccGetTexRef(texImage, modReduction, tex_info.name);
        hipaccBindTextureDrv(texImage, tex_info.image, tex_info.format,
                             tex_info.tex_type);
        hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block,
                           argsReduction2DArray, false);
      } else {
        hipaccLaunchKernel(exploreReduction2D, kernel2D, grid, block,
                           argsReduction2D, false);
      }
      float total_time = hipaccCudaTiming.get_last_kernel_timing();

      // second step: reduce partial blocks on GPU
      grid.y = 1;
      while (num_blocks > 1) {
        block.x = (num_blocks < max_threads) ? ((num_blocks + 32 - 1) / 32) * 32
                                             : max_threads;

        grid.x = (int)ceilf((float)(num_blocks) / (block.x * ppt));

        void *argsReduction1D[] = {(void *)&output, (void *)&output,
                                   (void *)&num_blocks, (void *)&ppt};

        hipaccLaunchKernel(exploreReduction1D, kernel1D, grid, block,
                           argsReduction1D, false);
        total_time += hipaccCudaTiming.get_last_kernel_timing();

        num_blocks = grid.x;
      }
      times.push_back(total_time);
    }

    std::sort(times.begin(), times.end());
    hipaccCudaTiming.set_gpu_timing(times[times.size() / 2]);

    if (hipaccCudaTiming.get_last_kernel_timing() < opt_time) {
      opt_time = hipaccCudaTiming.get_last_kernel_timing();
      opt_ppt = ppt;
    }

    // print timing
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> PPT: " + std::to_string(ppt) + ", " +
            std::to_string(hipaccCudaTiming.get_last_kernel_timing()) + " | " +
            std::to_string(times.front()) + " | " +
            std::to_string(times.back()) + " (median(" +
            std::to_string(HIPACC_NUM_ITERATIONS) +
            ") | minimum | maximum) ms");

    // cleanup
    CUresult err = cuModuleUnload(modReduction);
    checkErrDrv(err, "cuModuleUnload()");
  }
  hipaccCudaTiming.set_gpu_timing(opt_time);
  hipaccRuntimeLogTrivial(
      hipaccRuntimeLogLevel::INFO,
      "<HIPACC:> Best unroll factor for reduction kernel '" + kernel2D + "/" +
          kernel1D + "': " + std::to_string(opt_ppt) + ":" +
          std::to_string(opt_time) + "ms");

  // get reduced value
  err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  return result;
}
template <typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
                                  std::string kernel1D,
                                  const HipaccImageCuda &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  hipacc_tex_info tex_info, int cc) {
  HipaccAccessor acc(img);
  return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, acc,
                                            max_threads, pixels_per_thread,
                                            tex_info, cc);
}

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 128
#define MAX_SEGMENTS 512 // equals 65k bins (MAX_SEGMENTS*SEGMENT_SIZE)
#endif
template <typename T, typename T2>
T *hipaccApplyBinningSegmented(const void *kernel2D, std::string kernel2D_name,
                               HipaccAccessor &acc, unsigned int num_hists,
                               unsigned int num_warps, unsigned int num_bins,
                               const textureReference *tex) {
  T *output;                   // GPU memory for reduction
  T *result = new T[num_bins]; // host result

  dim3 grid(num_hists, (num_bins + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
  dim3 block(32, num_warps);

  cudaError_t err =
      cudaMalloc((void **)&output, sizeof(T) * num_hists * num_bins);
  checkErr(err, "cudaMalloc()");

  std::vector<void *> args;
  switch (acc.img->get_mem_type()) {
  default:
  case hipaccMemoryType::Global: {
    auto accImgDevMem = acc.img->get_device_memory();
    args.push_back(&accImgDevMem);
    break;
  }
  case hipaccMemoryType::Array2D:
    hipaccBindTexture<T>(hipaccMemoryType::Array2D, tex, acc.img);
    break;
  }

  args.push_back((void *)&output);
  args.push_back((void *)&acc.width);
  args.push_back((void *)&acc.height);
  auto accImgStride = acc.img->get_stride();
  args.push_back((void *)&accImgStride);
  args.push_back((void *)&num_bins);
  args.push_back((void *)&acc.offset_x);
  args.push_back((void *)&acc.offset_y);

  hipaccLaunchKernel(kernel2D, kernel2D_name, grid, block, args.data());

  err =
      cudaMemcpy(result, output, sizeof(T) * num_bins, cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  return result;
}

// Allocate memory for Pyramid image
template <typename T>
HipaccImageCuda hipaccCreatePyramidImage(const HipaccImageCuda &base,
                                         size_t width, size_t height) {
  switch (base->get_mem_type()) {
  default:
    if (base->get_alignment() > 0) {
      return hipaccCreateMemory<T>(NULL, width, height, base->get_alignment());
    } else {
      return hipaccCreateMemory<T>(NULL, width, height);
    }
  case hipaccMemoryType::Array2D:
    return hipaccCreateArray2D<T>(NULL, width, height);
  }
}

template <typename T>
HipaccPyramidCuda hipaccCreatePyramid(const HipaccImageCuda &img,
                                      size_t depth) {
  HipaccPyramidCuda p(depth);
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

#endif // __HIPACC_CU_TPP__
