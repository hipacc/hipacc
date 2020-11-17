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
HipaccImageCuda<T> createImage(T *host_mem, T *mem, size_t width, size_t height,
                            size_t stride, size_t alignment,
                            hipaccMemoryType mem_type) {
  HipaccImageCuda<T> img = std::make_shared<HipaccImageCudaRaw<T>>(
      static_cast<int>(width), static_cast<int>(height), static_cast<int>(stride), static_cast<int>(alignment), mem, mem_type);

  hipaccWriteMemory<T>(img, host_mem ? host_mem : img->get_host_memory());
  return img;
}

template <typename T> T *createMemory(size_t stride, size_t height) {
  T *mem{};
  cudaError_t err = cudaMalloc((void **)&mem, sizeof(T) * stride * height);
  checkErr(err, "cudaMalloc()");
  return mem;
}

// Allocate memory with alignment specified
template <typename T>
HipaccImageCuda<T> hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                   size_t alignment) {
  // alignment has to be a multiple of sizeof(T)
  alignment = (int)ceilf((float)alignment / sizeof(T)) * sizeof(T);
  size_t stride = (int)ceilf((float)(width) / (alignment / sizeof(T))) *
                  (alignment / sizeof(T));

  T *device_mem = createMemory<T>(stride, height);
  return createImage<T>(host_mem, device_mem, width, height, stride, alignment);
}

// Allocate memory without any alignment considerations
template <typename T>
HipaccImageCuda<T> hipaccCreateMemory(T *host_mem, size_t width, size_t height) {
  T *mem = createMemory<T>(width, height);
  return createImage<T>(host_mem, mem, width, height, width, 0);
}

// Allocate 2D array
template <typename T>
HipaccImageCuda<T> hipaccCreateArray2D(T *host_mem, size_t width, size_t height) {
  cudaArray *array{};
  int flags = cudaArraySurfaceLoadStore;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaError_t err =
      cudaMallocArray(&array, &channelDesc, width, height, flags); // MEM
  checkErr(err, "cudaMallocArray()");

  return createImage<T>(host_mem, (T*)array, width, height, width, 0,
                     hipaccMemoryType::Array2D);
}

// Write to memory
template <typename T>
void hipaccWriteMemory(HipaccImageCuda<T> &img, T *host_mem) {
  if (!img || host_mem == nullptr)
    return;

  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if (host_mem != img->get_host_memory()) // copy if user provides host data
    std::copy(host_mem, host_mem + width * height, img->get_host_memory());

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

// Write to memory
template <typename T>
void hipaccWriteMemoryCudaGraph(HipaccImageCuda<T> &img, T *host_mem, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaMemcpy3DParms &memcpyArgs) {
  if (!img || host_mem == nullptr)
    return;

  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();
  if (host_mem != img->get_host_memory()) // copy if user provides host data
    std::copy(host_mem, host_mem + width * height, img->get_host_memory());

  cudaError_t err;
  if (img->get_mem_type() == hipaccMemoryType::Array2D ||
      img->get_mem_type() == hipaccMemoryType::Surface) {
    // TODO
    assert(false && "hipacc memory type not supported for cuda graph");
  } else {
    if (stride > width) {
      memcpyArgs.srcArray = NULL;
      memcpyArgs.srcPos = make_cudaPos(0, 0, 0);
      memcpyArgs.srcPtr = make_cudaPitchedPtr(host_mem, sizeof(T)*width, width, height);
      memcpyArgs.dstArray = NULL;
      memcpyArgs.dstPos = make_cudaPos(0, 0, 0);
      memcpyArgs.dstPtr = make_cudaPitchedPtr(img->get_device_memory(), sizeof(T)*stride, stride, height);
      memcpyArgs.extent = make_cudaExtent(sizeof(T)*width, sizeof(T)*height, 1);
      memcpyArgs.kind = cudaMemcpyHostToDevice;
    } else {
      memcpyArgs.srcArray = NULL;
      memcpyArgs.srcPos = make_cudaPos(0, 0, 0);
      memcpyArgs.srcPtr = make_cudaPitchedPtr(host_mem, sizeof(T)*width*height, width*height, 1);
      memcpyArgs.dstArray = NULL;
      memcpyArgs.dstPos = make_cudaPos(0, 0, 0);
      memcpyArgs.dstPtr = make_cudaPitchedPtr(img->get_device_memory(), sizeof(T)*width*height, width*height, 1);
      memcpyArgs.extent = make_cudaExtent(sizeof(T)*width*height, 1, 1);
      memcpyArgs.kind = cudaMemcpyHostToDevice;
    }
    if (graphNodeDeps.empty()) {
      err = cudaGraphAddMemcpyNode(&graphNode, graph, NULL, 0, &memcpyArgs);
      checkErr(err, (std::string("cudaGraphAddMemcpyNode(") + __FUNCTION__ + ")"));
    } else {
      err = cudaGraphAddMemcpyNode(&graphNode, graph, graphNodeDeps.data(), graphNodeDeps.size(), &memcpyArgs);
      checkErr(err, (std::string("cudaGraphAddMemcpyNode(") + __FUNCTION__ + ")"));
    }
  }
}

// Read from memory
template <typename T> T *hipaccReadMemory(const HipaccImageCuda<T> &img) {
  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if (img->get_mem_type() == hipaccMemoryType::Array2D ||
      img->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DFromArray(
        img->get_host_memory(), stride * sizeof(T),
        (cudaArray *)img->get_device_memory(), 0, 0, width * sizeof(T), height,
        cudaMemcpyDeviceToHost);
    checkErr(err, "cudaMemcpy2DFromArray()");
  } else {
    if (stride > width) {
      cudaError_t err =
          cudaMemcpy2D(img->get_host_memory(), width * sizeof(T),
                       img->get_device_memory(), stride * sizeof(T),
                       width * sizeof(T), height, cudaMemcpyDeviceToHost);
      checkErr(err, "cudaMemcpy2D()");
    } else {
      cudaError_t err =
          cudaMemcpy(img->get_host_memory(), img->get_device_memory(),
                     sizeof(T) * width * height, cudaMemcpyDeviceToHost);
      checkErr(err, "cudaMemcpy()");
    }
  }
  return img->get_host_memory();
}

// Read from memory
template <typename T> T *hipaccReadMemoryCudaGraph(const HipaccImageCuda<T> &img, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaMemcpy3DParms &memcpyArgs) {
  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  cudaError_t err;
  if (img->get_mem_type() == hipaccMemoryType::Array2D ||
      img->get_mem_type() == hipaccMemoryType::Surface) {
    // TODO
    assert(false && "hipacc memory type not supported for cuda graph");
  } else {
    if (stride > width) {
      memcpyArgs.srcArray = NULL;
      memcpyArgs.srcPos = make_cudaPos(0, 0, 0);
      memcpyArgs.srcPtr = make_cudaPitchedPtr(img->get_device_memory(), sizeof(T)*stride, stride, height);
      memcpyArgs.dstArray = NULL;
      memcpyArgs.dstPos = make_cudaPos(0, 0, 0);
      memcpyArgs.dstPtr = make_cudaPitchedPtr(img->get_host_memory(), sizeof(T)*width, width, height);
      memcpyArgs.extent = make_cudaExtent(sizeof(T)*width, sizeof(T)*height, 1);
      memcpyArgs.kind = cudaMemcpyDeviceToHost;
    } else {
      memcpyArgs.srcArray = NULL;
      memcpyArgs.srcPos = make_cudaPos(0, 0, 0);
      memcpyArgs.srcPtr = make_cudaPitchedPtr(img->get_device_memory(), sizeof(T)*width*height, width*height, 1);
      memcpyArgs.dstArray = NULL;
      memcpyArgs.dstPos = make_cudaPos(0, 0, 0);
      memcpyArgs.dstPtr = make_cudaPitchedPtr(img->get_host_memory(), sizeof(T)*width*height, width*height, 1);
      memcpyArgs.extent = make_cudaExtent(sizeof(T)*width*height, 1, 1);
      memcpyArgs.kind = cudaMemcpyDeviceToHost;
    }
		err = cudaGraphAddMemcpyNode(&graphNode, graph, graphNodeDeps.data(), graphNodeDeps.size(), &memcpyArgs);
    checkErr(err, (std::string("cudaGraphAddMemcpyNode(") + __FUNCTION__ + ")"));
  }
  return img->get_host_memory();
}

// Bind memory to texture
template <typename T>
void hipaccBindTexture(hipaccMemoryType mem_type, const textureReference *tex,
                       const HipaccImageCuda<T> &img) {
  cudaError_t err = cudaSuccess;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

  switch (mem_type) {
  case hipaccMemoryType::Linear1D:
    assert(img->get_mem_type() <= hipaccMemoryType::Linear2D &&
           "expected linear memory");
    err = cudaBindTexture(nullptr, tex, img->get_device_memory(), &channelDesc,
                          sizeof(T) * img->get_stride() * img->get_height());
    checkErr(err, "cudaBindTexture()");
    break;
  case hipaccMemoryType::Linear2D:
    assert(img->get_mem_type() <= hipaccMemoryType::Linear2D &&
           "expected linear memory");
    err = cudaBindTexture2D(nullptr, tex, img->get_device_memory(), &channelDesc,
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
                       const HipaccImageCuda<T> &img) {
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
void hipaccReadSymbol(T *host_mem, const void *symbol, std::string const& symbol_name,
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

// Perform global reduction using memory fence operations and return result
template <typename T, class KernelFunc>
T hipaccApplyReductionShared(const KernelFunc &reductionKernel,
                                  const HipaccAccessor<T> &acc,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  HipaccExecutionParameterCuda const &ep,
                                  const textureReference *tex,
                                  bool print_timing) {
  T *output; // GPU memory for reduction
  T result;  // host result

  // single step reduction: reduce image (region) into linear memory and
  // reduce the linear memory using memory fence operations
  dim3 blockDim(max_threads, 1);
  dim3 gridDim((int)ceilf((float)(acc.img->get_width()) / (blockDim.x)),
            (int)ceilf((float)(acc.height) / pixels_per_thread));
  unsigned int num_blocks = gridDim.x * gridDim.y;
  unsigned int idle_left = 0;

  cudaError_t err = cudaMalloc((void **)&output, sizeof(T) * num_blocks);
  checkErr(err, "cudaMalloc()");

  if ((acc.offset_x || acc.offset_y) &&
      (acc.width != acc.img->get_width() || acc.height != acc.img->get_height())) {
    // reduce iteration space by idle blocks
    idle_left = acc.offset_x / blockDim.x;
    unsigned int idle_right =
        (acc.img->get_width() - (acc.offset_x + acc.width)) / blockDim.x;
    gridDim.x = (int)ceilf(
        (float)(acc.img->get_width() - (idle_left + idle_right) * blockDim.x) /
        (blockDim.x));

    // update number of blocks
    idle_left *= blockDim.x;
  }

  auto accImgDevMem = acc.img->get_device_memory();
  
  auto accImgWidth = acc.img->get_width();
  auto accImgHeight = acc.img->get_height();
  auto accImgStride = acc.img->get_stride();

  //reserve buffer Memory
  T *bufferImage;
  cudaMalloc(&bufferImage, gridDim.x*gridDim.y*sizeof(T));

  //initialize counter for threads finished with last reduction phase; last to finish writes result to "output"
  unsigned int *finishedThreadCounter;
  cudaMalloc(&finishedThreadCounter, sizeof(unsigned int));
  cudaMemset(finishedThreadCounter, 0, sizeof(unsigned int));


  switch (acc.img->get_mem_type()) {
    case hipaccMemoryType::Global: {
      hipaccLaunchKernel(reductionKernel, gridDim, blockDim, ep, print_timing, (blockDim.x + 1) * blockDim.y * sizeof(T), accImgDevMem, output, accImgWidth, accImgHeight, accImgStride, bufferImage, finishedThreadCounter);
      break;
    }
    case hipaccMemoryType::Array2D: {
      // FIXME: textures not yet supported
      std::cerr << "CUDA reductions with textures are not supported yet" << std::endl;
      exit(1);
      //hipaccBindTexture<T2>(hipaccMemoryType::Array2D, tex, acc.img);
      //hipaccLaunchKernel(kernel2D, grid, block, ep, print_timing, shared_memory_size, output, acc.width, acc.height, tex, acc.img->get_stride(), acc.offset_x, acc.offset_y, num_bins, finishedThreadCounter);
      break;
    }
  }


  err = cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(finishedThreadCounter);
  checkErr(err, "cudaFree()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  //deallocate buffer
  err = cudaFree(bufferImage);
  checkErr(err, "cudaFree()");

  return result;
}

// Perform global reduction using memory fence operations and return result
template <typename T, class KernelFunc>
T hipaccApplyReductionShared(const KernelFunc &kernel2D,
                                  const HipaccImageCuda<T> &img,
                                  unsigned int max_threads,
                                  unsigned int pixels_per_thread,
                                  HipaccExecutionParameterCuda const &ep,
                                  const textureReference *tex,
                                  bool print_timing) {
  HipaccAccessor<T> acc(img);
  return hipaccApplyReductionShared<T>(
      kernel2D, acc, max_threads, pixels_per_thread, ep, tex, print_timing);
}

template <typename T, typename T2, class KernelFunc, int SEGMENT_SIZE>
T *hipaccApplyBinningSegmented(KernelFunc const &kernel2D,
                                                const HipaccAccessor<T2> &acc,
                                                unsigned int num_warps,
                                                unsigned int num_units,
                                                unsigned int num_bins,
                                                HipaccExecutionParameterCuda const &ep,
                                                const textureReference *tex,
                                                bool print_timing) {

  T *output; //GPU memory for reduction
  T *result = new T[num_bins]; //host result
  auto num_segments = (num_bins+SEGMENT_SIZE-1)/SEGMENT_SIZE;
  dim3 block(32, num_warps);
  dim3 grid(num_units, num_segments);

  size_t shared_memory_size = SEGMENT_SIZE * num_warps * sizeof(T);

  cudaError_t err =
      cudaMalloc((void **)&output, sizeof(T) * num_units * num_bins);
  checkErr(err, "cudaMalloc()");

  //initialize counter for threads finished with last reduction phase
  unsigned int *finishedThreadCounter;
  cudaMalloc(&finishedThreadCounter, sizeof(unsigned int)*num_segments);
  cudaMemset(finishedThreadCounter, 0, sizeof(unsigned int)*num_segments);

  switch (acc.img->get_mem_type()) {
    case hipaccMemoryType::Global: {
      hipaccLaunchKernel(kernel2D, grid, block, ep, print_timing, shared_memory_size, acc.img->get_device_memory(), output, acc.width, acc.height, acc.img->get_stride(), acc.offset_x, acc.offset_y, num_bins, finishedThreadCounter);
      break;
    }
    case hipaccMemoryType::Array2D: {
      // FIXME: textures not yet supported
      std::cerr << "CUDA reductions with textures are not supported yet" << std::endl;
      exit(1);
      hipaccBindTexture<T2>(hipaccMemoryType::Array2D, tex, acc.img);
      //hipaccLaunchKernel(kernel2D, grid, block, ep, print_timing, shared_memory_size, output, acc.width, acc.height, tex, acc.img->get_stride(), acc.offset_x, acc.offset_y, num_bins, finishedThreadCounter);
      break;
    }
  }

  err =
      cudaMemcpy(result, output, sizeof(T) * num_bins, cudaMemcpyDeviceToHost);
  checkErr(err, "cudaMemcpy()");

  err = cudaFree(finishedThreadCounter);
  checkErr(err, "cudaFree()");

  err = cudaFree(output);
  checkErr(err, "cudaFree()");

  return result;

}

// Allocate memory for Pyramid image
template <typename T>
HipaccImageCuda<T> hipaccCreatePyramidImage(const HipaccImageCuda<T> &base,
                                         size_t width, size_t height) {
  switch (base->get_mem_type()) {
  default:
    if (base->get_alignment() > 0) {
      return hipaccCreateMemory<T>(nullptr, width, height, base->get_alignment());
    } else {
      return hipaccCreateMemory<T>(nullptr, width, height);
    }
  case hipaccMemoryType::Array2D:
    return hipaccCreateArray2D<T>(nullptr, width, height);
  }
}

template <typename T>
HipaccPyramidCuda<T> hipaccCreatePyramid(const HipaccImageCuda<T> &img,
                                      size_t depth) {
  HipaccPyramidCuda<T> p(depth);
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
