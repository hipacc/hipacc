//
// Copyright (c) 2014, Saarland University
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

#ifndef __HIPACC_CPU_TPP__
#define __HIPACC_CPU_TPP__

template <typename T>
HipaccImageCpu createImage(T *host_mem, void *mem, size_t width, size_t height,
                           size_t stride, size_t alignment,
                           hipaccMemoryType mem_type) {
  HipaccImageCpu img = std::make_shared<HipaccImageCpuRaw>(
      width, height, stride, alignment, sizeof(T), mem, mem_type);

  hipaccWriteMemory(img, host_mem ? host_mem : (T *)img->get_host_memory());
  return img;
}

// Allocate memory with alignment specified
template <typename T>
HipaccImageCpu hipaccCreateMemory(T *host_mem, size_t width, size_t height,
                                  size_t alignment) {
  // alignment has to be a multiple of sizeof(T)
  alignment = static_cast<int>(ceilf(static_cast<float>(alignment) / sizeof(T)) * sizeof(T));
  int stride = static_cast<int>(ceilf(static_cast<float>(width) / (alignment / sizeof(T))) *
               (alignment / sizeof(T)));

  T *mem = new T[stride * height];
  return createImage(host_mem, (void *)mem, width, height, stride, alignment);
}

// Allocate memory without any alignment considerations
template <typename T>
HipaccImageCpu hipaccCreateMemory(T *host_mem, size_t width, size_t height) {
  T *mem = new T[width * height];
  return createImage(host_mem, (void *)mem, width, height, width, 0);
}

// Write to memory
template <typename T> void hipaccWriteMemory(HipaccImageCpu &img, T *host_mem) {
  if (host_mem == nullptr)
    return;

  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if ((char *)host_mem != img->get_host_memory()) // if user provides host data
    std::copy(host_mem, host_mem + width * height, (T *)img->get_host_memory());

  if (stride > width) {
    for (size_t i = 0; i < height; ++i) {
      std::memcpy(&((T *)img->get_aligned_host_memory())[i * stride],
                  &host_mem[i * width], sizeof(T) * width);
    }
  } else {
    std::memcpy(img->get_aligned_host_memory(), host_mem,
                sizeof(T) * width * height);
  }
}

// Read from memory
template <typename T> T *hipaccReadMemory(const HipaccImageCpu &img) {
  size_t width = img->get_width();
  size_t height = img->get_height();
  size_t stride = img->get_stride();

  if (stride > width) {
    for (size_t i = 0; i < height; ++i) {
      std::memcpy(&((T *)img->get_host_memory())[i * width],
                  &((T *)img->get_aligned_host_memory())[i * stride],
                  sizeof(T) * width);
    }
  } else {
    std::memcpy((T *)img->get_host_memory(), img->get_aligned_host_memory(),
                sizeof(T) * width * height);
  }

  return (T *)img->get_host_memory();
}

// Infer non-const Domain from non-const Mask
template <typename T>
void hipaccWriteDomainFromMask(HipaccImageCpu &dom, T *host_mem) {
  size_t size = dom->get_width() * dom->get_height();
  uchar *dom_mem = new uchar[size];

  for (size_t i = 0; i < size; ++i) {
    dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
  }

  hipaccWriteMemory(dom, dom_mem);

  delete[] dom_mem;
}

// Allocate memory for Pyramid image
template <typename T>
HipaccImageCpu hipaccCreatePyramidImage(const HipaccImageCpu &base,
                                        size_t width, size_t height) {
  switch (base->get_mem_type()) {
  default:
    if (base->get_alignment() > 0) {
      return hipaccCreateMemory<T>(NULL, width, height, base->get_alignment());
    } else {
      return hipaccCreateMemory<T>(NULL, width, height);
    }
  case hipaccMemoryType::Array2D:
    assert(0 && "Wrong memory type for CPU");
    return 0;
  }
}

template <typename T>
HipaccPyramidCpu hipaccCreatePyramid(const HipaccImageCpu &img, size_t depth) {
  HipaccPyramidCpu p(depth);
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

#endif // __HIPACC_CPU_TPP__
