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

#ifndef __HIPACC_CU_RED_HPP__
#define __HIPACC_CU_RED_HPP__


#include <assert.h>
//include <device_launch_parameters.h>


#define WARP_SIZE 32
//#define REDUCE_BLOCKS_SEQUENTIAL


template<typename CalcType>
struct IdxVal {
	unsigned int idx;
	CalcType val;
};


template <typename CalcType>
using ReduceFunc = CalcType(*)(CalcType a, CalcType b);

template <typename CalcType, typename PixelType>
using BinningFunc = IdxVal<CalcType>(*)(unsigned int num_bins, unsigned int x, unsigned int y, PixelType pixel);


// row length == 32
template <typename CalcType, ReduceFunc<CalcType> reduce>
__device__ void hipacc_half_warp_reduce(volatile CalcType* sdata, int tid) {
  sdata[tid] = reduce(sdata[tid], sdata[tid + 16]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 8]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 4]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 2]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 1]);
}

// row length == 64
template <typename CalcType, ReduceFunc<CalcType> reduce>
__device__ void hipacc_warp_reduce(volatile CalcType *sdata, int tid) {
  sdata[tid] = reduce(sdata[tid], sdata[tid + 32]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 16]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 8]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 4]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 2]);
  sdata[tid] = reduce(sdata[tid], sdata[tid + 1]);
}

// row length > 64 && row_length % 32 == 0
template <typename CalcType, ReduceFunc<CalcType> reduce>
__device__ void warp_reduce_max(volatile CalcType* sdata, int tid, int row_length) {
  // reduce upper chunks of each 32 values
  for (int i = row_length - 32; i > 32; i -= 32) {
    sdata[tid] = reduce(sdata[tid], sdata[tid + i]);
  }
  hipacc_warp_reduce<CalcType, reduce>(sdata, tid);
}

// row length is unknown
template <typename CalcType, ReduceFunc<CalcType> reduce>
__device__ void hipacc_warp_reduce_odd(volatile CalcType* sdata, int tid, int row_length) {
  if (row_length >= 64) {
    // guarded reduce of overlap
    const int overlap_start = (row_length / 32) * 32;
    if (tid + overlap_start < row_length) { //check whether there is overlap
      sdata[tid] = reduce(sdata[tid], sdata[tid + overlap_start]);
    }
    // followed by normal warp reduce
    warp_reduce_max<CalcType, reduce>(sdata, tid, row_length);
  }
  else if (row_length >= 32) {
    // guarded reduce of overlap
    const int overlap_start = 32;
    if (tid + overlap_start < row_length) {
      sdata[tid] = reduce(sdata[tid], sdata[tid + overlap_start]);
    }
    // followed by normal half warp reduce
    hipacc_half_warp_reduce<CalcType, reduce>(sdata, tid);
  }
  else if (row_length > 1) {
    // reduce everything guarded
#pragma unroll 5
    for (int i = 16; i > 0; i /= 2) {
      if (tid + i < row_length) {
        sdata[tid] = reduce(sdata[tid], sdata[tid + i]);
      }
    }
  }
}

// NVCC does not accept multiple external shared memory arrays with same name
// but differen types. We need this workaround to solve this issue.
template <typename CalcType>
__device__ CalcType *hipacc_get_shared() {
  extern __shared__ unsigned char mem[];
  return reinterpret_cast<CalcType*>(mem);
}


/// \brief Reduces input values in parallel.
/// Must be called by every thread in the kernel. Every thread constributes a
/// single value for reduction. After the reduction is finished, this function
/// will return true for the last thread still active. Thereby, the kernel can
/// continue processing the reduced result with a single thread. Requires
/// shared memory of size: NUM_ELEMENTS * (blockDim.x + 1) * blockDim.y.
/// \param result  Result memory of size: NUM_ELEMENTS * sizeof(CalcType)
/// \param red_buf Helper buffer for reduction of size: NUM_ELEMENTS * GRID_SIZE
/// \param value   Single input value for reduction
/// \param counter Variable for atomic counting
/// \returns true if current thread is the last one still running


template<int NUM_ELEMENTS = 1, typename CalcType, ReduceFunc<CalcType> reduce>
__device__ bool hipacc_do_shared_reduction(CalcType *result, CalcType *red_buf, CalcType *value, unsigned int *counter) {
  // multiple of warp size
  assert(blockDim.x >= 32 && blockDim.x % 32 == 0);
  // we need enough threads to sum up block results
  assert(blockDim.x * blockDim.y >= gridDim.x);
  // check dimensions of helper buffer

  const int lidx = threadIdx.x;
  const int lidy = threadIdx.y;

  auto smem = hipacc_get_shared<CalcType>();
  const int smem_pitch = blockDim.x + 1;
  volatile CalcType* smem_row = &smem[smem_pitch * lidy];

  const int smem_slice = blockDim.y * smem_pitch;
  const int red_buf_slice = gridDim.y;

  int smem_offset = 0;
#pragma unroll NUM_ELEMENTS
  for (int s = 0; s < NUM_ELEMENTS; ++s) {
    smem_row[smem_offset + lidx] = value[s];
    smem_offset += smem_slice;
  }

  if (blockDim.x > 32) {
    // wait for all warps in row to finish
    __syncthreads();

    // only let one warp in
    if (lidx < 32) {
      smem_offset = 0;
#pragma unroll NUM_ELEMENTS
      for (int s = 0; s < NUM_ELEMENTS; ++s) {
        const int sidx = smem_offset + lidx;

        // reduce each row to first element (lock-step in warp)
        warp_reduce_max<CalcType, reduce>(smem_row, sidx, blockDim.x);

        smem_offset += smem_slice;
      }
    }
  }
  else {
    // only let half a warp in
    if (lidx < 16) {
#pragma unroll NUM_ELEMENTS
      for (int smem_offset = 0; smem_offset < NUM_ELEMENTS*smem_slice; smem_offset+=smem_slice) {
        const int sidx = smem_offset + lidx;

        // reduce each row to first element (lock-step in warp)
        hipacc_half_warp_reduce<CalcType, reduce>(smem_row, sidx);
      }
    }
  }

  __shared__ bool last_block;
  last_block = false;

  __syncthreads();

  CalcType acc[NUM_ELEMENTS];
  const int num_blocks = gridDim.x * gridDim.y;

  if (lidx == 0 && lidy == 0) {
    smem_offset = 0;
#pragma unroll NUM_ELEMENTS
    for (int s = 0; s < NUM_ELEMENTS; ++s) {

      acc[s] = smem_row[smem_offset];

      // accumulate first elements in rows
      int smem_row_offset = smem_offset + smem_pitch;
      for (int i = 1; i < blockDim.y; ++i) {
        acc[s] = reduce(acc[s], smem_row[smem_row_offset]);
        smem_row_offset += smem_pitch;
      }

      smem_offset += smem_slice;
    }

    if (num_blocks > 1) {
      // store result of this block in global memory
      int red_buf_offset = 0;
#pragma unroll NUM_ELEMENTS
      for (int s = 0; s < NUM_ELEMENTS; ++s) {
        red_buf[red_buf_offset + blockIdx.x + blockIdx.y * gridDim.x] = acc[s];
        red_buf_offset += red_buf_slice;
      }

      const unsigned int id = atomicInc(counter, num_blocks - 1);
      if (id == num_blocks - 1) {
        last_block = true;
      }
    }
  }

  if (num_blocks == 1) {
    if (lidx == 0 && lidy == 0) {
      // we are done here, as we already have the result
#pragma unroll NUM_ELEMENTS
      for (int s = 0; s < NUM_ELEMENTS; ++s) {
        result[s] = acc[s];
      }
      return true;
    }
  }
  else {
    // accumulate block results

    __syncthreads(); // wait for write to last_block variable

    // check if we are the last block that finished
    if (last_block) {

      // linear thread id;
      const int tid = lidx + lidy * blockDim.x;
      smem_row = &smem[0];

      if (tid < gridDim.x) {

        smem_offset = 0;
        int red_buf_offset = 0;
#pragma unroll NUM_ELEMENTS
        for (int s = 0; s < NUM_ELEMENTS; ++s) {

          // accumulate results from blocks (from global memory) into single row
          acc[s] = red_buf[red_buf_offset + tid];
          int offset = gridDim.x;
          for (int y = 1; y < gridDim.y; ++y) {
            acc[s] = reduce(acc[s], red_buf[red_buf_offset + tid + offset]);
            offset += gridDim.x;
          }
          smem_row[smem_offset + tid] = acc[s];

          smem_offset += gridDim.x;
          red_buf_offset += red_buf_slice;
        }
      }

      __syncthreads();

#ifdef REDUCE_BLOCKS_SEQUENTIAL

      // accumulate all values in final row in a single thread
      if (tid == 0) {
        smem_offset = 0;
#pragma unroll NUM_ELEMENTS
        for (int s = 0; s < NUM_ELEMENTS; ++s) {

          for (int x = 1; x < gridDim.x; ++x) {
            acc[s] = reduce(acc[s], smem_row[smem_offset + x]);
          }
          result[s] = acc[s];

          smem_offset += gridDim.x;
        }

        *counter = 0;
        return true;
      }

#else // more parallelism, significant speedup (~0.25 ms) for large images

      // accumulate all values in parallel using a single warp
      if (gridDim.x > 1 && tid < 32) {
        smem_offset = 0;
#pragma unroll NUM_ELEMENTS
        for (int s = 0; s < NUM_ELEMENTS; ++s) {
          // we do not know gridDim.x -> reduce with guards
          hipacc_warp_reduce_odd<CalcType, reduce>(&smem_row[smem_offset], tid, gridDim.x);

          smem_offset += gridDim.x;
        }
      }

      if (tid == 0) {
        smem_offset = 0;
#pragma unroll NUM_ELEMENTS
        for (int s = 0; s < NUM_ELEMENTS; ++s) {

          result[s] = smem_row[smem_offset];

          smem_offset += gridDim.x;
        }

        *counter = 0;
        return true;
      }
#endif // REDUCE_BLOCKS_SEQUENTIAL
    }
  }

  return false;
}

template<typename CalcType, ReduceFunc<CalcType> reduce>
__global__ void hipacc_shared_reduction(CalcType *image, CalcType *result, const unsigned int image_width, const unsigned int image_height, const unsigned int image_stride, CalcType *red_buf, unsigned int *counter) {
  const int gidx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int gidy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (gidx < image_width && gidy < image_height) {
    CalcType *pixel = image + gidy * image_stride + gidx;
	  CalcType acc;
	  if(hipacc_do_shared_reduction<1, CalcType, reduce>(&acc, red_buf, pixel, counter))
		  *result = acc;
  }
}

// Helper for tagging integers with thread ID (first 5 bits are used)
template<typename IntType>
__device__ IntType tag_uint(IntType value) {
  static_assert(std::is_integral<IntType>::value && std::is_unsigned<IntType>::value && sizeof(IntType) <= 4,
                "Only unsigned integer up to 4 bytes are supported");
  return (value) | (threadIdx.x << (3 + ((sizeof(IntType) - 1) * 8)));
}

// Helper for untagging bin values
template<typename BinType>
__device__ BinType untag(BinType value) {
  // default: nothing is done (see specialization below for unsigned int)
  return value;
}

// Helper for untagging integer values with thread ID (first 5 bits are used)
template<typename IntType>
__device__ IntType untag_uint(IntType value) {
  static_assert(std::is_integral<IntType>::value && std::is_unsigned<IntType>::value && sizeof(IntType) <= 4,
                "Only unsigned integer up to 4 bytes are supported");
  return (value) & ~(0xF8 << ((sizeof(IntType) - 1) * 8));
}

// Specialized helper for supported unsigned int types
template<> __device__ uint8_t  untag(uint8_t value)  { return untag_uint<uint8_t>(value); }
template<> __device__ uint16_t untag(uint16_t value) { return untag_uint<uint16_t>(value); }
template<> __device__ uint32_t untag(uint32_t value) { return untag_uint<uint32_t>(value); }

// Helper for accumulating integers
template<typename IntType, ReduceFunc<IntType> reduce>
__device__ void accumulate_uint(volatile IntType* address, IntType accu) {
  static_assert(std::is_integral<IntType>::value && std::is_unsigned<IntType>::value && sizeof(IntType) <= 4,
                "Only unsigned integer up to 4 bytes are supported");
  IntType old, val;
  do {
    // read current bin and remove thread ID tag
    old = untag_uint<IntType>(*address);
    // reduce value, add thread ID tag
    val = tag_uint<IntType>(reduce(old, accu));
    // write new value to current bin
    *address = val;
  } while (*address != val); // run until write was successful
}

#if (__CUDA_ARCH__ >= 700)
// Helper for accumulating 16bit types using atomicCAS
template<typename BinType, ReduceFunc<BinType> reduce>
__device__ void accumulate_cas_16(BinType* address, BinType accu) {
  BinType old, val;
  unsigned short int *oldi = (unsigned short int*)&old, *vali = (unsigned short int*)&val;
  do {
    old = *address;
    val = reduce(old, accu);
  } while (atomicCAS((unsigned short int*)address, *oldi, *vali) != *oldi);
}
#endif

// Helper for accumulating 32bit types using atomicCAS
template<typename BinType, ReduceFunc<BinType> reduce>
__device__ void accumulate_cas_32(BinType* address, BinType accu) {
  BinType old, val;
  unsigned int *oldi = (unsigned int*)&old, *vali = (unsigned int*)&val;
  do {
    old = *address;
    val = reduce(old, accu);
  } while (atomicCAS((unsigned int*)address, *oldi, *vali) != *oldi);
}

// Helper for accumulating 64bit types using atomicCAS
template<typename BinType, ReduceFunc<BinType> reduce>
__device__ void accumulate_cas_64(BinType* address, BinType accu) {
  BinType old, val;
  unsigned long long int *oldi = (unsigned long long int*)&old, *vali = (unsigned long long int*)&val;
  do {
    old = *address;
    val = reduce(old, accu);
  } while (atomicCAS((unsigned long long int*)address, *oldi, *vali) != *oldi);
}

// Helper for accumulating >128bit types using atomicCAS
template<typename BinType, ReduceFunc<BinType> reduce>
__device__ void accumulate_cas_gt64(BinType* address, BinType accu) {
  BinType old, val;
  unsigned long long int *oldi = (unsigned long long int*)&old, *vali = (unsigned long long int*)&val;
  bool run = true;
  do {
    old = *address;
    val = reduce(old, accu);
    if (atomicCAS((unsigned long long int*)address, *oldi, *vali) == *oldi) {
      run = false;
      // 64bit CAS succeeded, this is the winning thread in the current warp => write remaining bits
      *address = val;
    }
  } while (run);
}

// Helper for translating reduce functions with unknown int types to known int types
template <typename IntType, typename BinType, ReduceFunc<BinType> reduce>
__device__ IntType reduce_uint(IntType a, IntType b) {
  BinType ret = reduce(*reinterpret_cast<BinType*>(&a), *reinterpret_cast<BinType*>(&b));
  return *reinterpret_cast<IntType*>(&ret);
}

// Helper for dispatching to the correct accumulation implementation
template<typename BinType, ReduceFunc<BinType> reduce>
__device__ void accumulate(BinType* address, BinType accu) {
  if (std::is_integral<BinType>::value && std::is_unsigned<BinType>::value && sizeof(BinType) <= 4) {
    switch (sizeof(BinType)) {
      case 1:
        accumulate_uint<uint8_t, reduce_uint<uint8_t, BinType, reduce>>(reinterpret_cast<uint8_t*>(address), *reinterpret_cast<uint8_t*>(&accu));
        break;
      case 2:
        accumulate_uint<uint16_t, reduce_uint<uint16_t, BinType, reduce>>(reinterpret_cast<uint16_t*>(address), *reinterpret_cast<uint16_t*>(&accu));
        break;
      case 4:
        accumulate_uint<uint32_t, reduce_uint<uint32_t, BinType, reduce>>(reinterpret_cast<uint32_t*>(address), *reinterpret_cast<uint32_t*>(&accu));
        break;
      default:
        break;
    }
  }
  else {
    switch (sizeof(BinType)) {
    case 2: // short
#if (__CUDA_ARCH__ >= 700)
      accumulate_cas_16<BinType, reduce>(address, accu);
      break;
#else
            // fall through
#endif
    case 1: // bool/char
      assert(false && "Unsupported type");
      break;
    case 4: // int/float
      accumulate_cas_32<BinType, reduce>(address, accu);
      break;
    case 8: // double
      accumulate_cas_64<BinType, reduce>(address, accu);
      break;
    default: // 128 bit int
      accumulate_cas_gt64<BinType, reduce>(address, accu);
      break;
    }
  }
}


// Binning and reduce with conflicts solved via thread ID (uint) or CAS (else)
//
// Variables:
//  - num_bins:     Number of bins
//
// Configuration:
//  - BinType:      Type of bins
//  - CalcType:     Type of input pixels
//  - NUM_WARPS:    Warps per block (affects block size and shared memory size)
//  - NUM_UNITS:    Number of partial results (affects number of blocks)
//  - SEGMENT_SIZE: Size of segment of partial results (lower -> less conflicts)
//
// Constants:
//  - WARP_SIZE:    Threads per warp (32 for NVIDIA)
//
// Constraints:
//  - WARP_SIZE * NUM_WARPS >= SEGMENT_SIZE
//  - num_segments == ceil(num_bins/SEGMENT_SIZE)
//
// Settings:
//  - Block size:              WARP_SIZE x NUM_WARPS
//  - Grid size:               NUM_UNITS x num_segments
//  - Shared memory required:  SEGMENT_SIZE * NUM_WARPS * sizeof(BinType)
//  - Global memory required:  num_bins * NUM_UNITS * sizeof(BinType)
//
// Steps:
//  1) Each warp computes a single SEGMENT in shared memory.
//  2) SEGMENTS of all warps within a block are assembled to single SEGMENT
//     and stored in global memory.
//      - num_segments represent partial (full size, not entire image) result,
//  3) y partial results are merged by x blocks to a single final result vector.
//      - where y = NUM_UNITS
template<typename BinType, typename CalcType, BinningFunc<BinType, CalcType> binning, ReduceFunc<BinType> reduce, int NUM_WARPS, int NUM_UNITS, int SEGMENT_SIZE, int PPT>
__device__ bool hipacc_binning_and_reduce(BinType* result, CalcType* input, int pitch, int pos_x, int pos_y, int width, int height, int num_bins, unsigned int* counter) {
  static_assert(WARP_SIZE * NUM_WARPS >= SEGMENT_SIZE, "Not enough warps");
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == NUM_WARPS);
  assert(gridDim.x == NUM_UNITS);
  assert(gridDim.y == ((num_bins + SEGMENT_SIZE - 1) / SEGMENT_SIZE));

  const int num_warps = NUM_WARPS;
  const int segment_size = SEGMENT_SIZE;

  int lid = threadIdx.x + threadIdx.y * WARP_SIZE;

  auto warp_result = hipacc_get_shared<BinType>();
  BinType* lresult = &warp_result[threadIdx.y * SEGMENT_SIZE];

  // initialize shared memory
#pragma unroll segment_size
  for (int i = 0; i < SEGMENT_SIZE; i += WARP_SIZE) {
    lresult[threadIdx.x + i] = BinType{ 0 };
  }

  __syncthreads();

  // compute result segments
  int increment = NUM_UNITS * WARP_SIZE * NUM_WARPS;
  int gpos = ((WARP_SIZE * NUM_WARPS) * blockIdx.x) + (threadIdx.y * WARP_SIZE) + threadIdx.x;
  int end = width * height / PPT;
  int offset = blockIdx.y * SEGMENT_SIZE;

  // iterate over the entire input
  for (int i = gpos; i < end; i += increment) {
    int gid_y = pos_y + (i / width);
    int gid_x = pos_x + (i % width);
    int ipos = gid_x + gid_y * pitch;
    const int inc = height / PPT * pitch;
#pragma unroll PPT
    for (int p = 0; p < PPT; ++p) {
      int y = gid_y + p * height / PPT;

      CalcType pixel = input[ipos];
      ipos += inc;

      // compute idx/val for bin from pixel
      IdxVal<BinType> bin = binning(num_bins, gid_x, y, pixel);

      // project to current segment
      bin.idx -= offset;

      // write to bin in current segment
      if (bin.idx < SEGMENT_SIZE) {
        accumulate<BinType, reduce>(&lresult[bin.idx], bin.val);
      }
    }
  }

  __syncthreads();

  BinType bin{ 0 };

  // assemble segments and write partial results
  if (lid < min(SEGMENT_SIZE, num_bins)) {
    bin = untag<BinType>(warp_result[lid]);
#pragma unroll num_warps
    for (int i = 1; i < NUM_WARPS; ++i) {
      bin += untag<BinType>(warp_result[i * SEGMENT_SIZE + lid]);
    }
    result[offset + lid + (blockIdx.x * num_bins)] = bin;
  }

  // merge partial results
  if (gridDim.x > 1) {
    __shared__ bool last_block_for_segment;

    if (lid == 0) {
      int ticket = atomicInc(&counter[blockIdx.y], gridDim.x - 1);
      last_block_for_segment = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (last_block_for_segment) {
      int blocksize = WARP_SIZE * NUM_WARPS;
      int runs = (SEGMENT_SIZE + blocksize - 1) / blocksize;

      for (int i = 0; i < runs; ++i) {
        if (lid < SEGMENT_SIZE) {
          bin = result[offset + lid];

          for (unsigned yi = 1; yi < gridDim.x; ++yi) {
            bin = reduce(bin, result[offset + yi * num_bins + lid]);
          }

          result[offset + lid] = bin;
        }
        lid += blocksize;
      }

      if (lid == 0) {
        counter[blockIdx.y] = 0;
        return true;
      }
    }
  }
  else {
    if (lid == 0) return true;
  }

  return false;
}


template<typename BinType, typename CalcType, BinningFunc<BinType, CalcType> binning, ReduceFunc<BinType> reduce, int NUM_WARPS, int NUM_UNITS, int SEGMENT_SIZE=DEFAULT_SEGMENT_SIZE, int PPT=1>
__global__ void hipacc_binning_reduction(CalcType *image, BinType *result, int width, int height, int stride, int offset_x, int offset_y, int num_bins, unsigned int *counter) {
  hipacc_binning_and_reduce<BinType, CalcType, binning, reduce, NUM_WARPS, NUM_UNITS, SEGMENT_SIZE, PPT>(result, image, stride, offset_x, offset_y, width, height, num_bins, counter);
}



#endif  // __HIPACC_CU_RED_HPP__
