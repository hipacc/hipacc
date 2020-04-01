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

#ifndef PPT
#define PPT 1
#endif
#ifndef BS
#define BS 32
#endif
// define offset parameters required to specify a sub-region on the image on
// that the reduction should be applied
#ifdef USE_OFFSETS
#define OFFSETS                                                                \
  , const unsigned int offset_x, const unsigned int offset_y,                  \
      const unsigned int is_width, const unsigned int is_height,               \
      const unsigned int offset_block
#define IS_HEIGHT is_height
#define OFFSET_BLOCK offset_block
#define OFFSET_Y offset_y
#define OFFSET_CHECK_X gid_x >= offset_x &&gid_x < is_width + offset_x
#define OFFSET_CHECK_X_STRIDE                                                  \
  gid_x + blockDim.x >= offset_x &&gid_x + blockDim.x < is_width + offset_x
#else
#define OFFSETS
#define IS_HEIGHT height
#define OFFSET_BLOCK 0
#define OFFSET_Y 0
#define OFFSET_CHECK_X gid_x < width
#define OFFSET_CHECK_X_STRIDE gid_x + blockDim.x < width
#endif
#ifdef USE_ARRAY_2D
#define READ(INPUT, X, Y, STRIDE) tex2D(INPUT, X, Y)
#define INPUT_PARM(DATA_TYPE, INPUT_NAME)
#else
#define READ(INPUT, X, Y, STRIDE) INPUT[(X) + (Y)*STRIDE]
#define INPUT_PARM(DATA_TYPE, INPUT_NAME) const DATA_TYPE *INPUT_NAME,
#endif
#include <assert.h>
//include <device_launch_parameters.h>


//#define REDUCE_BLOCKS_SEQUENTIAL

template <typename CalcType>
using ReduceFunc = CalcType(*)(CalcType a, CalcType b);
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

// global variable used by thread-fence reduction to count how many blocks have
// finished -> defined by the compiler (otherwise redefined for multiple
// reductions)
//__device__ unsigned int finished_blocks = 0;

// single step reduction:
// reduce a 2D block and store the reduced values to linear memory; use
// thread-fence synchronization to reduce the values stored to linear memory in
// one go - CUDA 2.x hardware required

// Helper macros for accumulating integers in segmented binning/reduction
// (first 5 bits are used for tagging with thread id)
#define UNTAG_INT(BIN_TYPE, VAL) (VAL) & ~(0xF8 << ((sizeof(BIN_TYPE) - 1) * 8))

#define TAG_INT(BIN_TYPE, VAL)                                                 \
  (VAL) | (threadIdx.x << (3 + ((sizeof(BIN_TYPE) - 1) * 8)))

#define ACCU_INT(BIN_TYPE, PTR, REDUCE)                                        \
  volatile BIN_TYPE *address = PTR;                                            \
  BIN_TYPE old, val;                                                           \
  do {                                                                         \
    old = UNTAG_INT(BIN_TYPE, *address);                                       \
    val = TAG_INT(BIN_TYPE, REDUCE(old, bin));                                 \
    *address = val;                                                            \
  } while (*address != val);

// Helper macros for accumulating using atomicCAS in segmented binning/reduction
#define UNTAG_NONE(BIN_TYPE, VAL) (VAL)

#define ACCU_CAS_32(BIN_TYPE, PTR, REDUCE)                                     \
  BIN_TYPE *address = PTR;                                                     \
  BIN_TYPE old, val;                                                           \
  unsigned int *oldi = (unsigned int *)&old, *vali = (unsigned int *)&val;     \
  do {                                                                         \
    old = *address;                                                            \
    val = REDUCE(old, bin);                                                    \
  } while (atomicCAS((unsigned int *)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_64(BIN_TYPE, PTR, REDUCE)                                     \
  BIN_TYPE *address = PTR;                                                     \
  BIN_TYPE old, val;                                                           \
  unsigned long long int *oldi = (unsigned long long int *)&old,               \
                         *vali = (unsigned long long int *)&val;               \
  do {                                                                         \
    old = *address;                                                            \
    val = REDUCE(old, bin);                                                    \
  } while (atomicCAS((unsigned long long int *)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_GT64(BIN_TYPE, PTR, REDUCE)                                   \
  BIN_TYPE *address = PTR;                                                     \
  BIN_TYPE old, val;                                                           \
  unsigned long long int *oldi = (unsigned long long int *)&old,               \
                         *vali = (unsigned long long int *)&val;               \
  bool run = true;                                                             \
  do {                                                                         \
    old = *address;                                                            \
    val = REDUCE(old, bin);                                                    \
    if (atomicCAS((unsigned long long int *)address, *oldi, *vali) == *oldi) { \
      run = false;                                                             \
      /* 64bit CAS succeeded, winning thread in warp => write remaining bits   \
       */                                                                      \
      *address = val;                                                          \
    }                                                                          \
  } while (run);

// Binning and reduction with conflicts solved via AtomicCAS or ThreadIdx
//
// Variables:
//  - PIXEL_TYPE: Type of image pixels
//  - BIN_TYPE:   Type of histogram bins
//  - NUM_BINS:   Number of histogram bins
//
// Configuration:
//  - WARP_SIZE: Threads per warp (32 for NVIDIA)
//  - NUM_WARPS: Warps per block (affects block size and shared memory size)
//  - NUM_HISTS: Partial histograms (affects number of blocks)
//
// Constants:
//  - SEGMENT_SIZE: 128 (higher -> less segments and redundancy, more conflicts)
//
// Settings:
//  - Block size:              WARP_SIZE x NUM_WARPS
//  - Shared memory per block: SEGMENT_SIZE * NUM_WARPS * sizeof(BIN_TYPE)
//  - Number of segments:      NUM_SEGMENTS = ceil(NUM_BINS/SEGMENT_SIZE)
//  - Grid size:               NUM_HISTS x NUM_SEGMENTS
//
// Steps:
//  1) Each warp computes a single SEGMENT in shared memory.
//  2) SEGMENTS of all warps within a block are assembled to single SEGMENT
//     and stored in global memory.
//      - x SEGMENTS represent partial (full size, not entire image) histogram,
//      - where x = ceil(NUM_BINS/SEGMENT_SIZE)
//  3) y partial histograms are merged by x blocks to a single final histogram.
//      - where y = NUM_HISTS

#ifdef _Pragma
#define NVCC_UNROLL _Pragma("unroll")
#else
#ifdef __pragma
#define NVCC_UNROLL __pragma(unroll)
#else
#define NVCC_UNROLL
#endif
#endif 


#define BINNING_CUDA_2D_SEGMENTED(                                             \
    NAME, PIXEL_TYPE, BIN_TYPE, REDUCE, BINNING, ACCU, UNTAG, WARP_SIZE,       \
    NUM_WARPS, NUM_HISTS, PPT, SEGMENT_SIZE, ZERO, INPUT_NAME)                 \
  __device__ inline void BINNING##Put(BIN_TYPE *lmem, uint offset, uint idx,   \
                                      BIN_TYPE val) {                          \
    idx -= offset;                                                             \
    if (idx < SEGMENT_SIZE) {                                                  \
                                                                               \
      /* set bin value */                                                      \
      BIN_TYPE bin = val;                                                      \
                                                                               \
      /* accumulate using reduce function */                                   \
      ACCU(BIN_TYPE, &lmem[idx], REDUCE);                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  __global__ void __launch_bounds__(WARP_SIZE *NUM_WARPS)                      \
      NAME(INPUT_PARM(PIXEL_TYPE, INPUT_NAME) BIN_TYPE *output,                \
           const unsigned int width, const unsigned int height,                \
           const unsigned int stride, const unsigned int num_bins,             \
           const unsigned int offset_x, const unsigned int offset_y) {         \
    unsigned int lid = threadIdx.x + threadIdx.y * WARP_SIZE;                  \
                                                                               \
    __shared__ BIN_TYPE warp_hist[NUM_WARPS * SEGMENT_SIZE];                   \
    BIN_TYPE *lhist = &warp_hist[threadIdx.y * SEGMENT_SIZE];                  \
                                                                               \
    /* initialize shared memory */                                             \
    NVCC_UNROLL for (unsigned int i = 0; i < SEGMENT_SIZE;               \
                           i += WARP_SIZE) {                                   \
      lhist[threadIdx.x + i] = ZERO;                                           \
    }                                                                          \
                                                                               \
    __syncthreads();                                                           \
                                                                               \
    /* compute histogram segments */                                           \
    unsigned int increment = NUM_HISTS * WARP_SIZE * NUM_WARPS;                \
    unsigned int gpos = ((WARP_SIZE * NUM_WARPS) * blockIdx.x) +               \
                        (threadIdx.y * WARP_SIZE) + threadIdx.x;               \
    unsigned int end = width * height / PPT;                                   \
    unsigned int offset = blockIdx.y * SEGMENT_SIZE;                           \
                                                                               \
    BIN_TYPE bin = ZERO;                                                       \
    NVCC_UNROLL for (unsigned int i = gpos; i < end; i += increment) {   \
      unsigned int gid_y = offset_y + (i / width);                             \
      unsigned int gid_x = offset_x + (i % width);                             \
      uint ipos = gid_x + gid_y * stride;                                      \
      const uint inc = height / PPT * stride;                                  \
      NVCC_UNROLL for (unsigned int p = 0; p < PPT; ++p) {               \
        uint y = gid_y + p * height / PPT;                                     \
                                                                               \
        PIXEL_TYPE pixel = INPUT_NAME[ipos];                                   \
        ipos += inc;                                                           \
                                                                               \
        BINNING(lhist, offset, num_bins, gid_x, y, pixel);                     \
      }                                                                        \
    }                                                                          \
                                                                               \
    __syncthreads();                                                           \
                                                                               \
    /* assemble segments and write partial histograms */                       \
    if (lid < min(SEGMENT_SIZE, num_bins)) {                                   \
      bin = UNTAG(BIN_TYPE, warp_hist[lid]);                                   \
      NVCC_UNROLL for (unsigned int i = 1; i < NUM_WARPS; ++i) {         \
        bin = REDUCE(bin, UNTAG(BIN_TYPE, warp_hist[i * SEGMENT_SIZE + lid])); \
      }                                                                        \
      output[offset + lid + (blockIdx.x * num_bins)] = bin;                    \
    }                                                                          \
                                                                               \
    /* merge partial histograms */                                             \
    if (gridDim.x > 1) {                                                       \
      __shared__ bool last_block_for_segment;                                  \
                                                                               \
      if (lid == 0) {                                                          \
        unsigned int ticket =                                                  \
            atomicInc(&finished_blocks_##NAME[blockIdx.y], gridDim.x - 1);     \
        last_block_for_segment = (ticket == gridDim.x - 1);                    \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (last_block_for_segment) {                                            \
        unsigned int blocksize = WARP_SIZE * NUM_WARPS;                        \
        unsigned int runs = (SEGMENT_SIZE + blocksize - 1) / blocksize;        \
                                                                               \
        NVCC_UNROLL for (unsigned int i = 0; i < runs; ++i) {            \
          if (lid < SEGMENT_SIZE) {                                            \
            bin = output[offset + lid];                                        \
                                                                               \
            NVCC_UNROLL for (unsigned yi = 1; yi < gridDim.x; ++yi) {    \
              bin = REDUCE(bin, output[offset + yi * num_bins + lid]);         \
            }                                                                  \
                                                                               \
            output[offset + lid] = bin;                                        \
          }                                                                    \
          lid += blocksize;                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#endif  // __HIPACC_CU_RED_HPP__
