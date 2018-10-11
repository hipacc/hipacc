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

//#ifndef __HIPACC_CU_RED_HPP__
//#define __HIPACC_CU_RED_HPP__

#ifndef PPT
#define PPT 1
#endif
#ifndef BS
#define BS 32
#endif
// define offset parameters required to specify a sub-region on the image on
// that the reduction should be applied
#ifdef USE_OFFSETS
#define OFFSETS , const unsigned int offset_x, const unsigned int offset_y, const unsigned int is_width, const unsigned int is_height, const unsigned int offset_block
#define IS_HEIGHT is_height
#define OFFSET_BLOCK offset_block
#define OFFSET_Y offset_y
#define OFFSET_CHECK_X gid_x >= offset_x && gid_x < is_width + offset_x
#define OFFSET_CHECK_X_STRIDE gid_x + blockDim.x >= offset_x && gid_x + blockDim.x < is_width + offset_x
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


// global variable used by thread-fence reduction to count how many blocks have
// finished -> defined by the compiler (otherwise redefined for multiple
// reductions)
//__device__ unsigned int finished_blocks = 0;

// single step reduction:
// reduce a 2D block and store the reduced values to linear memory; use
// thread-fence synchronization to reduce the values stored to linear memory in
// one go - CUDA 2.x hardware required
#define REDUCTION_CUDA_2D_THREAD_FENCE(NAME, DATA_TYPE, REDUCE, INPUT_NAME) \
__global__ void __launch_bounds__ (BS) NAME(INPUT_PARM(DATA_TYPE, INPUT_NAME) \
        DATA_TYPE *output, const unsigned int width, const unsigned int height, \
        const unsigned int stride OFFSETS) { \
    const unsigned int gid_x = 2*blockDim.x * blockIdx.x + threadIdx.x + OFFSET_BLOCK; \
    const unsigned int gid_y = PPT*blockDim.y * blockIdx.y + threadIdx.y; \
    const unsigned int tid = threadIdx.x; \
 \
    __shared__ DATA_TYPE sdata[BS]; \
 \
    DATA_TYPE val; \
 \
    if (OFFSET_CHECK_X) { \
        if (OFFSET_CHECK_X_STRIDE) { \
            val = REDUCE(READ(INPUT_NAME, gid_x, gid_y + OFFSET_Y, stride), READ(INPUT_NAME, gid_x + blockDim.x, gid_y + OFFSET_Y, stride)); \
        } else { \
            val = READ(INPUT_NAME, gid_x, gid_y + OFFSET_Y, stride); \
        } \
    } else { \
        val = READ(INPUT_NAME, gid_x + blockDim.x, gid_y + OFFSET_Y, stride); \
    } \
 \
    for (int j=1; j < PPT; ++j) { \
        if (j+gid_y < IS_HEIGHT) { \
            if (OFFSET_CHECK_X) { \
                val = REDUCE(val, READ(INPUT_NAME, gid_x, j+gid_y + OFFSET_Y, stride)); \
            } \
            if (OFFSET_CHECK_X_STRIDE) { \
                val = REDUCE(val, READ(INPUT_NAME, gid_x+blockDim.x, j+gid_y + OFFSET_Y, stride)); \
            } \
        } \
    } \
    sdata[tid] = val; \
 \
    __syncthreads(); \
 \
    for (int s=blockDim.x/2; s>32; s>>=1) { \
        if (tid < s) { \
            sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
        } \
        __syncthreads(); \
    } \
 \
    if (tid < 32) { \
        volatile DATA_TYPE *smem = sdata; \
        smem[tid] = val = REDUCE(val, smem[tid + 32]); \
        smem[tid] = val = REDUCE(val, smem[tid + 16]); \
        smem[tid] = val = REDUCE(val, smem[tid +  8]); \
        smem[tid] = val = REDUCE(val, smem[tid +  4]); \
        smem[tid] = val = REDUCE(val, smem[tid +  2]); \
        smem[tid] = val = REDUCE(val, smem[tid +  1]); \
    } \
 \
    if (tid == 0) output[blockIdx.x + gridDim.x*blockIdx.y] = sdata[0]; \
 \
    if (gridDim.x * gridDim.y > 1) { \
        __shared__ bool last_block; \
 \
        __threadfence(); \
 \
        if (tid == 0) { \
            unsigned int ticket = atomicInc(&finished_blocks_##NAME, gridDim.x*gridDim.y); \
            last_block = (ticket == gridDim.x*gridDim.y-1); \
        } \
        __syncthreads(); \
 \
        if (last_block) { \
            unsigned int i = tid; \
            val = output[tid]; \
            i += blockDim.x; \
 \
            while (i < gridDim.x*gridDim.y) { \
                val = REDUCE(val, output[i]); \
                i += blockDim.x; \
            } \
            sdata[tid] = val; \
 \
            __syncthreads(); \
 \
            for (int s=blockDim.x/2; s>32; s>>=1) { \
                if (tid < s) { \
                    sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
                } \
                __syncthreads(); \
            } \
 \
            if (tid < 32) { \
                volatile DATA_TYPE *smem = sdata; \
                smem[tid] = val = REDUCE(val, smem[tid + 32]); \
                smem[tid] = val = REDUCE(val, smem[tid + 16]); \
                smem[tid] = val = REDUCE(val, smem[tid +  8]); \
                smem[tid] = val = REDUCE(val, smem[tid +  4]); \
                smem[tid] = val = REDUCE(val, smem[tid +  2]); \
                smem[tid] = val = REDUCE(val, smem[tid +  1]); \
            } \
 \
            if (tid == 0) { \
                output[0] = sdata[0]; \
                finished_blocks_##NAME = 0; \
            } \
        } \
    } \
}


// step 1:
// reduce a 2D block and store the reduced value to linear memory
#define REDUCTION_CUDA_2D(NAME, DATA_TYPE, REDUCE, INPUT_NAME) \
__global__ void __launch_bounds__ (BS) NAME(INPUT_PARM(DATA_TYPE, INPUT_NAME) \
        DATA_TYPE *output, const unsigned int width, const unsigned int height, \
        const unsigned int stride OFFSETS) { \
    const unsigned int gid_x = 2*blockDim.x * blockIdx.x + threadIdx.x + OFFSET_BLOCK; \
    const unsigned int gid_y = PPT*blockDim.y * blockIdx.y + threadIdx.y; \
    const unsigned int tid = threadIdx.x; \
 \
    __shared__ DATA_TYPE sdata[BS]; \
 \
    DATA_TYPE val; \
 \
    if (OFFSET_CHECK_X) { \
        if (OFFSET_CHECK_X_STRIDE) { \
            val = REDUCE(READ(INPUT_NAME, gid_x, gid_y + OFFSET_Y, stride), READ(INPUT_NAME, gid_x + blockDim.x, gid_y + OFFSET_Y, stride)); \
        } else { \
            val = READ(INPUT_NAME, gid_x, gid_y + OFFSET_Y, stride); \
        } \
    } else { \
        val = READ(INPUT_NAME, gid_x + blockDim.x, gid_y + OFFSET_Y, stride); \
    } \
 \
    for (int j=1; j < PPT; ++j) { \
        if (j+gid_y < IS_HEIGHT) { \
            if (OFFSET_CHECK_X) { \
                val = REDUCE(val, READ(INPUT_NAME, gid_x, j+gid_y + OFFSET_Y, stride)); \
            } \
            if (OFFSET_CHECK_X_STRIDE) { \
                val = REDUCE(val, READ(INPUT_NAME, gid_x+blockDim.x, j+gid_y + OFFSET_Y, stride)); \
            } \
        } \
    } \
    sdata[tid] = val; \
 \
    __syncthreads(); \
 \
    for (int s=blockDim.x/2; s>32; s>>=1) { \
        if (tid < s) { \
            sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
        } \
        __syncthreads(); \
    } \
 \
    if (tid < 32) { \
        volatile DATA_TYPE *smem = sdata; \
        smem[tid] = val = REDUCE(val, smem[tid + 32]); \
        smem[tid] = val = REDUCE(val, smem[tid + 16]); \
        smem[tid] = val = REDUCE(val, smem[tid +  8]); \
        smem[tid] = val = REDUCE(val, smem[tid +  4]); \
        smem[tid] = val = REDUCE(val, smem[tid +  2]); \
        smem[tid] = val = REDUCE(val, smem[tid +  1]); \
    } \
 \
    if (tid == 0) output[blockIdx.x + gridDim.x*blockIdx.y] = sdata[0]; \
}


// step 2:
// reduce a 1D block and store the reduced value to the first element of linear
// memory
#define REDUCTION_CUDA_1D(NAME, DATA_TYPE, REDUCE) \
__global__ void NAME(const DATA_TYPE *input, DATA_TYPE *output, const unsigned \
        int num_elements, const unsigned int iterations) { \
    const unsigned int tid = threadIdx.x; \
    const unsigned int i = blockIdx.x*(blockDim.x*iterations) + threadIdx.x; \
 \
    __shared__ DATA_TYPE sdata[BS]; \
 \
    DATA_TYPE val = input[i]; \
 \
    for (int j=1; j < iterations; ++j) { \
        if (i + j*blockDim.x < num_elements) { \
            val = REDUCE(val, input[i + j*blockDim.x]); \
        } \
    } \
    sdata[tid] = val; \
 \
    __syncthreads(); \
 \
    for (int s=blockDim.x/2; s>0; s>>=1) { \
        if (tid < s) { \
            sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
        } \
        __syncthreads(); \
    } \
 \
    if (tid == 0) output[blockIdx.x] = sdata[0]; \
}


// Helper macros for accumulating integers in segmented binning/reduction
// (first 5 bits are used for tagging with thread id)
#define UNTAG_INT(BIN_TYPE, VAL) \
  (VAL) & ~(0xF8 << ((sizeof(BIN_TYPE)-1)*8))

#define TAG_INT(BIN_TYPE, VAL) \
  (VAL) | (threadIdx.x << (3 + ((sizeof(BIN_TYPE)-1)*8)))

#define ACCU_INT(BIN_TYPE, PTR, REDUCE) \
  volatile BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  do { \
    old = UNTAG_INT(BIN_TYPE, *address); \
    val = TAG_INT(BIN_TYPE, REDUCE(old, bin)); \
    *address = val; \
  } while (*address != val);


// Helper macros for accumulating using atomicCAS in segmented binning/reduction
#define UNTAG_NONE(BIN_TYPE, VAL) (VAL)

#define ACCU_CAS_32(BIN_TYPE, PTR, REDUCE) \
  BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  unsigned int *oldi = (unsigned int*)&old, *vali = (unsigned int*)&val; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
  } while (atomicCAS((unsigned int*)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_64(BIN_TYPE, PTR, REDUCE) \
  BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  unsigned long long int *oldi = (unsigned long long int*)&old, *vali = (unsigned long long int*)&val; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
  } while (atomicCAS((unsigned long long int*)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_GT64(BIN_TYPE, PTR, REDUCE) \
  BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  unsigned long long int *oldi = (unsigned long long int*)&old, *vali = (unsigned long long int*)&val; \
  bool run = true; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
    if (atomicCAS((unsigned long long int*)address, *oldi, *vali) == *oldi) { \
      run = false; \
      /* 64bit CAS succeeded, winning thread in warp => write remaining bits */ \
      *address = val; \
    } \
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
#define BINNING_CUDA_2D_SEGMENTED(NAME, PIXEL_TYPE, BIN_TYPE, REDUCE, BINNING, ACCU, UNTAG, WARP_SIZE, NUM_WARPS, NUM_HISTS, PPT, SEGMENT_SIZE, ZERO, INPUT_NAME) \
__device__ inline void BINNING##Put(BIN_TYPE *lmem, uint offset, uint idx, BIN_TYPE val) { \
  idx -= offset; \
  if (idx < SEGMENT_SIZE) { \
 \
    /* set bin value */ \
    BIN_TYPE bin = val; \
 \
    /* accumulate using reduce function */ \
    ACCU(BIN_TYPE, &lmem[idx], REDUCE); \
  } \
} \
 \
__global__ void __launch_bounds__ (WARP_SIZE*NUM_WARPS) NAME(INPUT_PARM(PIXEL_TYPE, INPUT_NAME) \
        BIN_TYPE *output, const unsigned int width, const unsigned int height, \
        const unsigned int stride, const unsigned int num_bins, \
        const unsigned int offset_x, const unsigned int offset_y) { \
  unsigned int lid = threadIdx.x + threadIdx.y * WARP_SIZE; \
 \
  __shared__ BIN_TYPE warp_hist[NUM_WARPS*SEGMENT_SIZE]; \
  BIN_TYPE* lhist = &warp_hist[threadIdx.y * SEGMENT_SIZE]; \
 \
  /* initialize shared memory */ \
  _Pragma("unroll") \
  for (unsigned int i = 0; i < SEGMENT_SIZE; i += WARP_SIZE) { \
    lhist[threadIdx.x + i] = ZERO; \
  } \
 \
  __syncthreads(); \
 \
  /* compute histogram segments */ \
  unsigned int increment = NUM_HISTS * WARP_SIZE * NUM_WARPS; \
  unsigned int gpos = ((WARP_SIZE * NUM_WARPS) * blockIdx.x) + (threadIdx.y * WARP_SIZE) + threadIdx.x; \
  unsigned int end = width * height/PPT; \
  unsigned int offset = blockIdx.y * SEGMENT_SIZE; \
 \
  BIN_TYPE bin = ZERO; \
  _Pragma("unroll") \
  for (unsigned int i = gpos; i < end; i += increment) { \
    unsigned int gid_y = offset_y + (i / width); \
    unsigned int gid_x = offset_x + (i % width); \
    uint ipos = gid_x + gid_y * stride; \
    const uint inc = height/PPT*stride; \
    _Pragma("unroll") \
    for (unsigned int p = 0; p < PPT; ++p) { \
      uint y = gid_y + p*height/PPT; \
   \
      PIXEL_TYPE pixel = INPUT_NAME[ipos]; \
      ipos += inc; \
   \
      BINNING(lhist, offset, num_bins, gid_x, y, pixel); \
    } \
  } \
 \
  __syncthreads(); \
 \
  /* assemble segments and write partial histograms */ \
  if (lid < min(SEGMENT_SIZE,num_bins)) { \
    bin = UNTAG(BIN_TYPE, warp_hist[lid]); \
    _Pragma("unroll") \
    for (unsigned int i = 1; i < NUM_WARPS; ++i) { \
      bin = REDUCE(bin, UNTAG(BIN_TYPE, warp_hist[i * SEGMENT_SIZE + lid])); \
    } \
    output[offset + lid + (blockIdx.x * num_bins)] = bin; \
  } \
 \
  /* merge partial histograms */ \
  if (gridDim.x > 1) { \
    __shared__ bool last_block_for_segment; \
 \
    if (lid == 0) { \
      unsigned int ticket = atomicInc(&finished_blocks_##NAME[blockIdx.y], gridDim.x); \
      last_block_for_segment = (ticket == gridDim.x-1); \
    } \
    __syncthreads(); \
 \
    if (last_block_for_segment) { \
      unsigned int blocksize = WARP_SIZE * NUM_WARPS; \
      unsigned int runs = (SEGMENT_SIZE + blocksize - 1) / blocksize; \
 \
      _Pragma("unroll") \
      for (unsigned int i = 0; i < runs; ++i) { \
        if (lid < SEGMENT_SIZE) { \
          bin = output[offset + lid]; \
 \
          _Pragma("unroll") \
          for (unsigned yi = 1; yi < gridDim.x; ++yi) { \
            bin = REDUCE(bin, output[offset + yi*num_bins + lid]); \
          } \
 \
          output[offset + lid] = bin; \
        } \
        lid += blocksize; \
      } \
    } \
  } \
}

//#endif  // __HIPACC_CU_RED_HPP__
