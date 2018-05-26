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

//#ifndef __HIPACC_CL_RED_HPP__
//#define __HIPACC_CL_RED_HPP__

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
#define OFFSET_CHECK_X_STRIDE gid_x + get_local_size(0) >= offset_x && gid_x + get_local_size(0) < is_width + offset_x
#else
#define OFFSETS
#define IS_HEIGHT height
#define OFFSET_BLOCK 0
#define OFFSET_Y 0
#define OFFSET_CHECK_X gid_x < width
#define OFFSET_CHECK_X_STRIDE gid_x + get_local_size(0) < width
#endif
#ifdef USE_ARRAY_2D
#define READ(INPUT, X, Y, STRIDE, METHOD) METHOD(INPUT, img_sampler, (int2)(X, Y)).x
#define INPUT_PARM(DATA_TYPE, INPUT_NAME) __read_only image2d_t INPUT_NAME
#else
#define READ(INPUT, X, Y, STRIDE, METHOD) INPUT[(X) + (Y)*STRIDE]
#define INPUT_PARM(DATA_TYPE, INPUT_NAME) __global DATA_TYPE *INPUT_NAME
#endif


// step 1:
// reduce a 2D block stored to linear memory or an Image object and store the reduced value to linear memory
__constant sampler_t img_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define REDUCTION_CL_2D(NAME, DATA_TYPE, REDUCE, IMG_ACC) \
__kernel __attribute__((reqd_work_group_size(BS, 1, 1))) void NAME( \
        INPUT_PARM(DATA_TYPE, input), __global DATA_TYPE *output, \
        const unsigned int width, const unsigned int height, \
        const unsigned int stride OFFSETS) { \
    const unsigned int gid_x =   2*get_local_size(0) * get_group_id(0) + get_local_id(0) + OFFSET_BLOCK; \
    const unsigned int gid_y = PPT*get_local_size(1) * get_group_id(1) + get_local_id(1); \
    const unsigned int tid = get_local_id(0); \
 \
    __local DATA_TYPE sdata[BS]; \
 \
    DATA_TYPE val; \
 \
    if (OFFSET_CHECK_X) { \
        if (OFFSET_CHECK_X_STRIDE) { \
            val = REDUCE(READ(input, gid_x, gid_y + OFFSET_Y, stride, IMG_ACC), READ(input, gid_x + get_local_size(0), gid_y + OFFSET_Y, stride, IMG_ACC)); \
        } else { \
            val = READ(input, gid_x, gid_y + OFFSET_Y, stride, IMG_ACC); \
        } \
    } else { \
        val = READ(input, gid_x + get_local_size(0), gid_y + OFFSET_Y, stride, IMG_ACC); \
    } \
 \
    for (int j=1; j < PPT; ++j) { \
        if (j+gid_y < IS_HEIGHT) { \
            if (OFFSET_CHECK_X) { \
                val = REDUCE(val, READ(input, gid_x, j+gid_y + OFFSET_Y, stride, IMG_ACC)); \
            } \
            if (OFFSET_CHECK_X_STRIDE) { \
                val = REDUCE(val, READ(input, gid_x+get_local_size(0), j+gid_y + OFFSET_Y, stride, IMG_ACC)); \
            } \
        } \
    } \
    sdata[tid] = val; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (int s=get_local_size(0)/2; s>0; s>>=1) { \
        if (tid < s) { \
            sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (tid == 0) output[get_group_id(0) + get_num_groups(0)*get_group_id(1)] = sdata[0]; \
}


// step 2:
// reduce a 1D block and store the reduced value to the first element of linear
// memory
#define REDUCTION_CL_1D(NAME, DATA_TYPE, REDUCE) \
__kernel void NAME(__global const DATA_TYPE *input, __global DATA_TYPE *output, \
        const unsigned int num_elements, const unsigned int iterations) { \
    const unsigned int tid = get_local_id(0); \
    const unsigned int i = get_group_id(0)*(get_local_size(0)*iterations) + get_local_id(0); \
 \
    __local DATA_TYPE sdata[BS]; \
 \
    DATA_TYPE val = input[i]; \
 \
    for (int j=1; j < iterations; ++j) { \
        if (i + j*get_local_size(0) < num_elements) { \
            val = REDUCE(val, input[i + j*get_local_size(0)]); \
        } \
    } \
    sdata[tid] = val; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (int s=get_local_size(0)/2; s>0; s>>=1) { \
        if (tid < s) { \
            sdata[tid] = val = REDUCE(val, sdata[tid + s]); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (tid == 0) output[get_group_id(0)] = sdata[0]; \
}


// Helper macros for accumulating integers in segmented binning/reduction
// (first 5 bits are used for tagging with thread id)
#define UNTAG_INT(BIN_TYPE, VAL) \
  (VAL) & ~(0xF8 << ((sizeof(BIN_TYPE)-1)*8))

#define TAG_INT(BIN_TYPE, VAL) \
  (VAL) | (get_local_id(0) << (3 + ((sizeof(BIN_TYPE)-1)*8)))

#define ACCU_INT(BIN_TYPE, PTR, REDUCE) \
  __local volatile BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  do { \
    old = UNTAG_INT(BIN_TYPE, *address); \
    val = TAG_INT(BIN_TYPE, REDUCE(old, bin)); \
    *address = val; \
  } while (*address != val);


// Helper macros for accumulating using cmpxchg in segmented binning/reduction
#define UNTAG_NONE(BIN_TYPE, VAL) (VAL)

#define ACCU_CAS_32(BIN_TYPE, PTR, REDUCE) \
  __local BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  unsigned int *oldi = (unsigned int*)&old, *vali = (unsigned int*)&val; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
  } while (atomic_cmpxchg((__local unsigned int*)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_64(BIN_TYPE, PTR, REDUCE) \
  __local BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  ulong *oldi = (ulong*)&old, *vali = (ulong*)&val; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
  } while (atom_cmpxchg((__local ulong*)address, *oldi, *vali) != *oldi);

#define ACCU_CAS_GT64(BIN_TYPE, PTR, REDUCE) \
  __local BIN_TYPE* address = PTR; \
  BIN_TYPE old, val; \
  ulong *oldi = (ulong*)&old, *vali = (ulong*)&val; \
  bool run = true; \
  do { \
    old = *address; \
    val = REDUCE(old, bin); \
    if (atom_cmpxchg((__local ulong*)address, *oldi, *vali) == *oldi) { \
      run = false; \
      /* 64bit CAS succeeded, winning thread in warp => write remaining bits */ \
      *address = val; \
    } \
  } while (run);


#ifndef SEGMENT_SIZE
# define SEGMENT_SIZE 128
#endif


// Binning and reduction with conflicts solved via cmpxchg or thread id
//
// Variables:
//  - PIXEL_TYPE: Type of image pixels
//  - BIN_TYPE:   Type of histogram bins
//  - NUM_BINS:   Number of histogram bins
//
// Configuration:
//  - WARP_SIZE: Threads per warp (32 for NVIDIA, 64 for AMD)
//  - NUM_WARPS: Warps per block (affects block size and local memory size)
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
// Kernel2D steps:
//  1) Each warp computes a single SEGMENT in local memory.
//  2) SEGMENTS of all warps within a block are assembled to single SEGMENT
//     and stored in global memory.
//      - x SEGMENTS represent partial (full size, not entire image) histogram,
//      - where x = ceil(NUM_BINS/SEGMENT_SIZE)
//
// Kernel1D steps:
//  1) y partial histograms are merged by NUM_BIN many threads blocks to a
//     single final histogram.
//      - where y = NUM_HISTS
#define BINNING_CL_2D_SEGMENTED(NAME2D, NAME1D, PIXEL_TYPE, BIN_TYPE, REDUCE, BINNING, ACCU, UNTAG, WARP_SIZE, NUM_WARPS, NUM_HISTS, PPT, ZERO) \
inline void BINNING##Put(__local BIN_TYPE *lmem, uint offset, uint idx, BIN_TYPE val) { \
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
__kernel __attribute__((reqd_work_group_size(WARP_SIZE, NUM_WARPS, 1))) void NAME2D(__global const PIXEL_TYPE *input, \
        __global BIN_TYPE *output, const unsigned int width, const unsigned int height, \
        const unsigned int stride, const unsigned int num_bins, \
        const unsigned int offset_x, const unsigned int offset_y) { \
  unsigned int lid = get_local_id(0) + get_local_id(1) * WARP_SIZE; \
 \
  __local BIN_TYPE warp_hist[NUM_WARPS*SEGMENT_SIZE]; \
  __local BIN_TYPE* lhist = &warp_hist[get_local_id(1) * SEGMENT_SIZE]; \
 \
  /* initialize local memory */ \
  _Pragma("unroll") \
  for (unsigned int i = 0; i < SEGMENT_SIZE; i += WARP_SIZE) { \
    lhist[get_local_id(0) + i] = ZERO; \
  } \
 \
  barrier(CLK_LOCAL_MEM_FENCE); \
 \
  /* compute histogram segments */ \
  unsigned int increment = NUM_HISTS * WARP_SIZE * NUM_WARPS; \
  unsigned int gpos = ((WARP_SIZE * NUM_WARPS) * get_group_id(0)) + (get_local_id(1) * WARP_SIZE) + get_local_id(0); \
  unsigned int end = width * height/PPT; \
  unsigned int offset = get_group_id(1) * SEGMENT_SIZE; \
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
      PIXEL_TYPE pixel = input[ipos]; \
      ipos += inc; \
   \
      BINNING(lhist, offset, num_bins, gid_x, y, pixel); \
    } \
  } \
 \
  barrier(CLK_LOCAL_MEM_FENCE); \
 \
  /* assemble segments and write partial histograms */ \
  if (lid < min((uint)SEGMENT_SIZE,num_bins)) { \
    bin = UNTAG(BIN_TYPE, warp_hist[lid]); \
    _Pragma("unroll") \
    for (unsigned int i = 1; i < NUM_WARPS; ++i) { \
      bin = REDUCE(bin, UNTAG(BIN_TYPE, warp_hist[i * SEGMENT_SIZE + lid])); \
    } \
    output[offset + lid + (get_group_id(0) * num_bins)] = bin; \
  } \
 \
} \
 \
__kernel void NAME1D(__global BIN_TYPE *output, const unsigned int num_bins) { \
  unsigned int gid = get_global_id(0); \
  if (gid < num_bins) { \
    BIN_TYPE bin = output[gid]; \
    _Pragma("unroll") \
    for (unsigned yi = 1; yi < NUM_HISTS; ++yi) { \
      gid += num_bins; \
      bin = REDUCE(bin, output[gid]); \
    } \
 \
    output[get_global_id(0)] = bin; \
  } \
}


//#endif  // __HIPACC_CL_RED_HPP__
