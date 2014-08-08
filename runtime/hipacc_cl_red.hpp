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

//#endif  // __HIPACC_CL_RED_HPP__

