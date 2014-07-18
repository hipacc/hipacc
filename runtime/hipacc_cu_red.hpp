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

//#endif  // __HIPACC_CU_RED_HPP__

