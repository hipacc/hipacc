//
// Copyright (c) 2013, University of Erlangen-Nuremberg
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

//#ifndef __HIPACC_RS_RED_HPP__
//#define __HIPACC_RS_RED_HPP__

// define offset parameters required to specify a sub-region on the image on
// that the reduction should be applied
#ifdef USE_OFFSETS
#define OFFSET_X _red_offset_x
#define OFFSET_Y _red_offset_y
#else
#define OFFSET_X 0
#define OFFSET_Y 0
#endif
#define ALL(data, idx_x, idx_y, _red_stride, method) method(data, idx_x, idx_y)

#ifdef FS
#define RET_TYPE DATA_TYPE __attribute__((kernel))
#define IS_PARM
#define COMMA
#define RETURN(method, val) return val
#else
#define RET_TYPE void
#define IS_PARM DATA_TYPE *_IS
#define COMMA ,
#define RETURN(method, val) method = val
#endif

// step 1:
// reduce a 2D block stored to linear memory and store the reduced value to linear memory
#define REDUCTION_RS_2D(NAME, DATA_TYPE, ACCESS, REDUCE) \
RET_TYPE NAME(IS_PARM COMMA uint32_t x, uint32_t y) { \
    const int gid_x = x; \
    const int gid_y = y; \
 \
    DATA_TYPE val = ACCESS(_red_Input, gid_x + OFFSET_X, gid_y + OFFSET_Y, _red_stride, rsGetElementAt##_##DATA_TYPE); \
 \
    for (int j=1; j<_red_is_height; ++j) { \
        val = REDUCE(val, ACCESS(_red_Input, gid_x + OFFSET_X, j + gid_y + OFFSET_Y, _red_stride, rsGetElementAt##_##DATA_TYPE)); \
    } \
 \
    RETURN(ACCESS(_red_Output, x, 0, 0, *(DATA_TYPE*)rsGetElementAt), val); \
}

// step 2:
// reduce a 1D block and store the reduced value to the first element of linear
// memory
#define REDUCTION_RS_1D(NAME, DATA_TYPE, ACCESS, REDUCE) \
RET_TYPE NAME(IS_PARM) { \
    DATA_TYPE val = ACCESS(_red_Output, 0, 0, 0, rsGetElementAt##_##DATA_TYPE); \
 \
    for (int j=1; j<_red_num_elements; ++j) { \
        val = REDUCE(val, ACCESS(_red_Output, j, 0, 0, rsGetElementAt##_##DATA_TYPE)); \
    } \
 \
    RETURN(*_IS, val); \
}

//#endif  // __HIPACC_RS_RED_HPP__
