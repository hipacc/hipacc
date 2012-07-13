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

#ifndef __HIPACC_CUDA_INTERPOLATE_HPP__
#define __HIPACC_CUDA_INTERPOLATE_HPP__

#define IMG_PARM(TYPE) const TYPE *img
#define TEX_PARM(TYPE) const struct texture<TYPE, cudaTextureType1D, cudaReadModeElementType> &texRef
#define CONST_PARM(TYPE) , const TYPE const_val
#define NO_PARM(TYPE)
#define IMG(x, y, stride, const_val) img[(x) + (y)*(stride)]
#define TEX(x, y, stride, const_val) tex1Dfetch(texRef, (x) + (y)*(stride))
#define IMG_CONST(x, y, stride, const_val) ((x)<0||(y)<0)<0?const_val:img[(x) + (y)*(stride)]
#define TEX_CONST(x, y, stride, const_val) ((x)<0||(y)<0)<0?const_val:tex1Dfetch(texRef, (x) + (y)*(stride))

// border handling: CLAMP
#define BH_CLAMP_LOWER(idx, lower, stride) bh_clamp_lower(idx, lower)
#define BH_CLAMP_UPPER(idx, upper, stride) bh_clamp_upper(idx, upper)
__device__ inline int bh_clamp_lower(int idx, int lower) {
    if (idx < lower) idx = lower;
    return idx;
}
__device__ inline int bh_clamp_upper(int idx, int upper) {
    if (idx >= upper) idx = upper-1;
    return idx;
}

// border handling: REPEAT
#define BH_REPEAT_LOWER(idx, lower, stride) bh_repeat_lower(idx, lower, stride)
#define BH_REPEAT_UPPER(idx, upper, stride) bh_repeat_upper(idx, upper, stride)
__device__ inline int bh_repeat_lower(int idx, int lower, int stride) {
    while (idx < lower) idx += stride;
    return idx;
}
__device__ inline int bh_repeat_upper(int idx, int upper, int stride) {
    while (idx >= upper) idx -= stride;
    return idx;
}

// border handling: MIRROR
#define BH_MIRROR_LOWER(idx, lower, stride) bh_mirror_lower(idx, lower)
#define BH_MIRROR_UPPER(idx, upper, stride) bh_mirror_upper(idx, upper)
__device__ inline int bh_mirror_lower(int idx, int lower) {
    if (idx < lower) idx = lower + (lower - idx-1);
    return idx;
}
__device__ inline int bh_mirror_upper(int idx, int upper) {
    if (idx >= upper) idx = upper - (idx+1 - upper);
    return idx;
}

// border handling: CONSTANT
#define BH_CONSTANT_LOWER(idx, lower, stride) bh_constant_lower(idx, lower)
#define BH_CONSTANT_UPPER(idx, upper, stride) bh_constant_upper(idx, upper)
__device__ inline int bh_constant_lower(int idx, int lower) {
    if (idx < lower) return -1;
    return idx;
}
__device__ inline int bh_constant_upper(int idx, int upper) {
    if (idx >= upper) return -1;
    return idx;
}

// border handling: UNDEFINED
#define NO_BH(idx, limit, stride) (idx)
__device__ inline int no_bh(int idx) {
    return idx;
}

// CLAMP, REPEAT, MIRROR, CONSTANT, UNDEFINED (NO_BH)
#define DEFINE_BH_VARIANTS(METHOD, IM, DATA_TYPE) \
DEFINE_BH_VARIANT(METHOD, interpolate_##IM##_clamp, DATA_TYPE, NO_PARM, IMG, TEX, BH_CLAMP_LOWER, BH_CLAMP_UPPER) \
DEFINE_BH_VARIANT(METHOD, interpolate_##IM##_repeat, DATA_TYPE, NO_PARM, IMG, TEX, BH_REPEAT_LOWER, BH_REPEAT_UPPER) \
DEFINE_BH_VARIANT(METHOD, interpolate_##IM##_mirror, DATA_TYPE, NO_PARM, IMG, TEX, BH_MIRROR_LOWER, BH_MIRROR_UPPER) \
DEFINE_BH_VARIANT(METHOD, interpolate_##IM##_constant, DATA_TYPE, CONST_PARM, IMG_CONST, TEX_CONST, BH_CONSTANT_LOWER, BH_CONSTANT_UPPER) \
METHOD(interpolate_##IM##_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), NO_PARM(DATA_TYPE), IMG, NO_BH, NO_BH, NO_BH, NO_BH) \
METHOD(interpolate_##IM##_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), NO_PARM(DATA_TYPE), TEX, NO_BH, NO_BH, NO_BH, NO_BH)

// xl, xu, yl, yu
#define DEFINE_BH_VARIANT(METHOD, NAME, DATA_TYPE, CPARM, IMGACC, TEXACC, BH_LOWER, BH_UPPER) \
METHOD(NAME##_l_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, BH_LOWER, NO_BH, NO_BH, NO_BH) \
METHOD(NAME##_r_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, NO_BH, BH_UPPER, NO_BH, NO_BH) \
METHOD(NAME##_t_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, NO_BH, NO_BH, BH_LOWER, NO_BH) \
METHOD(NAME##_b_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, NO_BH, NO_BH, NO_BH, BH_UPPER) \
METHOD(NAME##_tl_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, BH_LOWER, NO_BH, BH_LOWER, NO_BH) \
METHOD(NAME##_tr_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, NO_BH, BH_UPPER, BH_LOWER, NO_BH) \
METHOD(NAME##_bl_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, BH_LOWER, NO_BH, NO_BH, BH_UPPER) \
METHOD(NAME##_br_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, NO_BH, BH_UPPER, NO_BH, BH_UPPER) \
METHOD(NAME##_tblr_gmem, DATA_TYPE, IMG_PARM(DATA_TYPE), CPARM(DATA_TYPE), IMGACC, BH_LOWER, BH_UPPER, BH_LOWER, BH_UPPER) \
METHOD(NAME##_l_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, BH_LOWER, NO_BH, NO_BH, NO_BH) \
METHOD(NAME##_r_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, NO_BH, BH_UPPER, NO_BH, NO_BH) \
METHOD(NAME##_t_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, NO_BH, NO_BH, BH_LOWER, NO_BH) \
METHOD(NAME##_b_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, NO_BH, NO_BH, NO_BH, BH_UPPER) \
METHOD(NAME##_tl_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, BH_LOWER, NO_BH, BH_LOWER, NO_BH) \
METHOD(NAME##_tr_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, NO_BH, BH_UPPER, BH_LOWER, NO_BH) \
METHOD(NAME##_bl_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, BH_LOWER, NO_BH, NO_BH, BH_UPPER) \
METHOD(NAME##_br_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, NO_BH, BH_UPPER, NO_BH, BH_UPPER) \
METHOD(NAME##_tblr_tex, DATA_TYPE, TEX_PARM(DATA_TYPE), CPARM(DATA_TYPE), TEXACC, BH_LOWER, BH_UPPER, BH_LOWER, BH_UPPER) \


// Bilinear Interpolation
#define INTERPOLATE_LINEAR_FILTERING_CUDA(NAME, DATA_TYPE, PARM, CPARM, ACCESS, BHXL, BHXU, BHYL, BHYU) \
__device__ DATA_TYPE NAME(PARM, const int stride, float x_mapped, float y_mapped, const int rwidth, const int rheight, const int global_offset_x, const int global_offset_y CPARM) { \
    float xb = x_mapped - 0.5f; \
    float yb = y_mapped - 0.5f; \
    int x_int = xb; \
    int y_int = yb; \
    float x_frac = xb - x_int; \
    float y_frac = yb - y_int; \
    x_int += global_offset_x; \
    y_int += global_offset_y; \
 \
    return \
        (1.0f-x_frac) * (1.0f-y_frac) * ACCESS(BHXU(BHXL(x_int  , global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int  , global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) + \
              x_frac  * (1.0f-y_frac) * ACCESS(BHXU(BHXL(x_int+1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int  , global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) + \
        (1.0f-x_frac) *       y_frac  * ACCESS(BHXU(BHXL(x_int  , global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int+1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) + \
              x_frac  *       y_frac  * ACCESS(BHXU(BHXL(x_int+1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int+1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val); \
} \

DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, char)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, short int)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, int)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, unsigned char)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, unsigned short int)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, unsigned int)
DEFINE_BH_VARIANTS(INTERPOLATE_LINEAR_FILTERING_CUDA, lf, float)


// Cubic Interpolation
__device__ float bicubic_spline(float diff) { 
    diff = abs(diff); 
    float a = -0.5f; 
 
    if (diff < 1.0f) { 
        return (a + 2.0f) *diff*diff*diff - (a + 3.0f)*diff*diff + 1; 
    } else if (diff < 2.0f) { 
        return a * diff*diff*diff - 5.0f * a * diff*diff + 8.0f * a * diff - 4.0f * a; 
    } else return 0.0f; 
} 
 
#define INTERPOLATE_CUBIC_FILTERING_CUDA(NAME, DATA_TYPE, PARM, CPARM, ACCESS, BHXL, BHXU, BHYL, BHYU) \
__device__ DATA_TYPE NAME(PARM, const int stride, float x_mapped, float y_mapped, const int rwidth, const int rheight, const int global_offset_x, const int global_offset_y CPARM) { \
    float xb = x_mapped - 0.5f; \
    float yb = y_mapped - 0.5f; \
    int x_int = xb; \
    int y_int = yb; \
    float x_frac = xb - x_int; \
    float y_frac = yb - y_int; \
    x_int += global_offset_x; \
    y_int += global_offset_y; \
 \
    float y0 = \
        ACCESS(BHXU(BHXL(x_int - 1 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 3); \
    float y1 = \
        ACCESS(BHXU(BHXL(x_int - 1 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 3); \
    float y2 = \
        ACCESS(BHXU(BHXL(x_int - 1 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 3); \
    float y3 = \
        ACCESS(BHXU(BHXL(x_int - 1 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 1 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * bicubic_spline(x_frac - 1 + 3); \
 \
    return y0*bicubic_spline(y_frac - 1 + 0) + \
        y1*bicubic_spline(y_frac - 1 + 1) + \
        y2*bicubic_spline(y_frac - 1 + 2) + \
        y3*bicubic_spline(y_frac - 1 + 3); \
}

DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, char)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, short int)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, int)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, unsigned char)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, unsigned short int)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, unsigned int)
DEFINE_BH_VARIANTS(INTERPOLATE_CUBIC_FILTERING_CUDA, cf, float)


// Lanczos3 Interpolation
#define MY_PI 3.141592654f
__device__ float lanczos(float diff) {
    diff = fabsf(diff);
    float l = 3.0f;

    if (diff==0.0f) return 1.0f;
    else if (diff < l) {
        return l * (sinf(MY_PI*diff/l) * sinf(MY_PI*diff)) / (MY_PI*MY_PI*diff*diff);
    } else return 0.0f;
}

#define INTERPOLATE_LANCZOS_FILTERING_CUDA(NAME, DATA_TYPE, PARM, CPARM, ACCESS, BHXL, BHXU, BHYL, BHYU) \
__device__ DATA_TYPE NAME(PARM, const int stride, float x_mapped, float y_mapped, const int rwidth, const int rheight, const int global_offset_x, const int global_offset_y CPARM) { \
    float xb = x_mapped - 0.5f; \
    float yb = y_mapped - 0.5f; \
    int x_int = xb; \
    int y_int = yb; \
    float x_frac = xb - x_int; \
    float y_frac = yb - y_int; \
    x_int += global_offset_x; \
    y_int += global_offset_y; \
 \
    float y0 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 4) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 0, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
    float y1 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 1, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
    float y2 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 4) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 2, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
    float y3 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 4) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 3, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
    float y4 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 4) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 4, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
    float y5 = \
        ACCESS(BHXU(BHXL(x_int - 2 + 0, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 0) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 1, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 1) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 2, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 2) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 3, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 3) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 4, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 4) + \
        ACCESS(BHXU(BHXL(x_int - 2 + 5, global_offset_x, rwidth), global_offset_x+rwidth, rwidth), BHYU(BHYL(y_int - 1 + 5, global_offset_y, rheight), global_offset_y+rheight, rheight), stride, const_val) * lanczos(x_frac - 2 + 5); \
 \
    return y0*lanczos(y_frac - 2 + 0) + \
        y1*lanczos(y_frac - 2 + 1) + \
        y2*lanczos(y_frac - 2 + 2) + \
        y3*lanczos(y_frac - 2 + 3) + \
        y4*lanczos(y_frac - 2 + 4) + \
        y5*lanczos(y_frac - 2 + 5); \
}

DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, char)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, short int)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, int)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, unsigned char)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, unsigned short int)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, unsigned int)
DEFINE_BH_VARIANTS(INTERPOLATE_LANCZOS_FILTERING_CUDA, l3, float)

#endif  // __HIPACC_CUDA_INTERPOLATE_HPP__

