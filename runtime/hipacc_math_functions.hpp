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

#ifndef __HIPACC_MATH_FUNCTIONS_HPP__
#define __HIPACC_MATH_FUNCTIONS_HPP__

#ifndef __CUDACC_RTC__
#include <algorithm>
#include <cmath>
#include <cstdlib>
#endif

#include "hipacc_types.hpp"


// math operators
#define MAKE_MATH_BI(NEW_TYPE, BASIC_TYPE, RET_TYPE, SUFFIX) \
 \
 /* acos */ \
 \
ATTRIBUTES RET_TYPE acos##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(acos(a.x), acos(a.y), acos(a.z), acos(a.w)); \
} \
 \
 /* acosh */ \
 \
ATTRIBUTES RET_TYPE acosh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(acosh(a.x), acosh(a.y), acosh(a.z), acosh(a.w)); \
} \
 \
 /* asin */ \
 \
ATTRIBUTES RET_TYPE asin##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(asin(a.x), asin(a.y), asin(a.z), asin(a.w)); \
} \
 \
 /* asinh */ \
 \
ATTRIBUTES RET_TYPE asinh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(asinh(a.x), asinh(a.y), asinh(a.z), asinh(a.w)); \
} \
 /* atan */ \
 \
ATTRIBUTES RET_TYPE atan##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(atan(a.x), atan(a.y), atan(a.z), atan(a.w)); \
} \
 /* atan2 */ \
 \
ATTRIBUTES RET_TYPE atan2##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(atan2(a.x, b.x), atan2(a.y, b.y), atan2(a.z, b.z), atan2(a.w, b.w)); \
} \
 /* atanh */ \
 \
ATTRIBUTES RET_TYPE atanh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(atanh(a.x), atanh(a.y), atanh(a.z), atanh(a.w)); \
} \
 /* cbrt */ \
 \
ATTRIBUTES RET_TYPE cbrt##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(cbrt(a.x), cbrt(a.y), cbrt(a.z), cbrt(a.w)); \
} \
 /* ceil */ \
 \
ATTRIBUTES RET_TYPE ceil##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(ceil(a.x), ceil(a.y), ceil(a.z), ceil(a.w)); \
} \
 /* copysign */ \
 \
ATTRIBUTES RET_TYPE copysign##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(copysign(a.x, b.x), copysign(a.y, b.y), copysign(a.z, b.z), copysign(a.w, b.w)); \
} \
 /* cos */ \
 \
ATTRIBUTES RET_TYPE cos##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(cos(a.x), cos(a.y), cos(a.z), cos(a.w)); \
} \
 /* cosh */ \
 \
ATTRIBUTES RET_TYPE cosh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(cosh(a.x), cosh(a.y), cosh(a.z), cosh(a.w)); \
} \
 /* erfc */ \
 \
ATTRIBUTES RET_TYPE erfc##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(erfc(a.x), erfc(a.y), erfc(a.z), erfc(a.w)); \
} \
 /* erf */ \
 \
ATTRIBUTES RET_TYPE erf##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(erf(a.x), erf(a.y), erf(a.z), erf(a.w)); \
} \
 /* exp */ \
 \
ATTRIBUTES RET_TYPE exp##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(exp(a.x), exp(a.y), exp(a.z), exp(a.w)); \
} \
 /* exp2 */ \
 \
ATTRIBUTES RET_TYPE exp2##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(exp2(a.x), exp2(a.y), exp2(a.z), exp2(a.w)); \
} \
 /* exp10 -> not supported */ \
 \
 /* expm1 */ \
 \
ATTRIBUTES RET_TYPE expm1##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(expm1(a.x), expm1(a.y), expm1(a.z), expm1(a.w)); \
} \
 /* fabs */ \
 \
ATTRIBUTES RET_TYPE fabs##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w)); \
} \
 /* fdim */ \
 \
ATTRIBUTES RET_TYPE fdim##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(fdim(a.x, b.x), fdim(a.y, b.y), fdim(a.z, b.z), fdim(a.w, b.w)); \
} \
 /* floor */ \
 \
ATTRIBUTES RET_TYPE floor##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(floor(a.x), floor(a.y), floor(a.z), floor(a.w)); \
} \
 /* fma */ \
 \
ATTRIBUTES RET_TYPE fma##SUFFIX(NEW_TYPE a, NEW_TYPE b, NEW_TYPE c) { \
    return make_##RET_TYPE(fma(a.x, b.x, c.x), fma(a.y, b.y, c.y), fma(a.z, b.z, c.z), fma(a.w, b.w, c.w)); \
} \
 /* fmax */ \
 \
ATTRIBUTES RET_TYPE fmax##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), fmax(a.w, b.w)); \
} \
 /* fmin */ \
 \
ATTRIBUTES RET_TYPE fmin##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z), fmin(a.w, b.w)); \
} \
 /* fmod */ \
 \
ATTRIBUTES RET_TYPE fmod##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z), fmod(a.w, b.w)); \
} \
 /* fract -> not supported */ \
 \
 /* frexp */ \
 \
ATTRIBUTES RET_TYPE frexp##SUFFIX(NEW_TYPE a, int4 *b) { \
    int x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = frexp(a.x, &x); \
    tmp.y = frexp(a.y, &y); \
    tmp.z = frexp(a.z, &z); \
    tmp.w = frexp(a.w, &w); \
    b->x = x; b->y = y; b->z = z; b->w = w; \
    return tmp; \
} \
 /* hypot */ \
 \
ATTRIBUTES RET_TYPE hypot##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(hypot(a.x, b.x), hypot(a.y, b.y), hypot(a.z, b.z), hypot(a.w, b.w)); \
} \
 /* ilogb */ \
 \
ATTRIBUTES int4 ilogb##SUFFIX(NEW_TYPE a) { \
    return make_int4(ilogb(a.x), ilogb(a.y), ilogb(a.z), ilogb(a.w)); \
} \
 /* ldexp */ \
 \
ATTRIBUTES RET_TYPE ldexp##SUFFIX(NEW_TYPE a, int4 b) { \
    return make_##RET_TYPE(ldexp(a.x, b.x), ldexp(a.y, b.y), ldexp(a.z, b.z), ldexp(a.w, b.w)); \
} \
 /* lgamma */ \
 \
ATTRIBUTES RET_TYPE lgamma##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(lgamma(a.x), lgamma(a.y), lgamma(a.z), lgamma(a.w)); \
} \
 /* lgamma_r -> not supported */ \
 \
 /* log */ \
 \
ATTRIBUTES RET_TYPE log##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(log(a.x), log(a.y), log(a.z), log(a.w)); \
} \
 /* log2 */ \
 \
ATTRIBUTES RET_TYPE log2##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(log2(a.x), log2(a.y), log2(a.z), log2(a.w)); \
} \
 /* log10 */ \
 \
ATTRIBUTES RET_TYPE log10##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(log10(a.x), log10(a.y), log10(a.z), log10(a.w)); \
} \
 /* log1p */ \
 \
ATTRIBUTES RET_TYPE log1p##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(log1p(a.x), log1p(a.y), log1p(a.z), log1p(a.w)); \
} \
 /* logb */ \
 \
ATTRIBUTES RET_TYPE logb##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(logb(a.x), logb(a.y), logb(a.z), logb(a.w)); \
} \
 /* modf */ \
 \
ATTRIBUTES RET_TYPE modf##SUFFIX(NEW_TYPE a, NEW_TYPE *b) { \
    BASIC_TYPE x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = modf##SUFFIX(a.x, &x); \
    tmp.y = modf##SUFFIX(a.y, &y); \
    tmp.z = modf##SUFFIX(a.z, &z); \
    tmp.w = modf##SUFFIX(a.w, &w); \
    b->x = x; b->y = y; b->z = z; b->w = w; \
    return tmp; \
} \
 /* nearbyint -> not supported */ \
 \
 /* nextafter */ \
 \
ATTRIBUTES RET_TYPE nextafter##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(nextafter(a.x, b.x), nextafter(a.y, b.y), nextafter(a.z, b.z), nextafter(a.w, b.w)); \
} \
 /* pow */ \
 \
ATTRIBUTES RET_TYPE pow##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w)); \
} \
 /* pown */ \
 \
ATTRIBUTES RET_TYPE pown##SUFFIX(NEW_TYPE a, int4 b) { \
    return make_##RET_TYPE(pow(a.x, (BASIC_TYPE)b.x), pow(a.y, (BASIC_TYPE)b.y), pow(a.z, (BASIC_TYPE)b.z), pow(a.w, (BASIC_TYPE)b.w)); \
} \
 /* powr */ \
 \
ATTRIBUTES RET_TYPE powr##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w)); \
} \
 /* remainder */ \
 \
ATTRIBUTES RET_TYPE remainder##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(remainder(a.x, b.x), remainder(a.y, b.y), remainder(a.z, b.z), remainder(a.w, b.w)); \
} \
 /* remquo */ \
 \
ATTRIBUTES RET_TYPE remquo##SUFFIX(NEW_TYPE a, NEW_TYPE b, int4 *c) { \
    int x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = remquo##SUFFIX(a.x, b.x, &x); \
    tmp.y = remquo##SUFFIX(a.y, b.y, &y); \
    tmp.z = remquo##SUFFIX(a.z, b.z, &z); \
    tmp.w = remquo##SUFFIX(a.w, b.w, &w); \
    c->x = x; c->y = y; c->z = z; c->w = w; \
    return tmp; \
} \
 /* rint */ \
 \
ATTRIBUTES RET_TYPE rint##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(rint(a.x), rint(a.y), rint(a.z), rint(a.w)); \
} \
 /* round */ \
 \
ATTRIBUTES RET_TYPE round##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(round(a.x), round(a.y), round(a.z), round(a.w)); \
} \
 /* rsqrt -> not supported */ \
 \
 /* sin */ \
 \
ATTRIBUTES RET_TYPE sin##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(sin(a.x), sin(a.y), sin(a.z), sin(a.w)); \
} \
 /* sincos -> not supported */ \
 \
 /* sinh */ \
 \
ATTRIBUTES RET_TYPE sinh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(sinh(a.x), sinh(a.y), sinh(a.z), sinh(a.w)); \
} \
 /* sqrt */ \
 \
ATTRIBUTES RET_TYPE sqrt##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w)); \
} \
 /* tan */ \
 \
ATTRIBUTES RET_TYPE tan##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(tan(a.x), tan(a.y), tan(a.z), tan(a.w)); \
} \
 /* tanh */ \
 \
ATTRIBUTES RET_TYPE tanh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(tanh(a.x), tanh(a.y), tanh(a.z), tanh(a.w)); \
} \
 /* tgamma */ \
 \
ATTRIBUTES RET_TYPE tgamma##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(tgamma(a.x), tgamma(a.y), tgamma(a.z), tgamma(a.w)); \
} \
 /* trunc */ \
 \
ATTRIBUTES RET_TYPE trunc##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(trunc(a.x), trunc(a.y), trunc(a.z), trunc(a.w)); \
} \


MAKE_MATH_BI(float4,    float,  float4, f)
#if defined __ANDROID__ and not defined CL_VERSION_1_1
// double functions not supported on Android (Renderscript)
#else
MAKE_MATH_BI(double4,   double, double4, )
#endif


// generic math functions
#if defined __CUDACC__
#define MAKE_MATH_BI_GEN(NEW_TYPE, BASIC_TYPE) \
        MAKE_MATH_BI_GEN_VEC(NEW_TYPE, BASIC_TYPE)
#else
#define MAKE_MATH_BI_GEN(NEW_TYPE, BASIC_TYPE) \
        MAKE_MATH_BI_GEN_BAS(NEW_TYPE, BASIC_TYPE) \
        MAKE_MATH_BI_GEN_VEC(NEW_TYPE, BASIC_TYPE)
#endif


#define MAKE_MATH_BI_GEN_BAS(NEW_TYPE, BASIC_TYPE) \
 \
 /* min */ \
 \
ATTRIBUTES BASIC_TYPE min(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a < b ? a : b); \
} \
 \
 /* max */ \
 \
ATTRIBUTES BASIC_TYPE max(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a > b ? a : b); \
} \


#define MAKE_MATH_BI_GEN_VEC(NEW_TYPE, BASIC_TYPE) \
 \
 /* min */ \
 \
ATTRIBUTES NEW_TYPE min(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); \
} \
 \
ATTRIBUTES NEW_TYPE min(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b)); \
} \
 \
ATTRIBUTES NEW_TYPE min(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(min(a, b.x), min(a, b.y), min(a, b.z), min(a, b.w)); \
} \
 \
 /* max */ \
 \
ATTRIBUTES NEW_TYPE max(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); \
} \
 \
ATTRIBUTES NEW_TYPE max(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b)); \
} \
 \
ATTRIBUTES NEW_TYPE max(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(max(a, b.x), max(a, b.y), max(a, b.z), max(a, b.w)); \
}

MAKE_MATH_BI_GEN(char4,     char)
MAKE_MATH_BI_GEN(uchar4,    uchar)
MAKE_MATH_BI_GEN(short4,    short)
MAKE_MATH_BI_GEN(ushort4,   ushort)
MAKE_MATH_BI_GEN(int4,      int)
MAKE_MATH_BI_GEN(uint4,     uint)
#if defined __CUDACC__
#else
MAKE_MATH_BI_GEN(long4,     long)
MAKE_MATH_BI_GEN(ulong4,    ulong)
#endif
MAKE_MATH_BI_GEN(float4,    float)
MAKE_MATH_BI_GEN(double4,   double)


// integer math operators: abs, labs
#define MAKE_MATH_BI_INT(NEW_TYPE, BASIC_TYPE, RET_TYPE, PREFIX) \
 /* abs */ \
ATTRIBUTES RET_TYPE PREFIX##abs(NEW_TYPE a) { \
    return make_##RET_TYPE(PREFIX##abs(a.x), PREFIX##abs(a.y), PREFIX##abs(a.z), PREFIX##abs(a.w)); \
}

MAKE_MATH_BI_INT(int4,  int,    int4,    )
MAKE_MATH_BI_INT(long4, long,   long4,  l)

#endif  // __HIPACC_MATH_FUNCTIONS_HPP__

