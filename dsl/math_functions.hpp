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

#ifndef __MATH_FUNCTIONS_HPP__
#define __MATH_FUNCTIONS_HPP__

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "types.hpp"

namespace hipacc {
namespace math {

#if defined __clang__
#define ATTRIBUTES inline
#elif defined __GNUC__
#define ATTRIBUTES inline
#else
#error "Only Clang, and gcc compilers supported!"
#endif


// math operators
#define MAKE_MATH_BI(NEW_TYPE, BASIC_TYPE, RET_TYPE, SUFFIX) \
 \
 /* acos */ \
 \
ATTRIBUTES RET_TYPE acos##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::acos(a.x), std::acos(a.y), std::acos(a.z), std::acos(a.w)); \
} \
 \
 /* acosh */ \
 \
ATTRIBUTES RET_TYPE acosh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::acosh(a.x), std::acosh(a.y), std::acosh(a.z), std::acosh(a.w)); \
} \
 \
 /* asin */ \
 \
ATTRIBUTES RET_TYPE asin##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::asin(a.x), std::asin(a.y), std::asin(a.z), std::asin(a.w)); \
} \
 \
 /* asinh */ \
 \
ATTRIBUTES RET_TYPE asinh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::asinh(a.x), std::asinh(a.y), std::asinh(a.z), std::asinh(a.w)); \
} \
 /* atan */ \
 \
ATTRIBUTES RET_TYPE atan##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::atan(a.x), std::atan(a.y), std::atan(a.z), std::atan(a.w)); \
} \
 /* atan2 */ \
 \
ATTRIBUTES RET_TYPE atan2##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::atan2(a.x, b.x), std::atan2(a.y, b.y), std::atan2(a.z, b.z), std::atan2(a.w, b.w)); \
} \
 /* atanh */ \
 \
ATTRIBUTES RET_TYPE atanh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::atanh(a.x), std::atanh(a.y), std::atanh(a.z), std::atanh(a.w)); \
} \
 /* cbrt */ \
 \
ATTRIBUTES RET_TYPE cbrt##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::cbrt(a.x), std::cbrt(a.y), std::cbrt(a.z), std::cbrt(a.w)); \
} \
 /* ceil */ \
 \
ATTRIBUTES RET_TYPE ceil##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z), std::ceil(a.w)); \
} \
 /* copysign */ \
 \
ATTRIBUTES RET_TYPE copysign##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::copysign(a.x, b.x), std::copysign(a.y, b.y), std::copysign(a.z, b.z), std::copysign(a.w, b.w)); \
} \
 /* cos */ \
 \
ATTRIBUTES RET_TYPE cos##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::cos(a.x), std::cos(a.y), std::cos(a.z), std::cos(a.w)); \
} \
 /* cosh */ \
 \
ATTRIBUTES RET_TYPE cosh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::cosh(a.x), std::cosh(a.y), std::cosh(a.z), std::cosh(a.w)); \
} \
 /* erfc */ \
 \
ATTRIBUTES RET_TYPE erfc##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::erfc(a.x), std::erfc(a.y), std::erfc(a.z), std::erfc(a.w)); \
} \
 /* erf */ \
 \
ATTRIBUTES RET_TYPE erf##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::erf(a.x), std::erf(a.y), std::erf(a.z), std::erf(a.w)); \
} \
 /* exp */ \
 \
ATTRIBUTES RET_TYPE exp##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::exp(a.x), std::exp(a.y), std::exp(a.z), std::exp(a.w)); \
} \
 /* exp2 */ \
 \
ATTRIBUTES RET_TYPE exp2##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::exp2(a.x), std::exp2(a.y), std::exp2(a.z), std::exp2(a.w)); \
} \
 /* exp10 -> not supported */ \
 \
 /* expm1 */ \
 \
ATTRIBUTES RET_TYPE expm1##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::expm1(a.x), std::expm1(a.y), std::expm1(a.z), std::expm1(a.w)); \
} \
 /* fabs */ \
 \
ATTRIBUTES RET_TYPE fabs##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::fabs(a.x), std::fabs(a.y), std::fabs(a.z), std::fabs(a.w)); \
} \
 /* fdim */ \
 \
ATTRIBUTES RET_TYPE fdim##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::fdim(a.x, b.x), std::fdim(a.y, b.y), std::fdim(a.z, b.z), std::fdim(a.w, b.w)); \
} \
 /* floor */ \
 \
ATTRIBUTES RET_TYPE floor##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::floor(a.x), std::floor(a.y), std::floor(a.z), std::floor(a.w)); \
} \
 /* fma */ \
 \
ATTRIBUTES RET_TYPE fma##SUFFIX(NEW_TYPE a, NEW_TYPE b, NEW_TYPE c) { \
    return make_##RET_TYPE(std::fma(a.x, b.x, c.x), std::fma(a.y, b.y, c.y), std::fma(a.z, b.z, c.z), std::fma(a.w, b.w, c.w)); \
} \
 /* fmax */ \
 \
ATTRIBUTES RET_TYPE fmax##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z), std::fmax(a.w, b.w)); \
} \
 /* fmin */ \
 \
ATTRIBUTES RET_TYPE fmin##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z), std::fmin(a.w, b.w)); \
} \
 /* fmod */ \
 \
ATTRIBUTES RET_TYPE fmod##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::fmod(a.x, b.x), std::fmod(a.y, b.y), std::fmod(a.z, b.z), std::fmod(a.w, b.w)); \
} \
 /* fract -> not supported */ \
 \
 /* frexp */ \
 \
ATTRIBUTES RET_TYPE frexp##SUFFIX(NEW_TYPE a, int4 *b) { \
    int x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = std::frexp(a.x, &x); \
    tmp.y = std::frexp(a.y, &y); \
    tmp.z = std::frexp(a.z, &z); \
    tmp.w = std::frexp(a.w, &w); \
    b->x = x; b->y = y; b->z = z; b->w = w; \
    return tmp; \
} \
 /* hypot */ \
 \
ATTRIBUTES RET_TYPE hypot##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::hypot(a.x, b.x), std::hypot(a.y, b.y), std::hypot(a.z, b.z), std::hypot(a.w, b.w)); \
} \
 /* ilogb */ \
 \
ATTRIBUTES int4 ilogb##SUFFIX(NEW_TYPE a) { \
    return make_int4(std::ilogb(a.x), std::ilogb(a.y), std::ilogb(a.z), std::ilogb(a.w)); \
} \
 /* ldexp */ \
 \
ATTRIBUTES RET_TYPE ldexp##SUFFIX(NEW_TYPE a, int4 b) { \
    return make_##RET_TYPE(std::ldexp(a.x, b.x), std::ldexp(a.y, b.y), std::ldexp(a.z, b.z), std::ldexp(a.w, b.w)); \
} \
 /* lgamma */ \
 \
ATTRIBUTES RET_TYPE lgamma##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::lgamma(a.x), std::lgamma(a.y), std::lgamma(a.z), std::lgamma(a.w)); \
} \
 /* lgamma_r -> not supported */ \
 \
 /* log */ \
 \
ATTRIBUTES RET_TYPE log##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::log(a.x), std::log(a.y), std::log(a.z), std::log(a.w)); \
} \
 /* log2 */ \
 \
ATTRIBUTES RET_TYPE log2##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::log2(a.x), std::log2(a.y), std::log2(a.z), std::log2(a.w)); \
} \
 /* log10 */ \
 \
ATTRIBUTES RET_TYPE log10##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::log10(a.x), std::log10(a.y), std::log10(a.z), std::log10(a.w)); \
} \
 /* log1p */ \
 \
ATTRIBUTES RET_TYPE log1p##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::log1p(a.x), std::log1p(a.y), std::log1p(a.z), std::log1p(a.w)); \
} \
 /* logb */ \
 \
ATTRIBUTES RET_TYPE logb##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::logb(a.x), std::logb(a.y), std::logb(a.z), std::logb(a.w)); \
} \
 /* modf */ \
 \
ATTRIBUTES RET_TYPE modf##SUFFIX(NEW_TYPE a, NEW_TYPE *b) { \
    BASIC_TYPE x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = std::modf(a.x, &x); \
    tmp.y = std::modf(a.y, &y); \
    tmp.z = std::modf(a.z, &z); \
    tmp.w = std::modf(a.w, &w); \
    b->x = x; b->y = y; b->z = z; b->w = w; \
    return tmp; \
} \
 /* nearbyint -> not supported */ \
 \
 /* nextafter */ \
 \
ATTRIBUTES RET_TYPE nextafter##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::nextafter(a.x, b.x), std::nextafter(a.y, b.y), std::nextafter(a.z, b.z), std::nextafter(a.w, b.w)); \
} \
 /* pow */ \
 \
ATTRIBUTES RET_TYPE pow##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::pow(a.x, b.x), std::pow(a.y, b.y), std::pow(a.z, b.z), std::pow(a.w, b.w)); \
} \
 /* pown */ \
 \
ATTRIBUTES RET_TYPE pown##SUFFIX(NEW_TYPE a, int4 b) { \
    return make_##RET_TYPE(std::pow(a.x, (BASIC_TYPE)b.x), std::pow(a.y, (BASIC_TYPE)b.y), std::pow(a.z, (BASIC_TYPE)b.z), std::pow(a.w, (BASIC_TYPE)b.w)); \
} \
 /* powr */ \
 \
ATTRIBUTES RET_TYPE powr##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::pow(a.x, b.x), std::pow(a.y, b.y), std::pow(a.z, b.z), std::pow(a.w, b.w)); \
} \
 /* remainder */ \
 \
ATTRIBUTES RET_TYPE remainder##SUFFIX(NEW_TYPE a, NEW_TYPE b) { \
    return make_##RET_TYPE(std::remainder(a.x, b.x), std::remainder(a.y, b.y), std::remainder(a.z, b.z), std::remainder(a.w, b.w)); \
} \
 /* remquo */ \
 \
ATTRIBUTES RET_TYPE remquo##SUFFIX(NEW_TYPE a, NEW_TYPE b, int4 *c) { \
    int x, y, z, w; /* clang does not allow to take the address of vector elements */ \
    RET_TYPE tmp; \
    tmp.x = std::remquo(a.x, b.x, &x); \
    tmp.y = std::remquo(a.y, b.y, &y); \
    tmp.z = std::remquo(a.z, b.z, &z); \
    tmp.w = std::remquo(a.w, b.w, &w); \
    c->x = x; c->y = y; c->z = z; c->w = w; \
    return tmp; \
} \
 /* rint */ \
 \
ATTRIBUTES RET_TYPE rint##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::rint(a.x), std::rint(a.y), std::rint(a.z), std::rint(a.w)); \
} \
 /* round */ \
 \
ATTRIBUTES RET_TYPE round##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::round(a.x), std::round(a.y), std::round(a.z), std::round(a.w)); \
} \
 /* rsqrt -> not supported */ \
 \
 /* sin */ \
 \
ATTRIBUTES RET_TYPE sin##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::sin(a.x), std::sin(a.y), std::sin(a.z), std::sin(a.w)); \
} \
 /* sincos -> not supported */ \
 \
 /* sinh */ \
 \
ATTRIBUTES RET_TYPE sinh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::sinh(a.x), std::sinh(a.y), std::sinh(a.z), std::sinh(a.w)); \
} \
 /* sqrt */ \
 \
ATTRIBUTES RET_TYPE sqrt##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z), std::sqrt(a.w)); \
} \
 /* tan */ \
 \
ATTRIBUTES RET_TYPE tan##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::tan(a.x), std::tan(a.y), std::tan(a.z), std::tan(a.w)); \
} \
 /* tanh */ \
 \
ATTRIBUTES RET_TYPE tanh##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::tanh(a.x), std::tanh(a.y), std::tanh(a.z), std::tanh(a.w)); \
} \
 /* tgamma */ \
 \
ATTRIBUTES RET_TYPE tgamma##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::tgamma(a.x), std::tgamma(a.y), std::tgamma(a.z), std::tgamma(a.w)); \
} \
 /* trunc */ \
 \
ATTRIBUTES RET_TYPE trunc##SUFFIX(NEW_TYPE a) { \
    return make_##RET_TYPE(std::trunc(a.x), std::trunc(a.y), std::trunc(a.z), std::trunc(a.w)); \
} \


MAKE_MATH_BI(float4,    float,  float4, f)
MAKE_MATH_BI(double4,   double, double4, )


// generic math functions
#define MAKE_MATH_BI_GEN(NEW_TYPE, BASIC_TYPE) \
 \
 /* min */ \
 \
ATTRIBUTES BASIC_TYPE min(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a < b ? a : b); \
} \
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
ATTRIBUTES BASIC_TYPE max(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a > b ? a : b); \
} \
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
MAKE_MATH_BI_GEN(long4,     long)
MAKE_MATH_BI_GEN(ulong4,    ulong)
MAKE_MATH_BI_GEN(float4,    float)
MAKE_MATH_BI_GEN(double4,   double)


// integer math operators: abs, labs
#define MAKE_MATH_BI_INT(NEW_TYPE, BASIC_TYPE, RET_TYPE, PREFIX) \
 /* abs */ \
ATTRIBUTES RET_TYPE PREFIX##abs(NEW_TYPE a) { \
    return make_##RET_TYPE(std::abs(a.x), std::abs(a.y), std::abs(a.z), std::abs(a.w)); \
}

MAKE_MATH_BI_INT(int4,  int,    int4,    )
MAKE_MATH_BI_INT(long4, long,   long4,  l)

} // end namespace math
} // end namespace hipacc

#endif // __MATH_FUNCTIONS_HPP__

