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

#ifndef __TYPES_HPP__
#define __TYPES_HPP__

namespace hipacc {

#if defined __clang__
typedef char                char4   __attribute__ ((ext_vector_type(4)));
typedef short int           short4  __attribute__ ((ext_vector_type(4)));
typedef int                 int4    __attribute__ ((ext_vector_type(4)));
typedef long int            long4   __attribute__ ((ext_vector_type(4)));
typedef unsigned char       uchar4  __attribute__ ((ext_vector_type(4)));
typedef unsigned short int  ushort4 __attribute__ ((ext_vector_type(4)));
typedef unsigned int        uint4   __attribute__ ((ext_vector_type(4)));
typedef unsigned long int   ulong4  __attribute__ ((ext_vector_type(4)));
typedef float               float4  __attribute__ ((ext_vector_type(4)));
typedef double              double4 __attribute__ ((ext_vector_type(4)));
#define ATTRIBUTES inline
#define MAKE_VEC(NEW_TYPE, BASIC_TYPE) \
    MAKE_COPS(NEW_TYPE, BASIC_TYPE)
#elif defined __GNUC__
typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;
typedef unsigned long       ulong;
#define ATTRIBUTES inline
#define MAKE_VEC(NEW_TYPE, BASIC_TYPE) \
    MAKE_TYPE(NEW_TYPE, BASIC_TYPE) \
    MAKE_COPS(NEW_TYPE, BASIC_TYPE) \
    MAKE_VOPS(NEW_TYPE, BASIC_TYPE)
#else
#error "Only Clang, and gcc compilers supported!"
#endif


// vector type definition
#define MAKE_TYPE(NEW_TYPE, BASIC_TYPE) \
_Pragma("pack(1)") \
struct NEW_TYPE { \
    BASIC_TYPE x, y, z, w; \
    void operator=(BASIC_TYPE b) { \
        x = b; y = b; z = b; w = b; \
    } \
}; \
typedef struct NEW_TYPE NEW_TYPE;


// custom operators
#define MAKE_COPS(NEW_TYPE, BASIC_TYPE) \
static ATTRIBUTES NEW_TYPE make_##NEW_TYPE(BASIC_TYPE x, BASIC_TYPE y, BASIC_TYPE z, BASIC_TYPE w) { \
    NEW_TYPE t; t.x = x; t.y = y; t.z = z; t.w = w; return t; \
} \
static ATTRIBUTES NEW_TYPE make_##NEW_TYPE(BASIC_TYPE s) \
{ \
    return make_##NEW_TYPE(s, s, s, s); \
} \
 \
 /* min */ \
 \
ATTRIBUTES BASIC_TYPE min(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a < b ? a : b); \
} \
 \
ATTRIBUTES NEW_TYPE min(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z),  min(a.w, b.w)); \
} \
 \
ATTRIBUTES NEW_TYPE min(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(min(a.x, b), min(a.y, b), min(a.z, b),  min(a.w, b)); \
} \
 \
ATTRIBUTES NEW_TYPE min(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(min(a, b.x), min(a, b.y), min(a, b.z),  min(a, b.w)); \
} \
 \
 /* max */ \
 \
ATTRIBUTES BASIC_TYPE max(BASIC_TYPE a, BASIC_TYPE b) { \
    return (a > b ? a : b); \
} \
 \
ATTRIBUTES NEW_TYPE max(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z),  max(a.w, b.w)); \
} \
 \
ATTRIBUTES NEW_TYPE max(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(max(a.x, b), max(a.y, b), max(a.z, b),  max(a.w, b)); \
} \
 \
ATTRIBUTES NEW_TYPE max(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(max(a, b.x), max(a, b.y), max(a, b.z),  max(a, b.w)); \
}


// vector operators
#define MAKE_VOPS(NEW_TYPE, BASIC_TYPE) \
 \
 /* addition */ \
 \
ATTRIBUTES NEW_TYPE operator+(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w); \
} \
ATTRIBUTES NEW_TYPE operator+(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(a.x + b, a.y + b, a.z + b,  a.w + b); \
} \
ATTRIBUTES NEW_TYPE operator+(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a + b.x, a + b.y, a + b.z,  a + b.w); \
} \
ATTRIBUTES void operator+=(NEW_TYPE &a, NEW_TYPE b) { \
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; \
} \
ATTRIBUTES void operator+=(NEW_TYPE a, BASIC_TYPE b) { \
    a.x += b, a.y += b, a.z += b,  a.w += b; \
} \
 \
 /* subtract */ \
 \
ATTRIBUTES NEW_TYPE operator-(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w); \
} \
ATTRIBUTES NEW_TYPE operator-(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(a.x - b, a.y - b, a.z - b,  a.w - b); \
} \
ATTRIBUTES NEW_TYPE operator-(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a - b.x, a - b.y, a - b.z,  a - b.w); \
} \
ATTRIBUTES void operator-=(NEW_TYPE &a, NEW_TYPE b) { \
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; \
} \
ATTRIBUTES void operator-=(NEW_TYPE a, BASIC_TYPE b) { \
    a.x -= b, a.y -= b, a.z -= b,  a.w -= b; \
} \
 \
 /* multiply */ \
 \
ATTRIBUTES NEW_TYPE operator*(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w); \
} \
ATTRIBUTES NEW_TYPE operator*(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(a.x * b, a.y * b, a.z * b,  a.w * b); \
} \
ATTRIBUTES NEW_TYPE operator*(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a * b.x, a * b.y, a * b.z,  a * b.w); \
} \
ATTRIBUTES void operator*=(NEW_TYPE &a, NEW_TYPE b) { \
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; \
} \
ATTRIBUTES void operator*=(NEW_TYPE a, BASIC_TYPE b) { \
    a.x *= b, a.y *= b, a.z *= b,  a.w *= b; \
} \
 \
 /* divide */ \
 \
ATTRIBUTES NEW_TYPE operator/(NEW_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w); \
} \
ATTRIBUTES NEW_TYPE operator/(NEW_TYPE a, BASIC_TYPE b) { \
    return make_##NEW_TYPE(a.x / b, a.y / b, a.z / b,  a.w / b); \
} \
ATTRIBUTES NEW_TYPE operator/(BASIC_TYPE a, NEW_TYPE b) { \
    return make_##NEW_TYPE(a / b.x, a / b.y, a / b.z,  a / b.w); \
} \
ATTRIBUTES void operator/=(NEW_TYPE &a, NEW_TYPE b) { \
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; \
} \
ATTRIBUTES void operator/=(NEW_TYPE a, BASIC_TYPE b) { \
    a.x /= b, a.y /= b, a.z /= b,  a.w /= b; \
}

MAKE_VEC(char4,     char)
MAKE_VEC(uchar4,    unsigned char)
MAKE_VEC(short4,    short)
MAKE_VEC(ushort4,   unsigned short)
MAKE_VEC(int4,      int)
MAKE_VEC(uint4,     unsigned int)
MAKE_VEC(long4,     long)
MAKE_VEC(ulong4,    unsigned long)
MAKE_VEC(float4,    float)
MAKE_VEC(double4,   double)

}

#endif // __TYPES_HPP__

