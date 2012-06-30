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

#ifndef __HIPACC_CUDA_VEC_HPP__
#define __HIPACC_CUDA_VEC_HPP__


typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;


// additional constructors
#define MAKE_VEC4(DATA_TYPE) \
inline __host__ __device__ DATA_TYPE##4 make_##DATA_TYPE##4(DATA_TYPE s) \
{ \
    return make_##DATA_TYPE##4(s, s, s, s); \
}

MAKE_VEC4(char)
MAKE_VEC4(uchar)
MAKE_VEC4(short)
MAKE_VEC4(ushort)
MAKE_VEC4(int)
MAKE_VEC4(uint)
MAKE_VEC4(long)
MAKE_VEC4(ulong)
MAKE_VEC4(float)
MAKE_VEC4(double)

#if 0
template < typename T1, typename T2 >
inline __host__ __device__ T1 &operator=(T2 const &b) {
    a.x = (T1) b.x;
    a.y = (T1) b.y;
    a.z = (T1) b.z;
    a.w = (T1) b.w;

    return *this;
}
inline __host__ __device__ int4 &operator=(uchar4 const &b) {
    a.x = (uchar4) b.x;
    a.y = (uchar4) b.y;
    a.z = (uchar4) b.z;
    a.w = (uchar4) b.w;

    return *this;
}
#endif

// addition
#define ADD_VEC4(DATA_TYPE) \
inline __host__ __device__ DATA_TYPE##4 operator+(DATA_TYPE##4 a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w); \
} \
inline __host__ __device__ DATA_TYPE##4 operator+(DATA_TYPE##4 a, DATA_TYPE b) { \
    return make_##DATA_TYPE##4(a.x + b, a.y + b, a.z + b,  a.w + b); \
} \
inline __host__ __device__ DATA_TYPE##4 operator+(DATA_TYPE a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a + b.x, a + b.y, a + b.z,  a + b.w); \
} \
inline __host__ __device__ void operator+=(DATA_TYPE##4 &a, DATA_TYPE##4 b) { \
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; \
} \
inline __host__ __device__ void operator+=(DATA_TYPE##4 a, DATA_TYPE b) { \
    a.x += b, a.y += b, a.z += b,  a.w += b; \
}

ADD_VEC4(char)
ADD_VEC4(uchar)
ADD_VEC4(short)
ADD_VEC4(ushort)
ADD_VEC4(int)
ADD_VEC4(uint)
ADD_VEC4(long)
ADD_VEC4(ulong)
ADD_VEC4(float)
ADD_VEC4(double)


// subtract
#define SUB_VEC4(DATA_TYPE) \
inline __host__ __device__ DATA_TYPE##4 operator-(DATA_TYPE##4 a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w); \
} \
inline __host__ __device__ DATA_TYPE##4 operator-(DATA_TYPE##4 a, DATA_TYPE b) { \
    return make_##DATA_TYPE##4(a.x - b, a.y - b, a.z - b,  a.w - b); \
} \
inline __host__ __device__ DATA_TYPE##4 operator-(DATA_TYPE a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a - b.x, a - b.y, a - b.z,  a - b.w); \
} \
inline __host__ __device__ void operator-=(DATA_TYPE##4 &a, DATA_TYPE##4 b) { \
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; \
} \
inline __host__ __device__ void operator-=(DATA_TYPE##4 a, DATA_TYPE b) { \
    a.x -= b, a.y -= b, a.z -= b,  a.w -= b; \
}

SUB_VEC4(char)
SUB_VEC4(uchar)
SUB_VEC4(short)
SUB_VEC4(ushort)
SUB_VEC4(int)
SUB_VEC4(uint)
SUB_VEC4(long)
SUB_VEC4(ulong)
SUB_VEC4(float)
SUB_VEC4(double)


// multiply
#define MUL_VEC4(DATA_TYPE) \
inline __host__ __device__ DATA_TYPE##4 operator*(DATA_TYPE##4 a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w); \
} \
inline __host__ __device__ DATA_TYPE##4 operator*(DATA_TYPE##4 a, DATA_TYPE b) { \
    return make_##DATA_TYPE##4(a.x * b, a.y * b, a.z * b,  a.w * b); \
} \
inline __host__ __device__ DATA_TYPE##4 operator*(DATA_TYPE a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a * b.x, a * b.y, a * b.z,  a * b.w); \
} \
inline __host__ __device__ void operator*=(DATA_TYPE##4 &a, DATA_TYPE##4 b) { \
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; \
} \
inline __host__ __device__ void operator*=(DATA_TYPE##4 a, DATA_TYPE b) { \
    a.x *= b, a.y *= b, a.z *= b,  a.w *= b; \
}

MUL_VEC4(char)
MUL_VEC4(uchar)
MUL_VEC4(short)
MUL_VEC4(ushort)
MUL_VEC4(int)
MUL_VEC4(uint)
MUL_VEC4(long)
MUL_VEC4(ulong)
MUL_VEC4(float)
MUL_VEC4(double)


// divide
#define DIV_VEC4(DATA_TYPE) \
inline __host__ __device__ DATA_TYPE##4 operator/(DATA_TYPE##4 a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w); \
} \
inline __host__ __device__ DATA_TYPE##4 operator/(DATA_TYPE##4 a, DATA_TYPE b) { \
    return make_##DATA_TYPE##4(a.x / b, a.y / b, a.z / b,  a.w / b); \
} \
inline __host__ __device__ DATA_TYPE##4 operator/(DATA_TYPE a, DATA_TYPE##4 b) { \
    return make_##DATA_TYPE##4(a / b.x, a / b.y, a / b.z,  a / b.w); \
} \
inline __host__ __device__ void operator/=(DATA_TYPE##4 &a, DATA_TYPE##4 b) { \
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; \
} \
inline __host__ __device__ void operator/=(DATA_TYPE##4 a, DATA_TYPE b) { \
    a.x /= b, a.y /= b, a.z /= b,  a.w /= b; \
}

DIV_VEC4(char)
DIV_VEC4(uchar)
DIV_VEC4(short)
DIV_VEC4(ushort)
DIV_VEC4(int)
DIV_VEC4(uint)
DIV_VEC4(long)
DIV_VEC4(ulong)
DIV_VEC4(float)
DIV_VEC4(double)


// vector commands for mathematic functions
inline __host__ __device__ float4 expf(float4 a) {
    return make_float4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
}


#endif  // __HIPACC_CUDA_VEC_HPP__

