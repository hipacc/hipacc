#include <cuda_runtime.h>

typedef unsigned char   uchar;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned long   ulong;

// avoid clash with CUDA vector types
#define __HIPACC_MATH_FUNCTIONS_HPP__

// do not include base implementations
#define __HIPACC_BASE_STANDALONE_HPP__

// Hipacc runtime definitions
#include <hipacc_cu_standalone.hpp>

