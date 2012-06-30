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

// undef macros defined for reductions 
#undef REDUCTION_CUDA_2D_THREAD_FENCE
#undef REDUCTION_CUDA_2D
#undef REDUCTION_CUDA_1D
#undef REDUCTION_OCL_2D
#undef REDUCTION_OCL_1D
#undef OFFSETS
#undef IS_HEIGHT
#undef OFFSET_Y
#undef OFFSET_CHECK_X
#undef OFFSET_CHECK_X_STRIDE
#undef USE_OFFSETS
// undef macros defined for interpolation 
#undef IMG_PARM
#undef TEX_PARM
#undef CONST_PARM
#undef NO_PARM
#undef IMG
#undef TEX
#undef IMG_CONST
#undef TEX_CONST
#undef BH_CLAMP_LOWER
#undef BH_CLAMP_UPPER
#undef BH_REPEAT_LOWER
#undef BH_REPEAT_UPPER
#undef BH_MIRROR_LOWER
#undef BH_MIRROR_UPPER
#undef BH_CONSTANT_LOWER
#undef BH_CONSTANT_UPPER
#undef NO_BH
#undef DEFINE_BH_VARIANTS
#undef DEFINE_BH_VARIANT
#undef INTERPOLATE_LINEAR_FILTERING_CUDA
#undef INTERPOLATE_CUBIC_FILTERING_CUDA
#undef INTERPOLATE_LANCZOS_FILTERING_CUDA

