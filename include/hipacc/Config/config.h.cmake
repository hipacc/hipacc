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

//===--- config.h - HIPACC Configuration ----------------------------------===//
//
// This file configures HIPACC.
//
//===----------------------------------------------------------------------===//

#ifndef _CONFIG_H_
#define _CONFIG_H_

#cmakedefine CUDA_FOUND
#cmakedefine OpenCL_FOUND
#cmakedefine USE_JIT_ESTIMATE

#define CU_COMPILER "${CU_COMPILER}"
#define CL_COMPILER "${CL_COMPILER}"
#define RUNTIME_INCLUDES "${RUNTIME_INCLUDES}"

#define CLANG_LIB_INCLUDE_DIR "${CLANG_LIB_INCLUDE_DIR}"

#ifdef _MSC_VER
#define HOST_COMPILER_INSTALL_PREFIX "${HOST_COMPILER_INSTALL_PREFIX}"
#endif // _MSC_VER

#define HIPACC_VERSION "${HIPACC_VERSION}"
#define GIT_REPOSITORY "${HIPACC_GIT_REPOSITORY}"
#define GIT_VERSION "${HIPACC_GIT_VERSION}"

#endif  // _CONFIG_H_

