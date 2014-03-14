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

//===--- OpenCL_CPU.cpp - Implements the OpenCL code generator for CPUs. -------------===//
//
// This file implements the OpenCL code generator for CPUs.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/OpenCL_CPU.h"

using namespace clang::hipacc::Backend;
using namespace clang::hipacc;
using namespace std;

OpenCL_CPU::CodeGenerator::Descriptor::Descriptor()
{
  SetTargetCode(::clang::hipacc::TARGET_OpenCLCPU);
  SetName("OpenCL for CPU");
  SetEmissionKey("opencl-cpu");
  SetDescription("Emit OpenCL code for CPU devices");
}

OpenCL_CPU::CodeGenerator::CodeGenerator(::clang::hipacc::CompilerOptions *pCompilerOptions)  : BaseType(pCompilerOptions, Descriptor())
{
  _InitSwitch< AcceleratorDeviceSwitches::ExploreConfig >(CompilerSwitchTypeEnum::ExploreConfig);
  _InitSwitch< AcceleratorDeviceSwitches::UseLocal      >(CompilerSwitchTypeEnum::UseLocal);
  _InitSwitch< AcceleratorDeviceSwitches::Vectorize     >(CompilerSwitchTypeEnum::Vectorize);
}


size_t OpenCL_CPU::CodeGenerator::_HandleSwitch(CompilerSwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex)
{
  string  strCurrentSwitch  = rvecArguments[szCurrentIndex];
  size_t  szReturnIndex     = szCurrentIndex;

  switch (eSwitch)
  {
  case CompilerSwitchTypeEnum::ExploreConfig:
    GetCompilerOptions().setExploreConfig(USER_ON);
    break;
  case CompilerSwitchTypeEnum::UseLocal:
    {
      GetCompilerOptions().setLocalMemory(_ParseOption< AcceleratorDeviceSwitches::UseLocal >(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::Vectorize:
    {
      GetCompilerOptions().setVectorizeKernels(_ParseOption< AcceleratorDeviceSwitches::Vectorize >(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strCurrentSwitch, GetName());
  }

  return szReturnIndex;
}


// vim: set ts=2 sw=2 sts=2 et ai:

