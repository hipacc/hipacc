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

//===--- OpenCL_GPU.cpp - Implements the OpenCL code generator for GPUs. -------------===//
//
// This file implements the OpenCL code generator for GPUs.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/OpenCL_GPU.h"
#include "hipacc/Device/TargetDescription.h"

using namespace clang::hipacc::Backend;

OpenCL_GPU::CodeGenerator::Descriptor::Descriptor()
{
  SetTargetLang(::clang::hipacc::Language::OpenCLGPU);
  SetName("OpenCL for GPU");
  SetEmissionKey("opencl-gpu");
  SetDescription("Emit OpenCL code for GPU devices");
}

OpenCL_GPU::CodeGenerator::CodeGenerator(::clang::hipacc::CompilerOptions *pCompilerOptions)  : BaseType(pCompilerOptions, Descriptor())
{
  _InitSwitch<AcceleratorDeviceSwitches::EmitPadding    >(CompilerSwitchTypeEnum::EmitPadding);
  _InitSwitch<AcceleratorDeviceSwitches::PixelsPerThread>(CompilerSwitchTypeEnum::PixelsPerThread);
  _InitSwitch<AcceleratorDeviceSwitches::ReduceConfig   >(CompilerSwitchTypeEnum::ReduceConfig);
  _InitSwitch<AcceleratorDeviceSwitches::Target         >(CompilerSwitchTypeEnum::Target);
  _InitSwitch<AcceleratorDeviceSwitches::UseConfig      >(CompilerSwitchTypeEnum::UseConfig);
  _InitSwitch<AcceleratorDeviceSwitches::UseLocal       >(CompilerSwitchTypeEnum::UseLocal);
  _InitSwitch<AcceleratorDeviceSwitches::UseTextures    >(CompilerSwitchTypeEnum::UseTextures);
  _InitSwitch<AcceleratorDeviceSwitches::Vectorize      >(CompilerSwitchTypeEnum::Vectorize);
  _InitSwitch<AcceleratorDeviceSwitches::ClCompilerPath >(CompilerSwitchTypeEnum::ClCompilerPath);
}


size_t OpenCL_GPU::CodeGenerator::_HandleSwitch(CompilerSwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex)
{
  std::string  strCurrentSwitch  = rvecArguments[szCurrentIndex];
  size_t  szReturnIndex     = szCurrentIndex;

  switch (eSwitch)
  {
  case CompilerSwitchTypeEnum::EmitPadding:
    {
      GetCompilerOptions().setPadding(_ParseOption<AcceleratorDeviceSwitches::EmitPadding>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::PixelsPerThread:
    {
      GetCompilerOptions().setPixelsPerThread(_ParseOption<AcceleratorDeviceSwitches::PixelsPerThread>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::ReduceConfig:
    {
      typedef AcceleratorDeviceSwitches::ReduceConfig  SwitchType;

      SwitchType::OptionParser::ReturnType  Value = _ParseOption<SwitchType>(rvecArguments, szCurrentIndex);

      GetCompilerOptions().setReduceConfig(Value.first, Value.second);
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::Target:
    {
      GetCompilerOptions().setTargetDevice(_ParseOption<AcceleratorDeviceSwitches::Target>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::UseConfig:
    {
      typedef AcceleratorDeviceSwitches::UseConfig  SwitchType;

      SwitchType::OptionParser::ReturnType  Value = _ParseOption<SwitchType>(rvecArguments, szCurrentIndex);

      GetCompilerOptions().setKernelConfig(Value.first, Value.second);
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::UseLocal:
    {
      GetCompilerOptions().setLocalMemory(_ParseOption<AcceleratorDeviceSwitches::UseLocal>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::UseTextures:
    {
      GetCompilerOptions().setTextureMemory(_ParseOption<AcceleratorDeviceSwitches::UseTextures>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::Vectorize:
    {
      GetCompilerOptions().setVectorizeKernels(_ParseOption<AcceleratorDeviceSwitches::Vectorize>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::ClCompilerPath:
    {
      GetCompilerOptions().setClCompilerPath(_ParseOption<AcceleratorDeviceSwitches::ClCompilerPath>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strCurrentSwitch, GetName());
  }

  return szReturnIndex;
}

void OpenCL_GPU::CodeGenerator::_CheckConfiguration()
{
  // Check base configuration
  BaseType::_CheckConfiguration();


  HipaccDevice ConfiguredTargetDevive(GetCompilerOptions());

  // Check target device
  if ( ! ( ConfiguredTargetDevive.isAMDGPU() || ConfiguredTargetDevive.isARMGPU() || ConfiguredTargetDevive.isNVIDIAGPU() ) )
  {
    throw RuntimeErrorException("OpenCL (GPU) code generation selected, but no OpenCL-capable GPU target device specified!\n  Please select correct target device/code generation back end combination.");
  }


  // Check texture support
  if (GetCompilerOptions().useTextureMemory(USER_ON))
  {
    // Only Array2D textures supported in OpenCL
    if (GetCompilerOptions().getTextureType() != Texture::Array2D)
    {
      llvm::errs() << "Warning: 'Linear1D', 'Linear2D', and 'Ldg' texture memory not supported by OpenCL!  Using 'Array2D' instead!\n";

      GetCompilerOptions().setTextureMemory(Texture::Array2D);
    }
  }
}


// vim: set ts=2 sw=2 sts=2 et ai:

