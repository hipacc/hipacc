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

//===--- CUDA.cpp - Implements the NVidia CUDA code generator. -----------------------===//
//
// This file implements the NVidia CUDA code generator.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/CUDA.h"
#include "hipacc/Device/TargetDescription.h"

using namespace clang::hipacc::Backend;

CUDA::CodeGenerator::Descriptor::Descriptor()
{
  SetTargetLang(::clang::hipacc::Language::CUDA);
  SetName("CUDA");
  SetEmissionKey("cuda");
  SetDescription("Emit CUDA code for GPU devices");
}

CUDA::CodeGenerator::CodeGenerator(::clang::hipacc::CompilerOptions *pCompilerOptions)  : BaseType(pCompilerOptions, Descriptor())
{
  _InitSwitch<AcceleratorDeviceSwitches::EmitPadding    >(CompilerSwitchTypeEnum::EmitPadding);
  _InitSwitch<AcceleratorDeviceSwitches::PixelsPerThread>(CompilerSwitchTypeEnum::PixelsPerThread);
  _InitSwitch<AcceleratorDeviceSwitches::ReduceConfig   >(CompilerSwitchTypeEnum::ReduceConfig);
  _InitSwitch<AcceleratorDeviceSwitches::Target         >(CompilerSwitchTypeEnum::Target);
  _InitSwitch<AcceleratorDeviceSwitches::UseConfig      >(CompilerSwitchTypeEnum::UseConfig);
  _InitSwitch<AcceleratorDeviceSwitches::UseLocal       >(CompilerSwitchTypeEnum::UseLocal);
  _InitSwitch<AcceleratorDeviceSwitches::UseTextures    >(CompilerSwitchTypeEnum::UseTextures);
  _InitSwitch<AcceleratorDeviceSwitches::Vectorize      >(CompilerSwitchTypeEnum::Vectorize);
  _InitSwitch<AcceleratorDeviceSwitches::NvccPath       >(CompilerSwitchTypeEnum::NvccPath);
  _InitSwitch<AcceleratorDeviceSwitches::CCBinPath      >(CompilerSwitchTypeEnum::CCBinPath);
  _InitSwitch<AcceleratorDeviceSwitches::FuseKernel     >(CompilerSwitchTypeEnum::FuseKernel);
  _InitSwitch<AcceleratorDeviceSwitches::UseGraph       >(CompilerSwitchTypeEnum::UseGraph);
}


size_t CUDA::CodeGenerator::_HandleSwitch(CompilerSwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex)
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
  case CompilerSwitchTypeEnum::NvccPath:
    {
      GetCompilerOptions().setNvccPath(_ParseOption<AcceleratorDeviceSwitches::NvccPath>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::CCBinPath:
    {
      GetCompilerOptions().setCCBinPath(_ParseOption<AcceleratorDeviceSwitches::CCBinPath>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::FuseKernel:
    {
      GetCompilerOptions().setFuseKernels(_ParseOption<AcceleratorDeviceSwitches::FuseKernel>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::UseGraph:
    {
      GetCompilerOptions().setUseGraph(_ParseOption<AcceleratorDeviceSwitches::UseGraph>(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strCurrentSwitch, GetName());
  }

  return szReturnIndex;
}

void CUDA::CodeGenerator::_CheckConfiguration()
{
  // Check base configuration
  BaseType::_CheckConfiguration();


  HipaccDevice ConfiguredTargetDevive(GetCompilerOptions());

  // Check target device
  if (! ConfiguredTargetDevive.isNVIDIAGPU())
  {
    throw RuntimeErrorException("CUDA code generation selected, but no CUDA - capable target device specified!\n  Please select correct target device/code generation back end combination.");
  }


  // Check texture support
  if (GetCompilerOptions().useTextureMemory(USER_ON))
  {
    // Writing to Array2D textures has been introduced with Fermi architecture
    if ( (GetCompilerOptions().getTextureType() == Texture::Array2D) && (GetCompilerOptions().getTargetDevice() < Device::Fermi_20) )
    {
      llvm::errs() << "Warning: 'Array2D' texture memory only supported for Fermi and later on (CC >= 2.0)!  Using 'Linear2D' instead!\n";

      GetCompilerOptions().setTextureMemory(Texture::Linear2D);
    }

    // Ldg (load via texture cache) was introduced with Kepler architecture
    if ( (GetCompilerOptions().getTextureType() == Texture::Ldg) && (GetCompilerOptions().getTargetDevice() < Device::Kepler_35) )
    {
      llvm::errs() << "Warning: 'Ldg' texture memory only supported for Kepler and later on (CC >= 3.5)!  Using 'Linear1D' instead!\n";

      GetCompilerOptions().setTextureMemory(Texture::Linear1D);
    }
  }
}


// vim: set ts=2 sw=2 sts=2 et ai:

