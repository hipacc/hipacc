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

//===--- Filterscript.cpp - Implements the Filterscript code generator. --------------===//
//
// This file implements the Filterscript code generator.
//
//===---------------------------------------------------------------------------------===//

#include "hipacc/Backend/Filterscript.h"

using namespace clang::hipacc::Backend;
using namespace std;

Filterscript::CodeGenerator::Descriptor::Descriptor()
{
  SetTargetCode(::clang::hipacc::TARGET_Filterscript);
  SetName("Filterscript");
  SetEmissionKey("filterscript");
  SetDescription("Emit Filterscript code for Android");
}

Filterscript::CodeGenerator::CodeGenerator(::clang::hipacc::CompilerOptions *pCompilerOptions)  : BaseType(pCompilerOptions, Descriptor())
{
  _InitSwitch< AndroidSwitches::RsPackage >(CompilerSwitchTypeEnum::RsPackage);
}


size_t Filterscript::CodeGenerator::_HandleSwitch(CompilerSwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex)
{
  string  strCurrentSwitch  = rvecArguments[szCurrentIndex];
  size_t  szReturnIndex     = szCurrentIndex;

  switch (eSwitch)
  {
  case CompilerSwitchTypeEnum::RsPackage:
    {
      GetCompilerOptions().setRSPackageName(_ParseOption< AndroidSwitches::RsPackage >(rvecArguments, szCurrentIndex));
      ++szReturnIndex;
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strCurrentSwitch, GetName());
  }

  return szReturnIndex;
}

void Filterscript::CodeGenerator::_CheckConfiguration()
{
  // Check base configuration
  BaseType::_CheckConfiguration();

  // Filterscript supports only one pixel thread
  GetCompilerOptions().setPixelsPerThread(1);
}

// vim: set ts=2 sw=2 sts=2 et ai:

