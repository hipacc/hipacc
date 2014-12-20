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

//===--- TargetDevices.h - List of target hardware devices ---------------====//
//
// This provides a list of supported target hardware devices.
//
//===----------------------------------------------------------------------===//

#ifndef _TARGET_DEVICES_H
#define _TARGET_DEVICES_H

#include <cstdint>

namespace clang {
namespace hipacc {
// supported target devices
enum class Device {
  Tesla_10          = 10,
  Tesla_11          = 11,
  Tesla_12          = 12,
  Tesla_13          = 13,
  Fermi_20          = 20,
  Fermi_21          = 21,
  Kepler_30         = 30,
  Kepler_35         = 35,
  Evergreen         = 58,
  NorthernIsland    = 69,
  //SouthernIsland    = 79
  Midgard           = 600,
  KnightsCorner     = 7120
};

// texture memory specification
enum class Texture : uint8_t {
  None,
  Linear1D,
  Linear2D,
  Array2D,
  Ldg
};
} // end namespace hipacc
} // end namespace clang

#endif  // _TARGET_DEVICES_H

// vim: set ts=2 sw=2 sts=2 et ai:

