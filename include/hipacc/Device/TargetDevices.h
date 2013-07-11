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

namespace clang {
namespace hipacc {
// supported target devices
enum TargetDevice {
  TESLA_10          = 10,
  TESLA_11          = 11,
  TESLA_12          = 12,
  TESLA_13          = 13,
  FERMI_20          = 20,
  FERMI_21          = 21,
  KEPLER_30         = 30,
  KEPLER_35         = 35,
  EVERGREEN         = 58,
  NORTHERN_ISLAND   = 69,
  //SOUTHERN_ISLAND   = 79
  MIDGARD           = 600
};

// texture memory specification
enum TextureType {
  NoTexture         = 0x0,
  Linear1D          = 0x1,
  Linear2D          = 0x2,
  Array2D           = 0x4,
  Ldg               = 0x8
};
} // end namespace hipacc
} // end namespace clang

#endif  // _TARGET_DEVICES_H

// vim: set ts=2 sw=2 sts=2 et ai:

