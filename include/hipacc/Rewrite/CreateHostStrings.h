//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// Copyright (c) 2010, ARM Limited
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

//===--- CreateHostStrings.h - Runtime string creator for the Rewriter ----===//
//
// This file implements functionality for printing Hipacc runtime code to
// strings.
//
//===----------------------------------------------------------------------===//

#ifndef _CREATE_HOST_STRINGS_H_
#define _CREATE_HOST_STRINGS_H_

#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/AST/ASTFuse.h"

#include <string>

namespace clang {
namespace hipacc {
class CreateHostStrings {
  private:
    CompilerOptions &options;
    HipaccDevice &device;
    unsigned literal_count;
    int num_indent, cur_indent;
    std::string indent;

    // fusion params
    std::map<HipaccKernel *, std::string> fusedKernelLaunchInfoMap;
    std::map<HipaccKernel *, std::string> fusedKernelCallMap;
    std::map<HipaccKernel *, std::string> fusedKernelPrepareLaunchMap;

    void inc_indent() {
      cur_indent += num_indent;
      indent = std::string(cur_indent, ' ');
    }
    void dec_indent() {
      cur_indent -= num_indent;
      if (cur_indent < 0) cur_indent = 0;
      indent = std::string(cur_indent, ' ');
    }

  public:
    CreateHostStrings(CompilerOptions &options, HipaccDevice &device) :
      options(options),
      device(device),
      literal_count(0),
      num_indent(4),
      cur_indent(num_indent),
      indent(cur_indent, ' ')
    {}

    std::string getIndent() { return indent; }
    void writeHeaders(std::string &resultStr);
    void writeInitialization(std::string &resultStr);
    void writeKernelCompilation(HipaccKernel *K, std::string &resultStr);
    void writeReductionDeclaration(HipaccKernel *K, std::string &resultStr);
    void writeMemoryAllocation(HipaccImage *Img, std::string const& width, std::string const&
        height, std::string const& host, std::string const& deep_copy, std::string &resultStr);
    void writeMemoryAllocationConstant(HipaccMask *Buf, std::string &resultStr);
    void writeMemoryMapping(HipaccImage *Img, std::string const& argument_name, std::string &resultStr);
    void writeMemoryTransfer(HipaccImage *Img, std::string mem,
        MemoryTransferDirection direction, std::string &resultStr);
    void addMemoryTransferGraph(HipaccImage *Img, std::string mem, MemoryTransferDirection direction,
        std::string &graphStr, std::string &nodeStr, std::string &nodeDepStr, std::string &nodeArgStr,
        std::string &resultStr);
    void writeMemoryTransfer(HipaccPyramid *Pyr, std::string idx,
        std::string mem, MemoryTransferDirection direction,
        std::string &resultStr);
    void writeMemoryTransferRegion(std::string dst, std::string src, std::string
        &resultStr);
    void writeMemoryTransferSymbol(HipaccMask *Mask, std::string mem,
        MemoryTransferDirection direction, std::string &resultStr);
    void writeMemoryTransferDomainFromMask(HipaccMask *Domain,
        HipaccMask *Mask, std::string &resultStr);
    void writeKernelCall(HipaccKernel *K, std::string &resultStr);
    void writeFusedKernelCall(HipaccKernel *K, std::string &resultStr, ASTFuse *kernelFuser);
    void writeReduceCall(HipaccKernel *K, std::string &resultStr);
    void writeBinningCall(HipaccKernel *K, std::string &resultStr);
    std::string getInterpolationDefinition(HipaccKernel *K, HipaccAccessor *Acc,
        std::string function_name, std::string type_suffix, Interpolate ip_mode,
        Boundary bh_mode);
    void writePyramidAllocation(std::string pyrName, std::string type,
        std::string img, std::string depth, std::string &resultStr);
    void writePyramidMapping(std::string pyrName, std::string type,
        std::string assigned_pyramid, std::string &resultStr);
};
} // namespace hipacc
} // namespace clang

#endif  // _CREATE_HOST_STRINGS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

