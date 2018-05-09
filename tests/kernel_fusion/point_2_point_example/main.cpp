//
// Copyright (c) 2018, University of Erlangen-Nuremberg
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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "hipacc.hpp"

//#define CHECK_RESULT


using namespace hipacc;

// kernel description in Hipacc 
class GlobalOffsetCorrection : public Kernel<int> {
  private:
    Accessor<int> &Input;
    int offset;

  public:
    GlobalOffsetCorrection( IterationSpace<int> &IS, Accessor<int> &Input, int offset ) :
      Kernel(IS),
      Input(Input),
      offset(offset) {
        add_accessor(&Input);
      }

    void kernel() {
      output() = Input() + offset;
    } 
};



/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int offseta = 3;
    const int offsetb = 6;

    // host memory for image of width*height pixels
    int *host_in = new int[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = (y*width + x) % 256;
      }
    }

    // hipacc object decls
    // input, buffer, and output image
    Image<int> In(width, height);
    Image<int> Buf(width, height);
    Image<int> Out(width, height);

    Accessor<int> AccIn(In);
    IterationSpace<int> IsBuf(Buf);
    Accessor<int> AccBuf(Buf);
    IterationSpace<int> IsOut(Out);

    GlobalOffsetCorrection GocKernelA(IsBuf, AccIn, offseta);
    GlobalOffsetCorrection GocKernelB(IsOut, AccBuf, offsetb);

    // kernel invocation
    In = host_in;
    GocKernelA.execute();
    GocKernelB.execute();

    // get pointer to result data
    int *output = Out.data();

#ifdef CHECK_RESULT
    std::cerr << "\nComparing results ...";
    // compute the golden results
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        int golden_out = (y*width + x) % 256 + offseta + offsetb;
        if (output[y*width + x] != golden_out) {
          std::cerr << "\nTest FAILED, result mismatch at x:" << x << " y:" << y << "\n";
          exit(EXIT_FAILURE);
        }
      }
    }
    std::cerr << "\nTest PASSED\n";
#endif

    // free memory
    delete[] host_in;

    return EXIT_SUCCESS;
}

