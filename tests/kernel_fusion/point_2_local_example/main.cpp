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

//#define PRINT_RESULT


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

class SumFilter : public Kernel<int> {
  private:
    Accessor<int> &Input;
    Domain &dom;
    Mask<int> &mask;

  public:
    SumFilter( IterationSpace<int> &IS, Accessor<int> &Input, Domain &dom, Mask<int> &mask ) :
      Kernel(IS),
      Input(Input),
      dom(dom),
      mask(mask)
      {
        add_accessor(&Input);
      }

    void kernel() {
      int sum = 0;
      iterate(dom, [&] () {
        sum += mask(dom) * Input(dom);
      });

      output() = sum;
    } 
};



/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int offset = 3;

    // filter mask for Gaussian blur filter
    const int filter_mask[3][3] = {
      { 1, 2, 1 },
      { 2, 4, 2 },
      { 1, 2, 1 }
    };
    Mask<int> mask(filter_mask);
    Domain DOM(mask);

    // host memory for image of width*height pixels
    int *host_in = new int[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = (y*width + x) % 17;
      }
    }

    // hipacc object decls
    // input, buffer, and output image
    Image<int> In(width, height);
    Image<int> Buf(width, height);
    Image<int> Out(width, height);

    Accessor<int> AccIn(In);
    IterationSpace<int> IsBuf(Buf);

    const BoundaryCondition<int> BCBufMirror(Buf, mask, Boundary::MIRROR);
    Accessor<int> AccBuf(BCBufMirror);
    IterationSpace<int> IsOut(Out);

    GlobalOffsetCorrection GocKernel(IsBuf, AccIn, offset);
    SumFilter SfKernel(IsOut, AccBuf, DOM, mask);

    // kernel invocation
    In = host_in;
    GocKernel.execute();
    SfKernel.execute();

    // get pointer to result data
    int *output = Out.data();

#ifdef PRINT_RESULT 
    std::cerr << "\nwriting output to file...\n";
    // compute the golden results
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        std::cout << "output at y " << y << ", x " << x << " : " << output[y*width + x] << "\n"; 
      }
    }
#endif

    // free memory
    delete[] host_in;

    return EXIT_SUCCESS;
}

