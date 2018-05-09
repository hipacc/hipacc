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
class AverageFilter : public Kernel<int> {
  private:
    Accessor<int> &Input;
    Domain &dom;
    Mask<int> &mask;

  public:
    AverageFilter( IterationSpace<int> &IS, Accessor<int> &Input, Domain &dom, Mask<int> &mask ) :
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

#ifdef PRINT_RESULT 
    const int width = 256;
    const int height = 256;
#else
    const int width = WIDTH;
    const int height = HEIGHT;
#endif

    // host memory for image of width*height pixels
    int *host_in = new int[width*height];

    // filter masks 
    const int filter_mask3[3][3] = {
      { 1, 2, 1 },
      { 2, 4, 2 },
      { 1, 2, 1 }
    };
    const int filter_mask5[5][5] = {
      { 1, 2, 3, 2, 1 },
      { 2, 3, 4, 4, 2 },
      { 2, 3, 4, 4, 2 },
      { 2, 3, 4, 4, 2 },
      { 1, 3, 4, 2, 1 }
    };

    Mask<int> mask1(filter_mask3);
    Domain DOM1(mask1);

    Mask<int> mask2(filter_mask5);
    Domain DOM2(mask2);

    // initialize data
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = (y*width + x) % 199;
      }
    }

    // hipacc object decls
    // input, buffer, and output image
    Image<int> In(width, height);
    Image<int> Buf(width, height);
    Image<int> Out(width, height);

    const BoundaryCondition<int> BCBufClamp1(In, mask1, Boundary::CLAMP);
    Accessor<int> AccIn(BCBufClamp1);
    IterationSpace<int> IsBuf(Buf);

    const BoundaryCondition<int> BCBufClamp2(Buf, mask2, Boundary::CLAMP);
    Accessor<int> AccBuf(BCBufClamp2);
    IterationSpace<int> IsOut(Out);

    AverageFilter AfKernelSize3Clamp(IsBuf, AccIn, DOM1, mask1);
    AverageFilter AfKernelSize5Clamp(IsOut, AccBuf, DOM2, mask2);

    // kernel invocation
    In = host_in;
    AfKernelSize3Clamp.execute();
    AfKernelSize5Clamp.execute();

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

