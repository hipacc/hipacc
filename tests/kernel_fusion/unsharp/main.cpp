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

#include <cstdlib>
#include <iostream>
#include <cmath>

#include "hipacc.hpp"


//#define PRINT_RESULT
#define data_t int

using namespace hipacc;

// kernel description in Hipacc
class Gauss : public Kernel<data_t> {
  private:
    Accessor<data_t> &Input;
    Mask<data_t> &cMask;

  public:
    Gauss(IterationSpace<data_t> &IS,
            Accessor<data_t> &Input, Mask<data_t> &cMask)
          : Kernel(IS),
            Input(Input),
            cMask(cMask) {
      add_accessor(&Input);
    }

    void kernel() {
      data_t sum = 0;
      sum += convolve(cMask, Reduce::SUM, [&] () -> float {
          return Input(cMask) * cMask();
      });
      output() = sum / 16;
    }
};


class Sharpen : public Kernel<data_t> {
  private:
    Accessor<data_t> &Input;
    Accessor<data_t> &InputBlur;

  public:
    Sharpen(IterationSpace<data_t> &IS, Accessor<data_t> &Input, Accessor<data_t> &InputBlur)
          : Kernel(IS), Input(Input), InputBlur(InputBlur) {
      add_accessor(&Input);
      add_accessor(&InputBlur);
    }

    void kernel() {
      output() = 2 * Input() - InputBlur();
    }
};


class Ratio : public Kernel<data_t> {
  private:
    Accessor<data_t> &Input;
    Accessor<data_t> &InputSharp;

  public:
    Ratio(IterationSpace<data_t> &IS, Accessor<data_t> &Input, Accessor<data_t> &InputSharp)
          : Kernel(IS), Input(Input), InputSharp(InputSharp) {
      add_accessor(&Input);
      add_accessor(&InputSharp);
    }

    void kernel() {
      output() = InputSharp() / Input();
    }
};


class Unsharp : public Kernel<data_t> {
  private:
    Accessor<data_t> &Input;
    Accessor<data_t> &InputRatio;

  public:
    Unsharp(IterationSpace<data_t> &IS, Accessor<data_t> &Input, Accessor<data_t> &InputRatio)
          : Kernel(IS), Input(Input), InputRatio(InputRatio) {
      add_accessor(&Input);
      add_accessor(&InputRatio);
    }

    void kernel() {
      output() = InputRatio() * Input();
    }
};



/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {

    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of width*height pixels
    data_t *host_in = new data_t[width*height];

    // convolution filter mask
    const data_t filter_xy[3][3] = {
        { 1, 2, 1 },
        { 2, 4, 2 },
        { 1, 2, 1 }
    };

    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = (3*(y*width + x)) % 199;
      }
    }

    // hipacc object decls
    Mask<data_t> G(filter_xy);

    Image<data_t> IN(width, height);
    Image<data_t> BLUR(width, height);
    Image<data_t> SHARP(width, height);
    Image<data_t> RATIO(width, height);
    Image<data_t> OUT(width, height);

    const BoundaryCondition<data_t> BcInClamp(IN, G, Boundary::CLAMP);
    Accessor<data_t> AccInClamp(BcInClamp);
    Accessor<data_t> AccIn(IN);

    IterationSpace<data_t> IsBlur(BLUR);
    Accessor<data_t> AccBlur(BLUR);
    IterationSpace<data_t> IsSharp(SHARP);
    Accessor<data_t> AccSharp(SHARP);
    IterationSpace<data_t> IsRatio(RATIO);
    Accessor<data_t> AccRatio(RATIO);
    IterationSpace<data_t> IS_OUT(OUT);

    Gauss GaussBlur(IsBlur, AccInClamp, G);
    Sharpen SharpKernel(IsSharp, AccIn, AccBlur);
    Ratio RatioKernel(IsRatio, AccIn, AccSharp);
    Unsharp UnsharpKernel(IS_OUT, AccIn, AccRatio);

    // kernel invocation
    IN = host_in;
    GaussBlur.execute();
    SharpKernel.execute();
    RatioKernel.execute();
    UnsharpKernel.execute();

    data_t *output = OUT.data();

#ifdef PRINT_RESULT
//    std::cerr << "\nwriting output to file...\n";
//    int ySkip = 17;
//    int xSkip = 13;
//    // compute the golden results
//    for (int y=0; y<height; y=y+ySkip) {
//      for (int x=0; x<width; x=x+xSkip) {
//        std::cout << "output at y " << y << ", x " << x << " : " <<
//          (int)output[y*width + x].x << " " << (int)output[y*width + x].y <<
//            " " << (int)output[y*width + x].z << "\n";
//      }
//    }
#endif


    // free memory
    delete[] host_in;

    return EXIT_SUCCESS;
}

