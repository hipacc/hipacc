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

//===---------- Image Enhancement Algorithm -------------------------------===//
//
// This file implements an image enhancement algorithm based on
// Suman S. et al. (2014) Image Enhancement Using Geometric Mean Filter and Gamma
// Correction for WCE Images. In: Loo C.K., Yap K.S., Wong K.W., Beng Jin A.T., 
// Huang K. (eds) Neural Information Processing. ICONIP 2014. 
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "hipacc.hpp"

//#define PRINT_RESULT
#define WIDTH  4032
#define HEIGHT 3024

using namespace hipacc;
using namespace hipacc::math;

// kernel description in Hipacc
class AverageFilter : public Kernel<float> {
  private:
    Accessor<float> &Input;
    Mask<float> &cMask;

  public:
    AverageFilter(IterationSpace<float> &IS,
            Accessor<float> &Input, Mask<float> &cMask)
          : Kernel(IS),
            Input(Input),
            cMask(cMask) {
      add_accessor(&Input);
    }

    void kernel() {
      float sum = 0.0f;
      sum += convolve(cMask, Reduce::SUM, [&] () -> float {
          return Input(cMask) * cMask();
      });
      output() = sum;
    }
};

class GlobalGain : public Kernel<float> {
  private:
    Accessor<float> &Input;
    int Gain;

  public:
    GlobalGain(IterationSpace<float> &IS,
            Accessor<float> &Input, int Gain)
          : Kernel(IS),
            Input(Input),
            Gain(Gain) {
      add_accessor(&Input);
    }

    void kernel() {
      output() = Input() * Gain;
    }
};

class GammaCorrection : public Kernel<float> {
  private:
    Accessor<float> &Input;
    float Gamma;

  public:
    GammaCorrection(IterationSpace<float> &IS,
            Accessor<float> &Input, float Gamma)
          : Kernel(IS),
            Input(Input),
            Gamma(Gamma) {
      add_accessor(&Input);
    }

    void kernel() {
      output() = powf(Input(), Gamma);
    }
};



/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    int gain = 2;
    float gamma = 0.6;
    // define filters
    const float average3[3][3] = {
        { 0.111111f, 0.111111f, 0.111111f },
        { 0.111111f, 0.111111f, 0.111111f },
        { 0.111111f, 0.111111f, 0.111111f }
    };
    const float average5[5][5] = {
        { 0.04f, 0.04f, 0.04f, 0.04f, 0.04f },
        { 0.04f, 0.04f, 0.04f, 0.04f, 0.04f },
        { 0.04f, 0.04f, 0.04f, 0.04f, 0.04f },
        { 0.04f, 0.04f, 0.04f, 0.04f, 0.04f },
        { 0.04f, 0.04f, 0.04f, 0.04f, 0.04f }
    };

    // load input
    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of width*height pixels
    float *host_in = new float[width*height];
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = float((y*width + x) % 199);
      }
    }

    // hipacc object decls
    Mask<float> MASK3(average3);
    Mask<float> MASK5(average5);

    Image<float> IN(width, height);
    Image<float> ImgF(width, height);
    Image<float> ImgG(width, height);
    Image<float> OUT(width, height);

    IterationSpace<float> IS_ImgF(ImgF);
    IterationSpace<float> IS_ImgG(ImgG);
    IterationSpace<float> IS_OUT(OUT);

    const BoundaryCondition<float> BcAtClamp(IN, MASK3, Boundary::CLAMP);
    Accessor<float> AccAtClamp(BcAtClamp);
    Accessor<float> AccImgF(ImgF);
    Accessor<float> AccImgG(ImgG);

    AverageFilter AF(IS_ImgF, AccAtClamp, MASK3);
    GlobalGain GG(IS_ImgG, AccImgF, gain);
    GammaCorrection GC(IS_OUT, AccImgG, gamma);

    // kernel invocation
    IN = host_in;
    AF.execute();
    GG.execute();
    GC.execute();
    float *output = OUT.data();

#ifdef PRINT_RESULT
    std::cerr << "\nwriting output to file...\n";
    int ySkip = 17;
    int xSkip = 13;
    // compute the golden results
    for (int y=0; y<height; y=y+ySkip) {
      for (int x=0; x<width; x=x+xSkip) {
        std::cout << "output at y " << y << ", x " << x << " : " <<
          (int)output[y*width + x].x << " " << (int)output[y*width + x].y <<
            " " << (int)output[y*width + x].z << "\n";
      }
    }
#endif


    // free memory
    delete[] host_in;

    return EXIT_SUCCESS;
}

