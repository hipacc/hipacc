//
// Copyright (c) 2019, University of Erlangen-Nuremberg
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Suman S. et al. (2014) Image Enhancement Using Geometric Mean Filter and
// Gamma Correction for WCE Images. In: Loo C.K., Yap K.S., Wong K.W., Beng Jin
// A.T., Huang K. (eds) Neural Information Processing. ICONIP 2014.

#include "hipacc.hpp"

#include <hipacc_helper.hpp>


#ifndef IMAGE_BASE_PATH
# define IMAGE_BASE_PATH ""
#endif
#include <iostream>

#define SIZE_X 3
#define SIZE_Y 3
#define WIDTH 4032
#define HEIGHT 3024
#define IMAGE IMAGE_BASE_PATH"/fuerte_ship.jpg"

#define data_t float

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class AverageFilter : public Kernel<data_t> {
private:
  Accessor<uchar> &Input;
  Mask<data_t> &cMask;

public:
  AverageFilter(IterationSpace<data_t> &IS, Accessor<uchar> &Input,
                Mask<data_t> &cMask)
      : Kernel(IS), Input(Input), cMask(cMask) {
    add_accessor(&Input);
  }

  void kernel() {
    data_t sum = 0.0f;
    sum += convolve(cMask, Reduce::SUM,
                    [&]() -> data_t { return Input(cMask) * cMask(); });
    output() = sum;
  }
};

class GlobalGain : public Kernel<data_t> {
private:
  Accessor<data_t> &Input;
  int Gain;

public:
  GlobalGain(IterationSpace<data_t> &IS, Accessor<data_t> &Input, int Gain)
      : Kernel(IS), Input(Input), Gain(Gain) {
    add_accessor(&Input);
  }

  void kernel() { output() = Input() * Gain; }
};

class GammaCorrection : public Kernel<data_t> {
private:
  Accessor<data_t> &Input;
  data_t Gamma;

public:
  GammaCorrection(IterationSpace<data_t> &IS, Accessor<data_t> &Input,
                  data_t Gamma)
      : Kernel(IS), Input(Input), Gamma(Gamma) {
    add_accessor(&Input);
  }

  void kernel() { output() = powf(Input(), Gamma); }
};

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
HIPACC_CODEGEN int main(int argc, const char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = SIZE_X;
  const int size_y = SIZE_Y;
  float timing = 0;
  int gain = 2;
  float gamma = 0.6;

  // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
  if (size_x != size_y || !(size_x == 3 || size_x == 5)) {
    std::cerr << "Wrong filter kernel size. "
              << "Currently supported values: 3x3 and 5x5!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // filter mask
  const float coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
    {0.111111f, 0.111111f, 0.111111f},
    {0.111111f, 0.111111f, 0.111111f},
    {0.111111f, 0.111111f, 0.111111f}
#endif
#if SIZE_X == 5
    {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
    {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
    {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
    {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
    {0.04f, 0.04f, 0.04f, 0.04f, 0.04f}
#endif
  };

  // host memory for image of width x height pixels
  uchar *input = load_data<uchar>(width, height, 1, IMAGE);
  data_t *ref_out = new data_t[width * height];

  std::cout << "Calculating Hipacc WCE Enhance App..." << std::endl;

  //************************************************************************//
  // input and output image of width x height pixels
  Image<uchar> in(width, height, input);
  Image<data_t> imgF(width, height);
  Image<data_t> imgG(width, height);
  Image<data_t> out(width, height);

  // define Mask for Gaussian filter
  Mask<float> mask(coef);

  IterationSpace<data_t> iterImgF(imgF);
  BoundaryCondition<uchar> bound(in, mask, Boundary::CLAMP);
  Accessor<uchar> accInBound(bound);
  AverageFilter AF(iterImgF, accInBound, mask);
  AF.execute();
  timing = hipacc_last_kernel_timing();

  IterationSpace<data_t> iterImgG(imgG);
  Accessor<data_t> accImgF(imgF);
  GlobalGain GG(iterImgG, accImgF, gain);
  GG.execute();
  timing = hipacc_last_kernel_timing();

  IterationSpace<data_t> iter(out);
  Accessor<data_t> accImgG(imgG);
  GammaCorrection GC(iter, accImgG, gamma);
  GC.execute();
  timing = hipacc_last_kernel_timing();

  // get pointer to result data
  data_t *output = out.data();

  //************************************************************************//

  std::cout << "Hipacc: " << timing << " ms, "
            << (width * height / timing) / 1000 << " Mpixel/s" << std::endl;

  save_data(width, height, 1, input, "input.jpg");
  save_data(width, height, 1, output, "output.jpg");
  show_data(width, height, 1, output, "output.jpg");

  // free memory
  delete[] input;
  delete[] ref_out;

  return EXIT_SUCCESS;
}
