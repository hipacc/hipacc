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


using namespace hipacc;
using namespace hipacc::math;


// Kernel description in Hipacc

class RGB2Gray : public Kernel<float> {
private:
  Accessor<uchar4> &input;

public:
  RGB2Gray(IterationSpace<float> &iter, Accessor<uchar4> &input)
      : Kernel(iter), input(input) {
    add_accessor(&input);
  }

  void kernel() {
    uchar4 pixel = input();
    float c = 1.0f/2.2f;

    // gamma correction
    float xg = powf((float)pixel.x, c);
    float yg = powf((float)pixel.y, c);
    float zg = powf((float)pixel.z, c);
    // Luminance
    output() = 0.2126f * xg + 0.7152 * yg + 0.0722 * zg;
  }
};


class GaussianBlur : public Kernel<float> {
private:
  Accessor<float> &input;
  Mask<float> &mask;

public:
  GaussianBlur(IterationSpace<float> &iter, Accessor<float> &input,
               Mask<float> &mask)
      : Kernel(iter), input(input), mask(mask) {
    add_accessor(&input);
  }

  void kernel() {
    output() = convolve(mask, Reduce::SUM,
               [&]() -> float { return mask() * input(mask); });
  }
};


class Sharpen : public Kernel<float> {
private:
  Accessor<float> &Gray;
  Accessor<float> &InputBlur;

public:
  Sharpen(IterationSpace<float> &IS, Accessor<float> &Gray,
          Accessor<float> &InputBlur)
      : Kernel(IS), Gray(Gray), InputBlur(InputBlur) {
    add_accessor(&Gray);
    add_accessor(&InputBlur);
  }

  void kernel() { output() = 2 * Gray() - InputBlur(); }
};


class Ratio : public Kernel<float> {
private:
  Accessor<float> &Gray;
  Accessor<float> &InputSharp;

public:
  Ratio(IterationSpace<float> &IS, Accessor<float> &Gray,
        Accessor<float> &InputSharp)
      : Kernel(IS), Gray(Gray), InputSharp(InputSharp) {
    add_accessor(&Gray);
    add_accessor(&InputSharp);
  }

  void kernel() { 
    float pixel = Gray();
    pixel = max(pixel, 0.01f); 
    output() = InputSharp() / pixel; 
  }
};


class Unsharp : public Kernel<uchar4> {
private:
  Accessor<uchar4> &Input;
  Accessor<float> &InputRatio;

public:
  Unsharp(IterationSpace<uchar4> &IS, Accessor<uchar4> &Input,
          Accessor<float> &InputRatio)
      : Kernel(IS), Input(Input), InputRatio(InputRatio) {
    add_accessor(&Input);
    add_accessor(&InputRatio);
  }

  void kernel() { 
    uchar4 in = Input();
    float r = InputRatio();
    in.x *= r;
    in.y *= r;
    in.z *= r;
    output() = in;
  }
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

  // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
  if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
    std::cerr << "Wrong filter kernel size. "
              << "Currently supported values: 3x3, 5x5, and 7x7!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // convolution filter mask
  const float coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
    {0.057118f, 0.124758f, 0.057118f},
    {0.124758f, 0.272496f, 0.124758f},
    {0.057118f, 0.124758f, 0.057118f}
#endif
#if SIZE_X == 5
    {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f},
    {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
    {0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f},
    {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
    {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f}
#endif
#if SIZE_X == 7
    {0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841},
    {0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010},
    {0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471},
    {0.008351, 0.029902, 0.064280, 0.082959, 0.064280, 0.029902, 0.008351},
    {0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471},
    {0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010},
    {0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841}
#endif
  };

  // host memory for image of width x height pixels
  uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4, IMAGE);

  std::cout << "Calculating Hipacc Unsharp filter ..." << std::endl;

  //************************************************************************//

  // input and output images
  Image<uchar4> in(width, height, input);
  Image<float> gray(width, height);
  Image<float> blur(width, height);
  Image<float> sharp(width, height);
  Image<float> ratio(width, height);
  Image<uchar4> out(width, height);

  // define Mask for Gaussian filter
  Mask<float> mask(coef);

  IterationSpace<float> iterGray(gray);
  Accessor<uchar4> accIn(in);
  RGB2Gray rgb2gray(iterGray, accIn);
  rgb2gray.execute();
  timing = hipacc_last_kernel_timing();

  BoundaryCondition<float> bound(gray, mask, Boundary::CLAMP);
  Accessor<float> accGrayBound(bound);
  Accessor<float> accGray(gray);
  IterationSpace<float> iterBlur(blur);
  GaussianBlur GB(iterBlur, accGrayBound, mask);
  GB.execute();
  timing = hipacc_last_kernel_timing();

  IterationSpace<float> iterSharp(sharp);
  Accessor<float> accBlur(blur);
  Sharpen Sha(iterSharp, accGray, accBlur);
  Sha.execute();
  timing = hipacc_last_kernel_timing();

  IterationSpace<float> iterRatio(ratio);
  Accessor<float> accSharp(sharp);
  Ratio Rat(iterRatio, accGray, accSharp);
  Rat.execute();
  timing = hipacc_last_kernel_timing();

  IterationSpace<uchar4> iter(out);
  Accessor<float> accRatio(ratio);
  Unsharp Uns(iter, accIn, accRatio);
  Uns.execute();
  timing = hipacc_last_kernel_timing();

  // get pointer to result data
  uchar4 *output = out.data();

  //************************************************************************//

  std::cout << "Hipacc: " << timing << " ms, "
            << (width * height / timing) / 1000 << " Mpixel/s" << std::endl;

  save_data(width, height, 4, (uchar*)input, "input.jpg");
  save_data(width, height, 4, (uchar*)output, "output.jpg");
  show_data(width, height, 4, (uchar*)output, "output.jpg");

  // free memory
  delete[] input;

  return EXIT_SUCCESS;
}


