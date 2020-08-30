//
// Copyright (c) 2020, University of Erlangen-Nuremberg
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

#include "hipacc.hpp"

#include <iostream>
#include <hipacc_helper.hpp>

#define WIDTH 512
#define HEIGHT 512
#define SIZE_X 5
#define SIZE_Y 5
#define TYPE uchar

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class LocalOperatorExample : public Kernel<TYPE> {
  private:
    Accessor<TYPE> &Input;
    Mask<float> &mask;

  public:
    LocalOperatorExample(IterationSpace<TYPE> &IS, Accessor<TYPE> &Input,
             Mask<float> &mask)
          : Kernel(IS), Input(Input), mask(mask) {
        add_accessor(&Input);
    }

    void kernel() {
        output() = (TYPE)(convolve(mask, Reduce::SUM, [&] () -> float {
                return mask() * Input(mask);
            }) + 0.5f);
    }
};

// forward declaration of reference implementation
void kernel_fusion(TYPE *in, TYPE *out, float *filter,
                   int size_x, int size_y, int width, int height);

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
HIPACC_CODEGEN int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    const int offset_x = size_x >> 1;
    const int offset_y = size_y >> 1;

    // convolution filter mask
    const float coef[SIZE_Y][SIZE_X] = {
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f }
    };

    // host memory for image of width x height pixels, random
    TYPE *input = (TYPE*)load_data<TYPE>(width, height);
    TYPE *ref_out = new TYPE[width*height];

    std::cout << "Testing Hipacc kernel fusion ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<TYPE> in(width, height, input);
    Image<TYPE> out(width, height);

    // test local to local kernel fusion
    // e.g., l -> l
    Mask<float> mask(coef);
    BoundaryCondition<TYPE> bound0(in, mask, Boundary::CLAMP);
    Accessor<TYPE> acc0(bound0);
    Image<TYPE> buf0(width, height);
    IterationSpace<TYPE> iter0(buf0);
    LocalOperatorExample localOp0(iter0, acc0, mask);

    BoundaryCondition<TYPE> bound1(buf0, mask, Boundary::CLAMP);
    Accessor<TYPE> acc1(bound1);
    IterationSpace<TYPE> iter1(out);
    LocalOperatorExample localOp1(iter1, acc1, mask);

    // execution after all decls
    localOp0.execute();
    localOp1.execute();

    // get pointer to result data
    TYPE *output = out.data();

    //************************************************************************//
    std::cout << "Calculating reference ..." << std::endl;
    kernel_fusion(input, ref_out, (float*)coef, size_x, size_y, width, height);
    compare_results(output, ref_out, width, height, offset_x*2, offset_y*2);

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}

// kernel fusion reference
void local_kernel(TYPE *in, TYPE *out, float *filter,
                     int size_x, int size_y, int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            float sum = 0.5f;
            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]
                            * in[(y+yf)*width + x + xf];
                }
            }
            out[y*width + x] = (TYPE) (sum);
        }
    }
}

void kernel_fusion(TYPE *in, TYPE *out, float *filter,
                   int size_x, int size_y, int width, int height) {
  TYPE *ref_buf0 = new TYPE[width*height];
  local_kernel(in, ref_buf0, filter, size_x, size_y, width, height);
  local_kernel(ref_buf0, out, filter, size_x, size_y, width, height);
  delete[] ref_buf0;
}
