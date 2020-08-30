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
#define TYPE uchar

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class PointOperatorExample : public Kernel<TYPE> {
    private:
        Accessor<TYPE> &in;

    public:
        PointOperatorExample(IterationSpace<TYPE> &iter, Accessor<TYPE> &acc)
              : Kernel(iter), in(acc) {
            add_accessor(&in);
        }

        void kernel() {
            TYPE interm_pixel = in();
            interm_pixel += 3;
            output() = interm_pixel;
        }
};

// forward declaration of reference implementation
void kernel_fusion(TYPE *in, TYPE *out, int width, int height);

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
HIPACC_CODEGEN int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of width x height pixels, random
    TYPE *input = (TYPE*)load_data<TYPE>(width, height);
    TYPE *ref_out = new TYPE[width*height];

    std::cout << "Testing Hipacc kernel fusion ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<TYPE> in(width, height, input);
    Image<TYPE> out(width, height);

    // test point to point kernel fusion
    // e.g., p -> p -> ... -> p
    Accessor<TYPE> acc0(in);
    Image<TYPE> buf0(width, height);
    IterationSpace<TYPE> iter0(buf0);
    PointOperatorExample pointOp0(iter0, acc0);

    Accessor<TYPE> acc1(buf0);
    Image<TYPE> buf1(width, height);
    IterationSpace<TYPE> iter1(buf1);
    PointOperatorExample pointOp1(iter1, acc1);

    Accessor<TYPE> acc2(buf1);
    Image<TYPE> buf2(width, height);
    IterationSpace<TYPE> iter2(out);
    PointOperatorExample pointOp2(iter2, acc2);

    // execution after all decls
    pointOp0.execute();
    pointOp1.execute();
    pointOp2.execute();

    // get pointer to result data
    TYPE *output = out.data();

    //************************************************************************//
    std::cout << "Calculating reference ..." << std::endl;
    kernel_fusion(input, ref_out, width, height);
    compare_results(output, ref_out, width, height);

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}

// kernel fusion reference
void point_kernel(TYPE *in, TYPE *out, int width, int height) {
    for (int p = 0; p < width*height; ++p) {
        TYPE interm_pixel = in[p];
        interm_pixel += 3;
        out[p] = interm_pixel;
    }
}

void kernel_fusion(TYPE *in, TYPE *out, int width, int height) {
  TYPE *ref_buf0 = new TYPE[width*height];
  TYPE *ref_buf1 = new TYPE[width*height];
  point_kernel(in, ref_buf0, width, height);
  point_kernel(ref_buf0, ref_buf1, width, height);
  point_kernel(ref_buf1, out, width, height);
  delete[] ref_buf0;
  delete[] ref_buf1;
}
