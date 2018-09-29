//
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


#define WIDTH  4096
#define HEIGHT 4096


using namespace hipacc;
using namespace hipacc::math;


// Kernel description in Hipacc
class Reduction : public Kernel<float> {
    private:
        Accessor<float> &in;

    public:
        Reduction(IterationSpace<float> &iter, Accessor<float> &in)
              : Kernel(iter), in(in) {
            add_accessor(&in);
        }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) const {
            return max(left,right);
        }
};


// forward declaration of reference implementation
void reduction(float *in, float *out, int width, int height);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    float timing = 0;

    // host memory for image of width x height pixels
    float *input = load_data<float>(width, height);
    float ref_out;

    std::cout << "Calculating Hipacc reduction ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<float> in(width, height, input);
    Image<float> out(width, height);

    Accessor<float> acc(in);

    IterationSpace<float> iter(out);
    Reduction filter(iter, acc);

    filter.execute();
    float output = filter.reduced_data();
    timing = hipacc_last_kernel_timing();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    reduction(input, &ref_out, width, height);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(&output, &ref_out);

    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}


// reduction reference
void reduction(float *in, float *out, int width, int height) {
    for (int p = 0; p < width * height; ++p) {
        *out += max(*out,in[p]);
    }
}
