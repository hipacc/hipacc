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
class Histogram : public Kernel<float,uint> {
    private:
        Accessor<float> &in;

    public:
        Histogram(IterationSpace<float> &iter, Accessor<float> &in)
              : Kernel(iter), in(in) {
            add_accessor(&in);
        }

        void kernel() {
            output() = in();
        }

        void binning(unsigned int x, unsigned int y, float pixel) {
            bin(pixel/255.0f*num_bins()) = 1;
        }

        uint reduce(uint left, uint right) const {
            return left + right;
        }
};


// forward declaration of reference implementation
void histogram(float *in, uint *out, int width, int height, int num_bins);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int num_bins = 256;
    float timing = 0;

    // host memory for image of width x height pixels
    float *input = load_data<float>(width, height);
    uint *ref_out = new uint[num_bins]();

    std::cout << "Calculating Hipacc histogram ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<float> in(width, height, input);
    Image<float> out(width, height);

    Accessor<float> acc(in);

    IterationSpace<float> iter(out);
    Histogram filter(iter, acc);

    filter.execute();
    uint* output = filter.binned_data(num_bins);
    timing = hipacc_last_kernel_timing();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    histogram(input, ref_out, width, height, num_bins);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(output, ref_out, num_bins);

    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}


// histogram reference
void histogram(float *in, uint *out, int width, int height, int num_bins) {
    for (int p = 0; p < width * height; ++p) {
        float pixel = in[p];
        out[(uint)(pixel/255.0f*num_bins)] += 1;
    }
}
