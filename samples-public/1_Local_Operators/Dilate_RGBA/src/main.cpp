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


#ifndef IMAGE_BASE_PATH
# define IMAGE_BASE_PATH ""
#endif


#define SIZE_X 5
#define SIZE_Y 5
#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  IMAGE_BASE_PATH"/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


class Dilate : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &in;
        Domain &dom;

    public:
        Dilate(IterationSpace<uchar4> &iter, Accessor<uchar4> &in, Domain &dom)
              : Kernel(iter), in(in), dom(dom) {
            add_accessor(&in);
        }

        void kernel() {
            output() = reduce(dom, Reduce::MAX, [&] () -> uchar4 {
                    return in(dom);
                });
        }
};


// forward declaration of reference implementation
void dilate_filter(uchar4 *in, uchar4 *out, int size_x, int size_y,
                   int width, int height);


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
    float timing = 0;

    // host memory for image of width x height pixels
    uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4, IMAGE);
    uchar4 *ref_out = new uchar4[width*height];

    std::cout << "Calculating Hipacc dilate filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar4> out(width, height);

    // define Domain for dilate filter
    Domain dom(size_x, size_y);

    BoundaryCondition<uchar4> bound(in, dom, Boundary::CLAMP);
    Accessor<uchar4> acc(bound);

    IterationSpace<uchar4> iter(out);
    Dilate filter(iter, acc, dom);

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    uchar4 *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    dilate_filter(input, ref_out, size_x, size_y, width, height);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results((uchar*)output, (uchar*)ref_out, width*4, height, offset_x*4, offset_y);

    save_data(width, height, 4, (uchar*)input, "input.jpg");
    save_data(width, height, 4, (uchar*)output, "output.jpg");
    show_data(width, height, 4, (uchar*)output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// dilate filter reference
void dilate_filter(uchar4 *in, uchar4 *out, int size_x, int size_y,
                   int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            uchar4 val = {0, 0, 0, 0};

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    val = max(val, in[(y + yf)*width + x + xf]);
                }
            }
            out[y*width + x] = val;
        }
    }
}
