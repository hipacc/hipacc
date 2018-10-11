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


#define SIZE_X 5
#define SIZE_Y 5
#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


// Laplace filter in Hipacc
class LaplaceFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Domain &dom;
        Mask<int> &mask;

    public:
        LaplaceFilter(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                      Domain &dom, Mask<int> &mask)
              : Kernel(iter), input(input), dom(dom), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            int sum = reduce(dom, Reduce::SUM, [&] () -> int {
                    return mask(dom) * input(dom);
                });
            sum += 128;
            sum = min(sum, 255);
            sum = max(sum, 0);
            output() = (uchar) (sum);
        }
};


// forward declaration of reference implementation
void laplace_filter(uchar *in, uchar *out, int *filter, int size,
                    int width, int height);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    float timing = 0;

    // only filter kernel sizes 1x1, 3x3 and 5x5 implemented
    if (size_x != size_y || (size_x != 1 && size_x != 3 && size_x != 5)) {
        std::cerr << "Wrong filter kernel size. "
                  << "Currently supported values: 1x1, 3x3 and 5x5!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

#if SIZE_X == 1
# define SIZE 3
#elif SIZE_X == 3
# define SIZE 3
#else
# define SIZE 5
#endif

    int coef[SIZE][SIZE] = {
#if SIZE_X==1
        { 0,  1,  0 },
        { 1, -4,  1 },
        { 0,  1,  0 }
#endif
#if SIZE_X==3
        { 2,  0,  2 },
        { 0, -8,  0 },
        { 2,  0,  2 }
#endif
#if SIZE_X==5
        { 1,   1,   1,   1,   1 },
        { 1,   1,   1,   1,   1 },
        { 1,   1, -24,   1,   1 },
        { 1,   1,   1,   1,   1 },
        { 1,   1,   1,   1,   1 }
#endif
    };

    // host memory for image of width x height pixels
    uchar *input = load_data<uchar>(width, height, 1, IMAGE);
    uchar *ref_out = new uchar[width*height];

    std::cout << "Calculating Hipacc Laplace filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(width, height, input);
    Image<uchar> out(width, height);

    // define Mask and Domain for Laplace
    Mask<int> mask(coef);
    Domain dom(mask);

    BoundaryCondition<uchar> bound(in, dom, Boundary::CLAMP);
    Accessor<uchar> acc(bound);

    IterationSpace<uchar> iter(out);
    LaplaceFilter filter(iter, acc, dom, mask);

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    uchar *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    laplace_filter(input, ref_out, (int*)coef, SIZE, width, height);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(output, ref_out, width, height, SIZE/2, SIZE/2);

    save_data(width, height, 1, input, "input.jpg");
    save_data(width, height, 1, output, "output.jpg");
    show_data(width, height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// Laplace filter reference
void laplace_filter(uchar *in, uchar *out, int *filter, int size,
                    int width, int height) {
    const int size_x = size;
    const int size_y = size;
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]
                            * in[(y+yf)*width + x + xf];
                }
            }

            sum += 128;
            sum = min(sum, 255);
            sum = max(sum, 0);
            out[y*width + x] = sum;
        }
    }
}
