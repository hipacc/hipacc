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


#define SIZE_X 7
#define SIZE_Y 7
#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  IMAGE_BASE_PATH"/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


// Gaussian blur filter in Hipacc
class GaussianBlur : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &input;
        Mask<float> &mask;

    public:
        GaussianBlur(IterationSpace<uchar4> &iter, Accessor<uchar4> &input,
                     Mask<float> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            float4 sum = convolve(mask, Reduce::SUM, [&] () -> float4 {
                    return mask() * convert_float4(input(mask));
                });
            output() = convert_uchar4(sum + 0.5f);
        }
};


// forward declaration of reference implementation
void gaussian_filter(uchar4 *in, uchar4 *out, float *filter,
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
    float timing = 0;

    // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
    if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
        std::cerr << "Wrong filter kernel size. "
                  << "Currently supported values: 3x3, 5x5, and 7x7!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
#endif
#if SIZE_X == 5
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f }
#endif
#if SIZE_X == 7
        { 0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f },
        { 0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f },
        { 0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f },
        { 0.008351f, 0.029902f, 0.064280f, 0.082959f, 0.064280f, 0.029902f, 0.008351f },
        { 0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f },
        { 0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f },
        { 0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f }
#endif
    };

    // host memory for image of width x height pixels
    uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4, IMAGE);
    uchar4 *ref_out = new uchar4[width*height];

    std::cout << "Calculating Hipacc Gaussian filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar4> out(width, height);

    // define Mask for Gaussian filter
    Mask<float> mask(coef);

    BoundaryCondition<uchar4> bound(in, mask, Boundary::CLAMP);
    Accessor<uchar4> acc(bound);

    IterationSpace<uchar4> iter(out);
    GaussianBlur filter(iter, acc, mask);

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    uchar4 *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    gaussian_filter(input, ref_out, (float*)coef, size_x, size_y, width, height);
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


// Gaussian blur filter reference
void gaussian_filter(uchar4 *in, uchar4 *out, float *filter,
                     int size_x, int size_y, int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            float4 sum = {0.5f, 0.5f, 0.5f, 0.5f};

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]
                            * convert_float4(in[(y+yf)*width + x + xf]);
                }
            }
            out[y*width + x] = convert_uchar4(sum);
        }
    }
}
