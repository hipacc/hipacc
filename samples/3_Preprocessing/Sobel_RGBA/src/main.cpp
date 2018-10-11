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


#define SIZE_X 7
#define SIZE_Y 7
#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"

#if SIZE_X == 7
# define data_t int4
# define convert_output(x) x
#else
# define data_t short4
# define convert_output(x) convert_short4(x)
#endif


using namespace hipacc;
using namespace hipacc::math;


// Sobel filter in Hipacc
class Sobel : public Kernel<data_t> {
    private:
        Accessor<uchar4> &input;
        Domain &dom;
        Mask<int> &mask;

    public:
        Sobel(IterationSpace<data_t> &iter, Accessor<uchar4> &input,
              Domain &dom, Mask<int> &mask)
              : Kernel(iter), input(input), dom(dom), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            int4 sum = reduce(dom, Reduce::SUM, [&] () -> int4 {
                    return mask(dom) * convert_int4(input(dom));
                });
            output() = convert_output(sum);
        }
};

class SobelCombine : public Kernel<uchar4> {
    private:
        Accessor<data_t> &input1;
        Accessor<data_t> &input2;
        int norm;

    public:
        SobelCombine(IterationSpace<uchar4> &iter, Accessor<data_t> &input1,
                     Accessor<data_t> &input2, int norm)
              : Kernel(iter), input1(input1), input2(input2), norm(norm) {
            add_accessor(&input1);
            add_accessor(&input2);
        }

        void kernel() {
            data_t in1 = input1()/norm;
            data_t in2 = input2()/norm;
            float4 result = sqrtf(convert_float4(in1*in1 + in2*in2));
            result = min(result, 255.0f);
            result = max(result, 0.0f);
            output() = convert_uchar4(result);
        }
};


// forward declarations of reference implementation
void sobel_filter(uchar4 *in, data_t *out, int *filter, int size_x, int size_y,
                  int width, int height);
void sobel_combine(data_t *input1, data_t *input2, uchar4 *out,
                   int width, int height, int norm);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
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

#if SIZE_X==3
    const int norm = 4;

    const int coef_x[SIZE_Y][SIZE_X] = {
        { -1, 0,  1 },
        { -2, 0,  2 },
        { -1, 0,  1 }};

    const int coef_y[SIZE_Y][SIZE_X] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }};
#endif
#if SIZE_X==5
    const int norm = 48;

    const int coef_x[SIZE_Y][SIZE_X] = {
        { -1,  -2, 0,  2, 1 },
        { -4,  -8, 0,  8, 4 },
        { -6, -12, 0, 12, 6 },
        { -4,  -8, 0,  8, 4 },
        { -1,  -2, 0,  2, 1 }};

    const int coef_y[SIZE_Y][SIZE_X] = {
        { -1, -4, -6,  -4, -1 },
        { -2, -8, -12, -8, -2 },
        {  0,  0,  0,   0,  0 },
        {  2,  8,  12,  8,  2 },
        {  1,  4,  6,   4,  1 }};
#endif
#if SIZE_X==7
    const int norm = 640;

    const int coef_x[SIZE_Y][SIZE_X] = {
        {  -1,  -4,   -5, 0,   5,  4,  1 },
        {  -6, -24,  -30, 0,  30, 24,  6 },
        { -15, -60,  -75, 0,  75, 60, 15 },
        { -20, -80, -100, 0, 100, 80, 20 },
        { -15, -60,  -75, 0,  75, 60, 15 },
        {  -6, -24,  -30, 0,  30, 24,  6 },
        {  -1,  -4,   -5, 0,   5,  4,  1 }};

    const int coef_y[SIZE_Y][SIZE_X] = {
        { -1, -6,  -15, -20,  -15, -6,  -1 },
        { -4, -24, -60, -80,  -60, -24, -4 },
        { -5, -30, -75, -100, -75, -30, -5 },
        {  0,  0,   0,   0,    0,   0,   0 },
        {  5,  30,  75,  100,  75,  30,  5 },
        {  4,  24,  60,  80,   60,  24,  4 },
        {  1,  6,   15,  20,   15,  6,   1 }};
#endif

    // host memory for image of width x height pixels
    uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4, IMAGE);
    data_t *ref_tmp1 = new data_t[width*height];
    data_t *ref_tmp2 = new data_t[width*height];
    uchar4 *ref_out = new uchar4[width*height];

    std::cout << "Calculating Hipacc Sobel filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<data_t> tmp1(width, height);
    Image<data_t> tmp2(width, height);
    Image<uchar4> out(width, height);

    // define Masks and Domains for Sobel filter
    Mask<int> mask_x(coef_x);
    Mask<int> mask_y(coef_y);
    Domain dom_x(mask_x);
    Domain dom_y(mask_y);

    BoundaryCondition<uchar4> bound(in, dom_x, Boundary::CLAMP);
    Accessor<uchar4> acc(bound);
    Accessor<data_t> acc_tmp1(tmp1);
    Accessor<data_t> acc_tmp2(tmp2);

    IterationSpace<data_t> iter_tmp1(tmp1);
    IterationSpace<data_t> iter_tmp2(tmp2);
    IterationSpace<uchar4> iter(out);

    Sobel filter_x(iter_tmp1, acc, dom_x, mask_x);
    filter_x.execute();
    timing += hipacc_last_kernel_timing();

    Sobel filter_y(iter_tmp2, acc, dom_y, mask_y);
    filter_y.execute();
    timing += hipacc_last_kernel_timing();

    SobelCombine combine(iter, acc_tmp1, acc_tmp2, norm);
    combine.execute();
    timing += hipacc_last_kernel_timing();

    // get pointer to result data
    uchar4 *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    sobel_filter(input, ref_tmp1, (int*)coef_x, size_x, size_y, width, height);
    sobel_filter(input, ref_tmp2, (int*)coef_y, size_x, size_y, width, height);
    sobel_combine(ref_tmp1, ref_tmp2, ref_out, width, height, norm);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results((uchar*)output, (uchar*)ref_out, width*4, height, offset_x*4, offset_y);

    save_data(width, height, 4, (uchar*)input, "input.jpg");
    save_data(width, height, 4, (uchar*)output, "output.jpg");
    show_data(width, height, 4, (uchar*)output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_tmp1;
    delete[] ref_tmp2;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// Sobel filter reference
void sobel_filter(uchar4 *in, data_t *out, int *filter, int size_x, int size_y,
                  int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int4 sum = {0, 0, 0, 0};

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]
                            * convert_int4(in[(y+yf)*width + x + xf]);
                }
            }
            out[y*width + x] = convert_output(sum);
        }
    }
}

void sobel_combine(data_t *input1, data_t *input2, uchar4 *out,
                   int width, int height, int norm) {
    for (int p = 0; p < width*height; ++p) {
        data_t in1 = input1[p]/norm;
        data_t in2 = input2[p]/norm;
        float4 result = sqrtf(convert_float4(in1*in1 + in2*in2));
        result = min(result, 255.0f);
        result = max(result, 0.0f);
        out[p] = convert_uchar4(result);
    }
}
