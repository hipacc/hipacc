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
# define data_t int
#else
# define data_t short
#endif


using namespace hipacc;
using namespace hipacc::math;


// Sobel filter in Hipacc
class Sobel : public Kernel<data_t> {
    private:
        Accessor<uchar> &input;
        Domain &dom;
        Mask<int> &mask;

    public:
        Sobel(IterationSpace<data_t> &iter, Accessor<uchar> &input,
              Domain &dom, Mask<int> &mask)
              : Kernel(iter), input(input), dom(dom), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            output() = (data_t)(reduce(dom, Reduce::SUM, [&] () -> int {
                    return mask(dom) * input(dom);
                }));
        }
};

class SobelCombine : public Kernel<uchar> {
    private:
        Accessor<data_t> &input1;
        Accessor<data_t> &input2;
        data_t norm;

    public:
        SobelCombine(IterationSpace<uchar> &iter, Accessor<data_t> &input1,
                     Accessor<data_t> &input2, data_t norm)
              : Kernel(iter), input1(input1), input2(input2), norm(norm) {
            add_accessor(&input1);
            add_accessor(&input2);
        }

        void kernel() {
            data_t in1 = input1()/norm;
            data_t in2 = input2()/norm;
            float result = sqrt((float)(in1*in1 + in2*in2));
            result = min(result, 255.0f);
            result = max(result, 0.0f);
            output() = result;
        }
};


// forward declarations of reference implementation
void sobel_filter(uchar *in, data_t *out, int *filter, int size_x, int size_y,
                  int width, int height);
void sobel_combine(data_t *input1, data_t *input2, uchar *out,
                   int width, int height, data_t norm);


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
    uchar *input = load_data<uchar>(width, height, 1, IMAGE);
    data_t *ref_tmp1 = new data_t[width*height];
    data_t *ref_tmp2 = new data_t[width*height];
    uchar *ref_out = new uchar[width*height];

    std::cout << "Calculating Hipacc Sobel filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(width, height, input);
    Image<data_t> tmp1(width, height);
    Image<data_t> tmp2(width, height);
    Image<uchar> out(width, height);

    // define Masks and Domains for Sobel filter
    Mask<int> mask_x(coef_x);
    Mask<int> mask_y(coef_y);
    Domain dom_x(mask_x);
    Domain dom_y(mask_y);

    BoundaryCondition<uchar> bound(in, dom_x, Boundary::CLAMP);
    Accessor<uchar> acc(bound);
    Accessor<data_t> acc_tmp1(tmp1);
    Accessor<data_t> acc_tmp2(tmp2);

    IterationSpace<data_t> iter_tmp1(tmp1);
    IterationSpace<data_t> iter_tmp2(tmp2);
    IterationSpace<uchar> iter(out);

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
    uchar *output = out.data();

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

    compare_results(output, ref_out, width, height, offset_x, offset_y);

    save_data(width, height, 1, input, "input.jpg");
    save_data(width, height, 1, output, "output.jpg");
    show_data(width, height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_tmp1;
    delete[] ref_tmp2;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// Sobel filter reference
void sobel_filter(uchar *in, data_t *out, int *filter, int size_x, int size_y,
                  int width, int height) {
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
            out[y*width + x] = sum;
        }
    }
}

void sobel_combine(data_t *input1, data_t *input2, uchar *out,
                   int width, int height, data_t norm) {
    for (int p = 0; p < width*height; ++p) {
        data_t in1 = input1[p]/norm;
        data_t in2 = input2[p]/norm;
        float result = sqrt(in1*in1 + in2*in2);
        result = min(result, 255.0f);
        result = max(result, 0.0f);
        out[p] = result;
    }
}


