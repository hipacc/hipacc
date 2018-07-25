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


#define SIGMA_S 13
#define SIGMA_R 16
#define WIDTH   4032
#define HEIGHT  3024
#define IMAGE   "../../common/img/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


class BilateralFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &in;
        Mask<float> &mask;
        Domain &dom;
        int sigma_r;

    public:
        BilateralFilter(IterationSpace<uchar> &iter, Accessor<uchar> &in,
                        Mask<float> &mask, Domain &dom, int sigma_r)
              : Kernel(iter), in(in), mask(mask), dom(dom), sigma_r(sigma_r) {
            add_accessor(&in);
        }

        void kernel() {
            float c_r = 0.5f/(sigma_r*sigma_r);
            float d = 0.0f;
            float p = 0.0f;
            float center = in();

            iterate(dom, [&] () -> void {
                    float diff = (float)(in(dom)) - center;
                    float s = expf(-c_r * diff*diff) * mask(dom);
                    d += s;
                    p += s * in(dom);
                });

            output() = (uchar)(p/d+0.5f);
        }
};


// forward declaration of reference implementation
void bilateral_filter(uchar *in, uchar *out, float *filter,
                      int sigma_s, int sigma_r, int width, int height);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int sigma_s = SIGMA_S;
    const int sigma_r = SIGMA_R;
    const int offset = sigma_s >> 1;
    float timing = 0;

    // only filter kernel sizes 3x3, 5x5, 7x7, and 13x13 implemented
    if (sigma_s != 3 && sigma_s != 5 && sigma_s != 7 && sigma_s != 13) {
        std::cerr << "Wrong filter kernel size. "
                  << "Currently supported values: 3x3, 5x5, 7x7, and 13x13!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    const float coef[SIGMA_S][SIGMA_S] = {
#if SIGMA_S==3
        { 0.018316f, 0.135335f, 0.018316f },
        { 0.135335f, 1.000000f, 0.135335f },
        { 0.018316f, 0.135335f, 0.018316f }
#endif
#if SIGMA_S==5
        { 0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f },
        { 0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f },
        { 0.135335f, 0.606531f, 1.000000f, 0.606531f, 0.135335f },
        { 0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f },
        { 0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f }
#endif
#if SIGMA_S==7
        { 0.018316f, 0.055638f, 0.108368f, 0.135335f, 0.108368f, 0.055638f, 0.018316f },
        { 0.055638f, 0.169013f, 0.329193f, 0.411112f, 0.329193f, 0.169013f, 0.055638f },
        { 0.108368f, 0.329193f, 0.641180f, 0.800737f, 0.641180f, 0.329193f, 0.108368f },
        { 0.135335f, 0.411112f, 0.800737f, 1.000000f, 0.800737f, 0.411112f, 0.135335f },
        { 0.108368f, 0.329193f, 0.641180f, 0.800737f, 0.641180f, 0.329193f, 0.108368f },
        { 0.055638f, 0.169013f, 0.329193f, 0.411112f, 0.329193f, 0.169013f, 0.055638f },
        { 0.018316f, 0.055638f, 0.108368f, 0.135335f, 0.108368f, 0.055638f, 0.018316f }
#endif
#if SIGMA_S==13
        { 0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f, 0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f, 0.018316f },
        { 0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f, 0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f, 0.033746f },
        { 0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f, 0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f, 0.055638f },
        { 0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f, 0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f, 0.082085f },
        { 0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f, 0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f, 0.108368f },
        { 0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f, 0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f, 0.128022f },
        { 0.135335f, 0.249352f, 0.411112f, 0.606531f, 0.800737f, 0.945959f, 1.000000f, 0.945959f, 0.800737f, 0.606531f, 0.411112f, 0.249352f, 0.135335f },
        { 0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f, 0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f, 0.128022f },
        { 0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f, 0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f, 0.108368f },
        { 0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f, 0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f, 0.082085f },
        { 0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f, 0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f, 0.055638f },
        { 0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f, 0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f, 0.033746f },
        { 0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f, 0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f, 0.018316f }
#endif
    };

    // host memory for image of width x height pixels
    uchar *input = load_data<uchar>(width, height, 1, IMAGE);
    uchar *ref_out = new uchar[width*height];

    std::cout << "Calculating Hipacc bilateral filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(width, height, input);
    Image<uchar> out(width, height);

    // define Mask and Domain for bilateral filter
    Mask<float> mask(coef);
    Domain dom(mask);

    BoundaryCondition<uchar> bound(in, dom, Boundary::CLAMP);
    Accessor<uchar> acc(bound);

    IterationSpace<uchar> iter(out);
    BilateralFilter filter(iter, acc, mask, dom, sigma_r);

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    uchar *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    bilateral_filter(input, ref_out, (float*)coef, sigma_s, sigma_r, width, height);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(output, ref_out, width, height, offset, offset);

    save_data(width, height, 1, input, "input.jpg");
    save_data(width, height, 1, output, "output.jpg");
    show_data(width, height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// bilateral filter reference
void bilateral_filter(uchar *in, uchar *out, float *filter,
                      int sigma_s, int sigma_r, int width, int height) {
    int anchor = sigma_s >> 1;
    int upper_x = width  - anchor;
    int upper_y = height - anchor;

    for (int y=anchor; y<upper_y; ++y) {
        for (int x=anchor; x<upper_x; ++x) {
            float center = in[y*width + x];
            float c_r = 0.5f/(sigma_r*sigma_r);
            float d = 0.0f;
            float p = 0.0f;

            for (int yf = -anchor; yf<=anchor; ++yf) {
                for (int xf = -anchor; xf<=anchor; ++xf) {
                    float diff = (float)(in[(y + yf)*width + x + xf]) - center;
                    float s = std::exp(-c_r * diff*diff)
                            * filter[(yf+anchor)*sigma_s + xf+anchor];
                    d += s;
                    p += s * in[(y + yf)*width + x + xf];
                }
            }
            out[y*width + x] = (uchar)(p/d+0.5f);
        }
    }
}
