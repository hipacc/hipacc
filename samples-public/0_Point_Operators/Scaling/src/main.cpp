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


#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  IMAGE_BASE_PATH"/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


// Kernel description in Hipacc
class Scale : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;

    public:
        Scale(IterationSpace<uchar> &iter, Accessor<uchar> &input)
              : Kernel(iter), input(input) {
            add_accessor(&input);
        }

        void kernel() {
            // copy kernel: interpolation is implictly done by Hipacc
            output() = input();
        }
};


// interpolation type for reference implementation
enum class IntType { Nearest, Linear };

// forward declaration of reference implementation
void scale(uchar *in, uchar *out, int in_width, int in_height, int out_width, int out_height, IntType type);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
HIPACC_CODEGEN int main(int argc, const char **argv) {
    const int full_width = WIDTH;
    const int full_height = HEIGHT;
    const int small_width = WIDTH/2;
    const int small_height = HEIGHT/2;
    float timing = 0;

    // host memory for image of width x height pixels
    uchar *input = load_data<uchar>(full_width, full_height, 1, IMAGE);
    uchar *ref_tmp = new uchar[small_width*small_height];
    uchar *ref_out = new uchar[full_width*full_height];

    std::cout << "Calculating Hipacc image scale filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(full_width, full_height, input);
    Image<uchar> tmp(small_width, small_height);
    Image<uchar> out(full_width, full_height);

    // downscale with linear filter (LF) interpolation
    Accessor<uchar> acc1(in, Interpolate::LF);
    IterationSpace<uchar> iter1(tmp);
    Scale downscale(iter1, acc1);

    downscale.execute();
    timing = hipacc_last_kernel_timing();

    // upscale with nearest neighbor (NN) interpolation
    Accessor<uchar> acc2(tmp, Interpolate::NN);
    IterationSpace<uchar> iter2(out);
    Scale upscale(iter2, acc2);

    upscale.execute();
    timing += hipacc_last_kernel_timing();

    // get pointer to result data
    uchar *output = out.data();

    //************************************************************************//

    int pixels = small_width*small_height + full_width*full_height;
    std::cout << "Hipacc: " << timing << " ms, "
              << (pixels/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    scale(input, ref_tmp, full_width, full_height, small_width, small_height, IntType::Linear);
    scale(ref_tmp, ref_out, small_width, small_height, full_width, full_height, IntType::Nearest);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (pixels/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(output, ref_out, full_width, full_height);

    save_data(full_width, full_height, 1, input, "input.jpg");
    save_data(full_width, full_height, 1, output, "output.jpg");
    show_data(full_width, full_height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_out;
    delete[] ref_tmp;

    return EXIT_SUCCESS;
}


template<typename F, typename T>
F lerp(F w, T v1, T v2) {
    return (static_cast<F>(1)-w) * v1 + w * v2;
}


// image scale reference
void scale(uchar *in, uchar *out, int in_width, int in_height, int out_width, int out_height, IntType type) {
    float step_x = static_cast<float>(in_width) / out_width;
    float step_y = static_cast<float>(in_height) / out_height;

    for (int y=0; y<out_height; ++y) {
        for (int x=0; x<out_width; ++x) {
            // compute cell-centered input position
            float xf = step_x/2 + x*step_x;
            float yf = step_y/2 + y*step_y;

            switch (type) {
              case IntType::Nearest: {
                int x1 = static_cast<int>(xf);
                int y1 = static_cast<int>(yf);
                out[y*out_width + x] = in[y1*in_width + x1];
                break;
              }
              case IntType::Linear: {
                // move from cell-centered to absolute coordinate
                xf -= 0.5f;
                yf -= 0.5f;
                if (xf < 0.0f) xf = 0.0f;
                if (yf < 0.0f) yf = 0.0f;

                int x1 = static_cast<int>(xf);
                int y1 = static_cast<int>(yf);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                if (x2 >= in_width) x2 = in_width-1;
                if (y2 >= in_height) y2 = in_height-1;

                float wx = xf - x1;
                float wy = yf - y1;

                out[y*out_width + x] = static_cast<uchar>(
                    lerp(wy, lerp(wx, in[y1*in_width + x1],
                                      in[y1*in_width + x2]),
                             lerp(wx, in[y2*in_width + x1],
                                      in[y2*in_width + x2])));
                break;
              }
              default:
                std::cerr << "Unsupported interpolation type" << std::endl;
                exit(1);
            }
        }
    }
}
