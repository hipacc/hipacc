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


#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"


using namespace hipacc;
using namespace hipacc::math;


// Kernel description in Hipacc
class WindowingFilter : public Kernel<float> {
    private:
        Accessor<float> &in;
        float center;
        float wwidth;
        float scale;

    public:
        WindowingFilter(IterationSpace<float> &iter, Accessor<float> &acc,
                        float center, float wwidth, float scale)
              : Kernel(iter), in(acc), center(center), wwidth(wwidth),
                scale(scale) {
            add_accessor(&in);
        }

        void kernel() {
            float pixel = in();
            pixel = (pixel - (center - wwidth)) * scale;
            pixel = min(pixel, 255.0f);
            pixel = max(pixel, 0.0f);
            output() = pixel;
        }
};


// forward declaration of reference implementation
void windowing_filter(float *in, float *out, int width, int height,
                      float center, float wwidth, float scale);


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    float center = 128.0f;
    float wwidth = 64.0f;
    float scale = 255.0f / (2*wwidth);
    float timing = 0;

    // host memory for image of width x height pixels
    float *input = load_data<float>(width, height, 1, IMAGE);
    float *ref_out = new float[width*height];

    std::cout << "Calculating Hipacc windowing filter ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<float> in(width, height, input);
    Image<float> out(width, height);

    Accessor<float> acc(in);

    IterationSpace<float> iter(out);
    WindowingFilter filter(iter, acc, center, wwidth, scale);

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    float *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    std::cout << "Calculating reference ..." << std::endl;
    double start = time_ms();
    windowing_filter(input, ref_out, width, height, center, wwidth, scale);
    double end = time_ms();
    std::cout << "Reference: " << end-start << " ms, "
              << (width*height/(end-start))/1000 << " Mpixel/s" << std::endl;

    compare_results(output, ref_out, width, height);

    save_data(width, height, 1, input, "input.jpg");
    save_data(width, height, 1, output, "output.jpg");
    show_data(width, height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}


// windowing filter reference
void windowing_filter(float *in, float *out, int width, int height,
                      float center, float wwidth, float scale) {
    for (int p = 0; p < width*height; ++p) {
        float pixel = in[p];
        pixel = (pixel - (center - wwidth)) * scale;
        pixel = min(pixel, 255.0f);
        pixel = max(pixel, 0.0f);
        out[p] = pixel;
    }
}
