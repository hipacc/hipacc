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


#define WIDTH  1600
#define HEIGHT 900

#define PACK_INT

#ifdef PACK_INT
# define data_t uint
# define pack(a, b, c, d) \
    (uint)((uint)(a) | (uint)(b) << 8 | (uint)(c) << 16 | (uint)(d) << 24)
#else
# define data_t uchar4
# define pack(a, b, c, d) \
    ((uchar4){(uchar)a, (uchar)b, (uchar)c, (uchar)d})
#endif


using namespace hipacc;
using namespace hipacc::math;


class Mandelbrot : public Kernel<data_t> {
    private:
        const int width, height;
        const float scale, ox, oy;

    public:
        Mandelbrot(IterationSpace<data_t> &iter, const int w, const int h,
                   const float s, const float ox, const float oy)
              : Kernel(iter), width(w), height(h), scale(s), ox(ox), oy(oy) {}

        void kernel() {
            float x0 = ((float)x() / (float)width  * 3.5f) / scale - 2.5f + ox;
            float y0 = ((float)y() / (float)height * 2.0f) / scale - 1.0f + oy;
            float x = 0;
            float y = 0;
            int iteration = 0;
            int max_iteration = 1000;

            while ((x*x + y*y < 2*2) && (iteration < max_iteration)) {
                float xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                ++iteration;
            }
            
            uchar rout = 0;
            uchar gout = 0;
            uchar bout = 0;
            if (iteration < max_iteration) {
                float mu = (float)iteration + 1.0f - logf(logf(sqrtf(x*x + y*y))) / logf(2.0f);
                int c = (int)(mu / max_iteration * 768);
                if (c >= 512) {
                    bout = c - 512;
                    gout = 255 - rout;
                } else if (c >= 256) {
                    gout = c - 256;
                    rout = 255 - gout;
                } else {
                    rout = c;
                }
            }
            output() = pack(bout, gout, rout, 255);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    float timing = 0;

    data_t *output;

    std::cout << "Calculating Hipacc Mandelbrot ..." << std::endl;

    //************************************************************************//

    // output image of width x height pixels
    Image<data_t> out(width, height);

    IterationSpace<data_t> iter(out);

    const int max_steps = 1000;
    for (int i = 0; i <= max_steps; ++i) { 
        float step = (float)i/max_steps;
        float scale = 1.00f+step*9;

        Mandelbrot mandelbrot(iter, width, height, scale, step*.6f*width/100/scale, step*height/100/scale);
        mandelbrot.execute();

        timing = hipacc_last_kernel_timing();
        std::cout << "Hipacc: " << timing << " ms, "
                  << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

        // get pointer to result data
        output = out.data();
        
        if (i == 0 || i == max_steps) {
            save_data(width, height, 4, (uchar*)output, "output.jpg");
        }

        show_data(width, height, 4, (uchar*)output, "output.jpg", 1);
    }

    //************************************************************************//

    return EXIT_SUCCESS;
}
