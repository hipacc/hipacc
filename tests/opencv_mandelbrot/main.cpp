//
// Copyright (c) 2014, Saarland University
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

#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "hipacc.hpp"

using namespace cv;
using namespace hipacc;
using namespace hipacc::math;


class MandelbrotKernel : public Kernel<uchar4> {
    private:
        const int width, height;
        const float scale, ox, oy;

    public:
        MandelbrotKernel(IterationSpace<uchar4> &iter, const int w, const int h,
                const float s, const float ox, const float oy) :
            Kernel(iter), width(w), height(h), scale(s), ox(ox), oy(oy) { }

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
            
            uchar4 result = { 0, 0, 0, 0 };
            if (iteration < max_iteration) {
                float mu = (float)iteration + 1.0f - logf(logf(sqrtf(x*x + y*y))) / logf(2.0f);
                int c = (int)(mu / max_iteration * 768);
                if (c >= 512) {
                    result.z = c - 512;
                    result.y = 255 - result.z;
                } else if (c >= 256) {
                    result.y = c - 256;
                    result.x = 255 - result.y;
                } else {
                    result.x = c;
                }
            }
            output() = result;
        }
};


int main(int argc, const char **argv) {
    const int width = 1280;
    const int height = 732;

    // images
    Image<uchar4> img(width, height);
    IterationSpace<uchar4> iter_img(img);
    Mat frame(height, width, CV_8UC4);

    MandelbrotKernel mandelbrot(iter_img, width, height, 1.00f, 0.00f, 0.00f);
    mandelbrot.execute();
    std::cerr << "Hipacc Mandelbrot filter: " << hipacc_last_kernel_timing() << " ms" << std::endl;

    frame.data = (uchar *)img.data();
    imshow("Mandelbrot", frame);
    waitKey(0);

    MandelbrotKernel mandelbrot_z(iter_img, width, height, 10.00f, 0.97f, 0.83f);
    mandelbrot_z.execute();
    std::cerr << "Hipacc Mandelbrot filter (zoom): " << hipacc_last_kernel_timing() << " ms" << std::endl;

    frame.data = (uchar *)img.data();
    imshow("Mandelbrot (zoom)", frame);
    waitKey(0);

    return EXIT_SUCCESS;
}

