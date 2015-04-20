//
// Copyright (c) 2013, University of Erlangen-Nuremberg
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

#include <cfloat>
#include <cstdlib>
#include <iostream>

#include <sys/time.h>

//#define CPU
#ifdef OpenCV
#include <opencv2/opencv.hpp>
#ifndef CPU
#include <opencv2/gpu/gpu.hpp>
#endif
#endif

#include "hipacc.hpp"

// variables set by Makefile
//#define SIZE_X 5
//#define SIZE_Y 5
//#define WIDTH 4096
//#define HEIGHT 4096
#define CONST_MASK
#define USE_LAMBDA

#ifdef CPU
#define OFFSET_CV 0.5f
#else
#define OFFSET_CV 0.0f
#endif

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// box filter reference
void box_filter(uchar4 *in, uchar4 *out, int size_x, int size_y, int width, int
        height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    #ifdef OpenCV
    int upper_x = width-size_x+anchor_x;
    int upper_y = height-size_y+anchor_y;
    #else
    int upper_x = width-anchor_x;
    int upper_y = height-anchor_y;
    #endif

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int4 sum = { 0, 0, 0, 0 };

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += convert_int4(in[(y + yf)*width + x + xf]);
                }
            }
            out[y*width + x] = convert_uchar4(1.0f/(float)(size_x*size_y)*convert_float4(sum) + OFFSET_CV);
        }
    }
}


// Kernel description in Hipacc
class BoxFilter : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &in;
        Domain &dom;
        int size_x, size_y;

    public:
        BoxFilter(IterationSpace<uchar4> &iter, Accessor<uchar4> &in, Domain
                &dom, int size_x, int size_y) :
            Kernel(iter),
            in(in),
            dom(dom),
            size_x(size_x),
            size_y(size_y)
        { add_accessor(&in); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = convert_uchar4( (1/(float)(size_x*size_y)) *
                    (convert_float4(reduce(dom, Reduce::SUM, [&] () -> int4 {
                    return convert_int4(in(dom));
                    })) + OFFSET_CV));
        }
        #else
        void kernel() {
            int anchor_x = size_x >> 1;
            int anchor_y = size_y >> 1;
            int4 sum = { 0, 0, 0, 0 };

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += convert_int4(in(xf, yf));
                }
            }
            output() = convert_uchar4(1.0f/(float)(size_x*size_y)*convert_float4(sum) + OFFSET_CV);
        }
        #endif
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    double time0, time1, dt, min_dt;
    const int width = WIDTH;
    const int height = HEIGHT;
    const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    const int offset_x = size_x >> 1;
    const int offset_y = size_y >> 1;

    // only filter kernel sizes 3x3 and 5x5 implemented
    if (size_x != size_y && (size_x != 3 || size_x != 5)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // domain for box filter
    #ifdef CONST_MASK
    const
    #endif
    uchar domain[SIZE_Y][SIZE_X] = {
        #if SIZE_X == 3
        { 1, 1, 1 },
        { 1, 1, 1 },
        { 1, 1, 1 }
        #endif
        #if SIZE_X == 5
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 }
        #endif
    };

    // host memory for image of width x height pixels
    uchar4 *input = new uchar4[width*height];
    uchar4 *reference_in = new uchar4[width*height];
    uchar4 *reference_out = new uchar4[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            uchar4 val;
            val.x = (y*width + x + 1) % 256;
            val.y = (y*width + x + 2) % 256;
            val.z = (y*width + x + 3) % 256;
            val.w = (y*width + x + 4) % 256;
            input[y*width + x] = val;
            reference_in[y*width + x] = val;
            reference_out[y*width + x] = (uchar4){ 0, 0, 0, 0 };
        }
    }


    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar4> out(width, height);

    // define Domain for box filter
    Domain dom(domain);

    // use undefined boundary handling to access image pixels beyond region
    // defined by Accessor
    BoundaryCondition<uchar4> bound(in, size_x, size_y, Boundary::UNDEFINED);
    Accessor<uchar4> acc(bound, width-2*offset_x, height-2*offset_y, offset_x, offset_y);

    IterationSpace<uchar4> iter(out, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    BoxFilter filter(iter, acc, dom, size_x, size_y);

    std::cerr << "Calculating Hipacc box filter ..." << std::endl;
    float timing = 0.0f;

    filter.execute();
    timing = hipacc_last_kernel_timing();

    // get pointer to result data
    uchar4 *output = out.data();

    std::cerr << "Hipacc: " << timing << " ms, " << ((width-2*offset_x)*(height-2*offset_y)/timing)/1000 << " Mpixel/s" << std::endl;


    #ifdef OpenCV
    // OpenCV uses NPP library for filtering
    // image: 4096x4096
    // kernel size: 3x3
    // offset 3x3 shifted by 1 -> 1x1
    // output: 4096x4096 - 3x3 -> 4093x4093; start: 1,1; end: 4094,4094
    //
    // image: 4096x4096
    // kernel size: 4x4
    // offset 4x4 shifted by 1 -> 2x2
    // output: 4096x4096 - 4x4 -> 4092x4092; start: 2,2; end: 4094,4094
    #ifdef CPU
    std::cerr << std::endl << "Calculating OpenCV blur filter on the CPU ..." << std::endl;
    #else
    std::cerr << std::endl << "Calculating OpenCV blur filter on the GPU ..." << std::endl;
    #endif


    cv::Mat cv_data_in(height, width, CV_8UC4, input);
    cv::Mat cv_data_out(height, width, CV_8UC4, cv::Scalar(0));
    cv::Size ksize(size_x, size_y);

    #ifdef CPU
    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        cv::blur(cv_data_in, cv_data_out, ksize);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    #else
    cv::gpu::GpuMat gpu_in, gpu_out;
    gpu_in.upload(cv_data_in);

    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        cv::gpu::boxFilter(gpu_in, gpu_out, -1, ksize);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }

    gpu_out.download(cv_data_out);
    #endif
    std::cerr << "OpenCV: " << min_dt << " ms, " << ((width-size_x)*(height-size_y)/min_dt)/1000 << " Mpixel/s" << std::endl;

    // get pointer to result data
    output = (uchar4 *)cv_data_out.data;
    #endif


    std::cerr << std::endl << "Calculating reference ..." << std::endl;
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        box_filter(reference_in, reference_out, size_x, size_y, width, height);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    std::cerr << "Reference: " << min_dt << " ms, " << ((width-2*offset_x)*(height-2*offset_y)/min_dt)/1000 << " Mpixel/s" << std::endl;

    std::cerr << std::endl << "Comparing results ..." << std::endl;
    #ifdef OpenCV
    int upper_y = height-size_y+offset_y;
    int upper_x = width-size_x+offset_x;
    #else
    int upper_y = height-offset_y;
    int upper_x = width-offset_x;
    #endif
    // compare results
    for (int y=offset_y; y<upper_y; y++) {
        for (int x=offset_x; x<upper_x; x++) {
            if (reference_out[y*width + x].x != output[y*width + x].x ||
                reference_out[y*width + x].y != output[y*width + x].y ||
                reference_out[y*width + x].z != output[y*width + x].z ||
                reference_out[y*width + x].w != output[y*width + x].w) {
                std::cerr << "Test FAILED, at (" << x << "," << y << "): ("
                          << (int)reference_out[y*width + x].x << ","
                          << (int)reference_out[y*width + x].y << ","
                          << (int)reference_out[y*width + x].z << ","
                          << (int)reference_out[y*width + x].w << ") vs. ("
                          << (int)output[y*width + x].x << ","
                          << (int)output[y*width + x].y << ","
                          << (int)output[y*width + x].z << ","
                          << (int)output[y*width + x].w << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << "Test PASSED" << std::endl;

    // memory cleanup
    delete[] input;
    delete[] reference_in;
    delete[] reference_out;

    return EXIT_SUCCESS;
}

