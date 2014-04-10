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

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//#define CPU
#ifdef OpenCV
#include "opencv2/opencv.hpp"
#ifndef CPU
#include "opencv2/gpu/gpu.hpp"
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

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Dilate filter reference
void dilate_filter(uchar *in, uchar *out, int size_x, int size_y, int width, int
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
            uchar max_val = 0;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    max_val = max(max_val, in[(y + yf)*width + x + xf]);
                }
            }
            out[y*width + x] = max_val;
        }
    }
}


// Kernel description in HIPAcc
class DilateFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &in;
        Domain &dom;
        int size_x, size_y;

    public:
        DilateFilter(IterationSpace<uchar> &iter, Accessor<uchar> &in, Domain
                &dom, int size_x, int size_y) :
            Kernel(iter),
            in(in),
            dom(dom),
            size_x(size_x),
            size_y(size_y)
        { addAccessor(&in); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = reduce(dom, HipaccMAX, [&] () -> uchar {
                    return in(dom);
                    });
        }
        #else
        void kernel() {
            int anchor_x = size_x >> 1;
            int anchor_y = size_y >> 1;
            uchar max_val = 0;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    max_val = max(max_val, in(xf, yf));
                }
            }

            output() = max_val;
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
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!\n");
        exit(EXIT_FAILURE);
    }

    // domain for dilate filter
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
    uchar *input = (uchar *)malloc(sizeof(uchar)*width*height);
    uchar *reference_in = (uchar *)malloc(sizeof(uchar)*width*height);
    uchar *reference_out = (uchar *)malloc(sizeof(uchar)*width*height);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = (uchar)(y*width + x) % 256;
            reference_in[y*width + x] = (uchar)(y*width + x) % 256;
            reference_out[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<uchar> in(width, height);
    Image<uchar> out(width, height);

    // define Domain for Erode filter
    Domain dom(domain);

    // use undefined boundary handling to access image pixels beyond region
    // defined by Accessor
    BoundaryCondition<uchar> bound(in, size_x, size_y, BOUNDARY_UNDEFINED);
    Accessor<uchar> acc(bound, width-2*offset_x, height-2*offset_y, offset_x, offset_y);

    IterationSpace<uchar> iter(out, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    DilateFilter filter(iter, acc, dom, size_x, size_y);

    in = input;

    fprintf(stderr, "Calculating HIPAcc Dilate filter ...\n");
    float timing = 0.0f;

    filter.execute();
    timing = hipaccGetLastKernelTiming();

    // get pointer to result data
    uchar *output = out.getData();

    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, ((width-2*offset_x)*(height-2*offset_y)/timing)/1000);


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
    fprintf(stderr, "\nCalculating OpenCV Dilate filter on the CPU ...\n");
    #else
    fprintf(stderr, "\nCalculating OpenCV Dilate filter on the GPU ...\n");
    #endif


    cv::Mat cv_data_in(height, width, CV_8UC1, input);
    cv::Mat cv_data_out(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat kernel(cv::Mat::ones(size_x, size_y, CV_8U));

    #ifdef CPU
    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        cv::dilate(cv_data_in, cv_data_out, kernel);

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

        cv::gpu::dilate(gpu_in, gpu_out, kernel);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }

    gpu_out.download(cv_data_out);
    #endif
    fprintf(stderr, "OpenCV: %.3f ms, %.3f Mpixel/s\n", min_dt, ((width-size_x)*(height-size_y)/min_dt)/1000);

    // get pointer to result data
    output = (uchar *)cv_data_out.data;
    #endif


    fprintf(stderr, "\nCalculating reference ...\n");
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        dilate_filter(reference_in, reference_out, size_x, size_y, width, height);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", min_dt, ((width-2*offset_x)*(height-2*offset_y)/min_dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
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
            if (reference_out[y*width + x] != output[y*width + x]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %hhu vs. %hhu\n", x,
                        y, reference_out[y*width + x], output[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

