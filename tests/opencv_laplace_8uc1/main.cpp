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

#include <iostream>
#include <vector>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef OpenCV
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#include "hipacc.hpp"

// variables set by Makefile
//#define SIZE_X 3
//#define SIZE_Y 3
//#define WIDTH 4096
//#define HEIGHT 4096
//#define CPU
//#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Laplace filter reference
void laplace_filter(unsigned char *in, unsigned char *out, int *filter, int
        size, int width, int height) {
    const int size_x = size;
    const int size_y = size;
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
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]*in[(y+yf)*width + x + xf];
                }
            }

            sum = min(sum, 255);
            sum = max(sum, 0);
            out[y*width + x] = sum;
        }
    }
}


// Laplace filter in HIPAcc
class LaplaceFilter : public Kernel<unsigned char> {
    private:
        Accessor<unsigned char> &Input;
        Mask<int> &cMask;
        const int size;

    public:
        LaplaceFilter(IterationSpace<unsigned char> &IS, Accessor<unsigned char>
                &Input, Mask<int> &cMask, const int size) :
            Kernel(IS),
            Input(Input),
            cMask(cMask),
            size(size)
        { addAccessor(&Input); }

        #ifdef USE_LAMBDA
        void kernel() {
            int sum = convolve(cMask, HipaccSUM, [&] () -> int {
                    return cMask() * Input(cMask);
                    });
            sum = min(sum, 255);
            sum = max(sum, 0);
            output() = (unsigned char) (sum);
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            int sum = 0;

            for (int yf = -anchor; yf<=anchor; yf++) {
                for (int xf = -anchor; xf<=anchor; xf++) {
                    sum += cMask(xf, yf)*Input(xf, yf);
                }
            }

            sum = min(sum, 255);
            sum = max(sum, 0);
            output() = (unsigned char) (sum);
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
    std::vector<float> timings;
    float timing = 0.0f;

    // only filter kernel sizes 3x3 and 5x5 supported
    if (size_x != size_y || !(size_x == 3 || size_x == 5)) {
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!\n");
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    #ifdef CONST_MASK
    const
    #endif
    int mask[] = {
        #if SIZE_X==1
        0,  1,  0,
        1, -4,  1,
        0,  1,  0,
        #endif
        #if SIZE_X==3
        2,  0,  2,
        0, -8,  0,
        2,  0,  2,
        #endif
        #if SIZE_X==5
        1,   1,   1,   1,   1,
        1,   1,   1,   1,   1,
        1,   1, -24,   1,   1,
        1,   1,   1,   1,   1,
        1,   1,   1,   1,   1,
        #endif
    };

    // host memory for image of of width x height pixels
    unsigned char *host_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *host_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            reference_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            host_out[y*width + x] = 0;
            reference_out[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<unsigned char> IN(width, height);
    Image<unsigned char> OUT(width, height);

    // filter mask
    Mask<int> M(size_x, size_y);
    M = mask;

    IterationSpace<unsigned char> IsOut(OUT);

    IN = host_in;
    OUT = host_out;


#ifndef OpenCV
    fprintf(stderr, "Calculating Laplace filter ...\n");

    // BOUNDARY_UNDEFINED
    #ifdef RUN_UNDEF
    BoundaryCondition<unsigned char> BcInUndef(IN, size_x, BOUNDARY_UNDEFINED);
    Accessor<unsigned char> AccInUndef(BcInUndef);
    LaplaceFilter LFU(IsOut, AccInUndef, M, size_x);

    LFU.execute();
    timing = hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (UNDEFINED): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_CLAMP
    BoundaryCondition<unsigned char> BcInClamp(IN, size_x, BOUNDARY_CLAMP);
    Accessor<unsigned char> AccInClamp(BcInClamp);
    LaplaceFilter LFC(IsOut, AccInClamp, M, size_x);

    LFC.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CLAMP): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_REPEAT
    BoundaryCondition<unsigned char> BcInRepeat(IN, size_x, BOUNDARY_REPEAT);
    Accessor<unsigned char> AccInRepeat(BcInRepeat);
    LaplaceFilter LFR(IsOut, AccInRepeat, M, size_x);

    LFR.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (REPEAT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_MIRROR
    BoundaryCondition<unsigned char> BcInMirror(IN, size_x, BOUNDARY_MIRROR);
    Accessor<unsigned char> AccInMirror(BcInMirror);
    LaplaceFilter LFM(IsOut, AccInMirror, M, size_x);

    LFM.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (MIRROR): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_CONSTANT
    BoundaryCondition<unsigned char> BcInConst(IN, size_x, BOUNDARY_CONSTANT, '1');
    Accessor<unsigned char> AccInConst(BcInConst);
    LaplaceFilter LFConst(IsOut, AccInConst, M, size_x);

    LFConst.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CONSTANT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // get results
    host_out = OUT.getData();
#endif



#ifdef OpenCV
#ifdef CPU
    fprintf(stderr, "\nCalculating OpenCV Laplacian filter on the CPU ...\n");
#else
    fprintf(stderr, "\nCalculating OpenCV Laplacian filter on the GPU ...\n");
#endif


    cv::Mat cv_data_in(height, width, CV_8UC1, host_in);
    cv::Mat cv_data_out(height, width, CV_8UC1, host_out);
    int ddepth = CV_8U;
    double scale = 1.0f;
    double delta = 0.0f;

    for (int brd_type=0; brd_type<5; brd_type++) {
#ifdef CPU
        if (brd_type==cv::BORDER_WRAP) {
            // BORDER_WRAP is not supported on the CPU by OpenCV
            timings.push_back(0.0f);
            continue;
        }
        min_dt = DBL_MAX;
        for (int nt=0; nt<10; nt++) {
            time0 = time_ms();

            cv::Laplacian(cv_data_in, cv_data_out, ddepth, size_x, scale, delta, brd_type);

            time1 = time_ms();
            dt = time1 - time0;
            if (dt < min_dt) min_dt = dt;
        }
#else
        #if SIZE_X==5
        #error "OpenCV supports only 1x1 and 3x3 Laplace filters on the GPU!"
        #endif
        cv::gpu::GpuMat gpu_in, gpu_out;
        gpu_in.upload(cv_data_in);

        min_dt = DBL_MAX;
        for (int nt=0; nt<10; nt++) {
            time0 = time_ms();

            cv::gpu::Laplacian(gpu_in, gpu_out, -1, size_x, scale, brd_type);

            time1 = time_ms();
            dt = time1 - time0;
            if (dt < min_dt) min_dt = dt;
        }

        gpu_out.download(cv_data_out);
#endif

        fprintf(stderr, "OpenCV(");
        switch (brd_type) {
            case IPL_BORDER_CONSTANT:
                fprintf(stderr, "CONSTANT");
                break;
            case IPL_BORDER_REPLICATE:
                fprintf(stderr, "CLAMP");
                break;
            case IPL_BORDER_REFLECT:
                fprintf(stderr, "MIRROR");
                break;
            case IPL_BORDER_WRAP:
                fprintf(stderr, "REPEAT");
                break;
            case IPL_BORDER_REFLECT_101:
                fprintf(stderr, "MIRROR_101");
                break;
            default:
                break;
        }
        timings.push_back(min_dt);
        fprintf(stderr, "): %.3f ms, %.3f Mpixel/s\n", min_dt, (width*height/min_dt)/1000);
    }
#endif

    // print statistics
    for (unsigned int i=0; i<timings.size(); i++) {
        fprintf(stderr, "\t%.3f", timings.data()[i]);
    }
    fprintf(stderr, "\n\n");


    fprintf(stderr, "\nCalculating reference ...\n");
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        laplace_filter(reference_in, reference_out, (int *)mask, size_x, width, height);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", min_dt, (width*height/min_dt)/1000);

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
            if (reference_out[y*width + x] != host_out[y*width + x]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x,
                        y, reference_out[y*width + x], host_out[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(host_in);
    //free(host_out);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

