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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <vector>

//#define CPU
#ifdef OpenCV
#include "opencv2/opencv.hpp"
#ifndef CPU
#include "opencv2/gpu/gpu.hpp"
#endif
#endif

#include "hipacc.hpp"

// variables set by Makefile
#define SIZE_X 32
#define WIDTH 1048576
#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// FIR filter reference
void fir_filter(uchar4 *in, uchar4 *out, float *filter, int
        size_x, int width) {
    int anchor_x = size_x >> 1;

    for (int x=anchor_x; x<width-anchor_x; ++x) {
        float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };

        // size_x is even => -size_x/2 .. +size_x/2 - 1
        for (int xf = -anchor_x; xf<anchor_x; xf++) {
            sum += filter[xf + anchor_x]*convert_float4(in[x + xf]);
        }

        out[x] = convert_uchar4(sum);
    }
}


// FIR filter in HIPAcc
class FIRFilterMask : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &input;
        Mask<float> &mask;
        const int size_x;

    public:
        FIRFilterMask(IterationSpace<uchar4> &iter, Accessor<uchar4> &input,
                Mask<float> &mask, const int size_x) :
            Kernel(iter),
            input(input),
            mask(mask),
            size_x(size_x)
        { addAccessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };
            sum += convolve(mask, HipaccSUM, [&] () -> float4 {
                    return mask()*convert_float4(input(mask));
                    });

            output() = convert_uchar4(sum);
        }
        #else
        void kernel() {
            const int anchor_x = size_x >> 1;
            float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };

            // size_x is even => -size_x/2 .. +size_x/2 - 1
            for (int xf = -anchor_x; xf<anchor_x; xf++) {
                sum += mask(xf, 0)*convert_float4(input(xf, 0));
            }

            output() = convert_uchar4(sum);
        }
        #endif
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    double time0, time1, dt, min_dt;
    const int width = WIDTH;
    const int size_x = SIZE_X;
    const int offset_x = size_x >> 1;
    std::vector<float> timings;

    // filter coefficients
    #ifdef CONST_MASK
    // only filter kernel sizes 32, and 64 implemented
    if (size_x && !(size_x == 32 || size_x == 64)) {
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 32 and 64!\n");
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float filter_x[1][SIZE_X] = {
        #if SIZE_X == 32
        { 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f }
        #endif
        #if SIZE_X == 64
        { 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f }
        #endif
    };
    #else
    float filter_x[1][SIZE_X];

    for (int i=0; i < size_x; i++) {
        filter_x[0][i] = ((float)1)/SIZE_X;
    }
    #endif

    // input and output image of width pixels
    Image<uchar4> IN(width, 1);
    Image<uchar4> OUT(width, 1);

    // host memory for image of width pixels
    uchar4 *input = (uchar4 *)malloc(sizeof(uchar4)*width);
    uchar4 *reference_in = (uchar4 *)malloc(sizeof(uchar4)*width);
    uchar4 *reference_out = (uchar4 *)malloc(sizeof(uchar4)*width);

    // initialize data
    for (int x=0; x<width; ++x) {
        unsigned char val = x % 256;
        input[x] = (uchar4){ val, val, val, val };
        reference_in[x] = (uchar4){ val, val, val, val };
        reference_out[x] = (uchar4){ 0, 0, 0, 0 };
    }


    #ifndef OpenCV
    // filter mask
    Mask<float> MX(filter_x);

    IterationSpace<uchar4> IsOut(OUT);

    IN = input;

    fprintf(stderr, "Calculating HIPAcc FIR filter ...\n");
    float timing = 0.0f;

    // BOUNDARY_UNDEFINED
    #ifdef RUN_UNDEF
    BoundaryCondition<uchar4> BcInUndef2(IN, size_x, 1, BOUNDARY_UNDEFINED);
    Accessor<uchar4> AccInUndef2(BcInUndef2);
    FIRFilterMask FFU(IsOut, AccInUndef2, MX, size_x);

    FFU.execute();
    timing = hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (UNDEFINED): %.3f ms, %.3f Mpixel/s\n", timing, (width/timing)/1000);


    // BOUNDARY_CLAMP
    BoundaryCondition<uchar4> BcInClamp2(IN, size_x, 1, BOUNDARY_CLAMP);
    Accessor<uchar4> AccInClamp2(BcInClamp2);
    FIRFilterMask FFC(IsOut, AccInClamp2, MX, size_x);

    FFC.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CLAMP): %.3f ms, %.3f Mpixel/s\n", timing, (width/timing)/1000);


    // BOUNDARY_REPEAT
    BoundaryCondition<uchar4> BcInRepeat2(IN, size_x, 1, BOUNDARY_REPEAT);
    Accessor<uchar4> AccInRepeat2(BcInRepeat2);
    FIRFilterMask FFR(IsOut, AccInRepeat2, MX, size_x);

    FFR.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (REPEAT): %.3f ms, %.3f Mpixel/s\n", timing, (width/timing)/1000);


    // BOUNDARY_MIRROR
    BoundaryCondition<uchar4> BcInMirror2(IN, size_x, 1, BOUNDARY_MIRROR);
    Accessor<uchar4> AccInMirror2(BcInMirror2);
    FIRFilterMask FFM(IsOut, AccInMirror2, MX, size_x);

    FFM.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (MIRROR): %.3f ms, %.3f Mpixel/s\n", timing, (width/timing)/1000);


    // BOUNDARY_CONSTANT
    BoundaryCondition<uchar4> BcInConst2(IN, size_x, 1, BOUNDARY_CONSTANT, '1');
    Accessor<uchar4> AccInConst2(BcInConst2);
    FIRFilterMask FFConst(IsOut, AccInConst2, MX, size_x);

    FFConst.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CONSTANT): %.3f ms, %.3f Mpixel/s\n", timing, (width/timing)/1000);


    // get pointer to result data
    uchar4 *output = OUT.getData();
    #endif


    #ifdef OpenCV
    #ifdef CPU
    fprintf(stderr, "\nCalculating OpenCV FIR filter on the CPU ...\n");
    #else
    fprintf(stderr, "\nCalculating OpenCV FIR filter on the GPU ...\n");
    #endif


    cv::Mat cv_data_in(1, width, CV_8UC4, input);
    cv::Mat cv_data_out(1, width, CV_8UC4, cv::Scalar(0));

    cv::Mat kernel = cv::Mat::ones(size_x, 1, CV_32F ) / (float)(size_x);
    cv::Point anchor = cv::Point(-1,-1);
    double delta = 0;
    int ddepth = -1;

    for (int brd_type=0; brd_type<5; brd_type++) {
        #ifdef CPU
        if (brd_type==cv::BORDER_WRAP) {
            // BORDER_WRAP is not supported on the CPU by OpenCV
            timings.push_back(0.0f);
            continue;
        }
        min_dt = DBL_MAX;
        std::vector<double> times;
        for (int nt=0; nt<10; nt++) {
            time0 = time_ms();

            cv::filter2D(cv_data_in, cv_data_out, ddepth, kernel, anchor, delta, brd_type);

            time1 = time_ms();
            dt = time1 - time0;
            times.push_back(dt);
            if (dt < min_dt) min_dt = dt;
        }
        std::sort(times.begin(), times.end());
        min_dt = times.at(5);
        #else
        cv::gpu::GpuMat gpu_in, gpu_out;
        gpu_in.upload(cv_data_in);

        min_dt = DBL_MAX;
        std::vector<double> times;
        for (int nt=0; nt<10; nt++) {
            time0 = time_ms();

            cv::gpu::filter2D(gpu_in, gpu_out, ddepth, kernel, anchor, brd_type);

            time1 = time_ms();
            dt = time1 - time0;
            times.push_back(dt);
            if (dt < min_dt) min_dt = dt;
        }
        std::sort(times.begin(), times.end());
        min_dt = times.at(5);

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
        fprintf(stderr, "): %.3f ms, %.3f Mpixel/s\n", min_dt, (width/min_dt)/1000);
    }

    // get pointer to result data
    uchar4 *output = (uchar4 *)cv_data_out.data;
    #endif

    // print statistics
    fprintf(stderr, "HIPAcc: ");
    for (std::vector<float>::const_iterator it = timings.begin();
         it != timings.end(); ++it) {
        fprintf(stderr, "\t%.3f", *it);
    }
    fprintf(stderr, "\t");
    #ifdef CONST_MASK
    fprintf(stderr, "+ConstMask\t");
    #endif
    #ifdef USE_LAMBDA
    fprintf(stderr, "+Lambda\t");
    #endif
    fprintf(stderr, "\n\n");


    fprintf(stderr, "\nCalculating reference ...\n");
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        fir_filter(reference_in, reference_out, (float *)filter_x[0], size_x, width);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", min_dt, (width/min_dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
    #ifdef OpenCV
    int upper_x = width-size_x+offset_x;
    #else
    int upper_x = width-offset_x;
    #endif
    // compare results
    for (int x=offset_x; x<upper_x; x++) {
        if ((reference_out[x].x != output[x].x) &&
            (reference_out[x].y != output[x].y) &&
            (reference_out[x].z != output[x].z) &&
            (reference_out[x].w != output[x].w)) {
            fprintf(stderr, "Test FAILED, at (%d): (%hhu,%hhu,%hhu,%hhu) vs. (%hhu,%hhu,%hhu,%hhu)\n",
                    x,
                    reference_out[x].x,
                    reference_out[x].y,
                    reference_out[x].z,
                    reference_out[x].w,
                    output[x].x,
                    output[x].y,
                    output[x].z,
                    output[x].w);
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

