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
//#define SIZE_X 5
//#define SIZE_Y 5
//#define WIDTH 4096
//#define HEIGHT 4096
//#define YORDER
#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF
#define NO_SEP
#define INFER_DOMAIN
//#define ARRAY_DOMAIN
//#define CONST_DOMAIN

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Sobel filter reference
void sobel_filter(uchar *in, short *out, int *filter, int size_x, int size_y,
        int width, int height) {
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
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x] *
                           in[(y+yf)*width + x + xf];
                }
            }
            out[y*width + x] = sum;
        }
    }
}
void sobel_filter_row(uchar *in, short *out, int *filter, int size_x, int width,
        int height) {
    int anchor_x = size_x >> 1;
    #ifdef OpenCV
    int upper_x = width-size_x+anchor_x;
    #else
    int upper_x = width-anchor_x;
    #endif

    for (int y=0; y<height; ++y) {
        //for (int x=0; x<anchor_x; x++) out[y*width + x] = in[y*width + x];
        for (int x=anchor_x; x<upper_x; ++x) {
            int sum = 0;

            for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                sum += filter[xf+anchor_x] * in[(y)*width + x + xf];
            }
            out[y*width + x] = sum;
        }
        //for (int x=upper_x; x<width; x++) out[y*width + x] = in[y*width + x];
    }
}
void sobel_filter_column(short *in, short *out, int *filter, int size_y,
        int width, int height) {
    int anchor_y = size_y >> 1;
    #ifdef OpenCV
    int upper_y = height-size_y+anchor_y;
    #else
    int upper_y = height-anchor_y;
    #endif

    //for (int y=0; y<anchor_y; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = (uchar) in[y*width + x];
    //    }
    //}
    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=0; x<width; ++x) {
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                sum += filter[yf + anchor_y] * in[(y + yf)*width + x];
            }
            out[y*width + x] = sum;
        }
    }
    //for (int y=upper_y; y<height; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = (uchar) in[y*width + x];
    //    }
    //}
}


// Sobel filter in HIPAcc
#ifdef NO_SEP
class SobelFilterMask : public Kernel<short> {
    private:
        Accessor<uchar> &input;
        Domain &dom;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMask(IterationSpace<short> &iter, Accessor<uchar> &input,
                Domain &dom, Mask<int> &mask, const int size) :
            Kernel(iter),
            input(input),
            dom(dom),
            mask(mask),
            size(size)
        { addAccessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = (short)(reduce(dom, HipaccSUM, [&] () -> int {
                    return mask(dom) * input(dom);
                    }));
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            int sum = 0;

            for (int yf = -anchor; yf<=anchor; yf++) {
                for (int xf = -anchor; xf<=anchor; xf++) {
                    sum += mask(xf, yf)*input(xf, yf);
                }
            }

            output() = (short) sum;
        }
        #endif
};
#else
class SobelFilterMaskRow : public Kernel<short> {
    private:
        Accessor<uchar> &input;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMaskRow(IterationSpace<short> &iter, Accessor<uchar> &input,
                Mask<int> &mask, const int size):
            Kernel(iter),
            input(input),
            mask(mask),
            size(size)
        { addAccessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = (short)(convolve(mask, HipaccSUM, [&] () -> int {
                    return mask() * input(mask);
                    }));
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            int sum = 0;

            for (int xf = -anchor; xf<=anchor; xf++) {
                sum += mask(xf, 0)*input(xf, 0);
            }

            output() = (short) sum;
        }
        #endif
};
class SobelFilterMaskColumn : public Kernel<short> {
    private:
        Accessor<short> &input;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMaskColumn(IterationSpace<short> &iter, Accessor<short>
                &input, Mask<int> &mask, const int size):
            Kernel(iter),
            input(input),
            mask(mask),
            size(size)
        { addAccessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = (short)(convolve(mask, HipaccSUM, [&] () -> int {
                    return mask() * input(mask);
                    }));
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            int sum = 0;

            for (int yf = -anchor; yf<=anchor; yf++) {
                sum += mask(0, yf)*input(0, yf);
            }

            output() = (short) sum;
        }
        #endif
};
#endif


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

    // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
    if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 3x3, 5x5, and 7x7!\n");
        exit(EXIT_FAILURE);
    }

    // filter coefficients
    #ifdef YORDER
    #ifdef ARRAY_DOMAIN
    #ifdef CONST_DOMAIN
    const
    #endif
    uchar dom[SIZE_Y][SIZE_X] = {
        #if SIZE_X==3
         { 1,  1,  1 },
         { 0,  0,  0 },
         { 1,  1,  1 }
        #endif
        #if SIZE_X==5
         { 1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1 },
         { 0,  0,  0,  0,  0 },
         { 1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1 }
        #endif
        #if SIZE_X==7
         { 1,  1,  1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1,  1,  1 },
         { 0,  0,  0,  0,  0,  0,  0 },
         { 1,  1,  1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1,  1,  1 },
         { 1,  1,  1,  1,  1,  1,  1 }
        #endif
    };
    #endif
    #ifdef CONST_MASK
    const
    #endif
    int mask[SIZE_Y][SIZE_X] = {
        #if SIZE_X==3
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
        #endif
        #if SIZE_X==5
        { -1, -4, -6,  -4, -1 },
        { -2, -8, -12, -8, -2 },
        {  0,  0,  0,   0,  0 },
        {  2,  8,  12,  8,  2 },
        {  1,  4,  6,   4,  1 }
        #endif
        #if SIZE_X==7
        { -1, -6,  -15, -20,  -15, -6,  -1 },
        { -4, -24, -60, -80,  -60, -24, -4 },
        { -5, -30, -75, -100, -75, -30, -5 },
        {  0,  0,   0,   0,    0,   0,   0 },
        {  5,  30,  75,  100,  75,  30,  5 },
        {  4,  24,  60,  80,   60,  24,  4 },
        {  1,  6,   15,  20,   15,  6,   1 }
        #endif
    };
    #ifdef CONST_MASK
    const
    #endif
    int mask_x[1][SIZE_X] = {
        #if SIZE_X==3
        { 1, 2, 1 }
        #endif
        #if SIZE_X==5
        { 1, 4, 6, 4, 1 }
        #endif
        #if SIZE_X==7
        { 1, 6, 15, 20, 15, 6, 1 }
        #endif
    };
    #ifdef CONST_MASK
    const
    #endif
    int mask_y[SIZE_Y][1] = {
        #if SIZE_Y==3
        { -1 }, { 0 }, { +1 },
        #endif
        #if SIZE_Y==5
        { -1 }, { -2 }, { 0 }, { +2 }, { +1 },
        #endif
        #if SIZE_Y==7
        { -1 }, { -4 }, { -5 }, { 0 }, { 5 }, { 4 }, { 1 },
        #endif
    };
    #else
    #ifdef ARRAY_DOMAIN
    #ifdef CONST_DOMAIN
    const
    #endif
    uchar dom[size_y][size_x] = {
        #if SIZE_X==3
         { 1,  0,  1 },
         { 1,  0,  1 },
         { 1,  0,  1 }
        #endif
        #if SIZE_X==5
         { 1,  1,  0,  1,  1 },
         { 1,  1,  0,  1,  1 },
         { 1,  1,  0,  1,  1 },
         { 1,  1,  0,  1,  1 },
         { 1,  1,  0,  1,  1 }
        #endif
        #if SIZE_X==7
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 },
         { 1,  1,  1,  0,  1,  1,  1 }
        #endif
    };
    #endif
    #ifdef CONST_MASK
    const
    #endif
    int mask[SIZE_Y][SIZE_X] = {
        #if SIZE_X==3
        { -1, 0,  1 },
        { -2, 0,  2 },
        { -1, 0,  1 }
        #endif
        #if SIZE_X==5
        { -1,  -2, 0,  2, 1 },
        { -4,  -8, 0,  8, 4 },
        { -6, -12, 0, 12, 6 },
        { -4,  -8, 0,  8, 4 },
        { -1,  -2, 0,  2, 1 }
        #endif
        #if SIZE_X==7
        {  -1,  -4,   -5, 0,   5,  4,  1 },
        {  -6, -24,  -30, 0,  30, 24,  6 },
        { -15, -60,  -75, 0,  75, 60, 15 },
        { -20, -80, -100, 0, 100, 80, 20 },
        { -15, -60,  -75, 0,  75, 60, 15 },
        {  -6, -24,  -30, 0,  30, 24,  6 },
        {  -1,  -4,   -5, 0,   5,  4,  1 }
        #endif
    };
    #ifdef CONST_MASK
    const
    #endif
    int mask_x[1][SIZE_X] = {
        #if SIZE_X==3
        { -1, 0, +1 }
        #endif
        #if SIZE_X==5
        { -1, -2, 0, +2, +1 }
        #endif
        #if SIZE_X==7
        { -1, -4, -5, 0, 5, 4, 1 }
        #endif
    };
    #ifdef CONST_MASK
    const
    #endif
    int mask_y[SIZE_Y][1] = {
        #if SIZE_X==3
        { 1 }, { 2 }, { 1 },
        #endif
        #if SIZE_X==5
        { 1 }, { 4 }, { 6 }, { 4 }, { 1 },
        #endif
        #if SIZE_X==7
        { 1 }, { 6 }, { 15 }, { 20 }, { 15 }, { 6 }, { 1 },
        #endif
    };
    #endif

    // host memory for image of width x height pixels
    uchar *input = (uchar *)malloc(sizeof(uchar)*width*height);
    uchar *reference_in = (uchar *)malloc(sizeof(uchar)*width*height);
    short *reference_out = (short *)malloc(sizeof(short)*width*height);
    short *reference_tmp = (short *)malloc(sizeof(short)*width*height);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            uchar val = rand()%256;
            input[y*width + x] = val;
            reference_in[y*width + x] = input[y*width + x];
            reference_out[y*width + x] = 0;
            reference_tmp[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<uchar> IN(width, height);
    Image<short> OUT(width, height);
    Image<short> TMP(width, height);

    // filter mask
    Mask<int> M(mask);
    Mask<int> MX(mask_x);
    Mask<int> MY(mask_y);

    // filter domain
    #ifdef INFER_DOMAIN
    Domain D(M);
    #else
    #ifdef ARRAY_DOMAIN
    Domain D(dom);
    #else
    Domain D(size_x, size_y);
    D(0, 0) = 0;
    #ifdef YORDER
    D(-1, 0) = 0; D(1, 0) = 0;
    #if SIZE_X>3
    D(-2, 0) = 0; D(2, 0) = 0;
    #if SIZE_X>5
    D(-3, 0) = 0; D(3, 0) = 0;
    #endif
    #endif
    #else
    D(0, -1) = 0; D(0, 1) = 0;
    #if SIZE_Y>3
    D(0, -2) = 0; D(0, 2) = 0;
    #if SIZE_Y>5
    D(0, -3) = 0; D(0, 3) = 0;
    #endif
    #endif
    #endif
    #endif
    #endif

    IterationSpace<short> IsOut(OUT);
    IterationSpace<short> IsTmp(TMP);

    IN = input;


    #ifndef OpenCV
    fprintf(stderr, "Calculating Sobel filter ...\n");
    float timing = 0.0f;

    // BOUNDARY_UNDEFINED
    #ifdef RUN_UNDEF
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInUndef2(IN, M, BOUNDARY_UNDEFINED);
    Accessor<uchar> AccInUndef2(BcInUndef2);
    SobelFilterMask SFU(IsOut, AccInUndef2, D, M, size_x);

    SFU.execute();
    timing = hipaccGetLastKernelTiming();
    #else
    BoundaryCondition<uchar> BcInUndef(IN, MX, BOUNDARY_UNDEFINED);
    Accessor<uchar> AccInUndef(BcInUndef);
    SobelFilterMaskRow SFRU(IsTmp, AccInUndef, MX, size_x);

    BoundaryCondition<short> BcTmpUndef(TMP, MY, BOUNDARY_UNDEFINED);
    Accessor<short> AccTmpUndef(BcTmpUndef);
    SobelFilterMaskColumn SFCU(IsOut, AccTmpUndef, MY, size_y);

    SFRU.execute();
    timing = hipaccGetLastKernelTiming();
    SFCU.execute();
    timing += hipaccGetLastKernelTiming();
    #endif
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (UNDEFINED): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_CLAMP
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInClamp2(IN, M, BOUNDARY_CLAMP);
    Accessor<uchar> AccInClamp2(BcInClamp2);
    SobelFilterMask SFC(IsOut, AccInClamp2, D, M, size_x);

    SFC.execute();
    timing = hipaccGetLastKernelTiming();
    #else
    BoundaryCondition<uchar> BcInClamp(IN, MX, BOUNDARY_CLAMP);
    Accessor<uchar> AccInClamp(BcInClamp);
    SobelFilterMaskRow SFRC(IsTmp, AccInClamp, MX, size_x);

    BoundaryCondition<short> BcTmpClamp(TMP, MY, BOUNDARY_CLAMP);
    Accessor<short> AccTmpClamp(BcTmpClamp);
    SobelFilterMaskColumn SFCC(IsOut, AccTmpClamp, MY, size_y);

    SFRC.execute();
    timing = hipaccGetLastKernelTiming();
    SFCC.execute();
    timing += hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CLAMP): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_REPEAT
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInRepeat2(IN, M, BOUNDARY_REPEAT);
    Accessor<uchar> AccInRepeat2(BcInRepeat2);
    SobelFilterMask SFR(IsOut, AccInRepeat2, D, M, size_x);

    SFR.execute();
    timing = hipaccGetLastKernelTiming();
    #else
    BoundaryCondition<uchar> BcInRepeat(IN, MX, BOUNDARY_REPEAT);
    Accessor<uchar> AccInRepeat(BcInRepeat);
    SobelFilterMaskRow SFRR(IsTmp, AccInRepeat, MX, size_x);

    BoundaryCondition<short> BcTmpRepeat(TMP, MY, BOUNDARY_REPEAT);
    Accessor<short> AccTmpRepeat(BcTmpRepeat);
    SobelFilterMaskColumn SFCR(IsOut, AccTmpRepeat, MY, size_y);

    SFRR.execute();
    timing = hipaccGetLastKernelTiming();
    SFCR.execute();
    timing += hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (REPEAT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_MIRROR
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInMirror2(IN, M, BOUNDARY_MIRROR);
    Accessor<uchar> AccInMirror2(BcInMirror2);
    SobelFilterMask SFM(IsOut, AccInMirror2, D, M, size_x);

    SFM.execute();
    timing = hipaccGetLastKernelTiming();
    #else
    BoundaryCondition<uchar> BcInMirror(IN, MX, BOUNDARY_MIRROR);
    Accessor<uchar> AccInMirror(BcInMirror);
    SobelFilterMaskRow SFRM(IsTmp, AccInMirror, MX, size_x);

    BoundaryCondition<short> BcTmpMirror(TMP, MY, BOUNDARY_MIRROR);
    Accessor<short> AccTmpMirror(BcTmpMirror);
    SobelFilterMaskColumn SFCM(IsOut, AccTmpMirror, MY, size_y);

    SFRM.execute();
    timing = hipaccGetLastKernelTiming();
    SFCM.execute();
    timing += hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (MIRROR): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // BOUNDARY_CONSTANT
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInConst2(IN, M, BOUNDARY_CONSTANT, '1');
    Accessor<uchar> AccInConst2(BcInConst2);
    SobelFilterMask SFConst(IsOut, AccInConst2, D, M, size_x);

    SFConst.execute();
    timing = hipaccGetLastKernelTiming();
    #else
    BoundaryCondition<uchar> BcInConst(IN, MX, BOUNDARY_CONSTANT, '1');
    Accessor<uchar> AccInConst(BcInConst);
    SobelFilterMaskRow SFRConst(IsTmp, AccInConst, MX, size_x);

    BoundaryCondition<short> BcTmpConst(TMP, MY, BOUNDARY_CONSTANT, 1);
    Accessor<short> AccTmpConst(BcTmpConst);
    SobelFilterMaskColumn SFCConst(IsOut, AccTmpConst, MY, size_y);

    SFRConst.execute();
    timing = hipaccGetLastKernelTiming();
    SFCConst.execute();
    timing += hipaccGetLastKernelTiming();
    #endif
    timings.push_back(timing);
    fprintf(stderr, "HIPACC (CONSTANT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // get pointer to result data
    short *output = OUT.getData();
    #endif



    #ifdef OpenCV
    #ifdef CPU
    fprintf(stderr, "\nCalculating OpenCV Sobel filter on the CPU ...\n");
    #else
    fprintf(stderr, "\nCalculating OpenCV Sobel filter on the GPU ...\n");
    #endif


    cv::Mat cv_data_in(height, width, CV_8UC1, input);
    cv::Mat cv_data_out(height, width, CV_16SC1, cv::Scalar(0));
    int ddepth = CV_16S;
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

            #ifdef YORDER
            cv::Sobel(cv_data_in, cv_data_out, ddepth, 0, 1, size_y, scale, delta, brd_type);
            #else
            cv::Sobel(cv_data_in, cv_data_out, ddepth, 1, 0, size_x, scale, delta, brd_type);
            #endif

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

            #ifdef YORDER
            cv::gpu::Sobel(gpu_in, gpu_out, -1, 0, 1, size_y, scale, brd_type);
            #else
            cv::gpu::Sobel(gpu_in, gpu_out, -1, 1, 0, size_x, scale, brd_type);
            #endif

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

    // get pointer to result data
    short *output = (short *)cv_data_out.data;
    #endif

    // print statistics
    for (std::vector<float>::const_iterator it = timings.begin();
         it != timings.end(); ++it) {
        fprintf(stderr, "\t%.3f", *it);
    }
    fprintf(stderr, "\n\n");


    fprintf(stderr, "\nCalculating reference ...\n");
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        #ifdef NO_SEP
        sobel_filter(reference_in, reference_out, (int *)mask, size_x, size_y, width, height);
        #else
        sobel_filter_row(reference_in, reference_tmp, (int *)mask_x, size_x, width, height);
        sobel_filter_column(reference_tmp, reference_out, (int *)mask_y, size_y, width, height);
        #endif

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", min_dt, (width*height/min_dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
    #ifdef OpenCV
    #ifndef CPU
    fprintf(stderr, "\nWarning: OpenCV implementation on the GPU is currently broken (wrong results) ...\n");
    #endif
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
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x,
                        y, reference_out[y*width + x], output[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(reference_in);
    free(reference_tmp);
    free(reference_out);

    return EXIT_SUCCESS;
}

