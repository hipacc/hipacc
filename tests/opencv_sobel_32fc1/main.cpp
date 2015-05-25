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

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

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
//#define YORDER
#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF
//#define NO_SEP

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Sobel filter reference
void sobel_filter(float *in, float *out, int *filter, int size_x, int size_y,
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
void sobel_filter_row(float *in, float *out, int *filter, int size_x, int width,
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
void sobel_filter_column(float *in, float *out, int *filter, int size_y,
        int width, int height) {
    int anchor_y = size_y >> 1;
    #ifdef OpenCV
    int upper_y = height-size_y+anchor_y;
    #else
    int upper_y = height-anchor_y;
    #endif

    //for (int y=0; y<anchor_y; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = in[y*width + x];
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
    //        out[y*width + x] = in[y*width + x];
    //    }
    //}
}


// Sobel filter in Hipacc
#ifdef NO_SEP
class SobelFilterMask : public Kernel<float> {
    private:
        Accessor<float> &input;
        Domain &dom;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMask(IterationSpace<float> &iter, Accessor<float> &input,
                Domain &dom, Mask<int> &mask, const int size) :
            Kernel(iter),
            input(input),
            dom(dom),
            mask(mask),
            size(size)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = reduce(dom, Reduce::SUM, [&] () -> float {
                    return mask(dom) * input(dom);
                    });
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            float sum = 0;

            for (int yf = -anchor; yf<=anchor; yf++) {
                for (int xf = -anchor; xf<=anchor; xf++) {
                    sum += mask(xf, yf)*input(xf, yf);
                }
            }

            output() = sum;
        }
        #endif
};
#else
class SobelFilterMaskRow : public Kernel<float> {
    private:
        Accessor<float> &input;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMaskRow(IterationSpace<float> &iter, Accessor<float> &input,
                Mask<int> &mask, const int size):
            Kernel(iter),
            input(input),
            mask(mask),
            size(size)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    });
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            float sum = 0;

            for (int xf = -anchor; xf<=anchor; xf++) {
                sum += mask(xf, 0)*input(xf, 0);
            }

            output() = sum;
        }
        #endif
};
class SobelFilterMaskColumn : public Kernel<float> {
    private:
        Accessor<float> &input;
        Mask<int> &mask;
        const int size;

    public:
        SobelFilterMaskColumn(IterationSpace<float> &iter, Accessor<float>
                &input, Mask<int> &mask, const int size):
            Kernel(iter),
            input(input),
            mask(mask),
            size(size)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    });
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            float sum = 0;

            for (int yf = -anchor; yf<=anchor; yf++) {
                sum += mask(0, yf)*input(0, yf);
            }

            output() = sum;
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
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3, 5x5, and 7x7!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // filter coefficients
    #ifdef YORDER
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
    float *input = new float[width*height];
    float *reference_in = new float[width*height];
    float *reference_out = new float[width*height];
    float *reference_tmp = new float[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = rand()%256;
            reference_in[y*width + x] = input[y*width + x];
            reference_out[y*width + x] = 0;
            reference_tmp[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<float> IN(width, height, input);
    Image<float> OUT(width, height);
    Image<float> TMP(width, height);

    // filter mask
    Mask<int> M(mask);
    Mask<int> MX(mask_x);
    Mask<int> MY(mask_y);

    // filter domain
    Domain D(M);

    IterationSpace<float> IsOut(OUT);
    IterationSpace<float> IsTmp(TMP);


    #ifndef OpenCV
    std::cerr << "Calculating Hipacc Sobel filter ..." << std::endl;
    float timing = 0.0f;

    // UNDEFINED
    #ifdef RUN_UNDEF
    #ifdef NO_SEP
    BoundaryCondition<float> BcInUndef2(IN, M, Boundary::UNDEFINED);
    Accessor<float> AccInUndef2(BcInUndef2);
    SobelFilterMask SFU(IsOut, AccInUndef2, D, M, size_x);

    SFU.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<float> BcInUndef(IN, MX, Boundary::UNDEFINED);
    Accessor<float> AccInUndef(BcInUndef);
    SobelFilterMaskRow SFRU(IsTmp, AccInUndef, MX, size_x);

    BoundaryCondition<float> BcTmpUndef(TMP, MY, Boundary::UNDEFINED);
    Accessor<float> AccTmpUndef(BcTmpUndef);
    SobelFilterMaskColumn SFCU(IsOut, AccTmpUndef, MY, size_y);

    SFRU.execute();
    timing = hipacc_last_kernel_timing();
    SFCU.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (UNDEFINED): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CLAMP
    #ifdef NO_SEP
    BoundaryCondition<float> BcInClamp2(IN, M, Boundary::CLAMP);
    Accessor<float> AccInClamp2(BcInClamp2);
    SobelFilterMask SFC(IsOut, AccInClamp2, D, M, size_x);

    SFC.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<float> BcInClamp(IN, MX, Boundary::CLAMP);
    Accessor<float> AccInClamp(BcInClamp);
    SobelFilterMaskRow SFRC(IsTmp, AccInClamp, MX, size_x);

    BoundaryCondition<float> BcTmpClamp(TMP, MY, Boundary::CLAMP);
    Accessor<float> AccTmpClamp(BcTmpClamp);
    SobelFilterMaskColumn SFCC(IsOut, AccTmpClamp, MY, size_y);

    SFRC.execute();
    timing = hipacc_last_kernel_timing();
    SFCC.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // REPEAT
    #ifdef NO_SEP
    BoundaryCondition<float> BcInRepeat2(IN, M, Boundary::REPEAT);
    Accessor<float> AccInRepeat2(BcInRepeat2);
    SobelFilterMask SFR(IsOut, AccInRepeat2, D, M, size_x);

    SFR.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<float> BcInRepeat(IN, MX, Boundary::REPEAT);
    Accessor<float> AccInRepeat(BcInRepeat);
    SobelFilterMaskRow SFRR(IsTmp, AccInRepeat, MX, size_x);

    BoundaryCondition<float> BcTmpRepeat(TMP, MY, Boundary::REPEAT);
    Accessor<float> AccTmpRepeat(BcTmpRepeat);
    SobelFilterMaskColumn SFCR(IsOut, AccTmpRepeat, MY, size_y);

    SFRR.execute();
    timing = hipacc_last_kernel_timing();
    SFCR.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (REPEAT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // MIRROR
    #ifdef NO_SEP
    BoundaryCondition<float> BcInMirror2(IN, M, Boundary::MIRROR);
    Accessor<float> AccInMirror2(BcInMirror2);
    SobelFilterMask SFM(IsOut, AccInMirror2, D, M, size_x);

    SFM.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<float> BcInMirror(IN, MX, Boundary::MIRROR);
    Accessor<float> AccInMirror(BcInMirror);
    SobelFilterMaskRow SFRM(IsTmp, AccInMirror, MX, size_x);

    BoundaryCondition<float> BcTmpMirror(TMP, MY, Boundary::MIRROR);
    Accessor<float> AccTmpMirror(BcTmpMirror);
    SobelFilterMaskColumn SFCM(IsOut, AccTmpMirror, MY, size_y);

    SFRM.execute();
    timing = hipacc_last_kernel_timing();
    SFCM.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (MIRROR): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CONSTANT
    #ifdef NO_SEP
    BoundaryCondition<float> BcInConst2(IN, M, Boundary::CONSTANT, '1');
    Accessor<float> AccInConst2(BcInConst2);
    SobelFilterMask SFConst(IsOut, AccInConst2, D, M, size_x);

    SFConst.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<float> BcInConst(IN, MX, Boundary::CONSTANT, '1');
    Accessor<float> AccInConst(BcInConst);
    SobelFilterMaskRow SFRConst(IsTmp, AccInConst, MX, size_x);

    BoundaryCondition<float> BcTmpConst(TMP, MY, Boundary::CONSTANT, 1);
    Accessor<float> AccTmpConst(BcTmpConst);
    SobelFilterMaskColumn SFCConst(IsOut, AccTmpConst, MY, size_y);

    SFRConst.execute();
    timing = hipacc_last_kernel_timing();
    SFCConst.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (CONSTANT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // get pointer to result data
    float *output = OUT.data();
    #endif



    #ifdef OpenCV
    #ifdef CPU
    std::cerr << std::endl << "Calculating OpenCV Sobel filter on the CPU ..." << std::endl;
    #else
    std::cerr << std::endl << "Calculating OpenCV Sobel filter on the GPU ..." << std::endl;
    #endif


    cv::Mat cv_data_in(height, width, CV_32FC1, input);
    cv::Mat cv_data_out(height, width, CV_32FC1, cv::Scalar(0));
    int ddepth = CV_32F;
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

        std::cerr << "OpenCV (";
        switch (brd_type) {
            case IPL_BORDER_CONSTANT:    std::cerr << "CONSTANT";   break;
            case IPL_BORDER_REPLICATE:   std::cerr << "CLAMP";      break;
            case IPL_BORDER_REFLECT:     std::cerr << "MIRROR";     break;
            case IPL_BORDER_WRAP:        std::cerr << "REPEAT";     break;
            case IPL_BORDER_REFLECT_101: std::cerr << "MIRROR_101"; break;
            default: break;
        }
        std::cerr << "): " << min_dt << " ms, " << (width*height/min_dt)/1000 << " Mpixel/s" << std::endl;
        timings.push_back(min_dt);
    }

    // get pointer to result data
    float *output = (float *)cv_data_out.data;
    #endif

    // print statistics
    for (std::vector<float>::const_iterator it = timings.begin(); it != timings.end(); ++it) {
        std::cerr << "\t" << *it;
    }
    std::cerr << std::endl << std::endl;


    std::cerr << "Calculating reference ..." << std::endl;
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
    std::cerr << "Reference: " << min_dt << " ms, " << (width*height/min_dt)/1000 << " Mpixel/s" << std::endl;

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
            if (reference_out[y*width + x] != output[y*width + x]) {
                std::cerr << "Test FAILED, at (" << x << "," << y << "): "
                          << reference_out[y*width + x] << " vs. "
                          << output[y*width + x] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << "Test PASSED" << std::endl;

    // memory cleanup
    delete[] input;
    delete[] reference_in;
    delete[] reference_tmp;
    delete[] reference_out;

    return EXIT_SUCCESS;
}

