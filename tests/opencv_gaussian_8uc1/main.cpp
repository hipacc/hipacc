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


// Gaussian blur filter reference
void gaussian_filter(uchar *in, uchar *out, float *filter, int size_x, int
        size_y, int width, int height) {
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
            float sum = 0.5f;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x] *
                           in[(y+yf)*width + x + xf];
                }
            }
            out[y*width + x] = (uchar) (sum);
        }
    }
}
void gaussian_filter_row(uchar *in, float *out, float *filter, int size_x, int
        width, int height) {
    int anchor_x = size_x >> 1;
    #ifdef OpenCV
    int upper_x = width-size_x+anchor_x;
    #else
    int upper_x = width-anchor_x;
    #endif

    for (int y=0; y<height; ++y) {
        //for (int x=0; x<anchor_x; x++) out[y*width + x] = in[y*width + x];
        for (int x=anchor_x; x<upper_x; ++x) {
            float sum = 0;

            for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                sum += filter[xf+anchor_x] * in[(y)*width + x + xf];
            }
            out[y*width + x] = sum;
        }
        //for (int x=upper_x; x<width; x++) out[y*width + x] = in[y*width + x];
    }
}
void gaussian_filter_column(float *in, uchar *out, float *filter, int size_y,
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
            float sum = 0.5f;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                sum += filter[yf + anchor_y] * in[(y + yf)*width + x];
            }
            out[y*width + x] = (uchar) (sum);
        }
    }
    //for (int y=upper_y; y<height; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = (uchar) in[y*width + x];
    //    }
    //}
}


// Gaussian blur filter in Hipacc
#ifdef NO_SEP
class GaussianBlurFilterMask : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<float> &mask;
        const int size_x, size_y;

    public:
        GaussianBlurFilterMask(IterationSpace<uchar> &iter, Accessor<uchar>
                &input, Mask<float> &mask, const int size_x, const int size_y) :
            Kernel(iter),
            input(input),
            mask(mask),
            size_x(size_x),
            size_y(size_y)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    }) + 0.5f);
        }
        #else
        void kernel() {
            const int anchor_x = size_x >> 1;
            const int anchor_y = size_y >> 1;
            float sum = 0.5f;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += mask(xf, yf)*input(xf, yf);
                }
            }

            output() = (uchar) sum;
        }
        #endif
};
#else
class GaussianBlurFilterMaskRow : public Kernel<float> {
    private:
        Accessor<uchar> &input;
        Mask<float> &mask;
        const int size;

    public:
        GaussianBlurFilterMaskRow(IterationSpace<float> &iter, Accessor<uchar>
                &input, Mask<float> &mask, const int size) :
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
class GaussianBlurFilterMaskColumn : public Kernel<uchar> {
    private:
        Accessor<float> &input;
        Mask<float> &mask;
        const int size;

    public:
        GaussianBlurFilterMaskColumn(IterationSpace<uchar> &iter,
                Accessor<float> &input, Mask<float> &mask, const int size) :
            Kernel(iter),
            input(input),
            mask(mask),
            size(size)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    }) + 0.5f);
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            float sum = 0.5f;

            for (int yf = -anchor; yf<=anchor; yf++) {
                sum += mask(0, yf)*input(0, yf);
            }

            output() = (uchar) (sum);
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
    const double sigma1 = ((size_x-1)*0.5 - 1)*0.3 + 0.8;
    const double sigma2 = ((size_y-1)*0.5 - 1)*0.3 + 0.8;

    // filter coefficients
    #ifdef CONST_MASK
    // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
    if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3, 5x5, and 7x7!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float filter_x[1][SIZE_X] = {
        #if SIZE_X == 3
        { 0.238994f, 0.522011f, 0.238994f }
        #endif
        #if SIZE_X == 5
        { 0.070766f, 0.244460f, 0.369546f, 0.244460f, 0.070766f }
        #endif
        #if SIZE_X == 7
        { 0.028995f, 0.103818f, 0.223173f, 0.288026f, 0.223173f, 0.103818f, 0.028995f }
        #endif
    };
    const float filter_y[SIZE_Y][1] = {
        #if SIZE_Y == 3
        { 0.238994f }, { 0.522011f }, { 0.238994f }
        #endif
        #if SIZE_Y == 5
        { 0.070766f }, { 0.244460f }, { 0.369546f }, { 0.244460f }, { 0.070766f }
        #endif
        #if SIZE_Y == 7
        { 0.028995f }, { 0.103818f }, { 0.223173f }, { 0.288026f }, { 0.223173f }, { 0.103818f }, { 0.028995f }
        #endif
    };
    const float filter_xy[SIZE_Y][SIZE_X] = {
        #if SIZE_X == 3
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
        #endif
        #if SIZE_X == 5
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f }
        #endif
        #if SIZE_X == 7
        { 0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841 },
        { 0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010 },
        { 0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471 },
        { 0.008351, 0.029902, 0.064280, 0.082959, 0.064280, 0.029902, 0.008351 },
        { 0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471 },
        { 0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010 },
        { 0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841 }
        #endif
    };
    #else
    float filter_x[1][SIZE_X];
    float filter_y[SIZE_Y][1];
    float filter_xy[SIZE_Y][SIZE_X];

    double scale2X = -0.5/(sigma1*sigma1);
    double scale2Y = -0.5/(sigma2*sigma2);
    double sum_x = 0.;
    double sum_y = 0.;

    for (int i=0; i < size_x; i++) {
        double x = i - (size_x-1)*0.5;
        double t = exp(scale2X*x*x);

        filter_x[0][i] = (float)t;
        sum_x += filter_x[0][i];
    }
    for (int i=0; i < size_y; i++) {
        double x = i - (size_y-1)*0.5;
        double t = exp(scale2Y*x*x);

        filter_y[i][0] = (float)t;
        sum_y += filter_y[i][0];
    }

    sum_x = 1./sum_x;
    sum_y = 1./sum_y;
    for (int i=0; i < size_x; i++) {
        filter_x[0][i] = (float)(filter_x[0][i]*sum_x);
    }
    for (int i=0; i < size_y; i++) {
        filter_y[i][0] = (float)(filter_y[i][0]*sum_y);
    }

    for (int y=0; y < size_y; y++) {
        for (int x=0; x < size_x; x++) {
            filter_xy[y][x] = filter_x[0][x]*filter_y[y][0];
        }
    }
    #endif

    // host memory for image of width x height pixels
    uchar *input = new uchar[width*height];
    uchar *reference_in = new uchar[width*height];
    uchar *reference_out = new uchar[width*height];
    float *reference_tmp = new float[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = (uchar)(y*width + x) % 256;
            reference_in[y*width + x] = (uchar)(y*width + x) % 256;
            reference_out[y*width + x] = 0;
            reference_tmp[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<uchar> IN(width, height, input);
    Image<uchar> OUT(width, height);
    Image<float> TMP(width, height);

    // filter mask
    Mask<float> M(filter_xy);
    Mask<float> MX(filter_x);
    Mask<float> MY(filter_y);

    IterationSpace<uchar> IsOut(OUT);
    IterationSpace<float> IsTmp(TMP);


    #ifndef OpenCV
    std::cerr << "Calculating Hipacc Gaussian filter ..." << std::endl;
    float timing = 0.0f;

    // UNDEFINED
    #ifdef RUN_UNDEF
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInUndef2(IN, M, Boundary::UNDEFINED);
    Accessor<uchar> AccInUndef2(BcInUndef2);
    GaussianBlurFilterMask GFU(IsOut, AccInUndef2, M, size_x, size_y);

    GFU.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<uchar> BcInUndef(IN, MX, Boundary::UNDEFINED);
    Accessor<uchar> AccInUndef(BcInUndef);
    GaussianBlurFilterMaskRow GFRU(IsTmp, AccInUndef, MX, size_x);

    BoundaryCondition<float> BcTmpUndef(TMP, MY, Boundary::UNDEFINED);
    Accessor<float> AccTmpUndef(BcTmpUndef);
    GaussianBlurFilterMaskColumn GFCU(IsOut, AccTmpUndef, MY, size_y);

    GFRU.execute();
    timing = hipacc_last_kernel_timing();
    GFCU.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (UNDEFINED): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CLAMP
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInClamp2(IN, M, Boundary::CLAMP);
    Accessor<uchar> AccInClamp2(BcInClamp2);
    GaussianBlurFilterMask GFC(IsOut, AccInClamp2, M, size_x, size_y);

    GFC.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<uchar> BcInClamp(IN, MX, Boundary::CLAMP);
    Accessor<uchar> AccInClamp(BcInClamp);
    GaussianBlurFilterMaskRow GFRC(IsTmp, AccInClamp, MX, size_x);

    BoundaryCondition<float> BcTmpClamp(TMP, MY, Boundary::CLAMP);
    Accessor<float> AccTmpClamp(BcTmpClamp);
    GaussianBlurFilterMaskColumn GFCC(IsOut, AccTmpClamp, MY, size_y);

    GFRC.execute();
    timing = hipacc_last_kernel_timing();
    GFCC.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // REPEAT
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInRepeat2(IN, M, Boundary::REPEAT);
    Accessor<uchar> AccInRepeat2(BcInRepeat2);
    GaussianBlurFilterMask GFR(IsOut, AccInRepeat2, M, size_x, size_y);

    GFR.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<uchar> BcInRepeat(IN, MX, Boundary::REPEAT);
    Accessor<uchar> AccInRepeat(BcInRepeat);
    GaussianBlurFilterMaskRow GFRR(IsTmp, AccInRepeat, MX, size_x);

    BoundaryCondition<float> BcTmpRepeat(TMP, MY, Boundary::REPEAT);
    Accessor<float> AccTmpRepeat(BcTmpRepeat);
    GaussianBlurFilterMaskColumn GFCR(IsOut, AccTmpRepeat, MY, size_y);

    GFRR.execute();
    timing = hipacc_last_kernel_timing();
    GFCR.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (REPEAT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // MIRROR
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInMirror2(IN, M, Boundary::MIRROR);
    Accessor<uchar> AccInMirror2(BcInMirror2);
    GaussianBlurFilterMask GFM(IsOut, AccInMirror2, M, size_x, size_y);

    GFM.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<uchar> BcInMirror(IN, MX, Boundary::MIRROR);
    Accessor<uchar> AccInMirror(BcInMirror);
    GaussianBlurFilterMaskRow GFRM(IsTmp, AccInMirror, MX, size_x);

    BoundaryCondition<float> BcTmpMirror(TMP, MY, Boundary::MIRROR);
    Accessor<float> AccTmpMirror(BcTmpMirror);
    GaussianBlurFilterMaskColumn GFCM(IsOut, AccTmpMirror, MY, size_y);

    GFRM.execute();
    timing = hipacc_last_kernel_timing();
    GFCM.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (MIRROR): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CONSTANT
    #ifdef NO_SEP
    BoundaryCondition<uchar> BcInConst2(IN, M, Boundary::CONSTANT, '1');
    Accessor<uchar> AccInConst2(BcInConst2);
    GaussianBlurFilterMask GFConst(IsOut, AccInConst2, M, size_x, size_y);

    GFConst.execute();
    timing = hipacc_last_kernel_timing();
    #else
    BoundaryCondition<uchar> BcInConst(IN, MX, Boundary::CONSTANT, '1');
    Accessor<uchar> AccInConst(BcInConst);
    GaussianBlurFilterMaskRow GFRConst(IsTmp, AccInConst, MX, size_x);

    BoundaryCondition<float> BcTmpConst(TMP, MY, Boundary::CONSTANT, 1.0f);
    Accessor<float> AccTmpConst(BcTmpConst);
    GaussianBlurFilterMaskColumn GFCConst(IsOut, AccTmpConst, MY, size_y);

    GFRConst.execute();
    timing = hipacc_last_kernel_timing();
    GFCConst.execute();
    timing += hipacc_last_kernel_timing();
    #endif
    timings.push_back(timing);
    std::cerr << "Hipacc (CONSTANT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // get pointer to result data
    uchar *output = OUT.data();
    #endif



    #ifdef OpenCV
    #ifdef CPU
    std::cerr << std::endl << "Calculating OpenCV Gaussian filter on the CPU ..." << std::endl;
    #else
    std::cerr << std::endl << "Calculating OpenCV Gaussian filter on the GPU ..." << std::endl;
    #endif


    cv::Mat cv_data_in(height, width, CV_8UC1, input);
    cv::Mat cv_data_out(height, width, CV_8UC1, cv::Scalar(0));
    cv::Size ksize(size_x, size_y);

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

            cv::GaussianBlur(cv_data_in, cv_data_out, ksize, sigma1, sigma2, brd_type);

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

            cv::gpu::GaussianBlur(gpu_in, gpu_out, ksize, sigma1, sigma2, brd_type);

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
    uchar *output = (uchar *)cv_data_out.data;
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
        gaussian_filter(reference_in, reference_out, (float *)filter_xy, size_x, size_y, width, height);
        #else
        gaussian_filter_row(reference_in, reference_tmp, (float *)filter_x, size_x, width, height);
        gaussian_filter_column(reference_tmp, reference_out, (float *)filter_y, size_y, width, height);
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
                          << (int)reference_out[y*width + x] << " vs. "
                          << (int)output[y*width + x] << std::endl;
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

