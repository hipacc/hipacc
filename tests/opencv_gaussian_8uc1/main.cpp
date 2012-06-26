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
#include <float.h>
#include <math.h>
#include <stdio.h>
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

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Gaussian filter reference
void gaussian_filter(unsigned char *in, unsigned char *out, float *filter, int
        size_x, int size_y, int width, int height) {
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
            float sum = 0.0f;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]*in[(y+yf)*width + x + xf];
                }
            }
            out[y*width + x] = (unsigned char) (sum+0.5f);
        }
    }
}
void gaussian_filter_row(unsigned char *in, float *out, float *filter, int
        size_x, int width, int height) {
    int anchor_x = size_x >> 1;
#ifdef OpenCV
    int upper_x = width-size_x+anchor_x;
#else
    int upper_x = width-anchor_x;
#endif

    for (int y=0; y<height; ++y) {
        //for (int x=0; x<anchor_x; x++) out[y*width + x] = in[y*width + x];
        for (int x=anchor_x; x<upper_x; ++x) {
            float sum = 0.0f;

            for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                sum += filter[xf+anchor_x]*in[(y)*width + x + xf];
            }
            out[y*width + x] = sum;
        }
        //for (int x=upper_x; x<width; x++) out[y*width + x] = in[y*width + x];
    }
}
void gaussian_filter_column(float *in, unsigned char *out, float *filter, int
        size_y, int width, int height) {
    int anchor_y = size_y >> 1;
#ifdef OpenCV
    int upper_y = height-size_y+anchor_y;
#else
    int upper_y = height-anchor_y;
#endif

    //for (int y=0; y<anchor_y; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = (unsigned char) in[y*width + x];
    //    }
    //}
    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=0; x<width; ++x) {
            float sum = 0.0f;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                sum += filter[yf+anchor_y]*in[(y + yf)*width + x];
            }
            out[y*width + x] = (unsigned char) (sum+0.5f);
        }
    }
    //for (int y=upper_y; y<height; y++) {
    //    for (int x=0; x<width; ++x) {
    //        out[y*width + x] = (unsigned char) in[y*width + x];
    //    }
    //}
}


namespace hipacc {
class GaussianBlurFilter3 : public Kernel<unsigned char> {
    private:
        Accessor<unsigned char> &Input;
        float f0, f1, f2, f3, f4, f5, f6, f7, f8;
        int size_x, size_y;

    public:
        GaussianBlurFilter3(IterationSpace<unsigned char> &IS, Accessor<unsigned
                char> &Input, float f0, float f1, float f2, float f3, float f4,
                float f5, float f6, float f7, float f8, int size_x, int size_y)
            :
            Kernel(IS),
            Input(Input),
            f0(f0),
            f1(f1),
            f2(f2),
            f3(f3),
            f4(f4),
            f5(f5),
            f6(f6),
            f7(f7),
            f8(f8),
            size_x(size_x),
            size_y(size_y)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            sum = f0*Input(-1, -1) + f1*Input(0, -1) + f2*Input(1, -1)
                + f3*Input(-1, 0) + f4*Input(0, 0) + f5*Input(1, 0)
                + f6*Input(-1, 1) + f7*Input(0, 1) + f8*Input(1, 1);
            output() = (unsigned char) (sum+0.5f);
        }
};
class GaussianBlurFilterRow3 : public Kernel<float> {
    private:
        Accessor<unsigned char> &Input;
        float f0, f1, f2;
        int size_x;

    public:
        GaussianBlurFilterRow3(IterationSpace<float> &IS, Accessor<unsigned
                char> &Input, float f0, float f1, float f2, int size_x) :
            Kernel(IS),
            Input(Input),
            f0(f0),
            f1(f1),
            f2(f2),
            size_x(size_x)
        {
            addAccessor(&Input);
        }

        void kernel() {
            //int anchor_x = size_x >> 1;
            float sum = 0.0f;

            sum = f0*Input(-1, 0) + f1*Input(0, 0) + f2*Input(1, 0);
            //for (int xf = -anchor_x; xf<=anchor_x; xf++) {
            //    sum += filter[xf+anchor_x]*Input(xf, 0);
            //}
            output() = sum;
        }
};
class GaussianBlurFilterColumn3 : public Kernel<unsigned char> {
    private:
        Accessor<float> &Input;
        float f0, f1, f2;
        int size_y;

    public:
        GaussianBlurFilterColumn3(IterationSpace<unsigned char> &IS,
                Accessor<float> &Input, float f0, float f1, float f2, int
                size_y) :
            Kernel(IS),
            Input(Input),
            f0(f0),
            f1(f1),
            f2(f2),
            size_y(size_y)
        {
            addAccessor(&Input);
        }

        void kernel() {
            //int anchor_y = size_y >> 1;
            float sum = 0.0f;

            sum = f0*Input(0, -1) + f1*Input(0, 0) + f2*Input(0, 1);
            //for (int yf = -anchor_y; yf<=anchor_y; yf++) {
            //    sum += filter[yf+anchor_y]*Input(0, yf);
            //}
            output() = (unsigned char) (sum+0.5f);
        }
};
class GaussianBlurFilterRow5 : public Kernel<float> {
    private:
        Accessor<unsigned char> &Input;
        float f0, f1, f2, f3, f4;
        int size_x;

    public:
        GaussianBlurFilterRow5(IterationSpace<float> &IS, Accessor<unsigned
                char> &Input, float f0, float f1, float f2, float f3, float f4,
                int size_x) :
            Kernel(IS),
            Input(Input),
            f0(f0),
            f1(f1),
            f2(f2),
            f3(f3),
            f4(f4),
            size_x(size_x)
        {
            addAccessor(&Input);
        }

        void kernel() {
            //int anchor_x = size_x >> 1;
            float sum = 0.0f;

            sum = f0*Input(-2, 0) + f1*Input(-1, 0) + f2*Input(0, 0) +
                f3*Input(1, 0) + f4*Input(2, 0);
            //for (int xf = -anchor_x; xf<=anchor_x; xf++) {
            //    sum += filter[xf+anchor_x]*Input(xf, 0);
            //}
            output() = sum;
        }
};
class GaussianBlurFilterColumn5 : public Kernel<unsigned char> {
    private:
        Accessor<float> &Input;
        float f0, f1, f2, f3, f4;
        int size_y;

    public:
        GaussianBlurFilterColumn5(IterationSpace<unsigned char> &IS,
                Accessor<float> &Input, float f0, float f1, float f2, float f3,
                float f4, int size_y) :
            Kernel(IS),
            Input(Input),
            f0(f0),
            f1(f1),
            f2(f2),
            f3(f3),
            f4(f4),
            size_y(size_y)
        {
            addAccessor(&Input);
        }

        void kernel() {
            //int anchor_y = size_y >> 1;
            float sum = 0.0f;

            sum = f0*Input(0, -2) + f1*Input(0, -1) + f2*Input(0, 0) +
                f3*Input(0, 1) + f4*Input(0, 2);
            //for (int yf = -anchor_y; yf<=anchor_y; yf++) {
            //    sum += filter[yf+anchor_y]*Input(0, yf);
            //}
            output() = (unsigned char) (sum+0.5f);
        }
};
}


int main(int argc, const char **argv) {
    double time0, time1, dt, min_dt = DBL_MAX;
    int width = WIDTH;
    int height = HEIGHT;
    int size_x = SIZE_X;
    int size_y = SIZE_Y;
    int offset_x = size_x >> 1;
    int offset_y = size_y >> 1;
    const double sigma1 = ((size_x-1)*0.5 - 1)*0.3 + 0.8;
    const double sigma2 = ((size_y-1)*0.5 - 1)*0.3 + 0.8;

    // only filter kernel sizes 3x3 and 5x5 implemented
    if (size_x != size_y && (size_x != 3 || size_x != 5)) {
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!\n");
        exit(EXIT_FAILURE);
    }

    // host memory for image of of widthxheight pixels
    unsigned char *host_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *host_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    float *reference_tmp = (float *)malloc(sizeof(float)*width*height);

    // filter coefficients
    float *filter_x = (float *)malloc(sizeof(float)*size_x);
    float *filter_y = (float *)malloc(sizeof(float)*size_y);
    float *filter_xy = (float *)malloc(sizeof(float)*size_x*size_y);

    double scale2X = -0.5/(sigma1*sigma1);
    double scale2Y = -0.5/(sigma2*sigma2);
    double sum_x = 0.;
    double sum_y = 0.;
    
    for (int i=0; i < size_x; i++) {
        double x = i - (size_x-1)*0.5;
        double t = exp(scale2X*x*x);

        filter_x[i] = (float)t;
        sum_x += filter_x[i];
    }
    for (int i=0; i < size_y; i++) {
        double x = i - (size_y-1)*0.5;
        double t = exp(scale2Y*x*x);

        filter_y[i] = (float)t;
        sum_y += filter_y[i];
    }
    
    sum_x = 1./sum_x;
    sum_y = 1./sum_y;
    for (int i=0; i < size_x; i++) {
        filter_x[i] = (float)(filter_x[i]*sum_x);
    }
    for (int i=0; i < size_y; i++) {
        filter_y[i] = (float)(filter_y[i]*sum_y);
    }

    for (int y=0; y < size_y; y++) {
        for (int x=0; x < size_x; x++) {
            filter_xy[y*size_x + x] = filter_x[x]*filter_y[y];
        }
    }


    // input and output image of widthxheight pixels
    Image<unsigned char> IN(width, height);
    Image<unsigned char> OUT(width, height);
    Image<float> TMP(width, height);
    Accessor<unsigned char> AccIn(IN, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    Accessor<unsigned char> AccInR(IN, width-2*offset_x, height, offset_x, 0);
    Accessor<float> AccTmp(TMP, width, height-2*offset_y, 0, offset_y);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            reference_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            host_out[y*width + x] = 0;
            reference_out[y*width + x] = 0;
            reference_tmp[y*width + x] = 0;
        }
    }

    IterationSpace<float> GISR(TMP, width-2*offset_x, height, offset_x, 0);
    IterationSpace<unsigned char> GISC(OUT, width, height-2*offset_y, 0, offset_y);
    IterationSpace<unsigned char> GIS(OUT, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    GaussianBlurFilter3 GF(GIS, AccIn, filter_xy[0], filter_xy[1], filter_xy[2],
            filter_xy[3], filter_xy[4], filter_xy[5], filter_xy[6],
            filter_xy[7], filter_xy[8], size_x, size_y);
#if SIZE_X == 3
    GaussianBlurFilterRow3 GFR(GISR, AccInR, filter_x[0], filter_x[1],
            filter_x[2], size_x);
    GaussianBlurFilterColumn3 GFC(GISC, AccTmp, filter_y[0], filter_y[1],
            filter_y[2], size_y);
#else
    GaussianBlurFilterRow5 GFR(GISR, AccInR, filter_x[0], filter_x[1],
            filter_x[2], filter_x[3], filter_x[4], size_x);
    GaussianBlurFilterColumn5 GFC(GISC, AccTmp, filter_y[0], filter_y[1],
            filter_y[2], filter_y[3], filter_y[4], size_y);
#endif

    IN = host_in;
    OUT = host_out;

    fprintf(stderr, "Calculating Gaussian filter ...\n");

    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        //GF.execute();
        GFR.execute();
        GFC.execute();

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }

    // get results
    host_out = OUT.getData();

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    // NB: actually there are (width-d)*(height) output pixels
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", min_dt,
            ((width-2*offset_x)*(height-2*offset_y)/min_dt)/1000);



#ifdef OpenCV
    // OpenCV uses NPP library for filtering
    // image: 4096x4096
    // kernel size: 3x3
    // offset 3x3 shiftet by 1 -> 1x1
    // output: 4096x4096 - 3x3 -> 4093x4093; start: 1,1; end: 4094,4094
    //
    // image: 4096x4096
    // kernel size: 4x4
    // offset 4x4 shiftet by 1 -> 2x2
    // output: 4096x4096 - 4x4 -> 4092x4092; start: 2,2; end: 4094,4094
#ifdef CPU
    fprintf(stderr, "\nCalculating OpenCV Gaussian filter on the CPU ...\n");
#else
    fprintf(stderr, "\nCalculating OpenCV Gaussian filter on the GPU ...\n");
#endif


    cv::Mat cv_data_in(height, width, CV_8UC1, host_in);
    cv::Mat cv_data_out(height, width, CV_8UC1, host_out);
    cv::Size ksize(size_x, size_y);
#ifdef CPU
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        cv::GaussianBlur(cv_data_in, cv_data_out, ksize, sigma1, sigma2);

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

        cv::gpu::GaussianBlur(gpu_in, gpu_out, ksize, sigma1, sigma2);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }

    gpu_out.download(cv_data_out);
#endif

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    // NB: actually there are (width-d)*(height) output pixels
    fprintf(stderr, "OpenCV: %.3f ms, %.3f Mpixel/s\n", min_dt,
            ((width-size_x)*(height-size_y)/min_dt)/1000);
#endif



    fprintf(stderr, "\nCalculating reference ...\n");
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        // calculate reference
        //gaussian_filter(reference_in, reference_out, filter_xy, size_x, size_y, width, height);
        gaussian_filter_row(reference_in, reference_tmp, filter_x, size_x, width, height);
        gaussian_filter_column(reference_tmp, reference_out, filter_y, size_y, width, height);

        time1 = time_ms();
        dt = time1 - time0;
        if (dt < min_dt) min_dt = dt;
    }
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", min_dt,
            ((width-2*offset_x)*(height-2*offset_y)/min_dt)/1000);

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
            if (reference_out[y*width + x] != host_out[y*width +x]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %hhu vs. %hhu\n", x,
                        y, reference_out[y*width + x], host_out[y*width +x]);
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

