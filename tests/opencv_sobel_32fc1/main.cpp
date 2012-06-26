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


// Sobel filter reference
#if 1
void sobel_filter_x(float *in, float *out, int size, int width, int height) {
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

    if (size_x == 3) {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                float sum = 0.0f;

                // -1 0 1
                // -2 0 2
                // -1 0 1
                sum = - in[(y-1)*width + x-1] + in[(y-1)*width + x+1]
                    - 2*in[y*width + x-1] + 2*in[y*width + x+1]
                    - in[(y+1)*width + x-1] + in[(y+1)*width + x+1];
                out[y*width + x] = sum;
            }
        }
    } else {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                float sum = 0.0f;

                // -1   -2  0   2   1
                // -4   -8  0   8   4
                // -6   -12 0   12  6
                // -4   -8  0   8   4
                // -1   -2  0   2   1
                sum = - 1*in[(y-2)*width + x-2] - 2*in[(y-2)*width + x-1] + 2*in[(y-2)*width + x+1] + 1*in[(y-2)*width + x+2]
                    - 4*in[(y-1)*width + x-2] - 8*in[(y-1)*width + x-1] + 8*in[(y-1)*width + x+1] + 4*in[(y-1)*width + x+2]
                    - 6*in[(y)*width + x-2] - 12*in[(y)*width + x-1] + 12*in[(y)*width + x+1] + 6*in[(y)*width + x+2]
                    - 4*in[(y+1)*width + x-2] - 8*in[(y+1)*width + x-1] + 8*in[(y+1)*width + x+1] + 4*in[(y+1)*width + x+2]
                    - 1*in[(y+2)*width + x-2] - 2*in[(y+2)*width + x-1] + 2*in[(y+2)*width + x+1] + 1*in[(y+2)*width + x+2];
                out[y*width + x] = sum;

            }
        }
    }
}
#else
void sobel_filter_x(float *in, float *out, int *filter, int size, int width,
        int height) {
    const int size_x=size;
    const int size_y=size;
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

            for (int yf=-anchor_y; yf<=anchor_y; yf++) {
                for (int xf=-anchor_x; xf<=anchor_x; xf++) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x]*in[(y+yf)*width + x+xf];
                }
            }
            out[y*width + x] = sum;
            //sum = - in[(y-1)*width + x-1] + in[(y-1)*width + x+1]
            //    - 2*in[y*width + x-1] + 2*in[y*width + x+1]
            //    - in[(y+1)*width + x-1] + in[(y+1)*width + x+1];
            //out[y*width + x] = sum;

        }
    }
}
#endif
void sobel_filter_y(float *in, float *out, int size, int width, int height) {
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

    if (size_y == 3) {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                float sum = 0.0f;

                // -1 -2 -1
                // 0  0  0
                // 1  2  1
                sum = - in[(y-1)*width + x-1] - 2*in[(y-1)*width + x] - in[(y-1)*width + x+1]
                    + in[(y+1)*width + x-1] + 2*in[(y+1)*width + x] + in[(y+1)*width + x+1];
                out[y*width + x] = sum;
            }
        }
    } else {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                float sum = 0.0f;

                // -1   -4  -6  -4  -1
                // -2   -8  -12 -8  -2
                // 0    0   0   0   0
                // 2    8   12  8   2
                // 1    4   6   4   1
                sum = - 1*in[(y-2)*width + x-2] - 4*in[(y-2)*width + x-1] - 6*in[(y-2)*width + x] - 4*in[(y-2)*width + x+1] - 1*in[(y-2)*width + x+2]
                    - 2*in[(y-1)*width + x-2] - 8*in[(y-1)*width + x-1] - 12*in[(y-1)*width + x] - 8*in[(y-1)*width + x+1] - 2*in[(y-1)*width + x+2]
                    + 2*in[(y+1)*width + x-2] + 8*in[(y+1)*width + x-1] + 12*in[(y+1)*width + x] + 8*in[(y+1)*width + x+1] + 2*in[(y+1)*width + x+2]
                    + 1*in[(y+2)*width + x-2] + 4*in[(y+2)*width + x-1] + 6*in[(y+2)*width + x] + 4*in[(y+2)*width + x+1] + 1*in[(y+2)*width + x+2];
                out[y*width + x] = sum;
            }
        }
    }
}
// Scharr filter reference
void scharr_filter_x(float *in, float *out, int width, int height) {
    const int size_x=3;
    const int size_y=3;
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

            sum = - 3*in[(y-1)*width + x-1] + 3*in[(y-1)*width + x+1]
                - 10*in[y*width + x-1] + 10*in[y*width + x+1]
                - 3*in[(y+1)*width + x-1] + 3*in[(y+1)*width + x+1];
            out[y*width + x] = sum;
        }
    }
}
void scharr_filter_y(float *in, float *out, int width, int height) {
    const int size_x=3;
    const int size_y=3;
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

            sum = - 3*in[(y-1)*width + x-1] - 10*in[(y-1)*width + x] - 3*in[(y-1)*width + x+1]
                + 3*in[(y+1)*width + x-1] + 10*in[(y+1)*width + x] + 3*in[(y+1)*width + x+1];
            out[y*width + x] = sum;
        }
    }
}


namespace hipacc {
class SobelFilterX : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int size;

    public:
        SobelFilterX(IterationSpace<float> &IS, Accessor<float> &Input, int
                size) :
            Kernel(IS),
            Input(Input),
            size(size)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            if (size == 3) {
                // -1 0 1
                // -2 0 2
                // -1 0 1
                sum = - Input(-1, -1) + Input(1, -1)
                    - 2*Input(-1, 0) + 2*Input(1, 0)
                    - Input(-1, 1) + Input(1, 1);
            } else {
                // -1   -2  0   2   1
                // -4   -8  0   8   4
                // -6   -12 0   12  6
                // -4   -8  0   8   4
                // -1   -2  0   2   1
                sum = - 1*Input(-2, -2) - 2*Input(-1, -2) + 2*Input(1, -2) + 1*Input(2, -2)
                    - 4*Input(-2, -1) - 8*Input(-1, -1) + 8*Input(1, -1) + 4*Input(2, -1)
                    - 6*Input(-2, 0) - 12*Input(-1, 0) + 12*Input(1, 0) + 6*Input(2, 0)
                    - 4*Input(-2, 1) - 8*Input(-1, 1) + 8*Input(1, 1) + 4*Input(2, 1)
                    - 1*Input(-2, +2) - 2*Input(-1, +2) + 2*Input(1, +2) + 1*Input(2, +2);
            }
            output() = sum;
        }
};
class SobelFilterY : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int size;

    public:
        SobelFilterY(IterationSpace<float> &IS, Accessor<float> &Input, int
                size) :
            Kernel(IS),
            Input(Input),
            size(size)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            if (size == 3) {
                // -1 -2 -1
                // 0  0  0
                // 1  2  1
                sum = - Input(-1, -1) - 2*Input(0, -1) - Input(1, -1) +
                    Input(-1, 1) + 2*Input(0, 1) + Input(1, 1);
            } else {
                // -1   -4  -6  -4  -1
                // -2   -8  -12 -8  -2
                // 0    0   0   0   0
                // 2    8   12  8   2
                // 1    4   6   4   1
                sum = - 1*Input(-2, -2) - 4*Input(-1, -2) - 6*Input(0, -2) - 4*Input(1, -2) - 1*Input(2, -2)
                    - 2*Input(-2, -1) - 8*Input(-1, -1) - 12*Input(0, -1) - 8*Input(1, -1) - 2*Input(2, -1)
                    + 2*Input(-2, 1) + 8*Input(-1, 1) + 12*Input(0, 1) + 8*Input(1, 1) + 2*Input(2, 1)
                    + 1*Input(-2, 2) + 4*Input(-1, 2) + 6*Input(0, 2) + 4*Input(1, 2) + 1*Input(2, 2);
            }
            output() = sum;
        }
};
class ScharrFilterX : public Kernel<float> {
    private:
        Accessor<float> &Input;

    public:
        ScharrFilterX(IterationSpace<float> &IS, Accessor<float> &Input) :
            Kernel(IS),
            Input(Input)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            sum = - 3*Input(-1, -1) + 3*Input(1, -1)
                - 10*Input(-1, 0) + 10*Input(1, 0)
                - 3*Input(-1, 1) + 3*Input(1, 1);
            output() = sum;
        }
};
class ScharrFilterY : public Kernel<float> {
    private:
        Accessor<float> &Input;

    public:
        ScharrFilterY(IterationSpace<float> &IS, Accessor<float> &Input) :
            Kernel(IS),
            Input(Input)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            sum = - 3*Input(-1, -1) - 10*Input(0, -1) - 3*Input(1, -1)
                + 3*Input(-1, 1) + 10*Input(0, 1) + 3*Input(1, 1);
            output() = sum;
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

    // host memory for image of of widthxheight pixels
    float *host_in = (float *)malloc(sizeof(float)*width*height);
    float *host_out = (float *)malloc(sizeof(float)*width*height);
    float *reference_in = (float *)malloc(sizeof(float)*width*height);
    float *reference_out = (float *)malloc(sizeof(float)*width*height);

    // input and output image of widthxheight pixels
    Image<float> IN(width, height);
    Image<float> OUT(width, height);
    Accessor<float> AccIn(IN, width-2*offset_x, height-2*offset_y, offset_x, offset_y);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (float)((y*width + x) % 256);
            reference_in[y*width + x] = (float)((y*width + x) % 256);
            host_out[y*width + x] = 0;
            reference_out[y*width + x] = 0;
        }
    }

    IterationSpace<float> SIS(OUT, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    #ifdef SCHARR
    #ifdef YORDER
    ScharrFilterY SF(SIS, AccIn);
    #else
    ScharrFilterX SF(SIS, AccIn);
    #endif
    #else
    #ifdef YORDER
    SobelFilterY SF(SIS, AccIn, size_y);
    #else
    SobelFilterX SF(SIS, AccIn, size_x);
    #endif
    #endif

    IN = host_in;
    OUT = host_out;

    fprintf(stderr, "Calculating %s filter ...\n",
            #ifdef SCHARR
            "Scharr"
            #else
            "Sobel"
            #endif
    );

    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        SF.execute();

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
    fprintf(stderr, "\nCalculating OpenCV %s filter on the %s ...\n",
            #ifdef SCHARR
            "Scharr", 
            #else
            "Sobel", 
            #endif
            #ifdef CPU
            "CPU"
            #else
            "GPU"
            #endif
    );

    cv::Mat cv_data_in(height, width, CV_32FC1, host_in);
    cv::Mat cv_data_out(height, width, CV_32FC1, host_out);
#ifdef CPU
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        #ifdef SCHARR
        #ifdef YORDER
        cv::Scharr(cv_data_in, cv_data_out, -1, 0, 1);
        #else
        cv::Scharr(cv_data_in, cv_data_out, -1, 1, 0);
        #endif
        #else
        #ifdef YORDER
        cv::Sobel(cv_data_in, cv_data_out, -1, 0, 1, size_y);
        #else
        cv::Sobel(cv_data_in, cv_data_out, -1, 1, 0, size_x);
        #endif
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

        #ifdef SCHARR
        #ifdef YORDER
        cv::gpu::Scharr(gpu_in, gpu_out, -1, 0, 1);
        #else
        cv::gpu::Scharr(gpu_in, gpu_out, -1, 1, 0);
        #endif
        #else
        #ifdef YORDER
        cv::gpu::Sobel(gpu_in, gpu_out, -1, 0, 1, size_y);
        #else
        cv::gpu::Sobel(gpu_in, gpu_out, -1, 1, 0, size_x);
        #endif
        #endif

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
        #ifdef SCHARR
        #ifdef YORDER
        scharr_filter_y(reference_in, reference_out, width, height);
        #else
        scharr_filter_x(reference_in, reference_out, width, height);
        #endif
        #else
        #ifdef YORDER
        sobel_filter_y(reference_in, reference_out, size_y, width, height);
        #else
        sobel_filter_x(reference_in, reference_out, size_x, width, height);
        #endif
        #endif

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
                fprintf(stderr, "Test FAILED, at (%d,%d): %f vs. %f\n", x,
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

