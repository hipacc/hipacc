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

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Laplace filter reference
void laplace_filter(unsigned char *in, unsigned char *out, int size, int width,
        int height) {
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

    if (size == 5) {
        const int mask[5][5] = {
            {1,   1,   1,   1,   1},
            {1,   1,   1,   1,   1},
            {1,   1, -24,   1,   1},
            {1,   1,   1,   1,   1},
            {1,   1,   1,   1,   1}};
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                int sum = 0;

                for (int yf=-anchor_y; yf<=anchor_y; yf++) {
                    for (int xf=-anchor_x; xf<=anchor_x; xf++) {
                        sum += mask[(yf+anchor_y)][xf+anchor_x]*in[(y+yf)*width + x+xf];
                    }
                }

                sum = max(0, min(sum, 255));
                out[y*width + x] = (unsigned char) (sum);
            }
        }
    } else if (size == 3) {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                int sum = 0;

                // 2  0  2
                // 0 -8  0
                // 2  0  2
                sum = 2*in[(y-1)*width + x-1] + 2*in[(y-1)*width + x+1]
                    - 8*in[(y)*width + x]
                    + 2*in[(y+1)*width + x-1] + 2*in[(y+1)*width + x+1];
                sum = max(0, min(sum, 255));
                out[y*width + x] = (unsigned char) (sum);
            }
        }
    } else {
        for (int y=anchor_y; y<upper_y; ++y) {
            for (int x=anchor_x; x<upper_x; ++x) {
                int sum = 0;

                // 0  1  0
                // 1 -4  1
                // 0  1  0
                sum = in[(y-1)*width + x]
                    + in[(y)*width + x-1] - 4*in[(y)*width + x] + in[(y)*width + x+1]
                    + in[(y+1)*width + x];
                sum = max(0, min(sum, 255));
                out[y*width + x] = (unsigned char) (sum);
            }
        }
    }
}


namespace hipacc {
class LaplaceFilter : public Kernel<unsigned char> {
    private:
        Accessor<unsigned char> &Input;
        int size;

    public:
        LaplaceFilter(IterationSpace<unsigned char> &IS, Accessor<unsigned char>
                &Input, int size) :
            Kernel(IS),
            Input(Input),
            size(size)
        {
            addAccessor(&Input);
        }

        void kernel() {
            int sum = 0;

            if (size == 5) {
                #if 1
                sum = Input(-2, -2) + Input(-1, -2) + Input(0, -2) + Input(1, -2) + Input(2, -2)
                    + Input(-2, -1) + Input(-1, -1) + Input(0, -1) + Input(1, -1) + Input(2, -1)
                    + Input(-2, 0) + Input(-1, 0) - 24*Input(0, 0) + Input(1, 0) + Input(2, 0)
                    + Input(-2, 1) + Input(-1, 1) + Input(0, 1) + Input(1, 1) + Input(2, 1)
                    + Input(-2, 2) + Input(-1, 2) + Input(0, 2) + Input(1, 2) + Input(2, 2);
                #else
                int anchor = size >> 1;
                const int mask[5][5] = {
                    {1,   1,   1,   1,   1},
                    {1,   1,   1,   1,   1},
                    {1,   1, -24,   1,   1},
                    {1,   1,   1,   1,   1},
                    {1,   1,   1,   1,   1}};

                for (int yf=-anchor; yf<=anchor; yf++) {
                    for (int xf=-anchor; xf<=anchor; xf++) {
                        sum += mask[(yf+anchor)][xf+anchor]*Input(xf, yf);
                    }
                }
                #endif
            } else if (size == 3) {
                // 2  0  2
                // 0 -8  0
                // 2  0  2
                sum = 2*Input(-1, -1) + 2*Input(1, -1)
                    - 8*Input(0, 0)
                    + 2*Input(-1, 1) + 2*Input(1, 1);
            } else {
                // 0  1  0
                // 1 -4  1
                // 0  1  0
                sum = Input(0, -1)
                    + Input(-1, 0) - 4*Input(0, 0) + Input(1, 0)
                    + Input(0, 1);
            }
            if (sum > 255) sum = 255;
            else if (sum < 0) sum = 0;
            output() = (unsigned char) (sum);
        }
};
}


int main(int argc, const char **argv) {
    double time0, time1, dt, min_dt = DBL_MAX;
    int width = WIDTH;
    int height = HEIGHT;
    int size_x = SIZE_X;
    int size_y = SIZE_Y;
    // all filters are 3x3
    int offset_x = size_x >> 1;
    int offset_y = size_y >> 1;

    // only filter kernel sizes 3x3 and 5x5 implemented
    if (size_x != size_y && (size_x != 3 || size_x != 5)) {
        fprintf(stderr, "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!\n");
        exit(EXIT_FAILURE);
    }
#ifdef OpenCV
#ifndef CPU
#if SIZE_X==5
#error "OpenCV supports only 1x1 and 3x3 Laplace filters on the GPU!"
#endif
#endif
#endif

    // host memory for image of of widthxheight pixels
    unsigned char *host_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *host_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_in = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
    unsigned char *reference_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);

    // input and output image of widthxheight pixels
    Image<unsigned char> IN(width, height);
    Image<unsigned char> OUT(width, height);
    Accessor<unsigned char> AccIn(IN, width-2*offset_x, height-2*offset_y, offset_x, offset_y);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            reference_in[y*width + x] = (unsigned char)(y*width + x) % 256;
            host_out[y*width + x] = 0;
            reference_out[y*width + x] = 0;
        }
    }

    IterationSpace<unsigned char> LIS(OUT, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    LaplaceFilter LF(LIS, AccIn, size_x);

    IN = host_in;
    OUT = host_out;

    fprintf(stderr, "Calculating Laplace filter ...\n");

    min_dt = DBL_MAX;
    for (int nt=0; nt<10; nt++) {
        time0 = time_ms();

        LF.execute();

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
    fprintf(stderr, "\nCalculating OpenCV Laplace filter on the %s ...\n",
            #ifdef CPU
            "CPU"
            #else
            "GPU"
            #endif
    );

    cv::Mat cv_data_in(height, width, CV_8UC1, host_in);
    cv::Mat cv_data_out(height, width, CV_8UC1, host_out);
#ifdef CPU
    min_dt = DBL_MAX;
    for (int nt=0; nt<3; nt++) {
        time0 = time_ms();

        cv::Laplacian(cv_data_in, cv_data_out, -1, size_x);

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

        cv::gpu::Laplacian(gpu_in, gpu_out, -1, size_x);

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
        laplace_filter(reference_in, reference_out, size_x, width, height);

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
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x,
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

