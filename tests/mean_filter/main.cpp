//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// Copyright (c) 2010, ARM Limited
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

//#define HSCAN
//#define SIMPLE
#define EPS 0.02f

// variables set by Makefile
#define NT 100
#define WIDTH 5120
#define HEIGHT 3200

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// horizontal mean filter reference
void horizontal_mean_filter(float *in, float *out, int d, int t, int width, int height) {
    #ifdef SIMPLE
    for (int y=0; y<height; ++y) {
        for (int x=0; x<(width-d); ++x) {
            float sum = 0;

            for (int k=0; k<d; ++k) {
                sum += in[y*width + x + k];
            }
            out[y*width + x] = sum/(float)d;
        }
    }
    #else
    int N = width-d;

    for (int y=0; y<height; ++y) {
        for (int t0=0; t0<N; t0+=t) {
            float sum = 0;

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                sum += in[y*width + t0 + k];
            }
            out[y*width + t0] = sum/(float)d;

            // second phase: rolling sum
            for (int dt=1; dt<min(t, N-t0); ++dt) {
                int t = t0+dt;
                sum -= in[y*width + t-1];
                sum += in[y*width + t-1+d];
                out[y*width + t] = sum/(float)d;
            }
        }
    }
    #endif
}


// vertical mean filter reference
void vertical_mean_filter(float *in, float *out, int d, int t, int width, int height) {
    #ifdef SIMPLE
    for (int y=0; y<(height-d); ++y) {
        for (int x=0; x<width; ++x) {
            float sum = 0;

            for (int k=0; k<d; ++k) {
                sum += in[(y+k)*width + x];
            }
            out[y*width + x] = sum/(float)d;
        }
    }
    #else
    int N = height-d;

    for (int x=0; x<width; ++x) {
        for (int t0=0; t0<N; t0+=t) {
            float sum = 0;

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                sum += in[(t0 + k)*width + x];
            }
            out[t0*width + x] = sum/(float)d;

            // second phase: rolling sum
            for (int dt=1; dt<min(t, N-t0); ++dt) {
                int t = t0+dt;
                sum -= in[(t-1)*width + x];
                sum += in[(t-1+d)*width + x];
                out[t*width + x] = sum/(float)d;
            }
        }
    }
    #endif
}


// Kernel description in HIPAcc
class HorizontalMeanFilter : public Kernel<float> {
    private:
        Accessor<float> &input;
        int d, nt, width;

    public:
        HorizontalMeanFilter(IterationSpace<float> &iter, Accessor<float>
                &input, int d, int nt, int width) :
            Kernel(iter),
            input(input),
            d(d),
            nt(nt),
            width(width)
        { addAccessor(&input); }

        void kernel() {
            float sum = 0.0f;

            #ifdef SIMPLE
            for (int k=0; k<d; ++k) {
                sum += input(k, 0);
            }

            output() = sum/(float)d;
            #else
            int t0 = getX();

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                sum += input.getPixel(k + t0*nt, input.getY());
            }
            outputAtPixel(t0*nt, getY()) = sum/(float)d;

            // second phase: rolling sum
            for (int dt=1; dt<min(nt, width-d-(t0*nt)); ++dt) {
                int t = t0*nt + dt;
                sum -= input.getPixel(t-1, input.getY());
                sum += input.getPixel(t-1+d, input.getY());
                outputAtPixel(t, getY()) = sum/(float)d;
            }
            #endif
        }
};

class VerticalMeanFilter : public Kernel<float> {
    private:
        Accessor<float> &input;
        int d, nt, height;

    public:
        VerticalMeanFilter(IterationSpace<float> &iter, Accessor<float> &input,
                int d, int nt, int height) :
            Kernel(iter),
            input(input),
            d(d),
            nt(nt),
            height(height)
        { addAccessor(&input); }

        void kernel() {
            float sum = 0;

            #ifdef SIMPLE
            for (int k=0; k<d; ++k) {
                sum += input(0, k);
            }

            output() = sum/(float)d;
            #else
            int t0 = getY();

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                sum += input.getPixel(input.getX(), k + t0*nt);
            }
            outputAtPixel(getX(), t0*nt) = sum/(float)d;

            // second phase: rolling sum
            for (int dt=1; dt<min(nt, height-d-(t0*nt)); ++dt) {
                int t = t0*nt + dt;
                sum -= input.getPixel(input.getX(), t-1);
                sum += input.getPixel(input.getX(), t-1+d);
                outputAtPixel(getX(), t) = sum/(float)d;
            }
            #endif
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    int width = WIDTH;
    int height = HEIGHT;
    int d = 40;
    int t = NT;
    float timing = 0.0f;

    // host memory for image of width x height pixels
    float *input = (float *)malloc(sizeof(float)*width*height);
    float *reference_in = (float *)malloc(sizeof(float)*width*height);
    float *reference_out = (float *)malloc(sizeof(float)*width*height);

    // input and output image of width x height pixels
    Image<float> IN(width, height);
    Image<float> OUT(width, height);
    Accessor<float> AccIN(IN);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in[y*width + x] = (float) (x*height + y) * DELTA;
            reference_out[y*width + x] = (float) (3.12451);
        }
    }


    #ifdef HSCAN
    #ifdef SIMPLE
    IterationSpace<float> HIS(OUT, width-d, height);
    #else
    IterationSpace<float> HIS(OUT, (int)ceil(((float)width-d)/t), height);
    #endif
    HorizontalMeanFilter HMF(HIS, AccIN, d, t, width);
    #else
    #ifdef SIMPLE
    IterationSpace<float> VIS(OUT, width, height-d);
    #else
    IterationSpace<float> VIS(OUT, width, (int)ceil(((float)height-d)/t));
    #endif
    VerticalMeanFilter VMF(VIS, AccIN, d, t, height);
    #endif

    IN = input;

    fprintf(stderr, "Calculating mean filter ...\n");

    #ifdef HSCAN
    HMF.execute();
    #else
    VMF.execute();
    #endif
    timing = hipaccGetLastKernelTiming();

    // get pointer to result data
    float *output = OUT.getData();

    #ifdef HSCAN
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, ((width-d)*height/timing)/1000);
    #else
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*(height-d)/timing)/1000);
    #endif


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference
    #ifdef HSCAN
    horizontal_mean_filter(reference_in, reference_out, d, t, width, height);
    #else
    vertical_mean_filter(reference_in, reference_out, d, t, width, height);
    #endif

    time1 = time_ms();
    dt = time1 - time0;
    #ifdef HSCAN
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, ((width-d)*height/dt)/1000);
    #else
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*(height-d)/dt)/1000);
    #endif

    fprintf(stderr, "\nComparing results ...\n");
    // compare results
    #ifdef HSCAN
    float rms_err = 0.0f;   // RMS error
    for (int y=0; y<height; y++) {
        for (int x=0; x<width-d; x++) {
            float derr = reference_out[y*width + x] - output[y*width + x];
            rms_err += derr*derr;

            if (fabs(derr) > EPS) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %f vs. %f\n", x, y,
                        reference_out[y*width + x], output[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    rms_err = sqrtf(rms_err / (float((width-d)*height)));
    // check RMS error
    if (rms_err > EPS) {
        fprintf(stderr, "Test FAILED: RMS error in image: %.3f > %.3f, aborting...\n", rms_err, EPS);
        exit(EXIT_FAILURE);
    }
    #else
    float rms_err = 0.0f;   // RMS error
    for (int y=0; y<height-d; y++) {
        for (int x=0; x<width; x++) {
            float derr = reference_out[y*width + x] - output[y*width + x];
            rms_err += derr*derr;

            if (fabs(derr) > EPS) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %f vs. %f\n", x, y,
                        reference_out[y*width + x], output[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    rms_err = sqrtf(rms_err / (float(width*(height-d))));
    // check RMS error
    if (rms_err > EPS) {
        fprintf(stderr, "Test FAILED: RMS error in image: %.3f > %.3f, aborting...\n", rms_err, EPS);
        exit(EXIT_FAILURE);
    }
    #endif
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

