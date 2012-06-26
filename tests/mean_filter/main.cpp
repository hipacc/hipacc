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

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

#define min(a,b) (((a) < (b)) ? (a) : (b))
//#define HSCAN
//#define SIMPLE
#define USE_GETPIXEL
#define EPS 0.02f

// variables set by Makefile
#define NT 100
#define WIDTH 5120
#define HEIGHT 3200

using namespace hipacc;


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
            float sum = 0.0f;

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
            float sum = 0.0f;

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
            float sum = 0.0f;

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
            float sum = 0.0f;

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


namespace hipacc {
class HorizontalMeanFilter : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int d;

    public:
        HorizontalMeanFilter(IterationSpace<float> &IS, Accessor<float> &Input,
                int d) :
            Kernel(IS),
            Input(Input),
            d(d)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            #ifdef SIMPLE
            for (int k=0; k<d; ++k) {
                sum += Input(k, 0);
            }

            output() = sum/(float)d;
            #else
            int t0 = getX();

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                #ifdef USE_GETPIXEL
                sum += Input.getPixel(k + t0*NT, getY());
                #else
                sum += Input(k + (t0*NT-t0), 0);
                #endif
            }
            #ifdef USE_GETPIXEL
            outputAtPixel(t0*NT, getY()) = sum/(float)d;
            #else
            output((t0*NT-t0), 0) = sum/(float)d;
            #endif

            // second phase: rolling sum
            for (int dt=1; dt<min(NT, WIDTH-d-(t0*NT)); ++dt) {
                #ifdef USE_GETPIXEL
                int t = t0*NT + dt;
                sum -= Input.getPixel(t-1, getY());
                sum += Input.getPixel(t-1+d, getY());
                outputAtPixel(t, getY()) = sum/(float)d;
                #else
                int t = (t0*NT-t0) + dt;
                sum -= Input(t-1, 0);
                sum += Input(t-1+d, 0);
                output(t, 0) = sum/(float)d;
                #endif
            }
            #endif
        }
};

class VerticalMeanFilter : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int d;

    public:
        VerticalMeanFilter(IterationSpace<float> &IS, Accessor<float> &Input,
                int d) :
            Kernel(IS),
            Input(Input),
            d(d)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float sum = 0.0f;

            #ifdef SIMPLE
            for (int k=0; k<d; ++k) {
                sum += Input(0, k);
            }

            output() = sum/(float)d;
            #else
            int t0 = getY();

            // first phase: convolution
            for (int k=0; k<d; ++k) {
                #ifdef USE_GETPIXEL
                sum += Input.getPixel(getX(), k + t0*NT);
                #else
                sum += Input(0, k + (t0*NT-t0));
                #endif
            }
            #ifdef USE_GETPIXEL
            outputAtPixel(getX(), t0*NT) = sum/(float)d;
            #else
            output(0, (t0*NT-t0)) = sum/(float)d;
            #endif

            // second phase: rolling sum
            for (int dt=1; dt<min(NT, HEIGHT-d-(t0*NT)); ++dt) {
                #ifdef USE_GETPIXEL
                int t = t0*NT + dt;
                sum -= Input.getPixel(getX(), t-1);
                sum += Input.getPixel(getX(), t-1+d);
                outputAtPixel(getX(), t) = sum/(float)d;
                #else
                int t = (t0*NT-t0) + dt;
                sum -= Input(0, t-1);
                sum += Input(0, t-1+d);
                output(0, t) = sum/(float)d;
                #endif
            }
            #endif
        }
};
}


int main(int argc, const char **argv) {
    double time0, time1, dt;
    int width = WIDTH;
    int height = HEIGHT;
    int d = 40;
    int t = NT;

    // host memory for image of of width x height pixels
    float *host_in = (float *)malloc(sizeof(float)*width*height);
    float *host_out = (float *)malloc(sizeof(float)*width*height);
    float *reference_in = (float *)malloc(sizeof(float)*width*height);
    float *reference_out = (float *)malloc(sizeof(float)*width*height);

    // input and output image of width x height pixels
    Image<float> IN(width, height);
    Image<float> OUT(width, height);
    BoundaryCondition<float> BcIN(IN, 5, 5, BOUNDARY_MIRROR);
    BoundaryCondition<float> BcIN2(IN, 5, BOUNDARY_MIRROR);
    BoundaryCondition<float> BcIN3(IN, 5, 5, BOUNDARY_CONSTANT, 5.0f);
    BoundaryCondition<float> BcIN4(IN, 5, BOUNDARY_CONSTANT, 5.0f);
    Accessor<float> AccIN(IN);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in[y*width + x] = (float) (x*height + y) * DELTA;
            host_out[y*width + x] = (float) (3.12451);
            reference_out[y*width + x] = (float) (3.12451);
        }
    }


    #ifdef HSCAN
    #ifdef SIMPLE
    IterationSpace<float> HIS(OUT, width-d, height);
    #else
    IterationSpace<float> HIS(OUT, (int)ceil(((float)width-d)/t), height);
    #endif
    HorizontalMeanFilter HMF(HIS, IN, OUT, d);
    #else
    #ifdef SIMPLE
    IterationSpace<float> VIS(OUT, width, height-d);
    #else
    IterationSpace<float> VIS(OUT, width, (int)ceil(((float)height-d)/t));
    #endif
    VerticalMeanFilter VMF(VIS, AccIN, d);
    #endif

    IN = host_in;
    OUT = host_out;

    fprintf(stderr, "Calculating mean filter ...\n");
    time0 = time_ms();

    #ifdef HSCAN
    HMF.execute();
    #else
    VMF.execute();
    #endif

    time1 = time_ms();
    dt = time1 - time0;

    // get results
    host_out = OUT.getData();

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    // NB: actually there are (width-d)*(height) output pixels
    #ifdef HSCAN
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", dt, ((width-d)*height/dt)/1000);
    #else
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", dt, (width*(height-d)/dt)/1000);
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
            float derr = reference_out[y*width + x] - host_out[y*width +x];
            rms_err += derr*derr;

            if (abs(derr) > EPS) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %f vs. %f\n", x, y,
                        reference_out[y*width + x], host_out[y*width +x]);
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
            float derr = reference_out[y*width + x] - host_out[y*width +x];
            rms_err += derr*derr;

            if (abs(derr) > EPS) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %f vs. %f\n", x, y,
                        reference_out[y*width + x], host_out[y*width +x]);
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
    free(host_in);
    //free(host_out);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

