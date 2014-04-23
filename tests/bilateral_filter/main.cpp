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
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

// variables set by Makefile
#define SIGMA_D 3
//#define SIGMA_R 5
//#define WIDTH 4096
//#define HEIGHT 4096
#define CONST_MASK
#define USE_LAMBDA
//#define SIGMA_D SIZE_X
#define SIGMA_R SIZE_Y
#define CONVOLUTION_MASK
#define EPS 0.02f
#define CONSTANT 1.0f

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}

enum BoundaryMode {
    BH_UNDEFINED,
    BH_CLAMP,
    BH_REPEAT,
    BH_MIRROR,
    BH_CONSTANT
};

// wrapper function for border handling
template<typename data_t> data_t get_data(data_t *array, int x, int y, int width, int height, BoundaryMode mode) {
    data_t ret = CONSTANT;

    switch (mode) {
        default:
        case BH_UNDEFINED:
            ret = array[x + y*width];
            break;
        case BH_CLAMP:
            x = min(max(x, 0), width-1);
            y = min(max(y, 0), height-1);
            ret = array[x + y*width];
            break;
        case BH_REPEAT:
            while (x >= width) x -= width;
            while (y >= height) y -= height;
            while (x < 0) x += width;
            while (y < 0) y += height;
            ret = array[x + y*width];
            break;
        case BH_MIRROR:
            if (x < 0) x = -x - 1;
            if (y < 0) y = -y - 1;
            if (x >= width) x = width - (x+1 - width);
            if (y >= height) y = height - (y+1 - height);
            ret = array[x + y*width];
            break;
        case BH_CONSTANT:
            if (x < 0 || y < 0 || x >= width || y >= height) {
                ret = CONSTANT;
            } else {
                ret = array[x + y*width];
            }
            break;
    }

    return ret;
}


// bilateral filter reference
void bilateral_filter(float *in, float *out, int sigma_d, int sigma_r, int
        width, int height, BoundaryMode mode) {
    float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
    float c_d = 1.0f/(2.0f*sigma_d*sigma_d);

    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = y + yf;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = x + xf;
                    float diff = get_data(in, ix, iy, width, height, mode) - in[y*width + x];

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    p += s * get_data(in, ix, iy, width, height, mode);
                }
            }
            out[y*width + x] = p/d;
        }
    }
}


// Kernel description in HIPAcc
class BilateralFilter : public Kernel<float> {
    private:
        Accessor<float> &input;
        int sigma_d;
        int sigma_r;

    public:
        BilateralFilter(IterationSpace<float> &iter, Accessor<float> &input, int
                sigma_d, int sigma_r) :
            Kernel(iter),
            input(input),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        { addAccessor(&input); }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = input(xf, yf) - input();

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    p += s * input(xf, yf);
                }
            }

            output() = p/d;
        }
};

class BilateralFilterMask : public Kernel<float> {
    private:
        Accessor<float> &input;
        Mask<float> &mask;
        Domain &dom;
        int sigma_d, sigma_r;

    public:
        BilateralFilterMask(IterationSpace<float> &iter, Accessor<float> &input,
                Mask<float> &mask, Domain &dom, int sigma_d, int sigma_r) :
            Kernel(iter),
            input(input),
            mask(mask),
            dom(dom),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        { addAccessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float d = 0;
            float p = 0;

            iterate(dom, [&] () -> void {
                    float diff = input(dom) - input();

                    float s = expf(-c_r * diff*diff) * mask(dom);
                    d += s;
                    p += s * input(dom);
                    });

            output() = p/d;
        }
        #else
        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = input(xf, yf) - input();

                    float s = expf(-c_r * diff*diff) * mask(xf, yf);
                    d += s;
                    p += s * input(xf, yf);
                }
            }

            output() = p/d;
        }
        #endif
};

class BilateralFilterBHCLAMP : public Kernel<float> {
    private:
        Accessor<float> &input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHCLAMP(IterationSpace<float> &iter, Accessor<float>
                &input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(iter),
            input(input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        { addAccessor(&input); }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = input.getY() + yf;
                iy = min(max(iy, 0), height-1);
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = input.getX() + xf;
                    ix = min(max(ix, 0), width-1);
                    float diff = input.getPixel(ix, iy) - input();

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    p += s * input.getPixel(ix, iy);
                }
            }

            output() = p/d;
        }
};

class BilateralFilterBHREPEAT : public Kernel<float> {
    private:
        Accessor<float> &input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHREPEAT(IterationSpace<float> &iter, Accessor<float>
                &input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(iter),
            input(input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        { addAccessor(&input); }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = input.getY() + yf;
                while (iy < 0) iy += height;
                while (iy >= height) iy -= height;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = input.getX() + xf;
                    while (ix < 0) ix += width;
                    while (ix >= width) ix -= width;
                    float diff = input.getPixel(ix, iy) - input();

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    p += s * input.getPixel(ix, iy);
                }
            }

            output() = p/d;
        }
};

class BilateralFilterBHMIRROR : public Kernel<float> {
    private:
        Accessor<float> &input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHMIRROR(IterationSpace<float> &iter, Accessor<float>
                &input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(iter),
            input(input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        { addAccessor(&input); }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = input.getY() + yf;
                if (iy < 0) iy = -iy - 1;
                if (iy >= height) iy = height - (iy+1 - height);
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = input.getX() + xf;
                    if (ix < 0) ix = -ix - 1;
                    if (ix >= width) ix = width - (ix+1 - width);
                    float diff = input.getPixel(ix, iy) - input();

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    p += s * input.getPixel(ix, iy);
                }
            }

            output() = p/d;
        }
};

class BilateralFilterBHCONSTANT : public Kernel<float> {
    private:
        Accessor<float> &input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHCONSTANT(IterationSpace<float> &iter, Accessor<float>
                &input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(iter),
            input(input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        { addAccessor(&input); }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0;
            float p = 0;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = input.getY() + yf;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = input.getX() + xf;
                    float diff;
                    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
                        diff = CONSTANT;
                    } else {
                        diff = input.getPixel(ix, iy) - input();
                    }

                    float s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) *
                        expf(-c_d * yf*yf);
                    d += s;
                    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
                        p += s * CONSTANT;
                    } else {
                        p += s * input.getPixel(ix, iy);
                    }
                }
            }

            output() = p/d;
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;
    const int sigma_d = SIGMA_D;
    const int sigma_r = SIGMA_R;
    float timing = 0.0f;

    #ifdef CONST_MASK
    const float filter_mask[4*sigma_d+1][4*sigma_d+1] = {
        #if SIGMA_D==1
        { 0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f },
        { 0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f },
        { 0.135335f, 0.606531f, 1.000000f, 0.606531f, 0.135335f },
        { 0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f },
        { 0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f }
        #endif
        #if SIGMA_D==2
        { 0.018316f, 0.043937f, 0.082085f, 0.119433f, 0.135335f, 0.119433f, 0.082085f, 0.043937f, 0.018316f },
        { 0.043937f, 0.105399f, 0.196912f, 0.286505f, 0.324652f, 0.286505f, 0.196912f, 0.105399f, 0.043937f },
        { 0.082085f, 0.196912f, 0.367879f, 0.535261f, 0.606531f, 0.535261f, 0.367879f, 0.196912f, 0.082085f },
        { 0.119433f, 0.286505f, 0.535261f, 0.778801f, 0.882497f, 0.778801f, 0.535261f, 0.286505f, 0.119433f },
        { 0.135335f, 0.324652f, 0.606531f, 0.882497f, 1.000000f, 0.882497f, 0.606531f, 0.324652f, 0.135335f },
        { 0.119433f, 0.286505f, 0.535261f, 0.778801f, 0.882497f, 0.778801f, 0.535261f, 0.286505f, 0.119433f },
        { 0.082085f, 0.196912f, 0.367879f, 0.535261f, 0.606531f, 0.535261f, 0.367879f, 0.196912f, 0.082085f },
        { 0.043937f, 0.105399f, 0.196912f, 0.286505f, 0.324652f, 0.286505f, 0.196912f, 0.105399f, 0.043937f },
        { 0.018316f, 0.043937f, 0.082085f, 0.119433f, 0.135335f, 0.119433f, 0.082085f, 0.043937f, 0.018316f }
        #endif
        #if SIGMA_D==3
        { 0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f, 0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f, 0.018316f },
        { 0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f, 0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f, 0.033746f },
        { 0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f, 0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f, 0.055638f },
        { 0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f, 0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f, 0.082085f },
        { 0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f, 0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f, 0.108368f },
        { 0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f, 0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f, 0.128022f },
        { 0.135335f, 0.249352f, 0.411112f, 0.606531f, 0.800737f, 0.945959f, 1.000000f, 0.945959f, 0.800737f, 0.606531f, 0.411112f, 0.249352f, 0.135335f },
        { 0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f, 0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f, 0.128022f },
        { 0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f, 0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f, 0.108368f },
        { 0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f, 0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f, 0.082085f },
        { 0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f, 0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f, 0.055638f },
        { 0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f, 0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f, 0.033746f },
        { 0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f, 0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f, 0.018316f }
        #endif
    };
    #else
    float filter_mask[2*2*sigma_d+1][2*2*sigma_d+1];
    float mask_tmp[2*2*sigma_d+1];
    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
        mask_tmp[xf+2*sigma_d] = expf(-1/(2.0f*sigma_d*sigma_d)*(xf*xf));
    }
    for (int yf=-2*sigma_d; yf<=2*sigma_d; yf++) {
        for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
            filter_mask[yf+2*sigma_d][xf+2*sigma_d] = mask_tmp[yf+2*sigma_d] * mask_tmp[xf+2*SIGMA_D];
            fprintf(stderr, "%ff, ", filter_mask[yf+2*sigma_d][xf+2*sigma_d]);
        }
        fprintf(stderr, "\n");
    }
    #endif
    Mask<float> mask(filter_mask);

    // define Domain for blur filter
    Domain dom(4*sigma_d+1, 4*sigma_d+1);


    // host memory for image of width x height pixels
    float *input = (float *)malloc(sizeof(float)*width*height);
    float *reference_in = (float *)malloc(sizeof(float)*width*height);
    float *reference_out = (float *)malloc(sizeof(float)*width*height);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in[y*width + x] = (float) (x*height + y) * DELTA;
            reference_out[y*width + x] = (float) (-3.12451);
        }
    }


    // input and output image of width x height pixels
    Image<float> in(width, height);
    Image<float> out(width, height);

    // iteration space
    IterationSpace<float> iter(out);

    in = input;

    fprintf(stderr, "Calculating HIPAcc bilateral filter ...\n");

    // Image only
    Accessor<float> AccInUndef(in);
    #ifdef CONVOLUTION_MASK
    BilateralFilterMask BFNOBH(iter, AccInUndef, mask, dom, sigma_d, sigma_r);
    #else
    BilateralFilter BFNOBH(iter, AccInUndef, sigma_d, sigma_r);
    #endif


    BFNOBH.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc (NOBH): %.3f ms, %.3f Mpixel/s\n\n", timing, (width*height/timing)/1000);


    // BOUNDARY_CLAMP
    BoundaryCondition<float> BcInClamp(in, mask, BOUNDARY_CLAMP);
    Accessor<float> AccInClamp(BcInClamp);
    #ifdef CONVOLUTION_MASK
    BilateralFilterMask BF(iter, AccInClamp, mask, dom, sigma_d, sigma_r);
    #else
    BilateralFilter BF(iter, AccInClamp, sigma_d, sigma_r);
    #endif

    BF.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n\n", timing, (width*height/timing)/1000);


    // get pointer to result data
    float *output = out.getData();


    // manual border handling: CLAMP
    Accessor<float> AccIn(in);
    BilateralFilterBHCLAMP BFBHCLAMP(iter, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");

    BFBHCLAMP.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc(BHCLAMP): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // manual border handling: REPEAT
    BilateralFilterBHREPEAT BFBHREPEAT(iter, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");

    BFBHREPEAT.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc(BHREPEAT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // manual border handling: MIRROR
    BilateralFilterBHMIRROR BFBHMIRROR(iter, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");

    BFBHMIRROR.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc(BHMIRROR): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // manual border handling: CONSTANT
    BilateralFilterBHCONSTANT BFBHCONSTANT(iter, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");

    BFBHCONSTANT.execute();
    timing = hipaccGetLastKernelTiming();

    fprintf(stderr, "Hipacc(BHCONSTANT): %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference
    bilateral_filter(reference_in, reference_out, sigma_d, sigma_r, width, height, BH_CLAMP);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*height/dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
    // compare results
    float rms_err = 0.0f;   // RMS error
    for (int y=0; y<height; y++) {
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
    rms_err = sqrtf(rms_err / ((float)(width*height)));
    // check RMS error
    if (rms_err > EPS) {
        fprintf(stderr, "Test FAILED: RMS error in image: %.3f > %.3f, aborting...\n", rms_err, EPS);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

