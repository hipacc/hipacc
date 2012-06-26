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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

// variables set by Makefile
//#define SIGMA_D 3
//#define SIGMA_R 5
//#define WIDTH 4096
//#define HEIGHT 4096
#define SIGMA_D SIZE_X
#define SIGMA_R SIZE_Y
#define CONVOLUTION_MASK
#define EPS 0.02f
#define CONSTANT 1.0f

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// wrapper function for border handling
template<typename data_t> data_t get_data(data_t *array, int x, int y, int width, int height, hipaccBoundaryMode mode) {
    data_t ret = CONSTANT;

    switch (mode) {
        default:
        case BOUNDARY_UNDEFINED:
            ret = array[x + y*width];
            break;
        case BOUNDARY_CLAMP:
            x = std::min(std::max(x, 0), width-1);
            y = std::min(std::max(y, 0), height-1);
            ret = array[x + y*width];
            break;
        case BOUNDARY_REPEAT:
            while (x >= width) x -= width;
            while (y >= height) y -= height;
            while (x < 0) x += width;
            while (y < 0) y += height;
            ret = array[x + y*width];
            break;
        case BOUNDARY_MIRROR:
            if (x < 0) x = -x - 1;
            if (y < 0) y = -y - 1;
            if (x >= width) x = width - (x+1 - width);
            if (y >= height) y = height - (y+1 - height);
            ret = array[x + y*width];
            break;
        case BOUNDARY_CONSTANT:
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
        width, int height, hipaccBoundaryMode mode) {
    float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
    float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
    float s = 0.0f;

    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            float d = 0.0f;
            float p = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = y + yf;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = x + xf;
                    float diff = get_data(in, ix, iy, width, height, mode) - in[y*width + x];

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * get_data(in, ix, iy, width, height, mode);
                }
            }
            out[y*width + x] = (float) (p / d);
        }
    }
}


namespace hipacc {
class BilateralFilterNOBH : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;

    public:
        BilateralFilterNOBH(IterationSpace<float> &IS, Accessor<float> &Input,
                int sigma_d, int sigma_r) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = Input(xf, yf) - Input();

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * Input(xf, yf);
                }
            }
            output() = (float) (p / d);
        }
};

class BilateralFilterMaskNOBH : public Kernel<float> {
    private:
        Accessor<float> &Input;
        Mask<float> &sMask;
        int sigma_d, sigma_r;

    public:
        BilateralFilterMaskNOBH(IterationSpace<float> &IS, Accessor<float>
                &Input, Mask<float> &sMask, int sigma_d, int sigma_r) :
            Kernel(IS),
            Input(Input),
            sMask(sMask),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float d = 0.0f;
            float p = 0.0f;

            #if 0
            d = convolve(sMask, HipaccSUM, [&] () -> float {
                    float diff = Input(sMask) - Input();
                    return expf(-c_r * diff*diff) * sMask();
                    });
            p = convolve(sMask, HipaccSUM, [&] () -> float {
                    float diff = Input(sMask) - Input();
                    return expf(-c_r * diff*diff) * sMask() * Input(sMask);
                    });
            #else
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = Input(xf, yf) - Input();

                    s = expf(-c_r * diff*diff) * sMask(2*sigma_d + xf, 2*sigma_d + yf);
                    d += s;
                    p += s * Input(xf, yf);
                }
            }
            #endif

            output() = (float) (p / d);
        }
};

class BilateralFilter : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;

    public:
        BilateralFilter(IterationSpace<float> &IS, Accessor<float> &Input, int
                sigma_d, int sigma_r) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = Input(xf, yf) - Input();

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * Input(xf, yf);
                }
            }
            output() = (float) (p / d);
        }
};

class BilateralFilterMask : public Kernel<float> {
    private:
        Accessor<float> &Input;
        Mask<float> &sMask;
        int sigma_d, sigma_r;

    public:
        BilateralFilterMask(IterationSpace<float> &IS, Accessor<float> &Input,
                Mask<float> &sMask, int sigma_d, int sigma_r) :
            Kernel(IS),
            Input(Input),
            sMask(sMask),
            sigma_d(sigma_d),
            sigma_r(sigma_r)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float d = 0.0f;
            float p = 0.0f;

            #if 0
            d = convolve(sMask, HipaccSUM, [&] () -> float {
                    float diff = Input(sMask) - Input();
                    return expf(-c_r * diff*diff) * sMask();
                    });
            p = convolve(sMask, HipaccSUM, [&] () -> float {
                    float diff = Input(sMask) - Input();
                    return expf(-c_r * diff*diff) * sMask() * Input(sMask);
                    });
            #else
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    float diff = Input(xf, yf) - Input();

                    s = expf(-c_r * diff*diff) * sMask(2*sigma_d + xf, 2*sigma_d + yf);
                    d += s;
                    p += s * Input(xf, yf);
                }
            }
            #endif

            output() = (float) (p / d);
        }
};

class BilateralFilterBHCLAMP : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHCLAMP(IterationSpace<float> &IS, Accessor<float>
                &Input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = Input.getY() + yf;
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height-1;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = Input.getX() + xf;
                    if (ix < 0) ix = 0;
                    if (ix >= width) ix = width-1;
                    float diff = Input.getPixel(ix, iy) - Input();

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * Input.getPixel(ix, iy);
                }
            }
            output() = (float) (p / d);
        }
};

class BilateralFilterBHREPEAT : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHREPEAT(IterationSpace<float> &IS, Accessor<float>
                &Input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = Input.getY() + yf;
                while (iy < 0) iy += height;
                while (iy >= height) iy -= height;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = Input.getX() + xf;
                    while (ix < 0) ix += width;
                    while (ix >= width) ix -= width;
                    float diff = Input.getPixel(ix, iy) - Input();

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * Input.getPixel(ix, iy);
                }
            }
            output() = (float) (p / d);
        }
};

class BilateralFilterBHMIRROR : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHMIRROR(IterationSpace<float> &IS, Accessor<float>
                &Input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = Input.getY() + yf;
                if (iy < 0) iy = -iy - 1;
                if (iy >= height) iy = height - (iy+1 - height);
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = Input.getX() + xf;
                    if (ix < 0) ix = -ix - 1;
                    if (ix >= width) ix = width - (ix+1 - width);
                    float diff = Input.getPixel(ix, iy) - Input();

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    p += s * Input.getPixel(ix, iy);
                }
            }
            output() = (float) (p / d);
        }
};

class BilateralFilterBHCONSTANT : public Kernel<float> {
    private:
        Accessor<float> &Input;
        int sigma_d;
        int sigma_r;
        int width;
        int height;

    public:
        BilateralFilterBHCONSTANT(IterationSpace<float> &IS, Accessor<float>
                &Input, int sigma_d, int sigma_r, int width, int height) :
            Kernel(IS),
            Input(Input),
            sigma_d(sigma_d),
            sigma_r(sigma_r),
            width(width),
            height(height)
        {
            addAccessor(&Input);
        }

        void kernel() {
            float c_r = 1.0f/(2.0f*sigma_r*sigma_r);
            float c_d = 1.0f/(2.0f*sigma_d*sigma_d);
            float d = 0.0f;
            float p = 0.0f;
            float s = 0.0f;

            for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
                int iy = Input.getY() + yf;
                for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                    int ix = Input.getX() + xf;
                    float diff;
                    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
                        diff = CONSTANT;
                    } else {
                        diff = Input.getPixel(ix, iy) - Input();
                    }

                    s = expf(-c_r * diff*diff) * expf(-c_d * xf*xf) * expf(-c_d
                            * yf*yf);
                    d += s;
                    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
                        p += s * CONSTANT;
                    } else {
                        p += s * Input.getPixel(ix, iy);
                    }
                }
            }
            output() = (float) (p / d);
        }
};
}


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;
    const int sigma_d = SIGMA_D;
    const int sigma_r = SIGMA_R;

    // host memory for image of of widthxheight pixels
    float *host_in = (float *)malloc(sizeof(float)*width*height);
    float *host_out = (float *)malloc(sizeof(float)*width*height);
    float *reference_in = (float *)malloc(sizeof(float)*width*height);
    float *reference_out = (float *)malloc(sizeof(float)*width*height);

#if 0
    float gaussian_d[2*2*sigma_d+1][2*2*sigma_d+1];
    float gaussian[2*2*sigma_d+1];
    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
        gaussian[xf+2*sigma_d] = expf(-1/(2.0f*sigma_d*sigma_d)*(xf*xf));
    }
    for (int yf=-2*sigma_d; yf<=2*sigma_d; yf++) {
        for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
            gaussian_d[yf+2*sigma_d][xf+2*sigma_d] = gaussian[yf+2*sigma_d] * gaussian[xf+2*SIGMA_D];
            fprintf(stderr, "%f, ", gaussian_d[yf+2*sigma_d][xf+2*sigma_d]);
        }
        fprintf(stderr, "\n");
    }
#endif
#if SIGMA_D==1
    const float mask[] = {
        0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f, 
        0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f, 
        0.135335f, 0.606531f, 1.000000f, 0.606531f, 0.135335f, 
        0.082085f, 0.367879f, 0.606531f, 0.367879f, 0.082085f, 
        0.018316f, 0.082085f, 0.135335f, 0.082085f, 0.018316f, 
    };
#endif
#if SIGMA_D==3
    const float mask[] = {
        0.018316, 0.033746, 0.055638, 0.082085, 0.108368, 0.128022, 0.135335, 0.128022, 0.108368, 0.082085, 0.055638, 0.033746, 0.018316, 
        0.033746, 0.062177, 0.102512, 0.151240, 0.199666, 0.235877, 0.249352, 0.235877, 0.199666, 0.151240, 0.102512, 0.062177, 0.033746, 
        0.055638, 0.102512, 0.169013, 0.249352, 0.329193, 0.388896, 0.411112, 0.388896, 0.329193, 0.249352, 0.169013, 0.102512, 0.055638, 
        0.082085, 0.151240, 0.249352, 0.367879, 0.485672, 0.573753, 0.606531, 0.573753, 0.485672, 0.367879, 0.249352, 0.151240, 0.082085, 
        0.108368, 0.199666, 0.329193, 0.485672, 0.641180, 0.757465, 0.800737, 0.757465, 0.641180, 0.485672, 0.329193, 0.199666, 0.108368, 
        0.128022, 0.235877, 0.388896, 0.573753, 0.757465, 0.894839, 0.945959, 0.894839, 0.757465, 0.573753, 0.388896, 0.235877, 0.128022, 
        0.135335, 0.249352, 0.411112, 0.606531, 0.800737, 0.945959, 1.000000, 0.945959, 0.800737, 0.606531, 0.411112, 0.249352, 0.135335, 
        0.128022, 0.235877, 0.388896, 0.573753, 0.757465, 0.894839, 0.945959, 0.894839, 0.757465, 0.573753, 0.388896, 0.235877, 0.128022, 
        0.108368, 0.199666, 0.329193, 0.485672, 0.641180, 0.757465, 0.800737, 0.757465, 0.641180, 0.485672, 0.329193, 0.199666, 0.108368, 
        0.082085, 0.151240, 0.249352, 0.367879, 0.485672, 0.573753, 0.606531, 0.573753, 0.485672, 0.367879, 0.249352, 0.151240, 0.082085, 
        0.055638, 0.102512, 0.169013, 0.249352, 0.329193, 0.388896, 0.411112, 0.388896, 0.329193, 0.249352, 0.169013, 0.102512, 0.055638, 
        0.033746, 0.062177, 0.102512, 0.151240, 0.199666, 0.235877, 0.249352, 0.235877, 0.199666, 0.151240, 0.102512, 0.062177, 0.033746, 
        0.018316, 0.033746, 0.055638, 0.082085, 0.108368, 0.128022, 0.135335, 0.128022, 0.108368, 0.082085, 0.055638, 0.033746, 0.018316, 
    };
#endif
    Mask<float> M(4*sigma_d+1, 4*sigma_d+1);
    M = mask;

    const char umask[] = {1u, '2', 3U, 4L, 7, 6, 7, 8, 9};
    Mask<char> uM(3, 3);
    uM = umask;

    // input and output image of widthxheight pixels
    Image<float> IN(width, height);
    Image<float> OUT(width, height);

    // iteration space
    IterationSpace<float> BIS(OUT);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in[y*width + x] = (float) (x*height + y) * DELTA;
            host_out[y*width + x] = (float) (3.12451);
            reference_out[y*width + x] = (float) (-3.12451);
        }
    }

    IN = host_in;
    OUT = host_out;
    fprintf(stderr, "Calculating bilateral filter ...\n");

    // Image only
    Accessor<float> AccInUndef(IN);
#ifdef CONVOLUTION_MASK
    BilateralFilterMaskNOBH BFNOBH(BIS, AccInUndef, M, sigma_d, sigma_r);
#else
    BilateralFilterNOBH BFNOBH(BIS, AccInUndef, sigma_d, sigma_r);
#endif

    // warmup
    BFNOBH.execute();

    time0 = time_ms();

    BFNOBH.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc (NOBH): %.3f ms, %.3f Mpixel/s\n\n", dt,
            (width*height/dt)/1000);


    // BOUNDARY_CLAMP
    BoundaryCondition<float> BcInClamp(IN, M, BOUNDARY_CLAMP);
    Accessor<float> AccInClamp(BcInClamp);
#ifdef CONVOLUTION_MASK
    BilateralFilterMask BF(BIS, AccInClamp, M, sigma_d, sigma_r);
#else
    BilateralFilter BF(BIS, AccInClamp, sigma_d, sigma_r);
#endif

    time0 = time_ms();

    BF.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n\n", dt,
            (width*height/dt)/1000);


    // get results
    host_out = OUT.getData();


#if 1
    // manual border handling: CLAMP
    Accessor<float> AccIn(IN);
    BilateralFilterBHCLAMP BFBHCLAMP(BIS, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");
    time0 = time_ms();

    BFBHCLAMP.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc(BHCLAMP): %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);


    // manual border handling: REPEAT
    BilateralFilterBHREPEAT BFBHREPEAT(BIS, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");
    time0 = time_ms();

    BFBHREPEAT.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc(BHREPEAT): %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);


    // manual border handling: MIRROR
    BilateralFilterBHMIRROR BFBHMIRROR(BIS, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");
    time0 = time_ms();

    BFBHMIRROR.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc(BHMIRROR): %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);


    // manual border handling: CONSTANT
    BilateralFilterBHCONSTANT BFBHCONSTANT(BIS, AccIn, sigma_d, sigma_r, width, height);

    fprintf(stderr, "Calculating bilateral filter with manual border handling ...\n");
    time0 = time_ms();

    BFBHCONSTANT.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    fprintf(stderr, "Hipacc(BHCONSTANT): %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);
#endif


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference
    bilateral_filter(reference_in, reference_out, sigma_d, sigma_r, width, height, BOUNDARY_CLAMP);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);


    fprintf(stderr, "\nComparing results ...\n");
    // compare results
    float rms_err = 0.0f;   // RMS error
    for (int y=0; y<height; y++) {
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
    rms_err = sqrtf(rms_err / ((float)(width*height)));
    // check RMS error
    if (rms_err > EPS) {
        fprintf(stderr, "Test FAILED: RMS error in image: %.5f > %.5f, aborting...\n", rms_err, EPS);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(host_in);
    //free(host_out);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

