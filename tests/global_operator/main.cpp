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
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

// variables set by Makefile
//#define WIDTH 4096
//#define HEIGHT 4096

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}

// reference
template<typename data_t>
data_t calc_min_pixel(data_t *in, data_t neutral, int width, int height, int
        offset_x, int offset_y, int is_width, int is_height) {
    data_t min = neutral;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            if (in[x + y*width] < min) min = in[x + y*width];
        }
    }

    return min;
}
template<typename data_t>
data_t calc_min_pixel(data_t *in, data_t neutral, int width, int height) {
    return calc_min_pixel<data_t>(in, neutral, width, height, 0, 0, width,
            height);
}

template<typename data_t>
float calc_max_pixel(data_t *in, data_t neutral, int width, int height, int
        offset_x, int offset_y, int is_width, int is_height) {
    data_t max = neutral;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            if (in[x + y*width] > max) max = in[x + y*width];
        }
    }

    return max;
}
template<typename data_t>
float calc_max_pixel(data_t *in, data_t neutral, int width, int height) {
    return calc_max_pixel<data_t>(in, neutral, width, height, 0, 0, width,
            height);
}

template<typename data_t>
data_t calc_sum_pixel(data_t *in, data_t neutral, int width, int height, int
        offset_x, int offset_y, int is_width, int is_height) {
    data_t sum = neutral;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            sum += in[x + y*width];
        }
    }

    return sum;
}
template<typename data_t>
data_t calc_sum_pixel(data_t *in, data_t neutral, int width, int height) {
    return calc_sum_pixel<data_t>(in, neutral, width, height, 0, 0, width,
            height);
}


// Kernel description in HIPAcc
template<typename data_t>
struct MinReduction : public GlobalReduction<data_t> {
    public:
        using GlobalReduction<data_t>::reduce;

        MinReduction(Image<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 
        MinReduction(Accessor<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 

        data_t reduce(data_t left, data_t right) {
            if (left < right) return left;
            else return right;
        }
};
template<typename data_t>
struct MaxReduction : public GlobalReduction<data_t> {
    public:
        using GlobalReduction<data_t>::reduce;

        MaxReduction(Image<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 
        MaxReduction(Accessor<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 

        data_t reduce(data_t left, data_t right) {
            if (left > right) return left;
            else return right;
        }
};
template<typename data_t>
struct SumReduction : public GlobalReduction<data_t> {
    public:
        using GlobalReduction<data_t>::reduce;

        SumReduction(Image<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 
        SumReduction(Accessor<data_t> &img, data_t neutral) :
            GlobalReduction<data_t>(img, neutral)
        {} 

        data_t reduce(data_t left, data_t right) {
            return left + right;
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of of widthxheight pixels
    int *host_in_int = (int *)malloc(sizeof(int)*width*height);
    int *host_out_int = (int *)malloc(sizeof(int)*width*height);
    int *reference_in_int = (int *)malloc(sizeof(int)*width*height);
    int *reference_out_int = (int *)malloc(sizeof(int)*width*height);
    float *host_in_float = (float *)malloc(sizeof(float)*width*height);
    float *host_out_float = (float *)malloc(sizeof(float)*width*height);
    float *reference_in_float = (float *)malloc(sizeof(float)*width*height);
    float *reference_out_float = (float *)malloc(sizeof(float)*width*height);

    // input and output image of widthxheight pixels
    Image<int> INInt(width, height);
    Image<int> OUTInt(width, height);
    Image<float> INFloat(width, height);
    Image<float> OUTFloat(width, height);

    Accessor<int> AccInInt(INInt, width/3, height/3, width/3, height/3);
    Accessor<float> AccInFloat(INFloat, width/3, height/3, width/3, height/3);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in_int[y*width + x] = (int) (x*height + y) * DELTA;
            reference_in_int[y*width + x] = (int) (x*height + y) * DELTA;
            host_out_int[y*width + x] = (int) (3.12451);
            reference_out_int[y*width + x] = (int) (3.12451);
            host_in_float[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in_float[y*width + x] = (float) (x*height + y) * DELTA;
            host_out_float[y*width + x] = (float) (3.12451);
            reference_out_float[y*width + x] = (float) (3.12451);
        }
    }

    INInt = host_in_int;
    OUTInt = host_out_int;
    INFloat = host_in_float;
    OUTFloat = host_out_float;

    // global operation using functors: Images
    MinReduction<int> redMinINInt(INInt, INT_MAX);
    MaxReduction<int> redMaxINInt(INInt, INT_MIN);
    SumReduction<int> redSumINInt(INInt, 0);
    MinReduction<float> redMinINFloat(INFloat, FLT_MAX);
    MaxReduction<float> redMaxINFloat(INFloat, FLT_MIN);
    SumReduction<float> redSumINFloat(INFloat, 0.0f);
    // global operation using functors: Accessors
    MinReduction<int> redMinAccInInt(AccInInt, INT_MAX);
    MaxReduction<int> redMaxAccInInt(AccInInt, INT_MIN);
    SumReduction<int> redSumAccInInt(AccInInt, 0);
    MinReduction<float> redMinAccInFloat(AccInFloat, FLT_MAX);
    MaxReduction<float> redMaxAccInFloat(AccInFloat, FLT_MIN);
    SumReduction<float> redSumAccInFloat(AccInFloat, 0.0f);

    // warmup
    redMinINInt.reduce();
    redMaxINInt.reduce();
    redSumINInt.reduce();
    redMinINFloat.reduce();
    redMaxINFloat.reduce();
    redSumINFloat.reduce();
    redMinAccInInt.reduce();
    redMaxAccInInt.reduce();
    redSumAccInInt.reduce();
    redMinAccInFloat.reduce();
    redMaxAccInFloat.reduce();
    redSumAccInFloat.reduce();

    fprintf(stderr, "Calculating global reductions ...\n");
    time0 = time_ms();

#define FUNCTORS
#ifdef FUNCTORS
    // Images
    int min_pixel_functor_int_img = redMinINInt.reduce();
    fprintf(stderr, "reduction functor, min (img, int): %d\n", min_pixel_functor_int_img);

    int max_pixel_functor_int_img = redMaxINInt.reduce();
    fprintf(stderr, "reduction functor, max (img, int): %d\n", max_pixel_functor_int_img);

    int sum_pixel_functor_int_img = redSumINInt.reduce();
    fprintf(stderr, "reduction functor, sum (img, int): %d\n", sum_pixel_functor_int_img);

    float min_pixel_functor_float_img = redMinINFloat.reduce();
    fprintf(stderr, "reduction functor, min (img, float): %f\n", min_pixel_functor_float_img);

    float max_pixel_functor_float_img = redMaxINFloat.reduce();
    fprintf(stderr, "reduction functor, max (img, float): %f\n", max_pixel_functor_float_img);

    float sum_pixel_functor_float_img = redSumINFloat.reduce();
    fprintf(stderr, "reduction functor, sum (img, float): %f\n", sum_pixel_functor_float_img);

    // Accessors
    int min_pixel_functor_int_acc = redMinAccInInt.reduce();
    fprintf(stderr, "reduction functor, min (acc, int): %d\n", min_pixel_functor_int_acc);

    int max_pixel_functor_int_acc = redMaxAccInInt.reduce();
    fprintf(stderr, "reduction functor, max (acc, int): %d\n", max_pixel_functor_int_acc);

    int sum_pixel_functor_int_acc = redSumAccInInt.reduce();
    fprintf(stderr, "reduction functor, sum (acc, int): %d\n", sum_pixel_functor_int_acc);

    float min_pixel_functor_float_acc = redMinAccInFloat.reduce();
    fprintf(stderr, "reduction functor, min (acc, float): %f\n", min_pixel_functor_float_acc);

    float max_pixel_functor_float_acc = redMaxAccInFloat.reduce();
    fprintf(stderr, "reduction functor, max (acc, float): %f\n", max_pixel_functor_float_acc);

    float sum_pixel_functor_float_acc = redSumAccInFloat.reduce();
    fprintf(stderr, "reduction functor, sum (acc, float): %f\n", sum_pixel_functor_float_acc);
#else
    // global operation
    float min_pixel = reduce(IN, FLT_MAX, [&] (float left, float right) {
            if (left < right) return left;
            else return right;
            });
    fprintf(stderr, "reduction, min: %f\n", min_pixel);
    float max_pixel = reduce(IN, FLT_MIN, [&] (float left, float right) {
            if (left > right) return left;
            else return right;
            });
    fprintf(stderr, "reduction, max: %f\n", max_pixel);
    float sum_pixel = reduce(IN, 0.0f, [&] (float left, float right) {
            return left + right;
            });
    fprintf(stderr, "reduction, sum: %f\n", sum_pixel);
#endif

    time1 = time_ms();
    dt = time1 - time0;

    // get results
    host_out_int = OUTInt.getData();
    host_out_float = OUTFloat.getData();


    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    // NB: actually there are (width-d)*(height) output pixels
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", dt,
            ((width*height)/dt)/1000);


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference: Images
    int min_pixel_ref_int_img = calc_min_pixel(host_in_int, INT_MAX, width, height);
    int max_pixel_ref_int_img = calc_max_pixel(host_in_int, INT_MIN, width, height);
    int sum_pixel_ref_int_img = calc_sum_pixel(host_in_int, 0, width, height);
    float min_pixel_ref_float_img = calc_min_pixel(host_in_float, FLT_MAX, width, height);
    float max_pixel_ref_float_img = calc_max_pixel(host_in_float, FLT_MIN, width, height);
    float sum_pixel_ref_float_img = calc_sum_pixel(host_in_float, 0.0f, width, height);

    // calculate reference: Accessors
    int min_pixel_ref_int_acc = calc_min_pixel(host_in_int, INT_MAX, width, height, width/3, height/3, width/3, height/3);
    int max_pixel_ref_int_acc = calc_max_pixel(host_in_int, INT_MIN, width, height, width/3, height/3, width/3, height/3);
    int sum_pixel_ref_int_acc = calc_sum_pixel(host_in_int, 0, width, height, width/3, height/3, width/3, height/3);
    float min_pixel_ref_float_acc = calc_min_pixel(host_in_float, FLT_MAX, width, height, width/3, height/3, width/3, height/3);
    float max_pixel_ref_float_acc = calc_max_pixel(host_in_float, FLT_MIN, width, height, width/3, height/3, width/3, height/3);
    float sum_pixel_ref_float_acc = calc_sum_pixel(host_in_float, 0.0f, width, height, width/3, height/3, width/3, height/3);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt,
            (width*height/dt)/1000);

    // compare results: Images
    bool passed_all = true;
    fprintf(stderr, "\nComparing results ...\n");
    if (min_pixel_functor_int_img != min_pixel_ref_int_img) {
        fprintf(stderr, "Test FAILED for min reduction (img, int): %d vs %d, aborting...\n", min_pixel_functor_int_img, min_pixel_ref_int_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Min reduction (img, int): PASSED\n");
    }
    if (min_pixel_functor_float_img != min_pixel_ref_float_img) {
        fprintf(stderr, "Test FAILED for min reduction (img, float): %.5f vs %.5f, aborting...\n", min_pixel_functor_float_img, min_pixel_ref_float_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Min reduction (img, float): PASSED\n");
    }

    if (max_pixel_functor_int_img != max_pixel_ref_int_img) {
        fprintf(stderr, "Test FAILED for max reduction (img, int): %d vs %d, aborting...\n", max_pixel_functor_int_img, max_pixel_ref_int_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Max reduction (img, int): PASSED\n");
    }
    if (max_pixel_functor_float_img != max_pixel_ref_float_img) {
        fprintf(stderr, "Test FAILED for max reduction (img, float): %.5f vs %.5f, aborting...\n", max_pixel_functor_float_img, max_pixel_ref_float_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Max reduction (img, float): PASSED\n");
    }

    if (sum_pixel_functor_int_img != sum_pixel_ref_int_img) {
        fprintf(stderr, "Test FAILED for sum reduction (img, int): %d vs %d, aborting...\n", sum_pixel_functor_int_img, sum_pixel_ref_int_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Sum reduction (img, int): PASSED\n");
    }
    if (sum_pixel_functor_float_img != sum_pixel_ref_float_img) {
        fprintf(stderr, "Test FAILED for sum reduction (img, float): %.5f vs %.5f, aborting...\n", sum_pixel_functor_float_img, sum_pixel_ref_float_img);
        passed_all = false;
    } else {
        fprintf(stderr, "Sum reduction (img, float): PASSED\n");
    }
    // compare results: Accessors
    if (min_pixel_functor_int_acc != min_pixel_ref_int_acc) {
        fprintf(stderr, "Test FAILED for min reduction (acc, int): %d vs %d, aborting...\n", min_pixel_functor_int_acc, min_pixel_ref_int_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Min reduction (acc, int): PASSED\n");
    }
    if (min_pixel_functor_float_acc != min_pixel_ref_float_acc) {
        fprintf(stderr, "Test FAILED for min reduction (acc, float): %.5f vs %.5f, aborting...\n", min_pixel_functor_float_acc, min_pixel_ref_float_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Min reduction (acc, float): PASSED\n");
    }

    if (max_pixel_functor_int_acc != max_pixel_ref_int_acc) {
        fprintf(stderr, "Test FAILED for max reduction (acc, int): %d vs %d, aborting...\n", max_pixel_functor_int_acc, max_pixel_ref_int_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Max reduction (acc, int): PASSED\n");
    }
    if (max_pixel_functor_float_acc != max_pixel_ref_float_acc) {
        fprintf(stderr, "Test FAILED for max reduction (acc, float): %.5f vs %.5f, aborting...\n", max_pixel_functor_float_acc, max_pixel_ref_float_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Max reduction (acc, float): PASSED\n");
    }

    if (sum_pixel_functor_int_acc != sum_pixel_ref_int_acc) {
        fprintf(stderr, "Test FAILED for sum reduction (acc, int): %d vs %d, aborting...\n", sum_pixel_functor_int_acc, sum_pixel_ref_int_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Sum reduction (acc, int): PASSED\n");
    }
    if (sum_pixel_functor_float_acc != sum_pixel_ref_float_acc) {
        fprintf(stderr, "Test FAILED for sum reduction (acc, float): %.5f vs %.5f, aborting...\n", sum_pixel_functor_float_acc, sum_pixel_ref_float_acc);
        passed_all = false;
    } else {
        fprintf(stderr, "Sum reduction (acc, float): PASSED\n");
    }
    // print final result
    if (passed_all) {
        fprintf(stderr, "Tests PASSED\n");
    } else {
        fprintf(stderr, "Tests FAILED\n");
        exit(EXIT_FAILURE);
    }

    // memory cleanup
    free(host_in_int);
    free(host_in_float);
    //free(host_out_int);
    //free(host_out_float);
    free(reference_in_int);
    free(reference_in_float);
    free(reference_out_int);
    free(reference_out_float);

    return EXIT_SUCCESS;
}

