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
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

// variables set by Makefile
//#define WIDTH 4096
//#define HEIGHT 4096

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}

// reference
template<typename data_t>
data_t calc_min_pixel(data_t *in, int width, int height, int offset_x, int
        offset_y, int is_width, int is_height) {
    data_t min_val = in[offset_x + offset_y*width];

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            min_val = min(min_val, in[x + y*width]);
        }
    }

    return min_val;
}
template<typename data_t>
data_t calc_min_pixel(data_t *in, int width, int height) {
    return calc_min_pixel<data_t>(in, width, height, 0, 0, width, height);
}

template<typename data_t>
float calc_max_pixel(data_t *in, int width, int height, int offset_x, int
        offset_y, int is_width, int is_height) {
    data_t max_val = in[offset_x + offset_y*width];

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            max_val = max(max_val, in[x + y*width]);
        }
    }

    return max_val;
}
template<typename data_t>
float calc_max_pixel(data_t *in, int width, int height) {
    return calc_max_pixel<data_t>(in, width, height, 0, 0, width, height);
}

template<typename data_t>
data_t calc_sum_pixel(data_t *in, int width, int height, int offset_x, int
        offset_y, int is_width, int is_height) {
    data_t sum = 0;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            sum += in[x + y*width];
        }
    }

    return sum;
}
template<typename data_t>
data_t calc_sum_pixel(data_t *in, int width, int height) {
    return calc_sum_pixel<data_t>(in, width, height, 0, 0, width, height);
}


// Kernel description in HIPAcc
class MinReductionInt : public Kernel<int> {
    private:
        Accessor<int> &in;

    public:
        MinReductionInt(IterationSpace<int> &iter, Accessor<int> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) {
            return min(left, right);
        }
};
class MinReductionFloat : public Kernel<float> {
    private:
        Accessor<float> &in;

    public:
        MinReductionFloat(IterationSpace<float> &iter, Accessor<float> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) {
            return min(left, right);
        }
};
class MaxReductionInt : public Kernel<int> {
    private:
        Accessor<int> &in;

    public:
        MaxReductionInt(IterationSpace<int> &iter, Accessor<int> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) {
            return max(left, right);
        }
};
class MaxReductionFloat : public Kernel<float> {
    private:
        Accessor<float> &in;

    public:
        MaxReductionFloat(IterationSpace<float> &iter, Accessor<float> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) {
            return max(left, right);
        }
};
class SumReductionInt : public Kernel<int> {
    private:
        Accessor<int> &in;

    public:
        SumReductionInt(IterationSpace<int> &iter, Accessor<int> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) {
            return left + right;
        }
};
class SumReductionFloat : public Kernel<float> {
    private:
        Accessor<float> &in;

    public:
        SumReductionFloat(IterationSpace<float> &iter, Accessor<float> &in) :
            Kernel(iter),
            in(in)
        { addAccessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) {
            return left + right;
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of width x height pixels
    int *input_int = (int *)malloc(sizeof(int)*width*height);
    int *reference_in_int = (int *)malloc(sizeof(int)*width*height);
    int *reference_out_int = (int *)malloc(sizeof(int)*width*height);
    float *input_float = (float *)malloc(sizeof(float)*width*height);
    float *reference_in_float = (float *)malloc(sizeof(float)*width*height);
    float *reference_out_float = (float *)malloc(sizeof(float)*width*height);

    // input and output image of width x height pixels
    Image<int> in_int(width, height);
    Image<int> out_int(width, height);
    Image<float> in_float(width, height);
    Image<float> out_float(width, height);

    Accessor<int> img_in_int(in_int);
    Accessor<float> img_in_float(in_float);
    Accessor<int> acc_in_int(in_int, width/3, height/3, width/3, height/3);
    Accessor<float> acc_in_float(in_float, width/3, height/3, width/3, height/3);

    // iteration spaces
    IterationSpace<int> out_int_iter(out_int);
    IterationSpace<float> out_float_iter(out_float);
    IterationSpace<int> out_acc_int_iter(out_int, width/3, height/3, width/3, height/3);
    IterationSpace<float> out_acc_float_iter(out_float, width/3, height/3, width/3, height/3);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input_int[y*width + x] = (int) (x*height + y) * DELTA;
            reference_in_int[y*width + x] = (int) (x*height + y) * DELTA;
            reference_out_int[y*width + x] = (int) (3.12451);
            input_float[y*width + x] = (float) (x*height + y) * DELTA;
            reference_in_float[y*width + x] = (float) (x*height + y) * DELTA;
            reference_out_float[y*width + x] = (float) (3.12451);
        }
    }

    in_int = input_int;
    out_int = reference_out_int;
    in_float = input_float;
    out_float = reference_out_float;

    // global operation using functors: Images
    MinReductionInt redMinINInt(out_int_iter, img_in_int);
    MaxReductionInt redMaxINInt(out_int_iter, img_in_int);
    SumReductionInt redSumINInt(out_int_iter, img_in_int);
    MinReductionFloat redMinINFloat(out_float_iter, img_in_float);
    MaxReductionFloat redMaxINFloat(out_float_iter, img_in_float);
    SumReductionFloat redSumINFloat(out_float_iter, img_in_float);
    // global operation using functors: Accessors
    MinReductionInt redMinAccInInt(out_acc_int_iter, acc_in_int);
    MaxReductionInt redMaxAccInInt(out_acc_int_iter, acc_in_int);
    SumReductionInt redSumAccInInt(out_acc_int_iter, acc_in_int);
    MinReductionFloat redMinAccInFloat(out_acc_float_iter, acc_in_float);
    MaxReductionFloat redMaxAccInFloat(out_acc_float_iter, acc_in_float);
    SumReductionFloat redSumAccInFloat(out_acc_float_iter, acc_in_float);

    // warmup
    redMinINInt.execute();
    redMaxINInt.execute();
    redSumINInt.execute();
    redMinINFloat.execute();
    redMaxINFloat.execute();
    redSumINFloat.execute();
    redMinAccInInt.execute();
    redMaxAccInInt.execute();
    redSumAccInInt.execute();
    redMinAccInFloat.execute();
    redMaxAccInFloat.execute();
    redSumAccInFloat.execute();

    fprintf(stderr, "Calculating global reductions ...\n");
    time0 = time_ms();

    // Images
    int min_pixel_functor_int_img = redMinINInt.getReducedData();
    fprintf(stderr, "reduction functor, min (img, int): %d\n", min_pixel_functor_int_img);

    int max_pixel_functor_int_img = redMaxINInt.getReducedData();
    fprintf(stderr, "reduction functor, max (img, int): %d\n", max_pixel_functor_int_img);

    int sum_pixel_functor_int_img = redSumINInt.getReducedData();
    fprintf(stderr, "reduction functor, sum (img, int): %d\n", sum_pixel_functor_int_img);

    float min_pixel_functor_float_img = redMinINFloat.getReducedData();
    fprintf(stderr, "reduction functor, min (img, float): %f\n", min_pixel_functor_float_img);

    float max_pixel_functor_float_img = redMaxINFloat.getReducedData();
    fprintf(stderr, "reduction functor, max (img, float): %f\n", max_pixel_functor_float_img);

    float sum_pixel_functor_float_img = redSumINFloat.getReducedData();
    fprintf(stderr, "reduction functor, sum (img, float): %f\n", sum_pixel_functor_float_img);

    // Accessors
    int min_pixel_functor_int_acc = redMinAccInInt.getReducedData();
    fprintf(stderr, "reduction functor, min (acc, int): %d\n", min_pixel_functor_int_acc);

    int max_pixel_functor_int_acc = redMaxAccInInt.getReducedData();
    fprintf(stderr, "reduction functor, max (acc, int): %d\n", max_pixel_functor_int_acc);

    int sum_pixel_functor_int_acc = redSumAccInInt.getReducedData();
    fprintf(stderr, "reduction functor, sum (acc, int): %d\n", sum_pixel_functor_int_acc);

    float min_pixel_functor_float_acc = redMinAccInFloat.getReducedData();
    fprintf(stderr, "reduction functor, min (acc, float): %f\n", min_pixel_functor_float_acc);

    float max_pixel_functor_float_acc = redMaxAccInFloat.getReducedData();
    fprintf(stderr, "reduction functor, max (acc, float): %f\n", max_pixel_functor_float_acc);

    float sum_pixel_functor_float_acc = redSumAccInFloat.getReducedData();
    fprintf(stderr, "reduction functor, sum (acc, float): %f\n", sum_pixel_functor_float_acc);

    time1 = time_ms();
    dt = time1 - time0;


    // Mpixel/s = (width*height/1000000) / (dt/1000) = (width*height/dt)/1000
    // NB: actually there are (width-d)*(height) output pixels
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", dt, ((width*height)/dt)/1000);


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference: Images
    int min_pixel_ref_int_img = calc_min_pixel(input_int, width, height);
    int max_pixel_ref_int_img = calc_max_pixel(input_int, width, height);
    int sum_pixel_ref_int_img = calc_sum_pixel(input_int, width, height);
    float min_pixel_ref_float_img = calc_min_pixel(input_float, width, height);
    float max_pixel_ref_float_img = calc_max_pixel(input_float, width, height);
    float sum_pixel_ref_float_img = calc_sum_pixel(input_float, width, height);

    // calculate reference: Accessors
    int min_pixel_ref_int_acc = calc_min_pixel(input_int, width, height, width/3, height/3, width/3, height/3);
    int max_pixel_ref_int_acc = calc_max_pixel(input_int, width, height, width/3, height/3, width/3, height/3);
    int sum_pixel_ref_int_acc = calc_sum_pixel(input_int, width, height, width/3, height/3, width/3, height/3);
    float min_pixel_ref_float_acc = calc_min_pixel(input_float, width, height, width/3, height/3, width/3, height/3);
    float max_pixel_ref_float_acc = calc_max_pixel(input_float, width, height, width/3, height/3, width/3, height/3);
    float sum_pixel_ref_float_acc = calc_sum_pixel(input_float, width, height, width/3, height/3, width/3, height/3);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*height/dt)/1000);

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
    free(input_int);
    free(input_float);
    free(reference_in_int);
    free(reference_in_float);
    free(reference_out_int);
    free(reference_out_float);

    return EXIT_SUCCESS;
}

