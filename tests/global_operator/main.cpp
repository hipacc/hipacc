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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

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
data_t calc_min_pixel(data_t *in, int width, int height, int offset_x, int offset_y, int is_width, int is_height) {
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
float calc_max_pixel(data_t *in, int width, int height, int offset_x, int offset_y, int is_width, int is_height) {
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
data_t calc_sum_pixel(data_t *in, int width, int height, int offset_x, int offset_y, int is_width, int is_height) {
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


// Kernel description in Hipacc
class MinReductionInt : public Kernel<int> {
    private:
        Accessor<int> &in;

    public:
        MinReductionInt(IterationSpace<int> &iter, Accessor<int> &in) :
            Kernel(iter),
            in(in)
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) const {
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
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) const {
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
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) const {
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
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) const {
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
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        int reduce(int left, int right) const {
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
        { add_accessor(&in); }

        void kernel() {
            output() = in();
        }

        float reduce(float left, float right) const {
            return left + right;
        }
};


int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;

    // host memory for image of width x height pixels
    int *input_int = new int[width*height];
    int *reference_in_int = new int[width*height];
    int *reference_out_int = new int[width*height];
    float *input_float = new float[width*height];
    float *reference_in_float = new float[width*height];
    float *reference_out_float = new float[width*height];

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

    // input and output image of width x height pixels
    Image<int> in_int(width, height, input_int);
    Image<int> out_int(width, height, reference_out_int);
    Image<float> in_float(width, height, input_float);
    Image<float> out_float(width, height, reference_out_float);

    Accessor<int> img_in_int(in_int);
    Accessor<float> img_in_float(in_float);
    Accessor<int> acc_in_int(in_int, width/3, height/3, width/3, height/3);
    Accessor<float> acc_in_float(in_float, width/3, height/3, width/3, height/3);

    // iteration spaces
    IterationSpace<int> out_int_iter(out_int);
    IterationSpace<float> out_float_iter(out_float);
    IterationSpace<int> out_acc_int_iter(out_int, width/3, height/3, width/3, height/3);
    IterationSpace<float> out_acc_float_iter(out_float, width/3, height/3, width/3, height/3);

    // global operations on Images
    MinReductionInt redMinINInt(out_int_iter, img_in_int);
    MaxReductionInt redMaxINInt(out_int_iter, img_in_int);
    SumReductionInt redSumINInt(out_int_iter, img_in_int);
    MinReductionFloat redMinINFloat(out_float_iter, img_in_float);
    MaxReductionFloat redMaxINFloat(out_float_iter, img_in_float);
    SumReductionFloat redSumINFloat(out_float_iter, img_in_float);
    // global operations on Accessors
    MinReductionInt redMinAccInInt(out_acc_int_iter, acc_in_int);
    MaxReductionInt redMaxAccInInt(out_acc_int_iter, acc_in_int);
    SumReductionInt redSumAccInInt(out_acc_int_iter, acc_in_int);
    MinReductionFloat redMinAccInFloat(out_acc_float_iter, acc_in_float);
    MaxReductionFloat redMaxAccInFloat(out_acc_float_iter, acc_in_float);
    SumReductionFloat redSumAccInFloat(out_acc_float_iter, acc_in_float);


    std::cerr << "Calculating global reductions ..." << std::endl;
    double start = time_ms();

    // Images
    redMinINInt.execute();
    int min_pixel_kernel_int_img = redMinINInt.reduced_data();
    std::cerr << "min (img, int) reduction: " << min_pixel_kernel_int_img << std::endl;

    redMaxINInt.execute();
    int max_pixel_kernel_int_img = redMaxINInt.reduced_data();
    std::cerr << "max (img, int) reduction: " << max_pixel_kernel_int_img << std::endl;

    redSumINInt.execute();
    int sum_pixel_kernel_int_img = redSumINInt.reduced_data();
    std::cerr << "sum (img, int) reduction: " << sum_pixel_kernel_int_img << std::endl;

    redMinINFloat.execute();
    float min_pixel_kernel_float_img = redMinINFloat.reduced_data();
    std::cerr << "min (img, float) reduction: " << min_pixel_kernel_float_img << std::endl;

    redMaxINFloat.execute();
    float max_pixel_kernel_float_img = redMaxINFloat.reduced_data();
    std::cerr << "max (img, float) reduction: " << max_pixel_kernel_float_img << std::endl;

    redSumINFloat.execute();
    float sum_pixel_kernel_float_img = redSumINFloat.reduced_data();
    std::cerr << "sum (img, float) reduction: " << sum_pixel_kernel_float_img << std::endl;

    // Accessors
    redMinAccInInt.execute();
    int min_pixel_kernel_int_acc = redMinAccInInt.reduced_data();
    std::cerr << "min (acc, int) reduction: " << min_pixel_kernel_int_acc << std::endl;

    redMaxAccInInt.execute();
    int max_pixel_kernel_int_acc = redMaxAccInInt.reduced_data();
    std::cerr << "max (acc, int) reduction: " << max_pixel_kernel_int_acc << std::endl;

    redSumAccInInt.execute();
    int sum_pixel_kernel_int_acc = redSumAccInInt.reduced_data();
    std::cerr << "sum (acc, int) reduction: " << sum_pixel_kernel_int_acc << std::endl;

    redMinAccInFloat.execute();
    float min_pixel_kernel_float_acc = redMinAccInFloat.reduced_data();
    std::cerr << "min (acc, float) reduction: " << min_pixel_kernel_float_acc << std::endl;

    redMaxAccInFloat.execute();
    float max_pixel_kernel_float_acc = redMaxAccInFloat.reduced_data();
    std::cerr << "max (acc, float) reduction: " << max_pixel_kernel_float_acc << std::endl;

    redSumAccInFloat.execute();
    float sum_pixel_kernel_float_acc = redSumAccInFloat.reduced_data();
    std::cerr << "sum (acc, float) reduction: " << sum_pixel_kernel_float_acc << std::endl;

    double end = time_ms();
    float time = end - start;

    std::cerr << "Hipacc: " << time << " ms, " << (width*height/time)/1000 << " Mpixel/s" << std::endl;


    std::cerr << std::endl << "Calculating reference ..." << std::endl;
    start = time_ms();

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

    end = time_ms();
    time = end - start;
    std::cerr << "Reference: " << time << " ms, " << (width*height/time)/1000 << " Mpixel/s" << std::endl;

    // compare results: Images
    bool passed_all = true;
    std::cerr << std::endl << "Comparing results ..." << std::endl;
    if (min_pixel_kernel_int_img != min_pixel_ref_int_img) {
        std::cerr << "Test FAILED for min reduction (img, int): " << min_pixel_kernel_int_img << " vs. " << min_pixel_ref_int_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Min reduction (img, int): PASSED" << std::endl;
    }
    if (min_pixel_kernel_float_img != min_pixel_ref_float_img) {
        std::cerr << "Test FAILED for min reduction (img, float): " << min_pixel_kernel_float_img << " vs. " << min_pixel_ref_float_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Min reduction (img, float): PASSED" << std::endl;
    }

    if (max_pixel_kernel_int_img != max_pixel_ref_int_img) {
        std::cerr << "Test FAILED for max reduction (img, int): " << max_pixel_kernel_int_img << " vs. " << max_pixel_ref_int_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Max reduction (img, int): PASSED" << std::endl;
    }
    if (max_pixel_kernel_float_img != max_pixel_ref_float_img) {
        std::cerr << "Test FAILED for max reduction (img, float): " << max_pixel_kernel_float_img << " vs. " << max_pixel_ref_float_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Max reduction (img, float): PASSED" << std::endl;
    }

    if (sum_pixel_kernel_int_img != sum_pixel_ref_int_img) {
        std::cerr << "Test FAILED for sum reduction (img, int): " << sum_pixel_kernel_int_img << " vs. " << sum_pixel_ref_int_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Sum reduction (img, int): PASSED" << std::endl;
    }
    if (sum_pixel_kernel_float_img != sum_pixel_ref_float_img) {
        std::cerr << "Test FAILED for sum reduction (img, float): " << sum_pixel_kernel_float_img << " vs. " << sum_pixel_ref_float_img << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Sum reduction (img, float): PASSED" << std::endl;
    }

    // compare results: Accessors
    if (min_pixel_kernel_int_acc != min_pixel_ref_int_acc) {
        std::cerr << "Test FAILED for min reduction (acc, int): " << min_pixel_kernel_int_acc << " vs. " << min_pixel_ref_int_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Min reduction (acc, int): PASSED" << std::endl;
    }
    if (min_pixel_kernel_float_acc != min_pixel_ref_float_acc) {
        std::cerr << "Test FAILED for min reduction (acc, float): " << min_pixel_kernel_float_acc << " vs. " << min_pixel_ref_float_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Min reduction (acc, float): PASSED" << std::endl;
    }

    if (max_pixel_kernel_int_acc != max_pixel_ref_int_acc) {
        std::cerr << "Test FAILED for max reduction (acc, int): " << max_pixel_kernel_int_acc << " vs. " << max_pixel_ref_int_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Max reduction (acc, int): PASSED" << std::endl;
    }
    if (max_pixel_kernel_float_acc != max_pixel_ref_float_acc) {
        std::cerr << "Test FAILED for max reduction (acc, float): " << max_pixel_kernel_float_acc << " vs. " << max_pixel_ref_float_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Max reduction (acc, float): PASSED" << std::endl;
    }

    if (sum_pixel_kernel_int_acc != sum_pixel_ref_int_acc) {
        std::cerr << "Test FAILED for sum reduction (acc, int): " << sum_pixel_kernel_int_acc << " vs. " << sum_pixel_ref_int_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Sum reduction (acc, int): PASSED" << std::endl;
    }
    if (sum_pixel_kernel_float_acc != sum_pixel_ref_float_acc) {
        std::cerr << "Test FAILED for sum reduction (acc, float): " << sum_pixel_kernel_float_acc << " vs. " << sum_pixel_ref_float_acc << ", aborting ..." << std::endl;
        passed_all = false;
    } else {
        std::cerr << "Sum reduction (acc, float): PASSED" << std::endl;
    }

    if (passed_all) {
        std::cerr << "Tests PASSED" << std::endl;
    } else {
        std::cerr << "Tests FAILED" << std::endl;
        exit(EXIT_FAILURE);
    }

    // free memory
    delete[] input_int;
    delete[] input_float;
    delete[] reference_in_int;
    delete[] reference_in_float;
    delete[] reference_out_int;
    delete[] reference_out_float;

    return EXIT_SUCCESS;
}

