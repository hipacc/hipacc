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
void access_nn(data_t *in, data_t *out, int in_width, int in_height, int in_ox,
        int in_oy, int in_roi_width, int in_roi_height, int out_width, int
        out_height, int is_ox, int is_oy, int is_width, int is_height) {
    float stride_x = (in_roi_width)/(float)is_width;
    float stride_y = (in_roi_height)/(float)is_height;

    for (int y=is_oy; y<is_oy+is_height; ++y) {
        for (int x=is_ox; x<is_ox+is_width; ++x) {
            int x_nn = (int)(stride_x*(x-is_ox)) + in_ox;
            int y_nn = (int)(stride_y*(y-is_oy)) + in_oy;

            out[x + y*out_width] = in[x_nn + y_nn*in_width];
        }
    }
}
template<typename data_t>
void access_nn(data_t *in, data_t *out, int in_width, int in_height, int
        out_width, int out_height) {
    return access_nn<data_t>(in, out, in_width, in_height, 0, 0, in_width,
            in_height, out_width, out_height, 0, 0, out_width, out_height);
}


// Kernel description in HIPAcc
class CopyKernel : public Kernel<int> {
    private:
        Accessor<int> &input;

    public:
        CopyKernel(IterationSpace<int> &iter, Accessor<int> &input) :
            Kernel(iter),
            input(input)
        { addAccessor(&input); }

        void kernel() {
            output() = input();
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;
    const int offset_x = 5;
    const int offset_y = 5;
    const int is_width = WIDTH/2;
    const int is_height = HEIGHT/2;
    const int is_offset_x = 2;
    const int is_offset_y = 2;
    float timing = 0.0f;

    // host memory for image of width x height pixels
    int *input = (int *)malloc(sizeof(int)*width*height);
    int *out_init = (int *)malloc(sizeof(int)*is_width*is_height);
    int *reference_in = (int *)malloc(sizeof(int)*width*height);
    int *reference_out = (int *)malloc(sizeof(int)*is_width*is_height);

    // input and output image of width x height pixels
    Image<int> IN(width, height);
    Image<int> OUT(is_width, is_height);

    // use nearest neighbor interpolation
    AccessorNN<int> AccInNN(IN);
    // use linear filtering interpolation
    AccessorLF<int> AccInLF(IN);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = x*height + y;
            reference_in[y*width + x] = x*height + y;
        }
    }
    for (int y=0; y<is_height; ++y) {
        for (int x=0; x<is_width; ++x) {
            out_init[y*is_width + x] = 23;
            reference_out[y*is_width + x] = 23;
        }
    }

    IN = input;
    OUT = out_init;

    IterationSpace<int> CIS(OUT);

    // copy kernel using NN
    CopyKernel copy_nn(CIS, AccInNN);
    CopyKernel copy_lf(CIS, AccInLF);

    fprintf(stderr, "Executing copy (NN) kernel ...\n");

    copy_nn.execute();
    timing = hipaccGetLastKernelTiming();


    // get pointer to result data
    int *output = OUT.getData();

    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (is_width*is_height/timing)/1000);


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference
    access_nn(reference_in, reference_out, width, height, 0, 0, width, height,
            is_width, is_height, 0, 0, is_width, is_height);
    //access_nn(reference_in, reference_out, width, height, offset_x, offset_y, width-2*offset_x, height-2*offset_y,
    //        is_width, is_height, is_offset_x, is_offset_y, is_width-2*is_offset_x, is_height-2*is_offset_y);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (is_width*is_height/dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
    // compare results
    for (int y=0; y<is_height; y++) {
        for (int x=0; x<is_width; x++) {
            if (reference_out[y*is_width + x] != output[y*is_width + x]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        reference_out[y*is_width + x], output[y*is_width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "Test PASSED\n");

    // memory cleanup
    free(input);
    free(out_init);
    free(reference_in);
    free(reference_out);

    return EXIT_SUCCESS;
}

