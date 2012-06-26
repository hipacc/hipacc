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
//#define SIGMA_D 3
//#define SIGMA_R 5
//#define WIDTH 4096
//#define HEIGHT 4096
#define SIGMA_D SIZE_X
#define SIGMA_R SIZE_Y

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


namespace hipacc {
class CopyNN : public Kernel<int> {
    private:
        Accessor<int> &Input;

    public:
        CopyNN(IterationSpace<int> &IS, Accessor<int> &Input) :
            Kernel(IS),
            Input(Input)
        {
            addAccessor(&Input);
        }

        void kernel() {
            output() = Input();
        }
};
}


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

    // host memory for image of of widthxheight pixels
    int *host_in = (int *)malloc(sizeof(int)*width*height);
    int *host_out = (int *)malloc(sizeof(int)*is_width*is_height);
    int *reference_in = (int *)malloc(sizeof(int)*width*height);
    int *reference_out = (int *)malloc(sizeof(int)*is_width*is_height);

    // input and output image of widthxheight pixels
    Image<int> IN(width, height);
    Image<int> OUT(is_width, is_height);

    // use constant for boundary handling
    BoundaryCondition<int> BCIn(IN, offset_x, offset_y, BOUNDARY_CONSTANT, 0);

    // use nearest neighbor interpolation
    AccessorNN<int> AccInNN(IN, width-2*offset_x, height-2*offset_y, offset_x, offset_y);
    // use linear filtering interpolation
    AccessorLF<int> AccInLF(BCIn, width-2*offset_x, height-2*offset_y, offset_x, offset_y);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            host_in[y*width + x] = x*height + y;
            reference_in[y*width + x] = x*height + y;
        }
    }
    for (int y=0; y<is_height; ++y) {
        for (int x=0; x<is_width; ++x) {
            host_out[y*is_width + x] = 23;
            reference_out[y*is_width + x] = 23;
        }
    }

    IN = host_in;
    OUT = host_out;

    IterationSpace<int> CIS(OUT, is_width-2*is_offset_x, is_height-2*is_offset_y, is_offset_x, is_offset_y);

    // copy kernel using NN
    CopyNN copy_nn(CIS, AccInNN);
    CopyNN copy_lf(CIS, AccInLF);

    // warmup
    copy_lf.execute();

    fprintf(stderr, "Executing copy (NN) kernel ...\n");
    time0 = time_ms();

    copy_nn.execute();

    time1 = time_ms();
    dt = time1 - time0;

    // get results
    host_out = OUT.getData();

    // Mpixel/s = (is_width*is_height/1000000) / (dt/1000) = (is_width*is_height/dt)/1000
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", dt, (is_width*is_height/dt)/1000);


    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    // calculate reference
    access_nn(reference_in, reference_out, width, height, offset_x, offset_y, width-2*offset_x, height-2*offset_y,
            is_width, is_height, is_offset_x, is_offset_y, is_width-2*is_offset_x, is_height-2*is_offset_y);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (is_width*is_height/dt)/1000);

    fprintf(stderr, "\nComparing results ...\n");
    // compare results
    for (int y=0; y<is_height; y++) {
        for (int x=0; x<is_width; x++) {
            if (reference_out[y*is_width + x] != host_out[y*is_width +x]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        reference_out[y*is_width + x], host_out[y*is_width +x]);
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

