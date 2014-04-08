//
// Copyright (c) 2013, University of Erlangen-Nuremberg
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

struct roi_t {
    int img_width, img_height;
    int roi_width, roi_height;
    int roi_ox, roi_oy;
    roi_t(int width, int height) :
        img_width(width), img_height(height), roi_width(width),
        roi_height(height), roi_ox(0), roi_oy(0) {}
    roi_t(int width, int height, int r_width, int r_height, int ox, int oy) :
        img_width(width), img_height(height), roi_width(r_width),
        roi_height(r_height), roi_ox(ox), roi_oy(oy) {}
};

// reference
template<typename data_t>
void compare_results(data_t *ref, data_t *data, roi_t &ref_roi, roi_t &data_roi)
{
    // compare results
    assert(ref_roi.roi_width == data_roi.roi_width && ref_roi.roi_height ==
            data_roi.roi_height && "Image sizes have to be the same!");
    fprintf(stderr, "\nComparing results ...\n");
    for (int y=ref_roi.roi_oy; y<ref_roi.roi_oy+ref_roi.roi_height; y++) {
        for (int x=ref_roi.roi_ox; x<ref_roi.roi_ox+ref_roi.roi_width; x++) {
            if (ref[y*ref_roi.img_width + x] !=
                data[(y-ref_roi.roi_oy+data_roi.roi_oy)*data_roi.img_width +
                x-ref_roi.roi_ox+data_roi.roi_ox]) {
                fprintf(stderr, "Test FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        ref[y*ref_roi.img_width + x],
                        data[(y-ref_roi.roi_oy+data_roi.roi_oy)*data_roi.img_width
                        + x-ref_roi.roi_ox+data_roi.roi_ox]);
                return;
            }
        }
    }
    fprintf(stderr, "Test PASSED\n");
}


int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int roi_width = WIDTH/2;
    const int roi_height = HEIGHT/2;
    const int roi_offset_x = 2;
    const int roi_offset_y = 2;

    // host memory for image of width x height pixels
    int *img0 = (int *)malloc(sizeof(int)*width*height);
    int *img1 = (int *)malloc(sizeof(int)*width*height);
    int *img2 = (int *)malloc(sizeof(int)*roi_width*roi_height);
    int *img3 = (int *)malloc(sizeof(int)*roi_width*roi_height);

    // input and output image of width x height pixels
    Image<int> IMG0(width, height);
    Image<int> IMG1(width, height);
    Image<int> IMG2(roi_width, roi_height);
    Image<int> IMG3(roi_width, roi_height);
    roi_t img_roi(width, height);

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            img0[y*width + x] = x*height + y;
            img1[y*width + x] = 23;
        }
    }
    for (int y=0; y<roi_height; ++y) {
        for (int x=0; x<roi_width; ++x) {
            img2[y*roi_width + x] = x*roi_height + y;
            img3[y*roi_width + x] = 23;
        }
    }

    IMG0 = img0;
    IMG1 = img1;
    IMG2 = img2;
    IMG3 = img3;
    int *out = NULL;

    // Image = Image
    IMG1 = IMG0;
    out = IMG1.getData();
    compare_results(img0, out, img_roi, img_roi);

    // Accessor = Image
    AccessorNN<int> AccImg1NN(IMG1);
    AccImg1NN = IMG0;
    out = IMG1.getData();
    compare_results(img0, out, img_roi, img_roi);

    // Accessor = Accessor
    Accessor<int> AccImg0(IMG0, roi_width, roi_height, roi_width/2, roi_height/2);
    AccessorLF<int> AccImg1LF(IMG1, roi_width, roi_height, roi_offset_x, roi_offset_y);
    roi_t acc_roi0(width, height, roi_width, roi_height, roi_width/2, roi_height/2);
    roi_t acc_roi1(width, height, roi_width, roi_height, roi_offset_x, roi_offset_y);
    AccImg1LF = AccImg0;
    out = IMG1.getData();
    compare_results(img0, out, acc_roi0, acc_roi1);

    // Image = Accessor
    IMG2 = AccImg0;
    roi_t img_roi2(roi_width, roi_height);
    out = IMG2.getData();
    compare_results(img0, out, acc_roi0, img_roi2);

    // Image = Image.getData()
    IMG1 = IMG0.getData();
    out = IMG1.getData();
    compare_results(img0, out, img_roi, img_roi);


    // memory cleanup
    free(img0);
    free(img1);
    free(img2);
    free(img3);

    return EXIT_SUCCESS;
}

