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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <sys/time.h>

#ifdef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef OPENCV_CUDA_FOUND
#include <opencv2/cudafilters.hpp>
#endif
#endif

#include "hipacc.hpp"

// variables set by Makefile
//#define SIZE_X 5
//#define SIZE_Y 5
//#define WIDTH 4096
//#define HEIGHT 4096

// code variants
#define CONST_MASK
#define USE_LAMBDA

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// box filter reference
void box_filter(uchar4 *in, uchar4 *out, int size_x, int size_y, int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int4 sum = { 0, 0, 0, 0 };

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += convert_int4(in[(y + yf)*width + x + xf]);
                }
            }
            out[y*width + x] = convert_uchar4(1.0f/(float)(size_x*size_y)*convert_float4(sum));
        }
    }
}


// Kernel description in Hipacc
class BoxFilter : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &in;
        Domain &dom;
        int size_x, size_y;

    public:
        BoxFilter(IterationSpace<uchar4> &iter, Accessor<uchar4> &in, Domain
                &dom, int size_x, int size_y) :
            Kernel(iter), in(in), dom(dom),
            size_x(size_x), size_y(size_y)
        { add_accessor(&in); }

        #ifdef USE_LAMBDA
        void kernel() {
            output() = convert_uchar4( (1.0f/(float)(size_x*size_y)) *
                    convert_float4(reduce(dom, Reduce::SUM, [&] () -> int4 {
                        return convert_int4(in(dom));
                    })));
        }
        #else
        void kernel() {
            int anchor_x = size_x >> 1;
            int anchor_y = size_y >> 1;
            int4 sum = { 0, 0, 0, 0 };

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += convert_int4(in(xf, yf));
                }
            }
            output() = convert_uchar4((1.0f/(float)(size_x*size_y))*convert_float4(sum));
        }
        #endif
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    const int offset_x = size_x >> 1;
    const int offset_y = size_y >> 1;

    // only filter kernel sizes 3x3 and 5x5 implemented
    if (size_x != size_y && (size_x != 3 || size_x != 5)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // domain for box filter
    #ifdef CONST_MASK
    const
    #endif
    uchar domain[SIZE_Y][SIZE_X] = {
        #if SIZE_X == 3
        { 1, 1, 1 },
        { 1, 1, 1 },
        { 1, 1, 1 }
        #endif
        #if SIZE_X == 5
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1 }
        #endif
    };

    // host memory for image of width x height pixels
    uchar4 *input = new uchar4[width*height];
    uchar4 *reference_in = new uchar4[width*height];
    uchar4 *reference_out = new uchar4[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            uchar4 val;
            val.x = (y*width + x + 1) % 256;
            val.y = (y*width + x + 2) % 256;
            val.z = (y*width + x + 3) % 256;
            val.w = (y*width + x + 4) % 256;
            input[y*width + x] = val;
            reference_in[y*width + x] = val;
            reference_out[y*width + x] = (uchar4){ 0, 0, 0, 0 };
        }
    }


    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar4> out(width, height);

    // define Domain for box filter
    Domain dom(domain);

    #ifndef OPENCV
    std::cerr << "Calculating Hipacc box filter ..." << std::endl;
    std::vector<float> timings_hipacc;
    float timing = 0;

    BoundaryCondition<uchar4> bound(in, dom, Boundary::CLAMP);
    Accessor<uchar4> acc(bound);

    IterationSpace<uchar4> iter(out);
    BoxFilter filter(iter, acc, dom, size_x, size_y);

    filter.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    // get pointer to result data
    uchar4 *output = out.data();

    if (timings_hipacc.size()) {
        std::cerr << "Hipacc:";
        for (std::vector<float>::const_iterator it = timings_hipacc.begin(); it != timings_hipacc.end(); ++it)
            std::cerr << "\t" << *it;
        std::cerr << std::endl;
    }
    #endif


    #ifdef OPENCV
    auto opencv_bench = [] (std::function<void(int)> init, std::function<void(int)> launch, std::function<void(float)> finish) {
        for (int brd_type=0; brd_type<5; ++brd_type) {
            init(brd_type);

            std::vector<float> timings;
            try {
                for (int nt=0; nt<10; ++nt) {
                    auto start = time_ms();
                    launch(brd_type);
                    auto end = time_ms();
                    timings.push_back(end - start);
                }
            } catch (const cv::Exception &ex) {
                std::cerr << ex.what();
                timings.push_back(0);
            }

            std::cerr << "OpenCV (";
            switch (brd_type) {
                case IPL_BORDER_CONSTANT:    std::cerr << "CONSTANT";   break;
                case IPL_BORDER_REPLICATE:   std::cerr << "CLAMP";      break;
                case IPL_BORDER_REFLECT:     std::cerr << "MIRROR";     break;
                case IPL_BORDER_WRAP:        std::cerr << "REPEAT";     break;
                case IPL_BORDER_REFLECT_101: std::cerr << "MIRROR_101"; break;
                default: break;
            }
            std::sort(timings.begin(), timings.end());
            float time = timings[timings.size()/2];
            std::cerr << "): " << time << " ms, " << (width*height/time)/1000 << " Mpixel/s" << std::endl;

            finish(time);
        }
    };

    cv::Mat cv_data_src(height, width, CV_8UC4, input);
    cv::Mat cv_data_dst(height, width, CV_8UC4, cv::Scalar(0));
    cv::Size ksize(size_x, size_y);
    cv::Point anchor = cv::Point(-1,-1);
    std::vector<float> timings_cpu;
    std::vector<float> timings_ocl;
    std::vector<float> timings_cuda;

    auto compute_tapi = [&] (std::vector<float> &timings) {
        cv::UMat dev_src, dev_dst;
        opencv_bench(
            [&] (int) {
                cv_data_src.copyTo(dev_src);
            },
            [&] (int brd_type) {
                cv::blur(dev_src, dev_dst, ksize, anchor, brd_type);
                if (cv::ocl::useOpenCL())
                    cv::ocl::finish();
            },
            [&] (float timing) {
                timings.push_back(timing);
                dev_dst.copyTo(cv_data_dst);
            }
        );
    };

    // OpenCV - CPU
    cv::ocl::setUseOpenCL(false);
    std::cerr << std::endl
              << "Calculating OpenCV-CPU box filter on CPU" << std::endl;
    compute_tapi(timings_cpu);

    // OpenCV - OpenCL
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cerr << std::endl
                  << "Calculating OpenCV-OCL box filter on "
                  << cv::ocl::Device::getDefault().name() << std::endl;
        compute_tapi(timings_ocl);
    }

    // OpenCV - CUDA
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        #ifdef OPENCV_CUDA_FOUND
        std::cerr << std::endl
                  << "Calculating OpenCV-CUDA box filter" << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

        cv::cuda::GpuMat dev_src, dev_dst;
        cv::Ptr<cv::cuda::Filter> box;

        opencv_bench(
            [&] (int brd_type) {
                dev_src.upload(cv_data_src);
                box = cv::cuda::createBoxFilter(dev_src.type(), dev_src.type(), ksize, anchor, brd_type);
            },
            [&] (int) {
                box->apply(dev_src, dev_dst);
            },
            [&] (float timing) {
                timings_cuda.push_back(timing);
                dev_dst.download(cv_data_dst);
            }
        );
        #endif
    }

    // get pointer to result data
    uchar4 *output = (uchar4 *)cv_data_dst.data;

    if (timings_cpu.size()) {
        std::cerr << "CV-CPU: ";
        for (auto time : timings_cpu)
            std::cerr << "\t" << time;
        std::cerr << std::endl;
    }
    if (timings_ocl.size()) {
        std::cerr << "CV-OCL: ";
        for (auto time : timings_ocl)
            std::cerr << "\t" << time;
        std::cerr << std::endl;
    }
    if (timings_cuda.size()) {
        std::cerr << "CV-CUDA:";
        for (auto time : timings_cuda)
            std::cerr << "\t" << time;
        std::cerr << std::endl;
    }
    #endif


    std::cerr << "Calculating reference ..." << std::endl;
    std::vector<float> timings_reference;
    for (int nt=0; nt<3; ++nt) {
        double start = time_ms();

        box_filter(reference_in, reference_out, size_x, size_y, width, height);

        double end = time_ms();
        timings_reference.push_back(end - start);
    }
    std::sort(timings_reference.begin(), timings_reference.end());
    float time = timings_reference[timings_reference.size()/2];
    std::cerr << "Reference: " << time << " ms, " << (width*height/time)/1000 << " Mpixel/s" << std::endl;


    std::cerr << "Comparing results ..." << std::endl;
    #ifdef OPENCV
    std::cerr << "Warning: The CPU, OCL, and CUDA modules in OpenCV use different implementations and yield inconsistent results." << std::endl
              << "         This is the case even for different filter sizes within the same module!" << std::endl;
    #endif
    for (int y=offset_y; y<height-offset_y; ++y) {
        for (int x=offset_x; x<width-offset_x; ++x) {
            if (reference_out[y*width + x].x != output[y*width + x].x ||
                reference_out[y*width + x].y != output[y*width + x].y ||
                reference_out[y*width + x].z != output[y*width + x].z ||
                reference_out[y*width + x].w != output[y*width + x].w) {
                std::cerr << "Test FAILED, at (" << x << "," << y << "): ("
                          << (int)reference_out[y*width + x].x << ","
                          << (int)reference_out[y*width + x].y << ","
                          << (int)reference_out[y*width + x].z << ","
                          << (int)reference_out[y*width + x].w << ") vs. ("
                          << (int)output[y*width + x].x << ","
                          << (int)output[y*width + x].y << ","
                          << (int)output[y*width + x].z << ","
                          << (int)output[y*width + x].w << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << "Test PASSED" << std::endl;

    // free memory
    delete[] input;
    delete[] reference_in;
    delete[] reference_out;

    return EXIT_SUCCESS;
}

