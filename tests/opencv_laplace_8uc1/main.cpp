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
//#define SIZE_X 3
//#define SIZE_Y 3
//#define WIDTH 4096
//#define HEIGHT 4096

// code variants
#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// Laplace filter reference
void laplace_filter(uchar *in, uchar *out, int *filter, int size, int width, int height) {
    const int size_x = size;
    const int size_y = size;
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += filter[(yf+anchor_y)*size_x + xf+anchor_x] * in[(y+yf)*width + x + xf];
                }
            }

            sum = min(sum, 255);
            sum = max(sum, 0);
            out[y*width + x] = sum;
        }
    }
}


// Laplace filter in Hipacc
class LaplaceFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Domain &dom;
        Mask<int> &mask;
        const int size;

    public:
        LaplaceFilter(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                Domain &dom, Mask<int> &mask, const int size) :
            Kernel(iter),
            input(input),
            dom(dom),
            mask(mask),
            size(size)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            int sum = reduce(dom, Reduce::SUM, [&] () -> int {
                    return mask(dom) * input(dom);
                    });
            sum = min(sum, 255);
            sum = max(sum, 0);
            output() = (uchar) (sum);
        }
        #else
        void kernel() {
            const int anchor = size >> 1;
            int sum = 0;

            for (int yf = -anchor; yf<=anchor; ++yf) {
                for (int xf = -anchor; xf<=anchor; ++xf) {
                    sum += mask(xf, yf) * input(xf, yf);
                }
            }

            sum = min(sum, 255);
            sum = max(sum, 0);
            output() = (uchar) (sum);
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

    // only filter kernel sizes 3x3 and 5x5 supported
    if (size_x != size_y || !(size_x == 3 || size_x == 5)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    #ifdef CONST_MASK
    const
    #endif
    #if SIZE_X == 1
    #define SIZE 3
    #elif SIZE_X == 3
    #define SIZE 3
    #else
    #define SIZE 5
    #endif
    int mask[SIZE][SIZE] = {
        #if SIZE_X==1
        { 0,  1,  0 },
        { 1, -4,  1 },
        { 0,  1,  0 }
        #endif
        #if SIZE_X==3
        { 2,  0,  2 },
        { 0, -8,  0 },
        { 2,  0,  2 }
        #endif
        #if SIZE_X==5
        { 1,   1,   1,   1,   1 },
        { 1,   1,   1,   1,   1 },
        { 1,   1, -24,   1,   1 },
        { 1,   1,   1,   1,   1 },
        { 1,   1,   1,   1,   1 }
        #endif
    };

    // host memory for image of width x height pixels
    uchar *input = new uchar[width*height];
    uchar *reference_in = new uchar[width*height];
    uchar *reference_out = new uchar[width*height];

    // initialize data
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input[y*width + x] = (uchar)(y*width + x) % 256;
            reference_in[y*width + x] = (uchar)(y*width + x) % 256;
            reference_out[y*width + x] = 0;
        }
    }


    // input and output image of width x height pixels
    Image<uchar> IN(width, height, input);
    Image<uchar> OUT(width, height);

    // filter mask
    Mask<int> M(mask);

    // filter domain
    Domain D(M);

    IterationSpace<uchar> IsOut(OUT);


    #ifndef OPENCV
    std::cerr << "Calculating Hipacc Laplace filter ..." << std::endl;
    std::vector<float> timings_hipacc;
    float timing = 0;

    // UNDEFINED
    #ifdef RUN_UNDEF
    BoundaryCondition<uchar> BcInUndef(IN, M, Boundary::UNDEFINED);
    Accessor<uchar> AccInUndef(BcInUndef);
    LaplaceFilter LFU(IsOut, AccInUndef, D, M, size_x);

    LFU.execute();
    timing = hipacc_last_kernel_timing();
    #endif
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (UNDEFINED): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CLAMP
    BoundaryCondition<uchar> BcInClamp(IN, M, Boundary::CLAMP);
    Accessor<uchar> AccInClamp(BcInClamp);
    LaplaceFilter LFC(IsOut, AccInClamp, D, M, size_x);

    LFC.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // REPEAT
    BoundaryCondition<uchar> BcInRepeat(IN, M, Boundary::REPEAT);
    Accessor<uchar> AccInRepeat(BcInRepeat);
    LaplaceFilter LFR(IsOut, AccInRepeat, D, M, size_x);

    LFR.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (REPEAT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // MIRROR
    BoundaryCondition<uchar> BcInMirror(IN, M, Boundary::MIRROR);
    Accessor<uchar> AccInMirror(BcInMirror);
    LaplaceFilter LFM(IsOut, AccInMirror, D, M, size_x);

    LFM.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (MIRROR): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // CONSTANT
    BoundaryCondition<uchar> BcInConst(IN, M, Boundary::CONSTANT, '1');
    Accessor<uchar> AccInConst(BcInConst);
    LaplaceFilter LFConst(IsOut, AccInConst, D, M, size_x);

    LFConst.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CONSTANT): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;


    // get pointer to result data
    uchar *output = OUT.data();

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

    cv::Mat cv_data_src(height, width, CV_8UC1, input);
    cv::Mat cv_data_dst(height, width, CV_8UC1, cv::Scalar(0));
    int ddepth = CV_8U;
    double scale = 1.0;
    double delta = 0.0;
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
                cv::Laplacian(dev_src, dev_dst, ddepth, size_x, scale, delta, brd_type);
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
              << "Calculating OpenCV-CPU Laplace filter on CPU" << std::endl;
    compute_tapi(timings_cpu);

    // OpenCV - OpenCL
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cerr << std::endl
                  << "Calculating OpenCV-OCL Laplace filter on "
                  << cv::ocl::Device::getDefault().name() << std::endl;
        compute_tapi(timings_ocl);
    }

    // OpenCV - CUDA
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        #ifdef OPENCV_CUDA_FOUND
        #if SIZE_X==5
        #warning "The CUDA module in OpenCV supports only 1x1 and 3x3 Laplace filters, using 3x3!"
        int size_x = 3;
        #endif
        std::cerr << std::endl
                  << "Calculating OpenCV-CUDA Laplace filter" << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

        cv::cuda::GpuMat dev_src, dev_dst;
        cv::Ptr<cv::cuda::Filter> laplace;

        opencv_bench(
            [&] (int brd_type) {
                dev_src.upload(cv_data_src);
                laplace = cv::cuda::createLaplacianFilter(dev_src.type(), dev_dst.type(), size_x, scale, brd_type);
            },
            [&] (int) {
                laplace->apply(dev_src, dev_dst);
            },
            [&] (float timing) {
                timings_cuda.push_back(timing);
                dev_dst.download(cv_data_dst);
            }
        );
        #endif
    }

    // get pointer to result data
    uchar *output = (uchar *)cv_data_dst.data;

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

        laplace_filter(reference_in, reference_out, (int *)mask, size_x, width, height);

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
            if (reference_out[y*width + x] != output[y*width + x]) {
                std::cerr << "Test FAILED, at (" << x << "," << y << "): "
                          << (int)reference_out[y*width + x] << " vs. "
                          << (int)output[y*width + x] << std::endl;
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

