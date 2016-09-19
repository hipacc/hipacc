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
#define SIZE_X 32
#define WIDTH 1048576

// code variants
#define CONST_MASK
#define USE_LAMBDA
//#define RUN_UNDEF

using namespace hipacc;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// FIR filter reference
void fir_filter(uchar4 *in, uchar4 *out, float *filter, int size_x, int width) {
    int anchor_x = size_x >> 1;
    int upper_x = width - anchor_x;

    for (int x=anchor_x; x<upper_x; ++x) {
        float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };

        // size_x is even => -size_x/2 .. +size_x/2 - 1
        for (int xf = -anchor_x; xf<anchor_x; ++xf) {
            sum += filter[xf + anchor_x] * convert_float4(in[x + xf]);
        }

        out[x] = convert_uchar4(sum);
    }
}


// FIR filter in Hipacc
class FIRFilterMask : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &input;
        Mask<float> &mask;
        const int size_x;

    public:
        FIRFilterMask(IterationSpace<uchar4> &iter, Accessor<uchar4> &input,
                Mask<float> &mask, const int size_x) :
            Kernel(iter),
            input(input),
            mask(mask),
            size_x(size_x)
        { add_accessor(&input); }

        #ifdef USE_LAMBDA
        void kernel() {
            float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };
            sum += convolve(mask, Reduce::SUM, [&] () -> float4 {
                    return mask() * convert_float4(input(mask));
                    });

            output() = convert_uchar4(sum);
        }
        #else
        void kernel() {
            const int anchor_x = size_x >> 1;
            float4 sum = { 0.5f, 0.5f, 0.5f, 0.5f };

            // size_x is even => -size_x/2 .. +size_x/2 - 1
            for (int xf = -anchor_x; xf<anchor_x; ++xf) {
                sum += mask(xf, 0) * convert_float4(input(xf, 0));
            }

            output() = convert_uchar4(sum);
        }
        #endif
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int size_x = SIZE_X;
    const int offset_x = size_x >> 1;

    // filter coefficients
    #ifdef CONST_MASK
    // only filter kernel sizes 32, and 64 implemented
    if (size_x && !(size_x == 32 || size_x == 64)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 32 and 64!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float filter_x[1][SIZE_X] = {
        #if SIZE_X == 32
        { 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f }
        #endif
        #if SIZE_X == 64
        { 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f }
        #endif
    };
    #else
    float filter_x[1][SIZE_X];

    for (int i=0; i < size_x; i++) {
        filter_x[0][i] = ((float)1)/SIZE_X;
    }
    #endif

    // host memory for image of width pixels
    uchar4 *input = new uchar4[width];
    uchar4 *reference_in = new uchar4[width];
    uchar4 *reference_out = new uchar4[width];

    // initialize data
    for (int x=0; x<width; ++x) {
        uchar val = x % 256;
        input[x] = (uchar4){ val, val, val, val };
        reference_in[x] = (uchar4){ val, val, val, val };
        reference_out[x] = (uchar4){ 0, 0, 0, 0 };
    }

    // input and output image of width pixels
    Image<uchar4> IN(width, 1, input);
    Image<uchar4> OUT(width, 1);


    #ifndef OPENCV
    std::cerr << "Calculating Hipacc FIR filter ..." << std::endl;
    std::vector<float> timings_hipacc;
    float timing = 0;

    // filter mask
    Mask<float> MX(filter_x);

    IterationSpace<uchar4> IsOut(OUT);

    // UNDEFINED
    #ifdef RUN_UNDEF
    BoundaryCondition<uchar4> BcInUndef2(IN, size_x, 1, Boundary::UNDEFINED);
    Accessor<uchar4> AccInUndef2(BcInUndef2);
    FIRFilterMask FFU(IsOut, AccInUndef2, MX, size_x);

    FFU.execute();
    timing = hipacc_last_kernel_timing();
    #endif
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (UNDEFINED): " << timing << " ms, " << (width/timing)/1000 << " Mpixel/s" << std::endl;


    // CLAMP
    BoundaryCondition<uchar4> BcInClamp2(IN, size_x, 1, Boundary::CLAMP);
    Accessor<uchar4> AccInClamp2(BcInClamp2);
    FIRFilterMask FFC(IsOut, AccInClamp2, MX, size_x);

    FFC.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width/timing)/1000 << " Mpixel/s" << std::endl;


    // REPEAT
    BoundaryCondition<uchar4> BcInRepeat2(IN, size_x, 1, Boundary::REPEAT);
    Accessor<uchar4> AccInRepeat2(BcInRepeat2);
    FIRFilterMask FFR(IsOut, AccInRepeat2, MX, size_x);

    FFR.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (REPEAT): " << timing << " ms, " << (width/timing)/1000 << " Mpixel/s" << std::endl;


    // MIRROR
    BoundaryCondition<uchar4> BcInMirror2(IN, size_x, 1, Boundary::MIRROR);
    Accessor<uchar4> AccInMirror2(BcInMirror2);
    FIRFilterMask FFM(IsOut, AccInMirror2, MX, size_x);

    FFM.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (MIRROR): " << timing << " ms, " << (width/timing)/1000 << " Mpixel/s" << std::endl;


    // CONSTANT
    BoundaryCondition<uchar4> BcInConst2(IN, size_x, 1, Boundary::CONSTANT, '1');
    Accessor<uchar4> AccInConst2(BcInConst2);
    FIRFilterMask FFConst(IsOut, AccInConst2, MX, size_x);

    FFConst.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CONSTANT): " << timing << " ms, " << (width/timing)/1000 << " Mpixel/s" << std::endl;


    // get pointer to result data
    uchar4 *output = OUT.data();

    if (timings_hipacc.size()) {
        std::cerr << "Hipacc:";
        for (std::vector<float>::const_iterator it = timings_hipacc.begin(); it != timings_hipacc.end(); ++it)
            std::cerr << "\t" << *it;
        std::cerr << "\t"
        #ifdef CONST_MASK
                  << "+ConstMask\t"
        #endif
        #ifdef USE_LAMBDA
                  << "+Lambda\t"
        #endif
                  << std::endl;
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
            std::cerr << "): " << time << " ms, " << (width/time)/1000 << " Mpixel/s" << std::endl;

            finish(time);
        }
    };

    cv::Mat cv_data_src(1, width, CV_8UC4, input);
    cv::Mat cv_data_dst(1, width, CV_8UC4, cv::Scalar(0));
    cv::Mat kernel = cv::Mat::ones(size_x, 1, CV_32F ) / (float)(size_x);
    cv::Point anchor = cv::Point(-1,-1);
    std::vector<float> timings_cpu;
    std::vector<float> timings_ocl;
    std::vector<float> timings_cuda;
    double delta = 0;
    int ddepth = -1;

    auto compute_tapi = [&] (std::vector<float> &timings) {
        cv::UMat dev_src, dev_dst;
        opencv_bench(
            [&] (int) {
                cv_data_src.copyTo(dev_src);
            },
            [&] (int brd_type) {
                cv::filter2D(dev_src, dev_dst, ddepth, kernel, anchor, delta, brd_type);
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
              << "Calculating OpenCV-CPU FIR filter on CPU" << std::endl;
    compute_tapi(timings_cpu);

    // OpenCV - OpenCL
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cerr << std::endl
                  << "Calculating OpenCV-OCL FIR filter on "
                  << cv::ocl::Device::getDefault().name() << std::endl;
        compute_tapi(timings_ocl);
    }

    // OpenCV - CUDA
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        #ifdef OPENCV_CUDA_FOUND
        std::cerr << std::endl
                  << "Calculating OpenCV-CUDA FIR filter" << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

        cv::cuda::GpuMat dev_src, dev_dst;
        cv::Ptr<cv::cuda::Filter> fir;

        opencv_bench(
            [&] (int brd_type) {
                dev_src.upload(cv_data_src);
                fir = cv::cuda::createLinearFilter(cv_data_src.type(), cv_data_dst.type(), kernel, anchor, brd_type);
            },
            [&] (int) {
                fir->apply(dev_src, dev_dst);
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

        fir_filter(reference_in, reference_out, (float *)filter_x[0], size_x, width);

        double end = time_ms();
        timings_reference.push_back(end - start);
    }
    std::sort(timings_reference.begin(), timings_reference.end());
    float time = timings_reference[timings_reference.size()/2];
    std::cerr << "Reference: " << time << " ms, " << (width/time)/1000 << " Mpixel/s" << std::endl;


    std::cerr << "Comparing results ..." << std::endl;
    #ifdef OPENCV
    std::cerr << "Warning: The CPU, OCL, and CUDA modules in OpenCV use different implementations and yield inconsistent results." << std::endl
              << "         This is the case even for different filter sizes within the same module!" << std::endl;
    #endif
    for (int x=offset_x; x<width-offset_x; ++x) {
        if (reference_out[x].x != output[x].x ||
            reference_out[x].y != output[x].y ||
            reference_out[x].z != output[x].z ||
            reference_out[x].w != output[x].w) {
            std::cerr << "Test FAILED, at (" << x << "): ("
                      << (int)reference_out[x].x << ","
                      << (int)reference_out[x].y << ","
                      << (int)reference_out[x].z << ","
                      << (int)reference_out[x].w << ") vs. ("
                      << (int)output[x].x << ","
                      << (int)output[x].y << ","
                      << (int)output[x].z << ","
                      << (int)output[x].w << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cerr << "Test PASSED" << std::endl;

    // free memory
    delete[] input;
    delete[] reference_in;
    delete[] reference_out;

    return EXIT_SUCCESS;
}

