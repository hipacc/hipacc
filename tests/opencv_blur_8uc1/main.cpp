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
//#define SIZE_X 5
//#define SIZE_Y 5
//#define WIDTH 4096
//#define HEIGHT 4096

// code variants
#define ARRAY_DOMAIN
#define CONST_DOMAIN
#define USE_LAMBDA

#define NT 100
#define SIMPLE

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// blur filter reference
#ifdef SIMPLE
void blur_filter(uchar *in, uchar *out, int size_x, int size_y, int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int y=anchor_y; y<upper_y; ++y) {
        for (int x=anchor_x; x<upper_x; ++x) {
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += in[(y + yf)*width + x + xf];
                }
            }
            out[y*width + x] = (uchar) ((1.0f/(float)(size_x*size_y))*sum);
        }
    }
}
#else
void blur_filter(uchar *in, uchar *out, int size_x, int size_y, int t, int width, int height) {
    int anchor_x = size_x >> 1;
    int anchor_y = size_y >> 1;
    int upper_x = width  - anchor_x;
    int upper_y = height - anchor_y;

    for (int x=anchor_x; x<upper_x; ++x) {
        for (int t0=anchor_y; t0<upper_y; t0+=t) {
            int sum = 0;

            // first phase: convolution
            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += in[(t0 + yf)*width + x + xf];
                }
            }
            out[t0*width + x] = (uchar) ((1.0f/(float)(size_x*size_y))*sum);

            // second phase: rolling sum
            for (int dt=1; dt<min(t, upper_y-t0); ++dt) {
                int t = t0+dt;
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum -= in[(t-anchor_y-1)*width + x + xf];
                    sum += in[(t-anchor_y-1+size_y)*width + x + xf];
                }
                out[t*width + x] = (uchar) ((1.0f/(float)(size_x*size_y))*sum);
            }
        }
    }
}
#endif


// Kernel description in Hipacc
class BlurFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &in;
        Domain &dom;
        int size_x, size_y;
        #ifndef SIMPLE
        int nt, height;
        #endif

    public:
        BlurFilter(IterationSpace<uchar> &iter, Accessor<uchar> &in, Domain
                &dom, int size_x, int size_y
                #ifndef SIMPLE
                , int nt, int height
                #endif
                ) :
            Kernel(iter), in(in), dom(dom),
            size_x(size_x), size_y(size_y)
            #ifndef SIMPLE
            , nt(nt), height(height)
            #endif
        { add_accessor(&in); }

        #ifdef SIMPLE
        void kernel() {
            #ifdef USE_LAMBDA
            output() = (uchar) ( (1.0f/(float)(size_x*size_y)) *
                    reduce(dom, Reduce::SUM, [&] () -> int {
                        return in(dom);
                    }));
            #else
            int anchor_x = size_x >> 1;
            int anchor_y = size_y >> 1;
            int sum = 0;

            for (int yf = -anchor_y; yf<=anchor_y; yf++) {
                for (int xf = -anchor_x; xf<=anchor_x; xf++) {
                    sum += in(xf, yf);
                }
            }
            output() = (uchar) ((1.0f/(float)(size_x*size_y))*sum);
            #endif
        }
        #else
        void kernel() {
            int anchor_x = size_x >> 1;
            int anchor_y = size_y >> 1;
            int sum = 0;

            int t0 = y();

            // first phase: convolution
            for (int yf = -anchor_y; yf<=anchor_y; ++yf) {
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum += in.pixel_at(in.x() + xf, t0*nt + yf);
                }
            }
            output_at(x(), t0*nt) = (uchar) ((1.0f/(float)(size_x*size_y))*sum);

            // second phase: rolling sum
            for (int dt=1; dt<min(nt, height-(t0*nt)); ++dt) {
                int t = t0*nt + dt;
                for (int xf = -anchor_x; xf<=anchor_x; ++xf) {
                    sum -= in.pixel_at(in.x() + xf, t-anchor_y-1);
                    sum += in.pixel_at(in.x() + xf, t-anchor_y-1+size_y);
                }
                output_at(x(), t) = (uchar) ((1.0f/(float)(size_x*size_y))*sum);
            }
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
    const int t = NT;

    // only filter kernel sizes 3x3 and 5x5 implemented
    if (size_x != size_y && (size_x != 3 || size_x != 5)) {
        std::cerr << "Wrong filter kernel size. Currently supported values: 3x3 and 5x5!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // domain for blur filter
    #ifdef ARRAY_DOMAIN
    #ifdef CONST_DOMAIN
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
    #endif

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
    Image<uchar> in(width, height, input);
    Image<uchar> out(width, height);

    // define Domain for blur filter
    #ifdef ARRAY_DOMAIN
    Domain dom(domain);
    #else
    Domain dom(size_x, size_y);
    #endif

    #ifndef OPENCV
    std::cerr << "Calculating Hipacc blur filter ..." << std::endl;
    std::vector<float> timings_hipacc;
    float timing = 0;

    BoundaryCondition<uchar> bound(in, dom, Boundary::CLAMP);
    Accessor<uchar> acc(bound);

    #ifdef SIMPLE
    IterationSpace<uchar> iter(out);
    BlurFilter filter(iter, acc, dom, size_x, size_y);
    #else
    IterationSpace<uchar> iter(out, width, (int)ceil((float)height/t));
    BlurFilter filter(iter, acc, dom, size_x, size_y, t, height);
    #endif

    filter.execute();
    timing = hipacc_last_kernel_timing();
    timings_hipacc.push_back(timing);
    std::cerr << "Hipacc (CLAMP): " << timing << " ms, " << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    // get pointer to result data
    uchar *output = out.data();

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
              << "Calculating OpenCV-CPU blur filter on CPU" << std::endl;
    compute_tapi(timings_cpu);

    // OpenCV - OpenCL
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cerr << std::endl
                  << "Calculating OpenCV-OCL blur filter on "
                  << cv::ocl::Device::getDefault().name() << std::endl;
        compute_tapi(timings_ocl);
    }

    // OpenCV - CUDA
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        #ifdef OPENCV_CUDA_FOUND
        std::cerr << std::endl
                  << "Calculating OpenCV-CUDA blur filter" << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

        cv::cuda::GpuMat dev_src, dev_dst;
        cv::Ptr<cv::cuda::Filter> blur;

        opencv_bench(
            [&] (int brd_type) {
                dev_src.upload(cv_data_src);
                blur = cv::cuda::createBoxFilter(dev_src.type(), dev_src.type(), ksize, anchor, brd_type);
            },
            [&] (int) {
                blur->apply(dev_src, dev_dst);
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

        #ifdef SIMPLE
        blur_filter(reference_in, reference_out, size_x, size_y, width, height);
        #else
        blur_filter(reference_in, reference_out, size_x, size_y, t, width, height);
        #endif

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

