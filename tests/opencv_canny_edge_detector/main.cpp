//
// Copyright (c) 2014, Saarland University
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

//
// The Canny edge detector with simplified thresholding (no hysteresis
// thresholding).
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "opencv2/opencv.hpp"

#include "hipacc.hpp"

#define VIDEO

using namespace cv;
using namespace hipacc;


class GaussianBlurFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<float> &mask;

    public:
        GaussianBlurFilter(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                Mask<float> &mask) :
            Kernel(iter),
            input(input),
            mask(mask)
        { addAccessor(&input); }

        void kernel() {
            output() = (uchar)(convolve(mask, HipaccSUM, [&] () -> float {
                    return mask() * input(mask);
                    }) + 0.5f);
        }
};

class GradFilter : public Kernel<float> {
    private:
        Accessor<uchar> &input;
        Mask<int> &mask_x, &mask_y;
        Domain &dom_x, &dom_y;

    public:
        GradFilter(IterationSpace<float> &iter, Accessor<uchar> &input,
                Mask<int> &mask_x, Mask<int> &mask_y, Domain &dom_x, Domain
                &dom_y) :
            Kernel(iter),
            input(input),
            mask_x(mask_x),
            mask_y(mask_y),
            dom_x(dom_x),
            dom_y(dom_y)
        { addAccessor(&input); }

        void kernel() {
            int gx = reduce(dom_x, HipaccSUM, [&] () -> int {
                    return mask_x(dom_x) * input(dom_x);
                    });
            int gy = reduce(dom_y, HipaccSUM, [&] () -> int {
                    return mask_y(dom_y) * input(dom_y);
                    });

            float tmp = gx*gx + gy*gy;
            output() = sqrt(tmp);
        }
};
class NMSFilter : public Kernel<int> {
    private:
        Accessor<float> &input;

    public:
        NMSFilter(IterationSpace<int> &iter, Accessor<float> &input) :
            Kernel(iter),
            input(input)
        { addAccessor(&input); }

        void kernel() {
            int pixel = input();

            if (pixel <= 27 || (pixel > 162 && pixel <= 207) ) {
                if (pixel < input(0, 1) || pixel < input(0, -1)) {
                    output() = 0; return;
                }
                output() = pixel; return;
            }
            if ((pixel > 27 && pixel <= 72) || (pixel > 207 && pixel <= 252)) {
                if (pixel < input(-1, 1) || pixel < input(1, -1)) {
                    output() = 0; return;
                }
                output() = pixel; return;
            }
            if ((pixel > 72 && pixel <= 117) || (pixel > 252 && pixel <= 297)) {
                if (pixel < input(-1, 0) || pixel < input(1, 0)) {
                    output() = 0; return;
                }
                output() = pixel; return;
            }
            if ((pixel > 117 && pixel <= 162) || (pixel > 297 && pixel <= 360)) {
                if (pixel < input(-1, -1) || pixel < input(1, 1)) {
                    output() = 0; return;
                }
                output() = pixel; return;
            }
        }
};
class ThresholdFilter : public Kernel<uchar> {
    private:
        Accessor<int> &input;

    public:
        ThresholdFilter(IterationSpace<uchar> &iter, Accessor<int> &input) :
            Kernel(iter),
            input(input)
        { addAccessor(&input); }

        void kernel() {
            int pixel = input();

            if (pixel <= 20) {
                pixel = 0;
            } else if (pixel > 20 && pixel < 40) {
                pixel = 0;
            } else {
                pixel = 255;
            }

            output() = pixel;
        }
};


int main(int argc, const char **argv) {
    float timing = 0, fps_timing = 0;
    // RGB image
    Mat frameRGB;
    Mat frame;

    #ifdef VIDEO
    // open default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        fprintf(stderr, "Error opening VideoCapture device!\n");
        return EXIT_FAILURE;
    }

    // get first frame
    cap >> frameRGB;
    cvtColor(frameRGB, frame, CV_BGR2GRAY);
    imshow("Canny", frameRGB);
    #else
    // input image
    frame = imread("0003.pgm", CV_LOAD_IMAGE_GRAYSCALE);
    if (frame.empty()) {
        fprintf(stderr, "Error reading image file '0003.pgm'!\n");
        return EXIT_FAILURE;
    }
    #endif


    // images
    Image<uchar> input_img(frame.cols, frame.rows);
    Image<uchar> gauss_img(frame.cols, frame.rows);
    Image<float> grad_img(frame.cols, frame.rows);
    Image<int> nms_img(frame.cols, frame.rows);
    Image<uchar> output_img(frame.cols, frame.rows);


    // filter mask
    const float filter_mask_gauss[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };
    Mask<float> gauss_mask(filter_mask_gauss);
    const int filter_mask_sobel_y[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };
    const int filter_mask_sobel_x[3][3] = {
        { -1, 0,  1 },
        { -2, 0,  2 },
        { -1, 0,  1 }
    };
    Mask<int> sobel_mask_x(filter_mask_sobel_x);
    Mask<int> sobel_mask_y(filter_mask_sobel_y);
    Domain sobel_dom_x(sobel_mask_x);
    Domain sobel_dom_y(sobel_mask_y);


    // blur input image
    input_img = frame.data;
    BoundaryCondition<uchar> bound_input_img(input_img, gauss_mask, BOUNDARY_CLAMP);
    Accessor<uchar> acc_input_img(bound_input_img);
    IterationSpace<uchar> iter_gauss(gauss_img);

    GaussianBlurFilter gauss(iter_gauss, acc_input_img, gauss_mask);

    // compute edge gradient
    BoundaryCondition<uchar> bound_gauss_img(gauss_img, gauss_mask, BOUNDARY_CLAMP);
    Accessor<uchar> acc_gauss_img(bound_gauss_img);
    IterationSpace<float> iter_grad(grad_img);

    GradFilter grad(iter_grad, acc_gauss_img, sobel_mask_x, sobel_mask_y, sobel_dom_x, sobel_dom_y);

    // non-maximum suppression
    BoundaryCondition<float> bound_grad_img(grad_img, gauss_mask, BOUNDARY_CLAMP);
    Accessor<float> acc_grad_img(bound_grad_img);
    IterationSpace<int> iter_nsm(nms_img);

    NMSFilter nms(iter_nsm, acc_grad_img);

    // thresholding
    Accessor<int> acc_nms_img(nms_img);
    IterationSpace<uchar> iter_output(output_img);

    ThresholdFilter threshold(iter_output, acc_nms_img);

    Mat frameCanny(frame.rows, frame.cols, CV_8UC1);
    #ifdef VIDEO
    while (1) {
        // get a RGB frame
        cap >> frameRGB;

        // convert to grayscale
        cvtColor(frameRGB, frame, CV_BGR2GRAY);
    #endif
        input_img = frame.data;

        // blur input image
        gauss.execute();
        timing = hipaccGetLastKernelTiming();
        fps_timing = timing;
        fprintf(stderr, "HIPAcc Gaussian blur filter: %.3f ms\n", timing);

        // compute edge gradient
        grad.execute();
        timing = hipaccGetLastKernelTiming();
        fps_timing += timing;
        fprintf(stderr, "HIPAcc edge gradient filter: %.3f ms\n", timing);

        // perform non-maximum suppression
        nms.execute();
        timing = hipaccGetLastKernelTiming();
        fps_timing += timing;
        fprintf(stderr, "HIPAcc NMS filter: %.3f ms\n", timing);

        // final thresholding
        threshold.execute();
        timing = hipaccGetLastKernelTiming();
        fps_timing += timing;
        fprintf(stderr, "HIPAcc threshold filter: %.3f ms\n", timing);

        // fps time
        fprintf(stderr, "HIPAcc canny: %.3f ms, %f fps\n", fps_timing, 1000.0f/fps_timing);

        frameCanny.data = output_img.getData();

        // display frame
        imshow("Canny", frameCanny);

    #ifdef VIDEO
        // exit when key is pressed
        if (waitKey(1) >= 0) break;
    }
    #else
    waitKey(0);
    #endif

    return EXIT_SUCCESS;
}

