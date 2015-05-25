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

//
// The optical flow implementation at hand uses the Census Transform:
//
// Stein, Fridtjof. "Efficient computation of optical flow using the census
// transform." Pattern Recognition. Springer Berlin Heidelberg, 2004. 79-86.
//

#include <iostream>

#include <opencv2/opencv.hpp>

#include "hipacc.hpp"

#define VIDEO
#define WINDOW_SIZE_X 30
#define WINDOW_SIZE_Y 30
#define EPSILON 30
//#define USE_LAMBDA

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
        { add_accessor(&input); }

        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    }) + 0.5f);
        }
};

inline uint ctn_t32(uchar data, uchar central, uint prev_result) {
    if (data > central + EPSILON) {
        return (prev_result << 2) | 0x01;
    } else if (data < central - EPSILON) {
        return (prev_result << 2) | 0x02;
    } else {
        return (prev_result << 2);
    }
}

class SignatureKernel : public Kernel<uint> {
    private:
        Accessor<uchar> &input;
        Domain &dom;

    public:
        SignatureKernel(IterationSpace<uint> &iter, Accessor<uchar> &input,
                        Domain &dom) :
            Kernel(iter),
            input(input),
            dom(dom)
        { add_accessor(&input); }

        void kernel() {
            // Census Transformation
            uchar z = input();
            uint c = 0u;
            #ifdef USE_LAMBDA
            iterate(dom, [&] () {
                c = ctn_t32(input(dom), z, c);
            });
            #else
            // row-4
            c = ctn_t32(input(-4, -4), z, c);
            c = ctn_t32(input( 0, -4), z, c);
            c = ctn_t32(input(+4, -4), z, c);
            // row-2
            c = ctn_t32(input(-2, -2), z, c);
            c = ctn_t32(input( 0, -2), z, c);
            c = ctn_t32(input(+2, -2), z, c);
            // row
            c = ctn_t32(input(-4,  0), z, c);
            c = ctn_t32(input(-2,  0), z, c);
            c = ctn_t32(input(+2,  0), z, c);
            c = ctn_t32(input(+4,  0), z, c);
            // row+2
            c = ctn_t32(input(-2, +2), z, c);
            c = ctn_t32(input( 0, +2), z, c);
            c = ctn_t32(input(+2, +2), z, c);
            // row+4
            c = ctn_t32(input(-4, +4), z, c);
            c = ctn_t32(input( 0, +4), z, c);
            c = ctn_t32(input(+4, +4), z, c);
            #endif

            output() = c;
        }
};

class VectorKernel : public Kernel<int> {
    private:
        Accessor<uint> &sig1, &sig2;
        Domain &dom;

    public:
        VectorKernel(IterationSpace<int> &iter, Accessor<uint> &sig1,
                Accessor<uint> &sig2, Domain &dom) :
            Kernel(iter),
            sig1(sig1),
            sig2(sig2),
            dom(dom)
        {
            add_accessor(&sig1);
            add_accessor(&sig2);
        }

        void kernel() {
            int vec_found = 0;
            int mem_loc = 0;

            uint reference = sig1();

            #ifdef USE_LAMBDA
            iterate(dom, [&] () -> void {
                    if (sig2(dom) == reference) {
                        vec_found++;
                        // encode ix and iy as upper and lower half-word of
                        // mem_loc
                        mem_loc = (dom.x() << 16) | (dom.y() & 0xffff);
                    }
                    });
            #else
            for (int iy=-WINDOW_SIZE_Y/4; iy<=WINDOW_SIZE_Y/4; iy++) {
                for (int ix=-WINDOW_SIZE_X/4; ix<=WINDOW_SIZE_X/4; ix++) {
                    if (sig2(ix, iy) == reference) {
                        if (iy==0 && ix==0) continue;
                        vec_found++;
                        // encode ix and iy as upper and lower half-word of
                        // mem_loc
                        mem_loc = (ix << 16) | (iy & 0xffff);
                    }
                }
            }
            #endif

            // save the vector, if exactly one was found
            if (vec_found!=1) {
                mem_loc = 0;
            }

            output() = mem_loc;
        }
};


static void filter(Mat& inputImage, Mat& outputImage) {
    uchar *inputBuffer = inputImage.data;
    uchar *outputBuffer = outputImage.data;
    int width = inputImage.cols;
    int height = inputImage.rows;

    // filter image
    for (int row = 1; row < height - 1; ++row) {
        for (int col = 1; col < width - 1; ++col) {
            // row-1
            uint c = inputBuffer[(row - 1) * width + (col - 1)];
            c += 2u * inputBuffer[(row - 1) * width + col];
            c += inputBuffer[(row - 1) * width + (col + 1)];
            // row
            c += 2u * inputBuffer[row * width + (col - 1)];
            c += 4u * inputBuffer[row * width + col];
            c += 2u * inputBuffer[row * width + (col + 1)];
            // row+1
            c += inputBuffer[(row + 1) * width + (col - 1)];
            c += 2u * inputBuffer[(row + 1) * width + col];
            c += inputBuffer[(row + 1) * width + (col + 1)];
            c /= 16u;
            outputBuffer[row * width + col] = (uchar) c;
        }
    }
}

static void generateSignature(Mat& inputImage, Mat& signature) {
    uchar *inputBuffer = inputImage.data;
    uint *outputBuffer = (uint *) signature.data;
    int width = inputImage.cols;
    int height = inputImage.rows;

    // signature generation
    for (int row = 4; row < height - 4; ++row) {
        for (int col = 4; col < width - 4; ++col) {
            // Census Transformation
            uchar z = inputBuffer[row * width + col];
            uint c = 0u;
            // row-4
            c = ctn_t32(inputBuffer[(row - 4) * width + (col - 4)], z, c);
            c = ctn_t32(inputBuffer[(row - 4) * width + col], z, c);
            c = ctn_t32(inputBuffer[(row - 4) * width + (col + 4)], z, c);
            // row-2
            c = ctn_t32(inputBuffer[(row - 2) * width + (col - 2)], z, c);
            c = ctn_t32(inputBuffer[(row - 2) * width + col], z, c);
            c = ctn_t32(inputBuffer[(row - 2) * width + (col + 2)], z, c);
            // row
            c = ctn_t32(inputBuffer[row * width + (col - 4)], z, c);
            c = ctn_t32(inputBuffer[row * width + (col - 2)], z, c);
            c = ctn_t32(inputBuffer[row * width + (col + 2)], z, c);
            c = ctn_t32(inputBuffer[row * width + (col + 4)], z, c);
            // row+2
            c = ctn_t32(inputBuffer[(row + 2) * width + (col - 2)], z, c);
            c = ctn_t32(inputBuffer[(row + 2) * width + col], z, c);
            c = ctn_t32(inputBuffer[(row + 2) * width + (col + 2)], z, c);
            // row+4
            c = ctn_t32(inputBuffer[(row + 4) * width + (col - 4)], z, c);
            c = ctn_t32(inputBuffer[(row + 4) * width + col], z, c);
            c = ctn_t32(inputBuffer[(row + 4) * width + (col + 4)], z, c);
            outputBuffer[row*width+col] = c;
        }
    }
}

static void generateVectors(Mat& signature1, Mat& signature2,
        vector<Point2i>& v0, vector< Point2i>& v1) {
    uint *referenceBuffer = (uint *) signature1.data;
    uint *checkBuffer = (uint *) signature2.data;
    int width = signature1.cols;
    int height = signature1.rows;
    int row_mem = 0;
    int col_mem = 0;
    // vector generation
    for (int row = (int) (WINDOW_SIZE_Y / 2); row < (height - (int) (WINDOW_SIZE_Y / 2)); row += 2) {
        for (int col = (int) (WINDOW_SIZE_X / 2); col < (width - (int) (WINDOW_SIZE_X / 2)); col += 2) {
            int vect_found = 0;
            uint reference = referenceBuffer[row * width + col];
            // check window
            for (int row_w = (row - (int) (WINDOW_SIZE_Y / 2)); row_w <= (row + (int) (WINDOW_SIZE_Y / 2)); row_w += 2) {
                for (int col_w = (col - (int) (WINDOW_SIZE_X / 2)); col_w <= (col + (int) (WINDOW_SIZE_X / 2)); col_w += 2) {
                    if (checkBuffer[row_w * width + col_w] == reference) {
                        if ((row_w == row) && (col_w == col))
                            continue;
                        vect_found++;
                        row_mem = row_w;
                        col_mem = col_w;
                    }
                }
            }

            // save the vector, if exactly one was found
            if (vect_found == 1) {
                Point p0(col, row);
                v0.push_back(p0);
                Point p1(col_mem, row_mem);
                v1.push_back(p1);
            }
        }
    }
}


int main(int argc, const char **argv) {
    float timing = 0, fps_timing = 0;
    // RGB image
    Mat frameRGB;
    Mat frame;

    #ifdef VIDEO
    // open default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening VideoCapture device!" << std::endl;
        return EXIT_FAILURE;
    }

    // get first frame
    cap >> frameRGB;
    cvtColor(frameRGB, frame, CV_BGR2GRAY);
    imshow("Optical Flow", frameRGB);
    #else
    // first frame
    std::string frame_name = "q5_00164.jpg";
    Mat frame = imread(frame_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (frame.empty()) {
        std::cerr << "Error reading image file '" << frame_name << "'!" << std::endl;
        return EXIT_FAILURE;
    }

    // second frame
    std::string frame2_name = "q5_00165.jpg";
    Mat frame2 = imread(frame2_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (frame2.empty()) {
        std::cerr << "Error reading image file '" << frame2_name << "'!" << std::endl;
        return EXIT_FAILURE;
    }
    #endif


    // images
    Image<uchar> img(frame.cols, frame.rows, frame.data);
    Image<uchar> prev(frame.cols, frame.rows);
    Image<uchar> filter_img(frame.cols, frame.rows);
    Image<uint> prev_signature(frame.cols, frame.rows);
    Image<uint> img_signature(frame.cols, frame.rows);


    // filter mask
    const float filter_mask[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };
    Mask<float> mask(filter_mask);

    // domain for signature kernel
    const uchar sig_coef[9][9] = {
      { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 0, 0, 1 }};
    Domain sig_dom(sig_coef);

    // vector image
    Image<int> img_vec((frame.cols-WINDOW_SIZE_X)/2, (frame.rows-WINDOW_SIZE_Y)/2);
    Accessor<uint> acc_img_sig(img_signature,   frame.cols-WINDOW_SIZE_X, frame.rows-WINDOW_SIZE_Y, WINDOW_SIZE_X/2, WINDOW_SIZE_Y/2, Interpolate::NN);
    Accessor<uint> acc_prev_sig(prev_signature, frame.cols-WINDOW_SIZE_X, frame.rows-WINDOW_SIZE_Y, WINDOW_SIZE_X/2, WINDOW_SIZE_Y/2, Interpolate::NN);
    IterationSpace<int> iter_vec(img_vec);

    // domain for vector kernel
    #ifdef USE_LAMBDA
    const uchar domain_vector[15][15] = {
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
        };
    Domain dom(domain_vector);
    #else
    Domain dom(WINDOW_SIZE_X/2, WINDOW_SIZE_Y/2);
    // do not process the center pixel
    dom(0, 0) = 0;
    #endif

    VectorKernel vector_kernel(iter_vec, acc_img_sig, acc_prev_sig, dom);


    // filter first image/frame
    BoundaryCondition<uchar> bound_img(img, mask, Boundary::CLAMP);
    Accessor<uchar> acc_img(bound_img);
    IterationSpace<uchar> iter_blur(filter_img);

    GaussianBlurFilter blur_img(iter_blur, acc_img, mask);

    blur_img.execute();
    timing = hipacc_last_kernel_timing();
    std::cerr << "Hipacc blur filter: " << timing << " ms" << std::endl;


    // generate signature for first image/frame
    BoundaryCondition<uchar> bound_fil(filter_img, sig_dom, Boundary::CLAMP);
    Accessor<uchar> acc_fil(bound_fil);
    IterationSpace<uint> iter_sig(img_signature);

    SignatureKernel sig_img(iter_sig, acc_fil, sig_dom);

    sig_img.execute();
    timing = hipacc_last_kernel_timing();
    std::cerr << "Hipacc signature kernel: " << timing << " ms" << std::endl;


    prev_signature = img_signature;
    prev = img;

    #ifdef VIDEO
    while (1) {
        // get a RGB frame
        cap >> frameRGB;

        // convert to grayscale
        cvtColor(frameRGB, frame, CV_BGR2GRAY);
        img = frame.data;

        // filter frame
        blur_img.execute();
        timing = hipacc_last_kernel_timing();
        fps_timing = timing;
        std::cerr << "Hipacc blur filter: " << timing << " ms" << std::endl;

        // generate signature for frame
        sig_img.execute();
        timing = hipacc_last_kernel_timing();
        fps_timing += timing;
        std::cerr << "Hipacc signature kernel: " << timing << " ms" << std::endl;

        // perform matching
        vector_kernel.execute();
        timing = hipacc_last_kernel_timing();
        fps_timing += timing;
        std::cerr << "Hipacc vector kernel: " << timing << " ms" << std::endl;

        // fps time
        std::cerr << "Hipacc optical flow: " << fps_timing << " ms, " << 1000.0f/fps_timing << " fps" << std::endl;

        int *vecs = img_vec.data();
        vector<Point2i> v0, v1;
        for (int y=0; y<img_vec.height(); y++) {
            for (int x=0; x<img_vec.width(); x++) {
                if (vecs[x + y*img_vec.width()]!=0) {
                    v0.push_back(Point(x*2 + WINDOW_SIZE_X/2, y*2 + WINDOW_SIZE_Y/2));
                    int loc = vecs[x + y*img_vec.width()];
                    int high = loc >> 16;
                    int low = (loc & 0xffff);
                    if (low >> 15) low |= 0xffff0000;
                    v1.push_back(Point(x*2 + high*2 + WINDOW_SIZE_X/2, y*2 + low*2 + WINDOW_SIZE_Y/2));
                }
            }
        }

        // draw lines
        for (size_t i = 0; i < v0.size(); i++) {
            line(frameRGB, v0[i], v1[i], CV_RGB(255,0,0), 1, CV_AA);
        }

        // switch pointers
        prev_signature = img_signature;
        prev = img;

        // display frame
        imshow("Optical Flow", frameRGB);

        // exit when key is pressed
        if (waitKey(1) >= 0) break;
    }
    #else
    // convert to grayscale
    img = frame2.data;

    // filter frame
    blur_img.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing = timing;
    std::cerr << "Hipacc blur filter: " << timing << " ms" << std::endl;

    // generate signature for frame
    sig_img.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    std::cerr << "Hipacc signature kernel: " << timing << " ms" << std::endl;

    // perform matching
    vector_kernel.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    std::cerr << "Hipacc vector kernel: " << timing << " ms" << std::endl;

    // fps time
    std::cerr << "Hipacc optical flow: " << fps_timing << " ms, " << 1000.0f/fps_timing << " fps" << std::endl;

    int *vecs = img_vec.data();
    vector<Point2i> v0, v1;
    for (int y=0; y<img_vec.height(); y++) {
        for (int x=0; x<img_vec.width(); x++) {
            if (vecs[x + y*img_vec.width()]!=0) {
                v0.push_back(Point(x*2 + WINDOW_SIZE_X/2, y*2 + WINDOW_SIZE_Y/2));
                int loc = vecs[x + y*img_vec.width()];
                int high = loc >> 16;
                int low = (loc & 0xffff);
                if (low >> 15) low |= 0xffff0000;
                v1.push_back(Point(x*2 + high*2 + WINDOW_SIZE_X/2, y*2 + low*2 + WINDOW_SIZE_Y/2));
            }
        }
    }

    // convert to color space
    cvtColor(frame2, frameRGB, CV_GRAY2BGR);

    // draw lines
    for (size_t i = 0; i < v0.size(); i++) {
        line(frameRGB, v0[i], v1[i], CV_RGB(255,0,0), 1, CV_AA);
    }

    // display frame
    imshow("Optical Flow", frameRGB);
    waitKey(0);
    #endif

    return EXIT_SUCCESS;
}

