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

#include "hipacc.hpp"

#include <iostream>
#include <hipacc_helper.hpp>


#define WIDTH  512
#define HEIGHT 512
#define IMAGE1 "../../common/img/q5_00164.jpg"
#define IMAGE2 "../../common/img/q5_00165.jpg"

#define SIZE_X        7
#define SIZE_Y        7
#define EPSILON       10
#define MAX_DIST_X    15
#define MAX_DIST_Y    3
#define THRESHOLD     100
#define TILE_SIZE     16

//#define USE_POLAR
#define PI            3.14159265f
#define HALF_PI       (PI/2.0f)
#define STRIDE_X      ((WIDTH+TILE_SIZE-1)/TILE_SIZE)
#define STRIDE_Y      ((HEIGHT+TILE_SIZE-1)/TILE_SIZE)


using namespace hipacc;
using namespace hipacc::math;


class GaussianBlur : public Kernel<uchar> {
    private:
        Accessor<uchar4> &input;
        Mask<float> &mask;

    public:
        GaussianBlur(IterationSpace<uchar> &iter, Accessor<uchar4> &input,
                     Mask<float> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                    uchar4 pixel = input(mask);
                    float intensity = .3f*pixel.x + .59f*pixel.y + .11f*pixel.z;
                    return mask() * intensity;
                }) + 0.5f);
        }
};

class SignatureKernel : public Kernel<uint> {
    private:
        Accessor<uchar> &input;
        Domain &dom;

    public:
        SignatureKernel(IterationSpace<uint> &iter, Accessor<uchar> &input,
                        Domain &dom)
              : Kernel(iter), input(input), dom(dom) {
            add_accessor(&input);
        }

        void kernel() {
            // Census Transformation
            uchar z = input();
            uint c = 0u;
            iterate(dom, [&] () {
                    uchar data = input(dom);
                    if (data > z + EPSILON) {
                        c = (c << 2) | 0x01;
                    } else if (data < z - EPSILON) {
                        c = (c << 2) | 0x02;
                    } else {
                        c = c << 2;
                    }
                });

            output() = c;
        }
};

class DeltaKernel : public Kernel<uint> {
    private:
        Accessor<uchar> &input1;
        Accessor<uchar> &input2;
        Domain &dom;

    public:
        DeltaKernel(IterationSpace<uint> &iter, Accessor<uchar> &input1,
                    Accessor<uchar> &input2, Domain &dom)
              : Kernel(iter), input1(input1), input2(input2), dom(dom) {
            add_accessor(&input1);
            add_accessor(&input2);
        }

        void kernel() {
            output() = reduce(dom, Reduce::SUM, [&] () -> uint {
                    int in1 = input1(dom);
                    int in2 = input2(dom);
                    int diff = in1 - in2;
                    if (diff < 0) diff *= -1;
                    return diff;
                });
        }
};

class VectorKernel : public Kernel<int> {
    private:
        Accessor<uint> &sig1, &sig2, &delta;
        Domain &dom;

    public:
        VectorKernel(IterationSpace<int> &iter, Accessor<uint> &sig1,
                     Accessor<uint> &sig2, Accessor<uint> &delta, Domain &dom)
              : Kernel(iter), sig1(sig1), sig2(sig2), delta(delta), dom(dom) {
            add_accessor(&sig1);
            add_accessor(&sig2);
        }

        void kernel() {
            int vec_found = 0;
            int mem_loc = 0;
            uint max_delta = 0;

            uint reference = sig1();

            iterate(dom, [&] () -> void {
                    if (sig2(dom) == reference) {
                        // BUG: ++operator is not recognized as assignment
                        vec_found = vec_found + 1;

                        // encode x and y as upper and lower half-word
                        mem_loc = (dom.x() << 16) | (dom.y() & 0xffff);

                        uint d = delta(dom);
                        if (d > max_delta) max_delta = d;
                    }
                });

            // save the vector, if exactly one was found
            if (vec_found!=1 || max_delta < THRESHOLD) {
                mem_loc = 0;
            }

            output() = mem_loc;
        }
};

class VectorMerge : public Kernel<int, float4> {
    private:
        Accessor<int> &input;

    public:
        VectorMerge(IterationSpace<int> &iter, Accessor<int> &input)
              : Kernel(iter), input(input) {
            add_accessor(&input);
        }

        void kernel() {
            output() = input();
        }

        void binning(uint x, uint y, int vector) {
            if (vector != 0) {
                float4 result = { 0.0f, 0.0f, 0.0f, 1.0f };

                int xi = vector >> 16;
                int yi = (short)(vector & 0xffff);

#ifdef USE_POLAR
                // Cartesian to polar
                float xf = (float)xi;
                float yf = (float)yi;
                float dist = sqrt(xf*xf+yf*yf);
                float angle = atan(yf/xf);
                if (xf < 0.0f) dist *= -1.0f;
                result.x = dist;
                result.y = angle+HALF_PI;
#else
                result.x = xi;
                result.y = yi;
#endif

                // target tile is at midway (xi/2 & yi/2) of vector
                uint xt = (x+xi/2) / TILE_SIZE;
                uint yt = (y+yi/2) / TILE_SIZE;
                bin((uint)(yt * STRIDE_X + xt)) = result;
            }
        }

        float4 reduce(float4 left, float4 right) const {
            if (left.w == 0.0f) {
                return right;
            } else if (right.w == 0.0f) {
                return left;
            } else {
                float4 result;

                // compute weights
                float ws = left.w + right.w;
                float wl = left.w/ws;
                float wr = 1.0f-wl;
                result.w = ws;

#ifdef USE_POLAR
                // average distance
                result.x = wl*left.x + wr*right.x;

                // average angle
                float al = left.y;
                float ar = right.y;
                float diff = al-ar;

                if (diff > HALF_PI || diff < -HALF_PI) {
                    if (ar < al) ar += PI;
                    else al += PI;
                }

                float as = wl*al + wr*ar;
                if (as > PI) as -= PI;
                result.y = as;
#else
                result.x = wl*left.x + wr*right.x;
                result.y = wl*left.y + wr*right.y;
#endif

                return result;
            }
        }
};


class Assemble : public Kernel<uchar4> {
    private:
        Accessor<uchar4> &input;
        Accessor<float4> &vecs;

    public:
        Assemble(IterationSpace<uchar4> &iter, Accessor<uchar4> &input,
                 Accessor<float4> &vecs)
              : Kernel(iter), input(input), vecs(vecs) {
            add_accessor(&input);
            add_accessor(&vecs);
        }

        void kernel() {
            int x = 0;
            int y = 0;
            float4 vector = vecs();

            if (vector.w != 0.0f) {
#ifdef USE_POLAR
                float angle = vector.y - HALF_PI;
                float dist = vector.x;

                // polar to Cartesian
                float xf = dist * cosf(angle);
                float yf = dist * sinf(angle);
#else
                float xf = vector.x;
                float yf = vector.y;
#endif

                // half distance and opposite direction
                xf *= -.5f;
                yf *= -.5f;

                // correct rounding
                x = (int)(xf + (xf < 0 ? -.5f : .5f));
                y = (int)(yf + (yf < 0 ? -.5f : .5f));
            }

            output() = input(x, y);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
		const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    float timing = 0;

		// only filter kernel sizes 3x3, 5x5, and 7x7 implemented
    if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
        std::cerr << "Wrong filter kernel size. "
                  << "Currently supported values: 3x3, 5x5, and 7x7!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float gauss_coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
#endif
#if SIZE_X == 5
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f }
#endif
#if SIZE_X == 7
        { 0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841 },
        { 0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010 },
        { 0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471 },
        { 0.008351, 0.029902, 0.064280, 0.082959, 0.064280, 0.029902, 0.008351 },
        { 0.006471, 0.023169, 0.049806, 0.064280, 0.049806, 0.023169, 0.006471 },
        { 0.003010, 0.010778, 0.023169, 0.029902, 0.023169, 0.010778, 0.003010 },
        { 0.000841, 0.003010, 0.006471, 0.008351, 0.006471, 0.003010, 0.000841 }
#endif
    };

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
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 }
    };

    // host memory for image of width x height pixels
    uchar4 *input1 = (uchar4*)load_data<uchar>(width, height, 4, IMAGE1);
    uchar4 *input2 = (uchar4*)load_data<uchar>(width, height, 4, IMAGE2);

    std::cout << "Calculating Hipacc motion interpolation ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar4> in1(width, height, input1);
    Image<uchar4> in2(width, height, input2);
    Image<uchar> tmp1(width, height);
    Image<uchar> tmp2(width, height);
    Image<uint> in1_sig(width, height);
    Image<uint> in2_sig(width, height);
    Image<uint> delta(width, height);
    Image<uchar4> out(width, height);
    Image<int> img_vec(width, height);
    Image<float4> merged_vec(STRIDE_X, STRIDE_Y);

    // define Mask for Gaussian blur filter
    Mask<float> mask(gauss_coef);

    // define Domain for signature kernel
    Domain sig_dom(sig_coef);

    // Domain for vector kernel
    Domain dom(MAX_DIST_X*2+1, MAX_DIST_Y*2+1);
    // do not process the center pixel
    dom(0,0) = 0;

    // filter first image
    BoundaryCondition<uchar4> bound_in1(in1, mask, Boundary::CLAMP);
    Accessor<uchar4> acc_in1(bound_in1);
    IterationSpace<uchar> iter_blur1(tmp1);
    GaussianBlur blur1(iter_blur1, acc_in1, mask);
    blur1.execute();
    timing += hipacc_last_kernel_timing();

    // generate signature for first image
    BoundaryCondition<uchar> bound_tmp1(tmp1, sig_dom, Boundary::CLAMP);
    Accessor<uchar> acc_tmp1(bound_tmp1);
    IterationSpace<uint> iter_in1_sig(in1_sig);
    SignatureKernel sig1(iter_in1_sig, acc_tmp1, sig_dom);
    sig1.execute();
    timing += hipacc_last_kernel_timing();

    // filter second image
    BoundaryCondition<uchar4> bound_in2(in2, mask, Boundary::CLAMP);
    Accessor<uchar4> acc_in2(bound_in2);
    IterationSpace<uchar> iter_blur2(tmp2);
    GaussianBlur blur2(iter_blur2, acc_in2, mask);
    blur2.execute();
    timing += hipacc_last_kernel_timing();

    // generate signature for second image
    BoundaryCondition<uchar> bound_tmp2(tmp2, sig_dom, Boundary::CLAMP);
    Accessor<uchar> acc_tmp2(bound_tmp2);
    IterationSpace<uint> iter_in2_sig(in2_sig);
    SignatureKernel sig2(iter_in2_sig, acc_tmp2, sig_dom);
    sig2.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<uint> iter_delta(delta);
    DeltaKernel del(iter_delta, acc_tmp1, acc_tmp2, sig_dom);
    del.execute();
    timing += hipacc_last_kernel_timing();

    // compute motion vectors
    BoundaryCondition<uint> bound_in2_sig(in2_sig, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_in2_sig(bound_in2_sig);
    BoundaryCondition<uint> bound_in1_sig(in1_sig, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_in1_sig(bound_in1_sig);
    BoundaryCondition<uint> bound_delta(delta, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_delta(bound_delta);
    IterationSpace<int> iter_vec(img_vec);
    VectorKernel vector_kernel(iter_vec, acc_in1_sig, acc_in2_sig, acc_delta, dom);
    vector_kernel.execute();
    timing += hipacc_last_kernel_timing();

    // merge motion vectors
    Accessor<int> acc_vec(img_vec);
    VectorMerge merge(iter_vec, acc_vec);

    // get merged vectors
    float4* vecs = merge.binned_data(STRIDE_X*STRIDE_Y);
    timing += hipacc_last_kernel_timing();

    // load vectors into image 'merged_vec'
    merged_vec = vecs;

    // assemble final image
    IterationSpace<uchar4> iter_out(out);
    Accessor<float4> acc_merged_vec(merged_vec, Interpolate::NN);
    BoundaryCondition<uchar4> bound_asm_in1(in1, dom, Boundary::CLAMP);
    Accessor<uchar4> acc_asm_in1(bound_asm_in1);
    Assemble assemble(iter_out, acc_in1, acc_merged_vec);
    assemble.execute();
    timing += hipacc_last_kernel_timing();

    uchar4 *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    save_data(width, height, 4, (uchar*)input1, "frame1.jpg");
    save_data(width, height, 4, (uchar*)output, "frame2.jpg");
    save_data(width, height, 4, (uchar*)input2, "frame3.jpg");
    show_data(width, height, 4, (uchar*)output, "frame2.jpg");

    // free memory
    delete[] input1;
    delete[] input2;

    return EXIT_SUCCESS;
}
