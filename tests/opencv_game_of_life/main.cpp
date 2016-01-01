//
// Copyright (c) 2016, Saarland University
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

#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "hipacc.hpp"

using namespace cv;
using namespace hipacc;


class GoLKernel : public Kernel<uchar> {
    private:
        Accessor<uchar> &in;
        Domain &dom;

    public:
        GoLKernel(IterationSpace<uchar> &iter, Accessor<uchar> &in, Domain &dom) :
            Kernel(iter), in(in), dom(dom) { add_accessor(&in); }

        void kernel() {
            // Any live cell with fewer than two live neighbors dies, as if caused by under-population.
            // Any live cell with two or three live neighbors lives on to the next generation.
            // Any live cell with more than three live neighbors dies, as if by over-population.
            // Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
            uchar neighbors = reduce(dom, Reduce::SUM, [&] () -> uchar {
                                return in(dom);
                              });
            uchar n2 = neighbors == 2;
            uchar n3 = neighbors == 3;
            output() = in() * n2 + n3;
        }
};


int main(int argc, const char **argv) {
    const int width  = 800;
    const int height = 600;

    const uchar mask[3][3] = {
        { 1, 1, 1 },
        { 1, 0, 1 },
        { 1, 1, 1 }
    };
    Domain dom(mask);

    Image<uchar> src(width, height);
    Image<uchar> dst(width, height);
    BoundaryCondition<uchar> bound_src(src, dom, Boundary::REPEAT);
    Accessor<uchar> acc_src(bound_src);
    IterationSpace<uchar> iter_dst(dst);

    Mat frame(height, width, CV_8UC1);
    randu(frame, Scalar::all(0), Scalar::all(2));
    src = frame.data;

    GoLKernel gol(iter_dst, acc_src, dom);
    imshow("Game of Life", frame * 255);

    while (1) {
        // filter frame
        gol.execute();
        std::cerr << "Hipacc GoL kernel: " << hipacc_last_kernel_timing() << " ms" << std::endl;

        // display frame
        frame.data = dst.data();
        src = dst;
        imshow("Game of Life", frame * 255);

        // exit when key is pressed
        if (waitKey(1) >= 0)
            break;
    }

    return EXIT_SUCCESS;
}
