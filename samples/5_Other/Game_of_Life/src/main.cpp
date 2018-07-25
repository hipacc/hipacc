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


#define WIDTH  800
#define HEIGHT 600


using namespace hipacc;
using namespace hipacc::math;


class GoL : public Kernel<uchar> {
    private:
        Accessor<uchar> &in;
        Domain &dom;

    public:
        GoL(IterationSpace<uchar> &iter, Accessor<uchar> &in, Domain &dom)
              : Kernel(iter), in(in), dom(dom) {
            add_accessor(&in);
        }

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


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;

    uchar* input = load_data<uchar>(width, height);
    for (int p = 0; p < width*height; ++p) {
       input[p] = input[p] < 128 ? 0 : 1;
    }

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(width, height, input);
    Image<uchar> out(width, height);

    // define Domain for blur filter
    Domain dom(3, 3);
    dom(0,0) = 0;

    BoundaryCondition<uchar> bound(in, dom, Boundary::REPEAT);
    Accessor<uchar> acc(bound);

    IterationSpace<uchar> iter(out);
    GoL kernel(iter, acc, dom);

    while (true) {
        // filter frame
        kernel.execute();
        std::cout << "Hipacc GoL kernel: "
                  << hipacc_last_kernel_timing() << " ms" << std::endl;

        uchar *output = out.data();
        for (int p = 0; p < width*height; ++p) {
           output[p] = output[p] * 255;
        }

        // display frame
        if (show_data(width, height, 1, output, "Game of Life", 1))
            break;

        in = out;
    }

    //************************************************************************//

    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}
