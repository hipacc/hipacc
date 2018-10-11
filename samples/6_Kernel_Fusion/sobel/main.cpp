//
// Copyright (c) 2018, University of Erlangen-Nuremberg
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
#include "hipacc.hpp"

#define PRINT_RESULT

using namespace hipacc;

// kernel description in Hipacc
class SobelFilter1DConv : public Kernel<float> {
    private:
        Accessor<float> &input;
        Mask<int> &mask;

    public:
        SobelFilter1DConv(IterationSpace<float> &iter, Accessor<float> &input,
                Mask<int> &mask):
            Kernel(iter),
            input(input),
            mask(mask)
        { add_accessor(&input); }

        void kernel() {
            output() = convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                    });
        }
};

class SobelFilterMagnitude : public Kernel<float> {
    private:
        Accessor<float> &inputX;
        Accessor<float> &inputY;

    public:
        SobelFilterMagnitude(IterationSpace<float> &iter, Accessor<float>
                &inputX, Accessor<float> &inputY):
            Kernel(iter),
            inputX(inputX),
            inputY(inputY)
        { add_accessor(&inputX); add_accessor(&inputY); }

        void kernel() {
            output() = sqrtf(inputX() * inputX() + inputY() * inputY());
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    // define filters
    int mask_x_row[3][3] = {
        { 0,  1,  0},
        { 0,  2,  0},
        { 0,  1,  0}
    };
    int mask_x_col[3][3] = {
        { 0,  0,  0},
        { 1,  0, -1},
        { 0,  0,  0}
    };

    int mask_y_row[3][3] = {
        { 0,  1,  0},
        { 0,  0,  0},
        { 0, -1,  0}
    };
    int mask_y_col[3][3] = {
        { 0,  0,  0},
        { 1,  2,  1},
        { 0,  0,  0}
    };

    // load input
    const int width = WIDTH;
    const int height = HEIGHT;

    float* input = new float[width*height];
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        input[y*width + x] = (y*width + x) % 199;
      }
    }

    // hipacc object decls
    Mask<int> MXR(mask_x_row);
    Mask<int> MXC(mask_x_col);
    Mask<int> MYR(mask_y_row);
    Mask<int> MYC(mask_y_col);

    Image<float> IN(width, height);
    Image<float> ImgRowBuf(width, height);
    Image<float> ImgRow(width, height);
    Image<float> ImgColBuf(width, height);
    Image<float> ImgCol(width, height);
    Image<float> OUT(width, height);

    IterationSpace<float> IS_OUT(OUT);
    IterationSpace<float> IS_ImgCol(ImgCol);
    IterationSpace<float> IS_ImgRow(ImgRow);
    IterationSpace<float> IS_ImgColBuf(ImgColBuf);
    IterationSpace<float> IS_ImgRowBuf(ImgRowBuf);

    const BoundaryCondition<float> BcInXClamp(IN, MXR, Boundary::CLAMP);
    Accessor<float> AccInXClamp(BcInXClamp);
    const BoundaryCondition<float> BcImgRBClamp(ImgRowBuf, MXC, Boundary::CLAMP);
    Accessor<float> AccImgRBClamp(BcImgRBClamp);
    Accessor<float> AccImgR(ImgRow);

    const BoundaryCondition<float> BcInYClamp(IN, MYR, Boundary::CLAMP);
    Accessor<float> AccInYClamp(BcInYClamp);
    const BoundaryCondition<float> BcImgCBClamp(ImgColBuf, MYC, Boundary::CLAMP);
    Accessor<float> AccImgCBClamp(BcImgCBClamp);
    Accessor<float> AccImgC(ImgCol);

    SobelFilter1DConv SFXR(IS_ImgRowBuf, AccInXClamp, MXR);
    SobelFilter1DConv SFXC(IS_ImgRow, AccImgRBClamp, MXC);

    SobelFilter1DConv SFYR(IS_ImgColBuf, AccInYClamp, MYR);
    SobelFilter1DConv SFYC(IS_ImgCol, AccImgCBClamp, MYC);

    SobelFilterMagnitude SFM(IS_OUT, AccImgR, AccImgC);

    // kernel invocation
    IN = input;
    SFXR.execute();
    SFYR.execute();
    SFXC.execute();
    SFYC.execute();
    SFM.execute();
    float *output = OUT.data();

#ifdef PRINT_RESULT
    std::cerr << "\nwriting output to file...\n";
    int ySkip = 17;
    int xSkip = 13;
    // compute the golden results
    for (int y=0; y<height; y=y+ySkip) {
      for (int x=0; x<width; x=x+xSkip) {
        std::cout << "output at y " << y << ", x " << x << " : " <<
          (int)output[y*width + x] <<"\n";
      }
    }
#endif


    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}

