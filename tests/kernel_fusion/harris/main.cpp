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

#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hipacc.hpp"

//#define PRINT_RESULT

using namespace hipacc;

// Harris Corner filter in HIPAcc
class Sobel : public Kernel<int> {
  private:
    Accessor<int> &Input;
    Mask<int> &cMask;
    Domain &dom;

  public:
    Sobel(IterationSpace<int> &IS,
            Accessor<int> &Input, Mask<int> &cMask, Domain &dom)
          : Kernel(IS),
            Input(Input),
            cMask(cMask),
            dom(dom) {
      add_accessor(&Input);
    }

    void kernel() {
      int sum = 0;
      sum += reduce(dom, Reduce::SUM, [&] () -> int {
          return Input(dom) * cMask(dom);
      });
      output() = sum / 6;
    }
};

class Square1 : public Kernel<int> {
  private:
    Accessor<int> &Input;

  public:
    Square1(IterationSpace<int> &IS,
            Accessor<int> &Input)
          : Kernel(IS),
            Input(Input) {
      add_accessor(&Input);
    }

    void kernel() {
      int in = Input();
      output() = in * in;
    }
};

class Square2 : public Kernel<int> {
  private:
    Accessor<int> &Input1;
    Accessor<int> &Input2;

  public:
    Square2(IterationSpace<int> &IS,
            Accessor<int> &Input1,
            Accessor<int> &Input2)
          : Kernel(IS),
            Input1(Input1),
            Input2(Input2) {
      add_accessor(&Input1);
      add_accessor(&Input2);
    }

    void kernel() {
      output() = Input1() * Input2();
    }
};

class Gauss : public Kernel<int> {
  private:
    Accessor<int> &Input;
    Mask<int> &cMask;

  public:
    Gauss(IterationSpace<int> &IS,
            Accessor<int> &Input, Mask<int> &cMask)
          : Kernel(IS),
            Input(Input),
            cMask(cMask) {
      add_accessor(&Input);
    }

    void kernel() {
      int sum = 0;
      sum += convolve(cMask, Reduce::SUM, [&] () -> float {
          return Input(cMask) * cMask();
      });
      output() = sum / 16;
    }
};

class HarrisCorner : public Kernel<int> {
  private:
    Accessor<int> &Dx;
    Accessor<int> &Dy;
    Accessor<int> &Dxy;

  public:
    HarrisCorner(IterationSpace<int> &IS,
            Accessor<int> &Dx, Accessor<int> &Dy, Accessor<int> &Dxy)
          : Kernel(IS),
            Dx(Dx),
            Dy(Dy),
            Dxy(Dxy) {
      add_accessor(&Dx);
      add_accessor(&Dy);
      add_accessor(&Dxy);
    }

    void kernel() {
      float k = 0.04f;
      float threshold = 20000.0f;
      int x = Dx();
      int y = Dy();
      int xy = Dxy();
      float R = 0;
      R = ((x * y) - (xy * xy)) /* det   */
          - (k * (x + y) * (x + y)); /* trace */
      int out = 0;
      if (R > threshold)
        out = 1;
      output() = out;
    }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {

#ifdef PRINT_RESULT 
    const int width = 24;
    const int height = 24;
#else
    const int width = WIDTH;
    const int height = HEIGHT;
#endif

    // host memory for image of width*height pixels
    int *host_in = new int[width*height];

    // initialize data
#ifdef PRINT_RESULT 
    int host_in_test[width*height] = { 
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
             0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,255,255,255,255,255,255,255,255, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };
    std::memcpy(host_in, host_in_test, width*height*4);
#else
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        host_in[y*width + x] = (y*width + x) % 199;
      }
    }
#endif

    // convolution filter mask
    const int filter_xy[3][3] = {
        { 1, 2, 1 },
        { 2, 4, 2 },
        { 1, 2, 1 }
    };
    const  int mask_x[3][3] = {  
        {-1,  0,  1},  
        {-1,  0,  1},  
        {-1,  0,  1} 
    };
    const  int mask_y[3][3] = {  
        {-1, -1, -1},  
        { 0,  0,  0},  
        { 1,  1,  1} 
    };

    Mask<int> G(filter_xy);
    Mask<int> MX(mask_x);
    Mask<int> MY(mask_y);
    Domain DomX(MX);
    Domain DomY(MY);

    // hipacc object decls
    // input, buffer, and output image
    Image<int> IN(width, height);
    Image<int> OUT(width, height);
    Image<int> DX(width, height);
    Image<int> DY(width, height);
    Image<int> SX(width, height);
    Image<int> SY(width, height);
    Image<int> SXY(width, height);
    Image<int> GX(width, height);
    Image<int> GY(width, height);
    Image<int> GXY(width, height);

    const BoundaryCondition<int> BcInClamp(IN, MX, Boundary::CLAMP);
    Accessor<int> AccInClamp(BcInClamp);

    IterationSpace<int> IsDx(DX);
    Accessor<int> AccDx(DX);
    IterationSpace<int> IsDy(DY);
    Accessor<int> AccDy(DY);

    IterationSpace<int> IsSx(SX);
    const BoundaryCondition<int> BcInClampSx(SX, G, Boundary::CLAMP);
    Accessor<int> AccInClampSx(BcInClampSx);

    IterationSpace<int> IsSy(SY);
    const BoundaryCondition<int> BcInClampSy(SY, G, Boundary::CLAMP);
    Accessor<int> AccInClampSy(BcInClampSy);

    IterationSpace<int> IsSxy(SXY);
    const BoundaryCondition<int> BcInClampSxy(SXY, G, Boundary::CLAMP);
    Accessor<int> AccInClampSxy(BcInClampSxy);

    IterationSpace<int> IsGx(GX);
    Accessor<int> AccGx(GX);
    IterationSpace<int> IsGy(GY);
    Accessor<int> AccGy(GY);
    IterationSpace<int> IsGxy(GXY);
    Accessor<int> AccGxy(GXY);

    IterationSpace<int> IsOut(OUT);

    Sobel DerivX(IsDx, AccInClamp, MX, DomX);
    Sobel DerivY(IsDy, AccInClamp, MY, DomY);

    Square1 SquareX(IsSx, AccDx);
    Square1 SquareY(IsSy, AccDy);
    Square2 SquareXY(IsSxy, AccDx, AccDy);

    Gauss GaussX(IsGx, AccInClampSx, G);
    Gauss GaussY(IsGy, AccInClampSy, G);
    Gauss GaussXY(IsGxy, AccInClampSxy, G);

    HarrisCorner HC(IsOut, AccGx, AccGy, AccGxy);

    // kernel invocation
    IN = host_in;
    DerivX.execute();
    DerivY.execute();
    SquareX.execute();
    SquareY.execute();
    SquareXY.execute();
    GaussX.execute();
    GaussY.execute();
    GaussXY.execute();
    HC.execute();

    // get pointer to result data
    int *output = OUT.data();

#ifdef PRINT_RESULT 
    int i,j;
    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            //fprintf(stdout,"%d ", host_out[i*width+j]);
            if (output[i*width+j] == 1) {
                fprintf(stdout,"X ");
            } else {
                fprintf(stdout,"- ");
            }
        }
        fprintf(stdout,"\n");
    }
#endif

    // free memory
    delete[] host_in;

    return EXIT_SUCCESS;
}

