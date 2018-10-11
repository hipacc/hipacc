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


#define SIZE_X 3
#define SIZE_Y 3
#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"

#if SIZE_X == 7
# define data_t ushort
#else
# define data_t uchar
#endif


using namespace hipacc;
using namespace hipacc::math;


// Harris Corner filter in Hipacc
class Sobel : public Kernel<short> {
  private:
    Accessor<uchar> &Input;
    Mask<char> &cMask;
    Domain &dom;

  public:
    Sobel(IterationSpace<short> &IS, Accessor<uchar> &Input, Mask<char> &cMask,
          Domain &dom)
          : Kernel(IS), Input(Input), cMask(cMask), dom(dom) {
        add_accessor(&Input);
    }

    void kernel() {
        short sum = 0;
        sum += reduce(dom, Reduce::SUM, [&] () -> short {
                return Input(dom) * cMask(dom);
            });
        output() = sum / 6;
    }
};

class Square1 : public Kernel<short> {
  private:
    Accessor<short> &Input;

  public:
    Square1(IterationSpace<short> &IS, Accessor<short> &Input)
          : Kernel(IS), Input(Input) {
        add_accessor(&Input);
    }

    void kernel() {
        short in = Input();
        output() = in * in;
    }
};

class Square2 : public Kernel<short> {
  private:
    Accessor<short> &Input1;
    Accessor<short> &Input2;

  public:
    Square2(IterationSpace<short> &IS, Accessor<short> &Input1,
            Accessor<short> &Input2)
          : Kernel(IS), Input1(Input1), Input2(Input2) {
        add_accessor(&Input1);
        add_accessor(&Input2);
    }

    void kernel() {
        output() = Input1() * Input2();
    }
};

class Gaussian : public Kernel<short> {
  private:
    Accessor<short> &Input;
    Mask<data_t> &cMask;
    short norm;

  public:
    Gaussian(IterationSpace<short> &IS, Accessor<short> &Input,
             Mask<data_t> &cMask, short norm)
          : Kernel(IS), Input(Input), cMask(cMask), norm(norm) {
        add_accessor(&Input);
    }

    void kernel() {
        int sum = 0;
        sum += convolve(cMask, Reduce::SUM, [&] () -> int {
                return Input(cMask) * cMask();
            });
        output() = sum / norm;
    }
};

class HarrisCorner : public Kernel<uchar> {
  private:
    Accessor<short> &Dx;
    Accessor<short> &Dy;
    Accessor<short> &Dxy;
    float k;
    float threshold;

  public:
    HarrisCorner(IterationSpace<uchar> &IS, Accessor<short> &Dx,
                 Accessor<short> &Dy, Accessor<short> &Dxy,
                 float k, float threshold)
          : Kernel(IS), Dx(Dx), Dy(Dy), Dxy(Dxy), k(k), threshold(threshold) {
        add_accessor(&Dx);
        add_accessor(&Dy);
        add_accessor(&Dxy);
    }

    void kernel() {
        int x = Dx();
        int y = Dy();
        int xy = Dxy();
        float R = 0;
        R = ((x * y) - (xy * xy))      /* det   */
            - (k * (x + y) * (x + y)); /* trace */
        uchar out = 0;
        if (R > threshold)
            out = 1;
        output() = out;
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
    float k = 0.04f;
    float threshold = 20000.0f;

    if (argc > 2) {
      k = atof(argv[1]);
      threshold = atof(argv[2]);
    }

    // convolution filter mask
#if SIZE_X == 3
    const short norm = 16;
    const data_t coef_xy[SIZE_Y][SIZE_X] = {
        { 1, 2, 1 },
        { 2, 4, 2 },
        { 1, 2, 1 }
    };
#endif
#if SIZE_X == 5
    const short norm = 256;
    const data_t coef_xy[SIZE_Y][SIZE_X] = {
        {  1,  4,  6,  4,  1 },
        {  4, 16, 24, 16,  2 },
        {  6, 24, 36, 24,  1 },
        {  4, 16, 24, 16,  2 },
        {  1,  4,  6,  4,  1 }
    };
#endif
#if SIZE_X == 7
    const short norm = 4096;
    const data_t coef_xy[SIZE_Y][SIZE_X] = {
        {   1,   6,  15,  20,  15,   6,   1 },
        {   6,  36,  90, 120,  90,  36,   6 },
        {  15,  90, 225, 300, 225,  90,  15 },
        {  20, 120, 300, 400, 300, 120,  20 },
        {  15,  90, 225, 300, 225,  90,  15 },
        {   6,  36,  90, 120,  90,  36,   6 },
        {   1,   6,  15,  20,  15,   6,   1 }
    };
#endif

    const char coef_x[3][3] = {
        {-1,  0,  1},
        {-1,  0,  1},
        {-1,  0,  1}
    };

    const char coef_y[3][3] = {
        {-1, -1, -1},
        { 0,  0,  0},
        { 1,  1,  1}
    };

    // host memory for image of width x height pixels
    uchar *input = load_data<uchar>(width, height, 1, IMAGE);
    uchar *output = load_data<uchar>(width, height, 1, IMAGE);

    std::cout << "Calculating Hipacc Harris corner ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in(width, height, input);
    Image<uchar> out(width, height);
    Image<short> dx(width, height);
    Image<short> dy(width, height);
    Image<short> dxy(width, height);
    Image<short> sx(width, height);
    Image<short> sy(width, height);
    Image<short> sxy(width, height);

    // define Masks for Gaussian blur and Sobel filter
    Mask<data_t> maskxy(coef_xy);
    Mask<char> maskx(coef_x);
    Mask<char> masky(coef_y);

    // define Domains for Sobel kernel
    Domain domx(maskx);
    Domain domy(masky);

    IterationSpace<short> iter_dx(dx);
    BoundaryCondition<uchar> bound_in(in, maskx, Boundary::CLAMP);
    Accessor<uchar> acc_in(bound_in);
    Sobel derivx(iter_dx, acc_in, maskx, domx);
    derivx.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<short> iter_dy(dy);
    Sobel derivy(iter_dy, acc_in, masky, domy);
    derivy.execute();
    timing += hipacc_last_kernel_timing();

    Accessor<short> acc_dx(dx);
    IterationSpace<short> iter_sx(sx);
    Square1 squarex(iter_sx, acc_dx);
    squarex.execute();
    timing += hipacc_last_kernel_timing();

    Accessor<short> acc_dy(dy);
    IterationSpace<short> iter_sy(sy);
    Square1 squarey(iter_sy, acc_dy);
    squarey.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<short> iter_sxy(sxy);
    Square2 squarexy(iter_sxy, acc_dx, acc_dy);
    squarexy.execute();
    timing += hipacc_last_kernel_timing();

    BoundaryCondition<short> bound_sx(sx, maskxy, Boundary::CLAMP);
    Accessor<short> acc_sx(bound_sx);
    Gaussian gaussx(iter_dx, acc_sx, maskxy, norm);
    gaussx.execute();
    timing += hipacc_last_kernel_timing();

    BoundaryCondition<short> bound_sy(sy, maskxy, Boundary::CLAMP);
    Accessor<short> acc_sy(bound_sy);
    Gaussian gaussy(iter_dy, acc_sy, maskxy, norm);
    gaussy.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<short> iter_dxy(dxy);
    BoundaryCondition<short> bound_sxy(sxy, maskxy, Boundary::CLAMP);
    Accessor<short> acc_sxy(bound_sxy);
    Gaussian gaussxy(iter_dxy, acc_sxy, maskxy, norm);
    gaussxy.execute();
    timing += hipacc_last_kernel_timing();


    IterationSpace<uchar> iter_out(out);
    Accessor<short> acc_dxy(dxy);
    HarrisCorner harris(iter_out, acc_dx, acc_dy, acc_dxy, k, threshold);
    harris.execute();
    timing += hipacc_last_kernel_timing();

    uchar *corners = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    // draw white crosses for visualization
    const int size = 5;
    for (int p = 0; p < width*height; ++p) {
        if (corners[p] != 0) {
            for (int i = -size; i <= size; ++i) {
                int posx = p+i;
                int posy = p+i*width;
                if (posx > 0 && posx < width*height) output[posx] = 255;
                if (posy > 0 && posy < width*height) output[posy] = 255;
            }
        }
    }

    save_data(width, height, 1, input, "input.jpg");
    save_data(width, height, 1, output, "output.jpg");
    show_data(width, height, 1, output, "output.jpg");

    // free memory
    delete[] input;
    delete[] output;

    return EXIT_SUCCESS;
}
