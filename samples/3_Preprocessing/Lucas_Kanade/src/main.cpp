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

#define WINDOW_SIZE 15


using namespace hipacc;
using namespace hipacc::math;


class Derive : public Kernel<char> {
private:
    Accessor<uchar> &input;
    Mask<char> &cMask; 
    Domain &dom;
    
public:
    Derive(IterationSpace<char> &iter, Accessor<uchar> &input,
           Mask<char> &cMask, Domain &dom)
          : Kernel(iter), input(input), cMask(cMask), dom(dom) {
        add_accessor(&input);
    }

    void kernel(){
        char sum = reduce(dom, Reduce::SUM, [&] () -> char {
                return input(dom) * cMask(dom);
            });
        output() = (char)(sum >> 1);
    }
};

class Diff : public Kernel<char> {
private:
    Accessor<uchar> &input1;
    Accessor<uchar> &input2;
    
public:
    Diff(IterationSpace<char> &iter, Accessor<uchar> &input1,
         Accessor<uchar> &input2)
          : Kernel(iter), input1(input1), input2(input2) {
        add_accessor(&input1);
        add_accessor(&input2);
    }

    void kernel() {
        output() = input1() - input2();
    }
};

class Square : public Kernel<int> {
private:
    Accessor<char> &input;
    
public:
    Square(IterationSpace<int> &iter, Accessor<char> &input)
          : Kernel(iter), input(input) {
        add_accessor(&input);
    }

    void kernel(){
        output() = input() * input();
    }
};

class Mult : public Kernel<int> {
private:
    Accessor<char> &input1;
    Accessor<char> &input2;
    
public:
    Mult(IterationSpace<int> &iter, Accessor<char> &input1,
         Accessor<char> &input2)
          : Kernel(iter), input1(input1), input2(input2) {
        add_accessor(&input1);
        add_accessor(&input2);
    }

    void kernel(){
        output() = input1() * input2();
    }
};

class LKanade : public Kernel<int> {
private:
    Accessor<int> &Ixx; 
    Accessor<int> &Iyy; 
    Accessor<int> &Ixy; 
    Accessor<int> &DelIx;
    Accessor<int> &DelIy;
    Domain &dom;
    
public:
    LKanade(IterationSpace<int> &iter, Accessor<int> &Ixx, 
            Accessor<int> &Iyy, Accessor<int> &Ixy, Accessor<int> &DelIx,
            Accessor<int> &DelIy, Domain &dom)
          : Kernel(iter), Ixx(Ixx), Iyy(Iyy), Ixy(Ixy), DelIx(DelIx),
            DelIy(DelIy), dom(dom) {
        add_accessor(&Ixx); 
        add_accessor(&Iyy); 
        add_accessor(&Ixy); 
        add_accessor(&DelIx); 
        add_accessor(&DelIy);
    }

    // Flow vector scaling factor
    #define FLOW_SCALING_FACTOR (1.0f/4.0f)
    void kernel(){
        int b_k0 = reduce(dom, Reduce::SUM, [&] () -> int {
            return DelIx(dom);
        });

        int b_k1 = reduce(dom, Reduce::SUM, [&] () -> int {
            return DelIy(dom);
        });

        int G0 = reduce(dom, Reduce::SUM, [&] () -> int {
            return Ixx(dom);
        });

        int G3 = reduce(dom, Reduce::SUM, [&] () -> int {
            return Iyy(dom);
        });

        int IXY = reduce(dom, Reduce::SUM, [&] () -> int {
            return Ixy(dom);
        });
        int G1 = IXY;
        int G2 = IXY;

        int vector = 0;

        //get_matrix_inv (G, G_inv);
        float detG = (float)G0 * G3 - (float)G1 * G2;
        if (detG > 1.0f) {
            float detG_inv = 1.0f / detG;
            float G_inv0 =  G3 * detG_inv;
            float G_inv1 = -G1 * detG_inv;
            float G_inv2 = -G2 * detG_inv;
            float G_inv3 =  G0 * detG_inv;

            float fx = G_inv0 * b_k0 + G_inv1 * b_k1;
            float fy = G_inv2 * b_k0 + G_inv3 * b_k1;
            fx = fx * FLOW_SCALING_FACTOR;
            fy = fy * FLOW_SCALING_FACTOR;

            // correct rounding
            fx += fx < 0 ? -.5f : .5f;
            fy += fy < 0 ? -.5f : .5f;

            vector = (int)((int)fx << 16) | ((int)fy & 0xffff);
        }

        output() = vector;
    }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    float timing = 0;

    const char coef_x[3][3] = {
        { 0, 0, 0},
        {-1, 0, 1},
        { 0, 0, 0}
    };

    const char coef_y[3][3] = {
        { 0,-1, 0},
        { 0, 0, 0},
        { 0, 1, 0}
    };

    // host memory for image of width x height pixels
    uchar *input1 = load_data<uchar>(width, height, 1, IMAGE1);
    uchar *input2 = load_data<uchar>(width, height, 1, IMAGE2);

    std::cout << "Calculating Hipacc Lucas Kanade ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in1(width, height, input1);
    Image<uchar> in2(width, height, input2);
    Image<char> dx(width, height);
    Image<char> dy(width, height);
    Image<char> delta(width, height);
    Image<int> sx(width, height);
    Image<int> sy(width, height);
    Image<int> sxy(width, height);
    Image<int> mx(width, height);
    Image<int> my(width, height);
    Image<int> out(width, height);

    Mask<char> maskx(coef_x);
    Mask<char> masky(coef_y);

    Domain domx(maskx);
    Domain domy(masky);
    Domain domxy(WINDOW_SIZE, WINDOW_SIZE);

    // Derive
    IterationSpace<char> iter_dx(dx);
    BoundaryCondition<uchar> bound_in1_x(in1, domx, Boundary::CLAMP);
    Accessor<uchar> acc_in1_x(bound_in1_x);
    Derive derivx(iter_dx, acc_in1_x, maskx, domx);
    derivx.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<char> iter_dy(dy);
    BoundaryCondition<uchar> bound_in2_y(in1, domy, Boundary::CLAMP);
    Accessor<uchar> acc_in1_y(bound_in2_y);
    Derive derivy(iter_dy, acc_in1_y, masky, domy);
    derivy.execute();
    timing += hipacc_last_kernel_timing();

    // Square
    IterationSpace<int> iter_sx(sx);
    Accessor<char> acc_dx(dx);
    Square squarex(iter_sx, acc_dx);
    squarex.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<int> iter_sy(sy);
    Accessor<char>  acc_dy(dy);
    Square squarey(iter_sy, acc_dy);
    squarey.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<char> iter_delta(delta);
    Accessor<uchar> acc_in1(in1);
    Accessor<uchar> acc_in2(in2);
    Diff diff(iter_delta, acc_in1, acc_in2);
    diff.execute();
    timing += hipacc_last_kernel_timing();

    // Mult
    IterationSpace<int> iter_sxy(sxy);
    Mult mulxy(iter_sxy, acc_dx, acc_dy);
    mulxy.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<int> iter_mx(mx);
    Accessor<char> acc_delta(delta);
    Mult mulx(iter_mx, acc_delta, acc_dx);
    mulx.execute();
    timing += hipacc_last_kernel_timing();

    IterationSpace<int> iter_my(my);
    Mult muly(iter_my, acc_delta, acc_dy);
    muly.execute();
    timing += hipacc_last_kernel_timing();

    // LKanade
    IterationSpace<int> iter_out(out);
    BoundaryCondition<int> bound_sx(sx, domxy, Boundary::CLAMP);
    BoundaryCondition<int> bound_sy(sy, domxy, Boundary::CLAMP);
    BoundaryCondition<int> bound_sxy(sxy, domxy, Boundary::CLAMP);
    BoundaryCondition<int> bound_mx(mx, domxy, Boundary::CLAMP);
    BoundaryCondition<int> bound_my(my, domxy, Boundary::CLAMP);
    Accessor<int> acc_sx(bound_sx);
    Accessor<int> acc_sy(bound_sy);
    Accessor<int> acc_sxy(bound_sxy);
    Accessor<int> acc_mx(bound_mx);
    Accessor<int> acc_my(bound_my);
    LKanade lkanade(iter_out, acc_sx, acc_sy, acc_sxy, acc_mx, acc_my, domxy);
    lkanade.execute();
    timing += hipacc_last_kernel_timing();

    int *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    // draw motion vectors for visualization
    for (int p = 0; p < width*height; ++p) {
        int vector = output[p];
        if (vector != 0) {
            float xf = vector >> 16;
            float yf = (short)(vector & 0xffff);
            float m = yf/xf;
            for (int i = 0; i <= abs(xf); ++i) {
                int xi = (xf < 0 ? -i : i);
                int yi = m*xi + (m*xi < 0 ? -.5f : .5f);
                int pos = p+yi*width+xi;
                if (pos > 0 && pos < width*height) {
                    input2[pos] = 255;
                }
            }
        }
    }

    save_data(width, height, 1, input1, "input.jpg");
    save_data(width, height, 1, input2, "output.jpg");
    show_data(width, height, 1, input2, "output.jpg");

    // free memory
    delete[] input1;
    delete[] input2;

    return EXIT_SUCCESS;
}
