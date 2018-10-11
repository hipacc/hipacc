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


#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"

#define FAST_EXP
#define PACK_INT

#ifdef PACK_INT
# define data_t uint
# define pack(a, b, c, d) \
    (uint)((uint)(a) | (uint)(b) << 8 | (uint)(c) << 16 | (uint)(d) << 24)
# define unpack(a, b, c, val) \
    a = val & 0xff; \
    val >>= 8; \
    b = val & 0xff; \
    val >>= 8; \
    c = val & 0xff;
#else
# define data_t uchar4
# define pack(a, b, c, d) \
    ((uchar4){(uchar)a, (uchar)b, (uchar)c, (uchar)d})
# define unpack(a, b, c, val) \
    a = val.x; \
    b = val.y; \
    c = val.z;
#endif

#define EXPF256(out, in) \
  float _x = 1.0f + in / 256.0f; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  _x *= _x; \
  out = _x


using namespace hipacc;


class Atrous : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;
        Domain &dom;
        Mask<float> &mask;

    public:
        Atrous(IterationSpace<data_t> &iter, Accessor<data_t> &input,
               Domain &dom, Mask<float> &mask)
              : Kernel(iter), input(input), dom(dom), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            data_t in = input();
            float rin, gin, bin;
            unpack(rin, gin, bin, in);
            rin /= 255.0f;
            gin /= 255.0f;
            bin /= 255.0f;
            float sum_weight = 0.0f;
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            
            iterate(dom, [&] () {
                    data_t pixel = input(dom);
                    float rpixel, gpixel, bpixel;
                    unpack(rpixel, gpixel, bpixel, pixel);
                    rpixel /= 255.0f;
                    gpixel /= 255.0f;
                    bpixel /= 255.0f;
                    float rd = rpixel - rin;
                    float gd = gpixel - gin;
                    float bd = bpixel - bin;
                    float weight = rd*rd + gd*gd + bd*bd;
#ifdef FAST_EXP
                    EXPF256(weight, -(weight));// * 1.0f));
#else
                    weight = expf(-(weight));// * 1.0f));
#endif
                    if (weight > 1.0f) {
                      weight = 1.0f;
                    }
                    weight *= mask(dom);
                    sum_weight += weight;
                    sum_r += rpixel * weight;
                    sum_g += gpixel * weight;
                    sum_b += bpixel * weight;
                });

            float rout = sum_r * 255.0f / sum_weight;
            float gout = sum_g * 255.0f / sum_weight;
            float bout = sum_b * 255.0f / sum_weight;
            output() = pack(bout, gout, rout, 255);
        }
};

class Scoto : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;

    public:
        Scoto(IterationSpace<data_t> &iter, Accessor<data_t> &input) :
            Kernel(iter),
            input(input)
        { add_accessor(&input); }

        void kernel() {
            data_t in = input();
            float rin, gin, bin;
            unpack(bin, gin, rin, in);
            float X =  0.5149f * rin + 0.3244f * gin + 0.1607f * bin;
            float Y = (0.2654f * rin + 0.6704f * gin + 0.0642f * bin) / 3.0f;
            float Z =  0.0248f * rin + 0.1248f * gin + 0.8504f * bin;
            float V = Y * (((((Y + Z) / X) + 1.0f) * 1.33f) - 1.68f);
            float W = X + Y + Z;
            float luma = 0.2126f * rin +  0.7152f * gin + 0.0722f * bin;
            float s = 0.0f;// luma / 2.0f;
            float x1 = X / W;
            float y1 = Y / W;
            const float xb = 0.25f;
            const float yb = 0.25f;
            
            x1 = ((1.0f - s) * xb) + (s * x1);
            y1 = ((1.0f - s) * yb) + (s * y1);
            Y = (V * 0.4468f * (1.0f - s)) + (s * Y);
            X = (x1 * Y) / y1;
            Z = (X / y1) - X - Y;

            float r =  2.562263f * X + -1.166107f * Y + -0.396157f * Z;
            float g = -1.021558f * X +  1.977828f * Y +  0.043730f * Z;
            float b =  0.075196f * X + -0.256248f * Y +  1.181053f * Z;

            if (r > 255.0f) r = 255.0f;
            else if (r < 0.0f) r = 0.0f;
            if (g > 255.0f) g = 255.0f;
            else if (g < 0.0f) g = 0.0f;
            if (b > 255.0f) b = 255.0f;
            else if (b < 0.0f) b = 0.0f;

            output() = pack(b, g, r, 255);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    float timing = 0.0f;

    // define filters
    const float coef0[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };
    const float coef1[5][5] = {
        { 0.057118f,      0.0f, 0.124758f,      0.0f, 0.057118f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f },
        { 0.124758f,      0.0f, 0.272496f,      0.0f, 0.124758f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f },
        { 0.057118f,      0.0f, 0.124758f,      0.0f, 0.057118f }
    };
    const float coef2[9][9] = {
        { 0.057118f,      0.0f,      0.0f,      0.0f, 0.124758f,     0.0f,      0.0f,       0.0f, 0.057118f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        { 0.124758f,      0.0f,      0.0f,      0.0f, 0.272496f,     0.0f,      0.0f,       0.0f, 0.124758f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        { 0.057118f,      0.0f,      0.0f,      0.0f, 0.124758f,     0.0f,      0.0f,       0.0f, 0.057118f }
    };
    const float coef3[17][17] = {
        { 0.057118f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f, 0.124758f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f, 0.057118f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        { 0.124758f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f, 0.272496f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f, 0.124758f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f,      0.0f },
        { 0.057118f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f, 0.124758f,      0.0f,      0.0f,      0.0f,      0.0f,     0.0f,      0.0f,       0.0f, 0.057118f }
    };

    // load input
    const int width = WIDTH;
    const int height = HEIGHT;
    data_t* input = (data_t*)load_data<uchar>(width, height, 4, IMAGE);

    //************************************************************************//

    Mask<float> mask0(coef0);
    Mask<float> mask1(coef1);
    Mask<float> mask2(coef2);
    Mask<float> mask3(coef3);

    Domain dom0(mask0);
    Domain dom1(mask1);
    Domain dom2(mask2);
    Domain dom3(mask3);

    Image<data_t> in(width, height, input);
    Image<data_t> at0(width, height);
    Image<data_t> at1(width, height);

    IterationSpace<data_t> iter_atrous0(at0);
    IterationSpace<data_t> iter_atrous1(at1);

    BoundaryCondition<data_t> BcAtClamp0(in, mask0, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp1(at0, mask1, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp2(at1, mask2, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp3(at0, mask3, Boundary::CLAMP);

    Accessor<data_t> AccAtClamp0(BcAtClamp0);
    Accessor<data_t> AccAtClamp1(BcAtClamp1);
    Accessor<data_t> AccAtClamp2(BcAtClamp2);
    Accessor<data_t> AccAtClamp3(BcAtClamp3);
    Accessor<data_t> AccSc(at1);

    Atrous atrous0(iter_atrous0, AccAtClamp0, dom0, mask0);
    atrous0.execute();
    timing += hipacc_last_kernel_timing();

    Atrous atrous1(iter_atrous1, AccAtClamp1, dom1, mask1);
    atrous1.execute();
    timing += hipacc_last_kernel_timing();

    Atrous atrous2(iter_atrous0, AccAtClamp2, dom2, mask2);
    atrous2.execute();
    timing += hipacc_last_kernel_timing();

    Atrous atrous3(iter_atrous1, AccAtClamp3, dom3, mask3);
    atrous3.execute();
    timing += hipacc_last_kernel_timing();

    Scoto scoto(iter_atrous0, AccSc);
    scoto.execute();
    timing += hipacc_last_kernel_timing();

    data_t *output = at0.data(); 

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    save_data(width, height, 4, (uchar*)input, "input.jpg");
    save_data(width, height, 4, (uchar*)output, "output.jpg");
    show_data(width, height, 4, (uchar*)output, "output.jpg");

    delete [] input;

    return EXIT_SUCCESS;
}

