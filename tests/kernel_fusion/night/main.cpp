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

//#define PRINT_RESULT
#define FAST_EXP
//#ifdef PACK_INT

#ifdef PACK_INT
# define data_t uint
# define pack(x, y, z, w) \
    (uint)((uint)(x) << 24 | (uint)(y) << 16 | (uint)(z) << 8 | (uint)(w))
# define unpack(r, g, b, val) \
    val >>= 8; \
    b = val & 0xff; \
    val >>= 8; \
    g = val & 0xff; \
    val >>= 8; \
    r = val & 0xff
#else
# define data_t uchar4
# define pack(x, y, z, w) \
    ((uchar4){(uchar)(x), (uchar)(y), (uchar)(z), (uchar)(w)})
# define unpack(r, g, b, val) \
    r = val.x; \
    g = val.y; \
    b = val.z
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


// kernel description in Hipacc 
class Atrous : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;
        Domain &dom;
        Mask<float> &mask;

    public:
        Atrous(IterationSpace<data_t> &iter, Accessor<data_t> &input,
               Domain &dom, Mask<float> &mask) :
            Kernel(iter),
            input(input),
            dom(dom),
            mask(mask)
        { add_accessor(&input); }

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
            output() = pack(rout, gout, bout, 255);
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
            unpack(rin, gin, bin, in);
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

            output() = pack(r, g, b, 255);
        }
};

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    // define filters
    const float atrous0[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };
    const float atrous1[5][5] = {
        { 0.057118f,      0.0f, 0.124758f,      0.0f, 0.057118f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f },
        { 0.124758f,      0.0f, 0.272496f,      0.0f, 0.124758f },
        {      0.0f,      0.0f,      0.0f,      0.0f,      0.0f },
        { 0.057118f,      0.0f, 0.124758f,      0.0f, 0.057118f }
    };

    // load input
    const int width = 1920;//FreeImage_GetWidth(img);
    const int height = 1200;//FreeImage_GetHeight(img);
    
    data_t* input = new data_t[width*height]; 
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < width; ++x) {
            input[x+y*width] = pack(x%5, y%15, (x+y)%25, 255);
        }           
    }

    // hipacc object decls
    Mask<float> MASK0(atrous0);
    Mask<float> MASK1(atrous1);

    Domain DOM0(MASK0);
    Domain DOM1(MASK1);

    Image<data_t> IN(width, height);
    Image<data_t> ATROUS0(width, height);
    Image<data_t> ATROUS1(width, height);
    Image<data_t> OUT(width, height);

    IterationSpace<data_t> IS_ATROUS0(ATROUS0);
    IterationSpace<data_t> IS_ATROUS1(ATROUS1);
    IterationSpace<data_t> IS_OUT(OUT);

    const BoundaryCondition<data_t> BcAtClamp0(IN, MASK0, Boundary::CLAMP);
    const BoundaryCondition<data_t> BcAtClamp1(ATROUS0, MASK1, Boundary::CLAMP);

    Accessor<data_t> AccAtClamp0(BcAtClamp0);
    Accessor<data_t> AccAtClamp1(BcAtClamp1);
    Accessor<data_t> AccSc(ATROUS1);

    Atrous Atrous0(IS_ATROUS0, AccAtClamp0, DOM0, MASK0);
    Atrous Atrous1(IS_ATROUS1, AccAtClamp1, DOM1, MASK1);
    Scoto SCOTO(IS_OUT, AccSc);

    // kernel invocation
		IN = input;
    Atrous0.execute();
    Atrous1.execute();
    SCOTO.execute();
    data_t *output = OUT.data(); 

#ifdef PRINT_RESULT 
    std::cerr << "\nwriting output to file...\n";
    int ySkip = 17;
    int xSkip = 13;
    // compute the golden results
    for (int y=0; y<height; y=y+ySkip) {
      for (int x=0; x<width; x=x+xSkip) {
        std::cout << "output at y " << y << ", x " << x << " : " << 
          (int)output[y*width + x].x << " " << (int)output[y*width + x].y <<
            " " << (int)output[y*width + x].z << "\n"; 
      }
    }
#endif


    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}

