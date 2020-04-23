#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <sys/time.h>
#include <FreeImage.h>
#include "hipacc.hpp"

// variables set by Makefile
//#define SIZE_X 5
//#define SIZE_Y 5
//#define WIDTH 4096
//#define HEIGHT 4096
#define PACK_INT
#define FAST_EXP

using namespace hipacc;

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
    ((uchar4){x, y, z, w})
# define unpack(r, g, b, val) \
    r = val.s0; \
    g = val.s1; \
    b = val.s2
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
    float timing = 0.0f;

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
    const float atrous2[9][9] = {
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
    const float atrous3[17][17] = {
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
    FIBITMAP* img = FreeImage_Load(FIF_JPEG, "tests/postproc/night/mountain.jpg");
    img = FreeImage_ConvertTo32Bits(img);
    const int width = FreeImage_GetWidth(img);
    const int height = FreeImage_GetHeight(img);
    data_t* d;
    
    data_t* input = new data_t[width*height]; 
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < width; ++x) {
            RGBQUAD color; 
            FreeImage_GetPixelColor(img, x, y, &color);
            input[x+y*width] = pack(color.rgbRed, color.rgbGreen, color.rgbBlue, 255);
        }           
    }

    //*************************************************************************//

    Mask<float> MASK0(atrous0);
    Mask<float> MASK1(atrous1);
    Mask<float> MASK2(atrous2);
    Mask<float> MASK3(atrous3);

    Domain DOM0(MASK0);
    Domain DOM1(MASK1);
    Domain DOM2(MASK2);
    Domain DOM3(MASK3);

    Image<data_t> IN(width, height, input);
    Image<data_t> ATROUS0(width, height);
    Image<data_t> ATROUS1(width, height);

    IterationSpace<data_t> IS_ATROUS0(ATROUS0);
    IterationSpace<data_t> IS_ATROUS1(ATROUS1);

    BoundaryCondition<data_t> BcAtClamp0(IN, MASK0, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp1(ATROUS0, MASK1, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp2(ATROUS1, MASK2, Boundary::CLAMP);
    BoundaryCondition<data_t> BcAtClamp3(ATROUS0, MASK3, Boundary::CLAMP);

    Accessor<data_t> AccAtClamp0(BcAtClamp0);
    Accessor<data_t> AccAtClamp1(BcAtClamp1);
    Accessor<data_t> AccAtClamp2(BcAtClamp2);
    Accessor<data_t> AccAtClamp3(BcAtClamp3);
    Accessor<data_t> AccSc(ATROUS1);

    Atrous Atrous0(IS_ATROUS0, AccAtClamp0, DOM0, MASK0);
    Atrous0.execute();
    timing += hipacc_last_kernel_timing();

    Atrous Atrous1(IS_ATROUS1, AccAtClamp1, DOM1, MASK1);
    Atrous1.execute();
    timing += hipacc_last_kernel_timing();

    Atrous Atrous2(IS_ATROUS0, AccAtClamp2, DOM2, MASK2);
    Atrous2.execute();
    timing += hipacc_last_kernel_timing();

    Atrous Atrous3(IS_ATROUS1, AccAtClamp3, DOM3, MASK3);
    Atrous3.execute();
    timing += hipacc_last_kernel_timing();

    Scoto SCOTO(IS_ATROUS0, AccSc);
    SCOTO.execute();
    timing += hipacc_last_kernel_timing();

    d = ATROUS0.data(); 

    for(size_t y = 0; y < height; ++y) { 
        for(size_t x = 0; x < width; ++x) { 
            RGBQUAD color;  
            unpack(color.rgbRed, color.rgbGreen, color.rgbBlue, d[x+y*width]);
            color.rgbReserved = 255; 
            FreeImage_SetPixelColor(img, x, y, &color); 
        } 
    } 
    FreeImage_Save(FIF_PNG, img, "NIGHT.png");
    //*************************************************************************//

    FreeImage_Unload(img);
    delete [] input;

    fprintf(stdout,"<HIPACC:> Overall time: %f(ms)\n", timing);

    return EXIT_SUCCESS;
}

