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
#define IMAGE  "../../common/img/fuerte_night.jpg"

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


using namespace hipacc;


class Bokeh : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;
        Domain &dom;
        Mask<float> &mask;
        float threshold;
        float amp;

    public:
        Bokeh(IterationSpace<data_t> &iter, Accessor<data_t> &input,
              Domain &dom, Mask<float> &mask, float threshold, float amp)
            : Kernel(iter), input(input), dom(dom), mask(mask),
              threshold(threshold), amp(amp) {
            add_accessor(&input);
        }

        void kernel() {
            float sum_weight = 0.0f;
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            
            iterate(dom, [&] () {
                    data_t pixel = input(dom);
                    float rpixel, gpixel, bpixel;
                    unpack(bpixel, gpixel, rpixel, pixel);
                    rpixel /= 255.0f;
                    gpixel /= 255.0f;
                    bpixel /= 255.0f;
                    float luma = 0.2126f * rpixel +  0.7152f * gpixel + 0.0722f * bpixel;
                    float weight = mask(dom);
                    if (luma > threshold) weight *= amp; // amplify light pixels
                    sum_r += rpixel * weight;
                    sum_g += gpixel * weight;
                    sum_b += bpixel * weight;
                    sum_weight += weight;
                });

            float rout = sum_r * 255.f / sum_weight;
            float gout = sum_g * 255.f / sum_weight;
            float bout = sum_b * 255.f / sum_weight;
            output() = pack(bout, gout, rout, 255);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    float threshold = 0.9f;
    float amp = 150.0f;

    if (argc > 2) {
      threshold = atof(argv[1]);
      amp = atof(argv[2]);
    }

    // define filters
    const float stencil[31][31] = {
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 2.000f, 4.000f, 6.000f, 11.000f, 17.000f, 19.000f, 13.000f, 9.000f, 7.000f, 3.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 1.000f, 2.000f, 2.000f, 4.000f, 10.000f, 27.000f, 48.000f, 72.000f, 89.000f, 89.000f, 63.000f, 26.000f, 3.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 2.000f, 4.000f, 6.000f, 14.000f, 19.000f, 37.000f, 64.000f, 100.000f, 133.000f, 171.000f, 198.000f, 200.000f, 159.000f, 92.000f, 30.000f, 1.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 2.000f, 8.000f, 9.000f, 11.000f, 27.000f, 55.000f, 91.000f, 131.000f, 170.000f, 204.000f, 230.000f, 243.000f, 249.000f, 248.000f, 236.000f, 188.000f, 107.000f, 34.000f, 1.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 2.000f, 8.000f, 12.000f, 29.000f, 59.000f, 102.000f, 146.000f, 183.000f, 217.000f, 241.000f, 251.000f, 254.000f, 254.000f, 254.000f, 253.000f, 254.000f, 244.000f, 197.000f, 119.000f, 44.000f, 3.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 3.000f, 13.000f, 22.000f, 47.000f, 91.000f, 146.000f, 199.000f, 234.000f, 248.000f, 252.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 252.000f, 250.000f, 210.000f, 127.000f, 45.000f, 4.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 1.000f, 3.000f, 13.000f, 32.000f, 74.000f, 133.000f, 192.000f, 229.000f, 246.000f, 251.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 252.000f, 213.000f, 126.000f, 44.000f, 4.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 3.000f, 8.000f, 32.000f, 88.000f, 156.000f, 216.000f, 248.000f, 254.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 255.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 252.000f, 205.000f, 116.000f, 32.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 7.000f, 16.000f, 68.000f, 164.000f, 232.000f, 250.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 250.000f, 248.000f, 193.000f, 94.000f, 23.000f, 2.000f, 0.000f, 0.000f, 0.000f, },
{ 9.000f, 32.000f, 104.000f, 208.000f, 253.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 254.000f, 240.000f, 170.000f, 73.000f, 11.000f, 4.000f, 1.000f, 0.000f, },
{ 10.000f, 43.000f, 123.000f, 218.000f, 252.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 251.000f, 229.000f, 140.000f, 51.000f, 11.000f, 2.000f, 1.000f, },
{ 10.000f, 45.000f, 125.000f, 218.000f, 252.000f, 254.000f, 254.000f, 254.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 251.000f, 204.000f, 116.000f, 32.000f, 3.000f, 2.000f, },
{ 10.000f, 46.000f, 126.000f, 218.000f, 252.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 243.000f, 179.000f, 74.000f, 14.000f, 4.000f, },
{ 10.000f, 46.000f, 127.000f, 220.000f, 252.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 220.000f, 118.000f, 34.000f, 2.000f, },
{ 9.000f, 42.000f, 121.000f, 216.000f, 252.000f, 253.000f, 254.000f, 252.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 232.000f, 135.000f, 40.000f, 1.000f, },
{ 8.000f, 37.000f, 113.000f, 213.000f, 252.000f, 254.000f, 254.000f, 253.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 221.000f, 117.000f, 29.000f, 2.000f, },
{ 7.000f, 31.000f, 104.000f, 210.000f, 252.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 247.000f, 188.000f, 78.000f, 13.000f, 2.000f, },
{ 6.000f, 26.000f, 93.000f, 200.000f, 250.000f, 254.000f, 254.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 251.000f, 222.000f, 141.000f, 43.000f, 5.000f, 2.000f, },
{ 6.000f, 20.000f, 80.000f, 182.000f, 244.000f, 253.000f, 254.000f, 254.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 252.000f, 247.000f, 185.000f, 93.000f, 22.000f, 2.000f, 1.000f, },
{ 3.000f, 11.000f, 62.000f, 160.000f, 231.000f, 249.000f, 253.000f, 254.000f, 253.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 251.000f, 230.000f, 140.000f, 52.000f, 12.000f, 3.000f, 1.000f, },
{ 1.000f, 6.000f, 44.000f, 137.000f, 215.000f, 247.000f, 252.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 245.000f, 193.000f, 95.000f, 21.000f, 7.000f, 2.000f, 0.000f, },
{ 1.000f, 3.000f, 30.000f, 115.000f, 204.000f, 250.000f, 250.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 226.000f, 136.000f, 51.000f, 10.000f, 3.000f, 0.000f, 0.000f, },
{ 1.000f, 2.000f, 18.000f, 87.000f, 183.000f, 250.000f, 253.000f, 252.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 251.000f, 244.000f, 179.000f, 75.000f, 17.000f, 5.000f, 0.000f, 0.000f, 0.000f, },
{ 1.000f, 1.000f, 12.000f, 54.000f, 144.000f, 233.000f, 255.000f, 253.000f, 252.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 253.000f, 251.000f, 251.000f, 201.000f, 111.000f, 28.000f, 3.000f, 2.000f, 0.000f, 0.000f, 0.000f, },
{ 1.000f, 2.000f, 8.000f, 26.000f, 85.000f, 167.000f, 220.000f, 240.000f, 244.000f, 249.000f, 251.000f, 252.000f, 252.000f, 253.000f, 253.000f, 254.000f, 254.000f, 254.000f, 254.000f, 254.000f, 250.000f, 252.000f, 225.000f, 136.000f, 52.000f, 12.000f, 2.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 1.000f, 4.000f, 9.000f, 28.000f, 74.000f, 129.000f, 173.000f, 202.000f, 224.000f, 237.000f, 246.000f, 248.000f, 249.000f, 249.000f, 250.000f, 252.000f, 252.000f, 253.000f, 251.000f, 253.000f, 230.000f, 162.000f, 72.000f, 18.000f, 6.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 1.000f, 3.000f, 8.000f, 22.000f, 49.000f, 79.000f, 109.000f, 137.000f, 161.000f, 184.000f, 202.000f, 215.000f, 226.000f, 233.000f, 239.000f, 240.000f, 244.000f, 244.000f, 230.000f, 173.000f, 89.000f, 31.000f, 9.000f, 2.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 2.000f, 7.000f, 10.000f, 17.000f, 33.000f, 53.000f, 72.000f, 94.000f, 116.000f, 136.000f, 152.000f, 164.000f, 173.000f, 178.000f, 186.000f, 182.000f, 153.000f, 96.000f, 39.000f, 14.000f, 5.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 3.000f, 4.000f, 8.000f, 13.000f, 19.000f, 25.000f, 33.000f, 44.000f, 54.000f, 60.000f, 66.000f, 72.000f, 75.000f, 69.000f, 48.000f, 29.000f, 16.000f, 7.000f, 1.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 2.000f, 2.000f, 4.000f, 5.000f, 8.000f, 10.000f, 12.000f, 13.000f, 14.000f, 14.000f, 11.000f, 6.000f, 5.000f, 5.000f, 2.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
{ 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 2.000f, 5.000f, 6.000f, 6.000f, 6.000f, 5.000f, 4.000f, 3.000f, 3.000f, 3.000f, 2.000f, 1.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, },
    };

    // load input
    const int width = WIDTH;
    const int height = HEIGHT;
    data_t *input = (data_t*)load_data<uchar>(width, height, 4, IMAGE);

    //************************************************************************//

    Mask<float> mask(stencil);

    Domain dom(mask);

    Image<data_t> in(width, height, input);
    Image<data_t> out(width, height);

    IterationSpace<data_t> iter(out);

    BoundaryCondition<data_t> BcAtClamp(in, mask, Boundary::CLAMP);

    Accessor<data_t> AccAtClamp(BcAtClamp);

    Bokeh bokeh(iter, AccAtClamp, dom, mask, threshold, amp);
    bokeh.execute();
    float timing = hipacc_last_kernel_timing();

    data_t *output = out.data(); 

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    save_data(width, height, 4, (uchar*)input, "input.jpg");
    save_data(width, height, 4, (uchar*)output, "output.jpg");
    show_data(width, height, 4, (uchar*)output, "output.jpg");

    delete [] input;

    return EXIT_SUCCESS;
}

