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

using namespace hipacc;


#ifdef PACK_INT
# define data_t uint
# define pack(x, y, z, w) \
    (uint)((uint)(x) << 24 | (uint)(y) << 16 | (uint)(z) << 8 | (uint)(w))
# define unpack(val, byte) \
    (((val) >> ((3-(byte))*8)) & 0xff)
#else
# define data_t uchar4
# define pack(x, y, z, w) \
    ((uchar4){x, y, z, w})
# define unpack(val, byte) \
    (val.s ## byte)
#endif

#define BILIN

class Chrome : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;
        int width;
        int height;
        float rindex;
        float gindex;

    public:
        Chrome(IterationSpace<data_t> &iter, Accessor<data_t> &input,
               int width, int height, float rindex, float gindex) :
            Kernel(iter),
            input(input),
            width(width),
            height(height),
            rindex(rindex),
            gindex(gindex)
        { add_accessor(&input); }

        void kernel() {
            int gposx = x();
            int gposy = y();
            float xpos = gposx - width/2;
            float ypos = gposy - height/2;

            float rout = 0;
            float gout = 0;
            float bout = 0;
            
            { /* red */
              float xshift = (xpos * rindex) + width/2;
              float yshift = (ypos * rindex) + height/2;

              int bx = (int)xshift-gposx;
              int by = (int)yshift-gposy;
#ifdef BILIN
              // bilinear filtering
              float xp = xshift-(uint)xshift;
              float yp = yshift-(uint)yshift;

              float val1 = unpack(input(bx  , by  ), 0);
              float val2 = unpack(input(bx+1, by  ), 0);
              float val3 = unpack(input(bx  , by+1), 0);
              float val4 = unpack(input(bx+1, by+1), 0);

              rout = ((1.0f-yp) * (((1.0f-xp) * val1) + (xp * val2)))
                     +      (yp * (((1.0f-xp) * val3) + (xp * val4)));
#else

              rout = unpack(input(bx, by), 0);
#endif
            }

            { /* green */
              float xshift = (xpos * gindex) + width/2;
              float yshift = (ypos * gindex) + height/2;

              int bx = (int)xshift-gposx;
              int by = (int)yshift-gposy;

#ifdef BILIN
              // bilinear filtering
              float xp = xshift-(uint)xshift;
              float yp = yshift-(uint)yshift;

              float val1 = unpack(input(bx  , by  ), 1);
              float val2 = unpack(input(bx+1, by  ), 1);
              float val3 = unpack(input(bx  , by+1), 1);
              float val4 = unpack(input(bx+1, by+1), 1);

              gout = ((1.0f-yp) * (((1.0f-xp) * val1) + (xp * val2)))
                     +      (yp * (((1.0f-xp) * val3) + (xp * val4)));
#else
              gout = unpack(input(bx, by), 1);
#endif
            }

            { /* blue */
              bout = unpack(input(), 2);
            }

            output() = pack(rout, gout, bout, 255);
        }
};

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    float rindex = 0.9975f;
    float gindex = 0.995f;

    if (argc > 2) {
      rindex = atof(argv[1]);
      gindex = atof(argv[2]);
    }

    // load input
    FIBITMAP* img = FreeImage_Load(FIF_JPEG, "tests/postproc/chrome/house.jpg");
    img = FreeImage_ConvertTo32Bits(img);
    const int width = 1920;//FreeImage_GetWidth(img);
    const int height = 1200;//FreeImage_GetHeight(img);
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

    Image<data_t> IN(width, height, input);
    Image<data_t> OUT(width, height);

    IterationSpace<data_t> IS(OUT);

    Accessor<data_t> AccAt(IN);

    Chrome CHROME(IS, AccAt, width, height, rindex, gindex);
    CHROME.execute();
    float timing = hipacc_last_kernel_timing();

    d = OUT.data(); 

    for(size_t y = 0; y < height; ++y) { 
        for(size_t x = 0; x < width; ++x) { 
            RGBQUAD color;  
            color.rgbRed = unpack(d[x+y*width], 0); 
            color.rgbGreen = unpack(d[x+y*width], 1); 
            color.rgbBlue = unpack(d[x+y*width], 2); 
            color.rgbReserved = 255; 
            FreeImage_SetPixelColor(img, x, y, &color); 
        } 
    } 
    FreeImage_Save(FIF_PNG, img, "CHROME.png");
    //*************************************************************************//

    FreeImage_Unload(img);
    delete [] input;

    fprintf(stdout,"<HIPACC:> Overall time: %f(ms)\n", timing);

    return EXIT_SUCCESS;
}

