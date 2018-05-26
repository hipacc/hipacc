#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hipacc.hpp"

#define FREEIMAGE

#define WIDTH  1024
#define HEIGHT 1024

#define WINDOW_SIZE_X 30
#define WINDOW_SIZE_Y 30
#define EPSILON       30
#define TILE_SIZE     32
#define STRIDE_X      ((WIDTH+TILE_SIZE-1)/TILE_SIZE)
#define STRIDE_Y      ((HEIGHT+TILE_SIZE-1)/TILE_SIZE)

#ifdef FREEIMAGE
#  include <FreeImage.h>

FIBITMAP* loadGrayscaleImage(const char *filename) {
    FIBITMAP* image = FreeImage_Load(FIF_PNG, filename);
    FIBITMAP* imGrayscale;

    FREE_IMAGE_COLOR_TYPE type = FreeImage_GetColorType(image);

    switch(type) {
        case FIC_MINISBLACK:
            imGrayscale = image;
            break;
        case FIC_RGB:
        case FIC_RGBALPHA:
            imGrayscale = FreeImage_ConvertToGreyscale(image);
            FreeImage_Unload(image);
            break;
        default:
            std::cerr << "Error: Image type unsupported" << std::endl;
            exit(EXIT_FAILURE);
    }
    std::cout << "Loaded " << filename << "." << std::endl;

    if (FreeImage_GetWidth(imGrayscale) != WIDTH &&
        FreeImage_GetHeight(imGrayscale) != HEIGHT) {
        std::cerr << "Error: Image dimension mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }

    return imGrayscale;
}
#endif

using namespace hipacc;


class GaussianBlurFilter : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<float> &mask;

    public:
        GaussianBlurFilter(IterationSpace<uchar> &iter, Accessor<uchar> &input, Mask<float> &mask) :
            Kernel(iter), input(input), mask(mask) { add_accessor(&input); }

        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                return mask() * input(mask);
            }) + 0.5f);
        }
};

class SignatureKernel : public Kernel<uint> {
    private:
        Accessor<uchar> &input;
        Domain &dom;

    public:
        SignatureKernel(IterationSpace<uint> &iter, Accessor<uchar> &input, Domain &dom) :
            Kernel(iter), input(input), dom(dom) { add_accessor(&input); }

        void kernel() {
            // Census Transformation
            short z = input();
            uint c = 0u;
            iterate(dom, [&] () {
                short data = input(dom);
                if (data > z + EPSILON) {
                    c = (c << 2) | 0x01;
                } else if (data < z - EPSILON) {
                    c = (c << 2) | 0x02;
                } else {
                    c = c << 2;
                }
            });

            output() = c;
        }
};

class VectorKernel : public Kernel<int, float4> {
    private:
        Accessor<uint> &sig1, &sig2;
        Domain &dom;

    public:
        VectorKernel(IterationSpace<int> &iter, Accessor<uint> &sig1, Accessor<uint> &sig2, Domain &dom) :
            Kernel(iter), sig1(sig1), sig2(sig2), dom(dom) {
            add_accessor(&sig1);
            add_accessor(&sig2);
        }

        void kernel() {
            int vec_found = 0;
            int mem_loc = 0;

            uint reference = sig1();

            iterate(dom, [&] () -> void {
                if (sig2(dom) == reference) {
                    //vec_found++; // BUG: Hipacc doesn't recognize ++-operator as assignment
                    vec_found = vec_found + 1;
                    // encode ix and iy as upper and lower half-word of
                    // mem_loc
                    mem_loc = (dom.x() << 16) | (dom.y() & 0xffff);
                }
            });

            // save the vector, if exactly one was found
            if (vec_found!=1) {
                mem_loc = 0;
            }

            output() = mem_loc;
        }

        void binning(uint x, uint y, int vector) {
            float4 result = { 0.0f, 0.0f, 0.0f, 0.0f };
            uint ix = x / TILE_SIZE;
            uint iy = y / TILE_SIZE;

            if (vector != 0) {
                // Cartesian to polar
                float x = vector >> 16;
                int iy = (vector & 0xffff);
                if (iy >> 15) iy |= 0xffff0000;
                float y = (float)iy;
                float dist = sqrt(x*x+y*y)/2.0f;
                float angle = atan((float)y/x);

                result.x = dist;
                result.y = angle;
                result.w = 1.0f;
            }

            bin((uint)(iy * STRIDE_X + ix)) = result;
        }

        float4 reduce(float4 left, float4 right) const {
            if (left.w == 0.0f) {
                return right;
            } else if (right.w == 0.0f) {
                return left;
            } else {
                float ws = left.w + right.w;
                float wl = left.w/ws;
                float wr = 1.0f-wl;
                float4 result = wl*left + wr*right;
                result.w = ws;
                return result;
            }
        }
};


class Assemble : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Accessor<float4> &vecs;

    public:
        Assemble(IterationSpace<uchar> &iter, Accessor<uchar> &input, Accessor<float4> &vecs) :
            Kernel(iter), input(input), vecs(vecs) {
            add_accessor(&input);
            add_accessor(&vecs);
        }

        void kernel() {
            int x = 0;
            int y = 0;
            float4 vector = vecs();

            if (vector.w != 0.0f) {
                // polar to Cartesian
                float xf = vector.x * cosf(vector.y);
                float yf = vector.x * sinf(vector.y);

                // correct rounding
                x = (int)(xf + 0.5f - (xf < 0 ? 1.0f : 0.0f));
                y = (int)(yf + 0.5f - (yf < 0 ? 1.0f : 0.0f));
            }

            output() = input(x, y);
        }
};



int main(int argc, const char **argv) {
    float timing = 0, fps_timing = 0;
    const int width = WIDTH;
    const int height = HEIGHT;

#ifdef FREEIMAGE
    FreeImage_Initialise();
    FIBITMAP* fiin1 = loadGrayscaleImage("input1.png");
    FIBITMAP* fiin2 = loadGrayscaleImage("input2.png");
    uchar* host_in1 = FreeImage_GetBits(fiin1);
    uchar* host_in2 = FreeImage_GetBits(fiin2);
#else
    uchar *host_in1 = (uchar *) malloc(sizeof(uchar)*width*height);
    uchar *host_in2 = (uchar *) malloc(sizeof(uchar)*width*height);
#endif

    // images
    Image<uchar> prev(width, height);
    Image<uchar> img(width, height);
    Image<uchar> filter_img(width, height);
    Image<uint> prev_signature(width, height);
    Image<uint> img_signature(width, height);
    Image<int> img_vec(width, height);
    Image<float4> merged_vec(STRIDE_X, STRIDE_Y);

    size_t tile_size = TILE_SIZE;

    // filter mask
    const float filter_mask[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };
    Mask<float> mask(filter_mask);

    // domain for signature kernel
    const uchar sig_coef[9][9] = {
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 }
    };
    Domain sig_dom(sig_coef);

    // domain for vector kernel
    Domain dom(WINDOW_SIZE_X/2, WINDOW_SIZE_Y/2);
    // do not process the center pixel
    dom(0,0) = 0;


    BoundaryCondition<uint> bound_img_sig(img_signature, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_img_sig(bound_img_sig);
    //Accessor<uint> acc_img_sig(img_signature);
    BoundaryCondition<uint> bound_prev_sig(prev_signature, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_prev_sig(bound_prev_sig);
    //Accessor<uint> acc_prev_sig(prev_signature);
    IterationSpace<int> iter_vec(img_vec);
    VectorKernel vector_kernel(iter_vec, acc_prev_sig, acc_img_sig, dom);

    prev = host_in1;
    img = host_in2;

    // filter previous image/frame
    BoundaryCondition<uchar> bound_prev(prev, mask, Boundary::CLAMP);
    Accessor<uchar> acc_prev(bound_prev);
    IterationSpace<uchar> iter_blur(filter_img);
    GaussianBlurFilter blur_prev(iter_blur, acc_prev, mask);
    blur_prev.execute();


    // generate signature for first image/frame
    BoundaryCondition<uchar> bound_fil(filter_img, sig_dom, Boundary::CLAMP);
    Accessor<uchar> acc_fil(bound_fil);
    IterationSpace<uint> iter_prev_sig(prev_signature);
    SignatureKernel sig_prev(iter_prev_sig, acc_fil, sig_dom);
    sig_prev.execute();

    // filter frame
    BoundaryCondition<uchar> bound_img(img, mask, Boundary::CLAMP);
    Accessor<uchar> acc_img(bound_img);
    GaussianBlurFilter blur_img(iter_blur, acc_img, mask);

    blur_img.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing = timing;
    fprintf(stdout, "HIPAcc blur filter: %.3f ms\n", timing);

    // generate signature for frame
    IterationSpace<uint> iter_img_sig(img_signature);
    SignatureKernel sig_img(iter_img_sig, acc_fil, sig_dom);
    sig_img.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    fprintf(stdout, "HIPAcc signature kernel: %.3f ms\n", timing);

    // perform matching and merge vectors
    vector_kernel.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    fprintf(stdout, "HIPAcc vector kernel: %.3f ms\n", timing);

    // get merged vectors
    float4* vecs = vector_kernel.binned_data(STRIDE_X*STRIDE_Y);
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    fprintf(stdout, "HIPAcc vector merge: %.3f ms\n", timing);

    merged_vec = vecs;

    // assemble final image
    IterationSpace<uchar> iter_prev(prev);
    Accessor<float4> acc_merged_vec(merged_vec, Interpolate::NN);
    BoundaryCondition<uchar> bound_asm_img(img, dom, Boundary::CLAMP);
    Accessor<uchar> acc_asm_img(bound_asm_img);
    Assemble assemble(iter_prev, acc_img, acc_merged_vec);
    assemble.execute();
    timing = hipacc_last_kernel_timing();
    fps_timing += timing;
    fprintf(stdout, "HIPAcc assembling kernel: %.3f ms\n", timing);

    // fps time
    fprintf(stdout, "<HIPACC:> motion interpolation: %.3f ms, %f fps\n", fps_timing, 1000.0f/fps_timing);

    host_in2 = prev.data();

#ifdef DEBUG
    for (int y=0; y<STRIDE_Y; y++) {
        for (int x=0; x<STRIDE_X; x++) {
            int pos = x + y*(STRIDE_X);
            if (vecs[pos].w != 0) {
                // polar to Cartesian
                float xf = vecs[pos].x * cos(vecs[pos].y);
                float yf = vecs[pos].x * sin(vecs[pos].y);

                // correct rounding
                int high = (int)(xf + 0.5f - (xf < 0 ? 1.0f : 0.0f));
                int low = (int)(yf + 0.5f - (yf < 0 ? 1.0f : 0.0f));

                int ox = x*tile_size + tile_size/2;
                int oy = y*tile_size + tile_size/2;
                fprintf(stdout, "(%d, %d) ---> (%d, %d)\n", ox + high, oy + low, ox, oy);
            }
        }
    }
#endif

#ifdef FREEIMAGE
    FIBITMAP* fiout = FreeImage_ConvertFromRawBits(host_in2, width, height, width,
                                                   8, 255, 255, 255, FALSE);
    FreeImage_Save(FIF_PNG, fiout, "output.png");

    for (int y=0; y<STRIDE_Y; y++) {
        for (int x=0; x<STRIDE_X; x++) {
            int pos = x + y*(STRIDE_X);
            int ox = x*tile_size + tile_size/2;
            int oy = y*tile_size + tile_size/2;
            if (vecs[pos].w != 0) {
                // polar to Cartesian
                float xf = vecs[pos].x * cos(vecs[pos].y);
                float yf = vecs[pos].x * sin(vecs[pos].y);

                // correct rounding
                int high = (int)(xf + 0.5f - (xf < 0 ? 1.0f : 0.0f));
                int low = (int)(yf + 0.5f - (yf < 0 ? 1.0f : 0.0f));

                // draw line
                float m = (float)low/(float)high;
                if (abs(low) > abs(high)) {
                    bool negative = low < 0;
                    for (int i=0; i<abs(low); ++i) {
                        int iy = oy + (negative ? -i : i);
                        int ix = ox + (negative ? -i/m : i/m);
                        int pos = ix + iy*width;
                        if (pos > 0 && pos < width*height)
                            host_in2[pos] = 255;
                    }
                } else {
                    bool negative = high < 0;
                    for (int i=0; i<abs(high); ++i) {
                        int ix = ox + (negative ? -i : i);
                        int iy = oy + (negative ? -i*m : i*m);
                        int pos = ix + iy*width;
                        if (pos > 0 && pos < width*height)
                            host_in2[pos] = 255;
                    }
                }
            }
        }
    }

    FIBITMAP* fidbg = FreeImage_ConvertFromRawBits(host_in2, width, height, width,
                                                   8, 255, 255, 255, FALSE);
    FreeImage_Save(FIF_PNG, fidbg, "debug.png");

    FreeImage_Unload(fiout);
    FreeImage_Unload(fidbg);
    FreeImage_Unload(fiin1);
    FreeImage_Unload(fiin2);
    FreeImage_DeInitialise();
#endif

    return EXIT_SUCCESS;
}
