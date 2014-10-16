//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// Copyright (c) 2010, ARM Limited
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

#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__

#include <algorithm>
#include <cmath>

#include "iterationspace.hpp"
#include "mask.hpp"

namespace hipacc {
// forward declaration
template<typename data_t> class Accessor;

enum class Boundary : uint8_t {
    UNDEFINED = 0,
    CLAMP,
    REPEAT,
    MIRROR,
    CONSTANT
};

enum class Interpolate : uint8_t {
    NO = 0,
    NN,
    LF,
    CF,
    L3
};

template<typename data_t>
class Image {
    private:
        const int width;
        const int height;
        data_t *array;
        unsigned int *refcount;

        data_t &getPixel(int x, int y) { return array[y*width + x]; }

    public:
        Image(int width, int height) :
            width(width),
            height(height),
            array(new data_t[width*height]),
            refcount(new unsigned int(1))
        {
            std::fill(array, array + width*height, 0);
        }

        Image(const Image &image) :
            width(image.width),
            height(image.height),
            array(image.array),
            refcount(image.refcount)
        {
            ++(*refcount);
        }

        ~Image() {
            --(*refcount);
            if (array != nullptr &&
                *refcount == 0) {
              delete refcount;
              delete[] array;
              array = nullptr;
            }
        }

        int getWidth() const { return width; }
        int getHeight() const { return height; }

        data_t *getData() { return array; }

        Image &operator=(data_t *other) {
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    array[y*width + x] = other[y*width + x];
                }
            }

            return *this;
        }
        void operator=(Image &other) {
            assert(width == other.getWidth() && height == other.getHeight() &&
                    "Image sizes have to be the same!");
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    getPixel(x, y) = other.getPixel(x, y);
                }
            }
        }
        void operator=(Accessor<data_t> &other) {
            assert(width == other.width && height == other.height &&
                    "Size of Image and Accessor have to be the same!");
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    getPixel(x, y) = other.img.getPixel(x + other.offset_x,
                            y + other.offset_y);
                }
            }
        }

    template<typename> friend class Accessor;
};


template<typename data_t>
class BoundaryCondition {
    protected:
        Image<data_t> &img;
        int size_x, size_y;
        Boundary mode;
        // dummy reference to return a reference for constants
        data_t const_val;
        data_t &dummy;

    public:
        BoundaryCondition(Image<data_t> &Img, int size_x, int size_y, Boundary mode) :
            img(Img),
            size_x(size_x),
            size_y(size_y),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, int size, Boundary mode) :
            img(Img),
            size_x(size),
            size_y(size),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, Boundary mode) :
            img(Img),
            size_x(Mask.getSizeX()),
            size_y(Mask.getSizeY()),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, int size_x, int size_y, Boundary mode, data_t val) :
            img(Img),
            size_x(size_x),
            size_y(size_y),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, int size, Boundary mode, data_t val) :
            img(Img),
            size_x(size),
            size_y(size),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, Boundary mode, data_t val) :
            img(Img),
            size_x(Mask.getSizeX()),
            size_y(Mask.getSizeY()),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        int clamp(int idx, int lower, int upper) {
            return std::min(std::max(idx, lower), upper-1);
        }
        int repeat(int idx, int lower, int upper) {
            if (idx  < lower) idx += lower + upper;
            if (idx >= upper) idx -= lower + upper;
            return idx;
        }
        int mirror(int idx, int lower, int upper) {
            if (idx  < lower) idx = lower + (lower - idx-1);
            if (idx >= upper) idx = upper - (idx+1 - upper);
            return idx;
        }

    template<typename> friend class Accessor;
};


template<typename data_t>
class Interpolation {
    protected:
        Interpolate mode;
        // dummy reference to return a reference for interpolation
        data_t interpol_init;
        data_t &interpol_val;

        virtual data_t &getPixelBH(int x, int y) = 0;

        data_t bicubic(float t, data_t a, data_t b, data_t c, data_t d) {
            return 0.5 * (c - a + (2.0f * a - 5.0f * b + 4.0f * c - d + (3.0f * (b - c) + d -a) * t) * t) * t + b;
        }

        data_t bicubic_spline(data_t diff) {
            // Cubic Convolution Interpolation for Digital Image Processing
            // Robert G. Keys
            //
            // Bicubic Spline Interpolation
            // a = -0.5 .. -.75
            // a = -0.5 for best approximation of original function as proposed
            //
            //        (a + 2)|x|^3 - (a + 3)|x|^2 + 1   0 <= |x| < 1
            // w(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a     1 <= |x| < 2
            //        0                                 2 <= |x|
            diff = abs(diff);
            float a = -0.5f;

            if (diff < 1.0f) {
                return (a + 2.0f) *diff*diff*diff - (a + 3.0f)*diff*diff + 1;
            } else if (diff < 2.0f) {
                return a * diff*diff*diff - 5.0f * a * diff*diff + 8.0f * a * diff - 4.0f * a;
            } else return (data_t)0;
        }

        constexpr double pi() const { return std::atan(1)*4; }

        data_t lanczos(data_t diff) {
            // Digital image processing: an algorithmic introduction using Java
            // Wilhelm Burger, Mark Burge
            //
            // Lanczos interpolation
            //          1                                        |x| = 0
            // wL3(x) = 3 * sin(pi()*x/3)*sin(pi()*x)       0 <  |x| < 3
            //          0                                   3 <= |x|
            diff = abs(diff);
            float l = 3.0f;

            if (diff==0.0f) {
                return (data_t)1;
            } else if (diff < l) {
                return l * (std::sin(pi()*diff/l) * std::sin(pi()*diff)) / (pi()*pi()*diff*diff);
            } else {
                return (data_t)0;
            }
        }

    public:
        Interpolation(Interpolate mode) :
            mode(mode), interpol_init(0), interpol_val(interpol_init) {}
        Interpolation() : Interpolation(Interpolate::NO) {}

        data_t &interpolate(ElementIterator *EI, int offset_x, int offset_y, int width, int height, int x, int y, int xf, int yf) {
            switch (mode) {
                case Interpolate::NO:
                    return getPixelBH(EI->getX() - EI->getOffsetX() + offset_x + xf,
                                      EI->getY() - EI->getOffsetY() + offset_y + yf);
                case Interpolate::NN: {
                    float stride_x = width/(float)EI->getWidth();
                    float stride_y = height/(float)EI->getHeight();
                    int x_mapped = offset_x + (int)(stride_x*(x - EI->getOffsetX() + xf));
                    int y_mapped = offset_y + (int)(stride_y*(y - EI->getOffsetY() + yf));

                    return getPixelBH(x_mapped, y_mapped);
                    }
                case Interpolate::LF: {
                    // first calculate the mapped address
                    float stride_x = width/(float)EI->getWidth();
                    float stride_y = height/(float)EI->getHeight();
                    float x_mapped = offset_x + stride_x*(x - EI->getOffsetX() + xf);
                    float y_mapped = offset_y + stride_y*(y - EI->getOffsetY() + yf);

                    // then do the interpolation
                    float xb = x_mapped - 0.5f;
                    float yb = y_mapped - 0.5f;
                    int x_int = xb;
                    int y_int = yb;
                    float x_frac = xb - x_int;
                    float y_frac = yb - y_int;

                    interpol_val =
                        (1.0f-x_frac) * (1.0f-y_frac) * getPixelBH(x_int  , y_int) +
                              x_frac  * (1.0f-y_frac) * getPixelBH(x_int+1, y_int) +
                        (1.0f-x_frac) *       y_frac  * getPixelBH(x_int  , y_int+1) +
                              x_frac  *       y_frac  * getPixelBH(x_int+1, y_int+1);

                    return interpol_val;
                    }
                case Interpolate::CF: {
                    // first calculate the mapped address
                    float stride_x = width/(float)EI->getWidth();
                    float stride_y = height/(float)EI->getHeight();
                    float x_mapped = offset_x + stride_x*(x - EI->getOffsetX() + xf);
                    float y_mapped = offset_y + stride_y*(y - EI->getOffsetY() + yf);

                    // then do the interpolation
                    float xb = x_mapped - 0.5f;
                    float yb = y_mapped - 0.5f;
                    int x_int = xb;
                    int y_int = yb;
                    float x_frac = xb - x_int;
                    float y_frac = yb - y_int;

                    #if 1
                    data_t y0 = getPixelBH(x_int - 1 + 0, y_int - 1 + 0) * bicubic_spline(x_frac - 1 + 0) +
                                getPixelBH(x_int - 1 + 1, y_int - 1 + 0) * bicubic_spline(x_frac - 1 + 1) +
                                getPixelBH(x_int - 1 + 2, y_int - 1 + 0) * bicubic_spline(x_frac - 1 + 2) +
                                getPixelBH(x_int - 1 + 3, y_int - 1 + 0) * bicubic_spline(x_frac - 1 + 3);
                    data_t y1 = getPixelBH(x_int - 1 + 0, y_int - 1 + 1) * bicubic_spline(x_frac - 1 + 0) +
                                getPixelBH(x_int - 1 + 1, y_int - 1 + 1) * bicubic_spline(x_frac - 1 + 1) +
                                getPixelBH(x_int - 1 + 2, y_int - 1 + 1) * bicubic_spline(x_frac - 1 + 2) +
                                getPixelBH(x_int - 1 + 3, y_int - 1 + 1) * bicubic_spline(x_frac - 1 + 3);
                    data_t y2 = getPixelBH(x_int - 1 + 0, y_int - 1 + 2) * bicubic_spline(x_frac - 1 + 0) +
                                getPixelBH(x_int - 1 + 1, y_int - 1 + 2) * bicubic_spline(x_frac - 1 + 1) +
                                getPixelBH(x_int - 1 + 2, y_int - 1 + 2) * bicubic_spline(x_frac - 1 + 2) +
                                getPixelBH(x_int - 1 + 3, y_int - 1 + 2) * bicubic_spline(x_frac - 1 + 3);
                    data_t y3 = getPixelBH(x_int - 1 + 0, y_int - 1 + 3) * bicubic_spline(x_frac - 1 + 0) +
                                getPixelBH(x_int - 1 + 1, y_int - 1 + 3) * bicubic_spline(x_frac - 1 + 1) +
                                getPixelBH(x_int - 1 + 2, y_int - 1 + 3) * bicubic_spline(x_frac - 1 + 2) +
                                getPixelBH(x_int - 1 + 3, y_int - 1 + 3) * bicubic_spline(x_frac - 1 + 3);

                    interpol_val = y0*bicubic_spline(y_frac - 1 + 0) +
                                   y1*bicubic_spline(y_frac - 1 + 1) +
                                   y2*bicubic_spline(y_frac - 1 + 2) +
                                   y3*bicubic_spline(y_frac - 1 + 3);
                    #else
                    data_t y0 = bicubic(x_frac,
                            getPixelBH(i-1, j-1),
                            getPixelBH(i,   j-1),
                            getPixelBH(i+1, j-1),
                            getPixelBH(i+2, j-1));
                    data_t y1 = bicubic(x_frac,
                            getPixelBH(i-1, j),
                            getPixelBH(i,   j),
                            getPixelBH(i+1, j),
                            getPixelBH(i+2, j));
                    data_t y2 = bicubic(x_frac,
                            getPixelBH(i-1, j+1),
                            getPixelBH(i,   j+1),
                            getPixelBH(i+1, j+1),
                            getPixelBH(i+2, j+1));
                    data_t y3 = bicubic(x_frac,
                            getPixelBH(i-1, j+2),
                            getPixelBH(i,   j+2),
                            getPixelBH(i+1, j+2),
                            getPixelBH(i+2, j+2));

                    interpol_val = bicubic(y_frac, y0, y1, y2, y3);
                    #endif

                    return interpol_val;
                    }
                case Interpolate::L3: {
                    // first calculate the mapped address
                    float stride_x = width/(float)EI->getWidth();
                    float stride_y = height/(float)EI->getHeight();
                    float x_mapped = offset_x + stride_x*(x - EI->getOffsetX() + xf);
                    float y_mapped = offset_y + stride_y*(y - EI->getOffsetY() + yf);

                    // then do the interpolation
                    float xb = x_mapped - 0.5f;
                    float yb = y_mapped - 0.5f;
                    int x_int = xb;
                    int y_int = yb;
                    float x_frac = xb - x_int;
                    float y_frac = yb - y_int;

                    data_t y0 = getPixelBH(x_int - 2 + 0, y_int - 1 + 0) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 0) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 0) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 0) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 0) * lanczos(x_frac - 2 + 4) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 0) * lanczos(x_frac - 2 + 5);
                    data_t y1 = getPixelBH(x_int - 2 + 0, y_int - 1 + 1) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 1) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 1) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 1) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 1) * lanczos(x_frac - 2 + 5) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 1) * lanczos(x_frac - 2 + 5);
                    data_t y2 = getPixelBH(x_int - 2 + 0, y_int - 1 + 2) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 2) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 2) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 2) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 2) * lanczos(x_frac - 2 + 4) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 2) * lanczos(x_frac - 2 + 5);
                    data_t y3 = getPixelBH(x_int - 2 + 0, y_int - 1 + 3) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 3) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 3) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 3) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 3) * lanczos(x_frac - 2 + 4) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 3) * lanczos(x_frac - 2 + 5);
                    data_t y4 = getPixelBH(x_int - 2 + 0, y_int - 1 + 4) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 4) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 4) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 4) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 4) * lanczos(x_frac - 2 + 4) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 4) * lanczos(x_frac - 2 + 5);
                    data_t y5 = getPixelBH(x_int - 2 + 0, y_int - 1 + 5) * lanczos(x_frac - 2 + 0) +
                                getPixelBH(x_int - 2 + 1, y_int - 1 + 5) * lanczos(x_frac - 2 + 1) +
                                getPixelBH(x_int - 2 + 2, y_int - 1 + 5) * lanczos(x_frac - 2 + 2) +
                                getPixelBH(x_int - 2 + 3, y_int - 1 + 5) * lanczos(x_frac - 2 + 3) +
                                getPixelBH(x_int - 2 + 4, y_int - 1 + 5) * lanczos(x_frac - 2 + 4) +
                                getPixelBH(x_int - 2 + 5, y_int - 1 + 5) * lanczos(x_frac - 2 + 5);

                    interpol_val = y0*lanczos(y_frac - 2 + 0) +
                                   y1*lanczos(y_frac - 2 + 1) +
                                   y2*lanczos(y_frac - 2 + 2) +
                                   y3*lanczos(y_frac - 2 + 3) +
                                   y4*lanczos(y_frac - 2 + 4) +
                                   y5*lanczos(y_frac - 2 + 5);

                    return interpol_val;
                    }
            }
        }
};


class AccessorBase {
    protected:
        const int width, height;
        const int offset_x, offset_y;
        ElementIterator *EI;

        void setEI(ElementIterator *ei) { EI = ei; }

    public:
        AccessorBase(int width, int height, int offset_x, int offset_y) :
            width(width),
            height(height),
            offset_x(offset_x),
            offset_y(offset_y),
            EI(nullptr)
        {}

    template<typename> friend class Kernel;
};


template<typename data_t>
class Accessor : public AccessorBase, BoundaryCondition<data_t>, Interpolation<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using BoundaryCondition<data_t>::repeat;
        using BoundaryCondition<data_t>::mirror;
        using Interpolation<data_t>::interpolate;

        data_t &interpolate(int x, int y, int xf=0, int yf=0) {
            return interpolate(EI, offset_x, offset_y, width, height, x, y, xf, yf);
        }

        // used by output Accessor: outputAtPixel(x, y)
        // and input Accessors: getPixel(x, y)
        // x and y refer to the area defined by the Accessor
        data_t &getPixelFromImg(int x, int y) {
            assert(EI && "ElementIterator not set!");
            return img.getPixel(x + offset_x,
                    y + offset_y);
        }

        virtual data_t &getPixelBH(int x, int y) override {
            data_t *ret = &dummy;
            int lower_x = offset_x;
            int lower_y = offset_y;
            int upper_x = offset_x + width;
            int upper_y = offset_y + height;

            switch (mode) {
                case Boundary::UNDEFINED:
                    ret = &img.getPixel(x, y);
                    break;
                case Boundary::CLAMP:
                    x = clamp(x, lower_x, upper_x);
                    y = clamp(y, lower_y, upper_y);
                    ret = &img.getPixel(x, y);
                    break;
                case Boundary::REPEAT:
                    x = repeat(x, lower_x, upper_x);
                    y = repeat(y, lower_y, upper_y);
                    ret = &img.getPixel(x, y);
                    break;
                case Boundary::MIRROR:
                    x = mirror(x, lower_x, upper_x);
                    y = mirror(y, lower_y, upper_y);
                    ret = &img.getPixel(x, y);
                    break;
                case Boundary::CONSTANT:
                    if (x < lower_x || x >= upper_x ||
                        y < lower_y || y >= upper_y) {
                        dummy = const_val;
                        ret = &dummy;
                    } else {
                        ret = &img.getPixel(x, y);
                    }
                    break;
            }

            return *ret;
        }


    public:
        Accessor(Image<data_t> &Img, Interpolate mode = Interpolate::NO) :
            AccessorBase(Img.getWidth(), Img.getHeight(), 0, 0),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, Boundary::CLAMP)),
            Interpolation<data_t>(mode)
        {}

        Accessor(Image<data_t> &Img, int width, int height, int xf, int yf,
                 Interpolate mode = Interpolate::NO) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, Boundary::CLAMP)),
            Interpolation<data_t>(mode)
        {}

        Accessor(BoundaryCondition<data_t> &BC, Interpolate mode = Interpolate::NO) :
            AccessorBase(BC.img.getWidth(), BC.img.getHeight(), 0, 0),
            BoundaryCondition<data_t>(BC),
            Interpolation<data_t>(mode)
        {}

        Accessor(BoundaryCondition<data_t> &BC, int width, int height, int xf,
                 int yf, Interpolate mode = Interpolate::NO) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BC),
            Interpolation<data_t>(mode)
        {}

        data_t &operator()(void) {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->getX(), EI->getY());
        }

        data_t &operator()(const int xf, const int yf) {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->getX(), EI->getY(), xf, yf);
        }

        data_t &operator()(MaskBase &M) {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->getX(), EI->getY(), M.getX(), M.getY());
        }


        void operator=(Image<data_t> &other) {
            assert(width == other.getWidth() && height == other.getHeight() &&
                    "Size of Accessor and Image have to be the same!");
            for (int y=offset_y; y<offset_y+height; ++y) {
                for (int x=offset_x; x<offset_x+width; ++x) {
                    img.getPixel(x, y) = other.getPixel(x - offset_x, y -
                            offset_y);
                }
            }
        }
        void operator=(Accessor<data_t> &other) {
            assert(width == other.width && height == other.height &&
                    "Accessor sizes have to be the same!");
            for (int y=offset_y; y<offset_y+height; ++y) {
                for (int x=offset_x; x<offset_x+width; ++x) {
                    img.getPixel(x, y) = other.img.getPixel(x - offset_x +
                            other.offset_x, y - offset_y + other.offset_y);
                }
            }
        }

        // low-level access methods
        data_t &getPixel(int x, int y) {
            return getPixelFromImg(x, y);
        }

        int getX(void) {
            assert(EI && "ElementIterator not set!");
            switch (mode) {
                case Interpolate::NO: return  EI->getX() - EI->getOffsetX();
                default:              return (EI->getX() - EI->getOffsetX()) *
                                              width/(float)EI->getWidth();
            }
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            switch (mode) {
                case Interpolate::NO: return  EI->getY() - EI->getOffsetY();
                default:              return (EI->getY() - EI->getOffsetY()) *
                                              height/(float)EI->getHeight();
            }
        }

    template<typename> friend class Image;
    template<typename> friend class Kernel;
};

} // end namespace hipacc

#endif // __IMAGE_HPP__

