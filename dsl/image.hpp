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
template<typename data_t> class AccessorLF;
template<typename data_t> class AccessorCF;
template<typename data_t> class AccessorL3;

enum hipaccBoundaryMode {
    BOUNDARY_UNDEFINED,
    BOUNDARY_CLAMP,
    BOUNDARY_REPEAT,
    BOUNDARY_MIRROR,
    BOUNDARY_CONSTANT
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
    template<typename> friend class AccessorNN;
    template<typename> friend class AccessorLF;
    template<typename> friend class AccessorCF;
    template<typename> friend class AccessorL3;
};


template<typename data_t>
class BoundaryCondition {
    protected:
        Image<data_t> &img;
        int size_x, size_y;
        hipaccBoundaryMode mode;
        // dummy reference to return a reference for constants
        data_t const_val;
        data_t &dummy;

    public:
        BoundaryCondition(Image<data_t> &Img, int size_x, int size_y, hipaccBoundaryMode mode) :
            img(Img),
            size_x(size_x),
            size_y(size_y),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode!=BOUNDARY_CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, int size, hipaccBoundaryMode mode) :
            img(Img),
            size_x(size),
            size_y(size),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode!=BOUNDARY_CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, hipaccBoundaryMode mode) :
            img(Img),
            size_x(Mask.getSizeX()),
            size_y(Mask.getSizeY()),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode!=BOUNDARY_CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, int size_x, int size_y, hipaccBoundaryMode mode, data_t val) :
            img(Img),
            size_x(size_x),
            size_y(size_y),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode==BOUNDARY_CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, int size, hipaccBoundaryMode mode, data_t val) :
            img(Img),
            size_x(size),
            size_y(size),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode==BOUNDARY_CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, hipaccBoundaryMode mode, data_t val) :
            img(Img),
            size_x(Mask.getSizeX()),
            size_y(Mask.getSizeY()),
            mode(mode),
            const_val(),
            dummy(const_val)
        {
            assert(mode==BOUNDARY_CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        int clamp(int coord, int lb, int ub) {
            return std::min(std::max(coord, lb), ub);
        }

    template<typename> friend class Accessor;
};


class AccessorBase {
    protected:
        const int width, height;
        const int offset_x, offset_y;
        ElementIterator *EI;

        virtual void setEI(ElementIterator *ei) {
            // TODO: enable this again for debugging
            //assert((!ei || ei->getWidth()==width) &&
            //        "For Accessor, width of Images and Iterationspace must be equal!");
            //assert((!ei || ei->getHeight()==height) &&
            //        "For Accessor, height of Image and Iterationspace must be equal!");
            EI = ei;
        }


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
class Accessor : public AccessorBase, BoundaryCondition<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;

        virtual data_t &interpolate(int x, int y, int xf=0, int yf=0) {
            return getPixelBH(EI->getX() - EI->getOffsetX() + offset_x + xf,
                    EI->getY() - EI->getOffsetY() + offset_y + yf);
        }

        // used by output Accessor: outputAtPixel(x, y)
        // and input Accessors: getPixel(x, y)
        // x and y refer to the area defined by the Accessor
        data_t &getPixelFromImg(int x, int y) {
            assert(EI && "ElementIterator not set!");
            return img.getPixel(x + offset_x,
                    y + offset_y);
        }

        data_t &getPixelBH(int x, int y) {
            data_t *ret = &dummy;

            switch (mode) {
                case BOUNDARY_UNDEFINED:
                    ret = &img.getPixel(x, y);
                    break;
                case BOUNDARY_CLAMP:
                    x = clamp(x, offset_x, offset_x+width-1);
                    y = clamp(y, offset_y, offset_y+height-1);
                    ret = &img.getPixel(x, y);
                    break;
                case BOUNDARY_REPEAT:
                    while (x < offset_x) x += width;
                    while (y < offset_y) y += height;
                    while (x >= offset_x+width) x -= width;
                    while (y >= offset_y+height) y -= height;
                    ret = &img.getPixel(x, y);
                    break;
                case BOUNDARY_MIRROR:
                    if (x < offset_x) x = offset_x + (offset_x - x - 1);
                    if (y < offset_y) y = offset_y + (offset_y - y - 1);
                    if (x >= offset_x+width) x = offset_x+width - (x + 1 - (offset_x+width));
                    if (y >= offset_y+height) y = offset_y+height - (y + 1 - (offset_y+height));
                    ret = &img.getPixel(x, y);
                    break;
                case BOUNDARY_CONSTANT:
                    if (x < offset_x || y < offset_y || x >=
                            offset_x+width || y >= offset_y+height) {
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
        Accessor(Image<data_t> &Img) :
            AccessorBase(Img.getWidth(), Img.getHeight(), 0, 0),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, BOUNDARY_CLAMP))
        {}

        Accessor(Image<data_t> &Img, int width, int height, int xf, int yf) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, BOUNDARY_CLAMP))
        {}

        Accessor(BoundaryCondition<data_t> &BC) :
            AccessorBase(BC.img.getWidth(), BC.img.getHeight(), 0, 0),
            BoundaryCondition<data_t>(BC)
        {}

        Accessor(BoundaryCondition<data_t> &BC, int width, int height, int xf,
                int yf) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BC)
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
            return EI->getX() - EI->getOffsetX();
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            return EI->getY() - EI->getOffsetY();
        }

    template<typename> friend class Image;
    template<typename> friend class Kernel;
    template<typename> friend class AccessorNN;
    template<typename> friend class AccessorLF;
    template<typename> friend class AccessorCF;
    template<typename> friend class AccessorL3;
};


template<typename data_t>
class AccessorNN : public Accessor<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using Accessor<data_t>::width;
        using Accessor<data_t>::height;
        using Accessor<data_t>::offset_x;
        using Accessor<data_t>::offset_y;
        using Accessor<data_t>::EI;
        using Accessor<data_t>::getPixel;
        using Accessor<data_t>::getPixelBH;

        void setEI(ElementIterator *ei) { EI = ei; }

        data_t &interpolate(int x, int y, int xf=0, int yf=0) {
            float stride_x = width/(float)EI->getWidth();
            float stride_y = height/(float)EI->getHeight();
            int x_mapped = offset_x + (int)(stride_x*(x - EI->getOffsetX() + xf));
            int y_mapped = offset_y + (int)(stride_y*(y - EI->getOffsetY() + yf));

            return getPixelBH(x_mapped, y_mapped);
        }


    public:
        using Accessor<data_t>::operator=;

        AccessorNN(Image<data_t> &Img) :
            Accessor<data_t>(Img)
        {}

        AccessorNN(Image<data_t> &Img, int width, int height, int xf=0, int
                yf=0) :
            Accessor<data_t>(Img, width, height, xf, yf)
        {}

        AccessorNN(BoundaryCondition<data_t> &BC) :
            Accessor<data_t>(BC)
        {}

        AccessorNN(BoundaryCondition<data_t> &BC, int width, int height, int
                xf=0, int yf=0) :
            Accessor<data_t>(BC, width, height, xf, yf)
        {}

        int getX(void) {
            assert(EI && "ElementIterator not set!");
            return (EI->getX() - EI->getOffsetX()) *
                width/(float)EI->getWidth();
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            return EI->getY() - EI->getOffsetY() *
                height/(float)EI->getHeight();
        }
};


template<typename data_t>
class AccessorLF : public Accessor<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using Accessor<data_t>::width;
        using Accessor<data_t>::height;
        using Accessor<data_t>::offset_x;
        using Accessor<data_t>::offset_y;
        using Accessor<data_t>::EI;
        using Accessor<data_t>::getPixel;
        using Accessor<data_t>::getPixelBH;
        // dummy reference to return a reference for interpolation
        data_t interpol_init;
        data_t &interpol_val;

        void setEI(ElementIterator *ei) { EI = ei; }

        data_t &interpolate(int x, int y, int xf=0, int yf=0) {
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


    public:
        using Accessor<data_t>::operator=;
        void operator=(AccessorLF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorCF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorL3<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }

        AccessorLF(Image<data_t> &Img) :
            Accessor<data_t>(Img),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorLF(Image<data_t> &Img, int width, int height, int xf=0, int
                yf=0) :
            Accessor<data_t>(Img, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorLF(BoundaryCondition<data_t> &BC) :
            Accessor<data_t>(BC),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorLF(BoundaryCondition<data_t> &BC, int width, int height, int
                xf=0, int yf=0) :
            Accessor<data_t>(BC, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        int getX(void) {
            assert(EI && "ElementIterator not set!");
            return (EI->getX() - EI->getOffsetX()) *
                width/(float)EI->getWidth();
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            return EI->getY() - EI->getOffsetY() *
                height/(float)EI->getHeight();
        }
};


template<typename data_t>
class AccessorCF : public Accessor<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using Accessor<data_t>::width;
        using Accessor<data_t>::height;
        using Accessor<data_t>::offset_x;
        using Accessor<data_t>::offset_y;
        using Accessor<data_t>::EI;
        using Accessor<data_t>::getPixel;
        using Accessor<data_t>::getPixelBH;
        // dummy reference to return a reference for interpolation
        data_t interpol_init;
        data_t &interpol_val;

        void setEI(ElementIterator *ei) { EI = ei; }

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

        data_t &interpolate(int x, int y, int xf=0, int yf=0) {
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


    public:
        using Accessor<data_t>::operator=;
        void operator=(AccessorLF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorCF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorL3<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }

        AccessorCF(Image<data_t> &Img) :
            Accessor<data_t>(Img),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorCF(Image<data_t> &Img, int width, int height, int xf=0, int
                yf=0) :
            Accessor<data_t>(Img, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorCF(BoundaryCondition<data_t> &BC) :
            Accessor<data_t>(BC),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorCF(BoundaryCondition<data_t> &BC, int width, int height, int
                xf=0, int yf=0) :
            Accessor<data_t>(BC, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        int getX(void) {
            assert(EI && "ElementIterator not set!");
            return (EI->getX() - EI->getOffsetX()) *
                width/(float)EI->getWidth();
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            return EI->getY() - EI->getOffsetY() *
                height/(float)EI->getHeight();
        }
};


template<typename data_t>
class AccessorL3 : public Accessor<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::size_x;
        using BoundaryCondition<data_t>::size_y;
        using BoundaryCondition<data_t>::mode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using Accessor<data_t>::width;
        using Accessor<data_t>::height;
        using Accessor<data_t>::offset_x;
        using Accessor<data_t>::offset_y;
        using Accessor<data_t>::EI;
        using Accessor<data_t>::getPixel;
        using Accessor<data_t>::getPixelBH;
        // dummy reference to return a reference for interpolation
        data_t interpol_init;
        data_t &interpol_val;

        void setEI(ElementIterator *ei) { EI = ei; }

        #define PI 3.14159265358979323846

        data_t lanczos(data_t diff) {
            // Digital image processing: an algorithmic introduction using Java
            // Wilhelm Burger, Mark Burge
            //
            // Lanczos interpolation
            //          1                                    |x| = 0
            // wL3(x) = 3 * sin(PI*x/3)*sin(PI*x)       0 <  |x| < 3
            //          0                               3 <= |x|
            diff = abs(diff);
            float l = 3.0f;

            if (diff==0.0f) return (data_t)1;
            else if (diff < l) {
                return l * (std::sin(PI*diff/l) * std::sin(PI*diff)) / (PI*PI*diff*diff);
            } else return (data_t)0;
        }

        data_t &interpolate(int x, int y, int xf=0, int yf=0) {
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


    public:
        using Accessor<data_t>::operator=;
        void operator=(AccessorLF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorCF<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }
        void operator=(AccessorL3<data_t> &other) {
            Accessor<data_t>::operator=(other);
        }

        AccessorL3(Image<data_t> &Img) :
            Accessor<data_t>(Img),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorL3(Image<data_t> &Img, int width, int height, int xf=0, int
                yf=0) :
            Accessor<data_t>(Img, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorL3(BoundaryCondition<data_t> &BC) :
            Accessor<data_t>(BC),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        AccessorL3(BoundaryCondition<data_t> &BC, int width, int height, int
                xf=0, int yf=0) :
            Accessor<data_t>(BC, width, height, xf, yf),
            interpol_init(0),
            interpol_val(interpol_init)
        {}

        int getX(void) {
            assert(EI && "ElementIterator not set!");
            return (EI->getX() - EI->getOffsetX()) *
                width/(float)EI->getWidth();
        }

        int getY(void) {
            assert(EI && "ElementIterator not set!");
            return EI->getY() - EI->getOffsetY() *
                height/(float)EI->getHeight();
        }
};
} // end namespace hipacc

#endif // __IMAGE_HPP__

