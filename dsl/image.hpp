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
        const int width_, height_;
        data_t *array;
        size_t *refcount;

        data_t &pixel(const int x, const int y) { return array[y*width_ + x]; }

    public:
        Image(const int width, const int height, data_t *init) :
            width_(width),
            height_(height),
            array(new data_t[width*height]),
            refcount(new size_t(1))
        {
            std::copy(init, init + width*height, array);
        }

        Image(const int width, const int height) :
            width_(width),
            height_(height),
            array(new data_t[width*height]),
            refcount(new size_t(1))
        {
            std::fill(array, array + width*height, 0);
        }

        Image(const Image &image) :
            width_(image.width_),
            height_(image.height_),
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

        int width() const { return width_; }
        int height() const { return height_; }

        data_t *data() const { return array; }

        Image &operator=(const data_t *other) {
            for (int y=0; y<height_; ++y) {
                for (int x=0; x<width_; ++x) {
                    array[y*width_ + x] = other[y*width_ + x];
                }
            }

            return *this;
        }
        Image &operator=(Image &other) {
            assert(width_ == other.width() && height_ == other.height() &&
                    "Image sizes have to be the same!");
            for (int y=0; y<height_; ++y) {
                for (int x=0; x<width_; ++x) {
                    pixel(x, y) = other.pixel(x, y);
                }
            }

            return *this;
        }
        Image &operator=(const Accessor<data_t> &other) {
            assert(width_ == other.width_ && height_ == other.height_ &&
                    "Size of Image and Accessor have to be the same!");
            for (int y=0; y<height_; ++y) {
                for (int x=0; x<width_; ++x) {
                    pixel(x, y) = other.img.pixel(x + other.offset_x_, y + other.offset_y_);
                }
            }

            return *this;
        }

    template<typename> friend class Accessor;
};


template<typename data_t>
class BoundaryCondition {
    private:
        Image<data_t> &img;
        const int size_x_, size_y_;
        const Boundary bmode;
        // dummy reference to return a reference for constants
        data_t const_val;
        data_t &dummy;

    public:
        BoundaryCondition(Image<data_t> &Img, const int size_x, const int size_y, const Boundary bmode) :
            img(Img),
            size_x_(size_x),
            size_y_(size_y),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, const int size, const Boundary bmode) :
            img(Img),
            size_x_(size),
            size_y_(size),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, const Boundary bmode) :
            img(Img),
            size_x_(Mask.size_x()),
            size_y_(Mask.size_y()),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode != Boundary::CONSTANT && "Boundary handling set to Constant, but no Constant specified.");
        }

        BoundaryCondition(Image<data_t> &Img, const int size_x, const int size_y, const Boundary bmode, const data_t val) :
            img(Img),
            size_x_(size_x),
            size_y_(size_y),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, const int size, const Boundary bmode, const data_t val) :
            img(Img),
            size_x_(size),
            size_y_(size),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        BoundaryCondition(Image<data_t> &Img, MaskBase &Mask, const Boundary bmode, const data_t val) :
            img(Img),
            size_x_(Mask.size_x()),
            size_y_(Mask.size_y()),
            bmode(bmode),
            const_val(),
            dummy(const_val)
        {
            assert(bmode == Boundary::CONSTANT && "Constant for boundary handling specified, but boundary mode is different.");
        }

        int clamp(int idx, const int lower, const int upper) {
            return std::min(std::max(idx, lower), upper-1);
        }
        int repeat(int idx, const int lower, const int upper) {
            if (idx  < lower) idx += lower + upper;
            if (idx >= upper) idx -= lower + upper;
            return idx;
        }
        int mirror(int idx, const int lower, const int upper) {
            if (idx  < lower) idx = lower + (lower - idx-1);
            if (idx >= upper) idx = upper - (idx+1 - upper);
            return idx;
        }

    template<typename> friend class Accessor;
};


template<typename data_t>
class Interpolation {
    protected:
        const Interpolate imode;
        // dummy reference to return a reference for interpolation
        data_t interpol_init;
        data_t &interpol_val;

        virtual data_t &pixel_bh(int x, int y) = 0;

        float bicubic_spline(float diff) const {
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
            diff = std::abs(diff);
            float a = -0.5f;

            if (diff < 1.0f) {
                return (a + 2.0f) *diff*diff*diff - (a + 3.0f)*diff*diff + 1.0f;
            } else if (diff < 2.0f) {
                return a * diff*diff*diff - 5.0f * a * diff*diff + 8.0f * a * diff - 4.0f * a;
            } else {
                return 0.0f;
            }
        }

        constexpr double pi() const { return std::atan(1.0)*4; }

        float lanczos(float diff) const {
            // Digital image processing: an algorithmic introduction using Java
            // Wilhelm Burger, Mark Burge
            //
            // Lanczos interpolation
            //          1                                                    |x| = 0
            // wL3(x) = 3 * (sin(pi()*x/3) * sin(pi()*x)) / (pi^2*x^2)  0 <  |x| < 3
            //          0                                               3 <= |x|
            diff = std::abs(diff);
            float l = 3.0f;

            if (diff==0.0f) {
                return 1.0f;
            } else if (diff < l) {
                return l * (std::sin(pi()*diff/l) * std::sin(pi()*diff)) / (pi()*pi()*diff*diff);
            } else {
                return 0.0f;
            }
        }

    public:
        explicit Interpolation(const Interpolate imode) :
            imode(imode), interpol_init(), interpol_val(interpol_init) {}
        Interpolation() : Interpolation(Interpolate::NO) {}

        data_t &interpolate(ElementIterator *EI, const int offset_x, const int offset_y, const int width, const int height,
                            const int x, const int y, const int xf, const int yf) {
            // calculate the mapped address
            float stride_x = width  / (float)EI->width();
            float stride_y = height / (float)EI->height();
            float x_mapped = offset_x + stride_x*(x - EI->offset_x() + xf);
            float y_mapped = offset_y + stride_y*(y - EI->offset_y() + yf);

            float xb = x_mapped - 0.5f;
            float yb = y_mapped - 0.5f;
            int x_int = xb;
            int y_int = yb;
            float x_frac = xb - x_int;
            float y_frac = yb - y_int;

            // do the interpolation
            switch (imode) {
                case Interpolate::NO:
                    return pixel_bh(EI->x() - EI->offset_x() + offset_x + xf, EI->y() - EI->offset_y() + offset_y + yf);
                case Interpolate::NN:
                    interpol_val = pixel_bh(x_mapped, y_mapped);
                    break;
                case Interpolate::LF:
                    interpol_val = convert<data_t>(
                        (1.0f - x_frac) * (1.0f - y_frac) * as_float(pixel_bh(x_int    , y_int)) +
                                x_frac  * (1.0f - y_frac) * as_float(pixel_bh(x_int + 1, y_int)) +
                        (1.0f - x_frac) *         y_frac  * as_float(pixel_bh(x_int    , y_int + 1)) +
                                x_frac  *         y_frac  * as_float(pixel_bh(x_int + 1, y_int + 1)));
                    break;
                case Interpolate::CF: {
                    auto y0 = as_float(pixel_bh(x_int - 1 + 0, y_int - 1 + 0)) * bicubic_spline(x_frac - 1 + 0) +
                              as_float(pixel_bh(x_int - 1 + 1, y_int - 1 + 0)) * bicubic_spline(x_frac - 1 + 1) +
                              as_float(pixel_bh(x_int - 1 + 2, y_int - 1 + 0)) * bicubic_spline(x_frac - 1 + 2) +
                              as_float(pixel_bh(x_int - 1 + 3, y_int - 1 + 0)) * bicubic_spline(x_frac - 1 + 3);
                    auto y1 = as_float(pixel_bh(x_int - 1 + 0, y_int - 1 + 1)) * bicubic_spline(x_frac - 1 + 0) +
                              as_float(pixel_bh(x_int - 1 + 1, y_int - 1 + 1)) * bicubic_spline(x_frac - 1 + 1) +
                              as_float(pixel_bh(x_int - 1 + 2, y_int - 1 + 1)) * bicubic_spline(x_frac - 1 + 2) +
                              as_float(pixel_bh(x_int - 1 + 3, y_int - 1 + 1)) * bicubic_spline(x_frac - 1 + 3);
                    auto y2 = as_float(pixel_bh(x_int - 1 + 0, y_int - 1 + 2)) * bicubic_spline(x_frac - 1 + 0) +
                              as_float(pixel_bh(x_int - 1 + 1, y_int - 1 + 2)) * bicubic_spline(x_frac - 1 + 1) +
                              as_float(pixel_bh(x_int - 1 + 2, y_int - 1 + 2)) * bicubic_spline(x_frac - 1 + 2) +
                              as_float(pixel_bh(x_int - 1 + 3, y_int - 1 + 2)) * bicubic_spline(x_frac - 1 + 3);
                    auto y3 = as_float(pixel_bh(x_int - 1 + 0, y_int - 1 + 3)) * bicubic_spline(x_frac - 1 + 0) +
                              as_float(pixel_bh(x_int - 1 + 1, y_int - 1 + 3)) * bicubic_spline(x_frac - 1 + 1) +
                              as_float(pixel_bh(x_int - 1 + 2, y_int - 1 + 3)) * bicubic_spline(x_frac - 1 + 2) +
                              as_float(pixel_bh(x_int - 1 + 3, y_int - 1 + 3)) * bicubic_spline(x_frac - 1 + 3);

                    interpol_val = convert<data_t>(
                            y0 * bicubic_spline(y_frac - 1 + 0) +
                            y1 * bicubic_spline(y_frac - 1 + 1) +
                            y2 * bicubic_spline(y_frac - 1 + 2) +
                            y3 * bicubic_spline(y_frac - 1 + 3));
                    break;
                }
                case Interpolate::L3: {
                    auto y0 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 0)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 0)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 0)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 0)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 0)) * lanczos(x_frac - 2 + 4) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 0)) * lanczos(x_frac - 2 + 5);
                    auto y1 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 1)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 1)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 1)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 1)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 1)) * lanczos(x_frac - 2 + 5) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 1)) * lanczos(x_frac - 2 + 5);
                    auto y2 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 2)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 2)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 2)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 2)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 2)) * lanczos(x_frac - 2 + 4) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 2)) * lanczos(x_frac - 2 + 5);
                    auto y3 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 3)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 3)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 3)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 3)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 3)) * lanczos(x_frac - 2 + 4) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 3)) * lanczos(x_frac - 2 + 5);
                    auto y4 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 4)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 4)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 4)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 4)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 4)) * lanczos(x_frac - 2 + 4) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 4)) * lanczos(x_frac - 2 + 5);
                    auto y5 = as_float(pixel_bh(x_int - 2 + 0, y_int - 1 + 5)) * lanczos(x_frac - 2 + 0) +
                              as_float(pixel_bh(x_int - 2 + 1, y_int - 1 + 5)) * lanczos(x_frac - 2 + 1) +
                              as_float(pixel_bh(x_int - 2 + 2, y_int - 1 + 5)) * lanczos(x_frac - 2 + 2) +
                              as_float(pixel_bh(x_int - 2 + 3, y_int - 1 + 5)) * lanczos(x_frac - 2 + 3) +
                              as_float(pixel_bh(x_int - 2 + 4, y_int - 1 + 5)) * lanczos(x_frac - 2 + 4) +
                              as_float(pixel_bh(x_int - 2 + 5, y_int - 1 + 5)) * lanczos(x_frac - 2 + 5);

                    interpol_val = convert<data_t>(
                            y0 * lanczos(y_frac - 2 + 0) +
                            y1 * lanczos(y_frac - 2 + 1) +
                            y2 * lanczos(y_frac - 2 + 2) +
                            y3 * lanczos(y_frac - 2 + 3) +
                            y4 * lanczos(y_frac - 2 + 4) +
                            y5 * lanczos(y_frac - 2 + 5));
                    break;
                }
            }

            return interpol_val;
        }
};


class AccessorBase {
    protected:
        const int width_, height_;
        const int offset_x_, offset_y_;
        ElementIterator *EI;

        void set_iterator(ElementIterator *ei) { EI = ei; }

    public:
        AccessorBase(const int width, const int height, const int offset_x, const int offset_y) :
            width_(width),
            height_(height),
            offset_x_(offset_x),
            offset_y_(offset_y),
            EI(nullptr)
        {}

    template<typename, typename> friend class Kernel;
};


template<typename data_t>
class Accessor : public AccessorBase, BoundaryCondition<data_t>, Interpolation<data_t> {
    private:
        using BoundaryCondition<data_t>::img;
        using BoundaryCondition<data_t>::bmode;
        using BoundaryCondition<data_t>::const_val;
        using BoundaryCondition<data_t>::dummy;
        using BoundaryCondition<data_t>::clamp;
        using BoundaryCondition<data_t>::repeat;
        using BoundaryCondition<data_t>::mirror;
        using Interpolation<data_t>::interpolate;
        using Interpolation<data_t>::imode;

        data_t &interpolate(const int x, const int y, const int xf=0, const int yf=0) {
            return interpolate(EI, offset_x_, offset_y_, width_, height_, x, y, xf, yf);
        }

        virtual data_t &pixel_bh(int x, int y) override {
            data_t *ret = &dummy;
            int lower_x = offset_x_;
            int lower_y = offset_y_;
            int upper_x = offset_x_ + width_;
            int upper_y = offset_y_ + height_;

            switch (bmode) {
                case Boundary::UNDEFINED:
                    ret = &img.pixel(x, y);
                    break;
                case Boundary::CLAMP:
                    x = clamp(x, lower_x, upper_x);
                    y = clamp(y, lower_y, upper_y);
                    ret = &img.pixel(x, y);
                    break;
                case Boundary::REPEAT:
                    x = repeat(x, lower_x, upper_x);
                    y = repeat(y, lower_y, upper_y);
                    ret = &img.pixel(x, y);
                    break;
                case Boundary::MIRROR:
                    x = mirror(x, lower_x, upper_x);
                    y = mirror(y, lower_y, upper_y);
                    ret = &img.pixel(x, y);
                    break;
                case Boundary::CONSTANT:
                    if (x < lower_x || x >= upper_x ||
                        y < lower_y || y >= upper_y) {
                        dummy = const_val;
                        ret = &dummy;
                    } else {
                        ret = &img.pixel(x, y);
                    }
                    break;
            }

            return *ret;
        }


    public:
        Accessor(Image<data_t> &Img, const Interpolate imode = Interpolate::NO) :
            AccessorBase(Img.width(), Img.height(), 0, 0),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, Boundary::CLAMP)),
            Interpolation<data_t>(imode)
        {}

        Accessor(Image<data_t> &Img, const int width, const int height, const int xf, const int yf, const Interpolate imode = Interpolate::NO) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BoundaryCondition<data_t>(Img, 0, 0, Boundary::CLAMP)),
            Interpolation<data_t>(imode)
        {}

        Accessor(const BoundaryCondition<data_t> &BC, const Interpolate imode = Interpolate::NO) :
            AccessorBase(BC.img.width(), BC.img.height(), 0, 0),
            BoundaryCondition<data_t>(BC),
            Interpolation<data_t>(imode)
        {}

        Accessor(const BoundaryCondition<data_t> &BC, const int width, const int height, const int xf, const int yf, const Interpolate imode = Interpolate::NO) :
            AccessorBase(width, height, xf, yf),
            BoundaryCondition<data_t>(BC),
            Interpolation<data_t>(imode)
        {}

        data_t &operator()() {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->x(), EI->y());
        }

        data_t &operator()(const int xf, const int yf) {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->x(), EI->y(), xf, yf);
        }

        data_t &operator()(MaskBase &M) {
            assert(EI && "ElementIterator not set!");
            return interpolate(EI->x(), EI->y(), M.x(), M.y());
        }


        Accessor<data_t> &operator=(Image<data_t> &other) {
            assert(width_ == other.width() && height_ == other.height() &&
                    "Size of Accessor and Image have to be the same!");
            for (int y=offset_y_; y<offset_y_+height_; ++y) {
                for (int x=offset_x_; x<offset_x_+width_; ++x) {
                    img.pixel(x, y) = other.pixel(x - offset_x_, y - offset_y_);
                }
            }

            return *this;
        }
        Accessor<data_t> &operator=(Accessor<data_t> &other) {
            assert(width_ == other.width_ && height_ == other.height_ &&
                    "Accessor sizes have to be the same!");
            for (int y=offset_y_; y<offset_y_+height_; ++y) {
                for (int x=offset_x_; x<offset_x_+width_; ++x) {
                    img.pixel(x, y) = other.img.pixel(x - offset_x_ + other.offset_x_, y - offset_y_ + other.offset_y_);
                }
            }

            return *this;
        }

        // low-level access methods
        data_t &pixel_at(const int x, const int y) {
            assert(EI && "ElementIterator not set!");
            // x and y refer to the area defined by the Accessor
            return img.pixel(x + offset_x_, y + offset_y_);
        }

        int x() const {
            assert(EI && "ElementIterator not set!");
            switch (imode) {
                case Interpolate::NO: return  EI->x() - EI->offset_x();
                default:              return (EI->x() - EI->offset_x()) * width_/(float)EI->width();
            }
        }

        int y() const {
            assert(EI && "ElementIterator not set!");
            switch (imode) {
                case Interpolate::NO: return  EI->y() - EI->offset_y();
                default:              return (EI->y() - EI->offset_y()) * height_/(float)EI->height();
            }
        }

    template<typename> friend class Image;
    template<typename, typename> friend class Kernel;
};

} // end namespace hipacc

#endif // __IMAGE_HPP__

