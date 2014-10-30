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

#ifndef __ITERATIONSPACE_HPP__
#define __ITERATIONSPACE_HPP__

#include "image.hpp"

namespace hipacc {
// forward declaration
template<typename data_t> class Image;

class Coordinate {
    public:
        int x, y;
    Coordinate() : x(0), y(0) {}
    Coordinate(int x, int y) : x(x), y(y) {}
};

class IterationSpaceBase {
    private:
        const int width_, height_;
        const int offset_x_, offset_y_;

    public:
        IterationSpaceBase(int width, int height, int offset_x=0, int
                offset_y=0) :
            width_(width),
            height_(height),
            offset_x_(offset_x),
            offset_y_(offset_y)
        {}

        virtual ~IterationSpaceBase() {}

        class ElementIterator {
            protected:
                const int min_x, min_y;
                const int max_x, max_y;
                const IterationSpaceBase *iteration_space;
                Coordinate coord;

            public:
                ElementIterator(const int width=0, const int height=0,
                                const int offset_x=0, const int offset_y=0,
                                const IterationSpaceBase *iteration_space=nullptr) :
                    min_x(offset_x),
                    min_y(offset_y),
                    max_x(offset_x+width),
                    max_y(offset_y+height),
                    iteration_space(iteration_space),
                    coord(offset_x, offset_y)
                {}

                // increment so we iterate over elements in a block
                ElementIterator &operator++() {
                    if (iteration_space) {
                        coord.x++;
                        if (coord.x >= max_x) {
                            coord.x = min_x;
                            coord.y++;
                            if (coord.y >= max_y) {
                                iteration_space = nullptr;
                            }
                        }
                    }
                    return *this;
                }

                operator const void*() { return iteration_space; }

                int x() const { return coord.x; }
                int y() const { return coord.y; }
                int width() const { return max_x - min_x; }
                int height() const { return max_y - min_y; }
                int offset_x() const { return min_x; }
                int offset_y() const { return min_y; }
        };

        ElementIterator begin() const {
            return ElementIterator(width_, height_, offset_x_, offset_y_, this);
        }
        ElementIterator end() const { return ElementIterator(); }

        int width()    const { return width_; }
        int height()   const { return height_; }
        int offset_x() const { return offset_x_; }
        int offset_y() const { return offset_y_; }
};


template<typename data_t>
class IterationSpace : public IterationSpaceBase {
    private:
        Image<data_t> &img;

    public:
        IterationSpace(Image<data_t> &img) :
            IterationSpaceBase(img.width(), img.height()),
            img(img)
        {}

        IterationSpace(Image<data_t> &img, const int width, const int height) :
            IterationSpaceBase(width, height),
            img(img)
        {}

        IterationSpace(Image<data_t> &img, const int width, const int height, const int offset_x, const int offset_y) :
            IterationSpaceBase(width, height, offset_x, offset_y),
            img(img)
        {}

        ~IterationSpace() {}

    template<typename> friend class Kernel;
};

// provide shortcut for ElementIterator
using ElementIterator = IterationSpaceBase::ElementIterator;
} // end namespace hipacc

#endif // __ITERATIONSPACE_HPP__

