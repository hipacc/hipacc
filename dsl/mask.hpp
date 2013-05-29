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

#ifndef __MASK_HPP__
#define __MASK_HPP__

#ifndef NO_BOOST
#include <boost/exception/all.hpp>
#include <boost/multi_array.hpp>
#endif

#include "iterationspace.hpp"
#include "types.hpp"

namespace hipacc {
// forward declaration
template<typename data_t> class BoundaryCondition;
enum HipaccConvolutionMode {
    HipaccSUM,
    HipaccMIN,
    HipaccMAX,
    HipaccPROD,
    HipaccMEDIAN
};

class MaskBase {
    protected:
        const int size_x, size_y;
        const int offset_x, offset_y;
        IterationSpaceBase iteration_space;
        ElementIterator *EI;


    public:
        MaskBase(int size_x, int size_y) :
            size_x(size_x),
            size_y(size_y),
            offset_x(-size_x/2),
            offset_y(-size_y/2),
            iteration_space(size_x, size_y),
            EI(NULL)
        {}

        int getX() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->getX();
        }
        int getY() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->getY();
        }

        ElementIterator begin() const {
            return ElementIterator(size_x, size_y, offset_x, offset_y,
                    &iteration_space);
        }
        ElementIterator end() const { return ElementIterator(); }

        virtual void setEI(ElementIterator *ei) {
            EI = ei;
        }
    template<typename> friend class BoundaryCondition;
};


template<typename data_t>
class Mask : public MaskBase {
    private:
        #ifdef NO_BOOST
        data_t *array;
        #else
        typedef boost::multi_array<data_t, 2> Array2D;
        Array2D array;
        #endif

    public:
        Mask(int size_x, int size_y) :
            MaskBase(size_x, size_y),
            #ifdef NO_BOOST
            array((data_t *)malloc(sizeof(data_t)*size_x*size_y))
            #else
            array(boost::extents[size_y][size_x])
            #endif
        {
            assert(size_x>0 && size_y>0 && "size for Mask must be positive!");
        }

        ~Mask() {
            #ifdef NO_BOOST
            free(array);
            #else
            #endif
        }

        data_t &operator()(void) {
            assert(EI && "ElementIterator for Mask not set!");
            #ifdef NO_BOOST
            return array[(EI->getY()-offset_y)*size_x + EI->getX()-offset_x];
            #else
            return array[EI->getY()-offset_y][EI->getX()-offset_x];
            #endif
        }
        data_t &operator()(const int xf, const int yf) {
            #ifdef NO_BOOST
            return array[(yf-offset_y)*size_x + xf-offset_x];
            #else
            return array[yf-offset_y][xf-offset_x];
            #endif
        }

        Mask &operator=(const data_t *other) {
            for (int y=0; y<size_y; ++y) {
                for (int x=0; x<size_x; ++x) {
                    #ifdef NO_BOOST
                    array[y*size_x + x] = other[y*size_x + x];
                    #else
                    array[y][x] = other[y*size_x + x];
                    #endif
                }
            }

            return *this;
        }

        void setEI(ElementIterator *ei) {
            EI = ei;
        }
};


template <typename data_t, typename Function>
auto convolve(Mask<data_t> &mask, HipaccConvolutionMode mode, const Function& fun) -> decltype(fun()) {
    ElementIterator end = mask.end();
    ElementIterator iter = mask.begin();

    // register mask
    mask.setEI(&iter);

    // initialize result - calculate first iteration
    auto result = fun();
    ++iter;

    // advance iterator and apply kernel to remaining iteration space
    while (iter != end) {
        switch (mode) {
            case HipaccSUM:
                result += fun();
                break;
            case HipaccMIN:
                {
                auto tmp = fun();
                result = hipacc::math::min(tmp, result);
                }
                break;
            case HipaccMAX:
                {
                auto tmp = fun();
                result = hipacc::math::max(tmp, result);
                }
                break;
            case HipaccPROD:
                result *= fun();
                break;
            case HipaccMEDIAN:
                assert(0 && "HipaccMEDIAN not implemented yet!");
                break;
        }
        ++iter;
    }

    // de-register mask
    mask.setEI(NULL);

    return result;
}
} // end namespace hipacc

#endif // __MASK_HPP__

