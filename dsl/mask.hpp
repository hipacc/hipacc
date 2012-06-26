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
        int getRelX() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->getX() + offset_x;
        }
        int getRelY() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->getY() + offset_y;
        }

        ElementIterator begin() const {
            return ElementIterator(size_x, size_y, 0, 0, &iteration_space);
        }
        ElementIterator end() const { return ElementIterator(); }

        virtual void setEI(ElementIterator *ei) {
            EI = ei;
        }
    template<typename> friend class BoundaryCondition;
};


template<typename data_t>
class Mask : public MaskBase {
    #ifndef NO_BOOST
    typedef boost::multi_array<data_t, 2> Array2D;
    #endif

    private:
        #ifndef NO_BOOST
        Array2D array;
        #else
        data_t *array;
        #endif

    public:
        Mask(int size_x, int size_y) :
            MaskBase(size_x, size_y),
            #ifndef NO_BOOST
            array(boost::extents[size_y][size_x])
            #else
            array((data_t *)malloc(sizeof(data_t)*size_x*size_y))
            #endif
        {
            assert(size_x>0 && size_y>0 && "size for Mask must be positive!");
        }

        ~Mask() {}

        data_t &operator()(void) {
            assert(EI && "ElementIterator for Mask not set!");
            #ifndef NO_BOOST
            return array[EI->getY()][EI->getX()];
            #else
            return array[EI->getY()*size_x + EI->getX()];
            #endif
        }
        data_t &operator()(const int xf, const int yf) {
            #ifndef NO_BOOST
            return array[yf][xf];
            #else
            return array[yf*size_x + xf];
            #endif
        }

        Mask &operator=(const data_t *other) {
            for (int y=0; y<size_y; ++y) {
                for (int x=0; x<size_x; ++x) {
                    #ifndef NO_BOOST
                    array[y][x] = other[y*size_x + x];
                    #else
                    array[y*size_x + x] = other[y*size_x + x];
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
data_t convolve(Mask<data_t> &mask, HipaccConvolutionMode mode, const Function& fun) {
    data_t tmp = 0, result = 0;
    bool first = true;

    ElementIterator end = mask.end();
    ElementIterator iter = mask.begin();

    // register mask
    mask.setEI(&iter);

    // advance iterator and apply kernel to whole iteration space
    while (iter != end) {
        switch (mode) {
            case HipaccSUM:
                if (first) {
                    result = fun();
                } else {
                    result += fun();
                }
                break;
            case HipaccMIN:
                if (first) {
                    result = fun();
                } else {
                    tmp = fun();
                    if (tmp < result) result = tmp;
                }
                break;
            case HipaccMAX:
                if (first) {
                    result = fun();
                } else {
                    tmp = fun();
                    if (tmp > result) result = tmp;
                }
                break;
            case HipaccPROD:
                if (first) {
                    result = fun();
                } else {
                    result *= fun();
                }
                break;
            case HipaccMEDIAN:
                assert(0 && "HipaccMEDIAN not implemented yet!");
                break;
        }
        first = false;
        ++iter;
    }

    // de-register mask
    mask.setEI(NULL);

    return result;
}
} // end namespace hipacc

#endif // __MASK_HPP__

