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

#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include <vector>

#include "iterationspace.hpp"


namespace hipacc {
// get time in milliseconds
double hipacc_time_ms() {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


template<typename data_t>
class Kernel {
    private:
        const IterationSpace<data_t> &iteration_space;
        Accessor<data_t> outImgAcc;
        ElementIterator iter;
        std::vector<AccessorBase *> images;

    public:
        Kernel(IterationSpace<data_t> &iteration_space) :
            iteration_space(iteration_space),
            outImgAcc(iteration_space.OutImg,
                        iteration_space.getWidth(), iteration_space.getHeight(),
                        iteration_space.getOffsetX(),
                        iteration_space.getOffsetY()),
            iter()
        {}

        virtual ~Kernel() {}
        virtual void kernel() = 0;

        void addAccessor(AccessorBase *Acc) { images.push_back(Acc); }

        void execute() {
            double time0, time1;
            ElementIterator end = iteration_space.end();
            iter = iteration_space.begin();

            // register input accessors
            for (std::vector<AccessorBase *>::iterator ei=images.begin(), ie=images.end();
                    ei!=ie; ++ei) {
                AccessorBase *Acc = *ei;
                Acc->setEI(&iter);
            }
            // register output accessors
            outImgAcc.setEI(&iter);

            // advance iterator and apply kernel to whole iteration space
            time0 = hipacc_time_ms();
            while (iter != end) {
                kernel();
                ++iter;
            }
            time1 = hipacc_time_ms();
            hipacc_last_timing = time1 - time0;

            // de-register input accessors
            for (std::vector<AccessorBase*>::iterator ei=images.begin(), ie=images.end();
                    ei!=ie; ++ei) {
                AccessorBase *Acc = *ei;
                Acc->setEI(NULL);
            }
            // de-register output accessors
            outImgAcc.setEI(NULL);

            // reset kernel iterator
            iter = ElementIterator();

        }

        // access output image
        data_t &output(void) {
            return outImgAcc();
        }


        // low-level access functions
        data_t &outputAtPixel(const int xf, const int yf) {
            return outImgAcc.getPixelFromImg(xf, yf);
        }

        int getX(void) {
            assert(iter!=ElementIterator() && "ElementIterator not set!");
            return iter.getX() - iter.getOffsetX();
        }

        int getY(void) {
            assert(iter!=ElementIterator() && "ElementIterator not set!");
            return iter.getY() - iter.getOffsetY();
        }
};


template<typename data_t>
class GlobalReduction {
    protected:
        Accessor<data_t> imgAcc;
        IterationSpace<data_t> redIS;
        data_t neutral;

    public:
        GlobalReduction(Image<data_t> &img, data_t neutral) :
            imgAcc(img, img.getWidth(), img.getHeight(), 0, 0),
            redIS(img),
            neutral(neutral)
        {} 
        GlobalReduction(Accessor<data_t> &acc, data_t neutral) :
            imgAcc(acc),
            redIS(acc.img, acc.width, acc.height, acc.offset_x, acc.offset_y),
            neutral(neutral)
        {} 

        virtual data_t reduce(data_t left, data_t right) = 0;

        data_t reduce(void) {
            data_t result = neutral;

            ElementIterator end = redIS.end();
            ElementIterator iter = redIS.begin();

            // register output accessors
            imgAcc.setEI(&iter);

            // advance iterator and apply kernel to whole iteration space
            while (iter != end) {
                result = reduce(result, imgAcc());
                ++iter;
            }

            // de-register output accessors
            imgAcc.setEI(NULL);

            return result;
        }
};


template <typename data_t, typename Function>
data_t reduce(Image<data_t> &img, data_t neutral, const Function& fun) {
    data_t result = neutral;
    IterationSpace<data_t> redIS(img);
    Accessor<data_t> imgAcc(img);

    ElementIterator end = redIS.end();
    ElementIterator iter = redIS.begin();

    // register output accessors
    imgAcc.setEI(&iter);

    // advance iterator and apply kernel to whole iteration space
    while (iter != end) {
        result = fun(result, imgAcc());
        ++iter;
    }

    // de-register output accessors
    imgAcc.setEI(NULL);

    return result;
}
} // end namespace hipacc

#endif // __KERNEL_HPP__

