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
    gettimeofday (&tv, nullptr);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


template<typename data_t>
class Kernel {
    private:
        const IterationSpace<data_t> &iteration_space;
        Accessor<data_t> out_acc;
        ElementIterator iter;
        std::vector<AccessorBase *> images;
        data_t reduction_result;

    public:
        Kernel(IterationSpace<data_t> &iteration_space) :
            iteration_space(iteration_space),
            out_acc(iteration_space.img,
                    iteration_space.getWidth(), iteration_space.getHeight(),
                    iteration_space.getOffsetX(), iteration_space.getOffsetY()),
            iter()
        {}

        virtual ~Kernel() {}
        virtual void kernel() = 0;
        virtual data_t reduce(data_t left, data_t right) { return left; }

        void addAccessor(AccessorBase *acc) { images.push_back(acc); }

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
            out_acc.setEI(&iter);

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
                Acc->setEI(nullptr);
            }
            // de-register output accessors
            out_acc.setEI(nullptr);

            // reset kernel iterator
            iter = ElementIterator();

            // apply reduction
            reduce();
        }

        void reduce(void) {
            ElementIterator end = iteration_space.end();
            ElementIterator iter = iteration_space.begin();

            // register output accessors
            out_acc.setEI(&iter);

            // first element
            data_t result = out_acc();
            ++iter;

            // advance iterator and apply kernel to whole iteration space
            while (iter != end) {
                result = reduce(result, out_acc());
                ++iter;
            }

            // de-register output accessors
            out_acc.setEI(nullptr);

            reduction_result = result;
        }

        data_t getReducedData() {
            return reduction_result;
        }


        // access output image
        data_t &output(void) {
            return out_acc();
        }


        // low-level access functions
        data_t &outputAtPixel(const int xf, const int yf) {
            return out_acc.getPixelFromImg(xf, yf);
        }

        int getX(void) {
            assert(iter!=ElementIterator() && "ElementIterator not set!");
            return iter.getX() - iter.getOffsetX();
        }

        int getY(void) {
            assert(iter!=ElementIterator() && "ElementIterator not set!");
            return iter.getY() - iter.getOffsetY();
        }

        // built-in functions: convolve, iterate, and reduce
        template <typename data_m, typename Function>
        auto convolve(Mask<data_m> &mask, HipaccConvolutionMode mode, const Function& fun) -> decltype(fun());
        template <typename Function>
        auto reduce(Domain &domain, HipaccConvolutionMode mode, const Function &fun) -> decltype(fun());
        template <typename Function>
        void iterate(Domain &domain, const Function &fun);
};


template <typename data_t> template <typename data_m, typename Function>
auto Kernel<data_t>::convolve(Mask<data_m> &mask, HipaccConvolutionMode mode, const Function& fun) -> decltype(fun()) {
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
    mask.setEI(nullptr);

    return result;
}


template <typename data_t> template <typename Function>
auto Kernel<data_t>::reduce(Domain &domain, HipaccConvolutionMode mode,
            const Function &fun) -> decltype(fun()) {
    Domain::DomainIterator end = domain.end();
    Domain::DomainIterator iter = domain.begin();

    // register domain
    domain.setDI(&iter);

    // initialize result - calculate first iteration
    auto result = fun();
    ++iter;

    // advance iterator and apply kernel to remaining iteration space
    while (iter != end) {
        switch (mode) {
            case HipaccSUM:
                result += fun();
                break;
            case HipaccMIN: {
                auto tmp = fun();
                result = hipacc::math::min(tmp, result);
                }
              break;
            case HipaccMAX: {
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

    // de-register domain
    domain.setDI(nullptr);

    return result;
}


template <typename data_t> template <typename Function>
void Kernel<data_t>::iterate(Domain &domain, const Function &fun) {
    Domain::DomainIterator end = domain.end();
    Domain::DomainIterator iter = domain.begin();

    // register domain
    domain.setDI(&iter);

    // advance iterator and apply kernel to iteration space
    while (iter != end) {
        fun();
        ++iter;
    }

    // de-register domain
    domain.setDI(nullptr);
}
} // end namespace hipacc

#endif // __KERNEL_HPP__

