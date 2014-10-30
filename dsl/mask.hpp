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

#include "iterationspace.hpp"
#include "types.hpp"

namespace hipacc {
class MaskBase {
    protected:
        const int size_x_, size_y_;
        uchar *domain_space;
        IterationSpaceBase iteration_space;

    public:
        MaskBase(const int size_x, const int size_y) :
            size_x_(size_x),
            size_y_(size_y),
            domain_space(new uchar[size_x*size_y]),
            iteration_space(size_x, size_y)
        {
            assert(size_x>0 && size_y>0 && "Size for Domain must be positive!");
            // initialize full domain
            for (int i = 0; i < size_x*size_y; ++i) {
              domain_space[i] = 1;
            }
        }

        MaskBase(const MaskBase &mask) :
            size_x_(mask.size_x_),
            size_y_(mask.size_y_),
            domain_space(new uchar[mask.size_x_*mask.size_y_]),
            iteration_space(mask.size_x_, mask.size_y_)
        {
            for (int y=0; y<size_y_; ++y) {
              for (int x=0; x<size_x_; ++x) {
                domain_space[y * size_x_ + x] = mask.domain_space[y * size_x_ + x];
              }
            }
        }

        ~MaskBase() {
            if (domain_space != nullptr) {
              delete[] domain_space;
              domain_space = nullptr;
            }
        }

        int size_x() const { return size_x_; }
        int size_y() const { return size_y_; }

        virtual int x() = 0;
        virtual int y() = 0;

    friend class Domain;
    template<typename> friend class Mask;
};


class Domain : public MaskBase {
    public:
        class DomainIterator : public ElementIterator {
            private:
                uchar *domain_space;

            public:
                DomainIterator(const int width=0, const int height=0,
                               const IterationSpaceBase *iterspace=nullptr,
                               uchar *domain_space=nullptr) :
                    ElementIterator(width, height, 0, 0, iterspace),
                    domain_space(domain_space)
                {
                    if (domain_space != nullptr) {
                        // set current coordinate before domain
                        coord.x = min_x-1;
                        coord.y = min_y;

                        // use increment to search first non-zero value
                        ++(*this);
                    }
                }

                ~DomainIterator() {}

                // increment so we iterate over elements in a block
                DomainIterator &operator++() {
                    do {
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
                    } while (nullptr != iteration_space &&
                             (nullptr == domain_space ||
                              0 == domain_space[(coord.y-min_y) * (max_x-min_x)
                                                + (coord.x-min_x)]));
                    return *this;
                }
            };

        // Helper class: Force Domain assignment
        //   D(1, 1) = 0;
        // to be an CXXOperatorCallExpr, which is easier to track.
        class DomainSetter {
            friend class Domain;
            private:
                uchar *domain_space;
                size_t pos;
                DomainSetter(uchar *domain_space, size_t pos) :
                    domain_space(domain_space), pos(pos)
                {}

            public:
                DomainSetter &operator=(const uchar val) {
                    domain_space[pos] = val;
                    return *this;
                }
        };

    protected:
        DomainIterator *DI;

    public:
        Domain(const int size_x, const int size_y) :
            MaskBase(size_x, size_y),
            DI(nullptr) {}

        template <int size_y, int size_x>
        Domain(const uchar (&domain)[size_y][size_x]) :
            MaskBase(size_x, size_y),
            DI(nullptr) {}

        Domain(const MaskBase &mask) :
            MaskBase(mask),
            DI(nullptr) {}

        Domain(const Domain &domain) :
            MaskBase(domain),
            DI(domain.DI) {}

        ~Domain() {}

        virtual int x() override {
            assert(DI && "DomainIterator for Domain not set!");
            return DI->x() - size_x_/2;
        }
        virtual int y() override {
            assert(DI && "DomainIterator for Domain not set!");
            return DI->y() - size_y_/2;
        }

        DomainSetter operator()(const int xf, const int yf) {
            assert(xf < size_x_/2 && yf < size_y_ &&
                    "out of bounds Domain access.");
            return DomainSetter(domain_space, (yf+size_x_/2)*size_x_ + xf+size_x_/2);
        }

        Domain &operator=(const uchar *other) {
            for (int y=0; y<size_y_; ++y) {
                for (int x=0; x<size_x_; ++x) {
                    domain_space[y * size_x_ + x] = other[y * size_x_ + x];
                }
            }

            return *this;
        }

        void operator=(const Domain &dom) {
            assert(size_x_==dom.size_x() && size_y_==dom.size_y() &&
                    "Domain sizes must be equal.");
            for (int y=0; y<size_y_; ++y) {
                for (int x=0; x<size_x_; ++x) {
                    domain_space[y * size_x_ + x] = dom.domain_space[y * size_x_ + x];
                }
            }
        }

        void operator=(const MaskBase &mask) {
            assert(size_x_==mask.size_x() && size_y_==mask.size_y() &&
                    "Domain and Mask size must be equal.");
            for (int y=0; y<size_y_; ++y) {
                for (int x=0; x<size_x_; ++x) {
                    domain_space[y * size_x_ + x] = mask.domain_space[y * size_x_ + x];
                }
            }
        }

        void setDI(DomainIterator *di) { DI = di; }
        DomainIterator begin() const {
            return DomainIterator(size_x_, size_y_, &iteration_space,
                                  domain_space);
        }
        DomainIterator end() const { return DomainIterator(); }
};


template<typename data_t>
class Mask : public MaskBase {
    private:
        ElementIterator *EI;
        data_t *array;

        template <int size_y, int size_x>
        void init(const data_t (&mask)[size_y][size_x]) {
            for (int y=0; y<size_y; ++y) {
                for (int x=0; x<size_x; ++x) {
                    array[y*size_x + x] = mask[y][x];
                    // set holes in underlying domain_space
                    if (mask[y][x] == 0) {
                        domain_space[y*size_x + x] = 0;
                    } else {
                        domain_space[y*size_x + x] = 1;
                    }
                }
            }
        }


    public:
        template <int size_y, int size_x>
        Mask(const data_t (&mask)[size_y][size_x]) :
            MaskBase(size_x, size_y),
            EI(nullptr),
            array(new data_t[size_x*size_y])
        {
            init(mask);
        }

        Mask(const Mask &mask) :
            MaskBase(mask.size_x, mask.size_y),
            array(new data_t[mask.size_x*mask.size_y])
        {
            init(mask.array);
        }

        ~Mask() {
            if (array != nullptr) {
              delete[] array;
              array = nullptr;
            }
        }

        int x() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->x() - size_x_/2;
        }
        int y() {
            assert(EI && "ElementIterator for Mask not set!");
            return EI->y() - size_y_/2;
        }

        data_t &operator()(void) {
            assert(EI && "ElementIterator for Mask not set!");
            return array[EI->y()*size_x_ + EI->x()];
        }
        data_t &operator()(const int xf, const int yf) {
            assert(xf < size_x_/2 && yf < size_y_/2 &&
                    "out of bounds Mask access.");
            return array[(yf+size_y_/2)*size_x_ + xf+size_x_/2];
        }
        data_t &operator()(Domain &D) {
            assert(D.size_x()==size_x_ && D.size_y()==size_y_ &&
                    "Domain and Mask size must be equal.");
            return array[(D.y()+D.size_y()/2)*size_x_ + D.x()+D.size_x()/2];
        }

        void setEI(ElementIterator *ei) { EI = ei; }
        ElementIterator begin() const {
            return ElementIterator(size_x_, size_y_, 0, 0, &iteration_space);
        }
        ElementIterator end() const { return ElementIterator(); }
};
} // end namespace hipacc

#endif // __MASK_HPP__

