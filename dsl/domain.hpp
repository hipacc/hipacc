//
// Copyright (c) 2013, University of Erlangen-Nuremberg
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef __DOMAIN_HPP__
#define __DOMAIN_HPP__


#include "iterationspace.hpp"
#include "types.hpp"

namespace hipacc {

class Domain {
public:
  // forward declaration
  class DomainIterator;

protected:
  const int size_x, size_y;
  const int offset_x, offset_y;
  IterationSpaceBase iteration_space;
  uchar* domain_space;
  DomainIterator *DI;

public:
  class DomainIterator : public ElementIterator {
  private:
    uchar* domain_space;

  public:
    DomainIterator(int width=0, int height=0,
                   int offsetx=0, int offsety=0,
                   const IterationSpaceBase* iterspace=NULL,
                   uchar* domain_space=NULL)
        : domain_space(domain_space),
          ElementIterator(width, height, offsetx, offsety, iterspace) {
      if (domain_space != NULL) {
        // set current coordinate before domain
        coord.x = min_x-1;
        coord.y = min_y;

        // use increment to search first non-zero value
        ++(*this);
      }
    }

    ~DomainIterator() {
    }

    // increment so we iterate over elements in a block
    DomainIterator &operator++() {
      do {
        if (iteration_space) {
          coord.x++;
          if (coord.x >= max_x) {
            coord.x = min_x;
            coord.y++;
            if (coord.y >= max_y) {
              iteration_space = NULL;
            }
          }
        }
      } while (NULL != iteration_space &&
               (NULL == domain_space ||
                0 == domain_space[(coord.y-min_y) * (max_x-min_x)
                                  + (coord.x-min_x)]));
      return *this;
    }
  };

public:
  Domain(int size_x, int size_y)
      : size_x(size_x),
        size_y(size_y),
        offset_x(-size_x/2),
        offset_y(-size_y/2),
        iteration_space(size_x, size_y),
        domain_space(new uchar[size_x*size_y]),
        DI(NULL) {
    assert(size_x>0 && size_y>0 && "Size for Domain must be positive!");
    // initialize full domain
    for (int i = 0; i < size_x*size_y; ++i) {
      domain_space[i] = 1;
    }
  }

  ~Domain() {
    if (domain_space != NULL) {
      delete[] domain_space;
      domain_space = NULL;
    }
  }

  int getX() {
    assert(DI && "DomainIterator for Domain not set!");
    return DI->getX();
  }
  int getY() {
    assert(DI && "DomainIterator for Domain not set!");
    return DI->getY();
  }

  DomainIterator begin() const {
    return DomainIterator(size_x, size_y, offset_x, offset_y,
                          &iteration_space, domain_space);
  }

  DomainIterator end() const {
    return DomainIterator();
  }

  void setDI(DomainIterator *di) {
    DI = di;
  }

  uchar& operator()(unsigned int x, unsigned int y) {
    x += size_x >> 1;
    y += size_y >> 1;
    if (x < size_x && y < size_y) {
      return domain_space[y * size_x + x];
    } else {
      return domain_space[0];
    }
  }

  Domain &operator=(const uchar* other) {
    for (int y=0; y<size_y; ++y) {
      for (int x=0; x<size_x; ++x) {
        domain_space[y * size_x + x] = other[y * size_x + x];
      }
    }

    return *this;
  }
};


template <typename Function>
auto reduce(Domain &domain, HipaccConvolutionMode mode,
            const Function& fun) -> decltype(fun()) {
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
  domain.setDI(NULL);

  return result;
}

} // end namespace hipacc

#endif // __DOMAIN_HPP__

