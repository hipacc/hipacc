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

#ifndef __PYRAMID_HPP__
#define __PYRAMID_HPP__

#include <vector>
#include <functional>

#include "types.hpp"
#include "image.hpp"

namespace hipacc {

// forward declaration
class Traversal;
class Recursion;

class PyramidBase {
    friend class Traversal;
    friend class Recursion;

    private:
        const int depth_;
        int level_;
        bool bound_;

        void increment() {
            ++level_;
        }

        void decrement() {
            --level_;
        }

        bool bind() {
            if (!bound_) {
                bound_ = true;
                level_ = 0;
                return true;
            } else {
                return false;
            }
        }

        void unbind() {
            bound_ = false;
        }

    public:
        explicit PyramidBase(const int depth)
            : depth_(depth), level_(0), bound_(false) {
        }

        int depth() const {
            return depth_;
        }

        int level() const {
            return level_;
        }

        bool is_top_level() const {
            return level_ == 0;
        }

        bool is_bottom_level() const {
            return level_ == depth_-1;
        }
};


template<typename data_t>
class Pyramid : public PyramidBase {
    private:
        std::vector<Image<data_t>> imgs_;

    public:
        Pyramid(Image<data_t> &img, const int depth)
            : PyramidBase(depth) {
            imgs_.push_back(img);
            int height = img.height()/2;
            int width = img.width()/2;
            for (int i=1; i<depth; ++i) {
                assert(width * height > 0 && "Pyramid stages too deep for image size.");
                Image<data_t> img(width, height);
                imgs_.push_back(img);
                height /= 2;
                width /= 2;
            }
        }

        Image<data_t> &operator()(const int relative) {
            assert(level() + relative >= 0 &&
                   level() + relative < (int)imgs_.size() &&
                   "Accessed pyramid stage is out of bounds.");
            return imgs_.at(level() + relative);
        }

        void swap(Pyramid<data_t> &other) {
            std::vector<Image<data_t>> tmp = other.imgs_;
            other.imgs_ = this->imgs_;
            this->imgs_ = tmp;
        }
};


std::vector<const std::function<void()>*> gTraverse;
std::vector<std::vector<PyramidBase*>> gPyramids;


class Traversal {
    private:
        const std::function<void()> &func_;
        std::vector<PyramidBase*> pyrs_;

    public:
        explicit Traversal(const std::function<void()> &func)
            : func_(func) {
        }

        ~Traversal() {
            for (auto pyr : pyrs_) pyr->unbind();
        }

        void add(PyramidBase &p) {
            bool bound = p.bind();
            assert(bound && "Pyramid already bound to another traversal.");
            pyrs_.push_back(&p);
        }

        void run() {
            gPyramids.push_back(pyrs_);
            gTraverse.push_back(&func_);

            (*gTraverse.back())();

            gTraverse.pop_back();
            gPyramids.pop_back();
        }
};


class Recursion {
    private:
        const std::function<void()> &func_;

    public:
        explicit Recursion(const std::function<void()> &func)
            : func_(func) {
            std::vector<PyramidBase*> pyrs = gPyramids.back();
            for (auto pyr : pyrs)
                pyr->increment();
        }

        ~Recursion() {
            std::vector<PyramidBase*> pyrs = gPyramids.back();
            for (auto pyr : pyrs)
                pyr->decrement();
        }

        void run(const int loop) {
            for (int i=0; i<loop; ++i) {
                (*gTraverse.back())();
                if (i < loop-1) {
                    func_();
                }
            }
        }
};


void traverse(PyramidBase &p0, const std::function<void()> &func) {
    Traversal t(func);
    t.add(p0);
    t.run();
}


void traverse(PyramidBase &p0, PyramidBase &p1,
              const std::function<void()> &func) {
    assert(p0.depth() == p1.depth() &&
           "Pyramid depths do not match.");

    Traversal t(func);
    t.add(p0);
    t.add(p1);
    t.run();
}


void traverse(PyramidBase &p0, PyramidBase &p1, PyramidBase &p2,
              const std::function<void()> &func) {
    assert(p0.depth() == p1.depth() &&
           p1.depth() == p2.depth() &&
           "Pyramid depths do not match.");

    Traversal t(func);
    t.add(p0);
    t.add(p1);
    t.add(p2);
    t.run();
}


void traverse(PyramidBase &p0, PyramidBase &p1, PyramidBase &p2,
              PyramidBase &p3, const std::function<void()> &func) {
    assert(p0.depth() == p1.depth() &&
           p1.depth() == p2.depth() &&
           p2.depth() == p3.depth() &&
           "Pyramid depths do not match.");

    Traversal t(func);
    t.add(p0);
    t.add(p1);
    t.add(p2);
    t.add(p3);
    t.run();
}


void traverse(PyramidBase &p0, PyramidBase &p1, PyramidBase &p2,
              PyramidBase &p3, PyramidBase &p4,
              const std::function<void()> &func) {
    assert(p0.depth() == p1.depth() &&
           p1.depth() == p2.depth() &&
           p2.depth() == p3.depth() &&
           p3.depth() == p4.depth() &&
           "Pyramid depths do not match.");

    Traversal t(func);
    t.add(p0);
    t.add(p1);
    t.add(p2);
    t.add(p3);
    t.add(p4);
    t.run();
}


void traverse(std::vector<PyramidBase*> pyrs, const std::function<void()> &func) {
    Traversal t(func);
    for (size_t i=0; i<pyrs.size(); ++i) {
        if (i < pyrs.size() - 1) {
            assert(pyrs[i]->depth() == pyrs[i+1]->depth() && "Pyramid depths do not match.");
        }
        t.add(*(pyrs[i]));
    }

    t.run();
}


void traverse(size_t loop=1, const std::function<void()> &func=[]{}) {
    assert(!gPyramids.empty() && "Traverse recursion called outside of traverse.");

    std::vector<PyramidBase*> pyrs = gPyramids.back();

    if (!pyrs.at(0)->is_bottom_level()) {
        Recursion r(func);
        r.run(loop);
    }
}

} // end namespace hipacc

#endif // __PYRAMID_HPP__

