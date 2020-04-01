//
// Copyright (c) 2013, University of Erlangen-Nuremberg
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
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

// This is the standalone (header-only) Hipacc base runtime

#include "hipacc_base.hpp"

#ifndef __HIPACC_BASE_STANDALONE_HPP__
#define __HIPACC_BASE_STANDALONE_HPP__

inline hipacc_launch_info::hipacc_launch_info(int size_x, int size_y, int is_width,
                                       int is_height, int offset_x,
                                       int offset_y, int pixels_per_thread,
                                       int simd_width)
    : size_x(size_x), size_y(size_y), is_width(is_width), is_height(is_height),
      offset_x(offset_x), offset_y(offset_y),
      pixels_per_thread(pixels_per_thread), simd_width(simd_width),
      bh_start_left(0), bh_start_right(0), bh_start_top(0), bh_start_bottom(0),
      bh_fall_back(0) {}

inline hipacc_launch_info::hipacc_launch_info(int size_x, int size_y,
                                       const HipaccAccessorBase &Acc,
                                       int pixels_per_thread, int simd_width)
    : size_x(size_x), size_y(size_y), is_width(Acc.width),
      is_height(Acc.height), offset_x(Acc.offset_x), offset_y(Acc.offset_y),
      pixels_per_thread(pixels_per_thread), simd_width(simd_width),
      bh_start_left(0), bh_start_right(0), bh_start_top(0), bh_start_bottom(0),
      bh_fall_back(0) {}

inline hipacc_smem_info::hipacc_smem_info(int size_x, int size_y, int pixel_size)
    : size_x(size_x), size_y(size_y), pixel_size(pixel_size) {}

inline void HipaccPyramidTraversor::pushFunc(const std::function<void()> *f) {
  hipaccTraverseFunc.push_back(f);
}

inline void HipaccPyramidTraversor::popFunc() { hipaccTraverseFunc.pop_back(); }

inline const std::function<void()> HipaccPyramidTraversor::getLastFunc() {
  return *hipaccTraverseFunc.back();
}

inline void HipaccPyramidTraversor::pushPyramids(std::vector<HipaccPyramid *> &pyrs) {
  hipaccPyramids.push_back(pyrs);
}

inline void HipaccPyramidTraversor::popPyramids() { hipaccPyramids.pop_back(); }

inline std::vector<HipaccPyramid *> HipaccPyramidTraversor::getLastPyramids() {
  return hipaccPyramids.back();
}

inline bool HipaccPyramidTraversor::hasPyramids() { return !hipaccPyramids.empty(); }

inline void HipaccPyramidTraversor::hipaccTraverse(HipaccPyramid &p0,
                                            const std::function<void()> &func) {
  assert(p0.bind() && "Pyramid already bound to another traversal.");

  std::vector<HipaccPyramid *> pyrs;
  pyrs.push_back(&p0);

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  p0.unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(HipaccPyramid &p0,
                                            HipaccPyramid &p1,
                                            const std::function<void()> &func) {
  assert(p0.depth() == p1.depth() && "Pyramid depths do not match.");

  assert(p0.bind() && "Pyramid already bound to another traversal.");
  assert(p1.bind() && "Pyramid already bound to another traversal.");

  std::vector<HipaccPyramid *> pyrs;
  pyrs.push_back(&p0);
  pyrs.push_back(&p1);

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  p0.unbind();
  p1.unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(HipaccPyramid &p0,
                                            HipaccPyramid &p1,
                                            HipaccPyramid &p2,
                                            const std::function<void()> &func) {
  assert(p0.depth() == p1.depth() && p1.depth() == p2.depth() &&
         "Pyramid depths do not match.");

  assert(p0.bind() && "Pyramid already bound to another traversal.");
  assert(p1.bind() && "Pyramid already bound to another traversal.");
  assert(p2.bind() && "Pyramid already bound to another traversal.");

  std::vector<HipaccPyramid *> pyrs;
  pyrs.push_back(&p0);
  pyrs.push_back(&p1);
  pyrs.push_back(&p2);

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  p0.unbind();
  p1.unbind();
  p2.unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(HipaccPyramid &p0,
                                            HipaccPyramid &p1,
                                            HipaccPyramid &p2,
                                            HipaccPyramid &p3,
                                            const std::function<void()> &func) {
  assert(p0.depth() == p1.depth() && p1.depth() == p2.depth() &&
         p2.depth() == p3.depth() && "Pyramid depths do not match.");

  assert(p0.bind() && "Pyramid already bound to another traversal.");
  assert(p1.bind() && "Pyramid already bound to another traversal.");
  assert(p2.bind() && "Pyramid already bound to another traversal.");
  assert(p3.bind() && "Pyramid already bound to another traversal.");

  std::vector<HipaccPyramid *> pyrs;
  pyrs.push_back(&p0);
  pyrs.push_back(&p1);
  pyrs.push_back(&p2);
  pyrs.push_back(&p3);

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  p0.unbind();
  p1.unbind();
  p2.unbind();
  p3.unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(
    HipaccPyramid &p0, HipaccPyramid &p1, HipaccPyramid &p2, HipaccPyramid &p3,
    HipaccPyramid &p4, const std::function<void()> &func) {
  assert(p0.depth() == p1.depth() && p1.depth() == p2.depth() &&
         p2.depth() == p3.depth() && p3.depth() == p4.depth() &&
         "Pyramid depths do not match.");

  assert(p0.bind() && "Pyramid already bound to another traversal.");
  assert(p1.bind() && "Pyramid already bound to another traversal.");
  assert(p2.bind() && "Pyramid already bound to another traversal.");
  assert(p3.bind() && "Pyramid already bound to another traversal.");
  assert(p4.bind() && "Pyramid already bound to another traversal.");

  std::vector<HipaccPyramid *> pyrs;
  pyrs.push_back(&p0);
  pyrs.push_back(&p1);
  pyrs.push_back(&p2);
  pyrs.push_back(&p3);
  pyrs.push_back(&p4);

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  p0.unbind();
  p1.unbind();
  p2.unbind();
  p3.unbind();
  p4.unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(std::vector<HipaccPyramid *> pyrs,
                                            const std::function<void()> &func) {
  for (size_t i = 0; i < pyrs.size(); ++i) {
    if (i < pyrs.size() - 1) {
      assert(pyrs[i]->depth() == pyrs[i + 1]->depth() &&
             "Pyramid depths do not match.");
    }
    assert(pyrs[i]->bind() && "Pyramid already bound to another traversal.");
  }

  pushPyramids(pyrs);
  pushFunc(&func);
  (getLastFunc())();
  popFunc();
  popPyramids();

  for (auto pyr : pyrs)
    pyr->unbind();
}

inline void HipaccPyramidTraversor::hipaccTraverse(unsigned int loop,
                                            const std::function<void()> &func) {
  assert(hasPyramids() && "Traverse recursion called outside of traverse.");

  std::vector<HipaccPyramid *> pyrs = getLastPyramids();

  if (!pyrs.at(0)->is_bottom_level()) {
    for (auto pyr : pyrs)
      pyr->levelInc();

    for (size_t i = 0; i < loop; i++) {
      (getLastFunc())();
      if (i < loop - 1) {
        func();
      }
    }

    for (auto pyr : pyrs)
      pyr->levelDec();
  }
}

#endif // __HIPACC_BASE_STANDALONE_HPP__
