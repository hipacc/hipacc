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

//===--- ClassRepresentation.h - Representation of the DSL C++ classes ----===//
//
// This provides the internal representation of the compiler-known DSL C++
// classes.
//
//===----------------------------------------------------------------------===//

#include "hipacc/DSL/ClassRepresentation.h"

using namespace clang;
using namespace hipacc;


std::string HipaccImage::getTextureType() {
  QualType QT = type;
  if (type->isVectorType()) {
    QT = QT->getAs<VectorType>()->getElementType();
  }
  const BuiltinType *BT = QT->getAs<BuiltinType>();

  switch (BT->getKind()) {
    case BuiltinType::WChar_U:
    case BuiltinType::WChar_S:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::LongDouble:
    case BuiltinType::Void:
    case BuiltinType::Bool:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::Double:
    default:
      Ctx.getDiagnostics().Report(VD->getLocation(),
          Ctx.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
            "BuiltinType %0 of Image %1 not supported for textures.")) <<
        BT->getName(PrintingPolicy(Ctx.getLangOpts())) << VD->getName();
      assert(0 && "BuiltinType for texture not supported");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      return "CU_AD_FORMAT_SIGNED_INT8";
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      return "CU_AD_FORMAT_UNSIGNED_INT8";
    case BuiltinType::Short:
      return "CU_AD_FORMAT_SIGNED_INT16";
    case BuiltinType::Char16:
    case BuiltinType::UShort:
      return "CU_AD_FORMAT_UNSIGNED_INT16";
    case BuiltinType::Int:
      return "CU_AD_FORMAT_SIGNED_INT32";
    case BuiltinType::Char32:
    case BuiltinType::UInt:
      return "CU_AD_FORMAT_UNSIGNED_INT32";
    case BuiltinType::Float:
      return "CU_AD_FORMAT_FLOAT";
  }
}


std::string HipaccImage::getImageReadFunction() {
  QualType QT = type;
  if (type->isVectorType()) {
    QT = QT->getAs<VectorType>()->getElementType();
  }
  const BuiltinType *BT = QT->getAs<BuiltinType>();

  switch (BT->getKind()) {
    case BuiltinType::WChar_U:
    case BuiltinType::WChar_S:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::LongDouble:
    case BuiltinType::Void:
    case BuiltinType::Bool:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::Double:
    default:
      Ctx.getDiagnostics().Report(VD->getLocation(),
          Ctx.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
            "BuiltinType %0 of Image %1 not supported for Image objects.")) <<
        BT->getName(PrintingPolicy(Ctx.getLangOpts())) << VD->getName();
      assert(0 && "BuiltinType for Image object not supported");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::Int:
      return "read_imagei";
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::Char16:
    case BuiltinType::UShort:
    case BuiltinType::Char32:
    case BuiltinType::UInt:
      return "read_imageui";
    case BuiltinType::Float:
      return "read_imagef";
  }
}


void HipaccBoundaryCondition::setConstExpr(APValue &val, ASTContext &Ctx) {
  QualType QT = getImage()->getPixelQualType();

  bool isVecType = QT->isVectorType();
  if (isVecType) {
      QT = QT->getAs<VectorType>()->getElementType();
  }
  const BuiltinType *BT = QT->getAs<BuiltinType>();

  switch (BT->getKind()) {
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::LongDouble:
    case BuiltinType::Void:
    case BuiltinType::Bool:
    default:
      assert(0 && "BuiltinType for Boundary handling constant not supported.");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      if (isVecType) {
        SmallVector<Expr *, 16> initExprs;

        for (unsigned int I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(new (Ctx)
              CharacterLiteral(lane.getInt().getSExtValue(),
                CharacterLiteral::Ascii, QT, SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(),
            llvm::makeArrayRef(initExprs.data(), initExprs.size()),
            SourceLocation());
        constExpr->setType(getImage()->getPixelQualType());
      } else {
        constExpr = new (Ctx) CharacterLiteral(val.getInt().getSExtValue(),
            CharacterLiteral::Ascii, QT, SourceLocation());
      }
      break;
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
      if (isVecType) {
        SmallVector<Expr *, 16> initExprs;

        for (unsigned int I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(new (Ctx) IntegerLiteral(Ctx, lane.getInt(), QT,
                SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(),
            llvm::makeArrayRef(initExprs.data(), initExprs.size()),
            SourceLocation());
        constExpr->setType(getImage()->getPixelQualType());
      } else {
        constExpr = new (Ctx) IntegerLiteral(Ctx, val.getInt(), QT,
            SourceLocation());
      }
      break;
    case BuiltinType::Float:
    case BuiltinType::Double:
      if (isVecType) {
        SmallVector<Expr *, 16> initExprs;

        for (unsigned int I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(FloatingLiteral::Create(Ctx,
                llvm::APFloat(lane.getFloat()), false, QT, SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(),
            llvm::makeArrayRef(initExprs.data(), initExprs.size()),
            SourceLocation());
        constExpr->setType(getImage()->getPixelQualType());
      } else {
        constExpr = FloatingLiteral::Create(Ctx, llvm::APFloat(val.getFloat()),
            false, QT, SourceLocation());
      }
      break;
  }
}


void HipaccIterationSpace::createOutputAccessor() {
  // create Accessor for accessing the image associated with the IterationSpace
  // during ASTTranslate
  HipaccBoundaryCondition *BC = new HipaccBoundaryCondition(img, VD);
  BC->setSizeX(0);
  BC->setSizeY(0);
  BC->setBoundaryHandling(BOUNDARY_UNDEFINED);

  acc = new HipaccAccessor(BC, InterpolateNO, VD);
}


void HipaccKernel::calcSizes() {
  for (std::map<FieldDecl *, HipaccAccessor *>::iterator iter = imgMap.begin(),
      eiter=imgMap.end(); iter!=eiter; ++iter) {
    // only Accessors with proper border handling mode
    if (iter->second->getSizeX() > max_size_x &&
        iter->second->getBoundaryHandling()!=BOUNDARY_UNDEFINED)
      max_size_x = iter->second->getSizeX();
    if (iter->second->getSizeY() > max_size_y &&
        iter->second->getBoundaryHandling()!=BOUNDARY_UNDEFINED)
      max_size_y = iter->second->getSizeY();
    // including Accessors with UNDEFINED border handling mode
    if (iter->second->getSizeX() > max_size_x_undef)
      max_size_x_undef = iter->second->getSizeX();
    if (iter->second->getSizeY() > max_size_y_undef)
      max_size_y_undef = iter->second->getSizeY();
  }
}


struct sortOccMap {
  bool operator()(const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
    if (left.second < right.second) return false;
    if (right.second < left.second) return true;
    return left.first < right.first;
  }
};


void HipaccKernel::calcConfig() {
  std::vector<std::pair<unsigned int, float> > occVec;
  unsigned int num_threads = max_threads_per_warp;
  bool use_shared = false;

  while (num_threads <= max_threads_per_block) {
    // allocations per thread block limits
    int warps_per_block = (int)ceil((float)num_threads /
        (float)max_threads_per_warp);
    int registers_per_block;
    if (isAMDGPU()) {
      // for AMD assume simple allocation strategy
      registers_per_block = warps_per_block * num_reg * max_threads_per_warp;
    } else {
      switch (allocation_granularity) {
        case BLOCK:
          // allocation in steps of two
          registers_per_block = (int)ceil((float)warps_per_block /
              (float)warp_register_alloc_size) * warp_register_alloc_size *
            num_reg * max_threads_per_warp;
          registers_per_block = (int)ceil((float)registers_per_block /
              (float)register_alloc_size) * register_alloc_size;
          break;
        case WARP:
          registers_per_block = (int)ceil((float)(num_reg *
                max_threads_per_warp) / (float)register_alloc_size) *
            register_alloc_size;
          registers_per_block *= (int)ceil((float)warps_per_block /
              (float)warp_register_alloc_size) * warp_register_alloc_size;
          break;
      }
    }

    unsigned int smem_used = 0;
    bool skip_config = false;
    // calculate shared memory usage for pixels staged to shared memory
    for (unsigned int i=0; i<KC->getNumImages(); i++) {
      HipaccAccessor *Acc = getImgFromMapping(KC->getImgFields().data()[i]);
      if (useLocalMemory(Acc)) {
        // check if the configuration suits our assumptions about shared memory
        if (num_threads % 32 == 0) {
          // fixed shared memory for x: 3*BSX
          int size_x = 32;
          if (Acc->getSizeX() > 1) {
            size_x *= 3;
          }
          // add padding to avoid bank conflicts
          size_x += 1;

          // size_y = ceil((PPT*BSY+SX-1)/BSY)
          int threads_y = num_threads/32;
          int size_y = (int)ceilf((float)(getPixelsPerThread()*threads_y +
                Acc->getSizeY()-1)/(float)threads_y) * threads_y;
          smem_used += size_x*size_y * Acc->getImage()->getPixelSize();
          use_shared = true;
        } else {
          skip_config = true;
        }
      }
    }

    if (skip_config || smem_used > max_total_shared_memory) {
      num_threads += max_threads_per_warp;
      continue;
    }

    int shared_memory_per_block = (int)ceil((float)smem_used /
        (float)shared_memory_alloc_size) * shared_memory_alloc_size;

    // maximum thread blocks per multiprocessor
    int lim_by_max_warps = std::min(max_blocks_per_multiprocessor, (unsigned
          int)floor((float)max_warps_per_multiprocessor /
            (float)warps_per_block));
    int lim_by_reg, lim_by_smem;
    if (num_reg > max_register_per_thread) {
      lim_by_reg = 0;
    } else {
      if (num_reg > 0) {
        lim_by_reg = (int)floor((float)max_total_registers /
            (float)registers_per_block);
      } else {
        lim_by_reg = max_blocks_per_multiprocessor;
      }
    }
    if (smem_used > 0) {
      lim_by_smem = (int)floor((float)max_total_shared_memory /
          (float)shared_memory_per_block);
    } else {
      lim_by_smem = max_blocks_per_multiprocessor;
    }

    // calculate GPU occupancy
    int active_thread_blocks_per_multiprocessor = std::min(std::min(lim_by_max_warps, lim_by_reg), lim_by_smem);
    if (active_thread_blocks_per_multiprocessor > 0) max_threads_for_kernel = num_threads;
    int active_warps_per_multiprocessor = active_thread_blocks_per_multiprocessor * warps_per_block;
    //int active_threads_per_multiprocessor = active_thread_blocks_per_multiprocessor * num_threads;
    float occupancy = (float)active_warps_per_multiprocessor/(float)max_warps_per_multiprocessor;
    //int max_simultaneous_blocks_per_GPU = active_thread_blocks_per_multiprocessor*max_multiprocessors_per_GPU;

    occVec.push_back(std::pair<int, float>(num_threads, occupancy));
    num_threads += max_threads_per_warp;
  }

  // sort configurations according to occupancy and number of threads
  std::sort(occVec.begin(), occVec.end(), sortOccMap());
  std::vector<std::pair<unsigned int, float> >::iterator iter;

  // calculate (optimal) kernel configuration from the kernel window sizes and
  // ignore the limitation of maximal threads per block
  unsigned int num_threads_x_opt = max_threads_per_warp;
  unsigned int num_threads_y_opt = 1;
  while (num_threads_x_opt < max_size_x>>1)
    num_threads_x_opt += max_threads_per_warp;
  while (num_threads_y_opt*getPixelsPerThread() < max_size_y>>1)
    num_threads_y_opt += 1;

  // Heuristic:
  // 0) maximize occupancy (e.g. to hide instruction latency
  // 1) - minimize #threads for border handling (e.g. prefer y over x)
  //    - prefer x over y when no border handling is necessary
  llvm::errs() << "\nCalculating kernel configuration for " << kernelName << "\n";
  llvm::errs() << "\toptimal configuration: " << num_threads_x_opt << "x" <<
    num_threads_y_opt << "(x" << getPixelsPerThread() << ")\n";
  for (iter=occVec.begin(); iter<occVec.end(); ++iter) {
    std::pair<unsigned int, float> occMap = *iter;
    llvm::errs() << "\t" << occMap.first << " threads:\t" << occMap.second << "\t";

    if (use_shared) {
      // start with warp_size or num_threads_x_opt if possible
      unsigned int num_threads_x = 32;
      unsigned int num_threads_y = occMap.first / num_threads_x;
      llvm::errs() << " -> " << num_threads_x << "x" << num_threads_y;
    } else {
      // make difference if we create border handling or not
      if (max_size_y > 1) {
        // start with warp_size or num_threads_x_opt if possible
        unsigned int num_threads_x = max_threads_per_warp;
        if (occMap.first >= num_threads_x_opt && occMap.first % num_threads_x_opt == 0) {
          num_threads_x = num_threads_x_opt;
        }
        unsigned int num_threads_y = occMap.first / num_threads_x;
        llvm::errs() << " -> " << num_threads_x << "x" << num_threads_y;
      } else {
        // use all threads for x direction
        llvm::errs() << " -> " << occMap.first << "x1";
      }
    }
    llvm::errs() << "(x" << getPixelsPerThread() << ")\n";
  }


  // fall back to default or user specified configuration
  unsigned int num_blocks_bh_x, num_blocks_bh_y;
  if (occVec.empty() || options.useKernelConfig()) {
    setDefaultConfig();
    num_blocks_bh_x = max_size_x<=1?0:(unsigned int)ceil((float)(max_size_x>>1) / (float)num_threads_x);
    num_blocks_bh_y = max_size_y<=1?0:(unsigned int)ceil((float)(max_size_y>>1) / (float)(num_threads_y*getPixelsPerThread()));
    llvm::errs() << "Using default configuration " << num_threads_x << "x"
                 << num_threads_y << " for kernel '" << kernelName << "'\n";
  } else {
    // start with first configuration
    iter = occVec.begin();
    std::pair<unsigned int, float> occMap = *iter;

    if (use_shared) {
      num_threads_x = 32;
      num_threads_y = occMap.first / num_threads_x;
    } else {
      // make difference if we create border handling or not
      if (max_size_y > 1) {
        // start with warp_size or num_threads_x_opt if possible
        num_threads_x = max_threads_per_warp;
        if (occMap.first >= num_threads_x_opt && occMap.first % num_threads_x_opt == 0) {
          num_threads_x = num_threads_x_opt;
        }
        num_threads_y = occMap.first / num_threads_x;
      } else {
        // use all threads for x direction
        num_threads_x = occMap.first;
        num_threads_y = 1;
      }
    }

    // estimate block required for border handling - the exact number depends on
    // offsets and is not known at compile time
    num_blocks_bh_x = max_size_x<=1?0:(unsigned int)ceil((float)(max_size_x>>1) / (float)num_threads_x);
    num_blocks_bh_y = max_size_y<=1?0:(unsigned int)ceil((float)(max_size_y>>1) / (float)(num_threads_y*getPixelsPerThread()));

    if ((max_size_y > 1) || num_threads_x != num_threads_x_opt || num_threads_y != num_threads_y_opt) {
      //std::vector<std::pair<unsigned int, float> >::iterator iter_n = occVec.begin()

      // look-ahead if other configurations match better
      while (++iter<occVec.end()) {
        std::pair<unsigned int, float> occMapNext = *iter;
        // bail out on lower occupancy
        if (occMapNext.second < occMap.second) break;

        // start with warp_size or num_threads_x_opt if possible
        unsigned int num_threads_x_tmp = max_threads_per_warp;
        if (occMapNext.first >= num_threads_x_opt && occMapNext.first % num_threads_x_opt == 0)
          num_threads_x_tmp = num_threads_x_opt;
        unsigned int num_threads_y_tmp = occMapNext.first / num_threads_x_tmp;

        // block required for border handling
        unsigned int num_blocks_bh_x_tmp = max_size_x<=1?0:(unsigned int)ceil((float)(max_size_x>>1) / (float)num_threads_x_tmp);
        unsigned int num_blocks_bh_y_tmp = max_size_y<=1?0:(unsigned int)ceil((float)(max_size_y>>1) / (float)(num_threads_y_tmp*getPixelsPerThread()));

        // use new configuration if we save blocks for border handling
        if (num_blocks_bh_x_tmp+num_blocks_bh_y_tmp < num_blocks_bh_x+num_blocks_bh_y) {
          num_threads_x = num_threads_x_tmp;
          num_threads_y = num_threads_y_tmp;
          num_blocks_bh_x = num_blocks_bh_x_tmp;
          num_blocks_bh_y = num_blocks_bh_y_tmp;
        }
      }
    }
    llvm::errs() << "Using configuration " << num_threads_x << "x"
                 << num_threads_y << "(occupancy: " << occMap.second
                 << ") for kernel '" << kernelName << "'\n";
  }

  llvm::errs() << "\t Blocks required for border handling: " <<
    num_blocks_bh_x << "x" << num_blocks_bh_y << "\n\n";
}

void HipaccKernel::setDefaultConfig() {
  max_threads_for_kernel = max_threads_per_block;
  num_threads_x = default_num_threads_x;
  num_threads_y = default_num_threads_y;
}

void HipaccKernel::addParam(QualType QT1, QualType QT2, QualType QT3,
    std::string typeC, std::string typeO, std::string name, FieldDecl *fd) {
  argTypesCUDA.push_back(QT1);
  argTypesOpenCL.push_back(QT2);
  argTypesC.push_back(QT3);

  argTypeNamesCUDA.push_back(typeC);
  argTypeNamesOpenCL.push_back(typeO);

  deviceArgNames.push_back(name);
  deviceArgFields.push_back(fd);
}

void HipaccKernel::createArgInfo() {
  if (argTypesCUDA.size()) return;

  SmallVector<HipaccKernelClass::argumentInfo, 16> arguments =
    KC->arguments;

  // normal parameters
  for (unsigned int i=0; i<KC->getNumArgs(); i++) {
    FieldDecl *FD = arguments.data()[i].field;
    QualType QT = arguments.data()[i].type;
    std::string name = arguments.data()[i].name;
    QualType QTtmp;

    switch (arguments.data()[i].kind) {
      case HipaccKernelClass::Normal:
        addParam(QT, QT, QT, QT.getAsString(), QT.getAsString(), name, FD);

        break;
      case HipaccKernelClass::IterationSpace:
        // add output image
        addParam(Ctx.getPointerType(QT), Ctx.getPointerType(QT),
            Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                  4096), ArrayType::Normal, false)),
            Ctx.getPointerType(QT).getAsString(), "cl_mem", name, NULL);

        break;
      case HipaccKernelClass::Image:
        // for textures use no pointer type
        if (useTextureMemory(getImgFromMapping(FD)) &&
            KC->getImgAccess(FD) == READ_ONLY &&
            // no texture required for __ldg() intrinsic
            !(useTextureMemory(getImgFromMapping(FD)) == Ldg)) {
          addParam(Ctx.getPointerType(QT), Ctx.getPointerType(QT),
              Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                    4096), ArrayType::Normal, false)), QT.getAsString(),
              "cl_mem", name, FD);
        } else {
          addParam(Ctx.getPointerType(QT), Ctx.getPointerType(QT),
              Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                    4096), ArrayType::Normal, false)),
              Ctx.getPointerType(QT).getAsString(), "cl_mem", name, FD);
        }

        // add types for image width/height plus stride
        addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
            Ctx.getConstType(Ctx.IntTy),
            Ctx.getConstType(Ctx.IntTy).getAsString(),
            Ctx.getConstType(Ctx.IntTy).getAsString(), name + "_width", NULL);
        addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
            Ctx.getConstType(Ctx.IntTy),
            Ctx.getConstType(Ctx.IntTy).getAsString(),
            Ctx.getConstType(Ctx.IntTy).getAsString(), name + "_height", NULL);

        // stride
        if (options.emitPadding() || getImgFromMapping(FD)->isCrop()) {
          addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy).getAsString(),
              Ctx.getConstType(Ctx.IntTy).getAsString(), name + "_stride",
              NULL);
        }

        // offset_x, offset_y
        if (getImgFromMapping(FD)->isCrop()) {
          addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy).getAsString(),
              Ctx.getConstType(Ctx.IntTy).getAsString(), name + "_offset_x",
              NULL);
          addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy),
              Ctx.getConstType(Ctx.IntTy).getAsString(),
              Ctx.getConstType(Ctx.IntTy).getAsString(), name + "_offset_y",
              NULL);
        }

        break;
      case HipaccKernelClass::Mask:
        QTtmp = Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                getMaskFromMapping(FD)->getSizeX()), ArrayType::Normal, false));
        // OpenCL non-constant mask
        if (!getMaskFromMapping(FD)->isConstant()) {
          addParam(QTtmp, Ctx.getPointerType(QT), QTtmp, QTtmp.getAsString(),
              Ctx.getPointerType(QT).getAsString(), name, FD);
        } else {
          addParam(QTtmp, QTtmp, QTtmp, QTtmp.getAsString(),
              QTtmp.getAsString(), name, FD);
        }

        break;
    }
  }

  // is_stride
  addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
      Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
      Ctx.getConstType(Ctx.IntTy).getAsString(), "is_stride", NULL);

  // is_width, is_height
  addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
      Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
      Ctx.getConstType(Ctx.IntTy).getAsString(), "is_width", NULL);
  addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
      Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
      Ctx.getConstType(Ctx.IntTy).getAsString(), "is_height", NULL);

  // is_offset_x, is_offset_y
  if (iterationSpace->isCrop()) {
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "is_offset_x", NULL);
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "is_offset_y", NULL);
  }

  // bh_start_left
  if (getMaxSizeX() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "bh_start_left", NULL);
  }
  // bh_start_right: always emit bh_start_right for iteration spaces not being a
  // multiple of the block size
  addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
      Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
      Ctx.getConstType(Ctx.IntTy).getAsString(), "bh_start_right", NULL);
  // bh_start_top
  if (getMaxSizeY() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "bh_start_top", NULL);
  }
  // bh_start_bottom: emit bh_start_bottom in case iteration space is not a
  // multiple of the block size
  if (getNumThreadsY()>1 || getPixelsPerThread()>1 || getMaxSizeY() ||
      options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "bh_start_bottom", NULL);
  }
  // bh_fall_back
  if (getMaxSizeX() || getMaxSizeY() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy),
        Ctx.getConstType(Ctx.IntTy), Ctx.getConstType(Ctx.IntTy).getAsString(),
        Ctx.getConstType(Ctx.IntTy).getAsString(), "bh_fall_back", NULL);
  }
}


void HipaccKernel::createHostArgInfo(ArrayRef<Expr *> hostArgs, std::string
    &hostLiterals, unsigned int &literalCount) {
  if (hostArgNames.size()) hostArgNames.clear();

  for (unsigned int i=0; i<KC->getNumArgs(); i++) {
    FieldDecl *FD = KC->arguments.data()[i].field;

    std::string Str;
    llvm::raw_string_ostream SS(Str);

    switch (KC->arguments.data()[i].kind) {
      case HipaccKernelClass::Normal:
        hostArgs[i]->printPretty(SS, 0, PrintingPolicy(Ctx.getLangOpts()));

        if (isa<DeclRefExpr>(hostArgs[i]->IgnoreParenCasts())) {
          hostArgNames.push_back(SS.str());
        } else {
          // get the text string for the argument and create a temporary
          std::stringstream LSS;
          LSS << "_tmpLiteral" << literalCount;
          literalCount++;

          hostLiterals += hostArgs[i]->IgnoreParenCasts()->getType().getAsString();
          hostLiterals += " ";
          hostLiterals += LSS.str();
          hostLiterals += " = ";
          hostLiterals += SS.str();
          hostLiterals += ";\n    ";
          hostArgNames.push_back(LSS.str());
        }

        break;
      case HipaccKernelClass::IterationSpace:
        // output image
        hostArgNames.push_back(iterationSpace->getName() + ".img");

        break;
      case HipaccKernelClass::Image: {
        // image
        HipaccAccessor *Acc = getImgFromMapping(FD);
        hostArgNames.push_back(Acc->getName() + ".img");

        // width, height
        hostArgNames.push_back(Acc->getName() + ".width");
        hostArgNames.push_back(Acc->getName() + ".height");

        // stride
        if (options.emitPadding() || Acc->isCrop()) {
          hostArgNames.push_back(Acc->getName() + ".img.stride");
        }

        // offset_x, offset_y
        if (Acc->isCrop()) {
          hostArgNames.push_back(Acc->getName() + ".offset_x");
          hostArgNames.push_back(Acc->getName() + ".offset_y");
        }

        break;
        }
      case HipaccKernelClass::Mask:
        hostArgNames.push_back(getMaskFromMapping(FD)->getName() + ".mem");

        break;
    }
  }
  // is_stride
  hostArgNames.push_back(iterationSpace->getName() + ".img.stride");

  // is_width, is_height
  hostArgNames.push_back(iterationSpace->getName() + ".width");
  hostArgNames.push_back(iterationSpace->getName() + ".height");

  // is_offset_x, is_offset_y
  if (iterationSpace->isCrop()) {
    hostArgNames.push_back(iterationSpace->getName() + ".offset_x");
    hostArgNames.push_back(iterationSpace->getName() + ".offset_y");
  }

  setInfoStr();
  // bh_start_left, bh_start_right
  if (getMaxSizeX() || options.exploreConfig()) {
    hostArgNames.push_back(getInfoStr() + ".bh_start_left");
  }
  hostArgNames.push_back(getInfoStr() + ".bh_start_right");
  // bh_start_top, bh_start_bottom
  if (getMaxSizeY() || options.exploreConfig()) {
    hostArgNames.push_back(getInfoStr() + ".bh_start_top");
  }
  if (getNumThreadsY()>1 || getPixelsPerThread()>1 || getMaxSizeY() ||
      options.exploreConfig()) {
    hostArgNames.push_back(getInfoStr() + ".bh_start_bottom");
  }
  // bh_fall_back
  if (getMaxSizeX() || getMaxSizeY() || options.exploreConfig()) {
    hostArgNames.push_back(getInfoStr() + ".bh_fall_back");
  }
}

// vim: set ts=2 sw=2 sts=2 et ai:

