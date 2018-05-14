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

#include <llvm/Support/Format.h>

#ifdef USE_JIT_ESTIMATE
#include <cuda_occupancy.h>
#endif

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


void HipaccBoundaryCondition::setConstVal(APValue &val, ASTContext &Ctx) {
  QualType QT = getImage()->getType();

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

        for (size_t I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(new (Ctx)
              CharacterLiteral(lane.getInt().getSExtValue(),
                CharacterLiteral::Ascii, QT, SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(), initExprs,
            SourceLocation());
        constExpr->setType(getImage()->getType());
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

        for (size_t I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(new (Ctx) IntegerLiteral(Ctx, lane.getInt(), QT,
                SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(), initExprs,
            SourceLocation());
        constExpr->setType(getImage()->getType());
      } else {
        constExpr = new (Ctx) IntegerLiteral(Ctx, val.getInt(), QT,
            SourceLocation());
      }
      break;
    case BuiltinType::Float:
    case BuiltinType::Double:
      if (isVecType) {
        SmallVector<Expr *, 16> initExprs;

        for (size_t I=0, N=val.getVectorLength(); I!=N; ++I) {
          APValue lane = val.getVectorElt(I);
          initExprs.push_back(FloatingLiteral::Create(Ctx,
                llvm::APFloat(lane.getFloat()), false, QT, SourceLocation()));
        }

        constExpr = new (Ctx) InitListExpr(Ctx, SourceLocation(), initExprs,
            SourceLocation());
        constExpr->setType(getImage()->getType());
      } else {
        constExpr = FloatingLiteral::Create(Ctx, llvm::APFloat(val.getFloat()),
            false, QT, SourceLocation());
      }
      break;
  }
}


void HipaccKernel::calcSizes() {
  for (auto map : imgMap) {
    // only Accessors with proper border handling mode
    if (map.second->getSizeX() > max_size_x &&
        map.second->getBoundaryMode()!=Boundary::UNDEFINED)
      max_size_x = map.second->getSizeX();
    if (map.second->getSizeY() > max_size_y &&
        map.second->getBoundaryMode()!=Boundary::UNDEFINED)
      max_size_y = map.second->getSizeY();
    // including Accessors with UNDEFINED border handling mode
    if (map.second->getSizeX() > max_size_x_undef) max_size_x_undef =
      map.second->getSizeX();
    if (map.second->getSizeY() > max_size_y_undef)
      max_size_y_undef = map.second->getSizeY();
  }
}


struct sortOccMap {
  bool operator()(const std::pair<unsigned, float> &left, const std::pair<unsigned, float> &right) {
    if (left.second < right.second) return false;
    if (right.second < left.second) return true;
    return left.first < right.first;
  }
};


void HipaccKernel::calcConfig() {
  #ifdef USE_JIT_ESTIMATE
  std::vector<std::pair<unsigned, float>> occVec;
  unsigned num_threads = max_threads_per_warp;
  bool use_shared = false;

  while (num_threads <= max_threads_per_block) {
    unsigned smem_used = 0;
    bool skip_config = false;
    // calculate shared memory usage for pixels staged to shared memory
    for (auto img : KC->getImgFields()) {
      HipaccAccessor *Acc = getImgFromMapping(img);
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
    max_threads_for_kernel = num_threads;

    int major = getTargetCC()/10;
    int minor = getTargetCC()%10;
    if (isAMDGPU()) {
      // set architecture to "Fermi"
      major = 2;
      minor = 0;
    }

    cudaOccDeviceState dev_state;
    cudaOccDeviceProp dev_props;
    cudaOccFuncAttributes fun_attrs;

    dev_props.computeMajor = major;
    dev_props.computeMinor = minor;
    dev_props.maxThreadsPerBlock = max_threads_per_block;
    dev_props.maxThreadsPerMultiprocessor = max_threads_per_multiprocessor;
    dev_props.regsPerBlock = max_total_registers;
    dev_props.regsPerMultiprocessor = max_total_registers;
    dev_props.warpSize = max_threads_per_warp;
    dev_props.sharedMemPerBlock = max_total_shared_memory;
    dev_props.sharedMemPerMultiprocessor = max_total_shared_memory;
    dev_props.numSms = 23;
    fun_attrs.maxThreadsPerBlock = max_threads_per_block;
    fun_attrs.numRegs = num_reg;
    fun_attrs.sharedSizeBytes = smem_used;

    size_t dynamic_smem_bytes = 0;
    cudaOccResult fun_occ;
    cudaOccMaxActiveBlocksPerMultiprocessor(&fun_occ, &dev_props, &fun_attrs, &dev_state, num_threads, dynamic_smem_bytes);
    int active_blocks = fun_occ.activeBlocksPerMultiprocessor;
    int min_grid_size, opt_block_size;
    cudaOccMaxPotentialOccupancyBlockSize(&min_grid_size, &opt_block_size, &dev_props, &fun_attrs, &dev_state, 0, dynamic_smem_bytes);
    int active_warps = active_blocks * (num_threads/max_threads_per_warp);

    // re-compute with optimal block size
    cudaOccMaxActiveBlocksPerMultiprocessor(&fun_occ, &dev_props, &fun_attrs, &dev_state, opt_block_size, dynamic_smem_bytes);
    int max_blocks = std::min(fun_occ.blockLimitRegs, std::min(fun_occ.blockLimitSharedMem, std::min(fun_occ.blockLimitWarps, fun_occ.blockLimitBlocks)));
    int max_warps = max_blocks * (opt_block_size/max_threads_per_warp);
    float occupancy = (float)active_warps/(float)max_warps;

    occVec.emplace_back(num_threads, occupancy);
    num_threads += max_threads_per_warp;
  }

  // sort configurations according to occupancy and number of threads
  std::sort(occVec.begin(), occVec.end(), sortOccMap());

  // calculate (optimal) kernel configuration from the kernel window sizes and
  // ignore the limitation of maximal threads per block
  unsigned num_threads_x_opt = max_threads_per_warp;
  unsigned num_threads_y_opt = 1;
  while (num_threads_x_opt < max_size_x>>1)
    num_threads_x_opt += max_threads_per_warp;
  while (num_threads_y_opt*getPixelsPerThread() < max_size_y>>1)
    num_threads_y_opt += 1;

  // Heuristic:
  // 0) maximize occupancy (e.g. to hide instruction latency
  // 1) - minimize #threads for border handling (e.g. prefer y over x)
  //    - prefer x over y when no border handling is necessary
  llvm::errs() << "\nCalculating kernel configuration for " << kernelName << "\n";
  llvm::errs() << "  optimal configuration: " << num_threads_x_opt << "x"
               << num_threads_y_opt << "(x" << getPixelsPerThread() << ")\n";
  for (auto map : occVec) {
    llvm::errs() << "    " << llvm::format("%5d", map.first) << " threads:"
                 << " occupancy = " << llvm::format("%*.2f", 6, map.second*100) << "%";

    unsigned num_threads_x = max_threads_per_warp;
    unsigned num_threads_y = 1;

    if (use_shared) {
      // use warp_size x N
      num_threads_y = map.first / num_threads_x;
    } else {
      // do we need border handling?
      if (max_size_y > 1) {
        // use N x M
        if (map.first >= num_threads_x_opt && map.first % num_threads_x_opt == 0) {
          num_threads_x = num_threads_x_opt;
        }
        num_threads_y = map.first / num_threads_x;
      } else {
        // use num_threads x 1
        num_threads_x = map.first;
      }
    }
    llvm::errs() << " -> " << llvm::format("%4d", num_threads_x)
                 << "x" << llvm::format("%-2d", num_threads_y)
                 << "(x" << getPixelsPerThread() << ")\n";
  }


  // fall back to default or user specified configuration
  unsigned num_blocks_bh_x, num_blocks_bh_y;
  if (occVec.empty() || options.useKernelConfig()) {
    setDefaultConfig();
    num_blocks_bh_x = max_size_x<=1?0:(unsigned)ceil((float)(max_size_x>>1) / (float)num_threads_x);
    num_blocks_bh_y = max_size_y<=1?0:(unsigned)ceil((float)(max_size_y>>1) / (float)(num_threads_y*getPixelsPerThread()));
    llvm::errs() << "Using default configuration " << num_threads_x << "x"
                 << num_threads_y << " for kernel '" << kernelName << "'\n";
  } else {
    // start with first configuration
    auto map = occVec.begin();

    num_threads_x = max_threads_per_warp;
    num_threads_y = 1;

    if (use_shared) {
      // use warp_size x N
      num_threads_y = map->first / num_threads_x;
    } else {
      // do we need border handling?
      if (max_size_y > 1) {
        // use N x M
        if (map->first >= num_threads_x_opt && map->first % num_threads_x_opt == 0) {
          num_threads_x = num_threads_x_opt;
        }
        num_threads_y = map->first / num_threads_x;
      } else {
        // use num_threads x 1
        num_threads_x = map->first;
      }
    }

    // estimate block required for border handling - the exact number depends on
    // offsets and is not known at compile time
    num_blocks_bh_x = max_size_x<=1?0:(unsigned)ceil((float)(max_size_x>>1) / (float)num_threads_x);
    num_blocks_bh_y = max_size_y<=1?0:(unsigned)ceil((float)(max_size_y>>1) / (float)(num_threads_y*getPixelsPerThread()));

    if ((max_size_y > 1) || num_threads_x != num_threads_x_opt || num_threads_y != num_threads_y_opt) {
      // look-ahead if other configurations match better
      auto map_next = occVec.begin();
      while (++map_next<occVec.end()) {
        // bail out on lower occupancy
        if (map_next->second < map->second) break;

        // start with warp_size or num_threads_x_opt if possible
        unsigned num_threads_x_tmp = max_threads_per_warp;
        if (map_next->first >= num_threads_x_opt && map_next->first % num_threads_x_opt == 0)
          num_threads_x_tmp = num_threads_x_opt;
        unsigned num_threads_y_tmp = map_next->first / num_threads_x_tmp;

        // block required for border handling
        unsigned num_blocks_bh_x_tmp = max_size_x<=1?0:(unsigned)ceil((float)(max_size_x>>1) / (float)num_threads_x_tmp);
        unsigned num_blocks_bh_y_tmp = max_size_y<=1?0:(unsigned)ceil((float)(max_size_y>>1) / (float)(num_threads_y_tmp*getPixelsPerThread()));

        // use new configuration if we save blocks for border handling
        if (num_blocks_bh_x_tmp+num_blocks_bh_y_tmp < num_blocks_bh_x+num_blocks_bh_y) {
          num_threads_x = num_threads_x_tmp;
          num_threads_y = num_threads_y_tmp;
          num_blocks_bh_x = num_blocks_bh_x_tmp;
          num_blocks_bh_y = num_blocks_bh_y_tmp;
        }
      }
    }
    llvm::errs() << "Using configuration " << num_threads_x << "x" << num_threads_y
                 << "(occupancy = " << llvm::format("%*.2f", 6, map->second*100)
                 << ") for kernel '" << kernelName << "'\n";
  }

  llvm::errs() << "  Blocks required for border handling: "
               << num_blocks_bh_x << "x" << num_blocks_bh_y << "\n\n";
  #else
  setDefaultConfig();
  #endif
}

void HipaccKernel::setDefaultConfig() {
  max_threads_for_kernel = max_threads_per_block;
  num_threads_x = default_num_threads_x;
  num_threads_y = default_num_threads_y;
}

void HipaccKernel::addParam(QualType QT1, QualType QT2, QualType QT3,
    std::string typeC, std::string typeO, std::string name, FieldDecl *fd) {
  switch (options.getTargetLang()) {
    case Language::C99:          argTypes.push_back(QT3);
                                 argTypeNames.push_back(typeC); break;
    case Language::CUDA:         argTypes.push_back(QT1);
                                 argTypeNames.push_back(typeC); break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:    argTypes.push_back(QT2);
                                 argTypeNames.push_back(typeO); break;
    case Language::Renderscript:
    case Language::Filterscript: argTypes.push_back(QT2);
                                 argTypeNames.push_back(typeC); break;
  }

  if (options.fuseKernels() && OptmOpt == OptimizationOption::KERNEL_FUSE) {
    deviceArgNames.push_back(name + "_" + kernelName);
  }
  else {
    deviceArgNames.push_back(name);
  }
  deviceArgFields.push_back(fd);
}

void HipaccKernel::createArgInfo() {
  if (argTypes.size()) return;

  // normal parameters
  for (auto arg : KC->getMembers()) {
    QualType QT = arg.type;
    QualType QTtmp;

    switch (arg.kind) {
      case HipaccKernelClass::FieldKind::Normal:
        addParam(QT, arg.name, arg.field);

        break;
      case HipaccKernelClass::FieldKind::IterationSpace:
      case HipaccKernelClass::FieldKind::Image:
        // for textures use no pointer type
        if (useTextureMemory(getImgFromMapping(arg.field)) != Texture::None &&
            useTextureMemory(getImgFromMapping(arg.field)) != Texture::Ldg) {
          addParam(Ctx.getPointerType(QT), Ctx.getPointerType(QT),
              Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                    getImgFromMapping(arg.field)->getImage()->getSizeX()),
                  ArrayType::Normal, false)), QT.getAsString(), "cl_mem",
              arg.name, arg.field);
        } else {
          addParam(Ctx.getPointerType(QT), Ctx.getPointerType(QT),
              Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                    getImgFromMapping(arg.field)->getImage()->getSizeX()),
                  ArrayType::Normal, false)),
              Ctx.getPointerType(QT).getAsString(), "cl_mem", arg.name,
              arg.field);
        }

        // add types for image width/height plus stride
        addParam(Ctx.getConstType(Ctx.IntTy), arg.name + "_width", nullptr);
        addParam(Ctx.getConstType(Ctx.IntTy), arg.name + "_height", nullptr);

        // stride
        if (options.emitPadding() || getImgFromMapping(arg.field)->isCrop()) {
          addParam(Ctx.getConstType(Ctx.IntTy), arg.name + "_stride", nullptr);
        }

        // offset_x, offset_y
        if (getImgFromMapping(arg.field)->isCrop()) {
          addParam(Ctx.getConstType(Ctx.IntTy), arg.name + "_offset_x",
              nullptr);
          addParam(Ctx.getConstType(Ctx.IntTy), arg.name + "_offset_y",
              nullptr);
        }

        break;
      case HipaccKernelClass::FieldKind::Mask:
        QTtmp = Ctx.getPointerType(Ctx.getConstantArrayType(QT, llvm::APInt(32,
                getMaskFromMapping(arg.field)->getSizeX()), ArrayType::Normal,
              false));
        // OpenCL non-constant mask
        if (!getMaskFromMapping(arg.field)->isConstant()) {
          addParam(QTtmp, Ctx.getPointerType(QT), QTtmp, QTtmp.getAsString(),
              Ctx.getPointerType(QT).getAsString(), arg.name, arg.field);
        } else {
          addParam(QTtmp, arg.name, arg.field);
        }

        break;
    }
  }

  // bh_start_left
  if (getMaxSizeX() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), "bh_start_left", nullptr);
  }
  // bh_start_right: always emit bh_start_right for iteration spaces not being a
  // multiple of the block size
  addParam(Ctx.getConstType(Ctx.IntTy), "bh_start_right", nullptr);
  // bh_start_top
  if (getMaxSizeY() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), "bh_start_top", nullptr);
  }
  // bh_start_bottom: emit bh_start_bottom in case iteration space is not a
  // multiple of the block size
  if (getNumThreadsY()>1 || getPixelsPerThread()>1 || getMaxSizeY() ||
      options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), "bh_start_bottom", nullptr);
  }
  // bh_fall_back
  if (getMaxSizeX() || getMaxSizeY() || options.exploreConfig()) {
    addParam(Ctx.getConstType(Ctx.IntTy), "bh_fall_back", nullptr);
  }
}


void HipaccKernel::createHostArgInfo(ArrayRef<Expr *> hostArgs, std::string
    &hostLiterals, unsigned &literalCount) {
  if (hostArgNames.size()) hostArgNames.clear();

  size_t i = 0;
  for (auto arg : KC->getMembers()) {
    switch (arg.kind) {
      case HipaccKernelClass::FieldKind::Normal: {
        std::string Str;
        llvm::raw_string_ostream SS(Str);
        hostArgs[i]->printPretty(SS, 0, PrintingPolicy(Ctx.getLangOpts()));

        if (isa<DeclRefExpr>(hostArgs[i]->IgnoreParenCasts())) {
          hostArgNames.push_back(SS.str());
        } else {
          // get the text string for the argument and create a temporary
          std::string tmp_lit("_tmpLiteral" + std::to_string(literalCount++));

          // use type of kernel class
          hostLiterals += arg.type.getAsString();
          hostLiterals += " ";
          hostLiterals += tmp_lit;
          hostLiterals += " = ";
          hostLiterals += SS.str();
          hostLiterals += ";\n    ";
          hostArgNames.push_back(tmp_lit);
        }

        break;
        }
      case HipaccKernelClass::FieldKind::IterationSpace:
      case HipaccKernelClass::FieldKind::Image: {
        // image
        HipaccAccessor *Acc = getImgFromMapping(arg.field);
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
      case HipaccKernelClass::FieldKind::Mask:
        hostArgNames.push_back(getMaskFromMapping(arg.field)->getName());

        break;
    }
    i++;
  }

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

