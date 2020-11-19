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

#ifndef _CLASS_REPRESENTATION_H_
#define _CLASS_REPRESENTATION_H_

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"

#include <clang/AST/ASTContext.h>

#include <locale>
#include <map>
#include <set>
#include <string>

namespace clang {
namespace hipacc {
// forward declaration
class HipaccKernel;

// direction of data transfers
enum MemoryTransferDirection {
  HOST_TO_DEVICE,
  DEVICE_TO_HOST,
  DEVICE_TO_DEVICE,
  HOST_TO_HOST
};

// boundary handling modes for images
enum class Boundary : uint8_t {
  UNDEFINED = 0,
  CLAMP,
  REPEAT,
  MIRROR,
  CONSTANT
};

// reduction modes for convolutions
enum class Reduce : uint8_t {
  SUM = 0,
  MIN,
  MAX,
  PROD,
  MEDIAN
};

// interpolation modes for accessors
enum class Interpolate : uint8_t {
  NO = 0,
  NN,
  LF,
  B5,
  CF,
  L3
};

// optimization modes for kernels
enum class OptimizationOption : uint8_t {
  NONE = 0,
  KERNEL_FUSE
};


// common base class for images, masks and pyramids
class HipaccSize {
  protected:
    unsigned size_x, size_y;
    std::string size_x_str, size_y_str;

  public:
    HipaccSize() :
      size_x(0), size_y(0),
      size_x_str(), size_y_str()
    {}

    void setSizeX(unsigned x) {
      size_x = x;
      size_x_str = std::to_string(x);
    }
    void setSizeY(unsigned y) {
      size_y = y;
      size_y_str = std::to_string(y);
    }
    unsigned getSizeX() { return size_x; }
    unsigned getSizeY() { return size_y; }
    std::string getSizeXStr() {
      assert(!size_x_str.empty());
      return size_x_str;
    }
    std::string getSizeYStr() {
      assert(!size_y_str.empty());
      return size_y_str;
    }
};


class HipaccMemory : public HipaccSize {
  protected:
    VarDecl *VD;
    std::string name;
    QualType type;

  public:
    HipaccMemory(VarDecl *VD, const std::string &name, QualType type) :
      HipaccSize(),
      VD(VD),
      name(name),
      type(type)
    {}

    const std::string &getName() const { return name; }
    VarDecl *getDecl() { return VD; }
    QualType getType() { return type; }
    std::string getTypeStr() { return type.getAsString(); }
};


class HipaccImage : public HipaccMemory {
  private:
    ASTContext &Ctx;

  public:
    HipaccImage(ASTContext &Ctx, VarDecl *VD, QualType QT) :
      HipaccMemory(VD, VD->getNameAsString(), QT),
      Ctx(Ctx)
    {}

    unsigned getPixelSize() { return static_cast<unsigned>(Ctx.getTypeSize(type))/8; }
    std::string getTextureType();
    std::string getImageReadFunction();
};


class HipaccPyramid : public HipaccImage {
  public:
    HipaccPyramid(ASTContext &Ctx, VarDecl *VD, QualType QT) :
      HipaccImage(Ctx, VD, QT)
    {}
};


class HipaccBoundaryCondition : public HipaccSize {
  private:
    VarDecl *VD;
    HipaccImage *img;
    Boundary mode;
    std::string pyr_idx_str;
    bool is_pyramid;
    Expr *constExpr;

  public:
    HipaccBoundaryCondition(VarDecl *VD, HipaccImage *img) :
      HipaccSize(),
      VD(VD),
      img(img),
      mode(Boundary::UNDEFINED),
      pyr_idx_str(),
      is_pyramid(false),
      constExpr(nullptr)
    {}

    void setPyramidIndex(const std::string &idx) {
      is_pyramid = true;
      pyr_idx_str = idx;
    }
    void setBoundaryMode(Boundary m) { mode = m; }
    void setConstVal(APValue &val, ASTContext &Ctx);
    VarDecl *getDecl() { return VD; }
    HipaccImage *getImage() { return img; }
    Boundary getBoundaryMode() { return mode; }
    std::string getPyramidIndex() { return pyr_idx_str; }
    bool isPyramid() { return is_pyramid; }
    Expr *getConstExpr() { return constExpr; }
};


class HipaccAccessor {
  private:
    VarDecl *VD;
    HipaccBoundaryCondition *bc;
    Interpolate mode;
    std::string name;
    bool crop;
    // kernel parameter name for width, height, and stride
    DeclRefExpr *widthDecl, *heightDecl, *strideDecl, *scaleXDecl, *scaleYDecl;
    DeclRefExpr *offsetXDecl, *offsetYDecl;

  public:
    HipaccAccessor(VarDecl *VD, HipaccBoundaryCondition *bc, Interpolate mode, bool crop) :
      VD(VD),
      bc(bc),
      mode(mode),
      name(VD->getNameAsString()),
      crop(crop),
      widthDecl(nullptr), heightDecl(nullptr), strideDecl(nullptr),
      scaleXDecl(nullptr), scaleYDecl(nullptr),
      offsetXDecl(nullptr), offsetYDecl(nullptr)
    {}

    void setWidthDecl(DeclRefExpr *width) { widthDecl = width; }
    void setHeightDecl(DeclRefExpr *height) { heightDecl = height; }
    void setStrideDecl(DeclRefExpr *stride) { strideDecl = stride; }
    void setScaleXDecl(DeclRefExpr *scale) { scaleXDecl = scale; }
    void setScaleYDecl(DeclRefExpr *scale) { scaleYDecl = scale; }
    void setOffsetXDecl(DeclRefExpr *ox) { offsetXDecl = ox; }
    void setOffsetYDecl(DeclRefExpr *oy) { offsetYDecl = oy; }
    VarDecl *getDecl() { return VD; }
    const std::string &getName() const { return name; }
    HipaccBoundaryCondition *getBC() { return bc; }
    Interpolate getInterpolationMode() { return mode; }
    HipaccImage *getImage() { return bc->getImage(); }
    unsigned getSizeX() { return bc->getSizeX(); }
    unsigned getSizeY() { return bc->getSizeY(); }
    std::string getSizeXStr() { return bc->getSizeXStr(); }
    std::string getSizeYStr() { return bc->getSizeYStr(); }
    DeclRefExpr *getWidthDecl() { return widthDecl; }
    DeclRefExpr *getHeightDecl() { return heightDecl; }
    DeclRefExpr *getStrideDecl() { return strideDecl; }
    DeclRefExpr *getScaleXDecl() { return scaleXDecl; }
    DeclRefExpr *getScaleYDecl() { return scaleYDecl; }
    DeclRefExpr *getOffsetXDecl() { return offsetXDecl; }
    DeclRefExpr *getOffsetYDecl() { return offsetYDecl; }
    void resetDecls() {
      widthDecl = heightDecl = strideDecl = nullptr;
      scaleXDecl = scaleYDecl = offsetXDecl = offsetYDecl = nullptr;
    }
    bool isCrop() { return crop; }
    Boundary getBoundaryMode() {
      return bc->getBoundaryMode();
    }
    Expr *getConstExpr() { return bc->getConstExpr(); }
};


class HipaccIterationSpace : public HipaccAccessor {
  private:
    HipaccImage *img;

  public:
    HipaccIterationSpace(VarDecl *VD, HipaccImage *img, bool crop) :
      HipaccAccessor(VD, new HipaccBoundaryCondition(VD, img), Interpolate::NO, crop),
      img(img)
    {}

    HipaccImage *getImage() { return img; }
};


class HipaccMask : public HipaccMemory {
  public:
    enum class MaskType : uint8_t {
      Mask,
      Domain
    };
  private:
    MaskType mask_type;
    InitListExpr *init_list;
    bool is_constant;
    bool is_printed;
    SmallVector<HipaccKernel *, 16> kernels;
    std::string hostMemName;
    bool *domain_space;
    HipaccMask *copy_mask;

  public:
    HipaccMask(VarDecl *VD, QualType QT, MaskType type) :
      HipaccMemory(VD, "_const" + VD->getNameAsString(), QT),
      mask_type(type),
      init_list(nullptr),
      is_constant(false),
      is_printed(false),
      kernels(0),
      hostMemName(),
      domain_space(nullptr),
      copy_mask(nullptr)
    {}

    ~HipaccMask() {
      if (domain_space)
        delete[] domain_space;
      if (copy_mask)
        delete copy_mask;
    }

    void setIsConstant(bool c) { is_constant = c; }
    void setIsPrinted(bool p) { is_printed = p; }
    void setInitList(InitListExpr *il) { init_list = il; }
    bool isDomain() { return mask_type==MaskType::Domain; }
    bool isConstant() { return is_constant; }
    bool isPrinted() { return is_printed; }
    Expr *getInitExpr(size_t x, size_t y) {
      return dyn_cast<InitListExpr>(init_list->getInit(static_cast<unsigned int>(y)))->getInit(static_cast<unsigned int>(x));
    }
    void addKernel(HipaccKernel *K) { kernels.push_back(K); }
    ArrayRef<HipaccKernel *> getKernels() { return kernels; }
    void setHostMemName(const std::string &name) { hostMemName = name; }
    std::string getHostMemName() { return hostMemName; }
    void setSizeX(unsigned x) {
      HipaccMemory::setSizeX(x);
      if (isDomain())
        setDomainSize(size_x*size_y);
    }
    void setSizeY(unsigned y) {
      HipaccMemory::setSizeY(y);
      if (isDomain())
        setDomainSize(size_x*size_y);
    }
    void setDomainSize(unsigned size) {
      if (domain_space) {
        delete[] domain_space;
        domain_space = nullptr;
      }
      if (size > 0) {
        domain_space = new bool[size];
        for (size_t i = 0; i < size; ++i) {
          domain_space[i] = true;
        }
      }
    }
    void setDomainDefined(unsigned pos, bool def) {
      if (domain_space)
        domain_space[pos] = def;
    }
    void setDomainDefined(unsigned x, unsigned y, bool def) {
      unsigned pos = (y * size_x) + x;
      setDomainDefined(pos, def);
    }
    bool isDomainDefined(unsigned x, unsigned y) {
      unsigned pos = (y * size_x) + x;
      return domain_space && domain_space[pos];
    }
    void setCopyMask(HipaccMask *mask) {
      copy_mask = mask;
    }
    bool hasCopyMask() {
      return copy_mask != nullptr;
    }
    HipaccMask *getCopyMask() {
      return copy_mask;
    }
};


class HipaccKernelClass {
  private:
    // type of kernel member
    enum class FieldKind : uint8_t {
      Normal,
      IterationSpace,
      Image,
      Mask
    };
    struct KernelMemberInfo {
      FieldKind kind;
      FieldDecl *field;
      QualType type;
      std::string name;
    };

    std::string name;
    CXXMethodDecl *kernelFunction, *reduceFunction, *binningFunction;
    QualType pixelType, binType;
    KernelStatistics *kernelStatistics;
    // kernel member information
    SmallVector<KernelMemberInfo, 16> members;
    SmallVector<FieldDecl *, 16> imgFields;
    SmallVector<FieldDecl *, 16> maskFields;
    SmallVector<FieldDecl *, 16> domainFields;
    FieldDecl *output_image;

  public:
    explicit HipaccKernelClass(const std::string &name) :
      name(name),
      kernelFunction(nullptr),
      reduceFunction(nullptr),
      binningFunction(nullptr),
      kernelStatistics(nullptr),
      members(0),
      imgFields(0),
      maskFields(0),
      domainFields(0),
      output_image(nullptr)
    {}

    const std::string &getName() const { return name; }

    void setKernelFunction(CXXMethodDecl *fun, CompilerKnownClasses &classes, bool verbose=false) {
      kernelFunction = fun;
      kernelStatistics = KernelStatistics::create(fun, name, output_image,
          classes, verbose);
    }

    void setReduceFunction(CXXMethodDecl *fun) { reduceFunction = fun; }
    void setBinningFunction(CXXMethodDecl *fun) { binningFunction = fun; }
    CXXMethodDecl *getKernelFunction() { return kernelFunction; }
    CXXMethodDecl *getReduceFunction() { return reduceFunction; }
    CXXMethodDecl *getBinningFunction() { return binningFunction; }

    void setPixelType(QualType type) { pixelType = type; }
    void setBinType(QualType type) { binType = type; }
    QualType getPixelType() { return pixelType; }
    QualType getBinType() { return binType; }

    KernelStatistics &getKernelStatistics(void) {
      return *kernelStatistics;
    }

    MemoryAccess getMemAccess(FieldDecl *decl) {
      return kernelStatistics->getMemAccess(decl);
    }
    MemoryPattern getMemPattern(FieldDecl *decl) {
      return kernelStatistics->getMemPattern(decl);
    }
    VectorInfo getVectorizeInfo(VarDecl *decl) {
      return kernelStatistics->getVectorizeInfo(decl);
    }
    VarDecl *getVarDeclByName(std::string name) {
      return kernelStatistics->getVarDeclByName(name);
    }
    KernelType getKernelType() {
      return kernelStatistics->getKernelType();
    }

    void addArg(FieldDecl *FD, QualType QT, StringRef Name) {
      KernelMemberInfo info = { FieldKind::Normal, FD, QT, Name };
      members.push_back(info);
    }
    void addImgArg(FieldDecl *FD, QualType QT, StringRef Name) {
      KernelMemberInfo info = { FieldKind::Image, FD, QT, Name };
      members.push_back(info);
      imgFields.push_back(FD);
    }
    void addMaskArg(FieldDecl *FD, QualType QT, StringRef Name) {
      KernelMemberInfo info = { FieldKind::Mask, FD, QT, Name};
      members.push_back(info);
      maskFields.push_back(FD);
    }
    void addISArg(FieldDecl *FD, QualType QT, StringRef Name) {
      KernelMemberInfo info = { FieldKind::IterationSpace, FD, QT, Name };
      members.push_back(info);
      imgFields.push_back(FD);
      output_image = FD;
    }

    ArrayRef<KernelMemberInfo> getMembers() { return members; }
    ArrayRef<FieldDecl *> getImgFields() { return imgFields; }
    ArrayRef<FieldDecl *> getMaskFields() { return maskFields; }
    FieldDecl *getOutField() { return output_image; }

    friend class HipaccKernel;
};


class HipaccKernelFeatures : public HipaccDevice {
  public:
    // memory type of image/mask
    enum MemoryType {
      Global    = 0x1,
      Constant  = 0x2,
      Texture_  = 0x4,
      Local     = 0x8
    };

    CompilerOptions &options;
    HipaccKernelClass *KC;
    std::map<HipaccAccessor *, MemoryType> memMap;
    std::map<HipaccAccessor *, Texture> texMap;

    void calcImgFeature(FieldDecl *decl, HipaccAccessor *acc) {
      MemoryType mem_type = Global;
      Texture tex_type = Texture::None;
      MemoryAccess mem_access = KC->getMemAccess(decl);
      MemoryPattern mem_pattern = KC->getMemPattern(decl);

      if (options.useTextureMemory() &&
          options.getTextureType() == Texture::Array2D) {
        mem_type = Texture_;
        tex_type = Texture::Array2D;
      } else {
        // for OpenCL image-objects and CUDA arrays we have to enable or disable
        // textures all the time otherwise, use texture memory only in case the
        // image is accessed with an offset to the x-coordinate
        if (options.emitCUDA()) {
          if (mem_pattern & NO_STRIDE) {
            if (require_textures[PointOperator] != Texture::None) {
              mem_type = Texture_;
              tex_type = require_textures[PointOperator];
            }
          }
          if ((mem_pattern & STRIDE_X) ||
              (mem_pattern & STRIDE_Y) ||
              (mem_pattern & STRIDE_XY)) {
            // possibly use textures only for stride_x ?
            if (require_textures[LocalOperator] != Texture::None) {
              mem_type = Texture_;
              tex_type = require_textures[LocalOperator];
            }
          } else if (mem_pattern & USER_XY) {
            if (require_textures[UserOperator] != Texture::None) {
              mem_type = Texture_;
              tex_type = require_textures[LocalOperator];
            }
          }

          if (mem_access == WRITE_ONLY && tex_type != Texture::Array2D) {
            mem_type = Global;
            tex_type = Texture::None;
          }
        }
      }

      if (acc->getSizeX() * acc->getSizeY() >= local_memory_threshold)
        mem_type = static_cast<MemoryType>(mem_type|Local);

      memMap[acc] = mem_type;
      texMap[acc] = tex_type;
    }

  public:
    HipaccKernelFeatures(CompilerOptions &options, HipaccKernelClass *KC) :
      HipaccDevice(options),
      options(options),
      KC(KC)
    {}

    bool useLocalMemory(HipaccAccessor *acc) {
      if (memMap.count(acc) && (memMap[acc] & Local))
        return true;
      return false;
    }

    Texture useTextureMemory(HipaccAccessor *acc) {
      if (memMap.count(acc) && (memMap[acc] & Texture_))
        return texMap[acc];
      return Texture::None;
    }

    bool vectorize() {
      return vectorization;
    }

    unsigned getPixelsPerThread() {
      return pixels_per_thread[KC->getKernelType()];
    }
};


class HipaccKernel : public HipaccKernelFeatures {
  private:
    ASTContext &Ctx;
    VarDecl *VD;
    std::string name;
    std::string kernelName, reduceName, binningName;
    std::string fileName;
    std::string reduceStr, binningStr, infoStr, numBinsStr;
    unsigned infoStrCnt, binningStrCnt;
    HipaccIterationSpace *iterationSpace;
    std::map<FieldDecl *, HipaccAccessor *> imgMap;
    std::map<FieldDecl *, HipaccMask *> maskMap;
    SmallVector<QualType, 16> argTypes;
    SmallVector<std::string, 16> argTypeNames;
    SmallVector<std::string, 16> hostArgNames;
    SmallVector<std::string, 16> deviceArgNames;
    SmallVector<FieldDecl *, 16> deviceArgFields;
    SmallVector<FunctionDecl *, 16> deviceFuncs;
    std::set<std::string> usedVars;
    unsigned max_threads_for_kernel;
    unsigned max_size_x, max_size_y;
    unsigned updated_size_x, updated_size_y;
    unsigned max_size_x_undef, max_size_y_undef;
    unsigned num_threads_x, num_threads_y;
    unsigned num_reg, num_lmem, num_smem, num_cmem;
    std::string graphName, graphNodeName, graphNodeDepName, graphNodeArgName;
    OptimizationOption OptmOpt;

    std::string executionParameter;

    void calcSizes();
    void calcConfig();
    void createArgInfo();
    void addParam(QualType QT1, QualType QT2, QualType QT3, std::string typeC,
        std::string typeO, std::string name, FieldDecl *fd);
    void addParam(QualType QT, const std::string &name, FieldDecl *fd) {
      addParam(QT, QT, QT, QT.getAsString(), QT.getAsString(), name, fd);
    }
    void createHostArgInfo(ArrayRef<Expr *> hostArgs, std::string &hostLiterals,
        unsigned &literalCount);

  public:
    HipaccKernel(ASTContext &Ctx, VarDecl *VD, HipaccKernelClass *KC,
        CompilerOptions &options) :
      HipaccKernelFeatures(options, KC),
      Ctx(Ctx),
      VD(VD),
      name(VD->getNameAsString()),
      kernelName(options.getTargetPrefix() + KC->getName() + name + "Kernel"),
      reduceName(options.getTargetPrefix() + KC->getName() + name + "Reduce"),
      binningName(options.getTargetPrefix() + KC->getName() + name + "Binning"),
      fileName(options.getTargetPrefix() + KC->getName() + VD->getNameAsString()),
      reduceStr(), binningStr(), infoStr(), numBinsStr(),
      infoStrCnt(0), binningStrCnt(0),
      iterationSpace(nullptr),
      imgMap(),
      maskMap(),
      argTypes(),
      argTypeNames(),
      hostArgNames(),
      deviceArgNames(),
      deviceArgFields(),
      deviceFuncs(),
      max_threads_for_kernel(0),
      max_size_x(0), max_size_y(0),
      updated_size_x(0), updated_size_y(0),
      max_size_x_undef(0), max_size_y_undef(0),
      num_threads_x(default_num_threads_x),
      num_threads_y(default_num_threads_y),
      num_reg(0),
      num_lmem(0),
      num_smem(0),
      num_cmem(0),
      graphName("graph_")
    {
      OptmOpt = OptimizationOption::NONE;
    }

    VarDecl *getDecl() const { return VD; }
    HipaccKernelClass *getKernelClass() const { return KC; }
    const std::string &getName() const { return name; }
    const std::string &getKernelName() const { return kernelName; }
    const std::string &getReduceName() const { return reduceName; }
    const std::string &getBinningName() const { return binningName; }
    const std::string &getFileName() const { return fileName; }
    const std::string &getInfoStr() const { return infoStr; }
    const std::string &getReduceStr() const { return reduceStr; }
    const std::string getBinningStr() const {
      return binningStr + "_" + std::to_string(binningStrCnt);
    }

    // keep track of variables used within kernel
    void setUsed(std::string name) { usedVars.insert(name); }
    void setUnused(std::string name) { usedVars.erase(name); }
    void setOptimizationOptions(OptimizationOption opt) { OptmOpt = opt; }
    void updateFusionSizeX(unsigned szX) { updated_size_x = szX; }
    void updateFusionSizeY(unsigned szY) { updated_size_y = szY; }
    void resetUsed() {
      usedVars.clear();
      deviceFuncs.clear();
      for (auto map : imgMap)
        map.second->resetDecls();
    }
    bool getUsed(std::string name) {
      return usedVars.find(name) != usedVars.end();
    }
    bool isFusible() {
      return OptmOpt == OptimizationOption::KERNEL_FUSE;
    }

    // keep track of functions called within kernel
    void addFunctionCall(FunctionDecl *FD) { deviceFuncs.push_back(FD); }
    ArrayRef<FunctionDecl *> getFunctionCalls() { return deviceFuncs; }

    HipaccIterationSpace *getIterationSpace() { return iterationSpace; }

    void insertMapping(FieldDecl *decl, HipaccIterationSpace *iter) {
      imgMap.emplace(decl, iter);
      calcImgFeature(decl, iter);
      iterationSpace = iter;
    }
    void insertMapping(FieldDecl *decl, HipaccAccessor *acc) {
      imgMap.emplace(decl, acc);
      calcImgFeature(decl, acc);
      calcSizes();
    }
    void insertMapping(FieldDecl *decl, HipaccMask *mask) {
      maskMap.emplace(decl, mask);
    }

    HipaccAccessor *getImgFromMapping(FieldDecl *decl) {
      auto iter = imgMap.find(decl);
      if (iter == imgMap.end())
        return nullptr;
      return iter->second;
    }
    HipaccMask *getMaskFromMapping(FieldDecl *decl) {
      auto iter = maskMap.find(decl);
      if (iter == maskMap.end())
        return nullptr;
      return iter->second;
    }

    ArrayRef<QualType> getArgTypes() {
      createArgInfo();
      return argTypes;
    }
    ArrayRef<std::string> getArgTypeNames() {
      createArgInfo();
      return argTypeNames;
    }
    ArrayRef<std::string> getDeviceArgNames() {
      createArgInfo();
      return deviceArgNames;
    }
    void setHostArgNames(ArrayRef<Expr *> hostArgs, std::string &hostLiterals,
        unsigned &literalCount) {
      std::string cnt(std::to_string(infoStrCnt++));
      infoStr = name + "_info" + cnt;
      reduceStr = name + "_red" + cnt;
      binningStr = name + "_bin" + cnt;
      createArgInfo();
      createHostArgInfo(hostArgs, hostLiterals, literalCount);
    }
    ArrayRef<std::string> getHostArgNames() {
      hipacc_require(hostArgNames.size(), "host argument names not set");
      return hostArgNames;
    }
    ArrayRef<FieldDecl *> getDeviceArgFields() {
      createArgInfo();
      return deviceArgFields;
    }

    void setResourceUsage(int reg, int lmem, int smem, int cmem) {
      num_reg = reg;
      num_lmem = lmem;
      num_smem = isAMDGPU() ? 4 * smem : smem; // only 1/4th of the actual usage is reported for AMD
      num_cmem = cmem;
      // calculate new configuration
      calcConfig();
      // reset parameter information since the tiling and corresponding
      // variables may have been changed
      argTypes.clear();
      argTypeNames.clear();
      // hostArgNames are set later on
      deviceArgNames.clear();
      deviceArgFields.clear();
      // recreate parameter information
      createArgInfo();
    }

    void setDefaultConfig();

    void printStats() {
      llvm::errs() << "Statistics for Kernel '" << fileName << "'\n";
      llvm::errs() << "  Vectorization: " << vectorize() << "\n";
      llvm::errs() << "  Pixels per thread: " << getPixelsPerThread() << "\n";

      for (auto map : memMap) {
        llvm::errs() << "  Image '" << map.first->getName() << "': ";
        if (map.second & Global)    llvm::errs() << "global";
        if (map.second & Constant)  llvm::errs() << "constant";
        if (map.second & Local)     llvm::errs() << "local";
        if (map.second & Texture_) {
          llvm::errs() << "texture ";
          switch (texMap[map.first]) {
            case Texture::None:     llvm::errs() << "(None)";     break;
            case Texture::Linear1D: llvm::errs() << "(Linear1D)"; break;
            case Texture::Linear2D: llvm::errs() << "(Linear2D)"; break;
            case Texture::Array2D:  llvm::errs() << "(Array2D)";  break;
            case Texture::Ldg:      llvm::errs() << "(Ldg)";      break;
          }
        }
        llvm::errs() << "\n";
      }
    }

    void setNumBinsStr(const std::string &numBins) {
      binningStrCnt++;
      numBinsStr = numBins;
    }
    std::string getNumBinsStr() { return numBinsStr; }
    unsigned getMaxThreadsForKernel() { return max_threads_for_kernel; }
    unsigned getMaxThreadsPerBlock() { return max_threads_per_block; }
    unsigned getMaxTotalSharedMemory() { return max_total_shared_memory; }
    unsigned getWarpSize() { return max_threads_per_warp; }
    unsigned getMaxSizeX() { return max_size_x<=1?0:max_size_x>>1; }
    unsigned getMaxSizeY() { return max_size_y<=1?0:max_size_y>>1; }
    unsigned getMaxSizeXUndef() {
      return max_size_x_undef<=1?0:max_size_x_undef>>1;
    }
    unsigned getMaxSizeYUndef() {
      return max_size_y_undef<=1?0:max_size_y_undef>>1;
    }
    unsigned getNumThreadsX() { return num_threads_x; }
    unsigned getNumThreadsY() { return num_threads_y; }
    unsigned getNumThreadsReduce() {
      return default_num_threads_x*default_num_threads_y;
    }
    unsigned getPixelsPerThreadReduce() {
      return pixels_per_thread[GlobalOperator];
    }

    std::string const& getExecutionParameter() const {
      return executionParameter;
    }

    void setExecutionParameter(std::string const& ep) {
      executionParameter = ep;
    }

    void setGraphName(std::string name) { graphName = name; }
    void setGraphNodeName(std::string name) { graphNodeName = name; }
    void setGraphNodeDepName(std::string name) { graphNodeDepName = name; }
    void setGraphNodeArgName(std::string name) { graphNodeArgName = name; }

    std::string getGraphName() const { return graphName; }
    std::string getGraphNodeName() const { return graphNodeName; };
    std::string getGraphNodeDepName() const { return graphNodeDepName; };
    std::string getGraphNodeArgName() const { return graphNodeArgName; };
};
} // namespace hipacc
} // namespace clang

#endif  // _CLASS_REPRESENTATION_H_

// vim: set ts=2 sw=2 sts=2 et ai:

