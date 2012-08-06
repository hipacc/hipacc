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

#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <llvm/ADT/StringRef.h>

#include <map>
#include <sstream>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"

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
enum BoundaryMode {
  BOUNDARY_UNDEFINED,
  BOUNDARY_CLAMP,
  BOUNDARY_REPEAT,
  BOUNDARY_MIRROR,
  BOUNDARY_CONSTANT
};

// reduction modes for convolutions
enum ConvolutionMode {
    HipaccSUM,
    HipaccMIN,
    HipaccMAX,
    HipaccPROD,
    HipaccMEDIAN
};

// interpolation modes for accessors
enum InterpolationMode {
    InterpolateNO,
    InterpolateNN,
    InterpolateLF,
    InterpolateCF,
    InterpolateL3
};


class HipaccImage {
  private:
    ASTContext &Ctx;
    VarDecl *VD;
    std::string name;
    // width, height, and stride initialization strings on the host side
    std::string width, height, stride;
    std::string widthType, heightType;
    QualType type;
    std::string pixelType;
    unsigned int pixelSize;

  public:
    HipaccImage(ASTContext &Ctx, VarDecl *VD) :
      Ctx(Ctx),
      VD(VD),
      name(VD->getNameAsString()),
      width(""), height(""), stride(""),
      widthType(""), heightType(""),
      type(),
      pixelType(""),
      pixelSize(0)
    {}

    const std::string &getName() const { return name; }
    void setWidth(std::string w) { width = w; }
    void setHeight(std::string h) { height = h; }
    void setStride(std::string s) { stride = s; }
    void setWidthType(std::string wt) { widthType = wt; }
    void setHeightType(std::string ht) { heightType = ht; }
    void setPixelType(QualType pt) {
      type = pt;
      pixelType = pt.getAsString();
      pixelSize = Ctx.getTypeSize(pt)/8;
    }
    VarDecl *getDecl() { return VD; }
    std::string getWidth() { return width; }
    std::string getHeight() { return height; }
    std::string getStride() {
      if (stride.empty()) return width;
      else return stride;
    }
    std::string getWidthType() { return widthType; }
    std::string getHeightType() { return heightType; }
    std::string getPixelType() { return pixelType; }
    unsigned int getPixelSize() { return pixelSize; }
    QualType getPixelQualType() { return type; }
    std::string getTextureType();
    std::string getImageReadFunction();
};


class HipaccBoundaryCondition {
  private:
    HipaccImage *img;
    VarDecl *VD;
    unsigned int size_x, size_y;
    std::string size_x_str, size_y_str;
    BoundaryMode boundaryHandling;
    Expr *constExpr;
    void setConstExpr(APValue &val, ASTContext &Ctx);

  public:
    HipaccBoundaryCondition(HipaccImage *img, VarDecl *VD) :
      img(img),
      VD(VD),
      size_x(0),
      size_y(0),
      size_x_str(""),
      size_y_str(""),
      boundaryHandling(BOUNDARY_UNDEFINED),
      constExpr(NULL)
    {}

    void setSizeX(unsigned int x) {
      std::string Str;
      llvm::raw_string_ostream SS(Str);
      SS << x;
      size_x_str = SS.str();
      size_x = x;
    }
    void setSizeY(unsigned int y) {
      std::string Str;
      llvm::raw_string_ostream SS(Str);
      SS << y;
      size_y_str = SS.str();
      size_y = y;
    }
    void setBoundaryHandling(BoundaryMode m) { boundaryHandling = m; }
    void setConstVal(APValue &val, ASTContext &Ctx) {
      setConstExpr(val, Ctx);
    }
    VarDecl *getDecl() { return VD; }
    HipaccImage *getImage() { return img; }
    unsigned int getSizeX() { return size_x; }
    unsigned int getSizeY() { return size_y; }
    std::string getSizeXStr() { return size_x_str; }
    std::string getSizeYStr() { return size_y_str; }
    BoundaryMode getBoundaryHandling() { return boundaryHandling; }
    Expr *getConstExpr() { return constExpr; }
};


class HipaccAccessor {
  private:
    HipaccBoundaryCondition *bc;
    InterpolationMode interpolation;
    VarDecl *VD;
    std::string name;
    bool crop;
    // width, height, and offset initialization strings on the host side
    std::string width, height;
    std::string offset_x, offset_y;
    std::string widthType, heightType;
    std::string offsetXType, offsetYType;
    // kernel parameter name for width, height, and stride
    std::string widthParm, heightParm, strideParm, offsetXParm, offsetYParm;
    DeclRefExpr *widthDecl, *heightDecl, *strideDecl, *scaleXDecl, *scaleYDecl;
    DeclRefExpr *offsetXDecl, *offsetYDecl;

  public:
    HipaccAccessor(HipaccBoundaryCondition *bc, InterpolationMode mode, VarDecl
        *VD) :
      bc(bc),
      interpolation(mode),
      VD(VD),
      name(VD->getNameAsString()),
      crop(true),
      width(""), height(""), offset_x(""), offset_y(""),
      widthType(""), heightType(""), offsetXType(""), offsetYType(""),
      widthParm(name + "_width"), heightParm(name + "_height"),
      strideParm(name + "_stride"),
      offsetXParm(name + "_offset_x"), offsetYParm(name + "_offset_y"),
      widthDecl(NULL), heightDecl(NULL), strideDecl(NULL),
      scaleXDecl(NULL), scaleYDecl(NULL), offsetXDecl(NULL), offsetYDecl(NULL)
    {}

    void setWidth(std::string w) { width = w; }
    void setHeight(std::string h) { height = h; }
    void setOffsetX(std::string x) { offset_x = x; }
    void setOffsetY(std::string y) { offset_y = y; }
    void setWidthType(std::string wt) { widthType = wt; }
    void setHeightType(std::string ht) { heightType = ht; }
    void setOffsetXType(std::string oxt) { offsetXType = oxt; }
    void setOffsetYType(std::string oyt) { offsetYType = oyt; }
    void setWidthDecl(DeclRefExpr *width) { widthDecl = width; }
    void setHeightDecl(DeclRefExpr *height) { heightDecl = height; }
    void setStrideDecl(DeclRefExpr *stride) { strideDecl = stride; }
    void setScaleXDecl(DeclRefExpr *scale) { scaleXDecl = scale; }
    void setScaleYDecl(DeclRefExpr *scale) { scaleYDecl = scale; }
    void setOffsetXDecl(DeclRefExpr *ox) { offsetXDecl = ox; }
    void setOffsetYDecl(DeclRefExpr *oy) { offsetYDecl = oy; }
    void setNoCrop() { crop = false; }
    VarDecl *getDecl() { return VD; }
    const std::string &getName() const { return name; }
    HipaccBoundaryCondition *getBC() { return bc; }
    InterpolationMode getInterpolation() { return interpolation; }
    std::string getWidth() { return width; }
    std::string getHeight() { return height; }
    std::string getOffsetX() { return offset_x; }
    std::string getOffsetY() { return offset_y; }
    HipaccImage *getImage() { return bc->getImage(); }
    unsigned int getSizeX() { return bc->getSizeX(); }
    unsigned int getSizeY() { return bc->getSizeY(); }
    std::string getSizeXStr() { return bc->getSizeXStr(); }
    std::string getSizeYStr() { return bc->getSizeYStr(); }
    std::string getWidthType() { return widthType; }
    std::string getHeightType() { return heightType; }
    std::string getOffsetXType() { return offsetXType; }
    std::string getOffsetYType() { return offsetYType; }
    std::string getWidthParm() { return widthParm; }
    std::string getHeightParm() { return heightParm; }
    std::string getStrideParm() { return strideParm; }
    std::string getOffsetXParm() { return offsetXParm; }
    std::string getOffsetYParm() { return offsetYParm; }
    DeclRefExpr *getWidthDecl() { return widthDecl; }
    DeclRefExpr *getHeightDecl() { return heightDecl; }
    DeclRefExpr *getStrideDecl() { return strideDecl; }
    DeclRefExpr *getScaleXDecl() { return scaleXDecl; }
    DeclRefExpr *getScaleYDecl() { return scaleYDecl; }
    DeclRefExpr *getOffsetXDecl() { return offsetXDecl; }
    DeclRefExpr *getOffsetYDecl() { return offsetYDecl; }
    bool isCrop() { return crop; }
    BoundaryMode getBoundaryHandling() {
      return bc->getBoundaryHandling();
    }
    Expr *getConstExpr() { return bc->getConstExpr(); }
};


class HipaccIterationSpace {
  private:
    HipaccImage *img;
    VarDecl *VD;
    std::string width;
    std::string height;
    std::string offsetX;
    std::string offsetY;
    std::string widthType;
    std::string heightType;
    std::string offsetXType;
    std::string offsetYType;
    // Accessor used during ASTTranslate to access the Output image
    HipaccAccessor *acc;

    void createOutputAccessor();

  public:
    HipaccIterationSpace(HipaccImage *img, VarDecl *VD) :
      img(img),
      VD(VD),
      width(""),
      height(""),
      offsetX(""),
      offsetY(""),
      widthType(""),
      heightType(""),
      offsetXType(""),
      offsetYType(""),
      acc(NULL)
    {
      createOutputAccessor();
    }

    void setWidth(std::string w) { width = w; }
    void setHeight(std::string h) { height = h; }
    void setOffsetX(std::string ox) { offsetX = ox; }
    void setOffsetY(std::string oy) { offsetY = oy; }
    void setWidthType(std::string wt) { widthType = wt; }
    void setHeightType(std::string ht) { heightType = ht; }
    void setOffsetXType(std::string oxt) { offsetXType = oxt; }
    void setOffsetYType(std::string oyt) { offsetYType = oyt; }
    VarDecl *getDecl() { return VD; }
    HipaccImage *getImage() { return img; }
    HipaccAccessor *getAccessor() { return acc; }
    std::string getWidth() { return width; }
    std::string getHeight() { return height; }
    std::string getOffsetX() { return offsetX; }
    std::string getOffsetY() { return offsetY; }
    std::string getWidthType() { return widthType; }
    std::string getHeightType() { return heightType; }
    std::string getOffsetXType() { return offsetXType; }
    std::string getOffsetYType() { return offsetYType; }
};


class HipaccMask {
  private:
    VarDecl *VD;
    std::string name;
    InitListExpr *init_list;
    unsigned int size_x, size_y;
    std::string size_x_str, size_y_str;
    QualType type;
    std::string typeStr;
    bool is_constant;
    bool is_printed;
    llvm::SmallVector<HipaccKernel *, 16> kernels;
    std::string hostMemName;
    Expr *hostMemExpr;

  public:
    HipaccMask(VarDecl *VD) :
      VD(VD),
      name("_const" + VD->getNameAsString()),
      init_list(NULL),
      size_x(0),
      size_y(0),
      size_x_str(""),
      size_y_str(""),
      type(),
      typeStr(""),
      is_constant(false),
      is_printed(false),
      kernels(0),
      hostMemName(""),
      hostMemExpr(NULL)
    {}

    const std::string &getName() const { return name; }
    void setSizeX(unsigned int x) {
      std::string Str;
      llvm::raw_string_ostream SS(Str);
      SS << x;
      size_x_str = SS.str();
      size_x = x;
    }
    void setSizeY(unsigned int y) {
      std::string Str;
      llvm::raw_string_ostream SS(Str);
      SS << y;
      size_y_str = SS.str();
      size_y = y;
    }
    void setType(QualType QT) {
      type = QT;
      typeStr = QT.getAsString();
    }
    void setIsConstant(bool c) { is_constant = c; }
    void setIsPrinted(bool p) { is_printed = p; }
    void setInitList(InitListExpr *il) { init_list = il; }
    unsigned int getSizeX() { return size_x; }
    unsigned int getSizeY() { return size_y; }
    std::string getSizeXStr() { return size_x_str; }
    std::string getSizeYStr() { return size_y_str; }
    QualType getType() { return type; }
    std::string getTypeStr() { return typeStr; }
    VarDecl *getDecl() { return VD; }
    bool isConstant() { return is_constant; }
    bool isPrinted() { return is_printed; }
    InitListExpr *getInitList() { return init_list; }
    void addKernel(HipaccKernel *K) { kernels.push_back(K); }
    llvm::SmallVector<HipaccKernel *, 16> &getKernels() {
      return kernels;
    }
    void setHostMemName(std::string name) { hostMemName = name; }
    std::string getHostMemName() { return hostMemName; }
    void setHostMemExpr(Expr *expr) { hostMemExpr = expr; }
    Expr *getHostMemExpr() { return hostMemExpr; }
};


class HipaccKernelClass {
  private:
    // type of argument
    enum ArgumentKind {
      Normal,
      IterationSpace,
      Image,
      Mask
    };
    // argument information
    struct argumentInfo {
      ArgumentKind kind;
      FieldDecl *field;
      QualType type;
      llvm::StringRef name;
    };

    std::string name;
    CXXMethodDecl *kernelFunction;
    KernelStatistics *kernelStatistics;
    // kernel parameter information
    llvm::SmallVector<argumentInfo, 16> arguments;
    llvm::SmallVector<FieldDecl *, 16> imgFields;
    llvm::SmallVector<FieldDecl *, 16> maskFields;

  public:
    HipaccKernelClass(std::string name) :
      name(name),
      kernelFunction(NULL),
      kernelStatistics(NULL),
      arguments(0),
      imgFields(0),
      maskFields(0)
    {}

    const std::string &getName() const { return name; }

    void setKernelFunction(CXXMethodDecl *fun) { kernelFunction = fun; }
    CXXMethodDecl *getKernelFunction() { return kernelFunction; }

    void setKernelStatistics(KernelStatistics *stats) {
      kernelStatistics = stats;
    }
    KernelStatistics &getKernelStatistics(void) {
      return *kernelStatistics;
    }

    MemoryAccess getImgAccess(FieldDecl *decl) {
      return kernelStatistics->getMemAccess(decl);
    }
    MemoryAccessDetail getImgAccessDetail(FieldDecl *decl) {
      return kernelStatistics->getMemAccessDetail(decl);
    }
    VectorInfo getVectorizeInfo(VarDecl *decl) {
      return kernelStatistics->getVectorizeInfo(decl);
    }
    KernelType getKernelType() {
      return kernelStatistics->getKernelType();
    }


    void addArg(FieldDecl *FD, QualType QT, llvm::StringRef Name) {
      argumentInfo a = {Normal, FD, QT, Name};
      arguments.push_back(a);
    }
    void addImgArg(FieldDecl *FD, QualType QT, llvm::StringRef Name) {
      argumentInfo a = {Image, FD, QT, Name};
      arguments.push_back(a);
      imgFields.push_back(FD);
    }
    void addMaskArg(FieldDecl *FD, QualType QT, llvm::StringRef Name) {
      argumentInfo a = {Mask, FD, QT, Name};
      arguments.push_back(a);
      maskFields.push_back(FD);
    }

    unsigned int getNumArgs() { return arguments.size(); }
    unsigned int getNumImages() { return imgFields.size(); }
    unsigned int getNumMasks() { return maskFields.size(); }

    llvm::SmallVector<argumentInfo, 16> &getArguments() { return arguments; }
    llvm::SmallVector<FieldDecl *, 16> &getImgFields() { return imgFields; }
    llvm::SmallVector<FieldDecl *, 16> &getMaskFields() { return maskFields; }

    friend class HipaccKernel;
};


class HipaccKernelFeatures : public HipaccDevice {
  public:
    // memory type of image/mask
    enum MemoryType {
      Global    = 0x1,
      Constant  = 0x2,
      Texture   = 0x4,
      Local     = 0x8
    };

    CompilerOptions &options;
    HipaccKernelClass *KC;
    std::map<HipaccAccessor *, MemoryType> memMap;
    std::map<HipaccAccessor *, TextureType> texMap;

    void calcISFeature(HipaccAccessor *acc) {
      MemoryType mem_type = Global;
      TextureType tex_type = NoTexture;

      if (options.useTextureMemory(USER_ON) &&
          options.getTextureType()==Array2D) {
        mem_type = Texture;
        tex_type = Array2D;
      }

      memMap[acc] = mem_type;
      texMap[acc] = tex_type;
    }

    void calcImgFeature(FieldDecl *decl, HipaccAccessor *acc) {
      MemoryType mem_type = Global;
      TextureType tex_type = NoTexture;
      MemoryAccessDetail memAccessDetail = KC->getImgAccessDetail(decl);

      if (options.useTextureMemory(USER_ON) &&
          options.getTextureType()==Array2D) {
        mem_type = Texture;
        tex_type = Array2D;
      } else {
        // for OpenCL image-objects and CUDA arrays we have to enable or disable
        // textures all the time otherwise, use texture memory only in case the
        // image is accessed with an offset to the x-coordinate
        if (options.emitCUDA()) {
          if (memAccessDetail & NO_STRIDE) {
            if (require_textures[PointOperator]) {
              mem_type = Texture;
              tex_type = require_textures[PointOperator];
            }
          }
          if ((memAccessDetail & STRIDE_X) || 
              (memAccessDetail & STRIDE_Y) || 
              (memAccessDetail & STRIDE_XY)) {
            // possibly use textures only for stride_x ?
            if (require_textures[LocalOperator]) {
              mem_type = Texture;
              tex_type = require_textures[LocalOperator];
            }
          } else if (memAccessDetail & USER_XY) {
            if (require_textures[UserOperator]) {
              mem_type = Texture;
              tex_type = require_textures[LocalOperator];
            }
          }
        }
      }

      if (acc->getSizeX() * acc->getSizeY() >= local_memory_threshold) {
        mem_type = (MemoryType) (mem_type|Local);
      }

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
      if (memMap.count(acc)) {
        if (memMap[acc] & Local) return true;
      }

      return false;
    }

    TextureType useTextureMemory(HipaccAccessor *acc) {
      if (memMap.count(acc)) {
        if (memMap[acc] & Texture) return texMap[acc];
      }

      return NoTexture;
    }

    bool propagateConstants() {
      return true;
    }

    bool vectorize() {
      return vectorization;
    }

    unsigned int getPixelsPerThread() {
      return pixels_per_thread[KC->getKernelType()];
    }
};


class HipaccKernel : public HipaccKernelFeatures {
  private:
    ASTContext &Ctx;
    VarDecl *VD;
    CompilerOptions &options;
    std::string name;
    std::string kernelName;
    std::string fileName;
    HipaccIterationSpace *iterationSpace;
    std::map<FieldDecl *, HipaccAccessor *> imgMap;
    std::map<FieldDecl *, HipaccMask *> maskMap;
    llvm::SmallVector<QualType, 16> argTypesCUDA;
    llvm::SmallVector<QualType, 16> argTypesOpenCL;
    llvm::SmallVector<QualType, 16> argTypesC;
    llvm::SmallVector<std::string, 16> argTypeNamesCUDA;
    llvm::SmallVector<std::string, 16> argTypeNamesOpenCL;
    llvm::SmallVector<llvm::StringRef, 16> argNames;
    llvm::SmallVector<std::string, 16> hostArgNames;
    llvm::SmallVector<FieldDecl *, 16> argFields;
    unsigned int max_threads_for_kernel;
    unsigned int max_size_x, max_size_y;
    unsigned int num_threads_x, num_threads_y;
    unsigned int num_blocks_bh_l, num_blocks_bh_r, num_blocks_bh_y;
    unsigned int num_reg, num_lmem, num_smem, num_cmem;

    void calcSizes();
    void calcConfig();
    void createArgInfo();
    void addParam(QualType QT1, QualType QT2, QualType QT3, std::string typeC,
        std::string typeO, llvm::StringRef name, FieldDecl *fd);
    void createHostArgInfo(Expr **hostArgs, std::string &hostLiterals, unsigned
        int &literalCount);

  public:
    HipaccKernel(ASTContext &Ctx, VarDecl *VD, HipaccKernelClass *KC,
        CompilerOptions &options) :
      HipaccKernelFeatures(options, KC),
      Ctx(Ctx),
      VD(VD),
      options(options),
      name(VD->getNameAsString()),
      kernelName(KC->getName()),
      fileName(""),
      iterationSpace(NULL),
      imgMap(),
      maskMap(),
      argTypesCUDA(),
      argTypesOpenCL(),
      argTypesC(),
      argTypeNamesCUDA(),
      argTypeNamesOpenCL(),
      argNames(),
      argFields(),
      max_threads_for_kernel(0),
      max_size_x(0),
      max_size_y(0),
      num_threads_x(default_num_threads),
      num_threads_y(1),
      num_blocks_bh_l(0),
      num_blocks_bh_r(0),
      num_blocks_bh_y(0),
      num_reg(0),
      num_lmem(0),
      num_smem(0),
      num_cmem(0)
    {}

    VarDecl *getDecl() const { return VD; }
    HipaccKernelClass *getKernelClass() const { return KC; }
    const std::string &getName() const { return name; }
    const std::string &getKernelName() const { return kernelName; }
    void setFileName(std::string name) { fileName = name; }
    const std::string &getFileName() const { return fileName; }

    void setIterationSpace(HipaccIterationSpace *IS) {
      iterationSpace = IS;
      calcISFeature(iterationSpace->getAccessor());
    }
    HipaccIterationSpace *getIterationSpace() { return iterationSpace; }

    void insertMapping(FieldDecl *decl, HipaccAccessor *acc) {
      imgMap.insert(std::pair<FieldDecl *, HipaccAccessor *>(decl, acc));
      calcImgFeature(decl, acc);
      calcSizes();
    }
    void insertMapping(FieldDecl *decl, HipaccMask *mask) {
      maskMap.insert(std::pair<FieldDecl *, HipaccMask *>(decl, mask));
    }

    HipaccAccessor *getImgFromMapping(FieldDecl *decl) {
      std::map<FieldDecl *, HipaccAccessor *>::iterator iter = imgMap.find(decl);

      if (iter == imgMap.end()) return NULL;
      else return iter->second;
    }
    HipaccMask *getMaskFromMapping(FieldDecl *decl) {
      std::map<FieldDecl *, HipaccMask *>::iterator iter = maskMap.find(decl);

      if (iter == maskMap.end()) return NULL;
      else return iter->second;
    }

    unsigned int getNumArgs() {
      createArgInfo();
      return argNames.size();
    }
    QualType *getArgTypes(ASTContext &Ctx, TargetCode target_code) {
      createArgInfo();

      switch (target_code) {
        case TARGET_CUDA:
          return argTypesCUDA.data();
        case TARGET_OpenCL:
        case TARGET_OpenCLx86:
          return argTypesOpenCL.data();
        case TARGET_C:
        default:
          return argTypesC.data();
      }
    }
    std::string *getArgTypeNames() {
      createArgInfo();
      if (options.emitCUDA()) return argTypeNamesCUDA.data();
      else return argTypeNamesOpenCL.data();
    }
    llvm::StringRef *getArgNames() {
      createArgInfo();
      return argNames.data();
    }
    void setHostArgNames(Expr **hostArgs, std::string &hostLiterals, unsigned
        int &literalCount) {
      createHostArgInfo(hostArgs, hostLiterals, literalCount);
    }
    std::string *getHostArgNames() {
      assert(hostArgNames.size() && "host argument names not set");
      return hostArgNames.data();
    }
    FieldDecl **getArgFields() {
      createArgInfo();
      return argFields.data();
    }

    void setResourceUsage(int reg, int lmem, int smem, int cmem) {
      num_reg = reg;
      num_lmem = lmem;
      if (isAMDGPU()) {
        // only 1/4th of the actual usage is reported
        num_smem = 4 * smem;
      } else {
        num_smem = smem;
      }
      num_cmem = cmem;
      calcConfig();
    }

    void setDefaultConfig();

    void printStats() {
      llvm::errs() << "Statistics for Kernel '" << fileName << "'\n";
      llvm::errs() << "  Loop unrolling & constant propagation: " <<
        propagateConstants() << "\n";
      llvm::errs() << "  Vectorization: " << vectorize() << "\n";
      llvm::errs() << "  Pixels per thread: " << getPixelsPerThread() << "\n";

      for (std::map<HipaccAccessor *, MemoryType>::iterator iter =
          memMap.begin(), eiter=memMap.end(); iter!=eiter; ++iter) {
        llvm::errs() << "  Image '" << iter->first->getName() << "': ";
        if (iter->second & Global) llvm::errs() << "global ";
        if (iter->second & Constant) llvm::errs() << "constant ";
        if (iter->second & Texture) llvm::errs() << "texture ";
        if (iter->second & Local) llvm::errs() << "local ";
        llvm::errs() << "\n";
      }
    }

    unsigned int getMaxThreadsForKernel() { return max_threads_for_kernel; }
    unsigned int getMaxThreadsPerBlock() { return max_threads_per_block; }
    unsigned int getMaxTotalSharedMemory() { return max_total_shared_memory; }
    unsigned int getWarpSize() { return max_threads_per_warp; }
    unsigned int getMaxSizeX() { return max_size_x<=1?0:max_size_x>>1; }
    unsigned int getMaxSizeY() { return max_size_y<=1?0:max_size_y>>1; }
    unsigned int getNumThreadsX() { return num_threads_x; }
    unsigned int getNumThreadsY() { return num_threads_y; }
    unsigned int getNumBlocksBHL() { return num_blocks_bh_l; }
    unsigned int getNumBlocksBHR() { return num_blocks_bh_r; }
    unsigned int getNumBlocksBHY() { return num_blocks_bh_y; }
};


class HipaccGlobalReductionClass {
  private:
    std::string name;
    CXXMethodDecl *reductionFunction;

  public:
    HipaccGlobalReductionClass(std::string name) :
      name(name),
      reductionFunction(NULL)
    {}

    const std::string &getName() const { return name; }

    void setReductionFunction(CXXMethodDecl *fun) { reductionFunction = fun; }
    CXXMethodDecl *getReductionFunction() { return reductionFunction; }
};


class HipaccGlobalReduction : public HipaccDevice {
  private:
    HipaccAccessor *acc;
    VarDecl *VD;
    HipaccGlobalReductionClass *GRC;
    std::string name;
    std::string type;
    std::string neutral;
    std::string fileName;
    CXXMethodDecl *reductionFunction;
    bool is_accessor;
    bool is_printed;
    unsigned int num_threads;

  public:
    HipaccGlobalReduction(HipaccAccessor *acc, VarDecl *VD,
        HipaccGlobalReductionClass *GRC, CompilerOptions &options, bool
        is_accessor) :
      HipaccDevice(options),
      acc(acc),
      VD(VD),
      GRC(GRC),
      name(VD->getNameAsString()),
      type(""),
      neutral(""),
      fileName(""),
      reductionFunction(NULL),
      is_accessor(is_accessor),
      is_printed(false),
      num_threads(256)
    {}

    void setType(std::string t) { type = t; }
    void setNeutral(std::string n) { neutral = n; }
    void setFileName(std::string name) { fileName = name; }
    void setReductionFunction(CXXMethodDecl *fun) { reductionFunction = fun; }
    void setIsPrinted(bool p) { is_printed = p; }

    const std::string &getName() const { return name; }
    const std::string getType() { return type; }
    const std::string getNeutral() { return neutral; }
    const std::string &getFileName() const { return fileName; }
    bool isAccessor() { return is_accessor; }
    bool isPrinted() { return is_printed; }

    VarDecl *getDecl() { return VD; }
    HipaccAccessor *getAccessor() { return acc; }
    CXXMethodDecl *getReductionFunction() { return reductionFunction; }
    HipaccGlobalReductionClass *getReductionClass() const { return GRC; }
    unsigned int getNumThreads() { return num_threads; }
    unsigned int getPixelsPerThread() {
      return pixels_per_thread[GlobalOperator];
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _CLASS_REPRESENTATION_H_

// vim: set ts=2 sw=2 sts=2 et ai:

