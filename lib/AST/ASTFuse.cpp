//
// Copyright (c) 2018, University of Erlangen-Nuremberg
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

//===--- ASTFuse.cpp - Kernel Fusion for the AST --------------------------===//
//
// This file implements the fusion and printing of the translated kernels.
//
//===--------------------------------------------------------------------===//

#include "hipacc/AST/ASTFuse.h"

using namespace clang;
using namespace hipacc;
using namespace ASTNode;
using namespace hipacc::Builtin;


FunctionDecl *ASTFuse::createFusedKernelDecl(std::list<HipaccKernel *> &l) {
  SmallVector<QualType, 16> argTypesKFuse;
  SmallVector<std::string, 16> deviceArgNamesKFuse;
  std::string name;
  size_t kfNumArg;
  for (auto K : l) {  // concatenate kernel arguments from the list
    kfNumArg = 0;
    for (auto arg : K->getArgTypes())
      argTypesKFuse.push_back(arg);
    for (auto argNam : K->getDeviceArgNames()) {
      deviceArgNamesKFuse.push_back(argNam);
      FuncDeclParamKernelMap[argNam] = K;
      FuncDeclParamDeclMap[argNam] = K->getDeviceArgFields()[kfNumArg++];
    }
    name += K->getName() + "_";
  }

  fusedKernelName = compilerOptions.getTargetPrefix() + name + "FusedKernel";
  for (auto K : l) {  // record kernel name
    fusedKernelNameMap[K] = fusedKernelName;
  }
  fusedFileName = compilerOptions.getTargetPrefix() + name + "Fused";
  fusedFileNamesAll.push_back(fusedFileName);
  return createFunctionDecl(Ctx, Ctx.getTranslationUnitDecl(),
      fusedKernelName, Ctx.VoidTy, argTypesKFuse, deviceArgNamesKFuse);
}

void ASTFuse::insertPrologFusedKernel() {
  for (auto VD : fusionRegVarDecls)
    curFusedKernelBody.push_back(createDeclStmt(Ctx, VD));
  for (auto VD : fusionIdxVarDecls)
    curFusedKernelBody.push_back(createDeclStmt(Ctx, VD));
}

void ASTFuse::insertEpilogFusedKernel() {
  // TODO
}

void ASTFuse::initKernelFusion() {
  fusionRegVarCount=0;
  fusionIdxVarCount=0;
  fusionRegVarDecls.clear();
  fusionIdxVarDecls.clear();
  curFusedKernelBody.clear();
  FuncDeclParamKernelMap.clear();
  FuncDeclParamDeclMap.clear();
  localKernelMemorySizeMap.clear();
  FusibleKernelSubListPosMap.clear();
  fusedFileNamesAll.clear();
  fusedKernelNameMap.clear();
}

void ASTFuse::markKernelPositionSublist(std::list<HipaccKernel *> &l) {
  // find number of local operators in the list
  SmallVector<unsigned, 16> vecLocalKernelIndices;
  vecLocalKernelIndices.clear();
  for (auto K : l) {
    HipaccKernelClass *KC = K->getKernelClass();
    if (KC->getKernelType() == LocalOperator) {
      vecLocalKernelIndices.push_back(dataDeps->getKernelIndex(K));
    }
    FusionTypeTags *tags = new FusionTypeTags;
    FusibleKernelSubListPosMap[K] = tags;
  }

  std::list<HipaccKernel *> pointKernelList;
  SmallVector<std::list<HipaccKernel *>, 16> vecLocalKernelLists;
  if (vecLocalKernelIndices.size() == 0) {
    // scenario 1: no local kernel sublists
    // e.g. p -> p -> ... -> p
    // sublisting
    pointKernelList.assign(l.begin(), l.end());
    // tag generation
    for (auto it = (pointKernelList.begin()); it != pointKernelList.end(); ++it) {
      HipaccKernel *K = *it;
      FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
      if (it == pointKernelList.begin() && std::next(it) == pointKernelList.end()) {
        KTag->Point2PointLoc = Undefined;  // single kernel sublist, no fusion
      } else if (it == pointKernelList.begin()) {
        KTag->Point2PointLoc = Source;
      } else if (std::next(it) == pointKernelList.end()) {
        KTag->Point2PointLoc = Destination;
      } else {
        KTag->Point2PointLoc = Intermediate;
      }
    }
  } else if (vecLocalKernelIndices.front() == 0) {
    // scenario 2: no point kernel sublist
    // e.g. l -> p -> ... -> l -> p -> ...
    // sublisting
    for (auto it = (vecLocalKernelIndices.begin()); it != vecLocalKernelIndices.end(); ++it) {
      std::list<HipaccKernel *> localKernelList;
      if (std::next(it) == vecLocalKernelIndices.end()) {
        auto itTemp = l.begin();
        std::advance(itTemp, *it);
        localKernelList.assign(itTemp, l.end());
      } else {
        auto itTempS = l.begin();
        std::advance(itTempS, *it);
        auto itTempE = l.begin();
        std::advance(itTempE, *(std::next(it)));
        localKernelList.assign(itTempS, itTempE);
      }
      vecLocalKernelLists.push_back(localKernelList);
    }
    // tag generation
    for (auto itLists = (vecLocalKernelLists.begin()); itLists != vecLocalKernelLists.end(); ++itLists) {
      std::list<HipaccKernel *> listLocal = *itLists;
      for (auto it = (listLocal.begin()); it != listLocal.end(); ++it) {
        HipaccKernel *K = *it;
        FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
        // local-to-point tag
        if (it == listLocal.begin() && std::next(it) == listLocal.end()) {
          KTag->Local2PointLoc = Undefined;
        } else if (it == listLocal.begin()) {
          KTag->Local2PointLoc = Source;
        } else if (std::next(it) == listLocal.end()) {
          KTag->Local2PointLoc = Destination;
        } else {
          KTag->Local2PointLoc = Intermediate;
        }
        // local-to-local tag
        if (itLists == vecLocalKernelLists.begin() && std::next(itLists) == vecLocalKernelLists.end()) {
          KTag->Local2LocalLoc = Undefined;
        } else if (itLists == vecLocalKernelLists.begin()) {
          KTag->Local2LocalLoc = Source;
        } else if (std::next(itLists) == vecLocalKernelLists.end()) {
          KTag->Local2LocalLoc = Destination;
        } else {
          KTag->Local2LocalLoc = Intermediate;
        }
      }
    }
  } else if (vecLocalKernelIndices.front() > 0) {
    // scenario 3: both point and local kernel sublists exist
    // e.g. p -> ... -> p -> l -> p -> ...
    // sublisting
    auto itTempP = l.begin();
    std::advance(itTempP, vecLocalKernelIndices.front());
    pointKernelList.assign(l.begin(), itTempP);
    for (auto it = (vecLocalKernelIndices.begin()); it != vecLocalKernelIndices.end(); ++it) {
      std::list<HipaccKernel *> localKernelList;
      if (std::next(it) == vecLocalKernelIndices.end()) {
        auto itTemp = l.begin();
        std::advance(itTemp, *it);
        localKernelList.assign(itTemp, l.end());
      } else {
        auto itTempS = l.begin();
        std::advance(itTempS, *it);
        auto itTempE = l.begin();
        std::advance(itTempE, *(std::next(it)));
        localKernelList.assign(itTempS, itTempE);
      }
      vecLocalKernelLists.push_back(localKernelList);
    }
    // tag generation
    for (auto it = (pointKernelList.begin()); it != pointKernelList.end(); ++it) {
      HipaccKernel *K = *it;
      FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
      if (it == pointKernelList.begin() && std::next(it) == pointKernelList.end()) {
        KTag->Point2PointLoc = Undefined;
        KTag->Point2LocalLoc = Source;
      } else if (it == pointKernelList.begin()) {
        KTag->Point2PointLoc = Source;
        KTag->Point2LocalLoc = Source;
      } else if (std::next(it) == pointKernelList.end()) {
        KTag->Point2PointLoc = Destination;
        KTag->Point2LocalLoc = Source;
      } else {
        KTag->Point2PointLoc = Intermediate;
        KTag->Point2LocalLoc = Source;
      }
    }
    for (auto itLists = (vecLocalKernelLists.begin()); itLists != vecLocalKernelLists.end(); ++itLists) {
      std::list<HipaccKernel *> listLocal = *itLists;
      for (auto it = (listLocal.begin()); it != listLocal.end(); ++it) {
        HipaccKernel *K = *it;
        FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
        // local-to-point tag
        if (it == listLocal.begin() && std::next(it) == listLocal.end()) {
          KTag->Local2PointLoc = Undefined;
        } else if (it == listLocal.begin()) {
          KTag->Local2PointLoc = Source;
        } else if (std::next(it) == listLocal.end()) {
          KTag->Local2PointLoc = Destination;
        } else {
          KTag->Local2PointLoc = Intermediate;
        }
        // point/local-to-local tag
        if (itLists == vecLocalKernelLists.begin() && std::next(itLists) == vecLocalKernelLists.end()) {
          KTag->Point2LocalLoc = Destination;
          KTag->Local2LocalLoc = Undefined;
        } else if (itLists == vecLocalKernelLists.begin()) {
          KTag->Point2LocalLoc = Destination;
          KTag->Local2LocalLoc = Source;
        } else if (std::next(itLists) == vecLocalKernelLists.end()) {
          KTag->Local2LocalLoc = Destination;
        } else {
          KTag->Local2LocalLoc = Intermediate;
        }
      }
    }
  }
}

void ASTFuse::recomputeMemorySizeLocal2LocalFusion(std::list<HipaccKernel *> &l) {
  // shared memory size update
  unsigned YSizeAcc = 1;
  unsigned XSizeAcc = 1;
  SmallVector<HipaccKernel *, 16> revList;
  for (auto K : l) {
    HipaccKernelClass *KC = K->getKernelClass();
    for (auto img : KC->getImgFields()) {
      HipaccAccessor *Acc = K->getImgFromMapping(img);
      MemoryAccess mem_acc = KC->getMemAccess(img);
      if (mem_acc == READ_ONLY && K->useLocalMemory(Acc)) {
        localKernelMemorySizeMap[K] = std::make_tuple(Acc->getSizeX(),
            Acc->getSizeY());
        revList.insert(revList.begin(), K);
      }
    }
  }
  for (auto K : revList) {
    XSizeAcc = XSizeAcc + std::get<0>(localKernelMemorySizeMap[K]) - 1;
    YSizeAcc = YSizeAcc + std::get<1>(localKernelMemorySizeMap[K]) - 1;
    localKernelMemorySizeMap[K] = std::make_tuple(XSizeAcc, YSizeAcc);
  }
}

void ASTFuse::HipaccFusion(std::list<HipaccKernel *>& l) {
  assert((l.size() >=2) && "at least two kernels shoud be recorded for fusion");
  initKernelFusion();
  curFusedKernelDecl = createFusedKernelDecl(l);
  createGidVarDecl();

  markKernelPositionSublist(l);

  // generating body for the fused kernel
  Stmt *curFusionBody;
  bool Local2LocalEndInsertion = false;
  VarDecl *idxXFused, *idxYFused;
  SmallVector<Stmt *, 16> vecFusionBody;
  SmallVector<Stmt *, 16> vecProducerP2LBody;
  std::queue<Stmt *> stmtsL2LProducerKernel;
  std::queue<Stmt *> stmtsL2LConsumerKernel;
  HipaccKernel *KLocalSrc = nullptr;


  for (auto it = (l.begin()); it != l.end(); ++it) {
    curFusionBody = nullptr;
    HipaccKernel *K = *it;
    HipaccKernelClass *KC = K->getKernelClass();
    FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
    FunctionDecl *kernelDecl = createFunctionDecl(Ctx,
        Ctx.getTranslationUnitDecl(), K->getKernelName(),
        Ctx.VoidTy, K->getArgTypes(), K->getDeviceArgNames());
    ASTTranslateFusion *Hipacc = new ASTTranslateFusion(Ctx, kernelDecl, K, KC,
        builtins, compilerOptions);

    // TODO, enable the composition of multiple fusions
    // domain-specific translation and fusion
    switch(KTag->Point2PointLoc) {
      default: break;
      case Source:
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->configSrcOperatorP2P(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Destination:
        Hipacc->configDestOperatorP2P(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->configIntermOperatorP2P(VDIn, VDOut);
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
    }
    switch(KTag->Local2PointLoc) {
      default: break;
      case Source:
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->configSrcOperatorP2P(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Destination:
        Hipacc->configDestOperatorP2P(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->configIntermOperatorP2P(VDIn, VDOut);
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
    }
    switch(KTag->Point2LocalLoc) {
      default: break;
      case Source:
        createIdx4FusionVarDecl();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->configSrcOperatorP2L(fusionRegVarDecls.back(), fusionIdxVarDecls.back());
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
      case Destination:
        Hipacc->configDestOperatorP2L(fusionRegVarDecls.back(), fusionIdxVarDecls.back(),
                                createCompoundStmt(Ctx, vecProducerP2LBody));
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->configIntermOperatorP2P(VDIn, VDOut);
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
    }
    switch(KTag->Local2LocalLoc) {
      default: break;
      case Source:
        recomputeMemorySizeLocal2LocalFusion(l);
        createReg4FusionVarDecl(KC->getOutField()->getType());
        createIdx4FusionVarDecl();
        idxXFused = fusionIdxVarDecls.back();
        createIdx4FusionVarDecl();
        idxYFused = fusionIdxVarDecls.back();
        Hipacc->configSrcOperatorL2L(fusionRegVarDecls.back(), idxXFused, idxYFused,
            std::get<1>(localKernelMemorySizeMap[K]));
        Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        stmtsL2LProducerKernel = Hipacc->getFusionLocalKernelBody();
        KLocalSrc = K;
        break;
      case Destination:
        Hipacc->configDestOperatorL2L(stmtsL2LProducerKernel,
          fusionRegVarDecls.back(), idxXFused, idxYFused,
          std::get<1>(localKernelMemorySizeMap[K]));
        Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        stmtsL2LConsumerKernel = Hipacc->getFusionLocalKernelBody();
        Local2LocalEndInsertion = true;
        break;
      case Intermediate:
        break;
    }

    if (curFusionBody) {
      vecFusionBody.push_back(curFusionBody);
    }
  }

  if (Local2LocalEndInsertion) {
    HipaccKernel *K = KLocalSrc;
    HipaccKernelClass *KC = K->getKernelClass();
    FunctionDecl *kernelDecl = createFunctionDecl(Ctx,
        Ctx.getTranslationUnitDecl(), K->getKernelName(),
        Ctx.VoidTy, K->getArgTypes(), K->getDeviceArgNames());
    ASTTranslateFusion *Hipacc = new ASTTranslateFusion(Ctx, kernelDecl, K, KC,
        builtins, compilerOptions);
    Hipacc->configEndSrcOperatorL2L(stmtsL2LConsumerKernel, idxXFused, idxYFused,
        std::get<1>(localKernelMemorySizeMap[K]));
    localKernelMaxAccSizeUpdated = localKernelMemorySizeMap[K];
    vecFusionBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
  }

  insertPrologFusedKernel();
  curFusedKernelBody.push_back(createCompoundStmt(Ctx, vecFusionBody));
  insertEpilogFusedKernel();
  curFusedKernelDecl->setBody(createCompoundStmt(Ctx, curFusedKernelBody));
}

void ASTFuse::setFusedKernelConfiguration(std::list<HipaccKernel *>& l) {
  #ifdef USE_JIT_ESTIMATE
  HipaccFusion(l);
  printFusedKernelFunction(l); // write fused kernel to file

  // JIT compile kernel in order to get resource usage
  std::string command = (l.back())->getCompileCommand(fusedKernelName,
      fusedFileName, compilerOptions.emitCUDA());

  int reg=0, lmem=0, smem=0, cmem=0;
  char line[FILENAME_MAX];
  SmallVector<std::string, 16> lines;
  FILE *fpipe;

  if (!(fpipe = (FILE *)popen(command.c_str(), "r"))) {
    perror("Problems with pipe");
    exit(EXIT_FAILURE);
  }

  while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
    lines.push_back(std::string(line));

    if (targetDevice.isNVIDIAGPU()) {
      char *ptr = line;
      char mem_type = 'x';
      int val1 = 0, val2 = 0;

      if (sscanf(ptr, "%d bytes %1c tack frame", &val1, &mem_type) == 2) {
        if (mem_type == 's') {
          lmem = val1;
          continue;
        }
      }

      if (sscanf(line, "ptxas info : Used %d registers", &reg) == 0)
        continue;

      while ((ptr = strchr(ptr, ','))) {
        ptr++;

        if (sscanf(ptr, "%d+%d bytes %1c mem", &val1, &val2, &mem_type) == 3) {
          switch (mem_type) {
            default: llvm::errs() << "wrong memory specifier '" << mem_type
                                  << "': " << ptr; break;
            case 'c': cmem += val1 + val2; break;
            case 'l': lmem += val1 + val2; break;
            case 's': smem += val1 + val2; break;
          }
          continue;
        }

        if (sscanf(ptr, "%d bytes %1c mem", &val1, &mem_type) == 2) {
          switch (mem_type) {
            default: llvm::errs() << "wrong memory specifier '" << mem_type
                                  << "': " << ptr; break;
            case 'c': cmem += val1; break;
            case 'l': lmem += val1; break;
            case 's': smem += val1; break;
          }
          continue;
        }

        if (sscanf(ptr, "%d texture %1c", &val1, &mem_type) == 2)
          continue;
        if (sscanf(ptr, "%d sampler %1c", &val1, &mem_type) == 2)
          continue;
        if (sscanf(ptr, "%d surface %1c", &val1, &mem_type) == 2)
          continue;

        // no match found
        llvm::errs() << "Unexpected memory usage specification: '" << ptr;
      }
    } else if (targetDevice.isAMDGPU()) {
      sscanf(line, "isa info : Used %d gprs, %d bytes lds", &reg, &smem);
    }
  }
  pclose(fpipe);

  if (reg == 0) {
    unsigned DiagIDCompile = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
        "Compiling kernel in file '%0.%1' failed, using default kernel configuration:\n%2");
    Diags.Report(DiagIDCompile)
      << fusedFileName << (const char*)(compilerOptions.emitCUDA()?"cu":"cl")
      << command.c_str();
    for (auto line : lines)
      llvm::errs() << line;
  } else {
    if (targetDevice.isNVIDIAGPU()) {
      llvm::errs() << "Resource usage for kernel '" << fusedKernelName << "'"
                   << ": " << reg << " registers, "
                   << lmem << " bytes lmem, "
                   << smem << " bytes smem, "
                   << cmem << " bytes cmem\n";
    } else if (targetDevice.isAMDGPU()) {
      llvm::errs() << "Resource usage for kernel '" << fusedKernelName << "'"
                   << ": " << reg << " gprs, "
                   << smem << " bytes lds\n";
    }
  }

  for (auto K : l) {
    K->updateFusionSizeX(std::get<1>(localKernelMaxAccSizeUpdated));
    K->updateFusionSizeY(std::get<1>(localKernelMaxAccSizeUpdated));
    K->setResourceUsage(reg, lmem, smem, cmem);
  }
  #else
  for (auto K : l)
    K->setDefaultConfig();
  #endif
}


bool ASTFuse::parseFusibleKernel(HipaccKernel *K) {
  if (!dataDeps->isFusible(K)) {
    return false;
  }

  // prepare fusible kernel list
  auto curList = vecFusibleKernelLists[dataDeps->getKernelListIndex(K)];
  auto it = curList.begin();
  std::advance(it, dataDeps->getKernelIndex(K));
  curList.insert(it, K);
  vecFusibleKernelLists[dataDeps->getKernelListIndex(K)] = curList;

  // fusion starts whenever a list is complete
  if (curList.size() == dataDeps->getKernelListSize(K)) {
    setFusedKernelConfiguration(curList);
    HipaccFusion(curList);
    printFusedKernelFunction(curList); // write fused kernel to file
  }
  return true;
}

// getters
bool ASTFuse::isSrcKernel(HipaccKernel *K) const { return dataDeps->isSrc(K); }
bool ASTFuse::isDestKernel(HipaccKernel *K) const { return dataDeps->isDest(K); }
HipaccKernel *ASTFuse::getProducerKernel(HipaccKernel *K) {
  auto curList = vecFusibleKernelLists[dataDeps->getKernelListIndex(K)];
  auto it = curList.begin();
  std::advance(it, dataDeps->getKernelIndex(K)-1);
  return *it;
}
SmallVector<std::string, 16> ASTFuse::getFusedFileNamesAll() const {
  return fusedFileNamesAll;
}
std::string ASTFuse::getFusedKernelName(HipaccKernel *K) { return fusedKernelNameMap[K]; }
unsigned ASTFuse::getNewYSizeLocalKernel(HipaccKernel *K) { return std::get<1>(localKernelMemorySizeMap[K]); }

void ASTFuse::createReg4FusionVarDecl(QualType QT) {
  std::string Name = "_reg_fusion" + std::to_string(fusionRegVarCount++);
  VarDecl *VD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), Name, QT);
  fusionRegVarDecls.push_back(VD);
}

void ASTFuse::createIdx4FusionVarDecl() {
  std::string Name = "_idx_fusion" + std::to_string(fusionIdxVarCount++);
  VarDecl *VD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), Name, Ctx.IntTy);
  fusionIdxVarDecls.push_back(VD);
}

void ASTFuse::createGidVarDecl() {
  VarDecl *gid_x = nullptr, *gid_y = nullptr;
  SmallVector<QualType, 16> uintDeclTypes;
  SmallVector<StringRef, 16> uintDeclNames;
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclNames.push_back("x");
  uintDeclNames.push_back("y");
  uintDeclNames.push_back("z");
  RecordDecl *uint3RD = createRecordDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "uint3", TTK_Struct, uintDeclTypes, uintDeclNames);
  VarDecl *threadIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "threadIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  VarDecl *blockIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  VarDecl *blockDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockDim", Ctx.getTypeDeclType(uint3RD), nullptr);
  DeclRefExpr *TIRef = createDeclRefExpr(Ctx, threadIdx);
  DeclRefExpr *BIRef = createDeclRefExpr(Ctx, blockIdx);
  DeclRefExpr *BDRef = createDeclRefExpr(Ctx, blockDim);
  VarDecl *xVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "x",
      Ctx.IntTy, nullptr);
  VarDecl *yVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "y",
      Ctx.IntTy, nullptr);
  Expr *local_id_x = createMemberExpr(Ctx, TIRef, false, xVD, xVD->getType());
  Expr *local_id_y = createMemberExpr(Ctx, TIRef, false, yVD, yVD->getType());
  Expr *block_id_x = createMemberExpr(Ctx, BIRef, false, xVD, xVD->getType());
  Expr *block_id_y = createMemberExpr(Ctx, BIRef, false, yVD, yVD->getType());
  Expr *local_size_x = createMemberExpr(Ctx, BDRef, false, xVD, xVD->getType());
  Expr *local_size_y = createMemberExpr(Ctx, BDRef, false, yVD, yVD->getType());
  gid_x = createVarDecl(Ctx, curFusedKernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, createBinaryOperator(Ctx, local_size_x,
          block_id_x, BO_Mul, Ctx.IntTy), local_id_x, BO_Add,
        Ctx.IntTy));
  Expr *YE = createBinaryOperator(Ctx, local_size_y, block_id_y, BO_Mul, Ctx.IntTy);
  gid_y = createVarDecl(Ctx, curFusedKernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, YE, local_id_y, BO_Add, Ctx.IntTy));
  fusionRegVarDecls.push_back(gid_x);
  fusionRegVarDecls.push_back(gid_y);
}


void ASTFuse::printFusedKernelFunction(std::list<HipaccKernel *>& l) {
  int fd;
  std::string filename(fusedFileName);
  std::string ifdef("_" + filename + "_");
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:          filename += ".cc"; ifdef += "CC_"; break;
    case Language::CUDA:         filename += ".cu"; ifdef += "CU_"; break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:    filename += ".cl"; ifdef += "CL_"; break;
    case Language::Renderscript: filename += ".rs"; ifdef += "RS_"; break;
    case Language::Filterscript: filename += ".fs"; ifdef += "FS_"; break;
  }

  // open file stream using own file descriptor. We need to call fsync() to
  // compile the generated code using nvcc afterwards.
  while ((fd = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0664)) < 0) {
    if (errno != EINTR) {
      std::string errorInfo("Error opening output file '" + filename + "'");
      perror(errorInfo.c_str());
    }
  }
  llvm::raw_fd_ostream OS(fd, false);

  // write ifndef, ifdef
  std::transform(ifdef.begin(), ifdef.end(), ifdef.begin(), ::toupper);
  OS << "#ifndef " + ifdef + "\n";
  OS << "#define " + ifdef + "\n\n";

  // preprocessor defines
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
      OS << "#include \"hipacc_types.hpp\"\n"
         << "#include \"hipacc_math_functions.hpp\"\n\n";
      break;
  }

  // declarations of textures, surfaces, variables, includes, definitions etc.
  size_t num_arg;
  for (auto K : l) {
    num_arg = 0;
    for (auto arg : K->getDeviceArgFields()) {
      auto cur_arg = num_arg++;
      std::string Name(K->getDeviceArgNames()[cur_arg]);
      std::string nameOrig = Name.substr(0, Name.find("_"+K->getKernelName()));
      if (!K->getUsed(Name) && !K->getUsed(nameOrig)){
        continue;
      }

      // constant memory declarations
      if (auto Mask = K->getMaskFromMapping(arg)) {
        if (Mask->isConstant()) {
          switch (compilerOptions.getTargetLang()) {
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              OS << "__constant ";
              break;
            case Language::CUDA:
              OS << "__device__ __constant__ ";
              break;
            case Language::C99:
            case Language::Renderscript:
            case Language::Filterscript:
              OS << "static const ";
              break;
          }
          OS << Mask->getTypeStr() << " " << Mask->getName() << K->getName() << "["
             << Mask->getSizeYStr() << "][" << Mask->getSizeXStr() << "] = {\n";

          // print Mask constant literals to 2D array
          for (size_t y=0; y<Mask->getSizeY(); ++y) {
            OS << "        {";
            for (size_t x=0; x<Mask->getSizeX(); ++x) {
              Mask->getInitExpr(x, y)->printPretty(OS, 0, Policy, 0);
              if (x < Mask->getSizeX()-1) {
                OS << ", ";
              }
            }
            if (y < Mask->getSizeY()-1) {
              OS << "},\n";
            } else {
              OS << "}\n";
            }
          }
          OS << "    };\n\n";
          Mask->setIsPrinted(true);
        } else {
          // emit declaration in CUDA and Renderscript
          // for other back ends, the mask will be added as kernel parameter
          switch (compilerOptions.getTargetLang()) {
            default: break;
            case Language::CUDA:
              OS << "__device__ __constant__ " << Mask->getTypeStr() << " "
                 << Mask->getName() << K->getName() << "[" << Mask->getSizeYStr()
                 << "][" << Mask->getSizeXStr() << "];\n\n";
              Mask->setIsPrinted(true);
              break;
            case Language::Renderscript:
            case Language::Filterscript:
              //OS << "rs_allocation " << K->getDeviceArgNames()[cur_arg]
              //   << ";\n\n";
              //Mask->setIsPrinted(true);
              break;
          }
        }
        continue;
      }

    }
  }

  // extern scope for CUDA
  OS << "\n";
  if (compilerOptions.emitCUDA())
    OS << "extern \"C\" {\n";

  // function definitions
  for (auto K : l) {
    for (auto fun : K->getFunctionCalls()) {
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          OS << "inline "; break;
        case Language::CUDA:
          OS << "__inline__ __device__ "; break;
        case Language::Renderscript:
        case Language::Filterscript:
          OS << "inline static "; break;
      }
      fun->print(OS, Policy);
    }
  }

  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
    case Language::Renderscript:
      break;
    case Language::CUDA:
      OS << "__global__ ";
      // TODO, configure the fused kernel, currently use the destination kernel config
      //if (compilerOptions.exploreConfig() && emitHints) {
      //  OS << "__launch_bounds__ (BSX_EXPLORE * BSY_EXPLORE) ";
      //} else {
        OS << "__launch_bounds__ (" << (l.back())->getNumThreadsX()
           << "*" << (l.back())->getNumThreadsY() << ") ";
      //}
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      break;
    case Language::Filterscript:
      break;
  }
  if (!compilerOptions.emitFilterscript())
    OS << "void ";
  OS << fusedKernelName;
  OS << "(";

  // write kernel parameters
  size_t comma = 0; num_arg = 0;
  for (auto param : curFusedKernelDecl->parameters()) {
    // back tracking the original kernels from the fused parameter list
    HipaccKernel *K = FuncDeclParamKernelMap[param->getNameAsString()];
    HipaccKernelClass *KC = K->getKernelClass();
    QualType T = param->getType();
    T.removeLocalConst();
    T.removeLocalRestrict();

    std::string Name(param->getNameAsString());
    std::string nameOrig = Name.substr(0, Name.find("_"+K->getKernelName()));
    FieldDecl *FD = FuncDeclParamDeclMap[Name];
    if (!K->getUsed(Name) && !K->getUsed(nameOrig)){
      continue;
    }

    // check if we have a Mask or Domain
    if (auto Mask = K->getMaskFromMapping(FD)) {
      if (Mask->isConstant())
        continue;
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
          if (comma++)
            OS << ", ";
          OS << "const "
             << Mask->getTypeStr()
             << " " << Mask->getName() << K->getName()
             << "[" << Mask->getSizeYStr() << "]"
             << "[" << Mask->getSizeXStr() << "]";
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          if (comma++)
            OS << ", ";
          OS << "__constant ";
          T.getAsStringInternal(Name, Policy);
          OS << Name;
          break;
        case Language::CUDA:
          // mask/domain is declared as constant memory
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          // mask/domain is declared as static memory
          break;
      }
      continue;
    }

    // check if we have an Accessor
    if (auto Acc = K->getImgFromMapping(FD)) {
      MemoryAccess mem_acc = KC->getMemAccess(FD);
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
          if (comma++)
            OS << ", ";
          if (mem_acc == READ_ONLY)
            OS << "const ";
          OS << Acc->getImage()->getTypeStr()
             << " " << Name
             << "[" << Acc->getImage()->getSizeYStr() << "]"
             << "[" << Acc->getImage()->getSizeXStr() << "]";
          // alternative for Pencil:
          // OS << "[static const restrict 2048][4096]";
          break;
        case Language::CUDA:
          if (K->useTextureMemory(Acc) != Texture::None &&
              K->useTextureMemory(Acc) != Texture::Ldg) // no parameter is emitted for textures
            continue;
          else {
            if (comma++)
              OS << ", ";
            if (mem_acc == READ_ONLY)
              OS << "const ";
            OS << T->getPointeeType().getAsString();
            OS << " * __restrict__ ";
            OS << Name;
          }
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          // __global keyword to specify memory location is only needed for OpenCL
          if (comma++)
            OS << ", ";
          if (K->useTextureMemory(Acc) != Texture::None) {
            if (mem_acc == WRITE_ONLY)
              OS << "__write_only image2d_t ";
            else
              OS << "__read_only image2d_t ";
          } else {
            OS << "__global ";
            if (mem_acc == READ_ONLY)
              OS << "const ";
            OS << T->getPointeeType().getAsString();
            OS << " * restrict ";
          }
          OS << Name;
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          break;
      }
      continue;
    }

    // normal arguments
    if (comma++)
      OS << ", ";
    T.getAsStringInternal(Name, Policy);
    OS << Name;

    // default arguments ...
    if (Expr *Init = param->getInit()) {
      CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init);
      if (!CCE || CCE->getConstructor()->isCopyConstructor())
        OS << " = ";
      Init->printPretty(OS, 0, Policy, 0);
    }
  }
  OS << ") ";

  // print kernel body
  curFusedKernelDecl->getBody()->printPretty(OS, 0, Policy, 0);

  if (compilerOptions.emitCUDA())
    OS << "}\n";
  OS << "\n";

  //if (KC->getReduceFunction()) {
  //  printReductionFunction(KC, K, OS);
  //}

  OS << "#endif //" + ifdef + "\n";
  OS << "\n";
  OS.flush();
  fsync(fd);
  close(fd);
}


void ASTTranslateFusion::configDestOperatorP2L(VarDecl *VDReg, VarDecl *VDIdx, Stmt *S) {
  bReplaceInputLocalExprs = true;
  exprInputIdxFusion = createDeclRefExpr(Ctx, VDIdx);
  exprOutputFusion = createDeclRefExpr(Ctx, VDReg);
  stmtProducerBodyP2L = S;
}

void ASTTranslateFusion::configSrcOperatorP2L(VarDecl *VDReg, VarDecl *VDIdx) {
  bReplaceInputIdxExpr = true;
  exprInputIdxFusion = createDeclRefExpr(Ctx, VDIdx);
  bReplaceOutputExpr = true;
  exprOutputFusion = createDeclRefExpr(Ctx, VDReg);
}

void ASTTranslateFusion::configSrcOperatorP2P(VarDecl *VD) {
  bReplaceOutputExpr = true;
  exprOutputFusion = createDeclRefExpr(Ctx, VD);
}

void ASTTranslateFusion::configDestOperatorP2P(VarDecl *VD) {
  bReplaceInputExpr = true;
  exprInputFusion = createDeclRefExpr(Ctx, VD);
}

void ASTTranslateFusion::configIntermOperatorP2P(VarDecl *VDIn, VarDecl *VDOut) {
  bReplaceInputExpr = true;
  exprInputFusion = createDeclRefExpr(Ctx, VDIn);
  bReplaceOutputExpr = true;
  exprOutputFusion = createDeclRefExpr(Ctx, VDOut);
}

void ASTTranslateFusion::configSrcOperatorL2L(VarDecl *VDRegOut, VarDecl *VDIdX,
    VarDecl *VDIdY, unsigned szY) {
  bRecordLocalKernelBody = true;
  bReplaceVarAccSizeY = true;
  FusionLocalVarAccSizeY = szY;
  bReplaceOutputExpr = true;
  exprOutputFusion = createDeclRefExpr(Ctx, VDRegOut);
  exprIdXShiftFusion = createDeclRefExpr(Ctx, VDIdX);
  exprIdYShiftFusion = createDeclRefExpr(Ctx, VDIdY);
}

std::queue<Stmt *> ASTTranslateFusion::getFusionLocalKernelBody() {
  return stmtsL2LKernelFusion;
}

void ASTTranslateFusion::configDestOperatorL2L(std::queue<Stmt *> stmtsLocal,
        VarDecl *VDRegIn, VarDecl *VDIdX, VarDecl *VDIdY, unsigned szY) {
  bReplaceVarAccSizeY = true;
  FusionLocalVarAccSizeY = szY;
  bInsertLocalKernelBody = true;
  stmtsProducerL2LKernelFusion = stmtsLocal;
  bRecordLocalKernelBody = true;
  bRecordLocalKernelBorderHandeling = true;
  bReplaceInputExpr = true;
  exprInputFusion = createDeclRefExpr(Ctx, VDRegIn);
  exprIdXShiftFusion = createDeclRefExpr(Ctx, VDIdX);
  exprIdYShiftFusion = createDeclRefExpr(Ctx, VDIdY);
}

void ASTTranslateFusion::configEndSrcOperatorL2L(std::queue<Stmt *> stmtsLocal,
        VarDecl *VDIdX, VarDecl *VDIdY, unsigned szY) {
  bReplaceVarAccSizeY = true;
  FusionLocalVarAccSizeY = szY;
  stmtsProducerL2LKernelFusion = stmtsLocal;
  exprIdXShiftFusion = createDeclRefExpr(Ctx, VDIdX);
  exprIdYShiftFusion = createDeclRefExpr(Ctx, VDIdY);
  bReplaceLocalKernelBody = true;
}


//*******************************************************************************
//
//******** Section: Overload ASTTranslate functions for kernel fusion ***********
//
//*******************************************************************************

Stmt *ASTTranslateFusion::Hipacc(Stmt *S) {
  if (S==nullptr) return nullptr;

  // search for image width and height parameters
  for (auto param : kernelDecl->parameters()) {
    auto parm_ref = createDeclRefExpr(Ctx, param);
    // the first parameter is the output image; create association between them.
    if (param == *kernelDecl->param_begin()) {
      outputImage = parm_ref;
      continue;
    }

    // search for boundary handling parameters
    if (param->getName().equals("bh_start_left"+kernelParamNameSuffix)) {
      bh_start_left = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_right"+kernelParamNameSuffix)) {
      bh_start_right = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_top"+kernelParamNameSuffix)) {
      bh_start_top = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_start_bottom"+kernelParamNameSuffix)) {
      bh_start_bottom = parm_ref;
      continue;
    }
    if (param->getName().equals("bh_fall_back"+kernelParamNameSuffix)) {
      bh_fall_back = parm_ref;
      continue;
    }

    if (compilerOptions.emitRenderscript() ||
        compilerOptions.emitFilterscript()) {
      // search for uint32_t x, uint32_t y parameters
      if (param->getName().equals("x"+kernelParamNameSuffix)) {
        // TODO: scan for uint32_t x
        continue;
      }
      if (param->getName().equals("y"+kernelParamNameSuffix)) {
        // TODO: scan for uint32_t y
        continue;
      }
    }


    // search for image width, height and stride parameters
    for (auto img : KernelClass->getImgFields()) {
      HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

      if (param->getName().equals(img->getNameAsString() + "_width"+kernelParamNameSuffix)) {
        Acc->setWidthDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_height"+kernelParamNameSuffix)) {
        Acc->setHeightDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_stride"+kernelParamNameSuffix)) {
        Acc->setStrideDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_offset_x"+kernelParamNameSuffix)) {
        Acc->setOffsetXDecl(parm_ref);
        continue;
      }
      if (param->getName().equals(img->getNameAsString() + "_offset_y"+kernelParamNameSuffix)) {
        Acc->setOffsetYDecl(parm_ref);
        continue;
      }
    }
  }

  // in case no stride was found, use image width as fallback
  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

    if (Acc->getStrideDecl() == nullptr) {
      Acc->setStrideDecl(Acc->getWidthDecl());
    }
  }

  // initialize target-specific variables and add gid_x and gid_y declarations
  // to kernel body
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  SmallVector<Stmt *, 16> kernelBody;
  FunctionDecl *barrier;
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
      initCPU(kernelBody, S);
      return createCompoundStmt(Ctx, kernelBody);
      break;
    case Language::CUDA:
      initCUDA(kernelBody);
      // void __syncthreads();
      barrier = builtins.getBuiltinFunction(CUDABI__syncthreads);
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      initOpenCL(kernelBody);
      // void barrier(cl_mem_fence_flags);
      barrier = builtins.getBuiltinFunction(OPENCLBIbarrier);
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      initRenderscript(kernelBody);
      break;
  }
  lidYRef = tileVars.local_id_y;
  gidYRef = tileVars.global_id_y;

  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);

    // add scale factor calculations for interpolation:
    // float acc_scale_x = (float)acc_width/is_width;
    // float acc_scale_y = (float)acc_height/is_height;
    if (Acc->getInterpolationMode() != Interpolate::NO) {
      Expr *scaleExprX = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getWidthDecl(Acc), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getWidthDecl(Kernel->getIterationSpace()), BO_Div, Ctx.FloatTy);
      Expr *scaleExprY = createBinaryOperator(Ctx, createCStyleCastExpr(Ctx,
            Ctx.FloatTy, CK_IntegralToFloating, getHeightDecl(Acc), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.FloatTy)),
          getHeightDecl(Kernel->getIterationSpace()), BO_Div, Ctx.FloatTy);
      VarDecl *scaleDeclX = createVarDecl(Ctx, kernelDecl, Acc->getName() +
          "scale_x", Ctx.FloatTy, scaleExprX);
      VarDecl *scaleDeclY = createVarDecl(Ctx, kernelDecl, Acc->getName() +
          "scale_y", Ctx.FloatTy, scaleExprY);
      DC->addDecl(scaleDeclX);
      DC->addDecl(scaleDeclY);
      kernelBody.push_back(createDeclStmt(Ctx, scaleDeclX));
      kernelBody.push_back(createDeclStmt(Ctx, scaleDeclY));
      Acc->setScaleXDecl(createDeclRefExpr(Ctx, scaleDeclX));
      Acc->setScaleYDecl(createDeclRefExpr(Ctx, scaleDeclY));
    }
  }

  // clear all stored decls before cloning, otherwise existing VarDecls will
  // be reused and we will miss declarations
  KernelDeclMapTex.clear();
  KernelDeclMapShared.clear();
  KernelDeclMapVector.clear();
  KernelDeclMapAcc.clear();
  KernelFunctionMap.clear();

  // add vector pointer declarations for images
  if (Kernel->vectorize() && !compilerOptions.emitC99()) {
    // search for member name in kernel parameter list
    for (auto param : kernelDecl->parameters()) {
      // output image - iteration space
      std::string strFuseName = (*kernelDecl->param_begin())->getName();
      strFuseName += kernelParamNameSuffix;
      if (param->getName().equals(strFuseName)) {
        // <type>4 *Output4 = (<type>4 *) Output;
        VarDecl *VD = CloneParmVarDecl(param);

        VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
              createDeclRefExpr(Ctx, param), nullptr,
              Ctx.getTrivialTypeSourceInfo(VD->getType())));

        kernelBody.push_back(createDeclStmt(Ctx, VD));

        // update output Image reference
        outputImage = createDeclRefExpr(Ctx, VD);
      }
    }

    for (auto img : KernelClass->getImgFields()) {
      StringRef name = img->getName();
      // search for member name in kernel parameter list
      for (auto param : kernelDecl->parameters()) {
        // parameter name matches
        std::string strFuseName = name;
        strFuseName += kernelParamNameSuffix;
        if (param->getName().equals(strFuseName)) {
          // <type>4 *Input4 = (<type>4 *) Input;
          VarDecl *VD = CloneParmVarDecl(param);

          VD->setInit(createCStyleCastExpr(Ctx, VD->getType(), CK_BitCast,
                createDeclRefExpr(Ctx, param), nullptr,
                Ctx.getTrivialTypeSourceInfo(VD->getType())));

          kernelBody.push_back(createDeclStmt(Ctx, VD));
        }
      }
    }
  }

  // add shared/local memory declarations
  bool use_shared = false;
  bool border_handling = false;
  bool kernel_x = false;
  bool kernel_y = false;
  for (auto img : KernelClass->getImgFields()) {
    HipaccAccessor *Acc = Kernel->getImgFromMapping(img);
    MemoryAccess mem_acc = KernelClass->getMemAccess(img);

    // bail out for user defined kernels
    if (KernelClass->getKernelType()==UserOperator) break;

    // check if we need border handling
    if (Acc->getBoundaryMode() != Boundary::UNDEFINED) {
      if (Acc->getSizeX() > 1 || Acc->getSizeY() > 1) border_handling = true;
      if (Acc->getSizeX() > 1) kernel_x = true;
      if (Acc->getSizeY() > 1) kernel_y = true;
    }

    // check if we need shared memory
    if (mem_acc == READ_ONLY && Kernel->useLocalMemory(Acc)) {
      std::string sharedName = "_smem";
      sharedName += img->getNameAsString();
      use_shared = true;

      VarDecl *VD;
      QualType QT;
      // __shared__ T _smemIn[SY-1 + BSY*PPT][3 * BSX];
      // for left and right halo, add 2*BSX
      if (!emitEstimation && compilerOptions.exploreConfig()) {
        Expr *SX = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSX_EXPLORE", Ctx.IntTy, nullptr));
        Expr *BSY = createDeclRefExpr(Ctx, createVarDecl(Ctx, kernelDecl,
              "BSY_EXPLORE", Ctx.IntTy, nullptr));
        Expr *SY = BSY;

        if (Acc->getSizeX() > 1) {
          // 3*BSX
          SX = createBinaryOperator(Ctx, createIntegerLiteral(Ctx, 3), SX,
              BO_Mul, Ctx.IntTy);
        }
        // add padding to avoid bank conflicts
        SX = createBinaryOperator(Ctx, SX, createIntegerLiteral(Ctx, 1), BO_Add,
            Ctx.IntTy);

        // size_y = ceil((PPT*BSY+SY-1)/BSY)
        // -> PPT*BSY + ((SY-2)/BSY + 1) * BSY
        if (Kernel->getPixelsPerThread() > 1) {
          SY = createBinaryOperator(Ctx, SY, createIntegerLiteral(Ctx,
                static_cast<int>(Kernel->getPixelsPerThread())), BO_Mul,
              Ctx.IntTy);
        }

        if (Acc->getSizeY() > 1) {
          SY = createBinaryOperator(Ctx, SY, createBinaryOperator(Ctx,
                createParenExpr(Ctx, createBinaryOperator(Ctx,
                    createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                        static_cast<int>(Acc->getSizeY()-2)), BSY, BO_Div,
                      Ctx.IntTy), createIntegerLiteral(Ctx, 1), BO_Add,
                    Ctx.IntTy)), BSY, BO_Mul, Ctx.IntTy), BO_Add, Ctx.IntTy);
        }

        QT = Acc->getImage()->getType();
        QT = Ctx.getVariableArrayType(QT, SX, ArrayType::Normal, 0,
                SourceLocation());
        QT = Ctx.getVariableArrayType(QT, SY, ArrayType::Normal, 0,
                SourceLocation());
      } else {
        llvm::APInt SX, SY;
        SX = llvm::APInt(32, Kernel->getNumThreadsX());

        unsigned accSizeY = Acc->getSizeY();
        unsigned accSizeX = Acc->getSizeY();
        if (bReplaceVarAccSizeY) {
          accSizeY = FusionLocalVarAccSizeY;
          accSizeX = FusionLocalVarAccSizeY;
        }

        if (Acc->getSizeX() > 1) {
          if (compilerOptions.allowMisAlignedAccess()) {
            // BSX+MaskX*2
            SX += llvm::APInt(32, static_cast<int32_t>(accSizeX/2)) * llvm::APInt(32, 2);
          } else {
            // 3*BSX
            SX *= llvm::APInt(32, 3);
          }
        }
        // add padding to avoid bank conflicts
        SX += llvm::APInt(32, 1);


        // size_y = ceil((PPT*BSY+SY-1)/BSY)
        int smem_size_y =
          static_cast<int>(ceilf(
                static_cast<float>(Kernel->getPixelsPerThread() *
                  Kernel->getNumThreadsY() + accSizeY - 1) /
                static_cast<float>(Kernel->getNumThreadsY())));
        SY = llvm::APInt(32, smem_size_y*Kernel->getNumThreadsY());

        QT = Acc->getImage()->getType();
        QT = Ctx.getConstantArrayType(QT, SX, ArrayType::Normal, 0);
        QT = Ctx.getConstantArrayType(QT, SY, ArrayType::Normal, 0);
      }

      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          VD = createVarDecl(Ctx, DC, sharedName, QT, nullptr);
          VD->addAttr(CUDASharedAttr::CreateImplicit(Ctx));
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          VD = createVarDecl(Ctx, DC, sharedName, Ctx.getAddrSpaceQualType(QT,
                LangAS::opencl_local), nullptr);
          break;
      }

      // search for member name in kernel parameter list
      for (auto param : kernelDecl->parameters()) {
        std::string imgNameTemp = img->getName();
        imgNameTemp += kernelParamNameSuffix;
        // parameter name matches
        if (param->getName().equals(imgNameTemp)) {
          // store mapping between ParmVarDecl and shared memory VarDecl
          KernelDeclMapShared[param] = VD;
          KernelDeclMapAcc[param] = Acc;

          break;
        }
      }

      // add VarDecl to current kernel DeclContext
      DC->addDecl(VD);
      kernelBody.push_back(createDeclStmt(Ctx, VD));
    }
  }

  // activate boundary handling for exploration
  if (compilerOptions.exploreConfig() && use_shared) {
    border_handling = true;
    kernel_x = true;
    kernel_y = true;
  }


  SmallVector<LabelDecl *, 16> LDS;
  LabelDecl *LDExit = createLabelDecl(Ctx, kernelDecl, "BH_EXIT");
  LabelStmt *LSExit = createLabelStmt(Ctx, LDExit, nullptr);
  GotoStmt *GSExit = createGotoStmt(Ctx, LDExit);


  // only create labels if we need border handling
  for (size_t i=0; i<=9 && border_handling; ++i) {
    LabelDecl *LD;
    Expr *if_goto = nullptr;

    switch (i) {
      default:
      case 0:
        // fall back: in case the image is too small, use code variant with
        // boundary handling for all borders
        LD = createLabelDecl(Ctx, kernelDecl, "BH_FB");
        if_goto = getBHFallBack();
        break;
      case 1:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.x < bh_start_left &&
        //              blockIdx.y < bh_start_top) goto BO_TL;
        // OpenCL:  if (get_group_id(0) < bh_start_left &&
        //              get_group_id(1) < bh_start_top) goto BO_TL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_TL");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartLeft(), BO_LT, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_y, getBHStartTop(), BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 2:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.x >= bh_start_right &&
        //              blockIdx.y < bh_start_top) goto BO_TR;
        // OpenCL:  if (get_group_id(0) >= bh_start_right &&
        //              get_group_id(1) < bh_start_top) goto BO_TR;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_TR");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartRight(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_y, getBHStartTop(), BO_LT, Ctx.BoolTy), BO_LAnd,
            Ctx.BoolTy);
        break;
      case 3:
        // check if we have only a row filter
        if (!kernel_y) continue;

        // CUDA:    if (blockIdx.y < bh_start_top) goto BO_T;
        // OpenCL:  if (get_group_id(1) < bh_start_top) goto BO_T;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_T");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartTop(), BO_LT, Ctx.BoolTy);
        break;
      case 4:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom &&
        //              blockIdx.x < bh_start_left) goto BO_BL;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom &&
        //              get_group_id(0) < bh_start_left) goto BO_BL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_BL");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_x, getBHStartLeft(), BO_LT, Ctx.BoolTy),
            BO_LAnd, Ctx.BoolTy);
        break;
      case 5:
        // check if we have only a row or column filter
        if (!kernel_x || !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom &&
        //              blockIdx.x >= bh_start_right) goto BO_BR;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom &&
        //              get_group_id(0) >= bh_start_right) goto BO_BL;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_BR");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        if_goto = createBinaryOperator(Ctx, if_goto, createBinaryOperator(Ctx,
              tileVars.block_id_x, getBHStartRight(), BO_GE, Ctx.BoolTy),
            BO_LAnd, Ctx.BoolTy);
        break;
      case 6:
        // this is not required for row filter, but for kernels where the
        // iteration space is not a multiple of the block size
        if (Kernel->getNumThreadsY()<=1 && Kernel->getPixelsPerThread()<=1 &&
            !kernel_y) continue;

        // CUDA:    if (blockIdx.y >= bh_start_bottom) goto BO_B;
        // OpenCL:  if (get_group_id(1) >= bh_start_bottom) goto BO_B;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_B");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_y,
            getBHStartBottom(), BO_GE, Ctx.BoolTy);
        break;
      case 7:
        // this is not required for column filters, but for kernels where the
        // iteration space is not a multiple of the block size

        // CUDA:    if (blockIdx.x >= bh_start_right) goto BO_R;
        // OpenCL:  if (get_group_id(0) >= bh_start_right) goto BO_R;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_R");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartRight(), BO_GE, Ctx.BoolTy);
        break;
      case 8:
        // check if we have only a column filter
        if (!kernel_x) continue;

        // CUDA:    if (blockIdx.x < bh_start_left) goto BO_L;
        // OpenCL:  if (get_group_id(0) < bh_start_left) goto BO_L;
        LD = createLabelDecl(Ctx, kernelDecl, "BH_L");
        if_goto = createBinaryOperator(Ctx, tileVars.block_id_x,
            getBHStartLeft(), BO_LT, Ctx.BoolTy);
        break;
      case 9:
        LD = createLabelDecl(Ctx, kernelDecl, "BH_NO");
        break;
    }
    LDS.push_back(LD);
    Stmt *GS = createGotoStmt(Ctx, LD);
    if (if_goto)
      GS = createIfStmt(Ctx, if_goto, GS);
    kernelBody.push_back(GS);
  }

  // add casts to tileVars if required
  updateTileVars();

  int ld_count = 0;
  for (size_t i=border_handling?0:9; i<=9; ++i) {
    // set border handling mode
    switch (i) {
      case 0:
        if (kernel_y) {
          bh_variant.borders.top = 1;
          bh_variant.borders.bottom = 1;
        }
        if (kernel_x) {
          bh_variant.borders.left = 1;
          bh_variant.borders.right = 1;
        }
        break;
      case 1:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.top = 1;
        bh_variant.borders.left = 1;
        break;
      case 2:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.top = 1;
        bh_variant.borders.right = 1;
        break;
      case 3:
        if (kernel_y) bh_variant.borders.top = 1;
        else continue;
        break;
      case 4:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.bottom = 1;
        bh_variant.borders.left = 1;
        break;
      case 5:
        if (!kernel_x || !kernel_y) continue;
        bh_variant.borders.bottom = 1;
        bh_variant.borders.right = 1;
        break;
      case 6:
        // this is not required for row filter, but for kernels where the
        // iteration space is not a multiple of the block size
        if (Kernel->getNumThreadsY()>1 || Kernel->getPixelsPerThread()>1 ||
            kernel_y) bh_variant.borders.bottom = 1;
        else continue;
        break;
      case 7:
        // this is not required for column filters, but for kernels where the
        // iteration space is not a multiple of the block size
        bh_variant.borders.right = 1;
        break;
      case 8:
        if (kernel_x) bh_variant.borders.left = 1;
        else continue;
        break;
      case 9:
        break;
      default:
        break;
    }

    if (bRecordLocalKernelBorderHandeling) {
      bRecordBorderHandelingStmts = true;
      stmtsBHFusion.clear();
    }

    // if (gid_x >= is_offset_x && gid_x < is_width+is_offset_x)
    BinaryOperator *check_bop = nullptr;
    if (border_handling) {
      // if (gid_x >= is_offset_x)
      if (Kernel->getIterationSpace()->getOffsetXDecl() &&
          !(kernel_x && !bh_variant.borders.left) && bh_variant.borderVal) {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getOffsetXDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
      }
      // if (gid_x < is_width+is_offset_x)
      if (!(kernel_x && !bh_variant.borders.right) && bh_variant.borderVal) {
        BinaryOperator *check_tmp = nullptr;
        if (Kernel->getIterationSpace()->getOffsetXDecl()) {
          check_tmp = createBinaryOperator(Ctx, tileVars.global_id_x,
              createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()),
                getOffsetXDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy),
              BO_LT, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, tileVars.global_id_x,
              getWidthDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
        }
        if (check_bop) {
          check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
              Ctx.BoolTy);
        } else {
          check_bop = check_tmp;
        }
      }
      // Renderscript iteration space is always the whole image, so we need to
      // check the y-dimension as well:
      // if (gid_y >= is_offset_y && gid_y < is_height+is_offset_y)
      if (compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        // if (gid_y >= is_offset_y)
        if (Kernel->getIterationSpace()->getOffsetYDecl() &&
            !(kernel_y && !bh_variant.borders.left) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = nullptr;
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
          if (check_bop) {
            check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
                Ctx.BoolTy);
          } else {
            check_bop = check_tmp;
          }
        }
        // if (gid_y < is_height+is_offset_y)
        if (!(kernel_y && !bh_variant.borders.right) && bh_variant.borderVal) {
          BinaryOperator *check_tmp = nullptr;
          if (Kernel->getIterationSpace()->getOffsetYDecl()) {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()),
                  getOffsetYDecl(Kernel->getIterationSpace()), BO_Add,
                  Ctx.IntTy), BO_LT, Ctx.BoolTy);
          } else {
            check_tmp = createBinaryOperator(Ctx, gidYRef,
                getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
          }
          if (check_bop) {
            check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
                Ctx.BoolTy);
          } else {
            check_bop = check_tmp;
          }
        }
      }
    } else {
      // if (gid_x < is_width+is_offset_x)
      if (Kernel->getIterationSpace()->getOffsetXDecl()) {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getOffsetXDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
        check_bop = createBinaryOperator(Ctx, check_bop,
            createBinaryOperator(Ctx, tileVars.global_id_x,
              createBinaryOperator(Ctx,
                getWidthDecl(Kernel->getIterationSpace()),
                getOffsetXDecl(Kernel->getIterationSpace()), BO_Add, Ctx.IntTy),
              BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
      } else {
        check_bop = createBinaryOperator(Ctx, tileVars.global_id_x,
            getWidthDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
      }
      // if (gid_y >= is_offset_y && gid_y < is_height+is_offset_y)
      // Renderscript iteration space is always the whole image, so we need to
      // check the y-dimension as well.
      if (compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        // if (gid_y < is_height+is_offset_y)
        BinaryOperator *check_tmp = nullptr;
        if (Kernel->getIterationSpace()->getOffsetYDecl()) {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getOffsetYDecl(Kernel->getIterationSpace()), BO_GE, Ctx.BoolTy);
          check_tmp = createBinaryOperator(Ctx, check_tmp,
              createBinaryOperator(Ctx, gidYRef, createBinaryOperator(Ctx,
                  getHeightDecl(Kernel->getIterationSpace()),
                  getOffsetYDecl(Kernel->getIterationSpace()), BO_Add,
                  Ctx.IntTy), BO_LT, Ctx.BoolTy), BO_LAnd, Ctx.BoolTy);
        } else {
          check_tmp = createBinaryOperator(Ctx, gidYRef,
              getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
        }
        if (check_bop) {
          check_bop = createBinaryOperator(Ctx, check_bop, check_tmp, BO_LAnd,
              Ctx.BoolTy);
        } else {
          check_bop = check_tmp;
        }
      }
    }


    // stage pixels into shared memory
    // ppt + ceil((size_y-1)/sy) iterations
    int p_add = 0;
    unsigned maxSizeYUndef = Kernel->getMaxSizeYUndef();
    if (bReplaceVarAccSizeY) {
      maxSizeYUndef = FusionLocalVarAccSizeY >> 1;
    }

    if (maxSizeYUndef) {
      p_add = static_cast<int>(ceilf(2*maxSizeYUndef /
            static_cast<float>(Kernel->getNumThreadsY())));
    }
    SmallVector<Stmt *, 16> labelBody;

    for (size_t p=0; use_shared && p<Kernel->getPixelsPerThread()+p_add; ++p) {
      if (compilerOptions.exploreConfig()) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
        // all iterations
        stageIterationToSharedMemoryExploration(labelBody);

        break;
      }
      if (p==0) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
        // first iteration
        stageIterationToSharedMemory(labelBody, p);
      } else {
        // update lid_y to lid_y + p*(int)local_size_y
        // update gid_y to gid_y + p*(int)local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                static_cast<int32_t>(p)), tileVars.local_size_y, BO_Mul,
              Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                static_cast<int32_t>(p)), tileVars.local_size_y, BO_Mul,
              Ctx.IntTy), BO_Add, Ctx.IntTy);
        // load next iteration to shared memory
        stageIterationToSharedMemory(labelBody, p);
      }
    }
    // synchronize shared memory
    if (use_shared) {
      // add memory barrier synchronization
      SmallVector<Expr *, 16> args;
      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          // CLK_LOCAL_MEM_FENCE -> 1
          // CLK_GLOBAL_MEM_FENCE -> 2
          args.push_back(createIntegerLiteral(Ctx, 1));
          labelBody.push_back(createFunctionCall(Ctx, barrier, args));
          break;
      }
    }

    for (size_t p=0; p<Kernel->getPixelsPerThread(); ++p) {
      // clear all stored decls before cloning, otherwise existing
      // VarDecls will be reused and we will miss declarations
      KernelDeclMap.clear();

      // calculate multiple pixels per thread
      SmallVector<Stmt *, 16> pptBody;

      if (p==0) {
        // initialize lid_y and gid_y
        lidYRef = tileVars.local_id_y;
        gidYRef = tileVars.global_id_y;
      } else {
        // update lid_y to lid_y + p*(int)local_size_y
        // update gid_y to gid_y + p*(int)local_size_y
        lidYRef = createBinaryOperator(Ctx, tileVars.local_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                static_cast<int32_t>(p)), tileVars.local_size_y, BO_Mul,
              Ctx.IntTy), BO_Add, Ctx.IntTy);
        gidYRef = createBinaryOperator(Ctx, tileVars.global_id_y,
            createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                static_cast<int32_t>(p)), tileVars.local_size_y, BO_Mul,
              Ctx.IntTy), BO_Add, Ctx.IntTy);
      }

      // convert kernel function body to CUDA/OpenCL kernel syntax
      Stmt *new_body = Clone(S);
      assert(isa<CompoundStmt>(new_body) && "CompoundStmt for kernel function body expected!");

      // add iteration space check when calculating multiple pixels per thread,
      // having a tiling with multiple threads in the y-dimension, or in case
      // exploration is done
      bool require_is_check = true;
      if (border_handling) {
        // code variant for column filter not processing the bottom
        if (kernel_y && !bh_variant.borders.bottom) require_is_check = false;
        // code variant without border handling
        if (!bh_variant.borderVal && !compilerOptions.exploreConfig())
          require_is_check = false;
        // number of threads is 1 and no exploration
        if (Kernel->getNumThreadsY()==1 && Kernel->getPixelsPerThread()==1 &&
            !compilerOptions.exploreConfig())
          require_is_check = false;
      } else {
        // exploration
        if (Kernel->getNumThreadsY()==1 && Kernel->getPixelsPerThread()==1 &&
            !compilerOptions.exploreConfig()) require_is_check = false;
      }
      if (require_is_check &&
          // Not necessary for Filterscript, gid_y has already been checked
          !compilerOptions.emitFilterscript()) {
        // if (gid_y + p < is_height)
        BinaryOperator *inner_check_bop = createBinaryOperator(Ctx, gidYRef,
            getHeightDecl(Kernel->getIterationSpace()), BO_LT, Ctx.BoolTy);
        IfStmt *inner_ispace_check = createIfStmt(Ctx, inner_check_bop,
            new_body);
        pptBody.push_back(inner_ispace_check);
      } else {
        pptBody.push_back(new_body);
      }

      // TODO remove this, enable this only for bandwidth profiling
      //pptBody.clear();

      // TODO whole body replacement for kernel fusion
      if (bReplaceLocalKernelBody) {
        pptBody.clear();
        pptBody.push_back(stmtsProducerL2LKernelFusion.front());
        stmtsProducerL2LKernelFusion.pop();
      }

      // add iteration space checking in case we have padded images and/or
      // padded block/grid configurations
      // TODO, disabled for optimization, assume same granularity
      //if (check_bop && !bReplaceInputIdxExpr) {
      if (check_bop) {
        IfStmt *ispace_check = createIfStmt(Ctx, check_bop,
            createCompoundStmt(Ctx, pptBody));
        labelBody.push_back(ispace_check);
      } else {
        for (auto stmt : pptBody)
          labelBody.push_back(stmt);
      }

      // record ppt body for l2l fusion
      if (bRecordLocalKernelBody) {
        stmtsL2LKernelFusion.push(createCompoundStmt(Ctx, pptBody));
      }
    }
    // add label statement if needed (boundary handling), else add body
    if (border_handling) {
      LabelStmt *LS = createLabelStmt(Ctx, LDS[ld_count++],
          createCompoundStmt(Ctx, labelBody));
      kernelBody.push_back(LS);
      kernelBody.push_back(GSExit);
    } else {
      kernelBody.push_back(createCompoundStmt(Ctx, labelBody));
    }

    if (bInsertLocalKernelBody) {
      stmtsProducerL2LKernelFusion.pop();
    }

    // reset image border configuration
    bh_variant.borderVal = 0;
    // reset lid_y and gid_y
    lidYRef = tileVars.local_id_y;
    gidYRef = tileVars.global_id_y;
  }

  if (border_handling) {
    kernelBody.push_back(LSExit);
  }

  if (compilerOptions.emitFilterscript()) {
    // in case no value was written, return the value of the iteration space
    Expr *result = accessMem(outputImage, Kernel->getIterationSpace(),
        READ_ONLY);
    setExprProps(outputImage, result);
    kernelBody.push_back(createReturnStmt(Ctx, result));
  }

  return createCompoundStmt(Ctx, kernelBody);
}


VarDecl *ASTTranslateFusion::CloneVarDecl(VarDecl *VD) {
  VarDecl *result = KernelDeclMap[VD];

  if (!result && (convMask || !redDomains.empty()))
    result = LambdaDeclMap[VD];

  if (!result) {
    QualType QT = VD->getType();
    TypeSourceInfo *TInfo = VD->getTypeSourceInfo();
    std::string name = VD->getName();

    if (KernelClass->getKernelType() == LocalOperator &&
        KernelClass->getVarDeclByName(name)) {
      name += kernelParamNameSuffix;
    }

    if (Kernel->vectorize() && KernelClass->getVectorizeInfo(VD) == VECTORIZE &&
        !compilerOptions.emitC99()) {
      QT = simdTypes.getSIMDType(VD, SIMD4);
      TInfo = Ctx.getTrivialTypeSourceInfo(QT);
    }

    DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
    result = VarDecl::Create(Ctx, DC, VD->getInnerLocStart(), VD->getLocation(),
        &Ctx.Idents.get(name), QT, TInfo, VD->getStorageClass());
    result->setIsUsed(); // set VarDecl as being used - required for CodeGen
    if (Kernel->vectorize() && KernelClass->getVectorizeInfo(VD) == VECTORIZE &&
        !compilerOptions.emitC99() ) {
      result->setInit(simdTypes.propagate(VD, Clone(VD->getInit())));
    } else {
      result->setInit(Clone(VD->getInit()));
    }
    result->setInitStyle(VD->getInitStyle());
    result->setTSCSpec(VD->getTSCSpec());

    // store mapping between original VarDecl and cloned VarDecl
    if (convMask || !redDomains.empty()) {
      LambdaDeclMap[VD] = result;
      LambdaDeclMap[result] = result;
    } else {
      KernelDeclMap[VD] = result;
      KernelDeclMap[result] = result;
    }

    // add VarDecl to current kernel DeclContext
    DC->addDecl(result);
  }

  return result;
}



Stmt *ASTTranslateFusion::VisitCompoundStmtTranslate(CompoundStmt *S) {
  CompoundStmt *result = new (Ctx) CompoundStmt(Ctx, MultiStmtArg(),
      S->getLBracLoc(), S->getLBracLoc());

  SmallVector<Stmt *, 16> body;
  for (auto stmt : S->body()) {
    curCStmt = S;
    Stmt *newS = Clone(stmt);
    curCStmt = S;

    if (preStmts.size()) {
      size_t num_stmts = 0;
      for (size_t i=0, e=preStmts.size(); i!=e; ++i) {
        if (preCStmt[i]==S) {
          body.push_back(preStmts[i]);
          num_stmts++;
        }
      }
      for (size_t i=0; i<num_stmts; ++i) {
        preStmts.pop_back();
        preCStmt.pop_back();
      }
    }

    if (bInsertLocalKernelBody && bInsertBeforeSmem) {
      Expr *offset_x, *offset_y;
      offset_x = createBinaryOperator(Ctx, exprIdXShiftFusion,
                    createIntegerLiteral(Ctx, curIdxXShiftFusion),
                      BO_Assign, Ctx.IntTy);
      offset_y = createBinaryOperator(Ctx, exprIdYShiftFusion,
                    createIntegerLiteral(Ctx, curIdxYShiftFusion),
                      BO_Assign, Ctx.IntTy);
      body.push_back(offset_x);
      body.push_back(offset_y);

      for (auto stmt : stmtsBHFusion) {
        body.push_back(stmt);
      }

      body.push_back(stmtsProducerL2LKernelFusion.front());
      bInsertBeforeSmem = false;
    }

    if (newS) body.push_back(newS);

    if (postStmts.size()) {
      size_t num_stmts = 0;
      for (size_t i=0, e=postStmts.size(); i!=e; ++i) {
        if (postCStmt[i]==S) {
          body.push_back(postStmts[i]);
          num_stmts++;
        }
      }
      for (size_t i=0; i<num_stmts; ++i) {
        postStmts.pop_back();
        postCStmt.pop_back();
      }
    }
  }

  result->setStmts(Ctx, body);

  return result;
}

// add border handling: CLAMP
Stmt *clamp_upper_fuse(ASTContext &Ctx, Expr *idx, Expr *upper, Expr *) {
  // if (idx >= upper) idx = upper-1;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createIntegerLiteral(Ctx, 1), BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr, nullptr);
}
Stmt *clamp_lower_fuse(ASTContext &Ctx,  Expr *idx, Expr *lower, Expr *) {
  // if (idx < lower) idx = lower;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx, lower,
        BO_Assign, Ctx.IntTy), nullptr, nullptr);
}


// add border handling: REPEAT
Stmt *repeat_upper_fuse(ASTContext &Ctx, Expr *idx, Expr *upper, Expr *stride) {
  // while (idx >= upper) idx -= is_width | is_height;
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createWhileStmt(Ctx, nullptr, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, stride, BO_Sub, Ctx.IntTy), BO_Assign,
        Ctx.IntTy));
}
Stmt *repeat_lower_fuse(ASTContext &Ctx, Expr *idx, Expr *lower, Expr *stride) {
  // while (idx < lower) idx += is_width | is_height;
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createWhileStmt(Ctx, nullptr, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, idx, stride, BO_Add, Ctx.IntTy), BO_Assign,
        Ctx.IntTy));
}


// add border handling: MIRROR
Stmt *mirror_upper_fuse(ASTContext &Ctx, Expr *idx, Expr *upper, Expr *) {
  // if (idx >= upper) idx = upper - (idx+1 - upper);
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_GE, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_upper, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, upper, createParenExpr(Ctx,
            createBinaryOperator(Ctx, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Add, Ctx.IntTy),
              createParenExpr(Ctx, upper), BO_Sub, Ctx.IntTy)) , BO_Sub,
          Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr, nullptr);
}
Stmt *mirror_lower_fuse(ASTContext &Ctx, Expr *idx, Expr *lower, Expr *) {
  // if (idx < lower) idx = lower + (lower - idx-1);
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_LT, Ctx.BoolTy);

  return createIfStmt(Ctx, bo_lower, createBinaryOperator(Ctx, idx,
        createBinaryOperator(Ctx, lower, createParenExpr(Ctx,
            createBinaryOperator(Ctx, lower, createBinaryOperator(Ctx, idx,
                createIntegerLiteral(Ctx, 1), BO_Sub, Ctx.IntTy), BO_Sub,
              Ctx.IntTy)) , BO_Add, Ctx.IntTy), BO_Assign, Ctx.IntTy), nullptr,
      nullptr);
}


// add border handling: CONSTANT
Expr *constant_upper_fuse(ASTContext &Ctx, Expr *idx, Expr *upper, Expr *cond) {
  // (idx < upper)
  Expr *bo_upper = createBinaryOperator(Ctx, idx, upper, BO_LT, Ctx.BoolTy);

  if (cond) {
    return createBinaryOperator(Ctx, bo_upper, cond, BO_LAnd, Ctx.BoolTy);
  }
  return bo_upper;
}
Expr *constant_lower_fuse(ASTContext &Ctx, Expr *idx, Expr *lower, Expr *cond) {
  // (idx >= lower)
  Expr *bo_lower = createBinaryOperator(Ctx, idx, lower, BO_GE, Ctx.BoolTy);

  if (cond) {
    return createBinaryOperator(Ctx, bo_lower, cond, BO_LAnd, Ctx.BoolTy);
  }
  return bo_lower;
}

// add border handling statements to the AST
Expr *ASTTranslateFusion::addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x,
    Expr *local_offset_y, HipaccAccessor *Acc) {
  return addBorderHandling(LHS, local_offset_x, local_offset_y, Acc, preStmts,
      preCStmt);
}

Expr *ASTTranslateFusion::addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x,
    Expr *local_offset_y, HipaccAccessor *Acc, SmallVector<Stmt *, 16> &bhStmts,
    SmallVector<CompoundStmt *, 16> &bhCStmt) {
  Expr *result = nullptr;
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);

  bool bRecordStmtsForKernelFusion = false;
  if (bRecordLocalKernelBorderHandeling && bRecordBorderHandelingStmts) {
    bRecordStmtsForKernelFusion = true;
  }

  std::string gidx_str;
  std::string gidy_str;
  std::string tmp_str;
  if (bRecordStmtsForKernelFusion) {
    gidx_str = "_gid_x_fusion" + std::to_string(FusionLocalLiteralCount);
    gidy_str = "_gid_y_fusion" + std::to_string(FusionLocalLiteralCount);
    tmp_str = "_tmp_fusion" + std::to_string(FusionLocalLiteralCount++);
  }
  else {
    gidx_str = "_gid_x" + std::to_string(literalCount);
    gidy_str = "_gid_y" + std::to_string(literalCount);
    tmp_str = "_tmp" + std::to_string(literalCount++);
  }

  Expr *lower_x, *upper_x, *lower_y, *upper_y;
  if (Acc->getOffsetXDecl()) {
    lower_x = getOffsetXDecl(Acc);
    upper_x = createBinaryOperator(Ctx, getOffsetXDecl(Acc), getWidthDecl(Acc),
        BO_Add, Ctx.IntTy);
  } else {
    lower_x = createIntegerLiteral(Ctx, 0);
    upper_x = getWidthDecl(Acc);
  }
  if (Acc->getOffsetYDecl()) {
    lower_y = getOffsetYDecl(Acc);
    upper_y = createBinaryOperator(Ctx, getOffsetYDecl(Acc), getHeightDecl(Acc),
        BO_Add, Ctx.IntTy);
  } else {
    lower_y = createIntegerLiteral(Ctx, 0);
    upper_y = getHeightDecl(Acc);
  }

  Expr *idx_x = tileVars.global_id_x;
  Expr *idx_y = gidYRef;
  Expr *idx_x_fusion = tileVars.global_id_x;
  Expr *idx_y_fusion = gidYRef;

  // step 0: add local offset: gid_[x|y] + local_offset_[x|y]
  idx_x = addLocalOffset(idx_x, local_offset_x);
  idx_y = addLocalOffset(idx_y, local_offset_y);
  if (bRecordStmtsForKernelFusion) {
    idx_x_fusion = addLocalOffset(idx_x_fusion, exprIdXShiftFusion);
    idx_y_fusion = addLocalOffset(idx_y_fusion, exprIdYShiftFusion);
  }
  // step 1: remove is_offset and add interpolation & boundary handling
  switch (Acc->getInterpolationMode()) {
    case Interpolate::NO:
      if (Acc!=Kernel->getIterationSpace()) {
        idx_x = removeISOffsetX(idx_x);
      }
      if ((compilerOptions.emitC99() ||
           compilerOptions.emitRenderscript() ||
           compilerOptions.emitFilterscript()) &&
          Acc!=Kernel->getIterationSpace()) {
        idx_y = removeISOffsetY(idx_y);
      }
      break;
    case Interpolate::NN:
      idx_x = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationX(Acc, idx_x)), nullptr,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      idx_y = createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
          createParenExpr(Ctx, addNNInterpolationY(Acc, idx_y)), nullptr,
          Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      break;
    case Interpolate::LF:
    case Interpolate::CF:
    case Interpolate::L3:
      return addInterpolationCall(LHS, Acc, idx_x, idx_y);
  }

  // step 2: add global Accessor/Iteration Space offset
  if (Acc!=Kernel->getIterationSpace()) {
    idx_x = addGlobalOffsetX(idx_x, Acc);
    idx_y = addGlobalOffsetY(idx_y, Acc);
  } else {
    if (!(compilerOptions.emitC99() ||
          compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript())) {
      idx_y = addGlobalOffsetY(idx_y, Acc);
    }
  }

  // TODO
  if (bRecordStmtsForKernelFusion) {
    idx_x = idx_x_fusion;
    idx_y = idx_y_fusion;
  }
  // add temporary variables for updated idx_x and idx_y
  if (local_offset_x) {
    VarDecl *tmp_x = createVarDecl(Ctx, kernelDecl, gidx_str, Ctx.IntTy, idx_x);
    DC->addDecl(tmp_x);
    idx_x = createDeclRefExpr(Ctx, tmp_x);
    Stmt *bhTemp = createDeclStmt(Ctx, tmp_x);
    if (bRecordStmtsForKernelFusion)
      (stmtsBHFusion).push_back(bhTemp);
    bhStmts.push_back(bhTemp);
    bhCStmt.push_back(curCStmt);
  }

  if (local_offset_y) {
    VarDecl *tmp_y = createVarDecl(Ctx, kernelDecl, gidy_str, Ctx.IntTy, idx_y);
    DC->addDecl(tmp_y);
    idx_y = createDeclRefExpr(Ctx, tmp_y);
    Stmt *bhTemp = createDeclStmt(Ctx, tmp_y);
    if (bRecordStmtsForKernelFusion)
      (stmtsBHFusion).push_back(bhTemp);
    bhStmts.push_back(bhTemp);
    bhCStmt.push_back(curCStmt);
  }

  if (Acc->getBoundaryMode() == Boundary::CONSTANT) {
    // <type> _tmp<0> = const_val;
    Expr *RHS = nullptr;
    Expr *const_val = Acc->getConstExpr();
    VarDecl *tmp_t = createVarDecl(Ctx, kernelDecl, tmp_str,
        const_val->getType(), const_val);

    DC->addDecl(tmp_t);
    DeclRefExpr *tmp_t_ref = createDeclRefExpr(Ctx, tmp_t);

    Stmt *bhTemp = createDeclStmt(Ctx, tmp_t);
    bhStmts.push_back(bhTemp);
    if (bRecordStmtsForKernelFusion)
      (stmtsBHFusion).push_back(bhTemp);
    bhCStmt.push_back(curCStmt);

    Expr *bo_constant = nullptr;
    if (bh_variant.borders.right && local_offset_x) {
      // < _gid_x<0> >= offset_x+width >
      bo_constant = constant_upper_fuse(Ctx, idx_x, upper_x, bo_constant);
    }
    if (bh_variant.borders.bottom && local_offset_y) {
      // if (_gid_y<0> >= offset_y+height)
      bo_constant = constant_upper_fuse(Ctx, idx_y, upper_y, bo_constant);
    }
    if (bh_variant.borders.left && local_offset_x) {
      // if (_gid_x<0> < offset_x)
      bo_constant = constant_lower_fuse(Ctx, idx_x, lower_x, bo_constant);
    }
    if (bh_variant.borders.top && local_offset_y) {
      // if (_gid_y<0> < offset_y)
      bo_constant = constant_lower_fuse(Ctx, idx_y, lower_y, bo_constant);
    }

    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
          RHS = accessMem2DAt(LHS, idx_x, idx_y);
          break;
      case Language::CUDA:
        if (Kernel->useTextureMemory(Acc) != Texture::None) {
          RHS = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        // fall through
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (Kernel->useTextureMemory(Acc) != Texture::None) {
          RHS = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        RHS = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        RHS = accessMemAllocAt(LHS, READ_ONLY, idx_x, idx_y);
        break;
    }
    setExprProps(LHS, RHS);

    // tmp<0> = RHS;
    if (bo_constant) {
      Stmt *bhTemp = createIfStmt(Ctx, bo_constant, createBinaryOperator(Ctx,
                      tmp_t_ref, RHS, BO_Assign, tmp_t_ref->getType()), nullptr,
                        nullptr);
      bhStmts.push_back(bhTemp);
      if (bRecordStmtsForKernelFusion)
        (stmtsBHFusion).push_back(bhTemp);
      bhCStmt.push_back(curCStmt);
    } else {
      Stmt *bhTemp = createBinaryOperator(Ctx, tmp_t_ref, RHS, BO_Assign,
                  tmp_t_ref->getType());
      bhStmts.push_back(bhTemp);
      if (bRecordStmtsForKernelFusion)
        (stmtsBHFusion).push_back(bhTemp);
      bhCStmt.push_back(curCStmt);
    }
    result = tmp_t_ref;
  } else {
    std::function<Stmt*(ASTContext &, Expr *, Expr *, Expr *)>
      lower_fun = nullptr, upper_fun = nullptr;
    switch (Acc->getBoundaryMode()) {
      case Boundary::CLAMP:  lower_fun = clamp_lower_fuse;
                             upper_fun = clamp_upper_fuse;
                             break;
      case Boundary::REPEAT: lower_fun = repeat_lower_fuse;
                             upper_fun = repeat_upper_fuse;
                             break;
      case Boundary::MIRROR: lower_fun = mirror_lower_fuse;
                             upper_fun = mirror_upper_fuse;
                             break;
      case Boundary::UNDEFINED:
        // in case of exploration boundary handling variants are required
        if (!compilerOptions.exploreConfig()) {
          assert(0 && "addBorderHandling && Boundary::UNDEFINED!");
        }
        break;
      case Boundary::CONSTANT:
        assert(0 && "addBorderHandling && Boundary::CONSTANT!");
        break;
    }

    auto stride_x = getWidthDecl(Acc);
    auto stride_y = getHeightDecl(Acc);
    if (upper_fun) {
      if (bh_variant.borders.right && local_offset_x) {
        Stmt *bhTemp = upper_fun(Ctx, idx_x, upper_x, stride_x);
        bhStmts.push_back(bhTemp);
        if (bRecordStmtsForKernelFusion)
          (stmtsBHFusion).push_back(bhTemp);
        bhCStmt.push_back(curCStmt);
      }
      if (bh_variant.borders.bottom && local_offset_y) {
        Stmt *bhTemp = upper_fun(Ctx, idx_y, upper_y, stride_y);
        bhStmts.push_back(bhTemp);
        if (bRecordStmtsForKernelFusion)
          (stmtsBHFusion).push_back(bhTemp);
        bhCStmt.push_back(curCStmt);
      }
    }
    if (lower_fun) {
      if (bh_variant.borders.left && local_offset_x) {
        Stmt *bhTemp = lower_fun(Ctx, idx_x, lower_x, stride_x);
        bhStmts.push_back(bhTemp);
        if (bRecordStmtsForKernelFusion)
          (stmtsBHFusion).push_back(bhTemp);
        bhCStmt.push_back(curCStmt);
      }
      if (bh_variant.borders.top && local_offset_y) {
        Stmt *bhTemp = lower_fun(Ctx, idx_y, lower_y, stride_y);
        bhStmts.push_back(bhTemp);
        if (bRecordStmtsForKernelFusion)
          (stmtsBHFusion).push_back(bhTemp);
        bhCStmt.push_back(curCStmt);
      }
    }

    // get data
    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
          result = accessMem2DAt(LHS, idx_x, idx_y);
          break;
      case Language::CUDA:
        if (Kernel->useTextureMemory(Acc) != Texture::None) {
          result = accessMemTexAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        // fall through
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (Kernel->useTextureMemory(Acc) != Texture::None) {
          result = accessMemImgAt(LHS, Acc, READ_ONLY, idx_x, idx_y);
          break;
        }
        result = accessMemArrAt(LHS, getStrideDecl(Acc), idx_x, idx_y);
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        result = accessMemAllocAt(LHS, READ_ONLY, idx_x, idx_y);
        break;
    }
    setExprProps(LHS, result);
  }

  if (bRecordStmtsForKernelFusion) {
    Expr *end_offset_x, *end_offset_y;
    end_offset_x = createBinaryOperator(Ctx, exprIdXShiftFusion,
                    createBinaryOperator(Ctx, idx_x, tileVars.global_id_x, BO_Sub, Ctx.IntTy),
                      BO_Assign, Ctx.IntTy);
    end_offset_y = createBinaryOperator(Ctx, exprIdYShiftFusion,
                    createBinaryOperator(Ctx, idx_y, gidYRef, BO_Sub, Ctx.IntTy),
                      BO_Assign, Ctx.IntTy);
    (stmtsBHFusion).push_back(end_offset_x);
    (stmtsBHFusion).push_back(end_offset_y);

    bRecordBorderHandelingStmts = false;
  }
  return result;
}

Expr *ASTTranslateFusion::VisitCXXOperatorCallExprTranslate(CXXOperatorCallExpr *E) {
  Expr *result = nullptr;

  // assume that all CXXOperatorCallExpr are memory access functions, since we
  // don't support function calls
  assert(isa<MemberExpr>(E->getArg(0)) && "Memory access function assumed.");
  MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(0));

  // get FieldDecl of the MemberExpr
  assert(isa<FieldDecl>(ME->getMemberDecl()) && "Image must be a C++-class member.");
  FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());

  // MemberExpr is converted to DeclRefExpr when cloning
  DeclRefExpr *LHS = dyn_cast<DeclRefExpr>(Clone(E->getArg(0)));


  // look for Mask user class member variable
  if (auto mask = Kernel->getMaskFromMapping(FD)) {
    MemoryAccess mem_acc = KernelClass->getMemAccess(FD);
    assert(mem_acc == READ_ONLY &&
        "only read-only memory access to Mask supported");

    switch (E->getNumArgs()) {
      default:
        assert(0 && "0, 1, or 2 arguments for Mask operator() expected!");
        break;
      case 1:
        assert(convMask && convMask == mask &&
            "0 arguments for Mask operator() only allowed within"
            "convolution lambda-function.");
        // within convolute lambda-function
        if (mask->isConstant()) {
          // propagate constants
          result = Clone(mask->getInitExpr(convIdxX, convIdxY));
        } else {
          // access mask elements
          Expr *midx_x = createIntegerLiteral(Ctx, convIdxX);
          Expr *midx_y = createIntegerLiteral(Ctx, convIdxY);

          // set Mask as being used within Kernel
          Kernel->setUsed(FD->getNameAsString());

          switch (compilerOptions.getTargetLang()) {
            case Language::C99:
            case Language::CUDA:
              // array subscript: Mask[conv_y][conv_x]
              result = accessMem2DAt(LHS, midx_x, midx_y);
              break;
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              // array subscript: Mask[(conv_y)*width + conv_x]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    static_cast<int>(mask->getSizeX())), midx_x, midx_y);
              break;
            case Language::Renderscript:
            case Language::Filterscript:
              // allocation access: rsGetElementAt(Mask, conv_x, conv_y)
              result = accessMemAllocAt(LHS, mem_acc, midx_x, midx_y);
              break;
          }
        }
        break;
      case 2:
        // 0: -> (this *) Mask class
        // 1: -> (dom) Domain class
        {
        assert(isa<MemberExpr>(E->getArg(1)) && "Memory access function assumed.");
        MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(1));
        assert(isa<FieldDecl>(ME->getMemberDecl()) && "Domain must be a C++-class member.");
        FieldDecl *domFD = dyn_cast<FieldDecl>(ME->getMemberDecl());

        // look for Domain user class member variable
        assert(Kernel->getMaskFromMapping(domFD) && "Could not find Domain variable.");
        HipaccMask *Domain = Kernel->getMaskFromMapping(domFD);
        (void)Domain; // silent compiler warning
        assert(Domain->isDomain() && "Domain required.");

        assert(mask->getSizeX()==Domain->getSizeX() &&
               mask->getSizeY()==Domain->getSizeY() &&
               "Mask and Domain size must be equal.");

        // within reduce/iterate lambda-function
        if (mask->isConstant()) {
          // propagate constants
          result = Clone(mask->getInitExpr(redIdxX.back(), redIdxY.back()));
        } else {
          // access mask elements
          Expr *midx_x = createIntegerLiteral(Ctx, redIdxX.back());
          Expr *midx_y = createIntegerLiteral(Ctx, redIdxY.back());

          // set Mask as being used within Kernel
          Kernel->setUsed(FD->getNameAsString());

          switch (compilerOptions.getTargetLang()) {
            case Language::C99:
            case Language::CUDA:
              // array subscript: Mask[conv_y][conv_x]
              result = accessMem2DAt(LHS, midx_x, midx_y);
              break;
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              // array subscript: Mask[(conv_y)*width + conv_x]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    static_cast<int>(mask->getSizeX())), midx_x, midx_y);
              break;
            case Language::Renderscript:
            case Language::Filterscript:
              // allocation access: rsGetElementAt(Mask, conv_x, conv_y)
              result = accessMemAllocAt(LHS, mem_acc, midx_x, midx_y);
              break;
          }
        }
        }
        break;
      case 3:
        // 0: -> (this *) Mask class
        // 1: -> x
        // 2: -> y

        // set Mask as being used within Kernel
        Kernel->setUsed(FD->getNameAsString());

        switch (compilerOptions.getTargetLang()) {
          case Language::C99:
          case Language::CUDA:
            // array subscript: Mask[y+size_y/2][x+size_x/2]
            result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                  Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                    static_cast<int>(mask->getSizeX()/2)), BO_Add, Ctx.IntTy),
                createBinaryOperator(Ctx, Clone(E->getArg(2)),
                  createIntegerLiteral(Ctx,
                    static_cast<int>(mask->getSizeY()/2)), BO_Add, Ctx.IntTy));
            break;
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            if (mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeX()/2)), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeY()/2)), BO_Add,
                    Ctx.IntTy));
            } else {
              // array subscript: Mask[(y+size_y/2)*width + x+size_x/2]
              result = accessMemArrAt(LHS, createIntegerLiteral(Ctx,
                    static_cast<int>(mask->getSizeX())),
                  createBinaryOperator(Ctx, Clone(E->getArg(1)),
                    createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeX()/2)), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeY()/2)), BO_Add,
                    Ctx.IntTy));
            }
            break;
          case Language::Renderscript:
          case Language::Filterscript:
            if (mask->isConstant()) {
              // array subscript: Mask[y+size_y/2][x+size_x/2]
              result = accessMem2DAt(LHS, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeX()/2)), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeY()/2)), BO_Add,
                    Ctx.IntTy));
            } else {
              // allocation access: rsGetElementAt(Mask, x+size_x/2, y+size_y/2)
              result = accessMemAllocAt(LHS, mem_acc, createBinaryOperator(Ctx,
                    Clone(E->getArg(1)), createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeX()/2)), BO_Add, Ctx.IntTy),
                  createBinaryOperator(Ctx, Clone(E->getArg(2)),
                    createIntegerLiteral(Ctx,
                      static_cast<int>(mask->getSizeY()/2)), BO_Add,
                    Ctx.IntTy));
            }
            break;
        }
        break;
    }
  }


  // look for Image user class member variable
  if (auto acc = Kernel->getImgFromMapping(FD)) {
    MemoryAccess mem_acc = KernelClass->getMemAccess(FD);

    if (bReplaceInputExpr && KernelClass->getKernelType() == PointOperator) {
      return LHS;
    }

    // Images are ParmVarDecls
    bool use_shared = false;
    DeclRefExpr *DRE = nullptr;
    if (!Kernel->vectorize()) { // Images are replaced by local pointers
      ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(LHS->getDecl());
      assert(PVD && "Image variable must be a ParmVarDecl!");

      if (KernelDeclMapShared[PVD]) {
        // shared/local memory
        use_shared = true;
        VarDecl *VD = KernelDeclMapShared[PVD];
        DRE = createDeclRefExpr(Ctx, VD);
      }
    }

    Expr *SY, *TX;
    Expr *SYOld, *TXOld;
    if (bReplaceVarAccSizeY) {
      // Index fusion for local kernel fusion
      if (acc->getSizeX() > 1) {
        if (compilerOptions.exploreConfig()) {
          TXOld = tileVars.local_size_x;
        } else if (compilerOptions.allowMisAlignedAccess()) {
          TXOld = createIntegerLiteral(Ctx, static_cast<int>(FusionLocalVarAccSizeY/2));
        } else {
          TXOld = createIntegerLiteral(Ctx, static_cast<int>(Kernel->getNumThreadsX()));
        }
      } else {
        TXOld = createIntegerLiteral(Ctx, 0);
      }
      TX = createBinaryOperator(Ctx, exprIdXShiftFusion, TXOld, BO_Add, Ctx.IntTy);
      if (acc->getSizeY() > 1) {
        SYOld = createIntegerLiteral(Ctx, static_cast<int>(FusionLocalVarAccSizeY/2));
      } else {
        SYOld = createIntegerLiteral(Ctx, 0);
      }
      SY = createBinaryOperator(Ctx, exprIdYShiftFusion, SYOld, BO_Add, Ctx.IntTy);
    } else {
      if (acc->getSizeX() > 1) {
        if (compilerOptions.exploreConfig()) {
          TX = tileVars.local_size_x;
        } else if (compilerOptions.allowMisAlignedAccess()) {
          TX = createIntegerLiteral(Ctx, static_cast<int>(acc->getSizeY()/2));
        } else {
          TX = createIntegerLiteral(Ctx, static_cast<int>(Kernel->getNumThreadsX()));
        }
      } else {
        TX = createIntegerLiteral(Ctx, 0);
      }
      if (acc->getSizeY() > 1) {
        SY = createIntegerLiteral(Ctx, static_cast<int>(acc->getSizeY()/2));
      } else {
        SY = createIntegerLiteral(Ctx, 0);
      }
    }

    HipaccMask *Mask = nullptr;
    int mask_idx_x = 0, mask_idx_y = 0;
    switch (E->getNumArgs()) {
      default:
        assert(0 && "0, 1, or 2 arguments for Accessor operator() expected!\n");
        break;
      case 1:
        // 0: -> (this *) Image Class
        if (use_shared) {
          result = accessMemShared(DRE, TX, SY);
        } else {
          result = accessMem(LHS, acc, mem_acc);
        }
        if (compilerOptions.fuseKernels() && KernelClass->getKernelType() == LocalOperator) {
          curIdxXShiftFusion = 0;
          curIdxYShiftFusion = 0;
        }
        break;
      case 2:
        // 0: -> (this *) Image Class
        // 1: -> Mask | Domain
        {
        assert(isa<MemberExpr>(E->getArg(1)->IgnoreImpCasts()) &&
            "Accessor operator() with 1 argument requires a"
            "convolution Mask or Domain as parameter.");
        MemberExpr *ME = dyn_cast<MemberExpr>(E->getArg(1)->IgnoreImpCasts());
        FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
        Mask = Kernel->getMaskFromMapping(FD);
        }
        if (convMask) {
          assert(convMask == Mask &&
              "the Mask parameter for Accessor operator(Mask) has to be"
              "the Mask parameter of the convolve method.");
          mask_idx_x = convIdxX;
          mask_idx_y = convIdxY;
        } else {
          bool found = false;
          for (unsigned int i = 0; i < redDomains.size(); ++i) {
            if (redDomains[i] == Mask) {
              mask_idx_x = redIdxX[i];
              mask_idx_y = redIdxY[i];
              found = true;
              break;
            }
          }
          assert(found &&
              "the Domain parameter for Accessor operator(Domain) has to be"
              "the Domain parameter of the reduce method.");
        }
      case 3:
        // 0: -> (this *) Image Class
        // 1: -> offset x
        // 2: -> offset y
        Expr *offset_x, *offset_y;
        if (E->getNumArgs()==2) {
          offset_x = createIntegerLiteral(Ctx,
              mask_idx_x-static_cast<int>(Mask->getSizeX()/2));
          offset_y = createIntegerLiteral(Ctx,
              mask_idx_y-static_cast<int>(Mask->getSizeY()/2));
          if (compilerOptions.fuseKernels() && KernelClass->getKernelType() == LocalOperator) {
            curIdxXShiftFusion = mask_idx_x-static_cast<int>(Mask->getSizeX()/2);
            curIdxYShiftFusion = mask_idx_y-static_cast<int>(Mask->getSizeY()/2);
          }
        } else {
          offset_x = Clone(E->getArg(1));
          offset_y = Clone(E->getArg(2));
          //TODO re-update the shift for kernel fusion
          if (compilerOptions.fuseKernels() && KernelClass->getKernelType() == LocalOperator) {
            curIdxXShiftFusion = 0;
            curIdxYShiftFusion = 0;
          }
        }

        if (use_shared) {
          result = accessMemShared(DRE, createBinaryOperator(Ctx, offset_x,
                TX, BO_Add, Ctx.IntTy), createBinaryOperator(Ctx, offset_y,
                  SY, BO_Add, Ctx.IntTy));
        } else {
          switch (mem_acc) {
            case READ_ONLY:
              if (bh_variant.borderVal) {
                return addBorderHandling(LHS, offset_x, offset_y, acc);
              }
              // fall through
            case WRITE_ONLY:
            case READ_WRITE:
            case UNDEFINED:
              result = accessMem(LHS, acc, mem_acc, offset_x, offset_y);
              break;
          }
        }
        break;
    }

    if (compilerOptions.fuseKernels() && KernelClass->getKernelType() == LocalOperator) {
     bInsertBeforeSmem = true;
    }
    if (bReplaceInputExpr && KernelClass->getKernelType() == LocalOperator) {
     return exprInputFusion;
    }
  }

  setExprProps(E, result);
  if (bReplaceInputIdxExpr) {
    ArraySubscriptExpr *tempASE = dyn_cast<ArraySubscriptExpr>(result);
    tempASE->setRHS(exprInputIdxFusion);
  }

  return result;
}


Expr *ASTTranslateFusion::VisitMemberExprTranslate(MemberExpr *E) {
  // TODO: create a map with all expressions not to be cloned ..
  if (E==tileVars.local_size_x->IgnoreParenCasts() ||
      E==tileVars.local_size_y->IgnoreParenCasts() ||
      E==tileVars.local_id_x->IgnoreParenCasts() ||
      E==tileVars.local_id_y->IgnoreParenCasts() ||
      E==tileVars.block_id_x->IgnoreParenCasts() ||
      E==tileVars.block_id_y->IgnoreParenCasts()) return E;

  // replace member class variables by kernel parameter references
  // (MemberExpr 0x4bd4af0 'int' ->d 0x4bd2330
  //  (CXXThisExpr 0x4bd4ac8 'class hipacc::VerticalMeanFilter *' this))
  // -->
  // (DeclRefExpr 0x4bda540 'int' ParmVar='d' 0x4bd8010)
  ValueDecl *VD = E->getMemberDecl();
  ValueDecl *paramDecl = nullptr;

  // search for member name in kernel parameter list
  for (auto param : kernelDecl->parameters()) {
    // parameter name matches
    std::string strFuseName = VD->getName();
    strFuseName += kernelParamNameSuffix;
    if (param->getName().equals(strFuseName)) {
      paramDecl = param;

      // get vector declaration
      if (Kernel->vectorize() && !compilerOptions.emitC99()) {
        if (KernelDeclMapVector.count(param)) {
          paramDecl = KernelDeclMapVector[param];
          llvm::errs() << "Vectorize: \n";
          paramDecl->dump();
          llvm::errs() << "\n";
        }
      }

      break;
    }
  }

  if (!paramDecl) {
    unsigned DiagIDParameter = Diags.getCustomDiagID(DiagnosticsEngine::Error,
        "Couldn't find initialization of kernel member variable '%0' in class constructor.");
    Diags.Report(E->getExprLoc(), DiagIDParameter) << VD->getName();
    exit(EXIT_FAILURE);
  }

  // check if the parameter is a Mask and replace it by a global VarDecl
  bool isMask = false;
  for (auto mask : KernelClass->getMaskFields()) {
    std::string strFuseName = mask->getName();
    strFuseName += kernelParamNameSuffix;
    if (paramDecl->getName().equals(strFuseName)) {
      if (auto Mask = Kernel->getMaskFromMapping(mask)) {
        isMask = true;
        if (Mask->isConstant() || compilerOptions.emitC99() ||
            compilerOptions.emitCUDA()) {
          // get Mask/Domain reference
          VarDecl *maskVar = lookup<VarDecl>(Mask->getName() +
              Kernel->getName(), Mask->getType());

          if (!maskVar) {
            maskVar = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
                Mask->getName()+Kernel->getName(), paramDecl->getType());

            DeclContext *DC =
              TranslationUnitDecl::castToDeclContext(Ctx.getTranslationUnitDecl());
            DC->addDecl(maskVar);
          }
          paramDecl = maskVar;
        }
      }
    }
  }

  if (!isMask) {
      // mark parameter as being used within the kernel unless for Masks and
      // Domains
      Kernel->setUsed(VD->getName());
  }

  Expr *result = createDeclRefExpr(Ctx, paramDecl);
  setExprProps(E, result);

  if (bReplaceInputExpr && KernelClass->getKernelType() == PointOperator &&
        Kernel->getImgFromMapping(dyn_cast<FieldDecl>(VD))) {
    return exprInputFusion;
  }
  return result;
}


Expr *ASTTranslateFusion::VisitCXXMemberCallExprTranslate(CXXMemberCallExpr *E) {
  assert(isa<MemberExpr>(E->getCallee()) &&
      "Hipacc: Stumbled upon unsupported expression or statement: CXXMemberCallExpr");
  MemberExpr *ME = cast<MemberExpr>(E->getCallee());

  auto mem_at_fun = [&] (HipaccAccessor *acc, DeclRefExpr *LHS,
                         MemoryAccess mem_acc) -> Expr * {
    assert(E->getNumArgs()==2 &&
           "x and y argument for pixel_at() or output_at() required!");
    Expr *idx_x = addGlobalOffsetX(Clone(E->getArg(0)), acc);
    Expr *idx_y = addGlobalOffsetY(Clone(E->getArg(1)), acc);
    Expr *result = nullptr;

    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
        result = accessMem2DAt(LHS, idx_x, idx_y);
        break;
      case Language::CUDA:
        if (Kernel->useTextureMemory(acc) != Texture::None) {
          result = accessMemTexAt(LHS, acc, mem_acc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(acc), idx_x, idx_y);
        }
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (Kernel->useTextureMemory(acc) != Texture::None) {
          result = accessMemImgAt(LHS, acc, mem_acc, idx_x, idx_y);
        } else {
          result = accessMemArrAt(LHS, getStrideDecl(acc), idx_x, idx_y);
        }
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        if (ME->getMemberNameInfo().getAsString() == "output_at" &&
            compilerOptions.emitFilterscript()) {
            assert(0 && "Filterscript does not support output_at().");
        }
        result = accessMemAllocAt(LHS, mem_acc, idx_x, idx_y);
        break;
    }

    setExprProps(E, result);

    return result;
  };

  if (isa<CXXThisExpr>(ME->getBase()->IgnoreImpCasts())) {
    // check if this is a convolve function call
    if (E->getDirectCallee() && (
          E->getDirectCallee()->getName().equals("convolve") ||
          E->getDirectCallee()->getName().equals("reduce") ||
          E->getDirectCallee()->getName().equals("iterate"))) {
      return convertConvolution(E);
    }

    // Kernel context -> use Iteration Space output Accessor
    auto LHS = outputImage;
    HipaccAccessor *acc = Kernel->getIterationSpace();
    MemoryAccess mem_acc = KernelClass->getMemAccess(KernelClass->getOutField());

    // x() method -> gid_x - is_offset_x
    if (ME->getMemberNameInfo().getAsString() == "x") {
      return createParenExpr(Ctx, removeISOffsetX(tileVars.global_id_x));
    }

    // y() method -> gid_y
    if (ME->getMemberNameInfo().getAsString() == "y") {
      if (compilerOptions.emitC99() ||
          compilerOptions.emitRenderscript() ||
          compilerOptions.emitFilterscript()) {
        return createParenExpr(Ctx, removeISOffsetY(gidYRef));
      }

      return gidYRef;
    }

    // output() method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output") {
      assert(E->getNumArgs()==0 && "no arguments for output() method supported!");
      Expr *result = nullptr;

      switch (compilerOptions.getTargetLang()) {
        case Language::Renderscript:
          if (Kernel->getPixelsPerThread() <= 1) {
            // write to output pixel pointed to by kernel parameter
            LHS = retValRef;
          }
          // fall through
        case Language::C99:
        case Language::CUDA:
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          result = accessMem(LHS, acc, mem_acc);
          break;
        case Language::Filterscript:
          postStmts.push_back(createReturnStmt(Ctx, retValRef));
          postCStmt.push_back(curCStmt);
          result = retValRef;
          break;
      }

      setExprProps(E, result);

      if (bReplaceOutputExpr) { return exprOutputFusion; }

      return result;
    }

    // output_at(x, y) method -> img[y][x]
    if (ME->getMemberNameInfo().getAsString() == "output_at") {
      return mem_at_fun(acc, LHS, mem_acc);
    }
  }

  if (auto base = dyn_cast<MemberExpr>(ME->getBase()->IgnoreImpCasts())) {
    FieldDecl *FD = dyn_cast<FieldDecl>(base->getMemberDecl());

    if (auto acc = Kernel->getImgFromMapping(FD)) {
      MemoryAccess mem_acc = KernelClass->getMemAccess(FD);

      // Acc.x() method -> acc_scale_x * (gid_x - is_offset_x)
      if (ME->getMemberNameInfo().getAsString() == "x") {
        if (acc->getInterpolationMode() == Interpolate::NO) {
          return createParenExpr(Ctx, removeISOffsetX(tileVars.global_id_x));
        }
        // remove is_offset_x and scale index to Accessor size
        return createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationX(acc,
                tileVars.global_id_x)), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      }

      // Acc.y() method -> acc_scale_y * gid_y
      if (ME->getMemberNameInfo().getAsString() == "y") {
        if (acc->getInterpolationMode() == Interpolate::NO) {
          if (compilerOptions.emitRenderscript() ||
              compilerOptions.emitFilterscript()) {
            return createParenExpr(Ctx, removeISOffsetY(gidYRef));
          }
          return gidYRef;
        }
        // scale index to Accessor size
        return createCStyleCastExpr(Ctx, Ctx.IntTy, CK_FloatingToIntegral,
            createParenExpr(Ctx, addNNInterpolationY(acc, gidYRef)), nullptr,
            Ctx.getTrivialTypeSourceInfo(Ctx.IntTy));
      }

      // Acc.pixel_at(x, y) method -> img[y][x]
      if (ME->getMemberNameInfo().getAsString() == "pixel_at") {
        // MemberExpr is converted to DeclRefExpr when cloning
        auto LHS = cast<DeclRefExpr>(Clone(ME->getBase()->IgnoreImpCasts()));
        return mem_at_fun(acc, LHS, mem_acc);
      }
    }

    if (auto mask = Kernel->getMaskFromMapping(FD)) {
      if (mask->isDomain()) {
        bool isDomainValid = false;
        int redDepth = 0;

        // search corresponding domain
        for (size_t i=0, e=redDomains.size(); i!=e; ++i) {
          if (mask == redDomains[i]) {
            isDomainValid = true;
            redDepth = i;
            break;
          }
        }

        assert(isDomainValid && "Getting Domain reduction IDs is only allowed "
                                "within reduction lambda-function.");
        // within convolute lambda-function
        if (ME->getMemberNameInfo().getAsString() == "x") {
          return createIntegerLiteral(Ctx, redIdxX[redDepth] -
              static_cast<int>(redDomains[redDepth]->getSizeX()/2));
        }
        if (ME->getMemberNameInfo().getAsString() == "y") {
          return createIntegerLiteral(Ctx, redIdxY[redDepth] -
              static_cast<int>(redDomains[redDepth]->getSizeY()/2));
        }
      } else {
        assert(mask == convMask && "Getting Mask convolution IDs is only allowed "
                                   "allowed within convolution lambda-function.");
        // within convolute lambda-function
        if (ME->getMemberNameInfo().getAsString() == "x") {
          return createIntegerLiteral(Ctx, convIdxX -
              static_cast<int>(mask->getSizeX()/2));
        }
        if (ME->getMemberNameInfo().getAsString() == "y") {
          return createIntegerLiteral(Ctx, convIdxY -
              static_cast<int>(mask->getSizeY()/2));
        }
      }
    }
  }

  llvm::errs() << "Hipacc: Stumbled upon unsupported expression:\n";
  E->dump();
  std::abort();
}


// stage single image line (warp size) to shared memory
void ASTTranslateFusion::stageLineToSharedMemory(ParmVarDecl *PVD,
    SmallVector<Stmt *, 16> &stageBody, Expr *local_offset_x, Expr
    *local_offset_y, Expr *global_offset_x, Expr *global_offset_y) {

  VarDecl *VD = KernelDeclMapShared[PVD];
  HipaccAccessor *Acc = KernelDeclMapAcc[PVD];
  DeclRefExpr *paramDRE = createDeclRefExpr(Ctx, PVD);

  Expr *LHS = accessMemShared(createDeclRefExpr(Ctx, VD), local_offset_x,
      local_offset_y);

  Expr *RHS;
  if (bh_variant.borderVal) {
    SmallVector<Stmt *, 16> bhStmts;
    SmallVector<CompoundStmt *, 16> bhCStmt;
    RHS = addBorderHandling(paramDRE, global_offset_x, global_offset_y, Acc,
        bhStmts, bhCStmt);

    // add border handling statements to stageBody
    for (auto stmt : bhStmts)
      stageBody.push_back(stmt);
  } else {
    RHS = accessMem(paramDRE, Acc, READ_ONLY, global_offset_x, global_offset_y);
  }

  if (bReplaceInputLocalExprs) {
    // extract and set global id
    ArraySubscriptExpr *tempASE = dyn_cast<ArraySubscriptExpr>(RHS);
    stageBody.push_back(createBinaryOperator(Ctx, exprInputIdxFusion,
      tempASE->getIdx(), BO_Assign, Acc->getImage()->getType()));
    // insert the producer body
    stageBody.push_back(stmtProducerBodyP2L);
    // replace the input
    stageBody.push_back(createBinaryOperator(Ctx, LHS, exprOutputFusion,
      BO_Assign, Acc->getImage()->getType()));
  }
  else {
    stageBody.push_back(createBinaryOperator(Ctx, LHS, RHS, BO_Assign,
          Acc->getImage()->getType()));
  }
}


// stage iteration p to shared memory
void ASTTranslateFusion::stageIterationToSharedMemory(SmallVector<Stmt *, 16>
    &stageBody, int p) {
  for (auto param : kernelDecl->parameters()) {
    if (KernelDeclMapShared[param]) {
      HipaccAccessor *Acc = KernelDeclMapAcc[param];

      unsigned varAccSizeY = Acc->getSizeY();
      if (bReplaceVarAccSizeY) {
        varAccSizeY = FusionLocalVarAccSizeY;
      }

      // check if the bottom apron has to be fetched
      if (p>=static_cast<int>(Kernel->getPixelsPerThread())) {
        int p_add = static_cast<int>(ceilf((varAccSizeY-1) /
              static_cast<float>(Kernel->getNumThreadsY())));
        if (p>=static_cast<int>(Kernel->getPixelsPerThread())+p_add) continue;
      }

      Expr *global_offset_x = nullptr, *global_offset_y = nullptr;
      Expr *SX2;

      if (Acc->getSizeX() > 1) {
        if (compilerOptions.exploreConfig()) {
          SX2 = tileVars.local_size_x;
        } else {
          SX2 = createIntegerLiteral(Ctx,
              static_cast<int32_t>(Kernel->getNumThreadsX()));
        }
      } else {
        SX2 = createIntegerLiteral(Ctx, 0);
      }
      if (varAccSizeY > 1) {
        global_offset_y = createParenExpr(Ctx, createUnaryOperator(Ctx,
              createIntegerLiteral(Ctx,
                static_cast<int32_t>(varAccSizeY/2)), UO_Minus, Ctx.IntTy));
      } else {
        global_offset_y = nullptr;
      }

      if (compilerOptions.allowMisAlignedAccess()) {
        Expr *local_offset_x = nullptr;
        // load line first half
        if (Acc->getSizeX() > 1) {
          local_offset_x = createIntegerLiteral(Ctx, static_cast<int32_t>(0));
          global_offset_x = createParenExpr(Ctx, createUnaryOperator(Ctx,
                createIntegerLiteral(Ctx,
                  static_cast<int32_t>(varAccSizeY/2)), UO_Minus, Ctx.IntTy));
        }
        stageLineToSharedMemory(param, stageBody, local_offset_x, nullptr,
            global_offset_x, global_offset_y);

        // load line second half (partially overlap)
        if (Acc->getSizeX() > 1) {
          local_offset_x = createIntegerLiteral(Ctx, static_cast<int32_t>(varAccSizeY/2)*2);
          global_offset_x = createParenExpr(Ctx, createUnaryOperator(Ctx,
                createIntegerLiteral(Ctx,
                  static_cast<int32_t>(varAccSizeY/2)), UO_Plus, Ctx.IntTy));
        }
        stageLineToSharedMemory(param, stageBody, local_offset_x, nullptr,
            global_offset_x, global_offset_y);
      } else {
        // check if we need to stage right apron
        size_t num_stages_x = 0;
        if (Acc->getSizeX() > 1) {
            num_stages_x = 2;
        }

        // load row (line)
        for (size_t i=0; i<=num_stages_x; ++i) {
          // _smem[lidYRef][(int)threadIdx.x + i*(int)blockDim.x] =
          //        Image[-SX/2 + i*(int)blockDim.x, -SY/2];
          Expr *local_offset_x = nullptr;
          if (Acc->getSizeX() > 1) {
            local_offset_x = createBinaryOperator(Ctx, createIntegerLiteral(Ctx,
                  static_cast<int32_t>(i)), tileVars.local_size_x, BO_Mul,
                Ctx.IntTy);
            global_offset_x = createBinaryOperator(Ctx, local_offset_x, SX2,
                BO_Sub, Ctx.IntTy);
          }

          stageLineToSharedMemory(param, stageBody, local_offset_x, nullptr,
              global_offset_x, global_offset_y);
        }
      }
    }
  }
}




void ASTTranslateFusion::initCUDA(SmallVector<Stmt *, 16> &kernelBody) {
  VarDecl *gid_x = nullptr, *gid_y = nullptr;
  SmallVector<QualType, 16> uintDeclTypes;
  SmallVector<StringRef, 16> uintDeclNames;
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclTypes.push_back(Ctx.UnsignedIntTy);
  uintDeclNames.push_back("x");
  uintDeclNames.push_back("y");
  uintDeclNames.push_back("z");

  // CUDA
  /*DEVICE_BUILTIN*/
  //struct uint3
  //{
  //  unsigned x, y, z;
  //};
  RecordDecl *uint3RD = createRecordDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "uint3", TTK_Struct, uintDeclTypes, uintDeclNames);

  /*DEVICE_BUILTIN*/
  //typedef struct uint3 uint3;

  /*DEVICE_BUILTIN*/
  //struct dim3
  //{
  //    unsigned x, y, z;
  //};

  /*DEVICE_BUILTIN*/
  //typedef struct dim3 dim3;

  //uint3 threadIdx;
  VarDecl *threadIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "threadIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  //uint3 blockIdx;
  VarDecl *blockIdx = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockIdx", Ctx.getTypeDeclType(uint3RD), nullptr);
  //dim3 blockDim;
  VarDecl *blockDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
      "blockDim", Ctx.getTypeDeclType(uint3RD), nullptr);
  //dim3 gridDim;
  //VarDecl *gridDim = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "gridDim", Ctx.getTypeDeclType(uint3RD), nullptr);
  //int warpSize;
  //VarDecl *warpSize = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
  //    "warpSize", Ctx.IntTy, nullptr);

  DeclRefExpr *TIRef = createDeclRefExpr(Ctx, threadIdx);
  DeclRefExpr *BIRef = createDeclRefExpr(Ctx, blockIdx);
  DeclRefExpr *BDRef = createDeclRefExpr(Ctx, blockDim);
  VarDecl *xVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "x",
      Ctx.IntTy, nullptr);
  VarDecl *yVD = createVarDecl(Ctx, Ctx.getTranslationUnitDecl(), "y",
      Ctx.IntTy, nullptr);

  tileVars.local_id_x = createMemberExpr(Ctx, TIRef, false, xVD,
      xVD->getType());
  tileVars.local_id_y = createMemberExpr(Ctx, TIRef, false, yVD,
      yVD->getType());
  tileVars.block_id_x = createMemberExpr(Ctx, BIRef, false, xVD,
      xVD->getType());
  tileVars.block_id_y = createMemberExpr(Ctx, BIRef, false, yVD,
      yVD->getType());
  tileVars.local_size_x = createMemberExpr(Ctx, BDRef, false, xVD,
      xVD->getType());
  tileVars.local_size_y = createMemberExpr(Ctx, BDRef, false, yVD,
      yVD->getType());
  //DeclRefExpr *GDRef = createDeclRefExpr(Ctx, gridDim);
  //tileVars.grid_size_x = createMemberExpr(Ctx, GDRef, false, xVD,
  //    xVD->getType());
  //tileVars.grid_size_y = createMemberExpr(Ctx, GDRef, false, yVD,
  //    yVD->getType());

  // CUDA: const int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
  gid_x = createVarDecl(Ctx, kernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, createBinaryOperator(Ctx, tileVars.local_size_x,
          tileVars.block_id_x, BO_Mul, Ctx.IntTy), tileVars.local_id_x, BO_Add,
        Ctx.IntTy));

  // CUDA: const int gid_y = blockDim.y*PPT*blockIdx.y + threadIdx.y;
  Expr *YE = createBinaryOperator(Ctx, tileVars.local_size_y,
      tileVars.block_id_y, BO_Mul, Ctx.IntTy);
  if (Kernel->getPixelsPerThread() > 1) {
    YE = createBinaryOperator(Ctx, YE, createIntegerLiteral(Ctx,
          static_cast<int>(Kernel->getPixelsPerThread())), BO_Mul, Ctx.IntTy);
  }
  gid_y = createVarDecl(Ctx, kernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, YE, tileVars.local_id_y, BO_Add, Ctx.IntTy));

  // add gid_x and gid_y statements
  DeclContext *DC = FunctionDecl::castToDeclContext(kernelDecl);
  DC->addDecl(gid_x);
  DC->addDecl(gid_y);

  if (!bSkipGidDecl) {
    kernelBody.push_back(createDeclStmt(Ctx, gid_x));
    kernelBody.push_back(createDeclStmt(Ctx, gid_y));
  }

  tileVars.global_id_x = createDeclRefExpr(Ctx, gid_x);
  tileVars.global_id_y = createDeclRefExpr(Ctx, gid_y);
}


// vim: set ts=2 sw=2 sts=2 et ai:


