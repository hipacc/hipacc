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



void ASTFuse::insertPrologFusedKernel() {
  for (auto VD : fusionRegVarDecls)
    curFusedKernelBody.push_back(createDeclStmt(Ctx, VD));
  for (auto VD : fusionIdxVarDecls)
    curFusedKernelBody.push_back(createDeclStmt(Ctx, VD));
  for (auto S : fusionRegSharedStmts)
    curFusedKernelBody.push_back(S);
}

void ASTFuse::insertEpilogFusedKernel() {
  // TODO
}


void ASTFuse::markKernelPositionSublist(std::list<HipaccKernel *> &l) {
  // find number of local operators in the list
  SmallVector<unsigned, 16> vecLocalKernelIndices;
  unsigned kLocalCnt = 0;
  for (auto K : l) {
    HipaccKernelClass *KC = K->getKernelClass();
    if (KC->getKernelType() == LocalOperator) {
      vecLocalKernelIndices.push_back(kLocalCnt);
    }
    FusionTypeTags *tags = new FusionTypeTags;
    FusibleKernelSubListPosMap[K] = tags;
    kLocalCnt++;
  }

  // container for sublists
  std::list<HipaccKernel *> pointKernelList;
  SmallVector<std::list<HipaccKernel *>, 16> vecLocalKernelLists;

  // analyze the kernel position in the fusion list
  if (vecLocalKernelIndices.empty()) {
    // scenario 1: no local kernels. e.g. p -> p -> ... -> p
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
    // scenario 2: first kernel is local e.g. l -> p -> ... -> l -> p -> ...
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

void ASTFuse::recomputeMemorySizeLocalFusion(std::list<HipaccKernel *> &l) {
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
    fusedLocalKernelMemorySizeMap[K] = std::make_tuple(XSizeAcc, YSizeAcc);
  }
}

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
  for (auto K : l) { fusedKernelNameMap[K] = fusedKernelName; }
  fusedFileName = compilerOptions.getTargetPrefix() + name + "Fused";
  fusedFileNamesAll.push_back(fusedFileName);

  return createFunctionDecl(Ctx, Ctx.getTranslationUnitDecl(),
      fusedKernelName, Ctx.VoidTy, argTypesKFuse, deviceArgNamesKFuse);
}

void ASTFuse::initKernelFusion() {
  fusedKernelName.clear();
  fusedFileName.clear();
  FuncDeclParamKernelMap.clear();
  FuncDeclParamDeclMap.clear();
  localKernelMemorySizeMap.clear();
  FusibleKernelSubListPosMap.clear();
  fusionRegVarCount=0;
  fusionIdxVarCount=0;
  fusionRegVarDecls.clear();
  fusionRegSharedStmts.clear();
  fusionIdxVarDecls.clear();
  curFusedKernelBody.clear();
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
  VarDecl *regVDSImg;
  SmallVector<Stmt *, 16> vecFusionBody;
  SmallVector<Stmt *, 16> vecProducerP2LBody;
  std::queue<Stmt *> stmtsL2LProducerKernel;
  std::queue<Stmt *> stmtsL2LConsumerKernel;
  HipaccKernel *KLocalSrc = nullptr;
  recomputeMemorySizeLocalFusion(l);

  for (auto it = (l.begin()); it != l.end(); ++it) {
    curFusionBody = nullptr;
    HipaccKernel *K = *it;
    HipaccKernelClass *KC = K->getKernelClass();
    FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
    FunctionDecl *kernelDecl = createFunctionDecl(Ctx,
        Ctx.getTranslationUnitDecl(), K->getKernelName(),
        Ctx.VoidTy, K->getArgTypes(), K->getDeviceArgNames());
    ASTTranslate *Hipacc = new ASTTranslate(Ctx, kernelDecl, K, KC,
        builtins, compilerOptions);

    // TODO, enable the composition of multiple fusions
    // domain-specific translation and fusion
    switch(KTag->Point2PointLoc) {
      default: break;
      case Source:
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->setFusionP2PSrcOperator(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Destination:
        Hipacc->setFusionP2PDestOperator(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->setFusionP2PIntermOperator(VDIn, VDOut);
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
    }
    switch(KTag->Local2PointLoc) {
      default: break;
      case Source:
        if (dataDeps->hasSharedIS(K)) {
          createReg4FusionVarDecl(KC->getOutField()->getType());
          regVDSImg = fusionRegVarDecls.back();
        }
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->setFusionP2PSrcOperator(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());

        if (dataDeps->hasSharedIS(K)) {
          fusionRegSharedStmts.push_back(Hipacc->getFusionSharedInputStmt(regVDSImg));
        }
        break;
      case Destination:
        if (dataDeps->hasSharedIS(K)) {
          Hipacc->setFusionL2PDestOperator(fusionRegVarDecls.back(), regVDSImg, dataDeps->getSharedISName(K));
        } else {
          Hipacc->setFusionP2PDestOperator(fusionRegVarDecls.back());
        }
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        if (dataDeps->hasSharedIS(K)) {
          Hipacc->setFusionL2PIntermOperator(VDIn, VDOut, regVDSImg, dataDeps->getSharedISName(K));
        } else {
          Hipacc->setFusionP2PIntermOperator(VDIn, VDOut);
        }
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
    }
    switch(KTag->Point2LocalLoc) {
      default: break;
      case Source:
        createIdx4FusionVarDecl();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->setFusionP2LSrcOperator(fusionRegVarDecls.back(), fusionIdxVarDecls.back());
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
      case Destination:
        Hipacc->setFusionP2LDestOperator(fusionRegVarDecls.back(), fusionIdxVarDecls.back(),
                                createCompoundStmt(Ctx, vecProducerP2LBody));
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->setFusionP2PIntermOperator(VDIn, VDOut);
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
    }
    switch(KTag->Local2LocalLoc) {
      default: break;
      case Source:
        createReg4FusionVarDecl(KC->getOutField()->getType());
        createIdx4FusionVarDecl();
        idxXFused = fusionIdxVarDecls.back();
        createIdx4FusionVarDecl();
        idxYFused = fusionIdxVarDecls.back();
        Hipacc->setFusionL2LSrcOperator(fusionRegVarDecls.back(), idxXFused, idxYFused,
            std::get<1>(localKernelMemorySizeMap[K]));
        Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        stmtsL2LProducerKernel = Hipacc->getFusionLocalKernelBody();
        KLocalSrc = K;
        break;
      case Destination:
        Hipacc->setFusionL2LDestOperator(stmtsL2LProducerKernel,
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
    localKernelMaxAccSizeUpdated = localKernelMemorySizeMap[K];
    ASTTranslate *Hipacc = new ASTTranslate(Ctx, kernelDecl, K, KC,
        builtins, compilerOptions);
    Hipacc->setFusionL2LEndSrcOperator(stmtsL2LConsumerKernel, idxXFused, idxYFused,
        std::get<1>(localKernelMemorySizeMap[K]));
    vecFusionBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
  }

  insertPrologFusedKernel();
  curFusedKernelBody.push_back(createCompoundStmt(Ctx, vecFusionBody));
  insertEpilogFusedKernel();
  curFusedKernelDecl->setBody(createCompoundStmt(Ctx, curFusedKernelBody));
}

void ASTFuse::setFusedKernelConfiguration(std::list<HipaccKernel *>& l) {
//  #ifdef USE_JIT_ESTIMATE
//  HipaccFusion(l);
//  printFusedKernelFunction(l); // write fused kernel to file
//
//  // JIT compile kernel in order to get resource usage
//  std::string command = (l.back())->getCompileCommand(fusedKernelName,
//      fusedFileName, compilerOptions.emitCUDA());
//
//  int reg=0, lmem=0, smem=0, cmem=0;
//  char line[FILENAME_MAX];
//  SmallVector<std::string, 16> lines;
//  FILE *fpipe;
//
//  if (!(fpipe = (FILE *)popen(command.c_str(), "r"))) {
//    perror("Problems with pipe");
//    exit(EXIT_FAILURE);
//  }
//
//  while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
//    lines.push_back(std::string(line));
//
//    if (targetDevice.isNVIDIAGPU()) {
//      char *ptr = line;
//      char mem_type = 'x';
//      int val1 = 0, val2 = 0;
//
//      if (sscanf(ptr, "%d bytes %1c tack frame", &val1, &mem_type) == 2) {
//        if (mem_type == 's') {
//          lmem = val1;
//          continue;
//        }
//      }
//
//      if (sscanf(line, "ptxas info : Used %d registers", &reg) == 0)
//        continue;
//
//      while ((ptr = strchr(ptr, ','))) {
//        ptr++;
//
//        if (sscanf(ptr, "%d+%d bytes %1c mem", &val1, &val2, &mem_type) == 3) {
//          switch (mem_type) {
//            default: llvm::errs() << "wrong memory specifier '" << mem_type
//                                  << "': " << ptr; break;
//            case 'c': cmem += val1 + val2; break;
//            case 'l': lmem += val1 + val2; break;
//            case 's': smem += val1 + val2; break;
//          }
//          continue;
//        }
//
//        if (sscanf(ptr, "%d bytes %1c mem", &val1, &mem_type) == 2) {
//          switch (mem_type) {
//            default: llvm::errs() << "wrong memory specifier '" << mem_type
//                                  << "': " << ptr; break;
//            case 'c': cmem += val1; break;
//            case 'l': lmem += val1; break;
//            case 's': smem += val1; break;
//          }
//          continue;
//        }
//
//        if (sscanf(ptr, "%d texture %1c", &val1, &mem_type) == 2)
//          continue;
//        if (sscanf(ptr, "%d sampler %1c", &val1, &mem_type) == 2)
//          continue;
//        if (sscanf(ptr, "%d surface %1c", &val1, &mem_type) == 2)
//          continue;
//
//        // no match found
//        llvm::errs() << "Unexpected memory usage specification: '" << ptr;
//      }
//    } else if (targetDevice.isAMDGPU()) {
//      sscanf(line, "isa info : Used %d gprs, %d bytes lds", &reg, &smem);
//    }
//  }
//  pclose(fpipe);
//
//  if (reg == 0) {
//    unsigned DiagIDCompile = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
//        "Compiling kernel in file '%0.%1' failed, using default kernel configuration:\n%2");
//    Diags.Report(DiagIDCompile)
//      << fusedFileName << (const char*)(compilerOptions.emitCUDA()?"cu":"cl")
//      << command.c_str();
//    for (auto line : lines)
//      llvm::errs() << line;
//  } else {
//    if (targetDevice.isNVIDIAGPU()) {
//      llvm::errs() << "Resource usage for kernel '" << fusedKernelName << "'"
//                   << ": " << reg << " registers, "
//                   << lmem << " bytes lmem, "
//                   << smem << " bytes smem, "
//                   << cmem << " bytes cmem\n";
//    } else if (targetDevice.isAMDGPU()) {
//      llvm::errs() << "Resource usage for kernel '" << fusedKernelName << "'"
//                   << ": " << reg << " gprs, "
//                   << smem << " bytes lds\n";
//    }
//  }
//
//  for (auto K : l) {
//    K->updateFusionSizeX(std::get<1>(localKernelMaxAccSizeUpdated));
//    K->updateFusionSizeY(std::get<1>(localKernelMaxAccSizeUpdated));
//    K->setResourceUsage(reg, lmem, smem, cmem);
//  }
//  #else
  for (auto K : l) {
    K->setDefaultConfig();
  }
//  #endif
}


bool ASTFuse::parseFusibleKernel(HipaccKernel *K) {
  if (!dataDeps->isFusible(K)) { return false; }

  // prepare fusible kernel set 
  unsigned blockCnt, listCnt;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  assert(FusibleKernelBlockLocation.count(kernelName) && "Kernel name has no record");
  std::tie(blockCnt, listCnt) = FusibleKernelBlockLocation[kernelName];
  auto PBl = fusibleKernelSet[blockCnt];
  auto it = PBl.begin(); std::advance(it, listCnt); PBl.insert(it, K);
  fusibleKernelSet[blockCnt] = PBl;

  // fusion starts whenever a fusible block is ready
  auto PBNam = *std::next(fusibleSetNames.begin(), blockCnt);
  if (PBl.size() == PBNam.size()) {
    PBl.sort([&](HipaccKernel *ka, HipaccKernel *kb){
        std::string kaNam = ka->getKernelClass()->getName() + ka->getName();
        std::string kbNam = kb->getKernelClass()->getName() + kb->getName();
        return std::none_of(PBNam.begin(), PBNam.end(), [&](std::list<std::string> ls){return ls.front() == kbNam && 
               std::find(ls.begin(), ls.end(), kaNam) != ls.end();});
        });
    fusibleKernelSet[blockCnt] = PBl;
    setFusedKernelConfiguration(PBl);
    HipaccFusion(PBl);
    printFusedKernelFunction(PBl); // write fused kernel to file
  }
  return true;
}

// getters
bool ASTFuse::isSrcKernel(HipaccKernel *K) {
  unsigned blockCnt, listCnt;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  assert(FusibleKernelBlockLocation.count(kernelName) && "Kernel name has no record");
  std::tie(blockCnt, listCnt) = FusibleKernelBlockLocation[kernelName];
  return fusibleKernelSet[blockCnt].front() == K;
}

bool ASTFuse::isDestKernel(HipaccKernel *K) { 
  unsigned blockCnt, listCnt;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  assert(FusibleKernelBlockLocation.count(kernelName) && "Kernel name has no record");
  std::tie(blockCnt, listCnt) = FusibleKernelBlockLocation[kernelName];
  return fusibleKernelSet[blockCnt].back() == K;
}

HipaccKernel *ASTFuse::getProducerKernel(HipaccKernel *K) {
  unsigned blockCnt, listCnt;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  assert(FusibleKernelBlockLocation.count(kernelName) && "Kernel name has no record");
  std::tie(blockCnt, listCnt) = FusibleKernelBlockLocation[kernelName];
  auto PBl = fusibleKernelSet[blockCnt];
  auto it = std::find(PBl.begin(), PBl.end(), K);
  return (it == PBl.begin()) ? nullptr : *std::prev(it); 
}

SmallVector<std::string, 16> ASTFuse::getFusedFileNamesAll() const {
  return fusedFileNamesAll;
}

std::string ASTFuse::getFusedKernelName(HipaccKernel *K) { return fusedKernelNameMap[K]; }
unsigned ASTFuse::getNewYSizeLocalKernel(HipaccKernel *K) { return std::get<1>(fusedLocalKernelMemorySizeMap[K]); }

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



// vim: set ts=2 sw=2 sts=2 et ai:


