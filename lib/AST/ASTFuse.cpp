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

#include "ASTFuse.h"

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


void ASTFuse::markKernelPositionSublist(std::list<HipaccKernel *> *l) {
  // initialize kernel location tags
  for (auto K : *l) {
    FusionTypeTags *tags = new FusionTypeTags;
    FusibleKernelSubListPosMap[K] = tags;
  }

  // sub-list indexing
  auto itSrc = l->begin();
  auto itLastLocal = l->begin();
  for (auto it = l->begin(); it != l->end(); ++it) {
    HipaccKernelClass *KC = (*it)->getKernelClass();
    // local to local fusion, e.g., l -> l
    if ((KC->getKernelType() == LocalOperator) && (it == itSrc) && (l->size() == 2) &&
        (l->back()->getKernelClass()->getKernelType() == LocalOperator)) {
      FusionTypeTags *PKTag = FusibleKernelSubListPosMap[l->front()];
      PKTag->Local2LocalLoc = Source;
      FusionTypeTags *CKTag = FusibleKernelSubListPosMap[l->back()];
      CKTag->Local2LocalLoc = Destination;
      break;
    }

    if ((KC->getKernelType() == LocalOperator) && (it != itSrc)) {
      // first search all the local kernels in the list
      // point to local fusion, e.g., p -> p -> ... -> l
      for (auto itSub = itSrc; itSub != it; ++itSub) {
        HipaccKernel *K = *itSub;
        FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
        if (itSub == itSrc) {
          KTag->Point2LocalLoc = Source;
        } else {
          KTag->Point2LocalLoc = Intermediate;
        }
      }
      HipaccKernel *KL = *it;
      FusionTypeTags *KTagL = FusibleKernelSubListPosMap[KL];
      KTagL->Point2LocalLoc = Destination;
      itSrc = std::next(it);
      itLastLocal = it;
    } else if (std::next(it) == l->end()) {
      // after found all the local kernels in the list
      // trace back to perform point-based kernels
      HipaccKernelClass *KC = (*itLastLocal)->getKernelClass();
      if (KC->getKernelType() == LocalOperator) {
        // local to point fusion, e.g., l -> p -> ... -> p
        for (auto itSub = itLastLocal; itSub != l->end(); ++itSub) {
          HipaccKernel *K = *itSub;
          FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
          if (itSub == itLastLocal) {
            KTag->Local2PointLoc = Source;
          } else if (std::next(itSub) == l->end()) {
            KTag->Local2PointLoc = Destination;
          } else {
            KTag->Local2PointLoc = Intermediate;
          }
        }
      } else {
        // point to point fusion, e.g., p -> p -> ... -> p
        for (auto itSub = itLastLocal; itSub != l->end(); ++itSub) {
          HipaccKernel *K = *itSub;
          FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
          if (itSub == itLastLocal) {
            KTag->Point2PointLoc = Source;
          } else if (std::next(itSub) == l->end()) {
            KTag->Point2PointLoc = Destination;
          } else {
            KTag->Point2PointLoc = Intermediate;
          }
        }
      }
    }
  }

  if (DEBUG) {
    std::cout << "[Kernel Fusion INFO] fusible sublist position:\n";
    for (auto K : *l) {
      FusionTypeTags *tags = FusibleKernelSubListPosMap[K];
      std::cout << " " << K->getKernelClass()->getName() + K->getName() << ":";
      std::cout << " Point2PointLoc(" << tags->Point2PointLoc << "),";
      std::cout << " Local2PointLoc(" << tags->Local2PointLoc << "),";
      std::cout << " Point2LocalLoc(" << tags->Point2LocalLoc << "),";
      std::cout << " Local2LocalLoc(" << tags->Local2LocalLoc << ")\n";
    }
  }
}

void ASTFuse::recomputeMemorySizeLocalFusion(std::list<HipaccKernel *> *l) {
  // shared memory size update
  unsigned YSizeAcc = 1;
  unsigned XSizeAcc = 1;
  SmallVector<HipaccKernel *, 16> revList;
  for (auto K : *l) {
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

FunctionDecl *ASTFuse::createFusedKernelDecl(std::list<HipaccKernel *> *l) {
  SmallVector<QualType, 16> argTypesKFuse;
  SmallVector<std::string, 16> deviceArgNamesKFuse;
  std::string name;

  // concatenate kernel arguments from the list
  // TODO: remove non-used paramaters
  for (auto K : *l) {
    size_t kfNumArg = 0;
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
  for (auto K : *l) { fusedKernelNameMap[K] = fusedKernelName; }
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

void ASTFuse::HipaccFusion(std::list<HipaccKernel *> *l) {
  hipacc_require((l->size() >=2), "at least two kernels shoud be recorded for fusion");
  initKernelFusion();
  curFusedKernelDecl = createFusedKernelDecl(l);
  createGidVarDecl();

  markKernelPositionSublist(l);

  // generating body for the fused kernel
  bool Local2LocalEndInsertion = false;
  VarDecl *idxXFused, *idxYFused;
  VarDecl *regVDSImg;
  SmallVector<Stmt *, 16> vecFusionBody;
  SmallVector<Stmt *, 16> vecProducerP2LBody;
  std::queue<Stmt *> stmtsL2LProducerKernel;
  std::queue<Stmt *> stmtsL2LConsumerKernel;
  HipaccKernel *KLocalSrc = nullptr;
  recomputeMemorySizeLocalFusion(l);

  if (DEBUG) {
    std::cout << "[Kernel Fusion INFO] domain-specific fusion:\n";
  }
  // fused kernel body generation
  for (auto it = (l->begin()); it != l->end(); ++it) {
    Stmt *curFusionBody = nullptr;
    HipaccKernel *K = *it;
    HipaccKernelClass *KC = K->getKernelClass();
    KernelType KernelType = KC->getKernelType();
    if (DEBUG) { std::cout << " Kernel " << KC->getName() + K->getName() << " executes "; }
    FusionTypeTags *KTag = FusibleKernelSubListPosMap[K];
    FunctionDecl *kernelDecl = createFunctionDecl(Ctx,
        Ctx.getTranslationUnitDecl(), K->getKernelName(),
        Ctx.VoidTy, K->getArgTypes(), K->getDeviceArgNames());
    ASTTranslate *Hipacc = new ASTTranslate(Ctx, kernelDecl, K, KC,
        builtins, compilerOptions);

    // Point-to-Point Transformation
    switch(KTag->Point2PointLoc) {
      default:
				break;
      case Source:
        if (DEBUG) { std::cout << "P2P source generate"; }
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->setFusionP2PSrcOperator(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Destination:
        if (DEBUG) { std::cout << "P2P Destination generate"; }
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        Hipacc->setFusionP2PDestOperator(fusionRegVarDecls.back());
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        if (DEBUG) { std::cout << "P2P Intermediate generate"; }
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->setFusionP2PIntermOperator(VDIn, VDOut);
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
    }

    // Local-to-Point Transformation
    switch(KTag->Local2PointLoc) {
      default:
				break;
      case Source:
        hipacc_require(KernelType == LocalOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "L2P source generate"; }
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
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "L2P Destination generate"; }
        if (dataDeps->hasSharedIS(K)) {
          Hipacc->setFusionL2PDestOperator(fusionRegVarDecls.back(), regVDSImg,
              dataDeps->getSharedISName(K));
        } else {
          Hipacc->setFusionP2PDestOperator(fusionRegVarDecls.back());
        }
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "L2P Intermediate generate"; }
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

    // Point-to-Local Transformation
    switch(KTag->Point2LocalLoc) {
      default:
				break;
      case Source:
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "P2L source generate"; }
        createIdx4FusionVarDecl();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        Hipacc->setFusionP2LSrcOperator(fusionRegVarDecls.back(), fusionIdxVarDecls.back());
        vecProducerP2LBody.clear();
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
      case Destination:
        hipacc_require(KernelType == LocalOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "P2L Destination generate"; }
        Hipacc->setFusionP2LDestOperator(fusionRegVarDecls.back(), fusionIdxVarDecls.back(),
                                createCompoundStmt(Ctx, vecProducerP2LBody));
        curFusionBody = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        break;
      case Intermediate:
        hipacc_require(KernelType == PointOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "P2L Intermediate generate"; }
        VarDecl *VDIn = fusionRegVarDecls.back();
        createReg4FusionVarDecl(KC->getOutField()->getType());
        VarDecl *VDOut = fusionRegVarDecls.back();
        Hipacc->setFusionP2PIntermOperator(VDIn, VDOut);
        vecProducerP2LBody.push_back(Hipacc->Hipacc(KC->getKernelFunction()->getBody()));
        break;
    }

    // Local-to-Local Transformation
    switch(KTag->Local2LocalLoc) {
      default:
				break;
      case Source:
        hipacc_require(KernelType == LocalOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "L2L source generate"; }
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
        hipacc_require(KernelType == LocalOperator, "Mismatch kernel type for fusion");
        if (DEBUG) { std::cout << "L2L Destination generate"; }
        Hipacc->setFusionL2LDestOperator(stmtsL2LProducerKernel,
          fusionRegVarDecls.back(), idxXFused, idxYFused,
          std::get<1>(localKernelMemorySizeMap[K]));
        Hipacc->Hipacc(KC->getKernelFunction()->getBody());
        stmtsL2LConsumerKernel = Hipacc->getFusionLocalKernelBody();
        Local2LocalEndInsertion = true;
        break;
      case Intermediate:
        hipacc_require(0, "Only two local kernels can be fused");
        break;
    }

    if (curFusionBody) {
      vecFusionBody.push_back(curFusionBody);
    }
    if (DEBUG) { std::cout << "\n"; }
  } // fused kernel body generation end

  // Additional handling needed for Local-to-Local Transformation
  if (Local2LocalEndInsertion) {
    if (DEBUG) { std::cout << " additional L2L transformation...\n"; }
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

  // combine kernel body with decl
  if (DEBUG) { std::cout << " combining transformed kernel body...\n"; }
  insertPrologFusedKernel();
  curFusedKernelBody.push_back(createCompoundStmt(Ctx, vecFusionBody));
  insertEpilogFusedKernel();
  curFusedKernelDecl->setBody(createCompoundStmt(Ctx, curFusedKernelBody));
  if (DEBUG) { std::cout << " fused kernel body done...\n"; }
}

void ASTFuse::setFusedKernelConfiguration(std::list<HipaccKernel *> *l) {
  for (auto K : *l) {
    K->setDefaultConfig();
  }
}

// only working on single-list
// TODO: implement parallel fusion
bool ASTFuse::parseFusibleKernel(HipaccKernel *K) {
  if (!dataDeps->isFusible(K)) { return false; }

  // prepare fusible kernel list
  unsigned PBlockID, KernelVecID;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(FusibleKernelBlockLocation.count(kernelName), "Kernel name has no record");
  std::tie(PBlockID, KernelVecID) = FusibleKernelBlockLocation[kernelName];
  auto PBl = fusibleKernelSet[PBlockID];
  PBl->push_back(K);

  // fusion starts whenever a fusible block is ready
  auto PBNam = *std::next(fusibleSetNames.begin(), PBlockID);
  if (PBl->size() == PBNam.size()) {
    // sort fusible list based on data dependence
    PBl->sort([&] (HipaccKernel *ka, HipaccKernel *kb) -> bool {
        std::string kaNam = ka->getKernelClass()->getName() + ka->getName();
        auto itKa = std::find_if(PBNam.begin(), PBNam.end(),
                    [&](std::list<std::string> ls) { return ls.front() == kaNam; });
        hipacc_require(itKa != PBNam.end(), "Kernel cannot be sorted");
        std::string kbNam = kb->getKernelClass()->getName() + kb->getName();
        return (std::find(itKa->begin(), itKa->end(), kbNam) != itKa->end());
    });

    if (DEBUG) {
      std::cout << "[Kernel Fusion INFO] fusible list:\n {";
      for (auto k : *PBl) {
        std::cout << " -> " << k->getKernelClass()->getName() + k->getName();
      }
      std::cout << " }" << std::endl;
    }
    setFusedKernelConfiguration(PBl);
    HipaccFusion(PBl);
    printFusedKernelFunction(PBl); // write fused kernel to file
  }
  return true;
}

// getters
bool ASTFuse::isSrcKernel(HipaccKernel *K) {
  unsigned PBlockID, KernelVecID;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(FusibleKernelBlockLocation.count(kernelName), "Kernel name has no record");
  std::tie(PBlockID, KernelVecID) = FusibleKernelBlockLocation[kernelName];
  return fusibleKernelSet[PBlockID]->front() == K;
}

bool ASTFuse::isDestKernel(HipaccKernel *K) {
  unsigned PBlockID, KernelVecID;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(FusibleKernelBlockLocation.count(kernelName), "Kernel name has no record");
  std::tie(PBlockID, KernelVecID) = FusibleKernelBlockLocation[kernelName];
  return fusibleKernelSet[PBlockID]->back() == K;
}

HipaccKernel *ASTFuse::getProducerKernel(HipaccKernel *K) {
  unsigned PBlockID, KernelVecID;
  std::string kernelName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(FusibleKernelBlockLocation.count(kernelName), "Kernel name has no record");
  std::tie(PBlockID, KernelVecID) = FusibleKernelBlockLocation[kernelName];
  auto PBl = fusibleKernelSet[PBlockID];
  auto it = std::find(PBl->begin(), PBl->end(), K);
  return (it == PBl->begin()) ? nullptr : *std::prev(it);
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
  VarDecl *gid_x = createVarDecl(Ctx, curFusedKernelDecl, "gid_x", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, createBinaryOperator(Ctx, local_size_x,
          block_id_x, BO_Mul, Ctx.IntTy), local_id_x, BO_Add,
        Ctx.IntTy));
  Expr *YE = createBinaryOperator(Ctx, local_size_y, block_id_y, BO_Mul, Ctx.IntTy);
  VarDecl *gid_y = createVarDecl(Ctx, curFusedKernelDecl, "gid_y", Ctx.getConstType(Ctx.IntTy),
      createBinaryOperator(Ctx, YE, local_id_y, BO_Add, Ctx.IntTy));
  fusionRegVarDecls.push_back(gid_x);
  fusionRegVarDecls.push_back(gid_y);
}

void ASTFuse::printFusedKernelFunction(std::list<HipaccKernel *> *l) {
  int fd;
  std::string filename(fusedFileName);
  std::string ifdef("_" + filename + "_");
  switch (compilerOptions.getTargetLang()) {
    case Language::CUDA:         filename += ".cu"; ifdef += "CU_"; break;
    case Language::C99:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU: break;
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
      // TODO: remove when fixed upstream (not yet fixed despite upstream patch having been committed on 2019-06-05)
      // https://reviews.llvm.org/D54258
      OS << "#ifndef __shared\n"
         << "#define __device __device__\n"
         << "#define __constant __constant__\n"
         << "#define __shared __shared__\n"
         << "#endif\n\n";
      break;
    case Language::C99:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU: break;
  }

  // declarations of textures, surfaces, variables, includes, definitions etc.
  size_t num_arg;
  for (auto K : *l) {
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
            case Language::CUDA:
              OS << "__device__ __constant__ ";
              break;
            case Language::C99:
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU: break;
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
            case Language::C99:
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU: break;
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
  for (auto K : *l) {
    for (auto fun : K->getFunctionCalls()) {
      switch (compilerOptions.getTargetLang()) {
        case Language::CUDA:
          OS << "__inline__ __device__ "; break;
        case Language::C99:
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU: break;
      }
      fun->print(OS, Policy);
    }
  }

  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    case Language::CUDA:
      OS << "__global__ ";
        OS << "__launch_bounds__ (" << (l->back())->getNumThreadsX()
           << "*" << (l->back())->getNumThreadsY() << ") ";
      break;
    case Language::C99:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU: break;
  }
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
        case Language::OpenCLACC:
        case Language::OpenCLCPU: break;
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

const bool ASTFuse::DEBUG =
#ifdef PRINT_DEBUG
    true;
#undef PRINT_DEBUG
#else
    false;
#endif


// vim: set ts=2 sw=2 sts=2 et ai:


