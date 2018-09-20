//
// Copyright (c) 2018, University of Erlangen-Nuremberg
// Copyright (c) 2014, Saarland University
// Copyright (c) 2015, University of Erlangen-Nuremberg
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

//===----------- HostDataDeps.cpp - Track data dependencies ---------------===//
//
// This file implements tracking of data dependencies to provide information
// for optimization such as kernel fusion
//
//===----------------------------------------------------------------------===//

#include "hipacc/Analysis/HostDataDeps.h"

namespace clang {
namespace hipacc {


void DependencyTracker::VisitDeclStmt(DeclStmt *S) {
  for (auto DI=S->decl_begin(), DE=S->decl_end(); DI!=DE; ++DI) {
    Decl *SD = *DI;

    if (SD->getKind() == Decl::Var) {
      VarDecl *VD = dyn_cast<VarDecl>(SD);

      // found Image decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Image)) {
        if (DEBUG) std::cout << "  Tracked Image declaration: "
                  << VD->getNameAsString() << std::endl;

        HipaccImage *Img = new HipaccImage(Context, VD,
            compilerClasses.getFirstTemplateType(VD->getType()));
        
        //// get the text string for the image width and height
        //std::string width_str  = convertToString(CCE->getArg(0));
        //std::string height_str = convertToString(CCE->getArg(1));

        // store Image definition
        imgDeclMap_[VD] = Img;
        dataDeps.addImage(VD, Img);
        break;
      }

      // found BoundaryCondition decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.BoundaryCondition)) {
        if (DEBUG) std::cout << "  Tracked BoundaryCondition declaration: "
                  << VD->getNameAsString() << std::endl;

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        HipaccBoundaryCondition *BC = nullptr;
        HipaccImage *Img = nullptr;
        size_t size_args = 0;

        for (size_t i=0, e=CCE->getNumArgs(); i!=e; ++i) {
          auto arg = CCE->getArg(i)->IgnoreParenCasts();

          auto dsl_arg = arg;
          if (auto call = dyn_cast<CXXOperatorCallExpr>(arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument specifies the image
            if (imgDeclMap_.count(DRE->getDecl())) {
              Img = imgDeclMap_[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Img);
              bcDeclMap_[VD] = BC;
              dataDeps.addBoundaryCondition(VD, BC, DRE->getDecl());
              continue;
            }

            // check if the argument is a Mask
            if (maskDeclMap_.count(DRE->getDecl())) {
              HipaccMask *Mask = maskDeclMap_[DRE->getDecl()];
              BC->setSizeX(Mask->getSizeX());
              BC->setSizeY(Mask->getSizeY());
              continue;
            }

            // check if the argument specifies the boundary mode
            if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
                DRE->getDecl()->getType().getAsString() ==
                "enum hipacc::Boundary") {
              auto lval = arg->EvaluateKnownConstInt(Context);
              auto cval = static_cast<std::underlying_type<Boundary>::type>(Boundary::CONSTANT);
              assert(lval.isNonNegative() && lval.getZExtValue() <= cval &&
                     "invalid Boundary mode");
              auto mode = static_cast<Boundary>(lval.getZExtValue());
              BC->setBoundaryMode(mode);

              if (mode == Boundary::CONSTANT) {
                // check if the parameter can be resolved to a constant
                auto const_arg = CCE->getArg(++i);
                if (!const_arg->isEvaluatable(Context)) {
                  //Diags.Report(arg->getExprLoc(), IDConstMode) << VD->getName();
                } else {
                  Expr::EvalResult val;
                  const_arg->EvaluateAsRValue(val, Context);
                  BC->setConstVal(val.Val, Context);
                }
              }
              continue;
            }
          }

          // check if the argument can be resolved to a constant
          if (size_args++ == 0) {
            BC->setSizeX(arg->EvaluateKnownConstInt(Context).getSExtValue());
            BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
          } else {
            BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
          }
        }

        break;
      }

      // found Accessor decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Accessor)) {
        if (DEBUG) std::cout << "  Tracked Accessor declaration: "
                << VD->getNameAsString() << std::endl;
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        HipaccAccessor *Acc = nullptr;
        HipaccBoundaryCondition *BC = nullptr;
        HipaccImage *Img = nullptr;
        Interpolate mode = Interpolate::NO;

        // check if the first argument is an Image
        DeclRefExpr *DRE = nullptr;

        for (auto arg : CCE->arguments()) {
          auto dsl_arg = arg->IgnoreParenCasts();
          if (isa<CXXDefaultArgExpr>(dsl_arg))
            continue;

          if (auto call = dyn_cast<CXXOperatorCallExpr>(dsl_arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }
          // match for DSL arguments
          if (isa<DeclRefExpr>(dsl_arg)) {
            DRE = dyn_cast<DeclRefExpr>(dsl_arg);
            // check if the argument specifies the boundary condition
            if (bcDeclMap_.count(DRE->getDecl())) {
              BC = bcDeclMap_[DRE->getDecl()];
              continue;
            }

            // check if the argument specifies the image
            //if (!BC && imgDeclMap_.count(DRE->getDecl())) 
            if (imgDeclMap_.count(DRE->getDecl())) {
              Img = imgDeclMap_[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Img);
              BC->setSizeX(1);
              BC->setSizeY(1);
              BC->setBoundaryMode(Boundary::CLAMP);
              continue;
            }
          }
        }

        assert(DRE != nullptr && "First Accessor argument is not a BC or Image");
        assert(BC != nullptr && "Expected BoundaryCondition in HostDataDep");

        Acc = new HipaccAccessor(VD, BC, mode, false);
        // store Accessor definition
        accDeclMap_[VD] = Acc;
        dataDeps.addAccessor(VD, Acc, DRE->getDecl());
        break;
      }

      // found IterationSpace decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.IterationSpace)) {
        if (DEBUG) std::cout << "  Tracked IterationSpace declaration: "
                << VD->getNameAsString() << std::endl;
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        HipaccIterationSpace *IS = nullptr;
        HipaccImage *Img = nullptr;

        // check if the first argument is an Image
        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the Image from the DRE if we have one
          if (imgDeclMap_.count(DRE->getDecl())) {
            if (DEBUG) std::cout << "    -> Based on Image: "
                    << DRE->getNameInfo().getAsString() << std::endl;

            Img = imgDeclMap_[DRE->getDecl()];
            IS = new HipaccIterationSpace(VD, Img, false);

            dataDeps.addIterationSpace(VD, IS, DRE->getDecl());
          }
        }

        // store IterationSpace
        iterDeclMap_[VD] = IS;
        break;
      }

      // found Mask decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Mask)) {
        if (DEBUG) std::cout << "  Tracked Mask declaration: "
                  << VD->getNameAsString() << std::endl;
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        QualType QT = compilerClasses.getFirstTemplateType(VD->getType());
        HipaccMask *Mask = new HipaccMask(VD, QT, HipaccMask::MaskType::Mask);

        // get initializer
        DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0)->IgnoreParenCasts());
        assert(DRE && "Mask must be initialized using a variable");
        VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        assert(V && "Mask must be initialized using a variable");
        bool isMaskConstant = V->getType().isConstant(Context);

        // extract size_y and size_x from type
        auto Array = Context.getAsConstantArrayType(V->getType());
        Mask->setSizeY(Array->getSize().getSExtValue());
        Array = Context.getAsConstantArrayType(Array->getElementType());
        Mask->setSizeX(Array->getSize().getSExtValue());

        // loop over initializers and check if each initializer is a constant
        if (isMaskConstant) {
          if (auto ILEY = dyn_cast<InitListExpr>(V->getInit())) {
            Mask->setInitList(ILEY);
            for (auto yinit : *ILEY) {
              auto ILEX = dyn_cast<InitListExpr>(yinit);
              for (auto xinit : *ILEX) {
                auto xexpr = dyn_cast<Expr>(xinit);
                if (!xexpr->isConstantInitializer(Context, false)) {
                  isMaskConstant = false;
                  break;
                }
              }
            }
          }
        }
        Mask->setIsConstant(isMaskConstant);
        Mask->setHostMemName(V->getName());
      
        // store Mask definition
        maskDeclMap_[VD] = Mask;
        dataDeps.addMask(VD, Mask);
        break;
      }

      // found Domain decl
      if (compilerClasses.isTypeOfClass(VD->getType(),
          compilerClasses.Domain)) {
        break;
      }

      // found Kernel decl
      if (VD->getType()->getTypeClass() == Type::Record) {
        std::string className =
            VD->getType()->getAsCXXRecordDecl()->getNameAsString();
        std::string varName = VD->getNameAsString();
        if (DEBUG) std::cout << "  Tracked Kernel declaration: " << className
                << " " << varName
                << std::endl;

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));
          if (iterDeclMap_.count(DRE->getDecl())) {
            if (DEBUG) std::cout << "    -> Based on IterationSpace: "
                    << DRE->getNameInfo().getAsString() << std::endl;
          }

          std::vector<ValueDecl*> accs;
          for (auto it = ++(CCE->arg_begin()); it != CCE->arg_end(); ++it) {
            if (isa<DeclRefExpr>(*it)) {
              DeclRefExpr *arg = dyn_cast<DeclRefExpr>(*it);
              if (accDeclMap_.count(arg->getDecl())) {
                if (DEBUG) std::cout << "    -> Based on Accessor: "
                        << arg->getNameInfo().getAsString()
                        << std::endl;
                accs.push_back(arg->getDecl());
              }
            }
          }
          dataDeps.addKernel(VD, DRE->getDecl(), accs);
          break;
        }
      }
    }
  }
}


void DependencyTracker::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  Expr *Ex = E->getCallee();
  if (isa<MemberExpr>(Ex)) {
    MemberExpr *ME = dyn_cast<MemberExpr>(Ex);
    Expr *base = ME->getBase()->ignoreParenBaseCasts();
    if (isa<DeclRefExpr>(base)) {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(base);
      CXXRecordDecl *CRD = E->getRecordDecl();
      if (CRD != nullptr) {
        if (CRD->getNameAsString() == "Kernel" &&
            E->getMethodDecl()->getNameAsString() == "execute") {
          std::string className =
              DRE->getType()->getAsCXXRecordDecl()->getNameAsString();
          std::string varName = DRE->getDecl()->getNameAsString();
          if (DEBUG) std::cout << "  Tracked Kernel call: "
                  << className << " " << varName
                  << std::endl;
          dataDeps.runKernel(DRE->getDecl());
        }
      }
    }
  }
}


void HostDataDeps::addImage(ValueDecl *VD, HipaccImage *img) {
  assert(!imgMap_.count(VD) && "Duplicate Image declaration");
  imgMap_[VD] = new Image(img);
}


void HostDataDeps::addMask(ValueDecl *VD, HipaccMask *mask) {
  assert(!maskMap_.count(VD) && "Duplicate BoundaryCondition declaration");
  maskMap_[VD] = new Mask(mask);
}


void HostDataDeps::addBoundaryCondition(
    ValueDecl *BCVD, HipaccBoundaryCondition *BC, ValueDecl *IVD) {
  assert(imgMap_.count(IVD) && "Image was not declared");
  assert(!bcMap_.count(BCVD) && "Duplicate BoundaryCondition declaration");
  bcMap_[BCVD] = new BoundaryCondition(BC, imgMap_[IVD]);
}


void HostDataDeps::addKernel(
    ValueDecl *KVD, ValueDecl *ISVD, std::vector<ValueDecl*> AVDS) {
  Kernel *kernel;
  assert(iterMap_.count(ISVD) && "IterationSpace was not declared");
  assert(KernelClassDeclMap.count(KVD->getType()->getAsCXXRecordDecl()) &&
          "Kernel class was not declared");
  assert(!kernelMap_.count(KVD) && "Duplicate Kernel declaration");

  kernel = new Kernel(
      KVD->getType()->getAsCXXRecordDecl()->getNameAsString()
        .append(KVD->getNameAsString()),
          iterMap_[ISVD], KVD,
            KernelClassDeclMap[KVD->getType()->getAsCXXRecordDecl()]);

  for (auto it = AVDS.begin(); it != AVDS.end(); ++it) {
    assert(accMap_.count(*it) && "Accessor was not declared");
    kernel->addAccessor(accMap_[*it]);
  }
  kernelMap_[KVD] = kernel;
}


void HostDataDeps::addAccessor(
    ValueDecl *AVD, HipaccAccessor *acc, ValueDecl* IVD) {
  Image *img;

  if (imgMap_.count(IVD)) {
    img = imgMap_[IVD];
  } else {
    if (!bcMap_.count(IVD)) {
      assert(false && "Image or BoundaryCondition was not declared");
    } else {
      img = bcMap_[IVD]->getImage();
    }
  }

  assert(!accMap_.count(AVD) && "Duplicate Accessor declaration");
  accMap_[AVD] = new Accessor(acc, img);
}


void HostDataDeps::addIterationSpace(
    ValueDecl *ISVD, HipaccIterationSpace *iter, ValueDecl *IVD) {
  assert(imgMap_.count(IVD) && "Image was not declared");
  assert(!iterMap_.count(ISVD) && "Duplicate IterationSpace declaration");
  iterMap_[ISVD] = new IterationSpace(iter, imgMap_[IVD]);
}


void HostDataDeps::runKernel(ValueDecl *VD) {
  assert(kernelMap_.count(VD) && "Kernel was not declared");
  Kernel *kernel = kernelMap_[VD];

  // Create new process and output space
  Space *space = new Space(kernel->getIterationSpace()->getImage());
  Process *proc = new Process(kernel, space);
  space->setSrcProcess(proc);
  spaces_.push_back(space);
  processes_.push_back(proc);

  // TODO to be rmd
  assert(!processMap_.count(kernel->getName()) &&
          "Duplicate process declaration, kernel name exists");
  processMap_[kernel->getName()] = proc;
  processVisitorMap_[proc] = false;
  assert(!spaceMap_.count(kernel->getIterationSpace()->getImage()->getName()) &&
          "Duplicate space declaration, image name exists");
  spaceMap_[kernel->getIterationSpace()->getImage()->getName()] = space;

  // Set process to destination for all predecessor spaces:
  std::vector<Accessor*> accs = kernel->getAccessors();
  for (auto it = accs.begin(); it != accs.end(); ++it) {
    Space *s = nullptr;
    for (auto it2 = spaces_.rbegin(); it2 != spaces_.rend(); ++it2) {
      if ((*it)->getImage() == (*it2)->getImage()) {
        s = *it2;
        break;
      }
    }
    if (s == nullptr) {
      s = new Space((*it)->getImage());
      (*it)->setSpace(s);
      spaces_.push_back(s);
    }
    s->addDstProcess(proc);
    proc->addInputSpace(s);
  }
}


//void HostDataDeps::dump(Process *proc) {
//  std::cout << " <- " << proc->getKernel()->getName();
//  std::vector<Space*> spaces = proc->getInSpaces();
//  for (auto it = spaces.begin(); it != spaces.end(); ++it) {
//    dump(*it);
//    if (it+1 != spaces.end()) {
//      std::cout << std::endl << "            ";
//      for (size_t i = 0; i < proc->getKernel()->getName().size(); ++i) {
//        std::cout << " ";
//      }
//      std::cout << "|";
//    }
//  }
//}
//
//
//void HostDataDeps::dump(Space *space) {
//  if (space->getAccessors().size() > 0) {
//    std::cout << " <- ";
//  }
//  std::cout << space->getImage()->getName();
//
//  Process *proc = space->getSrcProcess();
//  if (proc != nullptr) {
//    dump(proc);
//  }
//}


void HostDataDeps::dump() {
  std::cout << "  Application Graph:" << std::endl;
  for (auto pL : applicationGraph) {
    for (auto p : *pL) {
      std::cout << " --> " << p->getKernel()->getName();
    }
    std::cout << std::endl;
  }

  std::cout << "  Weight:" << std::endl;
  for (auto we : edgeWeightMap_) {
    std::cout << " " << (we.first.first)->getKernel()->getName() << " - "  << 
      (we.second) << " -> " << (we.first.second)->getKernel()->getName() << std::endl;
  }

  //for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
  //  IterationSpace *iter = (*it)->getIterationSpace();
  //  std::cout << "    - img: " << (*it)->getImage()->getName()
  //            << ", iter: " << (iter ? iter->getName() : "nullptr")
  //            << std::endl;
  //}

  //std::cout << "  Processes:" << std::endl;
  //for (auto it = processes_.begin(); it != processes_.end(); ++it) {
  //  std::cout << "    - kernel: " << (*it)->getKernel()->getName()
  //            << ", out: " << (*it)->getOutSpace()->getImage()->getName()
  //            << ", read dependent on: " << ((*it)->getReadDependentProcess() ?
  //                ((*it)->getReadDependentProcess())->getKernel()->getName() : "nullptr")
  //            << ", write dependent on: " << ((*it)->getWriteDependentProcess() ?
  //                ((*it)->getWriteDependentProcess())->getKernel()->getName() : "nullptr")
  //            << std::endl;
  //}

  //std::cout << "  Dependency Tree:" << std::endl;
  //int id = 1;
  //std::vector<Image*> done;
  //for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
  //  if (!findVector(done, (*it)->getImage()) &&
  //      (*it)->getAccessors().size() == 0) {
  //    std::cout << "    " << id << ":";
  //    dump(*it);
  //    std::cout << std::endl;
  //    ++id;
  //    done.push_back((*it)->getImage());
  //  }
  //}
}


std::vector<HostDataDeps::Space*> HostDataDeps::getInputSpaces() {
  std::vector<Space*> ret;
  for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
    if ((*it)->getSrcProcess() == nullptr) { ret.push_back(*it); }
  }
  return ret;
}


std::vector<HostDataDeps::Space*> HostDataDeps::getOutputSpaces() {
  std::vector<Space*> ret;
  for (auto it = spaces_.rbegin(); it != spaces_.rend(); ++it) {
    if ((*it)->getDstProcesses().empty()) { ret.push_back(*it); }
  }
  return ret;
}


void HostDataDeps::markProcess(Process *t) {
  for (auto S: t->getInSpaces()) { markSpace(S); }
}


void HostDataDeps::markSpace(Space *s) {
  std::vector<Process*> DstProcesses = s->getDstProcesses();
  Process *SrcProcess = s->getSrcProcess();
  if (SrcProcess) {
    if (!processVisitorMap_[SrcProcess]) {
      std::list<Process*> *list = new std::list<Process*>;
      list->push_back(SrcProcess);
      for (auto dp : DstProcesses) { list->push_back(dp); }
      applicationGraph.push_back(list);
      processVisitorMap_[SrcProcess] = true;
    }
    markProcess(SrcProcess);

    //if (DstProcesses.size() == 1 &&
    //     ((DstProcesses.back())->getInSpaces().size() == 1)) {
    //  SrcProcess->SetWriteDependentProcess(DstProcesses.back());
    //  DstProcesses.back()->SetReadDependentProcess(SrcProcess);
    //  recordFusibleProcessPair(SrcProcess, DstProcesses.back());
    //} else if (DstProcesses.size() == 1 &&
    //            ((DstProcesses.back())->getInSpaces().size() == 2)) {
    //  // get the second input img of the dest p
    //  Space *imgExt;
    //  if ((DstProcesses.back())->getInSpaces().front() == s) {
    //    imgExt = (DstProcesses.back())->getInSpaces().back();
    //  } else {
    //    imgExt = (DstProcesses.back())->getInSpaces().front();
    //  }
    //  // get the second input img of the dest p
    //  std::vector<Space*> vecImgEnt = SrcProcess->getInSpaces();
    //  bool shareExtSpace = (std::find(std::begin(vecImgEnt), std::end(vecImgEnt),
    //                         imgExt) != std::end(vecImgEnt)) ? true : false;
    //  if (shareExtSpace) {
    //    imgExt->setSpaceShared();
    //    // record as fusible kernel pair
    //    SrcProcess->SetWriteDependentProcess(DstProcesses.back());
    //    DstProcesses.back()->SetReadDependentProcess(SrcProcess);
    //    recordFusibleProcessPair(SrcProcess, DstProcesses.back());
    //  }
    //}
  }
}

//void HostDataDeps::recordFusibleProcessPair(Process *pSrc, Process *pDest) {
//  if (std::find(fusibleProcesses_.begin(), fusibleProcesses_.end(), pDest) == fusibleProcesses_.end())
//      fusibleProcesses_.push_back(pDest);
//  if (std::find(fusibleProcesses_.begin(), fusibleProcesses_.end(), pSrc) == fusibleProcesses_.end())
//      fusibleProcesses_.push_back(pSrc);
//}

void HostDataDeps::createSchedule() {
  for (auto S : getOutputSpaces()) { 
    markSpace(S); 
  }

  // weight computation and assignment 
  for (auto pL : applicationGraph) {
    Process *srcProcess = *(pL->begin());
    unsigned nALU = srcProcess->getKernel()->getKernelClass()->getKernelStatistics().getNumOpALUs(); 
    unsigned nSFU = srcProcess->getKernel()->getKernelClass()->getKernelStatistics().getNumOpSFUs(); 
    unsigned costOP = CALU * nALU + CSFU * nSFU;
    unsigned ISks = srcProcess->getKernel()->getKernelClass()->getKernelStatistics().getNumImgLoads();

    for (auto itEdge = std::next(pL->begin()); itEdge != pL->end(); ++itEdge) {
      Process *destProcess = *itEdge;
      unsigned w = 0;
      if ((srcProcess->getOutSpace()->getDstProcesses()).size() > 1 || 
          (destProcess->getInSpaces()).size() > 1) {        //illegal
        w = EPSILON;
      } else if (destProcess->getKernel()->getKernelClass()->getKernelType() == 
          PointOperator) {                                  // point-based
        w = TG;
      } else if (srcProcess->getKernel()->getKernelClass()->getKernelType() == 
          PointOperator) {                                  // point-to-local
        auto acc = destProcess->getKernel()->getAccessors().back();
        unsigned szKd =acc->getSizeX() * acc->getSizeY(); 
        unsigned costComp = costOP * ISks * szKd;
        w = TG - costComp;
      } else if (srcProcess->getKernel()->getKernelClass()->getKernelType() == 
          LocalOperator) {                                  // local-to-local
        auto accSrc = srcProcess->getKernel()->getAccessors().back();
        auto accDest = destProcess->getKernel()->getAccessors().back();
        auto szKf = (accDest->getSizeX() + static_cast<unsigned>(floor(accSrc->getSizeX() / 2)) * 2) *
                    (accDest->getSizeX() + static_cast<unsigned>(floor(accSrc->getSizeX() / 2)) * 2);
        unsigned costComp = costOP * ISks * szKf;
        w = TS - costComp;
      } else {                                              // unsupported scenario
        w = EPSILON;
      }
      unsigned we = std::max(w + GAMMA, EPSILON);
      edgeWeightMap_[std::make_pair(srcProcess, destProcess)] = we;
    }
  }

  //// create an initial set of fusion lists
  //for (auto p : fusibleProcesses_) {
  //  if (p->getReadDependentProcess() && !p->getWriteDependentProcess()) {
  //    std::list<Process*> *list = new std::list<Process*>;
  //    list->push_back(p);
  //    FusibleKernelListsMap[p] = list;
  //  } else if (p->getWriteDependentProcess()) {
  //    Process *pw = p->getWriteDependentProcess();
  //    std::list<Process*> *list = FusibleKernelListsMap[pw];
  //    auto it = std::find(list->begin(), list->end(), pw);
  //    assert((it != list->end()) && "cannot find write dependent process for kernel fusion");
  //    list->insert(it, p);
  //    FusibleKernelListsMap[p] = list;
  //  }
  //}
  //for (auto pair: FusibleKernelListsMap) {
  //  bool merge = true;
  //  for (auto l: vecFusibleKernelLists) {
  //    if (l == pair.second) { merge = false; break;}
  //  }
  //  if (merge) vecFusibleKernelLists.push_back(pair.second);
  //}
}


void HostDataDeps::minCutGlobal(partitionBlock PB, partitionBlock &PBRet0, 
                                partitionBlock &PBRet1) {
  // Stoer-Wagner Minimum Cut
  PBRet1.push_back(PB.back());
  PB.pop_back();
  PBRet0 = PB;
}


bool HostDataDeps::isLegal(partitionBlock const PB) {
  // external dependency detection
  if (PB.size() == 1) { return true; } 
  bool isLegal = true;
  std::vector<Space*> vecSrcSpaces;

  for (auto pL : PB) {
    Process *p = pL->begin();
    if (isDest(p)) {
      num_destP++;
    } else if (isSrc(p)) { // all src kernels can share the same input
      if (vecSrcSpaces.empty()) {
        vecSrcSpaces = p->getInSpaces();
      } else if (vecSrcSpaces != p->getInSpaces()) {
        isLegal = false; break;
      }
    } else {
      for (auto dp : p->getOutSpace()->getDstProcesses()) {


      }
    }
  }

  if (num_destP > 1) { isLegal = false; }
  return isLegal;
}


void HostDataDeps::fusibilityAnalysis() {
  std::set<partitionBlock> readySet;
  std::set<partitionBlock> workingSet;

  // initialization
  workingSet.insert(applicationGraph);
  while(!workingSet.empty()) {
    for (auto PB : workingSet) {
      if ((PB.size() == 1) || isLegal(PB)) {
        readySet.insert(PB);
        workingSet.erase(PB);
      } else {
        partitionBlock PBRet0, PBRet1;
        //llvm::errs() << "\n";
        //llvm::errs() << "before cut\n";
        //llvm::errs() << "PB size: " << PB.size();
        //llvm::errs() << "  PBRet0 size: " << PBRet0.size();
        //llvm::errs() << "  PBRet1 size: " << PBRet1.size();
        //llvm::errs() << "\n";
        minCutGlobal(PB, PBRet0, PBRet1);
        //llvm::errs() << "\n";
        //llvm::errs() << "after cut\n";
        //llvm::errs() << "PB size: " << PB.size();
        //llvm::errs() << "  PBRet0 size: " << PBRet0.size();
        //llvm::errs() << "  PBRet1 size: " << PBRet1.size();
        //llvm::errs() << "\n";
        workingSet.insert(PBRet0);
        workingSet.insert(PBRet1);
        workingSet.erase(PB);
      }
    }
  }
}


void HostDataDeps::recordFusibleKernelListInfo() {
  unsigned posCnt;
  unsigned listPosCnt = 0;
  llvm::errs() << "Fusibility analysis for Kernel Fusion: \n";
  for (auto l : vecFusibleKernelLists) {
    llvm::errs() << "  Fusible kernel list " << listPosCnt << ": \n";
    posCnt = 0;
    for (auto p : *l) {
      FusibleProcessInfoFinalMap[p] = std::make_tuple(listPosCnt, posCnt);
      FusibleProcessListSizeFinalMap[p] = l->size();
      posCnt++;
      llvm::errs() << "  --->  " << p->getKernel()->getName();
    }
    llvm::errs() << "\n";
    listPosCnt++;
  }
  llvm::errs() << "\n";
}

// new
bool HostDataDeps::isSrc(Process *P) {
  std::vector<Space*> S = P->getInSpaces();
  return std::any_of(S.begin(), S.end(), [](Space *s){return s->getSrcProcess() == nullptr;}) 
}

bool HostDataDeps::isDest(Process *P) {
  Space *s = P->getOutSpace();
  return s->getDstProcesses().empty();
}

// old TODO
bool HostDataDeps::isFusible(HipaccKernel *K) {
  bool isFusible = false;
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  if (FusibleProcessInfoFinalMap.count(p)) { isFusible = true; }
  return isFusible;
}

bool HostDataDeps::hasSharedIS(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  std::vector<Space*> spaces = processMap_[fullName]->getInSpaces();
  return std::any_of(spaces.begin(), spaces.end(), [](Space *s){return s->isSpaceShared();});
}

std::string HostDataDeps::getSharedISName(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  std::vector<Space*> spaces = processMap_[fullName]->getInSpaces();
  auto it = std::find_if(spaces.begin(), spaces.end(), [](Space *s){return s->isSpaceShared();});
  return (*it)->getImage()->getName();
}

bool HostDataDeps::isSrc(HipaccKernel *K) {
  bool isSrc = false;
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  if (std::get<1>(FusibleProcessInfoFinalMap[p]) == 0) { isSrc = true; }
  return isSrc;
}

bool HostDataDeps::isDest(HipaccKernel *K) {
  bool isDest = false;
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  if (std::get<1>(FusibleProcessInfoFinalMap[p]) ==
        FusibleProcessListSizeFinalMap[p]-1) {
    isDest = true;
  }
  return isDest;
}

unsigned HostDataDeps::getNumberOfFusibleKernelList() const {
  return vecFusibleKernelLists.size();
}

unsigned HostDataDeps::getKernelListIndex(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  assert(FusibleProcessInfoFinalMap.count(p) && "Kernel has no corresponding fusible process");
  return std::get<0>(FusibleProcessInfoFinalMap[p]);
}

unsigned HostDataDeps::getKernelIndex(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  assert(FusibleProcessInfoFinalMap.count(p) && "Kernel has no corresponding fusible process");
  return std::get<1>(FusibleProcessInfoFinalMap[p]);
}

unsigned HostDataDeps::getKernelListSize(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];
  assert(FusibleProcessListSizeFinalMap.count(p) && "Kernel has no corresponding fusible process");
  return FusibleProcessListSizeFinalMap[p];
}


const bool HostDataDeps::DEBUG =
#ifdef PRINT_DEBUG
    true;
#undef PRINT_DEBUG
#else
    false;
#endif

const bool DependencyTracker::DEBUG = HostDataDeps::DEBUG;


}
}





