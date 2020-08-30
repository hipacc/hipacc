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
        hipacc_require(isa<CXXConstructExpr>(VD->getInit()),
               "Expected BoundaryCondition definition (CXXConstructExpr).");
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
              hipacc_require(lval.isNonNegative() && lval.getZExtValue() <= cval,
                     "invalid Boundary mode");
              auto mode = static_cast<Boundary>(lval.getZExtValue());
              BC->setBoundaryMode(mode);

              if (mode == Boundary::CONSTANT) {
                // check if the parameter can be resolved to a constant
                auto const_arg = CCE->getArg(++i);
                if (!const_arg->isEvaluatable(Context)) {
                  //Diags.Report(arg->getExprLoc(), IDConstMode) << VD->getName();
                  hipacc_require(false, "require constant for Boundary Handling");
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
          if (!arg->isEvaluatable(Context)) {
            hipacc_require(false, "needs to be evaluable");
          }
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
        ValueDecl* BCVD = nullptr;
        HipaccImage *Img = nullptr;
        ValueDecl* ImgVD = nullptr;
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
              BCVD = DRE->getDecl();
              continue;
            }

            // check if the argument specifies the image
            //if (!BC && imgDeclMap_.count(DRE->getDecl()))
            if (imgDeclMap_.count(DRE->getDecl())) {
              Img = imgDeclMap_[DRE->getDecl()];
              ImgVD = DRE->getDecl();
              BC = new HipaccBoundaryCondition(VD, Img);
              BC->setSizeX(1);
              BC->setSizeY(1);
              BC->setBoundaryMode(Boundary::CLAMP);
              continue;
            }
          }
        }

        hipacc_require(DRE != nullptr, "First Accessor argument is not a BC or Image");
        hipacc_require(BC != nullptr, "Expected BoundaryCondition in HostDataDep");

        Acc = new HipaccAccessor(VD, BC, mode, false);
        // store Accessor definition
        accDeclMap_[VD] = Acc;

        if (BCVD) {
          dataDeps.addAccessor(VD, Acc, BCVD);
        } else if (ImgVD) {
          dataDeps.addAccessor(VD, Acc, ImgVD);
        } else {
          hipacc_require(false, "First Accessor argument is not a BC or Image");
        }
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
        hipacc_require(DRE, "Mask must be initialized using a variable");
        VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        hipacc_require(V, "Mask must be initialized using a variable");
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
        if (DEBUG) std::cout << "  Tracked Domain declaration: "
                  << VD->getNameAsString() << std::endl;

        HipaccMask *Domain = new HipaccMask(VD, Context.UnsignedCharTy,
                                            HipaccMask::MaskType::Domain);
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        if (CCE->getNumArgs() == 1) {
          // get initializer
          auto DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0)->IgnoreParenCasts());
          hipacc_require(DRE, "Domain must be initialized using a variable");
          VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
          hipacc_require(V, "Domain must be initialized using a variable");

          if (compilerClasses.isTypeOfTemplateClass(DRE->getType(),
                                                    compilerClasses.Mask)) {
            // copy from mask
            HipaccMask *Mask = maskDeclMap_[DRE->getDecl()];
            hipacc_require(Mask, "Mask to copy from was not declared");

            size_t size_x = Mask->getSizeX();
            size_t size_y = Mask->getSizeY();

            Domain->setSizeX(size_x);
            Domain->setSizeY(size_y);

            Domain->setIsConstant(Mask->isConstant());

            if (Mask->isConstant()) {
              for (size_t x=0; x<size_x; ++x) {
                for (size_t y=0; y<size_y; ++y) {
                  // copy values to compiler internal data structure
                  Expr::EvalResult val;
                  Mask->getInitExpr(x, y)->EvaluateAsRValue(val, Context);
                  if (val.Val.isInt()) {
                    Domain->setDomainDefined(x, y,
                        val.Val.getInt().getSExtValue() != 0);
                  } else if (val.Val.isFloat()) {
                    Domain->setDomainDefined(x, y,
                        !val.Val.getFloat().isZero());
                  } else {
                    hipacc_require(false, "Only builtin integer and floating point "
                                    "literals supported in copy Mask");
                  }
                }
              }
            } else {
              Domain->setCopyMask(Mask);
            }
          } else {
            // get from array
            bool isDomainConstant = V->getType().isConstant(Context);

            // extract size_y and size_x from type
            auto Array = Context.getAsConstantArrayType(V->getType());
            Domain->setSizeY(Array->getSize().getSExtValue());
            Array = Context.getAsConstantArrayType(Array->getElementType());
            Domain->setSizeX(Array->getSize().getSExtValue());

            // loop over initializers and check if each initializer is a
            // constant
            if (isDomainConstant) {
              if (auto ILEY = dyn_cast<InitListExpr>(V->getInit())) {
                Domain->setInitList(ILEY);
                for (size_t y=0; y<ILEY->getNumInits(); ++y) {
                  auto ILEX = dyn_cast<InitListExpr>(ILEY->getInit(y));
                  for (size_t x=0; x<ILEX->getNumInits(); ++x) {
                    auto xexpr = ILEX->getInit(x)->IgnoreParenCasts();
                    if (!xexpr->isConstantInitializer(Context, false)) {
                      isDomainConstant = false;
                      break;
                    }
                    // copy values to compiler internal data structure
                    if (auto val = dyn_cast<IntegerLiteral>(xexpr)) {
                      Domain->setDomainDefined(x, y, val->getValue() != 0);
                    } else {
                      hipacc_require(false, "Expected integer literal in domain initializer");
                    }
                  }
                }
              }
            }
            Domain->setIsConstant(isDomainConstant);
            Domain->setHostMemName(V->getName());
          }
        } else if (CCE->getNumArgs() == 2) {
          // check if the parameters can be resolved to a constant
          Expr *Arg0 = CCE->getArg(0);
          if (!Arg0->isEvaluatable(Context)) {
            hipacc_require(false, "parameter is not evaluable");
          }
          Domain->setSizeX(Arg0->EvaluateKnownConstInt(Context).getSExtValue());

          Expr *Arg1 = CCE->getArg(1);
          if (!Arg1->isEvaluatable(Context)) {
            hipacc_require(false, "parameter is not evaluable");
          }
          Domain->setSizeY(Arg1->EvaluateKnownConstInt(Context).getSExtValue());
          Domain->setIsConstant(true);
        } else {
          hipacc_require(false, "Domain definition requires exactly two arguments "
              "type constant integer or a single argument of type uchar[][] or "
              "Mask!");
        }
        // store Mask definition
        maskDeclMap_[VD] = Domain;
        dataDeps.addMask(VD, Domain);
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
        visitedKernelDecl_.push_back(VD);

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
          dataDeps.recordVisitedKernelDecl(DRE, visitedKernelDecl_);
        }
      }
    }
  }
}


void HostDataDeps::addImage(ValueDecl *VD, HipaccImage *img) {
  hipacc_require(!imgMap_.count(VD), "Duplicate Image declaration");
  imgMap_[VD] = new Image(img);
}


void HostDataDeps::addMask(ValueDecl *VD, HipaccMask *mask) {
  hipacc_require(!maskMap_.count(VD), "Duplicate BoundaryCondition declaration");
  maskMap_[VD] = new Mask(mask);
}


void HostDataDeps::addBoundaryCondition(
    ValueDecl *BCVD, HipaccBoundaryCondition *BC, ValueDecl *IVD) {
  hipacc_require(imgMap_.count(IVD), "Image was not declared");
  hipacc_require(!bcMap_.count(BCVD), "Duplicate BoundaryCondition declaration");
  bcMap_[BCVD] = new BoundaryCondition(BC, imgMap_[IVD]);
}


void HostDataDeps::addKernel(
    ValueDecl *KVD, ValueDecl *ISVD, std::vector<ValueDecl*> AVDS) {
  Kernel *kernel;
  hipacc_require(iterMap_.count(ISVD), "IterationSpace was not declared");
  hipacc_require(KernelClassDeclMap.count(KVD->getType()->getAsCXXRecordDecl()),
          "Kernel class was not declared");
  hipacc_require(!kernelMap_.count(KVD), "Duplicate Kernel declaration");

  kernel = new Kernel(
      KVD->getType()->getAsCXXRecordDecl()->getNameAsString()
        .append(KVD->getNameAsString()),
          iterMap_[ISVD], KVD,
            KernelClassDeclMap[KVD->getType()->getAsCXXRecordDecl()]);

  for (auto it = AVDS.begin(); it != AVDS.end(); ++it) {
    hipacc_require(accMap_.count(*it), "Accessor was not declared");
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
      hipacc_require(false, "Image or BoundaryCondition was not declared");
    } else {
      img = bcMap_[IVD]->getImage();
    }
  }

  hipacc_require(!accMap_.count(AVD), "Duplicate Accessor declaration");
  accMap_[AVD] = new Accessor(acc, img);
}


void HostDataDeps::addIterationSpace(
    ValueDecl *ISVD, HipaccIterationSpace *iter, ValueDecl *IVD) {
  hipacc_require(imgMap_.count(IVD), "Image was not declared");
  hipacc_require(!iterMap_.count(ISVD), "Duplicate IterationSpace declaration");
  iterMap_[ISVD] = new IterationSpace(iter, imgMap_[IVD]);
}


void HostDataDeps::recordVisitedKernelDecl(DeclRefExpr *DRE,
    std::vector<VarDecl *> &VKD) {
    std::vector<std::string> vProcName;
    // extract visited kernel decl
    for (auto VD : VKD) {
      std::string processName = VD->getType()->getAsCXXRecordDecl()->getNameAsString()
        .append(VD->getNameAsString());
      vProcName.push_back(processName);
    }

    std::string sProcName = DRE->getType()->getAsCXXRecordDecl()->getNameAsString()
      .append(DRE->getDecl()->getNameAsString());
    hipacc_require(processMap_.count(sProcName), "Missing process declaration");
    Process *sProc = processMap_[sProcName];
    visitedKernelDeclNameMap_[sProc] = vProcName;
}


void HostDataDeps::runKernel(ValueDecl *VD) {
  hipacc_require(kernelMap_.count(VD), "Kernel was not declared");
  Kernel *kernel = kernelMap_[VD];

  // Create new process and output space
  Space *space = new Space(kernel->getIterationSpace()->getImage());
  Process *proc = new Process(kernel, space);
  space->setSrcProcess(proc);
  spaces_.push_back(space);
  processes_.push_back(proc);

  hipacc_require(!processMap_.count(kernel->getName()),
          "Duplicate process declaration, kernel name exists");
  processMap_[kernel->getName()] = proc;
  processVisitorMap_[proc] = false;

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


void HostDataDeps::dump(partitionBlock &PB) {
  std::cout << "  Application Graph:" << std::endl;
  for (auto pL : PB) {
    for (auto p : *pL) {
      std::cout << " --> " << p->getKernel()->getName();
    }
    std::cout << std::endl;
  }
}


void HostDataDeps::dump(edgeWeight &wMap) {
  std::cout << "  Weight:" << std::endl;
  for (auto we : wMap) {
    std::cout << " " << (we.first.first)->getKernel()->getName() << " - "
      << (we.second) << " -> " << (we.first.second)->getKernel()->getName()
      << std::endl;
  }
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
  // for all non-input images
  if (SrcProcess) {
    if (!processVisitorMap_[SrcProcess]) {
      std::list<Process*> *list = new std::list<Process*>;
      list->push_back(SrcProcess);
      for (auto dp : DstProcesses) { list->push_back(dp); }
      applicationGraph.push_back(list);
      processVisitorMap_[SrcProcess] = true;
    }
    markProcess(SrcProcess);
  }
}


void HostDataDeps::generateSchedule() {
  for (auto S : getOutputSpaces()) {
    markSpace(S);
  }

  if (DEBUG) {
    std::cout << "  Host Data Dependence Graph: " << std::endl;
    dump(applicationGraph);
  }
}


// detect simple linear producer-consumer data dependence
void HostDataDeps::fusibilityAnalysisLinear() {
  partitionBlock workingBlock;
  partitionBlock readyBlock;
  for (auto pL : applicationGraph) {
    Process* producerP = pL->front();
    KernelType KT = producerP->getKernel()->getKernelClass()->getKernelType();
    // record only single comsumer kernel list
    if (pL->size() == 2 && (KT == PointOperator || KT == LocalOperator)) {
      workingBlock.push_back(pL);
    }
  }
  for (auto pL : workingBlock) {
    Process* consumerP = pL->back();
    KernelType KT = consumerP->getKernel()->getKernelClass()->getKernelType();
    // detect if the consumer kernel is used in other pairs (external input)
    bool hasExternInput = false;
    for (auto poL : applicationGraph) {
      if ((poL->size() > 1) && (pL != poL) && (poL->front() != consumerP)) {
        for (auto p : *poL) {
          if (p == consumerP) {
            hasExternInput = true;
            break;
          }
        }
      }
    }

    // record only single producer kernel list
    if (!hasExternInput && (KT == PointOperator || KT == LocalOperator)) {
      // check if the kernel decl has already been parsed for all process
      Process* producerP = pL->front();
      Process* consumerP = pL->back();
      std::string producerName = producerP->getKernel()->getName();
      std::string consumerName = consumerP->getKernel()->getName();
      auto VKVP = visitedKernelDeclNameMap_[producerP];
      auto VKVC = visitedKernelDeclNameMap_[consumerP];

      // generate hints for potential kernel fusion opportunities
      if ((std::find(VKVP.begin(), VKVP.end(), consumerName) != VKVP.end()) &&
         (std::find(VKVC.begin(), VKVC.end(), producerName) != VKVC.end())) {
        // check shared memory options for local-based fusion
        if ((KT == LocalOperator) && (!compilerOptions.useLocalMemory())) {
          llvm::errs() << "[Kernel Fusion INFO] hints:\n";
          llvm::errs() << " Kernel \"" << producerName << "\" and \"" << consumerName << "\" can be fused if shared memory option is enabled\n";
        } else {
          readyBlock.push_back(pL);
        }
      } else {
        llvm::errs() << "[Kernel Fusion INFO] hints:\n";
        llvm::errs() << " Kernel \"" << producerName << "\" and \"" << consumerName << "\" can be fused if all decls are positioned before execute() call\n";
      }
    }
  }

  // group all fusible pairs into partition blocks
  std::set<partitionBlock*> readySet;
  std::map<partitionBlock *, unsigned> LocalOpPBMap;
  std::reverse(readyBlock.begin(), readyBlock.end());
  for (auto pL : readyBlock) {
		bool isSamePB = false;
		bool skipPL = false;
    Process* producerP = pL->front();
    Process* consumerP = pL->back();
    int numLocalOpList = 0;
    for (auto p : *pL) {
      if (p->getKernel()->getKernelClass()->getKernelType() == LocalOperator) {
        numLocalOpList++;
      }
    }
  	for (auto pB : readySet) {
      // search data dependence within PB
      for (auto pfL : *pB) {
        Process* fproducerP = pfL->front();
        Process* fconsumerP = pfL->back();
        if ((producerP == fconsumerP) || (consumerP == fproducerP)) {
					isSamePB = true;
					break;
				}
			}

      if (isSamePB) {  // found the PB
        // check existing local operators in the pB
        // split local operator chains into different PBs
        if (LocalOpPBMap.find(pB) == LocalOpPBMap.end()) {
          pB->push_back(pL);
        } else {
          if (LocalOpPBMap[pB] < 1) {
            pB->push_back(pL);
          } else {
            isSamePB = false;
            skipPL = true;
          }
        }

        // update the number of local operator in the block
        if (LocalOpPBMap.find(pB) != LocalOpPBMap.end()) {
          LocalOpPBMap[pB] += numLocalOpList;
        } else {
          LocalOpPBMap[pB] = numLocalOpList;
        }
        break;
			}
		}
		if (!isSamePB && !skipPL) {
			partitionBlock *subG = new partitionBlock;
			subG->push_back(pL);
      readySet.insert(subG);
		}
  }

  // create new list for destination kernels, for PB completeness
  for (auto pB : readySet) {
	  std::list<Process*> destKList;
    for (auto pfL : *pB) {
			Process* fconsumerP = pfL->back();
      if (std::count_if(pB->begin(), pB->end(), [&](std::list<Process*> *pLL){
					return pLL->front() == fconsumerP;}) == 0){
				destKList.push_back(fconsumerP);
			}
		}
    for (auto p : destKList) {
      std::list<Process*> *list = new std::list<Process*>;
    	list->push_back(p);
			pB->push_back(list);
		}
	}

  // convert readySet to fusibleSetNames
  if (!readySet.empty()) {
    llvm::errs() << "[Kernel Fusion INFO] fusible kernels from linear analysis:\n";
    for (auto pB : readySet) {
      partitionBlockNames PBNam;
      llvm::errs() << " [ ";
      for (auto pL : *pB) {
        llvm::errs() << "{";
        std::list<std::string> lNam;
        for (auto p : *pL) {
          std::string kname = p->getKernel()->getName();
          llvm::errs() << " --> " << kname;
          lNam.push_back(kname);
        }
        llvm::errs() << "} ";
        PBNam.push_back(lNam);
      }
      llvm::errs() << "] \n";
      fusibleSetNames.insert(PBNam);
    }
  }
}


//***** Min-cut based graph partitioning functions*****//
void HostDataDeps::fusibilityAnalysis() {
  std::set<partitionBlock> readySet;
  std::set<partitionBlock> workingSet;

  // initialization
  workingSet.insert(applicationGraph);
  if (DEBUG) {
    dump(applicationGraph);
  }
  while(!workingSet.empty()) {
    std::set<partitionBlock> legalSet;
    std::set<partitionBlock> illegalSet;
    for (auto PB : workingSet) {
      if ((PB.size() == 1) || isLegal(PB)) {
        legalSet.insert(PB);
      } else {
        illegalSet.insert(PB);
      }
    }

    for (auto PB : legalSet) {
      workingSet.erase(PB);
      readySet.insert(PB);
    }
    for (auto PB : illegalSet) {
      partitionBlock PBRet0, PBRet1;
      minCutGlobal(PB, PBRet0, PBRet1);
      workingSet.insert(PBRet0);
      workingSet.insert(PBRet1);
      workingSet.erase(PB);
    }
  }

  // mark shared IS
  for (auto PB : readySet) {
    std::set<Space*> setSharedIS;
    for (auto pL : PB) {
      for (auto ss : pL->front()->getInSpaces()) {
        Process *sp = ss->getSrcProcess();
        if ((sp == nullptr) ||
           (std::count_if(PB.begin(), PB.end(), [&](std::list<Process*> *pLL){
                          return pLL->front() == sp;}) == 0)){
          if (setSharedIS.count(ss) == 0) {
            setSharedIS.insert(ss);
          } else {
            ss->setSpaceShared();
          }
        }
      }
    }
  }

  // recording analysis result
  llvm::errs() << "  Fusibility Analysis Result: \n";
  llvm::errs() << "--------------------------------\n\n";
  for (auto PB : readySet) {
    partitionBlockNames PBNam;
    for (auto pL : PB) {
      std::list<std::string> lNam;
      for (auto p : *pL) {
        std::string kname = p->getKernel()->getName();
        llvm::errs() << " --> " << kname;
        lNam.push_back(kname);
      }
      llvm::errs() << "\n";
      PBNam.push_back(lNam);
    }
    llvm::errs() << "--------------------------------\n";
    fusibleSetNames.insert(PBNam);
  }
}


void HostDataDeps::minCutGlobal(partitionBlock PB, partitionBlock &PBRet0,
                                partitionBlock &PBRet1) {
  // Stoer-Wagner Minimum Cut
  // initialization
  partitionBlock PBOrig;
  for (auto pL : PB) {
    std::list<Process*> *lPLocal = new std::list<Process*>;
    for (auto p : *pL) { lPLocal->push_back(p); }
    PBOrig.push_back(lPLocal);
  }

  edgeWeight curEdgeWeightMap;
  for (auto we : edgeWeightMap_) {
    if (std::any_of(PB.begin(), PB.end(), [&](std::list<Process*> *pL0){
          return (pL0->front() == we.first.first) &&
            (std::find(pL0->begin(), pL0->end(), we.first.second) != pL0->end());})) {
      auto value = we.second;
      auto key = we.first;
      curEdgeWeightMap[key] = value;
    }
  }

  // min cut
  unsigned wMin = UINT_MAX;
  partitionBlock PBContr;
  std::pair<Process *, Process *> STPair;
  for (auto i = PBOrig.size(); i != 1; i--) {
    unsigned wCur = minCutPhase(PB, curEdgeWeightMap, STPair);
    auto it = std::find_if(PBContr.begin(), PBContr.end(), [&](std::list<Process*> *pL0){return pL0->front() == STPair.first;});
    if (it == PBContr.end()) {
      std::list<Process*> *lPLocal = new std::list<Process*>;
      lPLocal->push_back(STPair.first);
      lPLocal->push_back(STPair.second);
      PBContr.push_back(lPLocal);
    } else {
      (*it)->push_back(STPair.second);
    }

    if (wCur < wMin) {
      wMin = wCur;
      PBRet0.clear();
      // compute PBRet0
      // get t and its represent nodes
      std::set<Process*> Ts;
      Ts.insert(STPair.second);
      for (auto pL : PBContr) {
        if (pL->front() == STPair.second) {
          for (auto p:*pL) {
            if (p != STPair.second) {Ts.insert(p);}
          }
        }
      }
      for (auto pL : PBOrig) {
        if (Ts.count(pL->front())==0) {
          std::list<Process*> *lPLocal = new std::list<Process*>;
          for (auto p: *pL) {
            if (Ts.count(p)==0) { lPLocal->push_back(p); }
          }
          PBRet0.push_back(lPLocal);
        }
      }
    }
  }

  // partitioning
  for (auto pL : PBOrig) {
    if (std::none_of(PBRet0.begin(), PBRet0.end(), [&](std::list<Process*> *pL0){return pL0->front() == pL->front();})) {
      std::list<Process*> *lPLocal = new std::list<Process*>;
      for (auto p: *pL) {
        if (std::none_of(PBRet0.begin(), PBRet0.end(), [&](std::list<Process*> *pL0){return pL0->front() == p;})) {
          lPLocal->push_back(p);
        }
      }
      PBRet1.push_back(lPLocal);
    }
  }
}


unsigned HostDataDeps::minCutPhase(partitionBlock &PB, edgeWeight &curEdgeWeightMap, std::pair<Process *, Process *> &ST) {
  // Stoer-Wagner Minimum Cut Phase contract
  Process *a = PB.front()->front();
  // max adjacency search
  std::vector<Process*> A;
  A.push_back(a);
  while(A.size() != PB.size()) {
    unsigned wAdjMax = 0;
    Process *pAdjMax;
    for (auto pL : PB) {
      if (std::find(A.begin(), A.end(), pL->front())==A.end()) {
        unsigned wTmp = 0;
        for (auto we : curEdgeWeightMap) {
          if ((we.first.first == pL->front()) && (std::find(A.begin(), A.end(), we.first.second)!=A.end())) {
            wTmp += we.second;
          } else if ((we.first.second == pL->front()) && (std::find(A.begin(), A.end(), we.first.first)!=A.end())) {
            wTmp += we.second;
          }
        }
        if (wTmp > wAdjMax) {
          wAdjMax = wTmp;
          pAdjMax = pL->front();
        }
      }
    }
    A.push_back(pAdjMax);
  }

  // cutting and weighting t
  Process *t = A.back();
  A.pop_back();
  Process *s = A.back();
  ST = std::make_pair(s, t);
  unsigned wCut = 0;
  for (auto we : curEdgeWeightMap) {
    if ((we.first.first == t) || (we.first.second == t)) {
      wCut += we.second;
    }
  }

  // updating PB by merging s and t
  bool runST = false;
  std::list<Process*> *pSTList = nullptr;
  std::list<Process*> *pSTListOld = nullptr;
  for (auto pL : PB) {
    if (pL->front() == t) {
      pL->remove_if([&](Process* p){return p == s;});
      if (!runST) {
        pL->front() = s;
        runST = true;
        pSTList = pL;
      } else {
        pL->remove_if([&](Process* p){return p == t;});
        pSTList->insert(pSTList->end(), pL->begin(), pL->end());
        pSTListOld = pL;
      }
    } else if (pL->front() == s) {
      pL->remove_if([&](Process* p){return p == t;});
      if (!runST) {
        runST = true;
        pSTList = pL;
      } else {
        pL->remove_if([&](Process* p){return p == s;});
        pSTList->insert(pSTList->end(), pL->begin(), pL->end());
        pSTListOld = pL;
      }
    } else {
      std::replace(pL->begin(), pL->end(), t, s);
      pL->unique();
    }
  }
  PB.erase(std::remove_if(PB.begin(), PB.end(), [&](std::list<Process*> *pL0){return pSTListOld == pL0;}), PB.end());

  // updating t edges in curEdgeWeightMap
  for (auto we : curEdgeWeightMap) {
    if ((we.first.first == t) && (we.first.second == s)) {
      curEdgeWeightMap.erase(we.first);
    } else if ((we.first.first == s) && (we.first.second == t)) {
      curEdgeWeightMap.erase(we.first);
    } else if ((we.first.first == t) && (we.first.second != s)) {
      auto it = curEdgeWeightMap.find(std::make_pair(s, we.first.second));
      if ( it != curEdgeWeightMap.end()) {
        (*it).second += we.second;
      } else {
        auto value = we.second;
        auto key = std::make_pair(s, we.first.second);
        curEdgeWeightMap[key] = value;
      }
      curEdgeWeightMap.erase(we.first);
    } else if ((we.first.first != s) && (we.first.second == t)) {
      auto it = curEdgeWeightMap.find(std::make_pair(we.first.first, s));
      if ( it != curEdgeWeightMap.end()) {
        (*it).second += we.second;
      } else {
        auto value = we.second;
        auto key = std::make_pair(we.first.first, s);
        curEdgeWeightMap[key] = value;
      }
      curEdgeWeightMap.erase(we.first);
    }
  }
  return wCut;
}


bool HostDataDeps::isLegal(const partitionBlock &PB) {
  if (PB.size() == 1) { return true; }
  // external dependency detection
  bool isLegalDependency = true;
  bool isLegalResource = true;
  unsigned numDepOut=0;
  unsigned numDepIn=0;
  std::vector<Space*> vecSrcSpaces;

  for (auto pL : PB) {
    Process *p = pL->front();
    for (auto dp : p->getOutSpace()->getDstProcesses()) {
      if (std::count_if(PB.begin(), PB.end(),
            [&](std::list<Process*> *pLL){return pLL->front() == dp;}) == 0) {
        numDepOut++;
      }
    }
    for (auto ds : p->getInSpaces()) {
      if (ds->getSrcProcess() && std::count_if(PB.begin(), PB.end(),
            [&](std::list<Process*> *pLL){return pLL->front() == ds->getSrcProcess();}) == 0) {
        numDepIn++;
        break;
      }
    }

    // global src and dest kernel
    if (isDest(p)) {
      numDepOut++;
    } else if (isSrc(p)) {
      numDepIn++;
      if (vecSrcSpaces.empty()) {
        vecSrcSpaces = p->getInSpaces();
      } else if (vecSrcSpaces != p->getInSpaces()) {
        isLegalDependency = false;
      }
    }
  }
  if (numDepOut > 1 || numDepIn > 1) { isLegalDependency = false; }

  // resource constraints
  unsigned YSizeAcc = 1;
  unsigned YSizeAccMax = 1;
  unsigned numLocalKernel = 0;
  std::set<Process*> setVisitedPro;
  for (auto pL : PB) {
    Process *p = pL->front();
    if (setVisitedPro.count(p) == 0 && p->getKernel()->getKernelClass()->getKernelType() == LocalOperator) {
      auto acc = p->getKernel()->getAccessors().back();
      YSizeAcc = YSizeAcc + acc->getSizeY() - 1;
      YSizeAccMax = std::max(YSizeAccMax, acc->getSizeY());
      setVisitedPro.insert(p);
      numLocalKernel++;
    }
  }
  if (numLocalKernel >= 2) {
    isLegalResource = (((static_cast<float>(YSizeAcc) / YSizeAccMax) <= CMS) &&
                       ((static_cast<float>(YSizeAcc) / YSizeAccMax) > CMSf)) ? true : false;
  }
  return (isLegalResource && isLegalDependency) ? true : false;
}


void HostDataDeps::computeGraphWeight() {
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
        w = ((TG / TS) > costComp) ? TG / TS - costComp : EPSILON;
      } else {                                              // unsupported scenario
        w = EPSILON;
      }
      unsigned we = std::max(w + GAMMA, EPSILON);
      edgeWeightMap_[std::make_pair(srcProcess, destProcess)] = we;
    }
  }
}


std::vector<HostDataDeps::Space*> HostDataDeps::getInputSpaces() {
  std::vector<Space*> ret;
  for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
    if ((*it)->getSrcProcess() == nullptr) { ret.push_back(*it); }
  }
  return ret;
}


bool HostDataDeps::isSrc(Process *P) {
  std::vector<Space*> S = P->getInSpaces();
  return std::all_of(S.begin(), S.end(), [](Space *s){return s->getSrcProcess() == nullptr;});
}

bool HostDataDeps::isDest(Process *P) {
  Space *s = P->getOutSpace();
  return s->getDstProcesses().empty();
}

std::set<HostDataDeps::partitionBlockNames> HostDataDeps::getFusibleSetNames() const {
  return fusibleSetNames;
}

bool HostDataDeps::isFusible(HipaccKernel *K) {
  bool isFusible = false;
  std::string fullName = K->getKernelClass()->getName() + K->getName();

  // Kernel name has no corresponding process or no execute() is called
  if (!processMap_.count(fullName)) {
    return isFusible;
  }
  // get Kernel Partition Block
  for (auto PBN : fusibleSetNames) {
    if (std::any_of(PBN.begin(), PBN.end(), [&](std::list<std::string> lNam){
      return (std::find(lNam.begin(), lNam.end(), fullName) != lNam.end()) &&
        (lNam.size() > 1);})) {
      isFusible = true;
      break;
    }
  }
  return isFusible;
}

bool HostDataDeps::hasSharedIS(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(processMap_.count(fullName), "Kernel name has no corresponding process");
  std::vector<Space*> spaces = processMap_[fullName]->getInSpaces();
  return std::any_of(spaces.begin(), spaces.end(), [](Space *s){return s->isSpaceShared();});
}

std::string HostDataDeps::getSharedISName(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName();
  hipacc_require(processMap_.count(fullName), "Kernel name has no corresponding process");
  std::vector<Space*> spaces = processMap_[fullName]->getInSpaces();
  auto it = std::find_if(spaces.begin(), spaces.end(), [](Space *s){return s->isSpaceShared();});
  return (*it)->getImage()->getName();
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

