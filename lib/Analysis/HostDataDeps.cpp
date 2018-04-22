//
// Copyright (c) 2018, University of Erlangen-Nuremberg
// Copyright (c) 2014, Saarland University
// Copyright (c) 2014, University of Erlangen-Nuremberg
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

        // check if the first argument is an Image
        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the Image from the DRE if we have one
          if (imgDeclMap_.count(DRE->getDecl())) {
            Img = imgDeclMap_[DRE->getDecl()];
            BC = new HipaccBoundaryCondition(VD, Img);

            dataDeps.addBoundaryCondition(VD, BC, DRE->getDecl());
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

        // check if the first argument is an Image
        DeclRefExpr *DRE = nullptr;

        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the BoundaryCondition from the DRE if we have one
          if (bcDeclMap_.count(DRE->getDecl())) {
            if (DEBUG) std::cout << "    -> Based on BoundaryCondition: "
                    << DRE->getNameInfo().getAsString() << std::endl;

            BC = bcDeclMap_[DRE->getDecl()];
          }

          // in case we have no BoundaryCondition, check if an Image is
          // specified and construct a BoundaryCondition
          if (!BC && imgDeclMap_.count(DRE->getDecl())) {
            if (DEBUG) std::cout << "    -> Based on Image: "
                    << DRE->getNameInfo().getAsString() << std::endl;

            Img = imgDeclMap_[DRE->getDecl()];
            BC = new HipaccBoundaryCondition(VD, Img);

            bcDeclMap_[VD] = BC;
          }
        }

        // TODO: Read from arguments
        Interpolate mode = Interpolate::NO;

        Acc = new HipaccAccessor(VD, BC, mode, false);

        // store Accessor definition
        accDeclMap_[VD] = Acc;

        assert(DRE != nullptr && "First Accessor argument is not a BC or Image");
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

      // TODO, not yet used
      // found Mask decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Mask)) {
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
  assert(!kernelMap_.count(KVD) && "Duplicate Kernel declaration");

  kernel = new Kernel(
      KVD->getType()->getAsCXXRecordDecl()->getNameAsString()
        .append(KVD->getNameAsString()),
          iterMap_[ISVD], KVD);

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

  assert(!processMap_.count(kernel->getName()) && "Duplicate process declaration, kernel name exists");
  processMap_[kernel->getName()] = proc;

  assert(!spaceMap_.count(kernel->getIterationSpace()->getImage()->getName()) && "Duplicate space declaration, image name exists");
  spaceMap_[kernel->getIterationSpace()->getImage()->getName()] = space;

  // Set process to destination for all predecessor spaces:
  std::vector<Accessor*> accs = kernel->getAccessors();
  for (auto it = accs.begin(); it != accs.end(); ++it) {
    Space *s = nullptr;// = (*it)->getSpace();
    // Lookup if space was written by previous process
    //for (auto it2 = processes_.rbegin();
    //          it2 != processes_.rend(); ++it2) {
    //  IterationSpace *iter = (*it2)->getKernel()->getIterationSpace();
    //  if (iter != nullptr && (*it)->getImage() == iter->getImage()) {
    //    s = (*it2)->getOutSpace();
    //    break;
    //  }
    //}
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

  // Mark that IterationSpace was most recently written by this process
  //IterationSpace *iter = kernel->getIterationSpace();

  // Update all accessors that are bound to the image currently written
  //for (auto it = accs.begin(); it != accs.end(); ++it) {
  //  if ((*it)->getImage() == iter->getImage()) {
  //    (*it)->setSpace(space);
  //  }
  //}
}


void HostDataDeps::dump(Process *proc) {
  std::cout << " <- " << proc->getKernel()->getName();

  std::vector<Space*> spaces = proc->getInSpaces();
  for (auto it = spaces.begin(); it != spaces.end(); ++it) {
    dump(*it);
    if (it+1 != spaces.end()) {
      std::cout << std::endl << "            ";
      for (size_t i = 0; i < proc->getKernel()->getName().size(); ++i) {
        std::cout << " ";
      }
      std::cout << "|";
    }
  }
}


void HostDataDeps::dump(Space *space) {
  if (space->getAccessors().size() > 0) {
    std::cout << " <- ";
  }
  std::cout << space->getImage()->getName();

  Process *proc = space->getSrcProcess();
  if (proc != nullptr) {
    dump(proc);
  }
}


void HostDataDeps::dump() {
  std::cout << "  Spaces:" << std::endl;
  for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
    IterationSpace *iter = (*it)->getIterationSpace();
    std::cout << "    - img: " << (*it)->getImage()->getName()
              << ", iter: " << (iter ? iter->getName() : "nullptr")
              << ", isShared: " << (*it)->isShared 
              << std::endl;
  }

  std::cout << "  Processes:" << std::endl;
  for (auto it = processes_.begin(); it != processes_.end(); ++it) {
    std::cout << "    - kernel: " << (*it)->getKernel()->getName()
              << ", out: " << (*it)->getOutSpace()->getImage()->getName()
              << ", read dependent on: " << ((*it)->GetReadDependentProcess() ? 
                  ((*it)->GetReadDependentProcess())->getKernel()->getName() : "nullptr")
              << ", write dependent on: " << ((*it)->GetWriteDependentProcess() ? 
                  ((*it)->GetWriteDependentProcess())->getKernel()->getName() : "nullptr")
              << std::endl;
  }

  std::cout << "  Dependency Tree:" << std::endl;
  int id = 1;
  std::vector<Image*> done;
  for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
    if (!findVector(done, (*it)->getImage()) &&
        (*it)->getAccessors().size() == 0) {
      std::cout << "    " << id << ":";
      dump(*it);
      std::cout << std::endl;
      ++id;
      done.push_back((*it)->getImage());
    }
  }
}


std::vector<HostDataDeps::Space*> HostDataDeps::getInputSpaces() {
  std::vector<Space*> ret;
  for (auto it = spaces_.begin(); it != spaces_.end(); ++it) {
    if ((*it)->getSrcProcess() == nullptr) {
      ret.push_back(*it);
    }
  }
  return ret;
}


std::vector<HostDataDeps::Space*> HostDataDeps::getOutputSpaces() {
  std::vector<Space*> ret;
  for (auto it = spaces_.rbegin(); it != spaces_.rend(); ++it) {
    if ((*it)->getDstProcesses().empty()) {
      ret.push_back(*it);
    }
  }
  return ret;
}


void HostDataDeps::markProcess(Process *t) {
  std::vector<Space*> spaces = t->getInSpaces();
  for (auto it = spaces.begin(); it != spaces.end(); ++it) {
    Space *s = *it;
    // all successors of p are planned
    if (s->getDstProcesses().size() == 1)   markSpace(s);
  }
}


void HostDataDeps::markSpace(Space *s) {
  std::vector<Process*> DstProcesses = s->getDstProcesses();
  Process *SrcProcess = s->getSrcProcess();
  if (SrcProcess != nullptr) {
    // set dependencies for linear dependency kernels
//    if (  DstProcesses.size() == 1 && 
//          ((DstProcesses.back())->getInSpaces().size() == 1) &&
//          !((DstProcesses.back())->getKernel()->getKernelType() == LocalOperator && // TODO
//          SrcProcess->getKernel()->getKernelType() == LocalOperator) ) {    // disable for L2L
    if (  DstProcesses.size() == 1 && 
          ((DstProcesses.back())->getInSpaces().size() == 1) ) {
      // mark the space as shared
      s->isShared = true;
      // record process dependencies in both directions
      SrcProcess->SetWriteDependentProcess(DstProcesses.back()); 
      DstProcesses.back()->SetReadDependentProcess(SrcProcess);
    }
    markProcess(SrcProcess);
  }
}


void HostDataDeps::createSchedule() {
  std::vector<Space*> outSpaces = getOutputSpaces();

  outId = tmpId = 0;
  for (auto it = outSpaces.begin(); it != outSpaces.end(); ++it) {
    markSpace(*it);
  }
}


//std::string HostDataDeps::prettyPrint(
//    std::map<std::string,std::vector<std::pair<std::string,std::string>>> args,
//    bool print) {
//  std::ostringstream retVal;
//  std::string indent = "";
//
//  retVal << indent << getEntrySignature(args, true) << " {" << std::endl;
//  retVal << "#pragma HLS dataflow" << std::endl;
//
//  indent = "  ";
//
//  //int cpyId = 0;
//  for (auto it = schedule.rbegin(); it != schedule.rend(); ++it) {
//    if ((*it)->isSpace()) {
//      Space *s = (Space*)*it;
//      if (s->cpyStreams.size() > 0) {
//        for (auto it2 = s->cpyStreams.begin();
//                  it2 != s->cpyStreams.end(); ++it2) {
//          retVal << indent << declareFifo(getTypeStr(s), *it2);
//        }
//#define NICO_LIB
//#ifdef NICO_LIB
//        retVal << indent << "splitStream";
//        if (compilerOptions.getPixelsPerThread() > 1) {
//          retVal << "VECT";
//        }
//        retVal << "<HIPACC_II_TARGET,HIPACC_MAX_WIDTH,HIPACC_MAX_HEIGHT,HIPACC_WINDOW_SIZE_X,HIPACC_WINDOW_SIZE_Y";
//        if (compilerOptions.getPixelsPerThread() > 1) {
//          retVal << ",HIPACC_PPT";
//        }
//        retVal << ">(" << s->stream;
//        for (auto it2 = s->cpyStreams.begin();
//                  it2 != s->cpyStreams.end(); ++it2) {
//          retVal << ", " << *it2;
//        }
//        retVal << ", HIPACC_MAX_WIDTH, HIPACC_MAX_HEIGHT);" << std::endl;
//#else // NICO_LIB
//        retVal << indent << "for (int i = 0; i < HIPACC_MAX_WIDTH*HIPACC_MAX_HEIGHT; ++i) {"
//               << std::endl;
//        retVal << indent << indent << getTypeStr(s) << " val;"
//               << std::endl;
//        retVal << indent << indent << s->stream << " >> val;" << std::endl;
//        for (auto it2 = s->cpyStreams.begin();
//                  it2 != s->cpyStreams.end(); ++it2) {
//          retVal << indent << indent << *it2 << " << val;" << std::endl;
//        }
//        retVal << indent << "}"
//               << std::endl;
//#endif // NICO_LIB
//
//        //std::ostringstream var;
//        //var << "_strmCpy" << cpyId;
//        //++cpyId;
//
//        //retVal << indent << getTypeStr(s) << " " << var.str() << ";"
//        //       << std::endl;
//        //retVal << indent << s->stream << " >> " << var.str() << ";"
//        //       << std::endl;
//        //for (auto it2 = s->cpyStreams.begin();
//        //          it2 != s->cpyStreams.end(); ++it2) {
//        //  retVal << indent << "stream<" << getTypeStr(s) << "> " << *it2 << ";"
//        //         << std::endl;
//        //  retVal << indent << *it2 << " << " << var.str() << ";"
//        //         << std::endl;
//        //}
//      }
//    } else {
//      Process *t = (Process*)*it;
//      if (!t->getOutSpace()->getDstProcesses().empty()) {
//        // do not print out stream (because it is function argument)
//        retVal << indent << declareFifo(getTypeStr(t->getOutSpace()), t->outStream);
//      }
//      retVal << indent << "cc" << t->getKernel()->getName() << "Kernel(";
//      retVal << t->outStream;
//      for (auto it2 = t->inStreams.begin();
//                it2 != t->inStreams.end(); ++it2) {
//        retVal << ", " << *it2;
//      }
//      if (args.find("cc" + t->getKernel()->getName() + "Kernel") != args.end()) {
//        std::vector<std::pair<std::string,std::string>> a =
//            args["cc" + t->getKernel()->getName() + "Kernel"];
//        for (auto it2 = a.begin(); it2 != a.end(); ++it2) {
//          retVal << ", " << it2->second;
//        }
//      }
//      retVal << ", HIPACC_MAX_WIDTH, HIPACC_MAX_HEIGHT);" << std::endl;
//    }
//  }
//
//  indent = "";
//  retVal << indent << "}" << std::endl;
//
//  if (print) {
//    std::cout << retVal.str() << std::endl;
//  }
//
//  return retVal.str();
//}


//std::string HostDataDeps::printEntryDecl(
//    std::map<std::string,std::vector<std::pair<std::string,std::string>>> args) {
//  return getEntrySignature(args, true) + ";\n";
//}


//std::string HostDataDeps::printEntryCall(
//    std::map<std::string,std::vector<std::pair<std::string,std::string>>> args,
//    std::string img) {
//  return getEntrySignature(args) + ";\n";
//}

//std::string HostDataDeps::printEntryDef(
//    std::map<std::string,std::vector<std::pair<std::string,std::string>>> args) {
//  return prettyPrint(args);
//}


ValueDecl *HostDataDeps::getSourceKernelValueDecl(HipaccKernel *K) {
  std::string fullName = K->getKernelClass()->getName() + K->getName(); 
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = (processMap_[fullName])->GetReadDependentProcess();
  return p ? p->getKernel()->getValueDecl() : nullptr;
}

bool HostDataDeps::isSourceKernel(HipaccKernel *K) { 
  bool isSrc = true;
  std::string fullName = K->getKernelClass()->getName() + K->getName(); 
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];

  if (p->GetReadDependentProcess()) { 
    isSrc = false;
  }
  return isSrc;
}

bool HostDataDeps::isDestinationKernel(HipaccKernel *K) { 
  bool isDest = true;
  std::string fullName = K->getKernelClass()->getName() + K->getName(); 
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];

  if (p->GetWriteDependentProcess()) { 
    isDest = false;
  }
  return isDest;
}

bool HostDataDeps::isFusibleKernel(HipaccKernel *K) { 
  bool isFusible = false;
  std::string fullName = K->getKernelClass()->getName() + K->getName(); 
  assert(processMap_.count(fullName) && "Kernel name has no corresponding process");
  Process *p = processMap_[fullName];

  if ((p->GetReadDependentProcess()) || (p->GetWriteDependentProcess())) {
    isFusible = true;
  }
  return isFusible;
}

bool HostDataDeps::isSharedSpace(ValueDecl *VD) {
  std::string fullName = VD->getNameAsString();
  assert(spaceMap_.count(fullName) && "image name has no corresponding space");
  Space *s = spaceMap_[fullName];
  bool isShared = s->isShared;
  return isShared;
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





