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

//===--- CompilerKnownClasses.h - List of compiler-known C++ classes ------===//
//
// This provides pointers to all compiler-known C++ classes.
//
//===----------------------------------------------------------------------===//

#ifndef _COMPILER_KNOWN_CLASSES_H_
#define _COMPILER_KNOWN_CLASSES_H_

#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>

namespace clang {
namespace hipacc {
class CompilerKnownClasses {
  public:
    CXXRecordDecl *Coordinate;
    CXXRecordDecl *Image;
    CXXRecordDecl *BoundaryCondition;
    CXXRecordDecl *AccessorBase;
    CXXRecordDecl *Accessor;
    CXXRecordDecl *IterationSpaceBase;
    CXXRecordDecl *IterationSpace;
    CXXRecordDecl *ElementIterator;
    CXXRecordDecl *Kernel;
    CXXRecordDecl *Mask;
    CXXRecordDecl *Domain;
    CXXRecordDecl *Pyramid;
    // End of Parsing
    CXXRecordDecl *HipaccEoP;

    CompilerKnownClasses() :
      Coordinate(nullptr),
      Image(nullptr),
      BoundaryCondition(nullptr),
      AccessorBase(nullptr),
      Accessor(nullptr),
      IterationSpaceBase(nullptr),
      IterationSpace(nullptr),
      ElementIterator(nullptr),
      Kernel(nullptr),
      Mask(nullptr),
      Domain(nullptr),
      Pyramid(nullptr),
      HipaccEoP(nullptr)
    {}

    bool isTypeOfClass(QualType QT, CXXRecordDecl *CRD) {
      if (QT->isReferenceType()) {
        QT = QT->getPointeeType();
      }

      if (QT == CRD->getTypeForDecl()->getCanonicalTypeInternal()) {
        return true;
      }

      return false;
    }

    bool isTypeOfTemplateClass(QualType QT, CXXRecordDecl *CRD) {
      if (QT->isReferenceType()) {
        QT = QT->getPointeeType();
      }

      // see also UnwrapTypeForDebugInfo() in CGDebugInfo.cpp
      if (QT->getTypeClass() == Type::Elaborated) {
        QT = cast<ElaboratedType>(QT)->getNamedType();
      }

      // class<type> ...
      if (QT->getTypeClass() == Type::TemplateSpecialization) {
        auto TST = dyn_cast<TemplateSpecializationType>(QT);

        ClassTemplateDecl *reference = CRD->getDescribedClassTemplate();
        ClassTemplateDecl *current;

        TemplateDecl *TD = TST->getTemplateName().getAsTemplateDecl();
        if (isa<ClassTemplateDecl>(TD)) {
          current = dyn_cast<ClassTemplateDecl>(TD);
        } else {
          return false;
        }

        do {
          if (reference == current) return true;
          current = current->getPreviousDecl();
        } while (current);
      }

      return false;
    }

    QualType getFirstTemplateType(QualType QT) {
      if (QT->isReferenceType()) {
        QT = QT->getPointeeType();
      }

      if (QT->getTypeClass() == Type::Elaborated) {
        QT = cast<ElaboratedType>(QT)->getNamedType();
      }

      // class<type> ...
      assert(QT->getTypeClass() == Type::TemplateSpecialization &&
          "instance of template class expected");
      auto TST = dyn_cast<TemplateSpecializationType>(QT);

      return TST->getArg(0).getAsType();
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _COMPILER_KNOWN_CLASSES_H_

// vim: set ts=2 sw=2 sts=2 et ai:

