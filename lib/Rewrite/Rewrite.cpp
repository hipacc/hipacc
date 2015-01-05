//
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

//===--- Rewrite.cpp - Mapping the DSL (AST nodes) to the runtime ---------===//
//
// This file implements functionality for mapping the DSL to the HIPAcc runtime.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Rewrite/Rewrite.h"
#include "hipacc/Analysis/KernelStatistics.h"
#ifdef USE_POLLY
#include "hipacc/Analysis/Polly.h"
#endif
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Config/config.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/DSL/CompilerKnownClasses.h"
#include "hipacc/Rewrite/CreateHostStrings.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

using namespace clang;
using namespace hipacc;
using namespace ASTNode;


namespace {
class Rewrite : public ASTConsumer,  public RecursiveASTVisitor<Rewrite> {
  private:
    // Clang internals
    CompilerInstance &CI;
    ASTContext &Context;
    DiagnosticsEngine &Diags;
    SourceManager &SM;
    llvm::raw_ostream &Out;
    Rewriter TextRewriter;
    Rewriter::RewriteOptions TextRewriteOptions;

    // HIPACC instances
    CompilerOptions &compilerOptions;
    HipaccDevice targetDevice;
    hipacc::Builtin::Context builtins;
    CreateHostStrings stringCreator;

    // compiler known/built-in C++ classes
    CompilerKnownClasses compilerClasses;

    // mapping between AST nodes and internal class representation
    llvm::DenseMap<RecordDecl *, HipaccKernelClass *> KernelClassDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccAccessor *> AccDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccBoundaryCondition *> BCDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccImage *> ImgDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccPyramid *> PyrDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccIterationSpace *> ISDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccKernel *> KernelDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccMask *> MaskDeclMap;

    // store interpolation methods required for CUDA
    SmallVector<std::string, 16> InterpolationDefinitionsGlobal;

    // pointer to main function
    FunctionDecl *mainFD;
    FileID mainFileID;
    unsigned literalCount;
    bool skipTransfer;

  public:
    Rewrite(CompilerInstance &CI, CompilerOptions &options, llvm::raw_ostream*
        o=nullptr) :
      CI(CI),
      Context(CI.getASTContext()),
      Diags(CI.getASTContext().getDiagnostics()),
      SM(CI.getASTContext().getSourceManager()),
      Out(o? *o : llvm::outs()),
      compilerOptions(options),
      targetDevice(options),
      builtins(CI.getASTContext()),
      stringCreator(CreateHostStrings(options, targetDevice)),
      compilerClasses(CompilerKnownClasses()),
      mainFD(nullptr),
      literalCount(0),
      skipTransfer(false)
    {}

    void HandleTranslationUnit(ASTContext &Context);
    bool HandleTopLevelDecl(DeclGroupRef D);

    bool VisitCXXRecordDecl(CXXRecordDecl *D);
    bool VisitDeclStmt(DeclStmt *D);
    bool VisitFunctionDecl(FunctionDecl *D);
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    bool VisitCallExpr (CallExpr *E);

    //bool shouldVisitTemplateInstantiations() const { return true; }

  private:
    void Initialize(ASTContext &Context) {
      // get the ID and start/end of the main file.
      mainFileID = SM.getMainFileID();
      TextRewriter.setSourceMgr(SM, Context.getLangOpts());
      TextRewriteOptions.RemoveLineIfEmpty = true;
    }

    void setKernelConfiguration(HipaccKernelClass *KC, HipaccKernel *K);
    void printReductionFunction(HipaccKernelClass *KC, HipaccKernel *K,
        PrintingPolicy Policy, llvm::raw_ostream *OS);
    void printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
        HipaccKernel *K, std::string file, bool emitHints);
};
}


ASTConsumer *HipaccRewriteAction::CreateASTConsumer(CompilerInstance &CI,
    StringRef file) {
  if (llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, file)) {
    return new Rewrite(CI, options, OS);
  }

  return nullptr;
}


void Rewrite::HandleTranslationUnit(ASTContext &Context) {
  assert(compilerClasses.Coordinate && "Coordinate class not found!");
  assert(compilerClasses.Image && "Image class not found!");
  assert(compilerClasses.BoundaryCondition && "BoundaryCondition class not found!");
  assert(compilerClasses.AccessorBase && "AccessorBase class not found!");
  assert(compilerClasses.Accessor && "Accessor class not found!");
  assert(compilerClasses.IterationSpaceBase && "IterationSpaceBase class not found!");
  assert(compilerClasses.IterationSpace && "IterationSpace class not found!");
  assert(compilerClasses.ElementIterator && "ElementIterator class not found!");
  assert(compilerClasses.Kernel && "Kernel class not found!");
  assert(compilerClasses.Mask && "Mask class not found!");
  assert(compilerClasses.Domain && "Domain class not found!");
  assert(compilerClasses.Pyramid && "Pyramid class not found!");
  assert(compilerClasses.HipaccEoP && "HipaccEoP class not found!");

  StringRef MainBuf = SM.getBufferData(mainFileID);
  const char *mainFileStart = MainBuf.begin();
  const char *mainFileEnd = MainBuf.end();
  SourceLocation locStart = SM.getLocForStartOfFile(mainFileID);

  size_t includeLen = strlen("include");
  size_t hipaccHdrLen = strlen("hipacc.hpp");
  size_t usingLen = strlen("using");
  size_t namespaceLen = strlen("namespace");
  size_t hipaccLen = strlen("hipacc");

  // loop over the whole file, looking for includes
  for (const char *bufPtr = mainFileStart; bufPtr < mainFileEnd; ++bufPtr) {
    if (*bufPtr == '#') {
      const char *startPtr = bufPtr;
      if (++bufPtr == mainFileEnd)
        break;
      while (*bufPtr == ' ' || *bufPtr == '\t')
        if (++bufPtr == mainFileEnd)
          break;
      if (!strncmp(bufPtr, "include", includeLen)) {
        const char *endPtr = bufPtr + includeLen;
        while (*endPtr == ' ' || *endPtr == '\t')
          if (++endPtr == mainFileEnd)
            break;
        if (*endPtr == '"') {
          if (!strncmp(endPtr+1, "hipacc.hpp", hipaccHdrLen)) {
            endPtr = strchr(endPtr+1, '"');
            // remove hipacc include
            SourceLocation includeLoc =
              locStart.getLocWithOffset(startPtr-mainFileStart);
            TextRewriter.RemoveText(includeLoc, endPtr-startPtr+1,
                TextRewriteOptions);
            bufPtr += endPtr-startPtr;
          }
        }
      }
    }
    if (*bufPtr == 'u') {
      const char *startPtr = bufPtr;
      if (!strncmp(bufPtr, "using", usingLen)) {
        const char *endPtr = bufPtr + usingLen;
        while (*endPtr == ' ' || *endPtr == '\t')
          if (++endPtr == mainFileEnd)
            break;
        if (*endPtr == 'n') {
          if (!strncmp(endPtr, "namespace", namespaceLen)) {
            endPtr += namespaceLen;
            while (*endPtr == ' ' || *endPtr == '\t')
              if (++endPtr == mainFileEnd)
                break;
            if (*endPtr == 'h') {
              if (!strncmp(endPtr, "hipacc", hipaccLen)) {
                endPtr = strchr(endPtr+1, ';');
                // remove using namespace line
                SourceLocation includeLoc =
                  locStart.getLocWithOffset(startPtr-mainFileStart);
                TextRewriter.RemoveText(includeLoc, endPtr-startPtr+1,
                    TextRewriteOptions);
                bufPtr += endPtr-startPtr;
              }
            }
          }
        }
      }
    }
  }


  // add include files for CUDA
  std::string newStr;

  // get include header string, including a header twice is fine
  stringCreator.writeHeaders(newStr);

  // add interpolation include and define interpolation functions for CUDA
  if (compilerOptions.emitCUDA() && InterpolationDefinitionsGlobal.size()) {
    newStr += "#include \"hipacc_cu_interpolate.hpp\"\n";

    // sort definitions and remove duplicate definitions
    std::sort(InterpolationDefinitionsGlobal.begin(),
        InterpolationDefinitionsGlobal.end());
    InterpolationDefinitionsGlobal.erase(
        std::unique(InterpolationDefinitionsGlobal.begin(),
          InterpolationDefinitionsGlobal.end()),
        InterpolationDefinitionsGlobal.end());

    // add interpolation definitions
    for (auto str : InterpolationDefinitionsGlobal)
      newStr += str;
    newStr += "\n";
  }

  // include .cu or .h files for normal kernels
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::C99:
      for (auto map : KernelDeclMap) {
        newStr += "#include \"";
        newStr += map.second->getFileName();
        newStr += ".cc\"\n";
      }
      break;
    case Language::CUDA:
      if (!compilerOptions.exploreConfig()) {
        for (auto map : KernelDeclMap) {
          newStr += "#include \"";
          newStr += map.second->getFileName();
          newStr += ".cu\"\n";
        }
      }
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      for (auto map : KernelDeclMap) {
        newStr += "#include \"ScriptC_";
        newStr += map.second->getFileName();
        newStr += ".h\"\n";
      }
      break;
  }


  // write constant memory declarations
  if (compilerOptions.emitCUDA()) {
    for (auto map : MaskDeclMap) {
      auto mask = map.second;
      if (mask->isPrinted()) continue;

      size_t i = 0;
      for (auto kernel : mask->getKernels()) {
        if (i++) newStr += "\n" + stringCreator.getIndent();

        newStr += "__device__ __constant__ ";
        newStr += mask->getTypeStr();
        newStr += " " + mask->getName() + kernel->getName();
        newStr += "[" + mask->getSizeYStr() + "][" + mask->getSizeXStr() +
          "];\n";
      }
    }
  }
  // rewrite header section
  TextRewriter.InsertTextBefore(locStart, newStr);


  // initialize CUDA/OpenCL
  assert(mainFD && "no main found!");

  CompoundStmt *CS = dyn_cast<CompoundStmt>(mainFD->getBody());
  assert(CS->size() && "CompoundStmt has no statements.");

  std::string initStr;

  // get initialization string for run-time
  stringCreator.writeInitialization(initStr);

  // load OpenCL kernel files and compile the OpenCL kernels
  if (!compilerOptions.exploreConfig()) {
    for (auto map : KernelDeclMap)
      stringCreator.writeKernelCompilation(map.second, initStr);
    initStr += "\n" + stringCreator.getIndent();
  }

  // write Mask transfers to Symbol in CUDA
  if (compilerOptions.emitCUDA()) {
    for (auto map : MaskDeclMap) {
      auto mask = map.second;

      if (!compilerOptions.exploreConfig()) {
        std::string newStr;
        if (mask->hasCopyMask()) {
          stringCreator.writeMemoryTransferDomainFromMask(mask,
              mask->getCopyMask(), newStr);
        } else {
          stringCreator.writeMemoryTransferSymbol(mask, mask->getHostMemName(),
              HOST_TO_DEVICE, newStr);
        }

        TextRewriter.InsertTextBefore(mask->getDecl()->getLocStart(), newStr);
      }
    }
  }

  // insert initialization before first statement
  auto BI = CS->body_begin();
  Stmt *S = *BI;
  TextRewriter.InsertTextBefore(S->getLocStart(), initStr);

  // insert memory release calls before last statement (return-statement)
  auto RBI = CS->body_rbegin();
  S = *RBI;
  // release all images
  for (auto map : ImgDeclMap) {
    auto img = map.second;
    std::string releaseStr;

    stringCreator.writeMemoryRelease(img, releaseStr);
    TextRewriter.InsertTextBefore(S->getLocStart(), releaseStr);
  }
  // release all non-const masks
  for (auto map : MaskDeclMap) {
    auto mask = map.second;
    std::string releaseStr;

    if (!compilerOptions.emitCUDA() && !mask->isConstant()) {
      stringCreator.writeMemoryRelease(mask, releaseStr);
      TextRewriter.InsertTextBefore(S->getLocStart(), releaseStr);
    }
  }
  // release all pyramids
  for (auto map : PyrDeclMap) {
    auto pyramid = map.second;
    std::string releaseStr;

    stringCreator.writeMemoryRelease(pyramid, releaseStr, true);
    TextRewriter.InsertTextBefore(S->getLocStart(), releaseStr);
  }

  // get buffer of main file id. If we haven't changed it, then we are done.
  if (auto RewriteBuf = TextRewriter.getRewriteBufferFor(mainFileID)) {
    Out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    llvm::errs() << "No changes to input file, something went wrong!\n";
  }
  Out.flush();
}


bool Rewrite::HandleTopLevelDecl(DeclGroupRef DGR) {
  for (auto decl : DGR) {
    if (compilerClasses.HipaccEoP) {
      // skip late template class instantiations when templated class instances
      // are created. this is the case if the expansion location is not within
      // the main file
      if (SM.getFileID(SM.getExpansionLoc(decl->getLocation()))!=mainFileID)
        continue;
    }
    TraverseDecl(decl);
  }

  return true;
}


bool Rewrite::VisitCXXRecordDecl(CXXRecordDecl *D) {
  // return if this is no Class definition
  if (!D->hasDefinition()) return true;

  // a) look for compiler known classes and remember them
  // b) look for user defined kernel classes derived from those stored in
  //    step a). If such a class is found:
  //    - create a mapping between kernel class constructor variables and
  //      kernel parameters and store that mapping.
  //    - analyze image memory access patterns for later usage.

  if (D->getTagKind() == TTK_Class && D->isCompleteDefinition()) {
    DeclContext *DC = D->getEnclosingNamespaceContext();
    if (DC->isNamespace()) {
      NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC);
      if (NS->getNameAsString() == "hipacc") {
        if (D->getNameAsString() == "Coordinate")
          compilerClasses.Coordinate = D;
        if (D->getNameAsString() == "Image") compilerClasses.Image = D;
        if (D->getNameAsString() == "BoundaryCondition")
          compilerClasses.BoundaryCondition = D;
        if (D->getNameAsString() == "AccessorBase")
          compilerClasses.AccessorBase = D;
        if (D->getNameAsString() == "Accessor") compilerClasses.Accessor = D;
        if (D->getNameAsString() == "IterationSpaceBase")
          compilerClasses.IterationSpaceBase = D;
        if (D->getNameAsString() == "IterationSpace")
          compilerClasses.IterationSpace = D;
        if (D->getNameAsString() == "ElementIterator")
          compilerClasses.ElementIterator = D;
        if (D->getNameAsString() == "Kernel") compilerClasses.Kernel = D;
        if (D->getNameAsString() == "Mask") compilerClasses.Mask = D;
        if (D->getNameAsString() == "Domain") compilerClasses.Domain = D;
        if (D->getNameAsString() == "Pyramid") compilerClasses.Pyramid = D;
        if (D->getNameAsString() == "HipaccEoP") compilerClasses.HipaccEoP = D;
      }
    }

    if (!compilerClasses.HipaccEoP) return true;

    HipaccKernelClass *KC = nullptr;

    for (auto base : D->bases()) {
      // found user kernel class
      if (compilerClasses.isTypeOfTemplateClass(base.getType(),
            compilerClasses.Kernel)) {
        KC = new HipaccKernelClass(D->getNameAsString());
        KernelClassDeclMap[D] = KC;
        // remove user kernel class (semicolon doesn't count to SourceRange)
        SourceLocation startLoc = D->getLocStart();
        SourceLocation endLoc = D->getLocEnd();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *endBuf = SM.getCharacterData(endLoc);
        const char *semiPtr = strchr(endBuf, ';');
        TextRewriter.RemoveText(startLoc, semiPtr-startBuf+1, TextRewriteOptions);

        break;
      }
    }

    if (!KC) return true;

    // find constructor
    CXXConstructorDecl *CCD = nullptr;
    for (auto ctor : D->ctors()) {
      if (ctor->isCopyOrMoveConstructor()) continue;
      CCD = ctor;
    }
    assert(CCD && "Couldn't find user kernel class constructor!");


    // iterate over constructor initializers
    for (auto param : CCD->params()) {
      // constructor initializer represent the parameters for the kernel. Match
      // constructor parameter with constructor initializer since the order may
      // differ, e.g.
      // kernel(int a, int b) : b(a), a(b) {}
      for (auto init : CCD->inits()) {
        QualType QT;

        // init->isMemberInitializer()
        if (auto DRE =
            dyn_cast<DeclRefExpr>(init->getInit()->IgnoreParenCasts())) {
          if (DRE->getDecl() == param) {
            FieldDecl *FD = init->getMember();

            // reference to Image variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Image)) {
              QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addImgArg(FD, QT, FD->getName());
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_width");
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_height");
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_stride");

              break;
            }

            // reference to Accessor variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Accessor)) {
              QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addImgArg(FD, QT, FD->getName());
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_width");
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_height");
              //KC->addArg(nullptr, Context.IntTy, FD->getNameAsString() + "_stride");

              break;
            }

            // reference to Mask variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Mask)) {
              QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addMaskArg(FD, QT, FD->getName());

              break;
            }

            // reference to Domain variable ?
            if (compilerClasses.isTypeOfClass(FD->getType(),
                                              compilerClasses.Domain)) {
              QT = Context.UnsignedCharTy;
              KC->addMaskArg(FD, QT, FD->getName());

              break;
            }

            // normal variable
            KC->addArg(FD, FD->getType(), FD->getName());

            break;
          }
        }

        // init->isBaseInitializer()
        if (auto CCE = dyn_cast<CXXConstructExpr>(init->getInit())) {
          assert(CCE->getNumArgs() == 1 &&
              "Kernel base class constructor requires exactly one argument!");

          if (auto DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0))) {
            if (DRE->getDecl() == param) {
              // create FieldDecl for the IterationSpace so it can be handled
              // like all other members
              QT = compilerClasses.getFirstTemplateType(param->getType());
              FieldDecl *FD = FieldDecl::Create(Context, D->getDeclContext(),
                  SourceLocation(), SourceLocation(),
                  &Context.Idents.get(param->getName()), QT,
                  Context.getTrivialTypeSourceInfo(QT), nullptr, false,
                  ICIS_NoInit);
              KC->addISArg(FD, QT, FD->getName());;
              //KC->addArg(nullptr, Context.IntTy, "is_width");
              //KC->addArg(nullptr, Context.IntTy, "is_height");
              //KC->addArg(nullptr, Context.IntTy, "is_stride");

              break;
            }
          }
        }
      }
    }

    // search for kernel and reduce functions
    for (auto method : D->methods()) {
      // kernel function
      if (method->getNameAsString() == "kernel") {
        // set kernel method
        KC->setKernelFunction(method);

        // define analysis context used for different checkers
        AnalysisDeclContext AC(/* AnalysisDeclContextManager */ 0, method);
        KernelStatistics::setAnalysisOptions(AC);

        // create kernel analysis pass, execute it and store it to kernel class
        KernelStatistics *stats = KernelStatistics::create(AC, D->getName(),
            compilerClasses);
        KC->setKernelStatistics(stats);

        continue;
      }

      // reduce function
      if (method->getNameAsString() == "reduce") {
        // set reduce method
        KC->setReduceFunction(method);

        continue;
      }
    }
  }

  return true;
}


bool Rewrite::VisitDeclStmt(DeclStmt *D) {
  if (!compilerClasses.HipaccEoP) return true;

  // a) convert Image declarations into memory allocations, e.g.
  //    Image<int> IN(width, height, data);
  //    =>
  //    HipaccImage IN = hipaccCreateMemory<int>(data, width, height, &stride, padding);
  // b) convert Pyramid declarations into pyramid creation, e.g.
  //    Pyramid<int> P(IN, 3);
  //    =>
  //    Pyramid P = hipaccCreatePyramid<int>(IN, 3);
  // c) save BoundaryCondition declarations, e.g.
  //    BoundaryCondition<int> BcIN(IN, 5, 5, Boundary::MIRROR);
  // d) save Accessor declarations, e.g.
  //    Accessor<int> AccIN(BcIN);
  // e) save Mask declarations, e.g.
  //    Mask<float> M(stencil);
  // f) save Domain declarations, e.g.
  //    Domain D(3, 3)
  //    Domain D(dom)
  //    Domain D(M)
  // g) save user kernel declarations, and replace it by kernel compilation
  //    for OpenCL, e.g.
  //    AddKernel K(IS, IN, OUT, 23);
  //    - create CUDA/OpenCL kernel AST by replacing accesses to Image data by
  //      global memory access and by replacing references to class member
  //      variables by kernel parameter variables.
  //    - print the CUDA/OpenCL kernel to a file.
  // h) save IterationSpace declarations, e.g.
  //    IterationSpace<int> VIS(OUT, width, height);
  for (auto decl : D->decls()) {
    if (decl->getKind() == Decl::Var) {
      VarDecl *VD = dyn_cast<VarDecl>(decl);

      // found Image decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Image)) {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert((CCE->getNumArgs() == 2 || CCE->getNumArgs() == 3) &&
               "Image definition requires two or three arguments!");

        HipaccImage *Img = new HipaccImage(Context, VD,
            compilerClasses.getFirstTemplateType(VD->getType()));

        // get the text string for the image width and height
        std::string width_str  = TextRewriter.ConvertToString(CCE->getArg(0));
        std::string height_str = TextRewriter.ConvertToString(CCE->getArg(1));

        if (compilerOptions.emitC99()) {
          // check if the parameter can be resolved to a constant
          unsigned IDConstant = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "Constant expression for %0 argument of Image %1 required (C/C++ only).");
          if (!CCE->getArg(0)->isEvaluatable(Context)) {
            Diags.Report(CCE->getArg(0)->getExprLoc(), IDConstant) << "width"
              << Img->getName();
          }
          if (!CCE->getArg(1)->isEvaluatable(Context)) {
            Diags.Report(CCE->getArg(1)->getExprLoc(), IDConstant) << "height"
              << Img->getName();
          }
          Img->setSizeX(CCE->getArg(0)->EvaluateKnownConstInt(Context).getSExtValue());
          Img->setSizeY(CCE->getArg(1)->EvaluateKnownConstInt(Context).getSExtValue());
        }

        // host memory
        std::string init_str = "NULL";
        if (CCE->getNumArgs() == 3) {
          init_str = TextRewriter.ConvertToString(CCE->getArg(2));
        }

        // create memory allocation string
        std::string newStr;
        stringCreator.writeMemoryAllocation(Img, width_str, height_str,
            init_str, newStr);

        // rewrite Image definition
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        // store Image definition
        ImgDeclMap[VD] = Img;

        break;
      }

      // found Pyramid decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Pyramid)) {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert(CCE->getNumArgs() == 2 &&
               "Pyramid definition requires exactly two arguments!");

        HipaccPyramid *Pyr = new HipaccPyramid(Context, VD,
            compilerClasses.getFirstTemplateType(VD->getType()));

        // get the text string for the pyramid image & depth
        std::string image_str = TextRewriter.ConvertToString(CCE->getArg(0));
        std::string depth_str = TextRewriter.ConvertToString(CCE->getArg(1));

        // create memory allocation string
        std::string newStr;
        stringCreator.writePyramidAllocation(VD->getName(),
            compilerClasses.getFirstTemplateType(VD->getType()).getAsString(),
            image_str, depth_str, newStr);

        // rewrite Pyramid definition
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        // store Pyramid definition
        PyrDeclMap[VD] = Pyr;

        break;
      }

      // found BoundaryCondition decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.BoundaryCondition)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected BoundaryCondition definition (CXXConstructExpr).");
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        unsigned IDConstMode = Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Constant value for BoundaryCondition %0 required.");
        unsigned IDConstSize = Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Constant expression for size argument of BoundaryCondition %1 required.");
        unsigned IDConstPyrIdx = Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Missing integer literal in Pyramid %0 call expression.");
        unsigned IDMode = Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Boundary handling constant for BoundaryCondition %0 required.");
        HipaccBoundaryCondition *BC = nullptr;
        HipaccImage *Img = nullptr;
        HipaccPyramid *Pyr = nullptr;
        size_t size_args = 0;

        for (size_t i=0, e=CCE->getNumArgs(); i!=e; ++i) {
          // img|pyramid-call, size_x, size_y, mode
          // img|pyramid-call, size, mode
          // img|pyramid-call, mask, mode
          // img|pyramid-call, size_x, size_y, mode, const_val
          // img|pyramid-call, size, mode, const_val
          // img|pyramid-call, mask, mode, const_val
          auto arg = CCE->getArg(i)->IgnoreParenCasts();

          auto dsl_arg = arg;
          if (auto call = dyn_cast<CXXOperatorCallExpr>(arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument specifies the image
            if (ImgDeclMap.count(DRE->getDecl())) {
              Img = ImgDeclMap[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Img);
              BCDeclMap[VD] = BC;
              continue;
            }

            // check if the argument is a pyramid call
            if (PyrDeclMap.count(DRE->getDecl())) {
              Pyr = PyrDeclMap[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Pyr);
              BCDeclMap[VD] = BC;

              // add call expression to pyramid argument
              auto call = dyn_cast<CXXOperatorCallExpr>(arg);
              auto index = call->getArg(1);
              if (!index->isEvaluatable(Context)) {
                Diags.Report(index->getExprLoc(), IDConstPyrIdx)
                  << Pyr->getName();
              }
              BC->setPyramidIndex(
                  index->EvaluateKnownConstInt(Context).toString(10));
              continue;
            }

            // check if the argument is a Mask
            if (MaskDeclMap.count(DRE->getDecl())) {
              HipaccMask *Mask = MaskDeclMap[DRE->getDecl()];
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
                  if (i+2 != e) {
                    Diags.Report(arg->getExprLoc(), IDMode) << VD->getName();
                  }
                  // check if the parameter can be resolved to a constant
                  auto const_arg = CCE->getArg(++i);
                  if (!const_arg->isEvaluatable(Context)) {
                    Diags.Report(arg->getExprLoc(), IDConstMode) <<
                      VD->getName();
                  } else {
                    Expr::EvalResult val;
                    const_arg->EvaluateAsRValue(val, Context);
                    BC->setConstVal(val.Val, Context);
                  }
              }
              continue;
            }

            // check if the argument can be resolved to a constant
            if (!arg->isEvaluatable(Context)) {
              Diags.Report(arg->getExprLoc(), IDConstSize) << VD->getName();
            }
            if (size_args++ == 0) {
              BC->setSizeX(arg->EvaluateKnownConstInt(Context).getSExtValue());
              BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
            } else {
              BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
            }
          }
        }

        assert((Img || Pyr) && "Expected first argument of BoundaryCondition "
                               "to be Image or Pyramid call.");


        // remove BoundaryCondition definition
        TextRewriter.RemoveText(D->getSourceRange());

        break;
      }

      // found Accessor decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Accessor)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected Accessor definition (CXXConstructExpr).");
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        HipaccAccessor *Acc = nullptr;
        HipaccBoundaryCondition *BC = nullptr;
        HipaccPyramid *Pyr = nullptr;
        Interpolate mode = Interpolate::NO;
        std::string Parms;
        size_t roi_args = 0;

        for (size_t i=0, e=CCE->getNumArgs(); i!=e; ++i) {
          auto arg = CCE->getArg(i)->IgnoreParenCasts();

          if (isa<CXXDefaultArgExpr>(arg))
            continue;

          auto dsl_arg = arg;
          if (auto call = dyn_cast<CXXOperatorCallExpr>(arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument specifies the boundary condition
            if (BCDeclMap.count(DRE->getDecl())) {
              BC = BCDeclMap[DRE->getDecl()];

              Parms = BC->getImage()->getName();
              if (BC->isPyramid()) {
                // add call expression to pyramid argument
                Parms += "(" + BC->getPyramidIndex() + ")";
              }
              continue;
            }

            // check if the argument specifies the image
            if (ImgDeclMap.count(DRE->getDecl())) {
              HipaccImage *Img = ImgDeclMap[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Img);
              BC->setSizeX(1);
              BC->setSizeY(1);
              BC->setBoundaryMode(Boundary::CLAMP);
              BCDeclMap[VD] = BC; // Fixme: store BoundaryCondition???

              Parms = BC->getImage()->getName();
              continue;
            }

            // check if the argument specifies is a pyramid call
            if (PyrDeclMap.count(DRE->getDecl())) {
              Pyr = PyrDeclMap[DRE->getDecl()];
              BC = new HipaccBoundaryCondition(VD, Pyr);
              BC->setSizeX(1);
              BC->setSizeY(1);
              BC->setBoundaryMode(Boundary::CLAMP);
              BCDeclMap[VD] = BC; // Fixme: store BoundaryCondition???

              // add call expression to pyramid argument
              Parms = TextRewriter.ConvertToString(arg);
              continue;
            }

            // check if the argument specifies the interpolate mode
            if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
                DRE->getDecl()->getType().getAsString() ==
                "enum hipacc::Interpolate") {
              auto lval = DRE->EvaluateKnownConstInt(Context);
              auto cval = static_cast<std::underlying_type<Interpolate>::type>(Interpolate::L3);
              assert(lval.isNonNegative() && lval.getZExtValue() <= cval &&
                     "invalid Interpolate mode");
              mode = static_cast<Interpolate>(lval.getZExtValue());
              continue;
            }
          }

          // get text string for arguments, argument order is:
          // img|bc|pyramid-call
          // img|bc|pyramid-call, width, height, xf, yf
          Parms += ", " + TextRewriter.ConvertToString(arg);
          roi_args++;
        }

        assert(BC && "Expected BoundaryCondition, Image or Pyramid call as "
                     "first argument to Accessor.");

        Acc = new HipaccAccessor(VD, BC, mode, roi_args == 4);

        std::string newStr;
        newStr = "HipaccAccessor " + Acc->getName() + "(" + Parms + ");";

        // replace Accessor decl by variables for width/height and offsets
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        // store Accessor definition
        AccDeclMap[VD] = Acc;

        break;
      }

      // found IterationSpace decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.IterationSpace)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected IterationSpace definition (CXXConstructExpr).");
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        HipaccIterationSpace *IS = nullptr;
        HipaccImage *Img = nullptr;
        HipaccPyramid *Pyr = nullptr;
        std::string Parms;
        size_t roi_args = 0;

        for (size_t i=0, e=CCE->getNumArgs(); i!=e; ++i) {
          auto arg = CCE->getArg(i)->IgnoreParenCasts();
          auto dsl_arg = arg;
          if (auto call = dyn_cast<CXXOperatorCallExpr>(arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument is an image 
            if (ImgDeclMap.count(DRE->getDecl())) {
              Img = ImgDeclMap[DRE->getDecl()];
              Parms = Img->getName();
              continue;
            }

            // check if the argument is a pyramid call
            if (PyrDeclMap.count(DRE->getDecl())) {
              Pyr = PyrDeclMap[DRE->getDecl()];
              // add call expression to pyramid argument
              Parms = TextRewriter.ConvertToString(arg);
              continue;
            }
          }

          // get text string for arguments, argument order is:
          // img[, is_width, is_height[, offset_x, offset_y]]
          Parms += ", " + TextRewriter.ConvertToString(arg);
          roi_args++;
        }

        assert((Img || Pyr) && "Expected first argument of IterationSpace to "
                               "be Image or Pyramid call.");

        IS = new HipaccIterationSpace(VD, Img ? Img : Pyr, roi_args == 4);
        ISDeclMap[VD] = IS; // store IterationSpace

        std::string newStr;
        newStr = "HipaccAccessor " + IS->getName() + "(" + Parms + ");";

        // replace iteration space decl by variables for width/height, and
        // offset
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        break;
      }

      HipaccMask *Mask = nullptr;
      // found Mask decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Mask)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected Mask definition (CXXConstructExpr).");

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert((CCE->getNumArgs() == 1) &&
               "Mask definition requires exactly one argument!");

        QualType QT = compilerClasses.getFirstTemplateType(VD->getType());
        Mask = new HipaccMask(VD, QT, HipaccMask::MaskType::Mask);

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
      }

      HipaccMask *Domain = nullptr;
      // found Domain decl
      if (compilerClasses.isTypeOfClass(VD->getType(),
                                        compilerClasses.Domain)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected Domain definition (CXXConstructExpr).");

        Domain = new HipaccMask(VD, Context.UnsignedCharTy,
                                            HipaccMask::MaskType::Domain);

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        if (CCE->getNumArgs() == 1) {
          // get initializer
          auto DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0)->IgnoreParenCasts());
          assert(DRE && "Domain must be initialized using a variable");
          VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
          assert(V && "Domain must be initialized using a variable");

          if (compilerClasses.isTypeOfTemplateClass(DRE->getType(),
                                                    compilerClasses.Mask)) {
            // copy from mask
            HipaccMask *Mask = MaskDeclMap[DRE->getDecl()];
            assert(Mask && "Mask to copy from was not declared");

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
                    assert(false && "Only builtin integer and floating point "
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
                      assert(false &&
                             "Expected integer literal in domain initializer");
                    }
                  }
                }
              }
            }
            Domain->setIsConstant(isDomainConstant);
            Domain->setHostMemName(V->getName());
          }
        } else if (CCE->getNumArgs() == 2) {
          unsigned DiagIDConstant =
              Diags.getCustomDiagID(DiagnosticsEngine::Error,
                  "Constant expression for %ordinal0 parameter to %1 %2 "
                  "required.");

          // check if the parameters can be resolved to a constant
          Expr *Arg0 = CCE->getArg(0);
          if (!Arg0->isEvaluatable(Context)) {
            Diags.Report(Arg0->getExprLoc(), DiagIDConstant)
              << 1 << "Domain" << VD->getName();
          }
          Domain->setSizeX(Arg0->EvaluateKnownConstInt(Context).getSExtValue());

          Expr *Arg1 = CCE->getArg(1);
          if (!Arg1->isEvaluatable(Context)) {
            Diags.Report(Arg1->getExprLoc(), DiagIDConstant)
              << 2 << "Domain" << VD->getName();
          }
          Domain->setSizeY(Arg1->EvaluateKnownConstInt(Context).getSExtValue());
          Domain->setIsConstant(true);
        } else {
          assert(false && "Domain definition requires exactly two arguments "
              "type constant integer or a single argument of type uchar[][] or "
              "Mask!");
        }
      }

      if (Mask || Domain) {
        HipaccMask *Buf = Domain ? Domain : Mask;

        std::string newStr;
        if (!Buf->isConstant() && !compilerOptions.emitCUDA()) {
          // create Buffer for Mask
          stringCreator.writeMemoryAllocationConstant(Buf, newStr);

          if (Buf->hasCopyMask()) {
            // create Domain from Mask and upload to Buffer
            stringCreator.writeMemoryTransferDomainFromMask(Buf,
                Buf->getCopyMask(), newStr);
          } else {
            // upload Mask to Buffer
            stringCreator.writeMemoryTransferSymbol(Buf, Buf->getHostMemName(),
                HOST_TO_DEVICE, newStr);
          }
        }

        // replace Mask declaration by Buffer allocation
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        // store Mask definition
        MaskDeclMap[VD] = Buf;

        break;
      }

      // found Kernel decl
      if (VD->getType()->getTypeClass() == Type::Record) {
        const RecordType *RT = cast<RecordType>(VD->getType());

        // get Kernel Class
        if (KernelClassDeclMap.count(RT->getDecl())) {
          HipaccKernelClass *KC = KernelClassDeclMap[RT->getDecl()];
          HipaccKernel *K = new HipaccKernel(Context, VD, KC, compilerOptions);
          KernelDeclMap[VD] = K;

          // remove kernel declaration
          TextRewriter.RemoveText(D->getSourceRange());

          // create map between Image or Accessor instances and kernel
          // variables; replace image instances by accessors with undefined
          // boundary handling
          assert(isa<CXXConstructExpr>(VD->getInit()) &&
               "Expected Image definition (CXXConstructExpr).");
          CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

          size_t num_img = 0, num_mask = 0;
          auto imgFields = KC->getImgFields();
          auto maskFields = KC->getMaskFields();
          for (size_t i=0; i<CCE->getNumArgs(); ++i) {
            auto arg = CCE->getArg(i);
            if (auto DRE = dyn_cast<DeclRefExpr>(arg->IgnoreParenCasts())) {
              // check if we have an Image
              if (ImgDeclMap.count(DRE->getDecl())) {
                unsigned DiagIDImage =
                  Diags.getCustomDiagID(DiagnosticsEngine::Error,
                      "Images are not supported within kernels, use Accessors instead:");
                Diags.Report(DRE->getLocation(), DiagIDImage);
              }

              // check if we have an IterationSpace
              if (ISDeclMap.count(DRE->getDecl())) {
                K->insertMapping(imgFields[num_img++],
                    ISDeclMap[DRE->getDecl()]);
                continue;
              }

              // check if we have an Accessor
              if (AccDeclMap.count(DRE->getDecl())) {
                K->insertMapping(imgFields[num_img++],
                    AccDeclMap[DRE->getDecl()]);
                continue;
              }

              // check if we have a Mask or Domain
              if (MaskDeclMap.count(DRE->getDecl())) {
                K->insertMapping(maskFields[num_mask++],
                    MaskDeclMap[DRE->getDecl()]);
                continue;
              }
            }
          }

          // set kernel configuration
          setKernelConfiguration(KC, K);

          // kernel declaration
          FunctionDecl *kernelDecl = createFunctionDecl(Context,
              Context.getTranslationUnitDecl(), K->getKernelName(),
              Context.VoidTy, K->getArgTypes(), K->getDeviceArgNames());

          // write CUDA/OpenCL kernel function to file clone old body,
          // replacing member variables
          ASTTranslate *Hipacc = new ASTTranslate(Context, kernelDecl, K, KC,
              builtins, compilerOptions);
          Stmt *kernelStmts =
            Hipacc->Hipacc(KC->getKernelFunction()->getBody());
          kernelDecl->setBody(kernelStmts);
          K->printStats();

          #ifdef USE_POLLY
          if (!compilerOptions.exploreConfig() && compilerOptions.emitC99()) {
            llvm::errs() << "\nPassing the following function to Polly:\n";
            kernelDecl->print(llvm::errs(), Context.getPrintingPolicy());
            llvm::errs() << "\n";

            Polly *polly_analysis = new Polly(Context, CI, kernelDecl);
            polly_analysis->analyzeKernel();
          }
          #endif

          // write kernel to file
          printKernelFunction(kernelDecl, KC, K, K->getFileName(), true);

          break;
        }
      }
    }
  }

  return true;
}


bool Rewrite::VisitFunctionDecl(FunctionDecl *D) {
  if (D->isMain()) {
    assert(D->getBody() && "main function has no body.");
    assert(isa<CompoundStmt>(D->getBody()) && "CompoundStmt for main body expected.");
    mainFD = D;
  }

  return true;
}


bool Rewrite::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  if (!compilerClasses.HipaccEoP) return true;

  // convert overloaded operator 'operator=' function into memory transfer,
  // a) Img = host_array;
  // b) Pyr(x) = host_array;
  // c) Img = Img;
  // d) Img = Acc;
  // e) Img = Pyr(x);
  // f) Acc = Acc;
  // g) Acc = Img;
  // h) Acc = Pyr(x);
  // i) Pyr(x) = Img;
  // j) Pyr(x) = Acc;
  // k) Pyr(x) = Pyr(x);
  // l) Domain(x, y) = literal; (return type of ()-operator is DomainSetter)
  if (E->getOperator() == OO_Equal) {
    if (E->getNumArgs() != 2) return true;

    HipaccImage *ImgLHS = nullptr, *ImgRHS = nullptr;
    HipaccAccessor *AccLHS = nullptr, *AccRHS = nullptr;
    HipaccPyramid *PyrLHS = nullptr, *PyrRHS = nullptr;
    HipaccMask *DomLHS = nullptr;
    std::string PyrIdxLHS, PyrIdxRHS;
    unsigned DomIdxX, DomIdxY;

    // check first parameter
    if (auto DRE = dyn_cast<DeclRefExpr>(E->getArg(0)->IgnoreParenCasts())) {
      // check if we have an Image at the LHS
      if (ImgDeclMap.count(DRE->getDecl())) {
        ImgLHS = ImgDeclMap[DRE->getDecl()];
      }
      // check if we have an Accessor at the LHS
      if (AccDeclMap.count(DRE->getDecl())) {
        AccLHS = AccDeclMap[DRE->getDecl()];
      }
    } else if (auto call = dyn_cast<CXXOperatorCallExpr>(E->getArg(0))) {
      // check if we have an Pyramid or Domain call at the LHS
      if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
        // get the Pyramid from the DRE if we have one
        if (PyrDeclMap.count(DRE->getDecl())) {
          PyrLHS = PyrDeclMap[DRE->getDecl()];

          // add call expression to pyramid argument
          unsigned DiagIDConstant =
            Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "Missing integer literal in Pyramid %0 call expression.");
          if (!call->getArg(1)->isEvaluatable(Context)) {
            Diags.Report(call->getArg(1)->getExprLoc(), DiagIDConstant)
              << PyrLHS->getName();
          }
          PyrIdxLHS =
            call->getArg(1)->EvaluateKnownConstInt(Context).toString(10);
        } else if (MaskDeclMap.count(DRE->getDecl())) {
          DomLHS = MaskDeclMap[DRE->getDecl()];

          assert(DomLHS->isConstant() &&
                 "Setting domain values only supported for constant Domains");

          unsigned DiagIDConstant =
            Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "Integer expression in Domain %0 is non-const.");
          if (!call->getArg(1)->isEvaluatable(Context)) {
            Diags.Report(call->getArg(1)->getExprLoc(), DiagIDConstant)
              << DomLHS->getName();
          }
          if (!call->getArg(2)->isEvaluatable(Context)) {
            Diags.Report(call->getArg(2)->getExprLoc(), DiagIDConstant)
              << DomLHS->getName();
          }
          DomIdxX = DomLHS->getSizeX()/2 +
            call->getArg(1)->EvaluateKnownConstInt(Context).getSExtValue();
          DomIdxY = DomLHS->getSizeY()/2 +
            call->getArg(2)->EvaluateKnownConstInt(Context).getSExtValue();
        }
      }
    }

    // check second parameter
    if (auto DRE = dyn_cast<DeclRefExpr>(E->getArg(1)->IgnoreParenCasts())) {
      // check if we have an Image at the RHS
      if (ImgDeclMap.count(DRE->getDecl())) {
        ImgRHS = ImgDeclMap[DRE->getDecl()];
      }
      // check if we have an Accessor at the RHS
      if (AccDeclMap.count(DRE->getDecl())) {
        AccRHS = AccDeclMap[DRE->getDecl()];
      }
    } else if (auto call = dyn_cast<CXXOperatorCallExpr>(E->getArg(1))) {
      // check if we have an Pyramid call at the RHS
      if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
        // get the Pyramid from the DRE if we have one
        if (PyrDeclMap.count(DRE->getDecl())) {
          PyrRHS = PyrDeclMap[DRE->getDecl()];

          // add call expression to pyramid argument
          unsigned DiagIDConstant =
            Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "Missing integer literal in Pyramid %0 call expression.");
          if (!call->getArg(1)->isEvaluatable(Context)) {
            Diags.Report(call->getArg(1)->getExprLoc(), DiagIDConstant)
              << PyrRHS->getName();
          }
          PyrIdxRHS =
            call->getArg(1)->EvaluateKnownConstInt(Context).toString(10);
        }
      }
    } else if (DomLHS) {
      // check for RHS literal to set domain value
      Expr *arg = E->getArg(1)->IgnoreParenCasts();

      assert(isa<IntegerLiteral>(arg) &&
             "RHS argument for setting specific domain value must be integer "
             "literal");

      // set domain value
      DomLHS->setDomainDefined(DomIdxX, DomIdxY,
          dyn_cast<IntegerLiteral>(arg)->getValue() != 0);

      SourceLocation startLoc = E->getLocStart();
      const char *startBuf = SM.getCharacterData(startLoc);
      const char *semiPtr = strchr(startBuf, ';');
      TextRewriter.RemoveText(startLoc, semiPtr-startBuf+1);

      return true;
    }

    if (ImgLHS || AccLHS || PyrLHS) {
      std::string newStr;

      if (ImgLHS && ImgRHS) {
        // Img1 = Img2;
        stringCreator.writeMemoryTransfer(ImgLHS, ImgRHS->getName(),
            DEVICE_TO_DEVICE, newStr);
      } else if (ImgLHS && AccRHS) {
        // Img1 = Acc2;
        stringCreator.writeMemoryTransferRegion("HipaccAccessor(" +
            ImgLHS->getName() + ")", AccRHS->getName(), newStr);
      } else if (ImgLHS && PyrRHS) {
        // Img1 = Pyr2(x2);
        stringCreator.writeMemoryTransfer(ImgLHS,
            PyrRHS->getName() + "(" + PyrIdxRHS + ")",
            DEVICE_TO_DEVICE, newStr);
      } else if (AccLHS && ImgRHS) {
        // Acc1 = Img2;
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            "HipaccAccessor(" + ImgRHS->getName() + ")", newStr);
      } else if (AccLHS && AccRHS) {
        // Acc1 = Acc2;
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            AccRHS->getName(), newStr);
      } else if (AccLHS && PyrRHS) {
        // Acc1 = Pyr2(x2);
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            "HipaccAccessor(" + PyrRHS->getName() + "(" + PyrIdxRHS + "))",
            newStr);
      } else if (PyrLHS && ImgRHS) {
        // Pyr1(x1) = Img2
        stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS, ImgRHS->getName(),
            DEVICE_TO_DEVICE, newStr);
      } else if (PyrLHS && AccRHS) {
        // Pyr1(x1) = Acc2
        stringCreator.writeMemoryTransferRegion(
            "HipaccAccessor(" + PyrLHS->getName() + "(" + PyrIdxLHS + "))",
            AccRHS->getName(), newStr);
      } else if (PyrLHS && PyrRHS) {
        // Pyr1(x1) = Pyr2(x2)
        stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS,
            PyrRHS->getName() + "(" + PyrIdxRHS + ")",
            DEVICE_TO_DEVICE, newStr);
      } else {
        bool write_pointer = true;
        // Img1 = Img2.data();
        // Img1 = Pyr2(x2).data();
        // Pyr1(x1) = Img2.data();
        // Pyr1(x1) = Pyr2(x2).data();
        if (auto mcall = dyn_cast<CXXMemberCallExpr>(E->getArg(1))) {
          // match only data() calls to Image instances
          if (mcall->getDirectCallee()->getNameAsString() == "data") {
            // side effect ! do not handle the next call to data()
            skipTransfer = true;
            if (auto DRE =
                dyn_cast<DeclRefExpr>(mcall->getImplicitObjectArgument())) {
              // check if we have an Image
              if (ImgDeclMap.count(DRE->getDecl())) {
                HipaccImage *Img = ImgDeclMap[DRE->getDecl()];

                if (PyrLHS) {
                  stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS,
                      Img->getName(), DEVICE_TO_DEVICE, newStr);
                } else {
                  stringCreator.writeMemoryTransfer(ImgLHS, Img->getName(),
                      DEVICE_TO_DEVICE, newStr);
                }
                write_pointer = false;
              }
            } else if (auto call = dyn_cast<CXXOperatorCallExpr>(
                                   mcall->getImplicitObjectArgument())) {
              // check if we have an Pyramid call
              if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
                // get the Pyramid from the DRE if we have one
                if (PyrDeclMap.count(DRE->getDecl())) {
                  HipaccPyramid *Pyr = PyrDeclMap[DRE->getDecl()];

                  // add call expression to pyramid argument
                  unsigned DiagIDConstant =
                    Diags.getCustomDiagID(DiagnosticsEngine::Error,
                        "Missing integer literal in Pyramid %0 call expression.");
                  if (!call->getArg(1)->isEvaluatable(Context)) {
                    Diags.Report(call->getArg(1)->getExprLoc(), DiagIDConstant)
                      << Pyr->getName();
                  }
                  std::string index =
                    call->getArg(1)->EvaluateKnownConstInt(Context).toString(10);

                  if (PyrLHS) {
                    stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS,
                        Pyr->getName() + "(" + index + ")", DEVICE_TO_DEVICE,
                        newStr);
                  } else {
                    stringCreator.writeMemoryTransfer(ImgLHS, Pyr->getName() +
                        "(" + index + ")", DEVICE_TO_DEVICE, newStr);
                  }
                  write_pointer = false;
                }
              }
            }
          }
        }

        if (write_pointer) {
          // get the text string for the memory transfer src
          std::string data_str = TextRewriter.ConvertToString(E->getArg(1));

          // create memory transfer string
          if (PyrLHS) {
            stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS, data_str,
                HOST_TO_DEVICE, newStr);
          } else {
            stringCreator.writeMemoryTransfer(ImgLHS, data_str, HOST_TO_DEVICE,
                newStr);
          }
        }
      }

      // rewrite Image assignment to memory transfer
      // get the start location and compute the semi location.
      SourceLocation startLoc = E->getLocStart();
      const char *startBuf = SM.getCharacterData(startLoc);
      const char *semiPtr = strchr(startBuf, ';');
      TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

      return true;
    }
  }

  return true;
}


bool Rewrite::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  if (!compilerClasses.HipaccEoP) return true;

  // a) convert invocation of 'execute' member function into kernel launch, e.g.
  //    K.execute()
  //    therefore, we need the declaration of K in order to get the parameters
  //    and the IterationSpace for the CUDA/OpenCL kernel, e.g.
  //    AddKernel K(IS, IN, OUT, 23);
  //    IS -> kernel launch configuration
  //    IN, OUT, 23 -> kernel parameters
  //    Image width, height, and stride -> kernel parameters
  // b) convert data() calls
  //    float *out = img.data();
  // c) convert reduced_data() calls
  //    float min = MinReduction.reduced_data();
  // d) convert width()/height() calls

  if (auto DRE =
      dyn_cast<DeclRefExpr>(E->getImplicitObjectArgument()->IgnoreParenCasts())) {
    // match execute calls to user kernel instances
    if (!KernelDeclMap.empty() &&
        E->getDirectCallee()->getNameAsString() == "execute") {
      // get the user Kernel class
      if (KernelDeclMap.count(DRE->getDecl())) {
        HipaccKernel *K = KernelDeclMap[DRE->getDecl()];
        VarDecl *VD = K->getDecl();
        std::string newStr;

        // this was checked before, when the user class was parsed
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert(CCE->getNumArgs()==K->getKernelClass()->getMembers().size() &&
            "number of arguments doesn't match!");

        // set host argument names and retrieve literals stored to temporaries
        K->setHostArgNames(llvm::makeArrayRef(CCE->getArgs(),
              CCE->getNumArgs()), newStr, literalCount);

        //
        // TODO: handle the case when only reduce function is specified
        //
        // create kernel call string
        stringCreator.writeKernelCall(K->getKernelName(), K->getKernelClass(),
            K, newStr);

        // create reduce call string
        if (K->getKernelClass()->getReduceFunction()) {
          newStr += "\n" + stringCreator.getIndent();
          stringCreator.writeReductionDeclaration(K, newStr);
          stringCreator.writeReduceCall(K->getKernelClass(), K, newStr);
        }

        // rewrite kernel invocation
        // get the start location and compute the semi location.
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
      }
    }
  }

  // data() & width()/height() MemberExpr calls
  if (auto ME = dyn_cast<MemberExpr>(E->getCallee())) {
    if (auto DRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreParenCasts())) {
      std::string newStr;

      // get the Kernel from the DRE if we have one
      if (KernelDeclMap.count(DRE->getDecl())) {
        // match for supported member calls
        if (ME->getMemberNameInfo().getAsString() == "reduced_data") {
          HipaccKernel *K = KernelDeclMap[DRE->getDecl()];

          // replace member function invocation
          SourceRange range(E->getLocStart(), E->getLocEnd());
          TextRewriter.ReplaceText(range, K->getReduceStr());

          return true;
        }
      }

      // get the Image from the DRE if we have one
      if (ImgDeclMap.count(DRE->getDecl())) {
        // match for supported member calls
        if (ME->getMemberNameInfo().getAsString() == "data") {
          if (skipTransfer) {
            skipTransfer = false;
            return true;
          }
          HipaccImage *Img = ImgDeclMap[DRE->getDecl()];
          // create memory transfer string
          stringCreator.writeMemoryTransfer(Img, "NULL", DEVICE_TO_HOST,
              newStr);
          // rewrite Image assignment to memory transfer
          // get the start location and compute the semi location.
          SourceLocation startLoc = E->getLocStart();
          const char *startBuf = SM.getCharacterData(startLoc);
          const char *semiPtr = strchr(startBuf, ';');
          TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

          return true;
        } else if (ME->getMemberNameInfo().getAsString() == "width") {
          newStr = "width";
        } else if (ME->getMemberNameInfo().getAsString() == "height") {
          newStr = "height";
        }
      }

      // get the Accessor from the DRE if we have one
      if (AccDeclMap.count(DRE->getDecl())) {
        // match for supported member calls
        if (ME->getMemberNameInfo().getAsString() == "width") {
          newStr = "img.width";
        } else if (ME->getMemberNameInfo().getAsString() == "height") {
          newStr = "img.height";
        }
      }

      if (!newStr.empty()) {
        // replace member function invocation
        SourceRange range(ME->getMemberLoc(), E->getLocEnd());
        TextRewriter.ReplaceText(range, newStr);
      }
    }
  }

  return true;
}


bool Rewrite::VisitCallExpr (CallExpr *E) {
  // rewrite function calls 'traverse' to 'hipaccTraverse'
  if (auto ICE = dyn_cast<ImplicitCastExpr>(E->getCallee())) {
    if (auto DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
      if (DRE->getDecl()->getNameAsString() == "traverse") {
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM.getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, '(');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf, "hipaccTraverse");
      }
    }
  }
  return true;
}


void Rewrite::setKernelConfiguration(HipaccKernelClass *KC, HipaccKernel *K) {
  #ifdef USE_JIT_ESTIMATE
  bool jit_compile = false;
  switch (compilerOptions.getTargetLang()) {
    default:
      jit_compile = false;
      break;
    case Language::CUDA:
    case Language::OpenCLGPU:
      if (targetDevice.isARMGPU()) {
        jit_compile = false;
      } else {
        jit_compile = true;
      }
      break;
  }

  if (!jit_compile) {
    K->setDefaultConfig();
    return;
  }

  // write kernel file to estimate resource usage
  // kernel declaration for CUDA
  FunctionDecl *kernelDeclEst = createFunctionDecl(Context,
      Context.getTranslationUnitDecl(), K->getKernelName(), Context.VoidTy,
      K->getArgTypes(), K->getDeviceArgNames());

  // create kernel body
  ASTTranslate *HipaccEst = new ASTTranslate(Context, kernelDeclEst, K, KC,
      builtins, compilerOptions, true);
  Stmt *kernelStmtsEst = HipaccEst->Hipacc(KC->getKernelFunction()->getBody());
  kernelDeclEst->setBody(kernelStmtsEst);

  // write kernel to file
  printKernelFunction(kernelDeclEst, KC, K, K->getFileName(), false);

  // compile kernel in order to get resource usage
  std::string command(K->getCompileCommand(compilerOptions.emitCUDA()) +
      K->getCompileOptions(K->getKernelName(), K->getFileName(),
        compilerOptions.emitCUDA()));

  int reg=0, lmem=0, smem=0, cmem=0;
  char line[FILENAME_MAX];
  SmallVector<std::string, 16> lines;
  FILE *fpipe;

  if (!(fpipe = (FILE *)popen(command.c_str(), "r"))) {
    perror("Problems with pipe");
    exit(EXIT_FAILURE);
  }

  std::string info;
  if (compilerOptions.emitCUDA()) {
    info = "ptxas info : Used %d registers";
  } else {
    if (targetDevice.isAMDGPU()) {
      info = "isa info : Used %d gprs, %d bytes lds";
    } else {
      info = "ptxas info : Used %d registers";
    }
  }

  while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
    lines.push_back(std::string(line));
    if (targetDevice.isAMDGPU()) {
      sscanf(line, info.c_str(), &reg, &smem);
    } else {
      char *ptr = line;
      int num_read = 0, val1 = 0, val2 = 0;
      char mem_type = 'x';

      if (compilerOptions.getTargetDevice() >= Device::Fermi_20) {
        // scan for stack size (shared memory)
        num_read = sscanf(ptr, "%d bytes %1c tack frame", &val1, &mem_type);

        if (num_read == 2 && mem_type == 's') {
          smem = val1;
          continue;
        }
      }

      num_read = sscanf(line, info.c_str(), &reg);
      if (!num_read) continue;

      while ((ptr = strchr(ptr, ','))) {
        ptr++;

        num_read = sscanf(ptr, "%d+%d bytes %1c mem", &val1, &val2, &mem_type);
        if (num_read == 3) {
          switch (mem_type) {
            default:
              llvm::errs() << "wrong memory specifier '" << mem_type
                           << "': " << ptr;
              break;
            case 'c':
              cmem += val1 + val2;
              break;
            case 'l':
              lmem += val1 + val2;
              break;
            case 's':
              smem += val1 + val2;
              break;
          }
          continue;
        }

        num_read = sscanf(ptr, "%d bytes %1c mem", &val1, &mem_type);
        if (num_read == 2) {
          switch (mem_type) {
            default:
              llvm::errs() << "wrong memory specifier '" << mem_type
                           << "': " << ptr;
              break;
            case 'c':
              cmem += val1;
              break;
            case 'l':
              lmem += val1;
              break;
            case 's':
              smem += val1;
              break;
          }
          continue;
        }

        num_read = sscanf(ptr, "%d texture %1c", &val1, &mem_type);
        if (num_read == 2) {
          continue;
        }
        num_read = sscanf(ptr, "%d sampler %1c", &val1, &mem_type);
        if (num_read == 2) {
          continue;
        }
        num_read = sscanf(ptr, "%d surface %1c", &val1, &mem_type);
        if (num_read == 2) {
          continue;
        }

        // no match found
        llvm::errs() << "Unexpected memory usage specification: '" << ptr;
      }
    }
  }
  pclose(fpipe);

  if (reg == 0) {
    unsigned DiagIDCompile = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
        "Compiling kernel in file '%0.%1' failed, using default kernel configuration:\n%2");
    Diags.Report(DiagIDCompile)
      << K->getFileName() << (const char*)(compilerOptions.emitCUDA()?"cu":"cl")
      << command.c_str();
    for (auto line : lines)
      llvm::errs() << line;
  } else {
    if (targetDevice.isAMDGPU()) {
      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
                   << ": " << reg << " gprs, "
                   << smem << " bytes lds\n";
    } else {
      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
                   << ": " << reg << " registers, "
                   << lmem << " bytes lmem, "
                   << smem << " bytes smem, "
                   << cmem << " bytes cmem\n";
    }
  }

  K->setResourceUsage(reg, lmem, smem, cmem);
  #else
  K->setDefaultConfig();
  #endif
}


void Rewrite::printReductionFunction(HipaccKernelClass *KC, HipaccKernel *K,
    PrintingPolicy Policy, llvm::raw_ostream *OS) {
  FunctionDecl *fun = KC->getReduceFunction();

  // preprocessor defines
  if (!compilerOptions.exploreConfig()) {
    *OS << "#define BS " << K->getNumThreadsReduce() << "\n"
        << "#define PPT " << K->getPixelsPerThreadReduce() << "\n";
  }
  if (K->getIterationSpace()->isCrop()) {
    *OS << "#define USE_OFFSETS\n";
  }
  switch (compilerOptions.getTargetLang()) {
    case Language::C99: break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType()==Texture::Array2D) {
        *OS << "#define USE_ARRAY_2D\n";
      }
      *OS << "#include \"hipacc_cl_red.hpp\"\n\n";
      break;
    case Language::CUDA:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType()==Texture::Array2D) {
        *OS << "#define USE_ARRAY_2D\n";
      }
      *OS << "#include \"hipacc_cu_red.hpp\"\n\n";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      *OS << "#pragma version(1)\n"
          << "#pragma rs java_package_name("
          << compilerOptions.getRSPackageName()
          << ")\n\n";
      if (compilerOptions.emitFilterscript()) {
        *OS << "#define FS\n";
      }
      *OS << "#define DATA_TYPE "
          << K->getIterationSpace()->getImage()->getTypeStr() << "\n"
          << "#include \"hipacc_rs_red.hpp\"\n\n";
      // input/output allocation definitions
      *OS << "rs_allocation _red_Input;\n";
      *OS << "rs_allocation _red_Output;\n";
      // offset specification
      if (K->getIterationSpace()->isCrop()) {
        *OS << "int _red_offset_x;\n";
        *OS << "int _red_offset_y;\n";
      }
      *OS << "int _red_stride;\n";
      *OS << "int _red_is_height;\n";
      *OS << "int _red_num_elements;\n";
      break;
  }


  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
      *OS << "extern \"C\" {\n";
      *OS << "__device__ ";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      *OS << "static ";
      break;
  }
  *OS << "inline " << fun->getReturnType().getAsString() << " "
      << K->getReduceName() << "(";
  // write kernel parameters
  size_t comma = 0;
  for (auto param : fun->params()) {
    std::string Name(param->getNameAsString());
    QualType T = param->getType();
    // normal arguments
    if (comma++) *OS << ", ";
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(fun))
      T = Parm->getOriginalType();
    T.getAsStringInternal(Name, Policy);
    *OS << Name;
  }
  *OS << ") ";

  // print kernel body
  fun->getBody()->printPretty(*OS, 0, Policy, 0);

  // instantiate reduction
  switch (compilerOptions.getTargetLang()) {
    case Language::C99: break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      // 2D reduction
      *OS << "REDUCTION_CL_2D(" << K->getReduceName() << "2D, "
          << fun->getReturnType().getAsString() << ", "
          << K->getReduceName() << ", "
          << K->getIterationSpace()->getImage()->getImageReadFunction()
          << ")\n";
      // 1D reduction
      *OS << "REDUCTION_CL_1D(" << K->getReduceName() << "1D, "
          << fun->getReturnType().getAsString() << ", "
          << K->getReduceName() << ")\n";
      break;
    case Language::CUDA:
      // print 2D CUDA array definition - this is only required on Fermi and if
      // Array2D is selected, but doesn't harm otherwise
      *OS << "texture<" << fun->getReturnType().getAsString()
          << ", cudaTextureType2D, cudaReadModeElementType> _tex"
          << K->getIterationSpace()->getImage()->getName() + K->getName()
          << ";\nconst textureReference *_tex"
          << K->getIterationSpace()->getImage()->getName() + K->getName()
          << "Ref;\n\n";
      // 2D reduction
      if (compilerOptions.getTargetDevice()>=Device::Fermi_20 &&
          !compilerOptions.exploreConfig()) {
        *OS << "__device__ unsigned finished_blocks_" << K->getReduceName()
            << "2D = 0;\n\n";
        *OS << "REDUCTION_CUDA_2D_THREAD_FENCE(";
      } else {
        *OS << "REDUCTION_CUDA_2D(";
      }
      *OS << K->getReduceName() << "2D, "
          << fun->getReturnType().getAsString() << ", "
          << K->getReduceName() << ", _tex"
          << K->getIterationSpace()->getImage()->getName() + K->getName() << ")\n";
      // 1D reduction
      if (compilerOptions.getTargetDevice() >= Device::Fermi_20 &&
          !compilerOptions.exploreConfig()) {
        // no second step required
      } else {
        *OS << "REDUCTION_CUDA_1D(" << K->getReduceName() << "1D, "
            << fun->getReturnType().getAsString() << ", "
            << K->getReduceName() << ")\n";
      }
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      *OS << "REDUCTION_RS_2D(" << K->getReduceName() << "2D, "
          << fun->getReturnType().getAsString() << ", ALL, "
          << K->getReduceName() << ")\n";
      // 1D reduction
      *OS << "REDUCTION_RS_1D(" << K->getReduceName() << "1D, "
          << fun->getReturnType().getAsString() << ", ALL, "
          << K->getReduceName() << ")\n";
      break;
  }

  if (compilerOptions.emitCUDA()) {
    *OS << "}\n";
  }
  *OS << "#include \"hipacc_undef.hpp\"\n";

  *OS << "\n";
}


void Rewrite::printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
    HipaccKernel *K, std::string file, bool emitHints) {
  PrintingPolicy Policy = Context.getPrintingPolicy();
  Policy.Indentation = 2;
  Policy.SuppressSpecifiers = false;
  Policy.SuppressTag = false;
  Policy.SuppressScope = false;
  Policy.ConstantArraySizeAsWritten = false;
  Policy.AnonymousTagLocations = true;
  Policy.PolishForDeclaration = false;

  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
      Policy.LangOpts.CUDA = 1; break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      Policy.LangOpts.OpenCL = 1; break;
  }

  int fd;
  std::string filename(file);
  std::string ifdef("_" + file + "_");
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
  llvm::raw_ostream *OS = &llvm::errs();
  while ((fd = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0664)) < 0) {
    if (errno != EINTR) {
      std::string errorInfo("Error opening output file '" + filename + "'");
      perror(errorInfo.c_str());
    }
  }
  OS = new llvm::raw_fd_ostream(fd, false);

  // write ifndef, ifdef
  std::transform(ifdef.begin(), ifdef.end(), ifdef.begin(), ::toupper);
  *OS << "#ifndef " + ifdef + "\n";
  *OS << "#define " + ifdef + "\n\n";

  // preprocessor defines
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
      *OS << "#include \"hipacc_types.hpp\"\n"
          << "#include \"hipacc_math_functions.hpp\"\n\n";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      *OS << "#pragma version(1)\n"
          << "#pragma rs java_package_name("
          << compilerOptions.getRSPackageName()
          << ")\n\n";
      break;
  }


  // interpolation includes & definitions
  bool inc=false;
  SmallVector<std::string, 16> InterpolationDefinitionsLocal;
  size_t num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    size_t i = num_arg++;

    if (!Acc || !K->getUsed(K->getDeviceArgNames()[i])) continue;

    if (Acc->getInterpolationMode() != Interpolate::NO) {
      if (!inc) {
        inc = true;
        switch (compilerOptions.getTargetLang()) {
          case Language::C99: break;
          case Language::CUDA:
            *OS << "#include \"hipacc_cu_interpolate.hpp\"\n\n";
            break;
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            *OS << "#include \"hipacc_cl_interpolate.hpp\"\n\n";
            break;
          case Language::Renderscript:
          case Language::Filterscript:
              *OS << "#include \"hipacc_rs_interpolate.hpp\"\n\n";
            break;
        }
      }

      // define required interpolation mode
      if (inc && Acc->getInterpolationMode() > Interpolate::NN) {
        std::string function_name(ASTTranslate::getInterpolationName(Context,
              builtins, compilerOptions, K, Acc, border_variant()));
        std::string suffix("_" +
            builtins.EncodeTypeIntoStr(Acc->getImage()->getType(), Context));

        std::string resultStr;
        stringCreator.writeInterpolationDefinition(K, Acc, function_name,
            suffix, Acc->getInterpolationMode(), Acc->getBoundaryMode(),
            resultStr);

        switch (compilerOptions.getTargetLang()) {
          default: InterpolationDefinitionsLocal.push_back(resultStr); break;
          case Language::C99: break;
        }

        resultStr.erase();
        stringCreator.writeInterpolationDefinition(K, Acc, function_name,
            suffix, Interpolate::NO, Boundary::UNDEFINED, resultStr);

        switch (compilerOptions.getTargetLang()) {
          default: InterpolationDefinitionsLocal.push_back(resultStr); break;
          case Language::C99: break;
        }
      }
    }
  }

  if (((compilerOptions.emitCUDA() && // CUDA, but no exploration or no hints
          (compilerOptions.exploreConfig() || !emitHints)) ||
        !compilerOptions.emitCUDA())  // or other targets
      && inc && InterpolationDefinitionsLocal.size()) {
    // sort definitions and remove duplicate definitions
    std::sort(InterpolationDefinitionsLocal.begin(),
        InterpolationDefinitionsLocal.end());
    InterpolationDefinitionsLocal.erase(std::unique(
          InterpolationDefinitionsLocal.begin(),
          InterpolationDefinitionsLocal.end()),
        InterpolationDefinitionsLocal.end());

    // add interpolation definitions
    while (InterpolationDefinitionsLocal.size()) {
      *OS << InterpolationDefinitionsLocal.pop_back_val();
    }
    *OS << "\n";
  } else {
    // emit interpolation definitions at the beginning at the file
    if (InterpolationDefinitionsLocal.size()) {
      while (InterpolationDefinitionsLocal.size()) {
        InterpolationDefinitionsGlobal.push_back(
            InterpolationDefinitionsLocal.pop_back_val());
      }
    }
  }

  // declarations of textures, surfaces, variables, etc.
  num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    auto cur_arg = num_arg++;
    if (!K->getUsed(K->getDeviceArgNames()[cur_arg])) continue;

    // output image declaration
    if (cur_arg==0) {
      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          // surface declaration
          if (compilerOptions.useTextureMemory() &&
              compilerOptions.getTextureType()==Texture::Array2D) {
            *OS << "surface<void, cudaSurfaceType2D> _surfOutput"
                << K->getName() << ";\n"
                << "const struct surfaceReference *_surfOutput"
                << K->getName() << "Ref;\n\n";
          }
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          *OS << "rs_allocation " << K->getDeviceArgNames()[cur_arg] << ";\n";
          break;
      }
      continue;
    }

    // global image declarations
    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    if (Acc) {
      QualType T = Acc->getImage()->getType();

      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          // texture declaration
          if (KC->getImgAccess(arg) == READ_ONLY &&
              K->useTextureMemory(Acc)!=Texture::None) {
            // no texture declaration for __ldg() intrinsic
            if (K->useTextureMemory(Acc) == Texture::Ldg) break;
            *OS << "texture<";
            *OS << T.getAsString();
            switch (K->useTextureMemory(Acc)) {
              default: assert(0 && "texture expected.");
              case Texture::Linear1D:
                *OS << ", cudaTextureType1D, cudaReadModeElementType> _tex";
                break;
              case Texture::Linear2D:
              case Texture::Array2D:
                *OS << ", cudaTextureType2D, cudaReadModeElementType> _tex";
                break;
            }
            *OS << arg->getNameAsString() << K->getName() << ";\n"
                << "const textureReference *_tex"
                << arg->getNameAsString() << K->getName() << "Ref;\n";
          }
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          *OS << "rs_allocation " << arg->getNameAsString() << ";\n";
          break;
      }
      continue;
    }

    // constant memory declarations
    HipaccMask *Mask = K->getMaskFromMapping(arg);
    if (Mask) {
      if (Mask->isConstant()) {
        switch (compilerOptions.getTargetLang()) {
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            *OS << "__constant ";
            break;
          case Language::CUDA:
            *OS << "__device__ __constant__ ";
            break;
          case Language::C99:
          case Language::Renderscript:
          case Language::Filterscript:
            *OS << "static const ";
            break;
        }
        *OS << Mask->getTypeStr() << " " << Mask->getName() << K->getName() << "["
            << Mask->getSizeYStr() << "][" << Mask->getSizeXStr() << "] = {\n";

        // print Mask constant literals to 2D array
        for (size_t y=0; y<Mask->getSizeY(); ++y) {
          *OS << "        {";
          for (size_t x=0; x<Mask->getSizeX(); ++x) {
            Mask->getInitExpr(x, y)->printPretty(*OS, 0, Policy, 0);
            if (x<Mask->getSizeX()-1) {
              *OS << ", ";
            }
          }
          if (y<Mask->getSizeY()-1) {
            *OS << "},\n";
          } else {
            *OS << "}\n";
          }
        }
        *OS << "    };\n\n";
        Mask->setIsPrinted(true);
      } else {
        // emit declaration in CUDA and Renderscript
        // for other back ends, the mask will be added as kernel parameter
        switch (compilerOptions.getTargetLang()) {
          default: break;
          case Language::CUDA:
            *OS << "__device__ __constant__ " << Mask->getTypeStr() << " "
                << Mask->getName() << K->getName() << "[" << Mask->getSizeYStr()
                << "][" << Mask->getSizeXStr() << "];\n\n";
            Mask->setIsPrinted(true);
            break;
          case Language::Renderscript:
          case Language::Filterscript:
            *OS << "rs_allocation " << K->getDeviceArgNames()[cur_arg]
                << ";\n\n";
            Mask->setIsPrinted(true);
            break;
        }
      }
      continue;
    }

    // normal variables - Renderscript|Filterscript only
    if (compilerOptions.emitRenderscript() ||
        compilerOptions.emitFilterscript()) {
      QualType QT = K->getArgTypes()[cur_arg];
      QT.removeLocalConst();
      *OS << QT.getAsString() << " " << K->getDeviceArgNames()[cur_arg]
          << ";\n";
      continue;
    }
  }

  // extern scope for CUDA
  *OS << "\n";
  if (compilerOptions.emitCUDA()) {
    *OS << "extern \"C\" {\n";
  }

  // function definitions
  for (auto fun : K->getFunctionCalls()) {
    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        *OS << "inline "; break;
      case Language::CUDA:
        *OS << "__inline__ __device__ "; break;
      case Language::Renderscript:
      case Language::Filterscript:
        *OS << "inline static "; break;
    }
    fun->print(*OS, Policy);
  }

  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
    case Language::Renderscript:
      break;
    case Language::CUDA:
      *OS << "__global__ ";
      if (compilerOptions.exploreConfig() && emitHints) {
        *OS << "__launch_bounds__ (BSX_EXPLORE * BSY_EXPLORE) ";
      } else {
        *OS << "__launch_bounds__ (" << K->getNumThreadsX() << "*"
            << K->getNumThreadsY() << ") ";
      }
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType()==Texture::Array2D) {
        *OS << "__constant sampler_t " << D->getNameInfo().getAsString()
            << "Sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | "
            << " CLK_FILTER_NEAREST; \n\n";
      }
      *OS << "__kernel ";
      if (compilerOptions.exploreConfig() && emitHints) {
        *OS << "__attribute__((reqd_work_group_size(BSX_EXPLORE, BSY_EXPLORE, "
            << "1))) ";
      } else {
        *OS << "__attribute__((reqd_work_group_size(" << K->getNumThreadsX()
            << ", " << K->getNumThreadsY() << ", 1))) ";
      }
      break;
    case Language::Filterscript:
      *OS << K->getIterationSpace()->getImage()->getTypeStr()
          << " __attribute__((kernel)) ";
      break;
  }
  if (!compilerOptions.emitFilterscript()) {
    *OS << "void ";
  }
  *OS << K->getKernelName();
  *OS << "(";

  // write kernel parameters
  size_t comma = 0; num_arg = 0;
  for (auto param : D->params()) {
    size_t i = num_arg++;
    std::string Name(param->getNameAsString());
    FieldDecl *FD = K->getDeviceArgFields()[i];

    if (!K->getUsed(Name) &&
        !compilerOptions.emitFilterscript() &&
        !compilerOptions.emitRenderscript()) {
        // Proceed for Filterscript, because output image is never explicitly used
        continue;
    }

    QualType T = param->getType();
    T.removeLocalConst();
    T.removeLocalRestrict();

    // check if we have a Mask or Domain
    HipaccMask *Mask = K->getMaskFromMapping(FD);
    if (Mask) {
      if (Mask->isConstant()) continue;
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
          if (comma++) *OS << ", ";
          *OS << "const "
              << Mask->getTypeStr()
              << " " << Mask->getName() << K->getName()
              << "[" << Mask->getSizeYStr() << "]"
              << "[" << Mask->getSizeXStr() << "]";
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          if (comma++) *OS << ", ";
          *OS << "__constant ";
          T.getAsStringInternal(Name, Policy);
          *OS << Name;
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
    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    MemoryAccess memAcc = KC->getImgAccess(FD);
    if (i==0) { // first argument is always the output image
      bool doBreak = false;
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          break;
        case Language::CUDA:
          if (compilerOptions.useTextureMemory() &&
              compilerOptions.getTextureType()==Texture::Array2D) {
              continue;
          }
          break;
        case Language::Renderscript:
          // parameters are set separately for Renderscript
          // add parameters for dummy allocation and indices
          *OS << K->getIterationSpace()->getImage()->getTypeStr()
              << " *_iter, uint32_t x, uint32_t y";
          doBreak = true;
          break;
        case Language::Filterscript:
          *OS << "uint32_t x, uint32_t y";
          doBreak = true;
          break;
      }
      if (doBreak) break;
    }

    if (Acc) {
      switch (compilerOptions.getTargetLang()) {
        case Language::C99:
          if (comma++) *OS << ", ";
          if (memAcc==READ_ONLY) *OS << "const ";
          *OS << Acc->getImage()->getTypeStr()
              << " " << Name
              << "[" << Acc->getImage()->getSizeYStr() << "]"
              << "[" << Acc->getImage()->getSizeXStr() << "]";
          // alternative for Pencil:
          // *OS << "[static const restrict 2048][4096]";
          break;
        case Language::CUDA:
          if (K->useTextureMemory(Acc)!=Texture::None && memAcc==READ_ONLY &&
              // parameter required for __ldg() intrinsic
              !(K->useTextureMemory(Acc) == Texture::Ldg)) {
            // no parameter is emitted for textures
            continue;
          } else {
            if (comma++) *OS << ", ";
            if (memAcc==READ_ONLY) *OS << "const ";
            *OS << T->getPointeeType().getAsString();
            *OS << " * __restrict__ ";
            *OS << Name;
          }
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          // __global keyword to specify memory location is only needed for OpenCL
          if (comma++) *OS << ", ";
          if (K->useTextureMemory(Acc)!=Texture::None) {
            if (memAcc==WRITE_ONLY) {
              *OS << "__write_only image2d_t ";
            } else {
              *OS << "__read_only image2d_t ";
            }
          } else {
            *OS << "__global ";
            if (memAcc==READ_ONLY) *OS << "const ";
            *OS << T->getPointeeType().getAsString();
            *OS << " * restrict ";
          }
          *OS << Name;
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          break;
      }
      continue;
    }

    // normal arguments
    if (comma++) *OS << ", ";
    T.getAsStringInternal(Name, Policy);
    *OS << Name;

    // default arguments ...
    if (Expr *Init = param->getInit()) {
      CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init);
      if (!CCE || CCE->getConstructor()->isCopyConstructor()) {
        *OS << " = ";
      }
      Init->printPretty(*OS, 0, Policy, 0);
    }
  }
  *OS << ") ";

  // print kernel body
  D->getBody()->printPretty(*OS, 0, Policy, 0);
  if (compilerOptions.emitCUDA()) {
    *OS << "}\n";
  }
  *OS << "\n";

  if (KC->getReduceFunction()) {
    printReductionFunction(KC, K, Policy, OS);
  }

  *OS << "#endif //" + ifdef + "\n";
  *OS << "\n";
  OS->flush();
  fsync(fd);
  close(fd);
}

// vim: set ts=2 sw=2 sts=2 et ai:

