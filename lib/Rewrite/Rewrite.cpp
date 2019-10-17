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
// This file implements functionality for mapping the DSL to the Hipacc runtime.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Rewrite/Rewrite.h"
#include "hipacc/Config/config.h"
#ifdef USE_POLLY
#include "hipacc/Analysis/Polly.h"
#endif
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Backend/ICodeGenerator.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/DSL/CompilerKnownClasses.h"
#include "hipacc/Rewrite/CreateHostStrings.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/Support/Path.h>

#include <errno.h>
#include <fcntl.h>

#ifdef _WIN32
# include <io.h>
# define popen(x,y) _popen(x,y)
# define pclose(x)  _pclose(x)
# define fsync(x)
#else
# include <unistd.h>
#endif

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
    std::unique_ptr<llvm::raw_pwrite_stream> Out;
    Rewriter TextRewriter;
    Rewriter::RewriteOptions TextRewriteOptions;
    PrintingPolicy Policy;

    // Hipacc instances
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
    Rewrite(CompilerInstance &CI, CompilerOptions &options,
        std::unique_ptr<llvm::raw_pwrite_stream> Out) :
      CI(CI),
      Context(CI.getASTContext()),
      Diags(CI.getASTContext().getDiagnostics()),
      SM(CI.getASTContext().getSourceManager()),
      Out(std::move(Out)),
      Policy(PrintingPolicy(getLangOpts(options))),
      compilerOptions(options),
      targetDevice(options),
      builtins(CI.getASTContext()),
      stringCreator(CreateHostStrings(options, targetDevice)),
      compilerClasses(CompilerKnownClasses()),
      mainFD(nullptr),
      literalCount(0),
      skipTransfer(false)
    {}

    // RecursiveASTVisitor
    bool VisitCXXRecordDecl(CXXRecordDecl *D);
    bool VisitDeclStmt(DeclStmt *D);
    bool VisitFunctionDecl(FunctionDecl *D);
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    bool VisitCallExpr(CallExpr *E);

  private:
    // ASTConsumer
    void HandleTranslationUnit(ASTContext &) override;
    bool HandleTopLevelDecl(DeclGroupRef D) override;
    void Initialize(ASTContext &Context) override {
      mainFileID = SM.getMainFileID();
      TextRewriter.setSourceMgr(SM, Context.getLangOpts());
      TextRewriteOptions.RemoveLineIfEmpty = true;
    }

    // Rewrite
    std::string convertToString(Stmt *from) {
      assert(from != nullptr && "Expected non-null Stmt");
      std::string SS;
      llvm::raw_string_ostream S(SS);
      from->printPretty(S, nullptr, Policy);
      return S.str();
    }

    void removeText(SourceLocation beg, SourceLocation end, const char add) {
      const char *beg_ptr = SM.getCharacterData(beg);
      const char *end_ptr = strchr(SM.getCharacterData(end), add);
      TextRewriter.RemoveText(beg, end_ptr-beg_ptr+1, TextRewriteOptions);
    }

    void replaceText(SourceLocation beg, SourceLocation end, const char add,
        std::string str) {
      const char *beg_ptr = SM.getCharacterData(beg);
      const char *end_ptr = strchr(SM.getCharacterData(end), add);
      TextRewriter.ReplaceText(beg, end_ptr-beg_ptr+1, str);
    }

    LangOptions getLangOpts(CompilerOptions &options) {
      LangOptions LO;
      switch (options.getTargetLang()) {
        default:
          LO.C99 = 1; break;
        case Language::CUDA:
          LO.CUDA = 1; break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          LO.OpenCL = 1; break;
      }
      return LO;
    }

    void setKernelConfiguration(HipaccKernelClass *KC, HipaccKernel *K);
    void printBinningFunction(HipaccKernelClass *KC, HipaccKernel *K,
        llvm::raw_fd_ostream &OS);
    void printReductionFunction(HipaccKernelClass *KC, HipaccKernel *K,
        llvm::raw_fd_ostream &OS);
    void printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
        HipaccKernel *K, std::string file, bool emitHints);
};
}


std::unique_ptr<ASTConsumer>
HipaccRewriteAction::CreateASTConsumer(CompilerInstance &CI,
                                       StringRef /*in_file*/) {
  std::string out;
  if (!out_file.empty()) {
    StringRef rel_path(out_file);
    SmallString<1024> abs_path = rel_path;
    std::error_code EC = llvm::sys::fs::make_absolute(abs_path);
    assert(!EC); (void)EC;
    llvm::sys::path::native(abs_path);
    out = abs_path.str();
  }

  std::unique_ptr<llvm::raw_pwrite_stream> OS =
    CI.createOutputFile(out, false, true, "", "", false);
  assert(OS && "Cannot create output stream.");

  return llvm::make_unique<Rewrite>(CI, options, std::move(OS));
}


void Rewrite::HandleTranslationUnit(ASTContext &) {
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
  for (const char *buf_ptr = mainFileStart; buf_ptr < mainFileEnd; ++buf_ptr) {
    if (*buf_ptr == '#') {
      const char *beg_ptr = buf_ptr;
      if (++buf_ptr == mainFileEnd)
        break;
      while (*buf_ptr == ' ' || *buf_ptr == '\t')
        if (++buf_ptr == mainFileEnd)
          break;
      if (!strncmp(buf_ptr, "include", includeLen)) {
        const char *end_ptr = buf_ptr + includeLen;
        while (*end_ptr == ' ' || *end_ptr == '\t')
          if (++end_ptr == mainFileEnd)
            break;
        if (*end_ptr == '"') {
          if (!strncmp(end_ptr+1, "hipacc.hpp", hipaccHdrLen)) {
            // remove hipacc include
            end_ptr = strchr(end_ptr+1, '"');
            removeText(locStart.getLocWithOffset(beg_ptr-mainFileStart),
                       locStart.getLocWithOffset(end_ptr-mainFileStart), '"');
            buf_ptr += end_ptr-beg_ptr;
          }
        }
      }
    }
    if (*buf_ptr == 'u') {
      const char *beg_ptr = buf_ptr;
      if (!strncmp(buf_ptr, "using", usingLen)) {
        const char *end_ptr = buf_ptr + usingLen;
        while (*end_ptr == ' ' || *end_ptr == '\t')
          if (++end_ptr == mainFileEnd)
            break;
        if (*end_ptr == 'n') {
          if (!strncmp(end_ptr, "namespace", namespaceLen)) {
            end_ptr += namespaceLen;
            while (*end_ptr == ' ' || *end_ptr == '\t')
              if (++end_ptr == mainFileEnd)
                break;
            if (*end_ptr == 'h') {
              if (!strncmp(end_ptr, "hipacc", hipaccLen)) {
                // remove using namespace line
                removeText(locStart.getLocWithOffset(beg_ptr-mainFileStart),
                           locStart.getLocWithOffset(end_ptr-mainFileStart), ';');
                buf_ptr += end_ptr-beg_ptr;
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
              InterpolationDefinitionsGlobal.end(), std::greater<std::string>());
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
          HipaccKernel* K = map.second;
          newStr += "#include \"";
          newStr += K->getFileName();
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
      if (mask->isPrinted())
        continue;

      size_t i = 0;
      for (auto kernel : mask->getKernels()) {
        if (i++)
          newStr += "\n" + stringCreator.getIndent();

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

        TextRewriter.InsertTextBefore(mask->getDecl()->getBeginLoc(), newStr);
      }
    }
  }

  // insert initialization before first statement
  TextRewriter.InsertTextBefore(CS->body_front()->getBeginLoc(), initStr);

  // get buffer of main file id. If we haven't changed it, then we are done.
  if (auto RewriteBuf = TextRewriter.getRewriteBufferFor(mainFileID)) {
    *Out << std::string(RewriteBuf->begin(), RewriteBuf->end());
    Out->flush();
  } else {
    llvm::errs() << "No changes to input file, something went wrong!\n";
  }
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
  if (!D->hasDefinition())
    return true;

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
        else if (D->getNameAsString() == "Image")
          compilerClasses.Image = D;
        else if (D->getNameAsString() == "BoundaryCondition")
          compilerClasses.BoundaryCondition = D;
        else if (D->getNameAsString() == "AccessorBase")
          compilerClasses.AccessorBase = D;
        else if (D->getNameAsString() == "Accessor")
          compilerClasses.Accessor = D;
        else if (D->getNameAsString() == "IterationSpaceBase")
          compilerClasses.IterationSpaceBase = D;
        else if (D->getNameAsString() == "IterationSpace")
          compilerClasses.IterationSpace = D;
        else if (D->getNameAsString() == "ElementIterator")
          compilerClasses.ElementIterator = D;
        else if (D->getNameAsString() == "Kernel")
          compilerClasses.Kernel = D;
        else if (D->getNameAsString() == "Mask")
          compilerClasses.Mask = D;
        else if (D->getNameAsString() == "Domain")
          compilerClasses.Domain = D;
        else if (D->getNameAsString() == "Pyramid")
          compilerClasses.Pyramid = D;
        else if (D->getNameAsString() == "HipaccEoP")
          compilerClasses.HipaccEoP = D;
      }
    }

    if (!compilerClasses.HipaccEoP)
      return true;

    HipaccKernelClass *KC = nullptr;

    for (auto base : D->bases()) {
      // found user kernel class
      if (compilerClasses.isTypeOfTemplateClass(base.getType(),
            compilerClasses.Kernel)) {
        KC = new HipaccKernelClass(D->getNameAsString());
        KC->setPixelType(compilerClasses.getFirstTemplateType(base.getType()));
        KC->setBinType(compilerClasses.getTemplateType(base.getType(),
              compilerClasses.getNumberOfTemplateArguments(base.getType())-1));
        KernelClassDeclMap[D] = KC;
        // remove user kernel class
        removeText(D->getBeginLoc(), D->getEndLoc(), ';');

        break;
      }
    }

    if (!KC)
      return true;

    // find constructor
    CXXConstructorDecl *CCD = nullptr;
    for (auto ctor : D->ctors()) {
      if (ctor->isCopyOrMoveConstructor())
        continue;
      CCD = ctor;
    }
    assert(CCD && "Couldn't find user kernel class constructor!");


    // iterate over constructor initializers
    for (auto param : CCD->parameters()) {
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
        KC->setKernelFunction(method, compilerClasses);
        continue;
      }

      // reduce function
      if (method->getNameAsString() == "reduce") {
        KC->setReduceFunction(method);
        continue;
      }

      // binning function
      if (method->getNameAsString() == "binning") {
        KC->setBinningFunction(method);
        continue;
      }
    }
  }

  return true;
}


bool Rewrite::VisitDeclStmt(DeclStmt *D) {
  if (!compilerClasses.HipaccEoP)
    return true;

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
        std::string width_str  = convertToString(CCE->getArg(0));
        std::string height_str = convertToString(CCE->getArg(1));

        // TODO: No need for images in C++ to be of constant size, but this
        //       might become useful for FPGA targets
        //if (compilerOptions.emitC99()) {
        //  // check if the parameter can be resolved to a constant
        //  unsigned IDConstant = Diags.getCustomDiagID(DiagnosticsEngine::Error,
        //        "Constant expression for %0 argument of Image %1 required (C/C++ only).");
        //  if (!CCE->getArg(0)->isEvaluatable(Context)) {
        //    Diags.Report(CCE->getArg(0)->getExprLoc(), IDConstant) << "width"
        //      << Img->getName();
        //  }
        //  if (!CCE->getArg(1)->isEvaluatable(Context)) {
        //    Diags.Report(CCE->getArg(1)->getExprLoc(), IDConstant) << "height"
        //      << Img->getName();
        //  }

        //  Img->setSizeX(CCE->getArg(0)->EvaluateKnownConstInt(Context).getSExtValue());
        //  Img->setSizeY(CCE->getArg(1)->EvaluateKnownConstInt(Context).getSExtValue());
        //}

        // host memory
        std::string init_str = "NULL";
        if (CCE->getNumArgs() == 3)
          init_str = convertToString(CCE->getArg(2));

        // create memory allocation string
        std::string newStr;
        stringCreator.writeMemoryAllocation(Img, width_str, height_str,
            init_str, newStr);

        // rewrite Image definition
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

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
        std::string image_str = convertToString(CCE->getArg(0));
        std::string depth_str = convertToString(CCE->getArg(1));

        // create memory allocation string
        std::string newStr;
        stringCreator.writePyramidAllocation(VD->getName(),
            compilerClasses.getFirstTemplateType(VD->getType()).getAsString(),
            image_str, depth_str, newStr);

        // rewrite Pyramid definition
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

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
              BC->setPyramidIndex(convertToString(call->getArg(1)));
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
                if (i+2 != e)
                  Diags.Report(arg->getExprLoc(), IDMode) << VD->getName();
                // check if the parameter can be resolved to a constant
                auto const_arg = CCE->getArg(++i);
                if (!const_arg->isEvaluatable(Context)) {
                  Diags.Report(arg->getExprLoc(), IDConstMode) << VD->getName();
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
          if (!arg->isEvaluatable(Context))
            Diags.Report(arg->getExprLoc(), IDConstSize) << VD->getName();
          if (size_args++ == 0) {
            BC->setSizeX(arg->EvaluateKnownConstInt(Context).getSExtValue());
            BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
          } else {
            BC->setSizeY(arg->EvaluateKnownConstInt(Context).getSExtValue());
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
        std::string parms;
        size_t roi_args = 0;

        for (auto arg : CCE->arguments()) {
          auto dsl_arg = arg->IgnoreParenCasts();

          if (isa<CXXDefaultArgExpr>(dsl_arg))
            continue;

          if (auto call = dyn_cast<CXXOperatorCallExpr>(dsl_arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument specifies the boundary condition
            if (BCDeclMap.count(DRE->getDecl())) {
              BC = BCDeclMap[DRE->getDecl()];

              parms = BC->getImage()->getName();
              if (BC->isPyramid()) {
                // add call expression to pyramid argument
                parms += "(" + BC->getPyramidIndex() + ")";
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

              parms = BC->getImage()->getName();
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
              parms = convertToString(arg);
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
          parms += ", " + convertToString(arg);
          roi_args++;
        }

        assert(BC && "Expected BoundaryCondition, Image or Pyramid call as "
                     "first argument to Accessor.");

        Acc = new HipaccAccessor(VD, BC, mode, roi_args == 4);

        std::string newStr;
        newStr = "HipaccAccessor " + Acc->getName() + "(" + parms + ");";

        // replace Accessor decl by variables for width/height and offsets
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

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
        std::string parms;
        std::string pyr_idx;
        size_t roi_args = 0;

        for (auto arg : CCE->arguments()) {
          auto dsl_arg = arg->IgnoreParenCasts();
          if (auto call = dyn_cast<CXXOperatorCallExpr>(dsl_arg)) {
            // for pyramid call use the first argument
            dsl_arg = call->getArg(0);
          }

          // match for DSL arguments
          if (auto DRE = dyn_cast<DeclRefExpr>(dsl_arg)) {
            // check if the argument is an image
            if (ImgDeclMap.count(DRE->getDecl())) {
              Img = ImgDeclMap[DRE->getDecl()];
              parms = Img->getName();
              continue;
            }

            // check if the argument is a pyramid call
            if (PyrDeclMap.count(DRE->getDecl())) {
              Pyr = PyrDeclMap[DRE->getDecl()];
              // add call expression to pyramid argument
              auto call = dyn_cast<CXXOperatorCallExpr>(arg);
              pyr_idx = convertToString(call->getArg(1));
              parms = Pyr->getName() + "(" + pyr_idx + ")";
              continue;
            }
          }

          // get text string for arguments, argument order is:
          // img[, is_width, is_height[, offset_x, offset_y]]
          parms += ", " + convertToString(arg);
          roi_args++;
        }

        assert((Img || Pyr) && "Expected first argument of IterationSpace to "
                               "be Image or Pyramid call.");

        IS = new HipaccIterationSpace(VD, Img ? Img : Pyr, roi_args == 4);
        if (Pyr)
          IS->getBC()->setPyramidIndex(pyr_idx);
        ISDeclMap[VD] = IS; // store IterationSpace

        std::string newStr;
        newStr = "HipaccAccessor " + IS->getName() + "(" + parms + ");";

        // replace iteration space decl by variables for width/height and offset
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

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
          newStr += "\n" + stringCreator.getIndent();

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
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

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
          for (auto arg : CCE->arguments()) {
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

          // translate kernel function, replaces member variables
          ASTTranslate *Hipacc = new ASTTranslate(Context, kernelDecl, K, KC,
              builtins, compilerOptions);
          Stmt *kernelStmts =
            Hipacc->Hipacc(KC->getKernelFunction()->getBody());
          kernelDecl->setBody(kernelStmts);
          K->printStats();

          // translate binning function if we have one
          if (KC->getBinningFunction()) {
            Stmt *binningStmts = Hipacc->translateBinning(
                KC->getBinningFunction()->getBody());
            KC->getBinningFunction()->setBody(binningStmts);
          }

          #ifdef USE_POLLY
          if (!compilerOptions.exploreConfig() && compilerOptions.emitC99()) {
            llvm::errs() << "\nPassing the following function to Polly:\n";
            kernelDecl->print(llvm::errs(), Policy);
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
  if (!compilerClasses.HipaccEoP)
    return true;

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

  auto skip_mte = [&] (Expr *expr) -> Expr * {
    if (auto mte = dyn_cast<MaterializeTemporaryExpr>(expr))
      return mte->GetTemporaryExpr();
    return expr;
  };

  if (E->getOperator() == OO_Equal) {
    if (E->getNumArgs() != 2)
      return true;

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
    } else if (auto call = dyn_cast<CXXOperatorCallExpr>(skip_mte(E->getArg(0)))) {
      // check if we have an Pyramid or Domain call at the LHS
      if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
        // get the Pyramid from the DRE if we have one
        if (PyrDeclMap.count(DRE->getDecl())) {
          PyrLHS = PyrDeclMap[DRE->getDecl()];
          PyrIdxLHS = convertToString(call->getArg(1));
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
      if (ImgDeclMap.count(DRE->getDecl()))
        ImgRHS = ImgDeclMap[DRE->getDecl()];
      // check if we have an Accessor at the RHS
      if (AccDeclMap.count(DRE->getDecl()))
        AccRHS = AccDeclMap[DRE->getDecl()];
    } else if (auto call = dyn_cast<CXXOperatorCallExpr>(E->getArg(1))) {
      // check if we have an Pyramid call at the RHS
      if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
        // get the Pyramid from the DRE if we have one
        if (PyrDeclMap.count(DRE->getDecl())) {
          PyrRHS = PyrDeclMap[DRE->getDecl()];
          PyrIdxRHS = convertToString(call->getArg(1));
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

      // remove domain value assignment
      removeText(E->getBeginLoc(), E->getEndLoc(), ';');

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
        if (auto mcall =
            dyn_cast<CXXMemberCallExpr>(E->getArg(1)->IgnoreParenCasts())) {
          // match only data() calls to Image instances
          if (mcall->getDirectCallee()->getNameAsString() == "data") {
            // side effect ! do not handle the next call to data()
            skipTransfer = true;
            if (auto DRE =
                dyn_cast<DeclRefExpr>(mcall->getImplicitObjectArgument()->IgnoreParenCasts())) {
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
                                   mcall->getImplicitObjectArgument()->IgnoreParenCasts())) {
              // check if we have an Pyramid call
              if (auto DRE = dyn_cast<DeclRefExpr>(call->getArg(0))) {
                // get the Pyramid from the DRE if we have one
                if (PyrDeclMap.count(DRE->getDecl())) {
                  HipaccPyramid *Pyr = PyrDeclMap[DRE->getDecl()];

                  // add call expression to pyramid argument
                  std::string index = convertToString(call->getArg(1));

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
          std::string data_str = convertToString(E->getArg(1));

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
      replaceText(E->getBeginLoc(), E->getEndLoc(), ';', newStr);

      return true;
    }
  }

  return true;
}


bool Rewrite::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  if (!compilerClasses.HipaccEoP)
    return true;

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
        assert(CCE->getNumArgs() == K->getKernelClass()->getMembers().size() &&
            "number of arguments doesn't match!");

        // set host argument names and retrieve literals stored to temporaries
        K->setHostArgNames(llvm::makeArrayRef(CCE->getArgs(),
              CCE->getNumArgs()), newStr, literalCount);

        //
        // TODO: handle the case when only reduce function is specified
        //
        // create kernel call string
        stringCreator.writeKernelCall(K, newStr);

        // rewrite kernel invocation
        replaceText(E->getBeginLoc(), E->getBeginLoc(), ';', newStr);
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
        if (ME->getMemberNameInfo().getAsString() == "binned_data"
            || ME->getMemberNameInfo().getAsString() == "reduced_data") {
          HipaccKernel *K = KernelDeclMap[DRE->getDecl()];

          std::string callStr, resultStr;
          if (ME->getMemberNameInfo().getAsString() == "binned_data") {
            auto numBinsExpr = E->getArg(0)->IgnoreImpCasts();
            std::string numBinsStr;
            llvm::raw_string_ostream SS(numBinsStr);
            numBinsExpr->printPretty(SS, 0, Policy);
            K->setNumBinsStr(SS.str());

            assert(K->getKernelClass()->getBinningFunction()
                   && "Called binned_data() but no binning function defined!");

            callStr += "\n" + stringCreator.getIndent();
            stringCreator.writeBinningCall(K, callStr);

            resultStr = K->getBinningStr();
          } else {
            assert(K->getKernelClass()->getReduceFunction()
                   && "Called reduced_data() but no reduce function defined!");

            callStr += "\n" + stringCreator.getIndent();
            stringCreator.writeReductionDeclaration(K, callStr);
            stringCreator.writeReduceCall(K, callStr);

            resultStr = K->getReduceStr();
          }

          // insert reduction call in the line before
          unsigned fileNum = SM.getSpellingLineNumber(E->getBeginLoc(), nullptr);
          SourceLocation callLoc = SM.translateLineCol(mainFileID, fileNum, 1);
          TextRewriter.InsertText(callLoc, callStr);

          //
          // TODO: make sure that kernel was executed before *_data call
          //
          // replace member function invocation
          SourceRange range(E->getBeginLoc(), E->getEndLoc());
          TextRewriter.ReplaceText(range, resultStr);

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
          replaceText(E->getBeginLoc(), E->getEndLoc(), ';', newStr);

          return true;
        }

        if (ME->getMemberNameInfo().getAsString() == "width") {
          newStr = "->width";
        } else if (ME->getMemberNameInfo().getAsString() == "height") {
          newStr = "->height";
        }
      }

      // get the Accessor from the DRE if we have one
      if (AccDeclMap.count(DRE->getDecl())) {
        // match for supported member calls
        if (ME->getMemberNameInfo().getAsString() == "width") {
          newStr = ".img->width";
        } else if (ME->getMemberNameInfo().getAsString() == "height") {
          newStr = ".img->height";
        }
      }

      if (!newStr.empty()) {
        // replace member function invocation
        SourceRange range(ME->getOperatorLoc(), E->getEndLoc());
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
        SourceRange range(E->getBeginLoc(),
                          E->getBeginLoc().getLocWithOffset(std::string("traverse").length()-1));
        TextRewriter.ReplaceText(range, "hipaccTraverse");
      }
    }
  }
  return true;
}


void Rewrite::setKernelConfiguration(HipaccKernelClass *KC, HipaccKernel *K) {
  #ifdef USE_JIT_ESTIMATE
  switch (compilerOptions.getTargetLang()) {
    default: return K->setDefaultConfig();
    case Language::CUDA:
    case Language::OpenCLGPU:
      if (!targetDevice.isARMGPU())
        break;
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
  std::string command = K->getCompileCommand(K->getKernelName(),
      K->getFileName(), compilerOptions.emitCUDA());

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
      << K->getFileName() << (const char*)(compilerOptions.emitCUDA()?"cu":"cl")
      << command.c_str();
    for (auto line : lines)
      llvm::errs() << line;
  } else {
    if (targetDevice.isNVIDIAGPU()) {
      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
                   << ": " << reg << " registers, "
                   << lmem << " bytes lmem, "
                   << smem << " bytes smem, "
                   << cmem << " bytes cmem\n";
    } else if (targetDevice.isAMDGPU()) {
      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
                   << ": " << reg << " gprs, "
                   << smem << " bytes lds\n";
    }
  }

  K->setResourceUsage(reg, lmem, smem, cmem);
  #else
  K->setDefaultConfig();
  #endif
}


void Rewrite::printBinningFunction(HipaccKernelClass *KC, HipaccKernel *K,
    llvm::raw_fd_ostream &OS) {
  FunctionDecl *bin_fun = KC->getBinningFunction();
  QualType pixelType = KC->getPixelType();
  QualType binType = KC->getBinType();
  std::string signatureBinning;

  if (compilerOptions.exploreConfig()) {
    assert(false && "Explorations not supported for multi-dimensional reductions");
  }

  // preprocessor defines
  std::string KID = K->getKernelName();
  switch (compilerOptions.getTargetLang()) {
    case Language::Renderscript:
    case Language::Filterscript:
      assert(false && "Multi-dimensional reductions is not supported for Renderscript");
      break;
    case Language::C99:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
    case Language::CUDA:
      OS << "#define " << KID << "PPT " << K->getPixelsPerThread() << "\n";
      break;
  }
  OS << "\n";

  // write binning signature and qualifiers
  if (compilerOptions.emitCUDA()) {
    OS << "extern \"C\" {\n";
    signatureBinning += "__device__ ";
  }
  signatureBinning += "inline void " + K->getBinningName() + "(";
  if (compilerOptions.emitOpenCL()) {
    signatureBinning += "__local ";
  }
  signatureBinning += binType.getAsString();
  signatureBinning += " *_lmem, uint _offset, uint _num_bins, ";

  // write other binning parameters
  size_t comma = 0;
  for (auto param : bin_fun->parameters()) {
    std::string Name(param->getNameAsString());
    QualType T = param->getType();
    // normal arguments
    if (comma++)
      signatureBinning += ", ";
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(bin_fun))
      T = Parm->getOriginalType();
    T.getAsStringInternal(Name, Policy);
    signatureBinning += Name;
  }
  signatureBinning += ")";

  // print forward declaration
  OS << signatureBinning << ";\n\n";

  // instantiate reduction
  switch (compilerOptions.getTargetLang()) {
    case Language::Renderscript:
    case Language::Filterscript:
      break;
    case Language::C99:
      OS << "BINNING_CPU_2D(";
      OS << K->getBinningName() << "2D, "
         << pixelType.getAsString() << ", "
         << binType.getAsString() << ", "
         << K->getReduceName() << ", "
         << K->getBinningName() << ", "
         << KID << "PPT"
         << ")\n\n";
      break;
    case Language::CUDA: {
      // 2D reduction
      OS << "__device__ unsigned finished_blocks_" << K->getBinningName()
         << "2D[MAX_SEGMENTS] = {0};\n\n";
      OS << "BINNING_CUDA_2D_SEGMENTED("
         << K->getBinningName() << "2D, ";
      // fall through!

    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      if (compilerOptions.emitOpenCL()) {
        OS << "BINNING_CL_2D_SEGMENTED("
           << K->getBinningName() << "2D, "
           << K->getBinningName() << "1D, ";
      }

      OS << pixelType.getAsString() << ", "
         << binType.getAsString() << ", "
         << K->getReduceName() << ", "
         << K->getBinningName() << ", ";

      size_t bitWidth = 32;
      if (isa<VectorType>(binType.getCanonicalType().getTypePtr())) {
        const VectorType *VT = dyn_cast<VectorType>(
            binType.getCanonicalType().getTypePtr());
        VectorTypeInfo info = createVectorTypeInfo(VT);
        bitWidth = info.elementCount * info.elementWidth;
      } else {
        bitWidth = getBuiltinTypeSize(binType->getAs<BuiltinType>());
      }

      if (bitWidth > 64) {
        // >64bit: Synchronize using 64bit atomicCAS (might cause errors)
        llvm::errs() << "WARNING: Potential data race if first 64 bits of bin write are identical to current bin value!\n";
        OS << "ACCU_CAS_GT64, UNTAG_NONE, ";
        // TODO: Implement synchronization using locks for bin types >64bit
        // TODO: Consider compiler switch to force locks for bin types >64bit
      } else {
        if (binType.getTypePtr()->isIntegerType()) {
          // INT: Synchronize using thread ID tagging
          llvm::errs() << "WARNING: First 5 bits of bin value are used for thread ID tagging!\n";
          OS << "ACCU_INT, UNTAG_INT, ";
          // TODO: Consider compiler switch to force CAS for full bit width
        } else {
          // CAS: Synchronize using atomicCAS (32 or 64 bit)
          OS << "ACCU_CAS_" << bitWidth << ", UNTAG_NONE, ";
        }
      }

      OS << K->getWarpSize() << ", "
         << compilerOptions.getReduceConfigNumWarps() << ", "
         << compilerOptions.getReduceConfigNumHists() << ", "
         << KID << "PPT, ";

      if (compilerOptions.emitCUDA()) {
        OS << "SEGMENT_SIZE, " // defined in "hipacc_cu.hpp"
           << (binType.getTypePtr()->isVectorType()
               ? "make_" + binType.getAsString() + "(0), "
               : "(0), ")
           << "_tex" << K->getIterationSpace()->getImage()->getName() + K->getName();
      } else {
        OS << (binType.getTypePtr()->isVectorType()
               ? "(" + binType.getAsString() + ")(0)"
               : "(0)");
      }

      OS << ")\n\n";
      }
      break;
  }

  // print binning function
  OS << signatureBinning << "\n";
  bin_fun->getBody()->printPretty(OS, 0, Policy, 0);
  OS << "\n";

  if (compilerOptions.emitCUDA())
    OS << "}\n";
  OS << "\n";
}


void Rewrite::printReductionFunction(HipaccKernelClass *KC, HipaccKernel *K,
    llvm::raw_fd_ostream &OS) {
  FunctionDecl *fun = KC->getReduceFunction();

  // preprocessor defines
  if (!compilerOptions.exploreConfig()) {
    OS << "#define BS " << K->getNumThreadsReduce() << "\n"
       << "#define PPT " << K->getPixelsPerThreadReduce() << "\n";
  }
  if (K->getIterationSpace()->isCrop()) {
    OS << "#define USE_OFFSETS\n";
  }
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
      if (compilerOptions.useOpenMP()) {
        OS << "#define USE_OPENMP\n";
      }
      OS << "#include \"hipacc_cpu_red.hpp\"\n\n";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType() == Texture::Array2D) {
        OS << "#define USE_ARRAY_2D\n";
      }
      OS << "#include \"hipacc_cl_red.hpp\"\n\n";
      break;
    case Language::CUDA:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType() == Texture::Array2D) {
        OS << "#define USE_ARRAY_2D\n";
      }
      OS << "#include \"hipacc_cu_red.hpp\"\n\n";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      OS << "#pragma version(1)\n"
         << "#pragma rs java_package_name("
         << compilerOptions.getRSPackageName()
         << ")\n\n";
      if (compilerOptions.emitFilterscript()) {
        OS << "#define FS\n";
      }
      OS << "#define DATA_TYPE "
         << K->getIterationSpace()->getImage()->getTypeStr() << "\n"
         << "#include \"hipacc_rs_red.hpp\"\n\n";
      // input/output allocation definitions
      OS << "rs_allocation _red_Input;\n";
      OS << "rs_allocation _red_Output;\n";
      // offset specification
      if (K->getIterationSpace()->isCrop()) {
        OS << "int _red_offset_x;\n";
        OS << "int _red_offset_y;\n";
      }
      OS << "int _red_stride;\n";
      OS << "int _red_is_height;\n";
      OS << "int _red_num_elements;\n";
      break;
  }


  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case Language::CUDA:
      OS << "extern \"C\" {\n";
      OS << "__device__ ";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      OS << "static ";
      break;
  }
  OS << "inline " << fun->getReturnType().getAsString() << " "
     << K->getReduceName() << "(";
  // write kernel parameters
  size_t comma = 0;
  for (auto param : fun->parameters()) {
    std::string Name(param->getNameAsString());
    QualType T = param->getType();
    // normal arguments
    if (comma++)
      OS << ", ";
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(fun))
      T = Parm->getOriginalType();
    T.getAsStringInternal(Name, Policy);
    OS << Name;
  }
  OS << ") ";

  // print kernel body
  fun->getBody()->printPretty(OS, 0, Policy, 0);

  // instantiate reduction
  switch (compilerOptions.getTargetLang()) {
    case Language::C99:
      // 2D reduction
      OS << "REDUCTION_CPU_2D(" << K->getReduceName() << "2D, "
         << fun->getReturnType().getAsString() << ", "
         << K->getReduceName() << ", "
         << "PPT)\n";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      // 2D reduction
      OS << "REDUCTION_CL_2D(" << K->getReduceName() << "2D, "
         << fun->getReturnType().getAsString() << ", "
         << K->getReduceName() << ", "
         << K->getIterationSpace()->getImage()->getImageReadFunction()
         << ")\n";
      // 1D reduction
      OS << "REDUCTION_CL_1D(" << K->getReduceName() << "1D, "
         << fun->getReturnType().getAsString() << ", "
         << K->getReduceName() << ")\n";
      break;
    case Language::CUDA:
      // 2D CUDA array definition - only required if Array2D is selected
      OS << "texture<" << fun->getReturnType().getAsString()
         << ", cudaTextureType2D, cudaReadModeElementType> _tex"
         << K->getIterationSpace()->getImage()->getName() + K->getName()
         << ";\n__device__ const textureReference *_tex"
         << K->getIterationSpace()->getImage()->getName() + K->getName()
         << "Ref;\n\n";
      // define reduction only if pixel and bin are of the same type
      if (KC->getPixelType() == KC->getBinType()) {
        // 2D reduction
        if (compilerOptions.exploreConfig()) {
          OS << "REDUCTION_CUDA_2D(";
        } else {
          OS << "__device__ unsigned finished_blocks_" << K->getReduceName()
             << "2D = 0;\n\n";
          OS << "REDUCTION_CUDA_2D_THREAD_FENCE(";
        }
        OS << K->getReduceName() << "2D, "
           << fun->getReturnType().getAsString() << ", "
           << K->getReduceName() << ", _tex"
           << K->getIterationSpace()->getImage()->getName() + K->getName() << ")\n";
        // 1D reduction
        if (compilerOptions.exploreConfig()) {
          OS << "REDUCTION_CUDA_1D(" << K->getReduceName() << "1D, "
             << fun->getReturnType().getAsString() << ", "
             << K->getReduceName() << ")\n";
        }
      }
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      OS << "REDUCTION_RS_2D(" << K->getReduceName() << "2D, "
         << fun->getReturnType().getAsString() << ", ALL, "
         << K->getReduceName() << ")\n";
      // 1D reduction
      OS << "REDUCTION_RS_1D(" << K->getReduceName() << "1D, "
         << fun->getReturnType().getAsString() << ", ALL, "
         << K->getReduceName() << ")\n";
      break;
  }

  if (compilerOptions.emitCUDA())
    OS << "}\n";
  OS << "#include \"hipacc_undef.hpp\"\n";
  OS << "\n";
}


void Rewrite::printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
    HipaccKernel *K, std::string file, bool emitHints) {
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
      // TODO: remove when fixed upstream
      // https://reviews.llvm.org/D54258
      OS << "#ifndef __shared\n"
         << "#define __device __device__\n"
         << "#define __constant __constant__\n"
         << "#define __shared __shared__\n"
         << "#endif\n\n";
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      OS << "#pragma version(1)\n"
         << "#pragma rs java_package_name("
         << compilerOptions.getRSPackageName()
         << ")\n\n";
      break;
  }

  // declarations of textures, surfaces, variables, includes, definitions etc.
  SmallVector<std::string, 16> InterpolationDefinitionsLocal;
  size_t num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    auto cur_arg = num_arg++;
    if (!K->getUsed(K->getDeviceArgNames()[cur_arg]))
      continue;

    // global image declarations and interpolation definitions
    if (auto Acc = K->getImgFromMapping(arg)) {
      QualType T = Acc->getImage()->getType();

      switch (compilerOptions.getTargetLang()) {
        default: break;
        case Language::CUDA:
          // texture and surface declarations
          if (KC->getMemAccess(arg) == WRITE_ONLY) {
            if (K->useTextureMemory(Acc) == Texture::Array2D)
              OS << "surface<void, cudaSurfaceType2D> _tex"
                 << arg->getNameAsString() << K->getName() << ";\n";
          } else {
            if (K->useTextureMemory(Acc) != Texture::None &&
                K->useTextureMemory(Acc) != Texture::Ldg) {
              OS << "texture<";
              OS << T.getAsString();
              switch (K->useTextureMemory(Acc)) {
                default: assert(0 && "texture expected.");
                case Texture::Linear1D:
                  OS << ", cudaTextureType1D, cudaReadModeElementType> _tex";
                  break;
                case Texture::Linear2D:
                case Texture::Array2D:
                  OS << ", cudaTextureType2D, cudaReadModeElementType> _tex";
                  break;
              }
              OS << arg->getNameAsString() << K->getName() << ";\n";
            }
          }
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          OS << "rs_allocation " << arg->getNameAsString() << ";\n";
          break;
      }

      if (Acc->getInterpolationMode() > Interpolate::NN) {
        switch (compilerOptions.getTargetLang()) {
          case Language::C99: break;
          case Language::CUDA:
            OS << "#include \"hipacc_cu_interpolate.hpp\"\n\n";
            break;
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            OS << "#include \"hipacc_cl_interpolate.hpp\"\n\n";
            break;
          case Language::Renderscript:
          case Language::Filterscript:
            OS << "#include \"hipacc_rs_interpolate.hpp\"\n\n";
            break;
        }

        // define required interpolation mode
        std::string function_name(ASTTranslate::getInterpolationName(
              compilerOptions, K, Acc));
        std::string suffix("_" +
            builtins.EncodeTypeIntoStr(Acc->getImage()->getType(), Context));

        auto bh_def = stringCreator.getInterpolationDefinition(K, Acc,
            function_name, suffix, Acc->getInterpolationMode(),
            Acc->getBoundaryMode());
        auto no_bh_def = stringCreator.getInterpolationDefinition(K, Acc,
            function_name, suffix, Interpolate::NO, Boundary::UNDEFINED);
        auto vec_conv = Acc->getImage()->getType()->isVectorType() ?
          "VECTOR_TYPE_FUNS(" + Acc->getImage()->getTypeStr() + ")\n" :
          "SCALAR_TYPE_FUNS(" + Acc->getImage()->getTypeStr() + ")\n";

        switch (compilerOptions.getTargetLang()) {
          default: InterpolationDefinitionsLocal.push_back(bh_def);
                   InterpolationDefinitionsLocal.push_back(no_bh_def);
                   InterpolationDefinitionsLocal.push_back(vec_conv);
                   break;
          case Language::C99: break;
        }
      }
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
            OS << "rs_allocation " << K->getDeviceArgNames()[cur_arg]
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
      OS << QT.getAsString() << " " << K->getDeviceArgNames()[cur_arg] << ";\n";
      continue;
    }
  }

  // interpolation definitions
  if (InterpolationDefinitionsLocal.size()) {
    // sort definitions and remove duplicate definitions
    std::sort(InterpolationDefinitionsLocal.begin(),
              InterpolationDefinitionsLocal.end(), std::greater<std::string>());
    InterpolationDefinitionsLocal.erase(
        std::unique(InterpolationDefinitionsLocal.begin(),
                    InterpolationDefinitionsLocal.end()),
        InterpolationDefinitionsLocal.end());

    if (compilerOptions.emitCUDA() &&
        !compilerOptions.exploreConfig() && emitHints) {
      // emit interpolation definitions at the beginning of main file
      for (auto str : InterpolationDefinitionsLocal)
        InterpolationDefinitionsGlobal.push_back(str);
    } else {
      // add interpolation definitions to kernel file
      for (auto str : InterpolationDefinitionsLocal)
        OS << str;
      OS << "\n";
    }
  }

  // extern scope for CUDA
  OS << "\n";
  if (compilerOptions.emitCUDA())
    OS << "extern \"C\" {\n";

  // function definitions
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

  // Launch the print method of the code generator
  compilerOptions.getCodeGenerator()->SetPrintingPolicy(&Policy);
  bool bKernelPrinted = compilerOptions.getCodeGenerator()->PrintKernelFunction(D, K, OS);

  // Check if the code generator handled the printing of the kernel function
  if (!bKernelPrinted) {
    // write kernel name and qualifiers
    switch (compilerOptions.getTargetLang()) {
      case Language::C99:
      case Language::Renderscript:
        break;
      case Language::CUDA:
        OS << "__global__ ";
        if (compilerOptions.exploreConfig() && emitHints) {
          OS << "__launch_bounds__ (BSX_EXPLORE * BSY_EXPLORE) ";
        } else {
          OS << "__launch_bounds__ (" << K->getNumThreadsX() << "*"
             << K->getNumThreadsY() << ") ";
        }
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (compilerOptions.useTextureMemory() &&
            compilerOptions.getTextureType() == Texture::Array2D) {
          OS << "__constant sampler_t " << D->getNameInfo().getAsString()
             << "Sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | "
             << " CLK_FILTER_NEAREST; \n\n";
        }
        OS << "__kernel ";
        if (compilerOptions.exploreConfig() && emitHints) {
          OS << "__attribute__((reqd_work_group_size(BSX_EXPLORE, BSY_EXPLORE, "
             << "1))) ";
        } else {
          OS << "__attribute__((reqd_work_group_size(" << K->getNumThreadsX()
             << ", " << K->getNumThreadsY() << ", 1))) ";
        }
        break;
      case Language::Filterscript:
        OS << K->getIterationSpace()->getImage()->getTypeStr()
           << " __attribute__((kernel)) ";
        break;
    }
    if (!compilerOptions.emitFilterscript())
      OS << "void ";
    OS << K->getKernelName();
    OS << "(";

    // write kernel parameters
    size_t comma = 0; num_arg = 0;
    for (auto param : D->parameters()) {
      // print default parameters for Renderscript and Filterscript only
      if (compilerOptions.emitFilterscript()) {
        OS << "uint32_t x, uint32_t y";
        break;
      }
      if (compilerOptions.emitRenderscript()) {
        OS << K->getIterationSpace()->getImage()->getTypeStr()
           << " *_iter, uint32_t x, uint32_t y";
        break;
      }

      size_t i = num_arg++;
      FieldDecl *FD = K->getDeviceArgFields()[i];

      QualType T = param->getType();
      T.removeLocalConst();
      T.removeLocalRestrict();

      std::string Name(param->getNameAsString());
      if (!K->getUsed(Name))
        continue;

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
    D->getBody()->printPretty(OS, 0, Policy, 0);
    if (compilerOptions.emitCUDA())
      OS << "}\n";
  }
  OS << "\n";

  if (KC->getReduceFunction())
    printReductionFunction(KC, K, OS);

  // ensure emitHints, otherwise binning will interfere with analytics
  if (emitHints && KC->getBinningFunction())
    printBinningFunction(KC, K, OS);

  OS << "#endif //" + ifdef + "\n";
  OS << "\n";
  OS.flush();
  fsync(fd);
  close(fd);
}

// vim: set ts=2 sw=2 sts=2 et ai:

