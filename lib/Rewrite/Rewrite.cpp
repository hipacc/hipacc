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

//===--- Rewrite.cpp - OpenCL/CUDA rewriter for the AST -------------------===//
//
// This file implements functionality for rewriting OpenCL/CUDA kernels.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Rewrite/Rewrite.h"

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
    llvm::raw_ostream &Out;
    bool dump;
    SourceManager *SM;
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
    llvm::DenseMap<RecordDecl *, HipaccGlobalReductionClass *> GlobalReductionClassDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccAccessor *> AccDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccBoundaryCondition *> BCDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccGlobalReduction *> GlobalReductionDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccImage *> ImgDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccIterationSpace *> ISDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccKernel *> KernelDeclMap;
    llvm::DenseMap<ValueDecl *, HipaccMask *> MaskDeclMap;

    // store .reduce() calls - the kernels are created when the TranslationUnit
    // is handled
    SmallVector<CXXMemberCallExpr *, 16> ReductionCalls;
    SmallVector<HipaccGlobalReduction *, 16> InvokedReductions;
    // store interpolation methods required for CUDA
    std::vector<std::string> InterpolationDefinitions;

    // pointer to main function
    FunctionDecl *mainFD;
    FileID mainFileID;
    unsigned int literalCount;
    unsigned int isLiteralCount;

  public:
    Rewrite(CompilerInstance &CI, CompilerOptions &options, llvm::raw_ostream*
        o=NULL, bool dump=false) :
      CI(CI),
      Context(CI.getASTContext()),
      Diags(CI.getASTContext().getDiagnostics()),
      Out(o? *o : llvm::outs()),
      dump(dump),
      compilerOptions(options),
      targetDevice(options),
      builtins(CI.getASTContext()),
      stringCreator(CreateHostStrings(options)),
      compilerClasses(CompilerKnownClasses()),
      mainFD(NULL),
      literalCount(0),
      isLiteralCount(0)
    {}

    void HandleTranslationUnit(ASTContext &Context);
    bool HandleTopLevelDecl(DeclGroupRef D);

    bool VisitCXXRecordDecl(CXXRecordDecl *D);
    bool VisitDeclStmt(DeclStmt *D);
    bool VisitFunctionDecl(FunctionDecl *D);
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    bool VisitBinaryOperator(BinaryOperator *E);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr *E);

    //bool shouldVisitTemplateInstantiations() const { return true; }

  private:
    void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();

      // get the ID and start/end of the main file.
      mainFileID = SM->getMainFileID();
      TextRewriter.setSourceMgr(Context.getSourceManager(),
          Context.getLangOpts());
      TextRewriteOptions.RemoveLineIfEmpty = true;
    }

    void generateReductionKernels();
    void printReductionFunction(FunctionDecl *D, HipaccGlobalReduction *GR,
        std::string file);
    void printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
        HipaccKernel *K, std::string file, bool emitHints);
};
}
ASTConsumer *CreateRewrite(CompilerInstance &CI, CompilerOptions &options,
    llvm::raw_ostream* out) {
  return new Rewrite(CI, options, out);
}


ASTConsumer *HipaccRewriteAction::CreateASTConsumer(CompilerInstance &CI,
    StringRef InFile) {
  if (llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, InFile)) {
    return CreateRewrite(CI, options, OS);
  }

  return NULL;
}


void Rewrite::HandleTranslationUnit(ASTContext &Context) {
  assert(compilerClasses.Coordinate && "Coordinate class not found!");
  assert(compilerClasses.Image && "Image class not found!");
  assert(compilerClasses.BoundaryCondition && "BoundaryCondition class not found!");
  assert(compilerClasses.AccessorBase && "AccessorBase class not found!");
  assert(compilerClasses.Accessor && "Accessor class not found!");
  assert(compilerClasses.AccessorNN && "AccessorNN class not found!");
  assert(compilerClasses.AccessorLF && "AccessorLF class not found!");
  assert(compilerClasses.AccessorCF && "AccessorCF class not found!");
  assert(compilerClasses.AccessorL3 && "AccessorL3 class not found!");
  assert(compilerClasses.IterationSpaceBase && "IterationSpaceBase class not found!");
  assert(compilerClasses.IterationSpace && "IterationSpace class not found!");
  assert(compilerClasses.ElementIterator && "ElementIterator class not found!");
  assert(compilerClasses.Kernel && "Kernel class not found!");
  assert(compilerClasses.GlobalReduction && "GlobalReduction class not found!");
  assert(compilerClasses.Mask && "Mask class not found!");
  assert(compilerClasses.HipaccEoP && "HipaccEoP class not found!");

  // generate reduction kernel calls
  if (ReductionCalls.size()) generateReductionKernels();

  StringRef MainBuf = SM->getBufferData(mainFileID);
  const char *mainFileStart = MainBuf.begin();
  const char *mainFileEnd = MainBuf.end();
  SourceLocation locStart = SM->getLocForStartOfFile(mainFileID);

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
  if (InterpolationDefinitions.size()) {
    newStr += "#include \"hipacc_cuda_interpolate.hpp\"\n";

    // sort definitions and remove duplicate definitions
    std::sort(InterpolationDefinitions.begin(), InterpolationDefinitions.end());
    InterpolationDefinitions.erase(std::unique(InterpolationDefinitions.begin(),
          InterpolationDefinitions.end()), InterpolationDefinitions.end());

    // add interpolation definitions
    for (unsigned int i=0, e=InterpolationDefinitions.size(); i!=e; ++i) {
      newStr += InterpolationDefinitions.data()[i];
    }
    newStr += "\n";
  }

  // include .cu files for normal kernels
  if (compilerOptions.emitCUDA() && !compilerOptions.exploreConfig()) {
    for (llvm::DenseMap<ValueDecl *, HipaccKernel *>::iterator
        it=KernelDeclMap.begin(), ei=KernelDeclMap.end(); it!=ei; ++it) {
      HipaccKernel *Kernel = it->second;

      newStr += "#include \"";
      newStr += Kernel->getFileName();
      newStr += ".cu\"\n";
    }
  }
  // include .cu files for global reduction kernels
  if (compilerOptions.emitCUDA() && !compilerOptions.exploreConfig()) {
    for (unsigned int i=0, e=InvokedReductions.size(); i!=e; ++i) {
      HipaccGlobalReduction *GR = InvokedReductions.data()[i];

      if (!GR->isPrinted()) {
        newStr += "#include \"";
        newStr += GR->getFileName();
        newStr += ".cu\"\n";
        GR->setIsPrinted(true);
      }
    }
  }
  // write constant memory declarations
  if (compilerOptions.emitCUDA()) {
    for (llvm::DenseMap<ValueDecl *, HipaccMask *>::iterator
        it=MaskDeclMap.begin(), ei=MaskDeclMap.end(); it!=ei; ++it) {
      HipaccMask *Mask = it->second;
      if (Mask->isPrinted()) continue;

      newStr += "__device__ __constant__ ";
      newStr += Mask->getTypeStr();
      newStr += " " + Mask->getName();
      newStr += "[" + Mask->getSizeYStr() + "][" + Mask->getSizeXStr() +
        "];\n";
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
  if (!compilerOptions.emitCUDA() && !compilerOptions.exploreConfig()) {
    for (llvm::DenseMap<ValueDecl *, HipaccKernel *>::iterator
        it=KernelDeclMap.begin(), ei=KernelDeclMap.end(); it!=ei; ++it) {
      HipaccKernel *Kernel = it->second;

      stringCreator.writeKernelCompilation(Kernel->getFileName(), initStr);
    }
    initStr += "\n    ";
  }
  // load OpenCL reduction kernel files and compile the OpenCL reduction kernels
  if (!compilerOptions.emitCUDA() && !compilerOptions.exploreConfig()) {
    for (unsigned int i=0, e=InvokedReductions.size(); i!=e; ++i) {
      HipaccGlobalReduction *GR = InvokedReductions.data()[i];

      if (!GR->isPrinted()) {
        stringCreator.writeKernelCompilation(GR->getFileName(), initStr, "2D");
        stringCreator.writeKernelCompilation(GR->getFileName(), initStr, "1D");
        GR->setIsPrinted(true);
      }
    }
  }

  // write Mask transfers to Symbol in CUDA
  if (compilerOptions.emitCUDA()) {
    for (llvm::DenseMap<ValueDecl *, HipaccMask *>::iterator
        it=MaskDeclMap.begin(), ei=MaskDeclMap.end(); it!=ei; ++it) {
      HipaccMask *Mask = it->second;
      std::string newStr;

      if (!compilerOptions.exploreConfig()) {
        stringCreator.writeMemoryTransferSymbol(Mask, Mask->getHostMemName(),
            HOST_TO_DEVICE, newStr);
      }

      Expr *E = Mask->getHostMemExpr();
      if (E) {
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
      }
    }
  }

  // insert initialization before first statement
  CompoundStmt::body_iterator BI = CS->body_begin();
  Stmt *S = *BI;
  TextRewriter.InsertTextBefore(S->getLocStart(), initStr);

  // get buffer of main file id. If we haven't changed it, then we are done.
  if (const RewriteBuffer *RewriteBuf =
      TextRewriter.getRewriteBufferFor(mainFileID)) {
    Out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    llvm::errs() << "No changes to input file, something went wrong!\n";
  }
  Out.flush();
}


bool Rewrite::HandleTopLevelDecl(DeclGroupRef DGR) {
  for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end(); I != E; ++I) {
    Decl *D = *I;

    if (compilerClasses.HipaccEoP) {
      // skip late template class instantiations when templated class instances
      // are created. this is the case if the expansion location is not within
      // the main file
      if (SM->getFileID(SM->getExpansionLoc(D->getLocation()))!=mainFileID)
        continue;
    }
    TraverseDecl(D);
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

  if (D->getTagKind() == TTK_Class) {
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
        if (D->getNameAsString() == "AccessorNN")
          compilerClasses.AccessorNN = D;
        if (D->getNameAsString() == "AccessorLF")
          compilerClasses.AccessorLF = D;
        if (D->getNameAsString() == "AccessorCF")
          compilerClasses.AccessorCF = D;
        if (D->getNameAsString() == "AccessorL3")
          compilerClasses.AccessorL3 = D;
        if (D->getNameAsString() == "IterationSpaceBase")
          compilerClasses.IterationSpaceBase = D;
        if (D->getNameAsString() == "IterationSpace")
          compilerClasses.IterationSpace = D;
        if (D->getNameAsString() == "ElementIterator")
          compilerClasses.ElementIterator = D;
        if (D->getNameAsString() == "Kernel") compilerClasses.Kernel = D;
        if (D->getNameAsString() == "GlobalReduction")
          compilerClasses.GlobalReduction = D;
        if (D->getNameAsString() == "Mask") compilerClasses.Mask = D;
        if (D->getNameAsString() == "HipaccEoP") compilerClasses.HipaccEoP = D;
      }
    }

    if (!compilerClasses.HipaccEoP) return true;

    HipaccKernelClass *KC = NULL;

    for (CXXRecordDecl::base_class_iterator I=D->bases_begin(),
        E=D->bases_end(); I!=E; ++I) {

      // found user kernel class
      if (compilerClasses.isTypeOfTemplateClass(I->getType(),
            compilerClasses.Kernel)) {
        KC = new HipaccKernelClass(D->getNameAsString());
        KernelClassDeclMap[D] = KC;
        // remove user kernel class (semicolon doesn't count to SourceRange)
        SourceLocation startLoc = D->getLocStart();
        SourceLocation endLoc = D->getLocEnd();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *endBuf = SM->getCharacterData(endLoc);
        const char *semiPtr = strchr(endBuf, ';');
        TextRewriter.RemoveText(startLoc, semiPtr-startBuf+1, TextRewriteOptions);

        break;
      }
    }

    if (!KC) return true;

    // find constructor
    CXXConstructorDecl *CCD = NULL;
    for (CXXRecordDecl::ctor_iterator I=D->ctor_begin(), E=D->ctor_end(); I!=E;
        ++I) {
      CXXConstructorDecl *CCDI = *I;

      if (CCDI->isCopyConstructor()) continue;

      CCD = CCDI;
    }
    assert(CCD && "Couldn't find user kernel class constructor!");


    // iterate over constructor initializers
    for (FunctionDecl::param_iterator I=CCD->param_begin(), E=CCD->param_end();
        I!=E; ++I) {
      ParmVarDecl *PVD = *I;

      // constructor initializer represent the parameters for the kernel. Match
      // constructor parameter with constructor initializer since the order may
      // differ, e.g.
      // kernel(int a, int b) : b(a), a(b) {}
      for (CXXConstructorDecl::init_iterator II=CCD->init_begin(),
          EE=CCD->init_end(); II!=EE; ++II) {
        CXXCtorInitializer *CBOMI =*II;

        // CBOMI->isMemberInitializer()
        if (isa<DeclRefExpr>(CBOMI->getInit()->IgnoreParenCasts())) {
          DeclRefExpr *DRE =
            dyn_cast<DeclRefExpr>(CBOMI->getInit()->IgnoreParenCasts());

          if (DRE->getDecl() == PVD) {
            FieldDecl *FD = CBOMI->getMember();

            // reference to Image variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Image)) {

              QualType QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addImgArg(FD, QT, FD->getName());

              break;
            }

            // reference to Accessor variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Accessor)) {

              QualType QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addImgArg(FD, QT, FD->getName());

              break;
            }

            // reference to Mask variable ?
            if (compilerClasses.isTypeOfTemplateClass(FD->getType(),
                  compilerClasses.Mask)) {

              QualType QT = compilerClasses.getFirstTemplateType(FD->getType());
              KC->addMaskArg(FD, QT, FD->getName());

              break;
            }

            // normal variable
            KC->addArg(FD, FD->getType(), FD->getName());

            break;
          }
        }

        // CBOMI->isBaseInitializer()
        if (isa<CXXConstructExpr>(CBOMI->getInit())) {
          CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(CBOMI->getInit());
          assert(CCE->getNumArgs() == 1 &&
              "Kernel base class constructor requires exactly one argument!");

          if (isa<DeclRefExpr>(CCE->getArg(0))) {
            DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));
            if (DRE->getDecl() == PVD) {
              // skip IterationSpace declaration, the iteration space is set on
              // the host side and an Accessor is created for the image
              // associated with the IterationSpace when the IterationSpace
              // declaration is found.
              break;
            }
          }
        }
      }
    }

    // search for kernel function
    for (CXXRecordDecl::method_iterator I=D->method_begin(), E=D->method_end();
        I!=E; ++I) {
      CXXMethodDecl *FD = *I;

      // kernel function
      if (FD->getNameAsString() == "kernel") {
        // set kernel method
        KC->setKernelFunction(FD);

        // define analysis context used for different checkers
        AnalysisDeclContext AC(/* AnalysisDeclContextManager */ 0, FD);
        KernelStatistics::setAnalysisOptions(AC);

        // create kernel analysis pass, execute it and store it to kernel class
        KernelStatistics *stats = KernelStatistics::create(AC, D->getName(),
            compilerClasses);
        KC->setKernelStatistics(stats);

        break;
      }
    }
  }

  if (D->getTagKind() == TTK_Struct) {
    if (!compilerClasses.HipaccEoP) return true;

    for (CXXRecordDecl::base_class_iterator I=D->bases_begin(),
        E=D->bases_end(); I!=E; ++I) {

      // found user global reduction class
      if (compilerClasses.isTypeOfTemplateClass(I->getType(),
            compilerClasses.GlobalReduction)) {
        HipaccGlobalReductionClass *GRC = new
          HipaccGlobalReductionClass(D->getNameAsString());
        GlobalReductionClassDeclMap[D] = GRC;

        // search for reduce function
        for (CXXRecordDecl::method_iterator I=D->method_begin(),
            E=D->method_end(); I!=E; ++I) {
          CXXMethodDecl *FD = *I;

          // reduce function
          if (FD->getNameAsString() == "reduce") {
            // set reduce method
            GRC->setReductionFunction(FD);
          }
        }
        assert(GRC->getReductionFunction() && "No reduction function found!");

        // remove user kernel class (semicolon doesn't count to SourceRange)
        SourceLocation startLoc, endLoc;
        if (D->getDescribedClassTemplate()) {
          startLoc = SM->getSpellingLoc(D->getDescribedClassTemplate()->getLocStart());
          endLoc = SM->getSpellingLoc(D->getDescribedClassTemplate()->getLocEnd());
        } else {
          startLoc = SM->getSpellingLoc(D->getLocStart());
          endLoc = SM->getSpellingLoc(D->getLocEnd());
        }
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *endBuf = SM->getCharacterData(endLoc);
        const char *semiPtr = strchr(endBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, "");

        break;
      }
    }
  }

  return true;
}


bool Rewrite::VisitDeclStmt(DeclStmt *D) {
  if (!compilerClasses.HipaccEoP) return true;

  // a) convert Image declarations into memory allocations, e.g.
  //    Image<int> IN(width, height);
  //    =>
  //    int *IN = hipaccCreateMemory<int>(NULL, width, height, &stride, padding);
  // b) save BoundaryCondition declarations, e.g.
  //    BoundaryCondition<int> BcIN(IN, 5, 5, BOUNDARY_MIRROR);
  // c) save Accessor declarations, e.g.
  //    Accessor<int> AccIN(BcIN);
  // d) save Mask declarations, e.g.
  //    Mask<float> M(2*sigma_d+1, 2*sigma_d+1);
  // d) save user kernel declarations, and replace it by kernel compilation
  //    for OpenCL, e.g.
  //    AddKernel K(IS, IN, OUT, 23);
  //    - create CUDA/OpenCL kernel AST by replacing accesses to Image data by
  //      global memory access and by replacing references to class member
  //      variables by kernel parameter variables.
  //    - print the CUDA/OpenCL kernel to a file.
  // e) save IterationSpace declarations, e.g.
  //    IterationSpace<int> VIS(OUT, width, height);
  // f) save GlobalReduction declarations, e.g.
  //    MinReduction<float> redMinIN(IN, FLT_MAX);
  for (DeclStmt::decl_iterator DI=D->decl_begin(), DE=D->decl_end(); DI!=DE;
      ++DI) {
    Decl *SD = *DI;

    if (SD->getKind() == Decl::Var) {
      VarDecl *VD = dyn_cast<VarDecl>(SD);

      // found Image decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Image)) {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert(CCE->getNumArgs() == 2 && "Image definition requires exactly two arguments!");

        HipaccImage *Img = new HipaccImage(Context, VD);

        std::string newStr;
        std::string pitchStr;

        // get the text string for the image width
        std::string widthStr;
        llvm::raw_string_ostream WS(widthStr);
        CCE->getArg(0)->printPretty(WS, 0, PrintingPolicy(CI.getLangOpts()));
        Img->setWidth(WS.str());
        Img->setWidthType(CCE->getArg(0)->getType().getAsString());

        // get the text string for the image height
        std::string heightStr;
        llvm::raw_string_ostream HS(heightStr);
        CCE->getArg(1)->printPretty(HS, 0, PrintingPolicy(CI.getLangOpts()));
        Img->setHeight(HS.str());
        Img->setHeightType(CCE->getArg(1)->getType().getAsString());

        // create memory allocation string
        stringCreator.writeMemoryAllocation(VD->getName(),
            compilerClasses.getFirstTemplateType(VD->getType()).getAsString(),
            WS.str(), HS.str(), pitchStr, newStr, targetDevice);

        // rewrite Image definition
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        // store Image definition
        Img->setPixelType(compilerClasses.getFirstTemplateType(VD->getType()));
        Img->setStride(pitchStr);
        ImgDeclMap[VD] = Img;

        break;
      }

      // found BoundaryCondition decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.BoundaryCondition)) {
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
            "Expected BoundaryCondition definition (CXXConstructExpr).");
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        HipaccBoundaryCondition *BC = NULL;
        HipaccImage *Img = NULL;

        // check if the first argument is an Image
        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the Image from the DRE if we have one
          if (ImgDeclMap.count(DRE->getDecl())) {
            Img = ImgDeclMap[DRE->getDecl()];
          }
        }
        assert(Img && "Expected Image as first argument to BoundaryCondition.");

        BC = new HipaccBoundaryCondition(Img, VD);

        // get text string for arguments, argument order is:
        // img, size_x, size_y, mode
        // img, size, mode
        // img, mask, mode
        // img, size_x, size_y, mode, const_val
        // img, size, mode, const_val
        // img, mask, mode, const_val
        Expr::EvalResult constVal;
        unsigned int DiagIDConstant =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Constant expression or Mask object for %ordinal0 parameter to BoundaryCondition %1 required.");
        unsigned int DiagIDNoMode =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Boundary handling mode for BoundaryCondition %0 required.");
        unsigned int DiagIDWrongMode =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Wrong boundary handling mode for BoundaryCondition %0 specified.");
        unsigned int DiagIDMode =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Boundary handling constant for BoundaryCondition %0 required.");


        unsigned int found_size = 0;
        bool found_mode = false;
        // get kernel window size
        for (unsigned int i=1, e=CCE->getNumArgs(); i!=e; ++i) {
          if (found_size == 0) {
            // check if the parameter is a Mask reference
            if (isa<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts())) {
              DeclRefExpr *DRE =
                dyn_cast<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts());

              // get the Mask from the DRE if we have one
              if (MaskDeclMap.count(DRE->getDecl())) {
                HipaccMask *Mask = MaskDeclMap[DRE->getDecl()];
                BC->setSizeX(Mask->getSizeX());
                BC->setSizeY(Mask->getSizeY());
                found_size++;
                found_size++;

                continue;
              }
            }

            // check if the parameter can be resolved to a constant
            if (!CCE->getArg(i)->isEvaluatable(Context)) {
              Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDConstant) << i+1
                << VD->getName();
            }
            BC->setSizeX(CCE->getArg(i)->EvaluateKnownConstInt(Context).getSExtValue());
            found_size++;
          } else {
            // check if the parameter specifies the boundary mode
            if (isa<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts())) {
              DeclRefExpr *DRE =
                dyn_cast<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts());
              // boundary mode found
              if (DRE->getDecl()->getKind() == Decl::EnumConstant &&
                  DRE->getDecl()->getType().getAsString() ==
                  "enum hipacc::hipaccBoundaryMode") {
                int64_t mode =
                  CCE->getArg(i)->EvaluateKnownConstInt(Context).getSExtValue();
                switch (mode) {
                  case BOUNDARY_UNDEFINED:
                  case BOUNDARY_CLAMP:
                  case BOUNDARY_REPEAT:
                  case BOUNDARY_MIRROR:
                    BC->setBoundaryHandling((BoundaryMode)mode);
                    break;
                  case BOUNDARY_CONSTANT:
                    BC->setBoundaryHandling((BoundaryMode)mode);
                    if (CCE->getNumArgs() != i+2) {
                      Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDMode) <<
                        VD->getName();
                    }
                    // check if the parameter can be resolved to a constant
                    if (!CCE->getArg(i+1)->isEvaluatable(Context)) {
                      Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDConstant)
                        << i+2 << VD->getName();
                    }
                    CCE->getArg(i+1)->EvaluateAsRValue(constVal, Context);
                    BC->setConstVal(constVal.Val, Context);
                    i++;
                    break;
                  default:
                    BC->setBoundaryHandling(BOUNDARY_UNDEFINED);
                    llvm::errs() << "invalid boundary handling mode specified, using default mode!\n";
                }
                found_mode = true;

                continue;
              }
            }

            if (found_size >= 2) {
              if (found_mode) {
                Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDWrongMode) <<
                  VD->getName();
              } else {
                Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDNoMode) <<
                  VD->getName();
              }
            }

            // check if the parameter can be resolved to a constant
            if (!CCE->getArg(i)->isEvaluatable(Context)) {
              Diags.Report(CCE->getArg(i)->getExprLoc(), DiagIDConstant) << i+1
                << VD->getName();
            }
            BC->setSizeY(CCE->getArg(i)->EvaluateKnownConstInt(Context).getSExtValue());
            found_size++;
          }
        }

        // store BoundaryCondition
        BCDeclMap[VD] = BC;

        // remove BoundaryCondition definition
        TextRewriter.RemoveText(D->getSourceRange());

        break;
      }

      // found Accessor decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Accessor) ||
          compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.AccessorNN) || 
          compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.AccessorLF) || 
          compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.AccessorCF) || 
          compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.AccessorL3)) {
        assert(VD->hasInit() && "Currently only Accessor definitions are supported, no declarations!");
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
            "Currently only Accessor definitions are supported, no declarations!");
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        std::string newStr;
        HipaccAccessor *Acc = NULL;
        HipaccBoundaryCondition *BC = NULL;

        // check if the first argument is an Image
        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the BoundaryCondition from the DRE if we have one
          if (BCDeclMap.count(DRE->getDecl())) {
            BC = BCDeclMap[DRE->getDecl()];
          }

          // in case we have no BoundaryCondition, check if an Image is
          // specified and construct a BoundaryCondition
          if (!BC && ImgDeclMap.count(DRE->getDecl())) {
            HipaccImage *Img = ImgDeclMap[DRE->getDecl()];
            BC = new HipaccBoundaryCondition(Img, VD);
            BC->setSizeX(1);
            BC->setSizeY(1);
            BC->setBoundaryHandling(BOUNDARY_CLAMP);

            // Fixme: store BoundaryCondition???
            BCDeclMap[VD] = BC;
          }
        }
        assert(BC && "Expected BoundaryCondition or Image as first argument to Accessor.");

        InterpolationMode mode;
        if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
              compilerClasses.Accessor)) mode = InterpolateNO;
        else if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
              compilerClasses.AccessorNN)) mode = InterpolateNN;
        else if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
              compilerClasses.AccessorLF)) mode = InterpolateLF;
        else if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
              compilerClasses.AccessorCF)) mode = InterpolateCF;
        else mode = InterpolateL3;

        Acc = new HipaccAccessor(BC, mode, VD);

        // get text string for arguments, argument order is:
        // img|bc
        // img|bc, width, height, xf, yf
        for (unsigned int i=1; i < 5; i++) {
          std::string Str, Var, Decl;
          llvm::raw_string_ostream SS(Str);

          if (i < CCE->getNumArgs()) {
            CCE->getArg(i)->printPretty(SS, 0,
                PrintingPolicy(CI.getLangOpts()));
            Decl = CCE->getArg(i)->getType().getAsString();
          } else if (i < 3) {
            // get width/height from image
            if (i==1) {
              SS << Acc->getImage()->getWidth();
              Decl = Acc->getImage()->getWidthType();
            } else {
              SS << Acc->getImage()->getHeight();
              Decl = Acc->getImage()->getHeightType();
            }
          } else {
            // omit offset_x and offset_y if not specified
            Acc->setNoCrop();
            break;
          }

          switch (i) {
            case 1:
              Var = "_" + VD->getNameAsString() + "width";
              Acc->setWidth(Var);
              Acc->setWidthType(Decl);
              break;
            case 2:
              Var = "_" + VD->getNameAsString() + "height";
              Acc->setHeight(Var);
              Acc->setHeightType(Decl);
              break;
            case 3:
              Var = "_" + VD->getNameAsString() + "offset_x";
              Acc->setOffsetX(Var);
              Acc->setOffsetXType(Decl);
              break;
            case 4:
              Var = "_" + VD->getNameAsString() + "offset_y";
              Acc->setOffsetY(Var);
              Acc->setOffsetYType(Decl);
              break;
            default:
              llvm::errs() << "Only four arguments for Accessor supported\n";
          }

          Decl += " " + Var + " = " + SS.str() + ";\n    ";
          newStr += Decl;
        }

        // replace Accessor decl by variables for width/height and offsets
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
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

        std::string newStr;
        std::stringstream LSS;
        LSS << isLiteralCount++;

        HipaccIterationSpace *IS = NULL;
        HipaccImage *Img = NULL;

        // check if the first argument is an Image
        if (isa<DeclRefExpr>(CCE->getArg(0))) {
          DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

          // get the Image from the DRE if we have one
          if (ImgDeclMap.count(DRE->getDecl())) {
            Img = ImgDeclMap[DRE->getDecl()];
          }
        }
        assert(Img && "Expected Image as first argument to IterationSpace.");

        IS = new HipaccIterationSpace(Img, VD);

        // get text string for arguments, argument order is:
        // img[, is_width, is_height[, offset_x, offset_y]]
        for (unsigned int i=1; i < 5; i++) {
          std::string Str, Var, Decl;
          llvm::raw_string_ostream SS(Str);

          if (i < CCE->getNumArgs()) {
            CCE->getArg(i)->printPretty(SS, 0,
                PrintingPolicy(CI.getLangOpts()));
            Decl = CCE->getArg(i)->getType().getAsString();
          } else {
            if (i==1) {
              // use width of image
              SS << Img->getWidth();
              Decl = Img->getWidthType();
            } else if (i==2) {
              // use height of image
              SS << Img->getHeight();
              Decl = Img->getHeightType();
            } else {
              // omit offset_x and offset_y if not specified
              break;
            }
          }

          switch (i) {
            case 1:
              Var = "_is_width_" + LSS.str();
              IS->setWidth(Var);
              IS->setWidthType(Decl);
              break;
            case 2:
              Var = "_is_height_" + LSS.str();
              IS->setHeight(Var);
              IS->setHeightType(Decl);
              break;
            case 3:
              Var = "_is_offset_x_" + LSS.str();
              IS->setOffsetX(Var);
              IS->setOffsetXType(Decl);
              break;
            case 4:
              Var = "_is_offset_y_" + LSS.str();
              IS->setOffsetY(Var);
              IS->setOffsetYType(Decl);
              break;
            default:
              llvm::errs() << "Only four arguments for IterationSpace supported\n";
          }

          Decl += " " + Var + " = " + SS.str() + ";\n    ";
          newStr += Decl;
        }

        // store IterationSpace
        ISDeclMap[VD] = IS;

        // replace iteration space decl by variables for width/height, and
        // offset
        // get the start location and compute the semi location.
        SourceLocation startLoc = D->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        break;
      }

      // found Mask decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Mask)) {
        assert(VD->hasInit() && "Currently only Mask definitions are supported, no declarations!");
        assert(isa<CXXConstructExpr>(VD->getInit()) &&
            "Currently only Mask definitions are supported, no declarations!");

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        assert((CCE->getNumArgs() == 2) &&
            "Mask definition requires exactly two arguments!");

        HipaccMask *Mask = new HipaccMask(VD);
        Mask->setType(compilerClasses.getFirstTemplateType(VD->getType()));

        Expr::EvalResult constVal;
        unsigned int DiagIDConstant =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
              "Constant expression for %ordinal0 parameter to Mask %1 required.");

        // check if the parameters can be resolved to a constant
        if (!CCE->getArg(0)->isEvaluatable(Context)) {
          Diags.Report(CCE->getArg(0)->getExprLoc(), DiagIDConstant) << 1 <<
            VD->getName();
        }
        Mask->setSizeX(CCE->getArg(0)->EvaluateKnownConstInt(Context).getSExtValue());

        if (!CCE->getArg(1)->isEvaluatable(Context)) {
          Diags.Report(CCE->getArg(1)->getExprLoc(), DiagIDConstant) << 2 <<
            VD->getName();
        }
        Mask->setSizeY(CCE->getArg(1)->EvaluateKnownConstInt(Context).getSExtValue());

        if (compilerOptions.emitCUDA() || Mask->isConstant()) {
          // remove Mask definition
          TextRewriter.RemoveText(D->getSourceRange());
        } else {
          std::string newStr, pitchStr;
          // create Buffer for Mask
          stringCreator.writeMemoryAllocationConstant(Mask->getName(),
              Mask->getTypeStr(), Mask->getSizeXStr(), Mask->getSizeYStr(),
              pitchStr, newStr);

          // replace Mask declaration by Buffer allocation
          // get the start location and compute the semi location.
          SourceLocation startLoc = D->getLocStart();
          const char *startBuf = SM->getCharacterData(startLoc);
          const char *semiPtr = strchr(startBuf, ';');
          TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
        }

        // store Mask definition
        MaskDeclMap[VD] = Mask;

        break;
      }

      // found GlobalReduction decl
      for (llvm::DenseMap<RecordDecl *, HipaccGlobalReductionClass *>::iterator
          it=GlobalReductionClassDeclMap.begin(),
          ei=GlobalReductionClassDeclMap.end(); it!=ei; ++it) {

        CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(it->first);
        if (compilerClasses.isTypeOfTemplateClass(VD->getType(), RD)) {
          assert(isa<CXXConstructExpr>(VD->getInit()) &&
              "Expected GlobalReduction definition (CXXConstructExpr).");
          CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

          HipaccGlobalReduction *GR = NULL;
          HipaccImage *Img = NULL;
          HipaccAccessor *Acc = NULL;

          // check if the first argument is an Image or Accessor
          if (isa<DeclRefExpr>(CCE->getArg(0))) {
            DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0));

            // get the Image from the DRE if we have one
            if (ImgDeclMap.count(DRE->getDecl())) {
              Img = ImgDeclMap[DRE->getDecl()];
            }

            // get the Accessor from the DRE if we have one
            if (AccDeclMap.count(DRE->getDecl())) {
              Acc = AccDeclMap[DRE->getDecl()];
            }
          }
          assert((Img || Acc) && "Expected an Image or Accessor as first argument to GlobalReduction.");
          assert(CCE->getNumArgs()==2 && "Expected exactly two arguments to GlobalReduction.");

          // create Accessor if none was provided
          if (!Acc) {
            HipaccBoundaryCondition *BC = new HipaccBoundaryCondition(Img, VD);
            BC->setSizeX(0);
            BC->setSizeY(0);
            BC->setBoundaryHandling(BOUNDARY_UNDEFINED);

            Acc = new HipaccAccessor(BC, InterpolateNO, VD);
          }

          // create GlobalReduction
          GR = new HipaccGlobalReduction(Acc, VD, it->second, compilerOptions,
              !Img);

          // get the string representation of the neutral element
          std::string neutralStr;
          llvm::raw_string_ostream NS(neutralStr);
          CCE->getArg(1)->printPretty(NS, 0, PrintingPolicy(CI.getLangOpts()));
          GR->setNeutral(NS.str());

          // get the template specialization type
          QualType QT = compilerClasses.getFirstTemplateType(VD->getType());
          GR->setType(QT.getAsString());

          // store GlobalReduction
          GlobalReductionDeclMap[VD] = GR;

          // remove GlobalReduction definition
          TextRewriter.RemoveText(D->getSourceRange());
        }
      }

      // found Kernel decl
      if (VD->getType().getTypePtr()->getTypeClass() == Type::Record) {
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
          assert(VD->hasInit() && "Currently only Kernel definitions are supported, no declarations!");
          assert(isa<CXXConstructExpr>(VD->getInit()) &&
              "Currently only Image definitions are supported, no declarations!");
          CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

          unsigned int num_img = 0, num_mask = 0;
          SmallVector<FieldDecl *, 16> imgFields = KC->getImgFields();
          SmallVector<FieldDecl *, 16> maskFields = KC->getMaskFields();
          for (unsigned int i=0; i<CCE->getNumArgs(); i++) {
            if (isa<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts())) {
              DeclRefExpr *DRE =
                dyn_cast<DeclRefExpr>(CCE->getArg(i)->IgnoreParenCasts());

              // check if we have an Image
              if (ImgDeclMap.count(DRE->getDecl())) {
                unsigned int DiagIDImage =
                  Diags.getCustomDiagID(DiagnosticsEngine::Error,
                      "Images are not supported within kernels, use Accessors instead:");
                Diags.Report(DRE->getLocation(), DiagIDImage);
              }

              // check if we have an Accessor
              if (AccDeclMap.count(DRE->getDecl())) {
                K->insertMapping(imgFields.data()[num_img],
                    AccDeclMap[DRE->getDecl()]);
                num_img++;
                continue;
              }

              // check if we have a Mask
              if (MaskDeclMap.count(DRE->getDecl())) {
                K->insertMapping(maskFields.data()[num_mask],
                    MaskDeclMap[DRE->getDecl()]);
                num_mask++;
                continue;
              }

              // check if we have an IterationSpace
              if (ISDeclMap.count(DRE->getDecl())) {
                K->setIterationSpace(ISDeclMap[DRE->getDecl()]);
                continue;
              }
            }
          }

          // create function declaration for kernel
          StringRef kernelName;
          std::string name;
          if (compilerOptions.emitCUDA()) {
            name = "cu" + KC->getName() + VD->getNameAsString();
          } else {
            name = "cl" + KC->getName() + VD->getNameAsString();
          }
          kernelName = StringRef(name);
          std::string filename = KC->getName() + VD->getNameAsString();
          K->setFileName(filename);


          #ifdef USE_JIT_ESTIMATE
          // write kernel file to estimate resource usage. The constants for
          // boundary handling are set later on.

          // kernel declaration for CUDA
          FunctionDecl *kernelDeclEst = createFunctionDecl(Context,
              Context.getTranslationUnitDecl(), kernelName, Context.VoidTy,
              K->getNumArgs(), K->getArgTypes(Context,
                compilerOptions.getTargetCode()), K->getArgNames());

          // create kernel body
          ASTTranslate *HipaccEst = new ASTTranslate(Context, kernelDeclEst, K,
              KC, builtins, compilerOptions, true);
          Stmt *kernelStmtsEst =
            HipaccEst->Hipacc(KC->getKernelFunction()->getBody());
          kernelDeclEst->setBody(kernelStmtsEst);

          // write kernel to file
          printKernelFunction(kernelDeclEst, KC, K, filename, false);

          // compile kernel in order to get resource usage
          std::string command = K->getCompileCommand(compilerOptions.emitCUDA())
            + K->getCompileOptions(kernelName.str(), filename,
                compilerOptions.emitCUDA());

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
              info = "isa info : Used %d gprs, %d bytes lds, stack size: %d";
            } else {
              info = "ptxas info : Used %d registers";
            }
          }
          while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
            lines.push_back(std::string(line));
            if (targetDevice.isAMDGPU()) {
              sscanf(line, info.c_str(), &reg, &smem, &lmem);
            } else {
              char *ptr = line;
              int num_read = 0, val1 = 0, val2 = 0;
              char mem_type = 'x';

              if (compilerOptions.getTargetDevice() >= FERMI_20) {
                // scan for stack size (shared memory)
                num_read = sscanf(ptr, "%d bytes %1c tack frame", &val1, &mem_type);

                if (num_read == 2 && mem_type == 's') {
                  smem = val1;
                  llvm::errs() << "stack size: " << val1 << "\n";
                  continue;
                }
              }

              num_read = sscanf(line, info.c_str(), &reg);
              if (!num_read) continue;

              while ((ptr = strchr(ptr, ','))) {
                ptr++;
                num_read = sscanf(ptr, "%d+%d bytes %1c mem", &val1, &val2,
                    &mem_type);
                if (num_read == 3) {
                  switch (mem_type) {
                    default:
                      llvm::errs() << "wrong memory specifier '" << mem_type <<
                        "': " << ptr;
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
                } else {
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
                  } else {
                    llvm::errs() << "Unexpected memory usage specification: '"
                      << ptr;
                  }
                }
              }
            }
          }
          pclose(fpipe);

          if (reg == 0) {
            unsigned int DiagIDCompile =
              Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                  "Compiling kernel in file '%0.cu' failed, using default kernel configuration:\n%1");
            Diags.Report(DiagIDCompile) << K->getFileName() << command.c_str();
            for (unsigned int i=0, e=lines.size(); i!=e; ++i) {
              llvm::errs() << lines.data()[i];
            }
          } else {
            if (targetDevice.isAMDGPU()) {
              llvm::errs() << "Resource usage for kernel '" << kernelName <<
                "': " << reg << " gprs, " << lmem << " bytes stack, " << smem <<
                " bytes lds\n";
            } else {
              llvm::errs() << "Resource usage for kernel '" << kernelName <<
                "': " << reg << " registers, " << lmem << " bytes lmem, " <<
                smem << " bytes smem, " << cmem << " bytes cmem\n";
            }
          }

          K->setResourceUsage(reg, lmem, smem, cmem);
          #else
          K->setDefaultConfig();
          #endif


          // kernel declaration
          FunctionDecl *kernelDecl = createFunctionDecl(Context,
              Context.getTranslationUnitDecl(), kernelName, Context.VoidTy,
              K->getNumArgs(), K->getArgTypes(Context,
                compilerOptions.getTargetCode()), K->getArgNames());

          // write CUDA/OpenCL kernel function to file clone old body,
          // replacing member variables
          ASTTranslate *Hipacc = new ASTTranslate(Context, kernelDecl, K, KC,
              builtins, compilerOptions);
          Stmt *kernelStmts = Hipacc->Hipacc(KC->getKernelFunction()->getBody());
          kernelDecl->setBody(kernelStmts);
          K->printStats();

          // write kernel to file
          printKernelFunction(kernelDecl, KC, K, filename, true);


          #ifdef USE_POLLY
          if (!compilerOptions.exploreConfig()) {
            // create kernel declaration for Polly
            FunctionDecl *kernelDeclPolly = createFunctionDecl(Context,
                Context.getTranslationUnitDecl(), kernelName, Context.VoidTy,
                K->getNumArgs(), K->getArgTypes(Context, TARGET_C),
                K->getArgNames());

            // call Polly ...
            ASTTranslate *HipaccPolly = new ASTTranslate(Context, kernelDeclPolly,
                K, KC, builtins, compilerOptions, false, true);
            Stmt *pollyStmts =
              HipaccPolly->Hipacc(KC->getKernelFunction()->getBody());
            kernelDeclPolly->setBody(pollyStmts);
            llvm::errs() << "\nPassing the following function to Polly:\n";
            kernelDeclPolly->print(llvm::errs(), Context.getPrintingPolicy());
            llvm::errs() << "\n";

            Polly *polly_analysis = new Polly(Context, CI, kernelDeclPolly);
            polly_analysis->analyzeKernel();
          }
          #endif

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
  // e.g. IN = host_array;
  if (E->getOperator() == OO_Equal) {
    if (E->getNumArgs() != 2) return true;

    if (isa<DeclRefExpr>(E->getArg(0))) {
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getArg(0));

      // check if we have an Image
      if (ImgDeclMap.count(DRE->getDecl())) {
        HipaccImage *Img = ImgDeclMap[DRE->getDecl()];
        std::string newStr;

        // get the text string for the memory transfer src
        std::string dataStr;
        llvm::raw_string_ostream DS(dataStr);
        E->getArg(1)->printPretty(DS, 0, PrintingPolicy(CI.getLangOpts()));

        // create memory transfer string
        stringCreator.writeMemoryTransfer(Img, DS.str(), HOST_TO_DEVICE,
            newStr);

        // rewrite Image assignment to memory transfer
        // get the start location and compute the semi location.
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        return true;
      }

      // check if we have a Mask
      if (MaskDeclMap.count(DRE->getDecl())) {
        HipaccMask *Mask = MaskDeclMap[DRE->getDecl()];
        std::string newStr;

        // get the text string for the memory transfer src
        std::string dataStr;
        llvm::raw_string_ostream DS(dataStr);
        E->getArg(1)->printPretty(DS, 0, PrintingPolicy(CI.getLangOpts()));

        DeclRefExpr *DREVar =
          dyn_cast<DeclRefExpr>(E->getArg(1)->IgnoreParenCasts());
        if (DREVar) {
          if (VarDecl *V = dyn_cast<VarDecl>(DREVar->getDecl())) {
            bool isMaskConstant = V->getType().isConstant(Context);

            if (!isMaskConstant) {
              unsigned int DiagIDConstant =
                Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                    "Mask '%0': Constant propagation only supported if coefficient array is declared as constant.");
              Diags.Report(V->getLocation(), DiagIDConstant) << Mask->getName();
            }

            // loop over initializers and check if each initializer is a
            // constant
            if (isMaskConstant && isa<InitListExpr>(V->getInit())) {
              InitListExpr *ILE = dyn_cast<InitListExpr>(V->getInit());
              Mask->setInitList(ILE);

              for (unsigned int i=0; i<ILE->getNumInits(); ++i) {
                if (!ILE->getInit(i)->isConstantInitializer(Context, false)) {
                  isMaskConstant = false;
                  break;
                }
              }
            }
            Mask->setIsConstant(isMaskConstant);

            // create memory transfer string for memory upload to constant
            // memory
            if (!isMaskConstant) {
              if (compilerOptions.emitCUDA()) {
                // store memory pointer and source location. The string is
                // replaced later on.
                Mask->setHostMemName(V->getName());
                Mask->setHostMemExpr(E);

                return true;
              } else {
                stringCreator.writeMemoryTransferSymbol(Mask, V->getName(),
                    HOST_TO_DEVICE, newStr);
              }
            }
          }
        }

        // rewrite Mask assignment to memory transfer
        // get the start location and compute the semi location.
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);

        return true;
      }
    }
  }

  return true;
}


bool Rewrite::VisitBinaryOperator(BinaryOperator *E) {
  if (!compilerClasses.HipaccEoP) return true;

  // convert Image assignments to a variable into memory transfer,
  // e.g. in_ptr = IN.getData();
  if (isa<CXXMemberCallExpr>(E->getRHS())) {
    CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(E->getRHS());

    // match only getData calls to Image instances
    if (MCE->getDirectCallee()->getNameAsString() != "getData") return true;

    if (isa<DeclRefExpr>(MCE->getImplicitObjectArgument())) {
      DeclRefExpr *DRE =
        dyn_cast<DeclRefExpr>(MCE->getImplicitObjectArgument());

      // check if we have an Image
      if (ImgDeclMap.count(DRE->getDecl())) {
        HipaccImage *Img = ImgDeclMap[DRE->getDecl()];

        std::string newStr;

        // get the text string for the memory transfer dst
        std::string dataStr;
        llvm::raw_string_ostream DS(dataStr);
        E->getLHS()->printPretty(DS, 0, PrintingPolicy(CI.getLangOpts()));

        // create memory transfer string
        stringCreator.writeMemoryTransfer(Img, DS.str(), DEVICE_TO_HOST,
            newStr);

        // rewrite Image assignment to memory transfer
        // get the start location and compute the semi location.
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
      }
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
  // b) convert invocation of 'reduce' member function into kernel launch, e.g.
  //    float min = MinReduction.reduce();

  if (E->getImplicitObjectArgument() &&
      isa<DeclRefExpr>(E->getImplicitObjectArgument()->IgnoreParenCasts())) {

    DeclRefExpr *DRE =
      dyn_cast<DeclRefExpr>(E->getImplicitObjectArgument()->IgnoreParenCasts());
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

        // constructor includes the iteration space -> -1
        assert(CCE->getNumArgs()-1==K->getKernelClass()->getNumArgs() &&
            "number of arguments doesn't match!");

        // set host argument names and retrieve literals stored to temporaries
        K->setHostArgNames(llvm::makeArrayRef(CCE->getArgs(),
              CCE->getNumArgs()), newStr, literalCount);

        // create kernel call string
        stringCreator.writeKernelCall(K->getKernelName() + K->getName(),
            K->getArgTypeNames(), K->getHostArgNames(), K->getKernelClass(), K,
            newStr);

        // rewrite kernel invocation
        // get the start location and compute the semi location.
        SourceLocation startLoc = E->getLocStart();
        const char *startBuf = SM->getCharacterData(startLoc);
        const char *semiPtr = strchr(startBuf, ';');
        TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
      }
    }

    // match reduce calls to user global reduction instances
    if (!GlobalReductionDeclMap.empty() &&
        E->getDirectCallee()->getNameAsString() == "reduce") {

      // store the reduction call expression - the specialized method bodies are
      // created before the translation unit is handled; reduction kernels are
      // generated then
      ReductionCalls.push_back(E);
    }
  }

  return true;
}


void Rewrite::generateReductionKernels() {
  for (unsigned int i=0; i<ReductionCalls.size(); i++) {
    CXXMemberCallExpr *E = ReductionCalls.data()[i];

    DeclRefExpr *DRE =
      dyn_cast<DeclRefExpr>(E->getImplicitObjectArgument()->IgnoreParenCasts());

    // get the user global reduction class
    if (GlobalReductionDeclMap.count(DRE->getDecl())) {
      std::string newStr;
      HipaccGlobalReduction *GR = GlobalReductionDeclMap[DRE->getDecl()];

      if (!GR->getReductionFunction()) {
        VarDecl *VD = GR->getDecl();

        if (VD->getType().getTypePtr()->getTypeClass() ==
            Type::TemplateSpecialization) {
          const TemplateSpecializationType *TST =
            cast<TemplateSpecializationType>(VD->getType());

          // get late instantiated class
          RecordDecl *RD = TST->getAsStructureType()->getDecl();
          CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD);

          for (CXXRecordDecl::method_iterator I=CXXRD->method_begin(),
              E=CXXRD->method_end(); I!=E; ++I) {
            CXXMethodDecl *FD = *I;

            if (FD->getNumParams()==2 && FD->getNameAsString() == "reduce") {
              std::string filename = GR->getReductionClass()->getName() + VD->getNameAsString();
              GR->setFileName(filename);
              GR->setReductionFunction(FD);

              // generate kernel code
              printReductionFunction(FD, GR, filename);

              // store global reduction as invoked (required to include header
              // later on)
              InvokedReductions.push_back(GR);

              break;
            }
          }
        }
      }

      // create kernel call string
      stringCreator.writeGlobalReductionCall(GR, newStr);

      // rewrite reduction invocation
      // get the start location and compute the semi location.
      SourceLocation startLoc = E->getLocStart();
      const char *startBuf = SM->getCharacterData(startLoc);
      const char *semiPtr = strchr(startBuf, ';');
      TextRewriter.ReplaceText(startLoc, semiPtr-startBuf+1, newStr);
    }
  }
}


void Rewrite::printReductionFunction(FunctionDecl *D, HipaccGlobalReduction *GR,
    std::string file) {
  PrintingPolicy Policy = Context.getPrintingPolicy();
  if (dump) Policy.DumpSourceManager = SM;
  Policy.Indentation = 2;
  Policy.SuppressSpecifiers = false;
  Policy.SuppressTag = false;
  Policy.SuppressScope = false;
  Policy.ConstantArraySizeAsWritten = false;
  Policy.AnonymousTagLocations = true;
  if (compilerOptions.emitCUDA()) {
    Policy.LangOpts.CUDA = 1;
  } else{
    Policy.LangOpts.OpenCL = 1;
  }

  int fd;
  std::string filename = file;
  if (compilerOptions.emitCUDA()) {
    filename += ".cu";
  } else {
    filename += ".cl";
  }

  // open file stream using own file descriptor. We need to call fsync() to
  // compile the generated code using nvcc afterwards.
  while ((fd = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0664)) < 0) {
    if (errno != EINTR) {
      std::string errorInfo = "Error opening output file '" + filename + "'";
      perror(errorInfo.c_str());
    }
  }
  llvm::raw_fd_ostream kernelOut(fd, false);

  // write ifndef, ifdef
  std::string ifdef = "_" + file + "_";
  if (compilerOptions.emitCUDA()) {
    ifdef += "CU_";
  } else {
    ifdef += "CL_";
  }
  std::transform(ifdef.begin(), ifdef.end(), ifdef.begin(), ::toupper); 
  kernelOut << "#ifndef " + ifdef + "\n";
  kernelOut << "#define " + ifdef + "\n\n";

  if (!compilerOptions.exploreConfig()) {
    kernelOut << "#define BS " << GR->getNumThreads() << "\n"
              << "#define PPT " << GR->getPixelsPerThread() << "\n";
  }
  if (GR->isAccessor()) {
    kernelOut << "#define USE_OFFSETS\n";
  }
  if (compilerOptions.emitCUDA() && compilerOptions.getTargetDevice()>=FERMI_20
      && compilerOptions.useTextureMemory() &&
      compilerOptions.getTextureType()==Array2D) {
    kernelOut << "#define USE_ARRAY_2D\n";
  }
  if (compilerOptions.emitCUDA()) {
    kernelOut << "#include \"hipacc_cuda_red.hpp\"\n\n";
  } else {
    kernelOut << "#include \"hipacc_ocl_red.hpp\"\n\n";
  }


  // write kernel name and qualifiers
  if (compilerOptions.emitCUDA()) {
    kernelOut << "extern \"C\" {\n";
    kernelOut << "__device__ ";
  }
  kernelOut << "inline "
            << D->getResultType().getAsString() << " "
            << GR->getName() << "Reduce(";
  // write kernel parameters
  unsigned int comma = 0;
  for (unsigned int i=0, e=D->getNumParams(); i!=e; ++i) {
    std::string Name = D->getParamDecl(i)->getNameAsString();

    // normal arguments
    if (comma++) kernelOut << ", ";
    QualType T = D->getParamDecl(i)->getType();
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
      T = Parm->getOriginalType();
    T.getAsStringInternal(Name, Policy);
    kernelOut << Name;
  }
  kernelOut << ") ";

  // print kernel body
  D->getBody()->printPretty(kernelOut, 0, Policy, 0);

  // instantiate reduction
  if (compilerOptions.emitCUDA()) {
    // print 2D CUDA array definition - this is only required on FERMI and if
    // Array2D is selected, but doesn't harm otherwise
    kernelOut << "texture<" << D->getResultType().getAsString()
              << ", cudaTextureType2D, cudaReadModeElementType> _tex"
              << GR->getAccessor()->getImage()->getName() + GR->getName()
              << ";\n\n";
    if (compilerOptions.getTargetDevice()>=FERMI_20 &&
        !compilerOptions.exploreConfig()) {
      kernelOut << "__device__ unsigned int finished_blocks_cu" <<
        GR->getFileName() << "2D = 0;\n\n";
      kernelOut << "REDUCTION_CUDA_2D_THREAD_FENCE(cu";
    } else {
      kernelOut << "REDUCTION_CUDA_2D(cu";
    }
    kernelOut << GR->getFileName() << "2D, "
              << D->getResultType().getAsString() << ", "
              << GR->getName() << "Reduce, _tex"
              << GR->getAccessor()->getImage()->getName() + GR->getName()
              << ")\n";
  } else {
    if (compilerOptions.useTextureMemory()) {
      kernelOut << "REDUCTION_OCL_2D_IMAGE(cl";
    } else {
      kernelOut << "REDUCTION_OCL_2D(cl";
    }
    kernelOut << GR->getFileName() << "2D, "
              << D->getResultType().getAsString() << ", "
              << GR->getName() << "Reduce";
    if (compilerOptions.useTextureMemory()) {
      kernelOut << ", " << GR->getAccessor()->getImage()->getImageReadFunction();
    }
    kernelOut << ")\n";
  }

  if (compilerOptions.emitCUDA() && compilerOptions.getTargetDevice() >=
      FERMI_20 && !compilerOptions.exploreConfig()) {
    // no second step required
  } else {
    if (compilerOptions.emitCUDA()) {
      kernelOut << "REDUCTION_CUDA_1D(cu";
    } else {
      kernelOut << "REDUCTION_OCL_1D(cl";
    }
    kernelOut << GR->getFileName() << "1D, "
              << D->getResultType().getAsString() << ", "
              << GR->getName() << "Reduce)\n";
  }

  if (compilerOptions.emitCUDA()) {
    kernelOut << "}\n";
  }
  kernelOut << "#include \"hipacc_undef.hpp\"\n";

  kernelOut << "\n";
  kernelOut << "#endif //" + ifdef + "\n";
  kernelOut << "\n";
  kernelOut.flush();
  fsync(fd);
  close(fd);
}


void Rewrite::printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
    HipaccKernel *K, std::string file, bool emitHints) {
  PrintingPolicy Policy = Context.getPrintingPolicy();
  if (dump) Policy.DumpSourceManager = SM;
  Policy.Indentation = 2;
  Policy.SuppressSpecifiers = false;
  Policy.SuppressTag = false;
  Policy.SuppressScope = false;
  Policy.ConstantArraySizeAsWritten = false;
  Policy.AnonymousTagLocations = true;
  if (compilerOptions.emitCUDA()) {
    Policy.LangOpts.CUDA = 1;
  } else{
    Policy.LangOpts.OpenCL = 1;
  }

  int fd;
  std::string filename = file;
  if (compilerOptions.emitCUDA()) {
    filename += ".cu";
  } else {
    filename += ".cl";
  }

  // open file stream using own file descriptor. We need to call fsync() to
  // compile the generated code using nvcc afterwards.
  while ((fd = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0664)) < 0) {
    if (errno != EINTR) {
      std::string errorInfo = "Error opening output file '" + filename + "'";
      perror(errorInfo.c_str());
    }
  }
  llvm::raw_fd_ostream kernelOut(fd, false);

  // write ifndef, ifdef
  std::string ifdef = "_" + file + "_";
  if (compilerOptions.emitCUDA()) {
    ifdef += "CU_";
  } else {
    ifdef += "CL_";
  }
  std::transform(ifdef.begin(), ifdef.end(), ifdef.begin(), ::toupper); 
  kernelOut << "#ifndef " + ifdef + "\n";
  kernelOut << "#define " + ifdef + "\n\n";

  if (compilerOptions.emitCUDA() && K->vectorize()) {
    kernelOut << "#include \"hipacc_cuda_vec.hpp\"\n\n";
  }

  bool inc=false;
  for (unsigned int i=0, e=KC->getNumImages(); i!=e; ++i) {
    FieldDecl *FD = KC->getImgFields().data()[i];
    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    
    if (Acc->getInterpolation()!=InterpolateNO) {
      if (!inc) {
        inc = true;
        if (compilerOptions.emitCUDA()) {
          if (!emitHints) {
            kernelOut << "#include \"hipacc_cuda_interpolate.hpp\"\n\n";
          }
        } else {
          kernelOut << "#include \"hipacc_ocl_interpolate.hpp\"\n\n";
        }
      }

      // define required interpolation mode
      if (inc && Acc->getInterpolation()>InterpolateNN) {
        std::string function_name = ASTTranslate::getInterpolationName(Context,
            builtins, compilerOptions, K, Acc, border_variant());
        std::string suffix = "_" +
          builtins.EncodeTypeIntoStr(Acc->getImage()->getPixelQualType(),
              Context);

        std::string resultStr;
        stringCreator.writeInterpolationDefinition(K, Acc, function_name,
            suffix, Acc->getInterpolation(), Acc->getBoundaryHandling(),
            resultStr);
        if (compilerOptions.emitCUDA()) {
          if (!emitHints) {
            kernelOut << resultStr;
          } else {
            // emit interpolation definitions at the beginning at the file
            InterpolationDefinitions.push_back(resultStr);
          }
        } else {
          kernelOut << resultStr;
        }

        resultStr.erase();
        stringCreator.writeInterpolationDefinition(K, Acc, function_name,
            suffix, InterpolateNO, BOUNDARY_UNDEFINED, resultStr);

        if (compilerOptions.emitCUDA()) {
          if (!emitHints) {
            kernelOut << resultStr;
          } else {
            // emit interpolation definitions at the beginning at the file
            InterpolationDefinitions.push_back(resultStr);
          }
        } else {
          kernelOut << resultStr;
        }
      }
    }
  }

  if (compilerOptions.emitCUDA()) {
    // write texture declarations
    for (unsigned int i=0, e=KC->getNumImages(); i!=e; ++i) {
      FieldDecl *FD = KC->getImgFields().data()[i];
      HipaccAccessor *Acc = K->getImgFromMapping(FD);

      if (KC->getImgAccess(FD) == READ_ONLY && K->useTextureMemory(Acc)) {
        QualType T = Acc->getImage()->getPixelQualType();

        kernelOut << "texture<";
        kernelOut << T.getAsString();
        switch (K->useTextureMemory(Acc)) {
          default:
          case Linear1D:
            kernelOut << ", cudaTextureType1D, cudaReadModeElementType> _tex";
            break;
          case Linear2D:
          case Array2D:
            kernelOut << ", cudaTextureType2D, cudaReadModeElementType> _tex";
            break;
        }
        kernelOut << FD->getNameAsString() << K->getName() << ";\n";
      }
    }

    // write surface declaration
    if (compilerOptions.useTextureMemory() &&
        compilerOptions.getTextureType()==Array2D) {
      kernelOut << "surface<void, cudaSurfaceType2D> _surfOutput";
      kernelOut << K->getName() << ";\n";
    }
    kernelOut << "\n";
  }

  // write constant memory declarations
  for (unsigned int i=0; i<KC->getNumMasks(); i++) {
    FieldDecl *FD = KC->getMaskFields().data()[i];
    HipaccMask *Mask = K->getMaskFromMapping(FD);

    if (Mask->isConstant()) {
      if (compilerOptions.emitCUDA()) {
        kernelOut << "__device__ __constant__ ";
      } else {
        kernelOut << "__constant ";
      }
      kernelOut << Mask->getTypeStr();
      kernelOut << " " << Mask->getName() << K->getName();
      kernelOut << "[" << Mask->getSizeYStr() << "][" << Mask->getSizeXStr() <<
        "] = {\n";

      InitListExpr *ILE = Mask->getInitList();
      unsigned int num_init = 0;

      // print Mask constant literals to 2D array
      for (unsigned int j=0; j<Mask->getSizeY(); j++) {
        kernelOut << "        {";
        for (unsigned int k=0; k<Mask->getSizeX(); k++) {
          ILE->getInit(num_init++)->printPretty(kernelOut, 0, Policy, 0);
          if (k<Mask->getSizeX()-1) {
            kernelOut << ", ";
          }
        }
        if (j<Mask->getSizeY()-1) {
          kernelOut << "},\n";
        } else {
          kernelOut << "}\n";
        }
      }
      kernelOut << "    };\n\n";
      Mask->setIsPrinted(true);
    } else {
      // mask is not constant. Emit declaration in CUDA, for OpenCL nothing has
      // to be printed - the Mask will be added as kernel parameter
      if (compilerOptions.emitCUDA()) {
        kernelOut << "__device__ __constant__ ";
        kernelOut << Mask->getTypeStr();
        kernelOut << " " << Mask->getName() << K->getName();
        kernelOut << "[" << Mask->getSizeYStr() << "][" << Mask->getSizeXStr()
          << "];\n\n";
        Mask->setIsPrinted(true);
      }
    }
  }

  // write kernel name and qualifiers
  if (compilerOptions.emitCUDA()) {
    kernelOut << "extern \"C\" {\n";
    kernelOut << "__global__ void ";
    if (!compilerOptions.exploreConfig() && emitHints) kernelOut << "__launch_bounds__ (" <<
      K->getNumThreadsX() << "*" << K->getNumThreadsY() << ") ";
  } else {
    if (compilerOptions.useTextureMemory() &&
        compilerOptions.getTextureType()==Array2D) {
      kernelOut << "__constant sampler_t "
        << D->getNameInfo().getAsString() << "Sampler = "
        << "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n\n";
    }
    kernelOut << "__kernel ";
    if (!compilerOptions.exploreConfig() && emitHints) kernelOut << "__attribute__((reqd_work_group_size(" <<
      K->getNumThreadsX() << ", " << K->getNumThreadsY() << ", 1))) ";
    kernelOut << "void ";
  }
  kernelOut << D->getNameInfo().getAsString();
  kernelOut << "(";

  // write kernel parameters
  unsigned int comma = 0;
  for (unsigned int i=0, e=D->getNumParams(); i!=e; ++i) {
    std::string Name = D->getParamDecl(i)->getNameAsString();

    // check if we have a Mask
    FieldDecl *FD = K->getArgFields()[i];
    HipaccMask *Mask = K->getMaskFromMapping(FD);
    if (Mask) {
      if (compilerOptions.emitCUDA() || Mask->isConstant()) {
        // skip mask parameter in CUDA, mask is declared as constant memory
      } else {
        if (comma++) kernelOut << ", ";
        kernelOut << "__constant ";
        QualType T = D->getParamDecl(i)->getType();
        if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
          T = Parm->getOriginalType();
        T.getAsStringInternal(Name, Policy);
        kernelOut << Name;
        kernelOut << " __attribute__ ((max_constant_size (" <<
          Mask->getSizeXStr() << "*" << Mask->getSizeYStr() <<
          "*sizeof(" << Mask->getTypeStr() << "))))";
      }
      continue;
    }

    // check if we have an Accessor
    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    MemoryAccess memAcc = UNDEFINED;
    if (i==0) { // first argument is always the output image
      if (compilerOptions.emitCUDA() && compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType()==Array2D) continue;
      Acc = K->getIterationSpace()->getAccessor();
      memAcc = WRITE_ONLY;
    } else if (Acc) {
      memAcc = KC->getImgAccess(FD);
    }

    if (Acc) {
      if (compilerOptions.emitCUDA()) {
        if (K->useTextureMemory(Acc) && memAcc==READ_ONLY) {
          // no parameter is emitted for textures
          continue;
        } else {
          if (comma++) kernelOut << ", ";
          QualType T = D->getParamDecl(i)->getType();
          if (memAcc==READ_ONLY && !T.isLocalConstQualified()) {
            kernelOut << "const ";
          }
          if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
            T = Parm->getOriginalType();
          T.getAsStringInternal(Name, Policy);
        }
      } else {
        // __global keyword to specify memory location is only needed for OpenCL
        if (comma++) kernelOut << ", ";
        if (K->useTextureMemory(Acc)) {
          if (memAcc==WRITE_ONLY) {
            kernelOut << "__write_only image2d_t ";
          } else {
            kernelOut << "__read_only image2d_t ";
          }
        } else {
          kernelOut << "__global ";
          QualType T = D->getParamDecl(i)->getType();
          if (memAcc==READ_ONLY && !T.isLocalConstQualified()) {
            kernelOut << "const ";
          }
          if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
            T = Parm->getOriginalType();
          T.getAsStringInternal(Name, Policy);
        }
      }
      kernelOut << Name;
      continue;
    }

    // normal arguments
    if (comma++) kernelOut << ", ";
    QualType T = D->getParamDecl(i)->getType();
    if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
      T = Parm->getOriginalType();
    T.getAsStringInternal(Name, Policy);
    kernelOut << Name;

    // default arguments ...
    if (Expr *Init = D->getParamDecl(i)->getInit()) {
      CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init);
      if (!CCE || CCE->getConstructor()->isCopyConstructor()) {
        kernelOut << " = ";
      }
      Init->printPretty(kernelOut, 0, Policy, 0);
    }
  }
  kernelOut << ") ";

  // print kernel body
  D->getBody()->printPretty(kernelOut, 0, Policy, 0);
  if (compilerOptions.emitCUDA()) {
    kernelOut << "}\n";
  }
  kernelOut << "\n";
  kernelOut << "#endif //" + ifdef + "\n";
  kernelOut << "\n";
  kernelOut.flush();
  fsync(fd);
  close(fd);
}

// vim: set ts=2 sw=2 sts=2 et ai:

