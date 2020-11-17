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
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/AST/ASTFuse.h"
#include "hipacc/Analysis/HostDataDeps.h"
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
#include <regex>
#include <fstream>

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

    // Analysis
    HostDataDeps *dataDeps;
    ASTFuse *kernelFuser;
    std::string cudaGraphStr;
    std::map<std::string, bool> outputImagesVisitorMap_;

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
    bool skipString;

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
      dataDeps(nullptr),
      kernelFuser(nullptr),
      cudaGraphStr("graph_"),
      compilerClasses(CompilerKnownClasses()),
      mainFD(nullptr),
      literalCount(0),
      skipTransfer(false),
      skipString(false)
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

    static void WriteAuxiliaryInterpolationDeclHeader(SmallVector<std::string, 16>& interpolations, char const* file_name, std::string const& target_include);

    // Rewrite
    std::string convertToString(Stmt *from) {
      hipacc_require(from != nullptr, "Expected non-null Stmt");
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
        case clang::hipacc::Language::CUDA:
          LO.CUDA = 1; break;
        case clang::hipacc::Language::OpenCLACC:
        case clang::hipacc::Language::OpenCLCPU:
        case clang::hipacc::Language::OpenCLGPU:
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

  std::unique_ptr<llvm::raw_pwrite_stream> OS = CI.createOutputFile(out, false, true, "", "", false);
  hipacc_require(OS != nullptr, "Cannot create output stream.");

  return std::make_unique<Rewrite>(CI, options, std::move(OS));
}


void Rewrite::WriteAuxiliaryInterpolationDeclHeader(SmallVector<std::string, 16>& interpolations, char const* file_name, std::string const& target_include) {
    std::ofstream interpol_hdr_stream(file_name);

    interpol_hdr_stream << "#pragma once" << std::endl
                        << target_include << std::endl << std::endl;

    // sort definitions and remove duplicate definitions
    std::sort(interpolations.begin(), interpolations.end(), std::greater<std::string>());
    interpolations.erase(std::unique(interpolations.begin(), interpolations.end()), interpolations.end());

    // add interpolation definitions
    for (auto str : interpolations)
      interpol_hdr_stream << str;

    interpol_hdr_stream << std::endl;
}

void Rewrite::HandleTranslationUnit(ASTContext &) {
  hipacc_require(compilerClasses.Coordinate, "Coordinate class not found!");
  hipacc_require(compilerClasses.Image, "Image class not found!");
  hipacc_require(compilerClasses.BoundaryCondition, "BoundaryCondition class not found!");
  hipacc_require(compilerClasses.AccessorBase, "AccessorBase class not found!");
  hipacc_require(compilerClasses.Accessor, "Accessor class not found!");
  hipacc_require(compilerClasses.IterationSpaceBase, "IterationSpaceBase class not found!");
  hipacc_require(compilerClasses.IterationSpace, "IterationSpace class not found!");
  hipacc_require(compilerClasses.ElementIterator, "ElementIterator class not found!");
  hipacc_require(compilerClasses.Kernel, "Kernel class not found!");
  hipacc_require(compilerClasses.Mask, "Mask class not found!");
  hipacc_require(compilerClasses.Domain, "Domain class not found!");
  hipacc_require(compilerClasses.Pyramid, "Pyramid class not found!");
  hipacc_require(compilerClasses.HipaccEoP, "HipaccEoP class not found!");

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

  // include .cu or .h files for normal kernels
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case clang::hipacc::Language::C99:
      for (auto map : KernelDeclMap) {
        newStr += "#include \"";
        newStr += map.second->getFileName();
        newStr += ".cc\"\n";
      }
      break;
    case clang::hipacc::Language::CUDA:
      if (compilerOptions.fuseKernels()) {
        for (auto map : KernelDeclMap) {
          if (!dataDeps->isFusible(map.second)) {
            newStr += "#include \"";
            newStr += map.second->getFileName();
            newStr += ".cu\"\n";
          }
        }
        for (auto fName : kernelFuser->getFusedFileNamesAll()) {
          newStr += "#include \"";
          newStr += fName;
          newStr += ".cu\"\n";
        }
      } else {
        for (auto map : KernelDeclMap) {
          newStr += "#include \"";
          newStr += map.second->getFileName();
          newStr += ".cu\"\n";
        }
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
  hipacc_require(mainFD, "no function with attribute 'HIPACC_CODEGEN' found!");

  CompoundStmt *CS = dyn_cast<CompoundStmt>(mainFD->getBody());
  hipacc_require(CS->size(), "CompoundStmt has no statements.", &Diags, mainFD->getLocation());

  std::string initStr, cleanupStr;

  // get initialization string for run-time

  bool skip_write_init{};

  if (mainFD->hasAttrs()) {
    std::string attr_annotate("annotate");
    std::string attr_hipacc_no_rt_init("annotate(\"hipacc_no_rt_init\")");

    for (auto attr : mainFD->getAttrs()) {
      std::string attr_name(attr->getSpelling());

      if (attr_name == attr_annotate) {
        std::string SS;
        llvm::raw_string_ostream S(SS);
        attr->printPretty(S, Policy);
        std::string attr_string(S.str());

        if (attr_string.find(attr_hipacc_no_rt_init) != std::string::npos)
            skip_write_init = true;
      }
    }
  }

  if(!skip_write_init)
      stringCreator.writeInitialization(initStr);

  // write graph global decl in CUDA
  if (compilerOptions.emitCUDA() && compilerOptions.useGraph()) {
    initStr += "\n" + stringCreator.getIndent();
    initStr += "// cuda graph decl\n" + stringCreator.getIndent();
    initStr += "cudaGraph_t " + cudaGraphStr + ";\n" + stringCreator.getIndent();
    initStr += "cudaGraphCreate(&" + cudaGraphStr + ", 0);\n" + stringCreator.getIndent();
    initStr += "cudaGraphExec_t " + cudaGraphStr + "exec_;\n" + stringCreator.getIndent();
    initStr += "cudaStream_t " + cudaGraphStr + "stream_;\n" + stringCreator.getIndent();
    initStr += "cudaStreamCreate(&" + cudaGraphStr + "stream_);\n" + stringCreator.getIndent();
    // graph node dep and args
    std::string depString("");
    initStr += "// cuda graph node, dependency and arguments decl\n" + stringCreator.getIndent();
    for (auto GMap : dataDeps->getGraphNodeDepMap()) {
      std::string nodeName = GMap.first;
      std::string nodeDepName = nodeName + "dep_";
      initStr += "cudaGraphNode_t " + nodeName + ";\n" + stringCreator.getIndent();
      initStr += "std::vector<cudaGraphNode_t> " + nodeDepName + ";\n" + stringCreator.getIndent();
      std::string nodeArgName = nodeName + "arg_";
      bool isMemcpyNode = (nodeName.find("_H2D_") != std::string::npos)||(nodeName.find("_D2H_") != std::string::npos);
      std::string nodeArgTypeStr = isMemcpyNode ? "cudaMemcpy3DParms" : "cudaKernelNodeParams";
      initStr += nodeArgTypeStr + " " + nodeArgName + " = {0};\n" + stringCreator.getIndent();
    }
  }

  // load OpenCL kernel files and compile the OpenCL kernels
  for (auto map : KernelDeclMap)
    stringCreator.writeKernelCompilation(map.second, initStr);
  initStr += "\n" + stringCreator.getIndent();

  // write Mask transfers to Symbol in CUDA
  if (compilerOptions.emitCUDA()) {
    for (auto map : MaskDeclMap) {
      auto mask = map.second;

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

  // insert initialization before first statement
  TextRewriter.InsertTextBefore(CS->body_front()->getBeginLoc(), initStr);

  // insert cleanup before last statement
  if (compilerOptions.emitCUDA() && compilerOptions.useGraph()) {
    cleanupStr += "// CUDA Graph clean up\n" + stringCreator.getIndent();
    cleanupStr += "cudaGraphExecDestroy(" + cudaGraphStr + "exec_);\n" + stringCreator.getIndent();
    cleanupStr += "cudaGraphDestroy(" + cudaGraphStr + ");\n" + stringCreator.getIndent();
    cleanupStr += "cudaStreamDestroy(" + cudaGraphStr + "stream_);\n" + stringCreator.getIndent();
  }
  TextRewriter.InsertTextBefore(CS->body_back()->getBeginLoc(), cleanupStr);

  // get buffer of main file id. If we haven't changed it, then we are done.
  if (auto RewriteBuf = TextRewriter.getRewriteBufferFor(mainFileID)) {
    std::string output_file_str(RewriteBuf->begin(), RewriteBuf->end());

    // remove all CR characters which are automatically added by clang on Windows to get the CRLF line endings
    output_file_str.erase(
      std::remove(output_file_str.begin(), output_file_str.end(), '\r')
      , output_file_str.end());

    *Out << output_file_str;
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

      // if (SM.getFileID(SM.getExpansionLoc(decl->getLocation()))!=mainFileID)
      //   continue;
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
    hipacc_require(CCD, "Couldn't find user kernel class constructor!", &Diags, D->getLocation());


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
          hipacc_require(CCE->getNumArgs() == 1,
              "Kernel base class constructor requires exactly one argument!", &Diags, CCE->getLocation());

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
        KC->setKernelFunction(method, compilerClasses, compilerOptions.printVerbose());
        continue;
      }

      // reduce function
      if (method->getNameAsString() == "reduce") {
        KC->setReduceFunction(method);
        //TODO: cuda graph is not supported for reduce function
        if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
          compilerOptions.setUseGraph(OFF);
        }
        continue;
      }

      // binning function
      if (method->getNameAsString() == "binning") {
        KC->setBinningFunction(method);
        //TODO: cuda graph is not supported for binning function
        if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
          compilerOptions.setUseGraph(OFF);
        }
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
  //    => HipaccImage* IN = hipaccCreateMemory<int>(data, width, height, &stride, padding);
  //    Image<int> IN(<any single expression>);
  //    => HipaccImage* IN = hipaccMapMemory<int>(<any single expression>);
  // b) convert Pyramid declarations into pyramid creation, e.g.
  //    Pyramid<int> P(IN, 3);
  //    => Pyramid P = hipaccCreatePyramid<int>(IN, 3);
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
    if (!decl->isUsed()) {
      // do not parse template code that has not been instanciated yet.
      // remove unused image decl
      if (decl->getKind() == Decl::Var) {
        VarDecl *VD = dyn_cast<VarDecl>(decl);
        if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Image)) {
          replaceText(D->getBeginLoc(), D->getEndLoc(), ';', "// unused image decl");
        }
      }
      continue;
    }

    if (decl->getKind() == Decl::Var) {
      VarDecl *VD = dyn_cast<VarDecl>(decl);

      // found Image decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Image)) {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());

        if (CCE == nullptr) {
          // In case image constructor is called with function as parameter
          // e.g. Image<ushort> image(converter::get_hipacc_image());
          ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(VD->getInit());
          if (EWC != nullptr) {
            CCE = dyn_cast<CXXConstructExpr>(EWC->getSubExpr());
          }
        }

        hipacc_require(CCE != nullptr, "Not a constructor expression of hipacc::Image", &Diags, VD->getLocation());
        hipacc_require(CCE->getConstructor() != nullptr, "Missing constructor declaration of hipacc::Image", &Diags, VD->getLocation());
        hipacc_require(CCE->getConstructor()->hasAttrs(), "Missing constructor attribute of hipacc::Image", &Diags, VD->getLocation());

        std::string constructor_type{};

        for(auto attrib: CCE->getConstructor()->getAttrs()) {
          if(attrib->getKind() != attr::Annotate)
            continue;

          constructor_type = cast<AnnotateAttr>(attrib)->getAnnotation();
          break;
        }

        HipaccImage *Img = new HipaccImage(Context, VD,
            compilerClasses.getFirstTemplateType(VD->getType()));

        if(constructor_type == "ArrayAssignment")
        {
          hipacc_require(CCE->getNumArgs() == 4, "Image ArrayAssignment constructor is expected to have four arguments", &Diags, CCE->getLocation());

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

          //  int64_t img_stride = CCE->getArg(0)->EvaluateKnownConstInt(Context).getSExtValue();
          //  int64_t img_height = CCE->getArg(1)->EvaluateKnownConstInt(Context).getSExtValue();

          //  if (compilerOptions.emitPadding()) {
          //    // respect alignment/padding for constantly sized CPU images
          //    int64_t alignment = compilerOptions.getAlignment()
          //                          / (Context.getTypeSize(Img->getType())/8);

          //    if (alignment > 1) {
          //      img_stride = ((img_stride+alignment-1) / alignment) * alignment;
          //    }
          //  }

          //  Img->setSizeX(img_stride);
          //  Img->setSizeY(img_height);
          //}

          // host memory
          std::string init_str = convertToString(CCE->getArg(2));
          std::string deep_copy_str = convertToString(CCE->getArg(3));

          // create memory allocation string
          std::string newStr;

          if(!init_str.empty()) {
            std::string h2dTransferStr;
            // decouple image H2D transfer from decl
            stringCreator.writeMemoryAllocation(Img, width_str, height_str, "", deep_copy_str, newStr);
            if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
              std::string nodeName = dataDeps->getGraphMemcpyNodeName(Img->getName(), init_str, "H2D");
              std::string nodeDepStr(nodeName + "dep_");
              std::string nodeArgStr(nodeName + "arg_");
              stringCreator.addMemoryTransferGraph(Img, init_str, HOST_TO_DEVICE, cudaGraphStr,
                                                   nodeName, nodeDepStr, nodeArgStr, h2dTransferStr);
              // add dependencies for other node
              h2dTransferStr += "\n" + stringCreator.getIndent();
              for (auto dStr : dataDeps->getGraphMemcpyNodeDepOn(Img->getName(), init_str, "H2D")) {
                h2dTransferStr += dStr + "dep_.push_back(" + nodeName + ");\n" + stringCreator.getIndent();
              }
            } else {
              stringCreator.writeMemoryTransfer(Img, init_str, HOST_TO_DEVICE, h2dTransferStr);
            }
            newStr += "\n" + stringCreator.getIndent();
            newStr += h2dTransferStr;
            newStr += "\n" + stringCreator.getIndent();
          } else {
            stringCreator.writeMemoryAllocation(Img, width_str, height_str,
                init_str, deep_copy_str, newStr);
          }

          // rewrite Image definition
          replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);
        } else if (constructor_type == "CustomImage") {
          hipacc_require(CCE->getNumArgs() >= 1, "Image constructor expects at least one argument", &Diags, CCE->getLocation());

          std::string newStr{};
          std::string assigned_image = convertToString(CCE->getArg(0));

          stringCreator.writeMemoryMapping(Img, assigned_image, newStr);

          // rewrite Image definition
          replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);
        } else {
          // TODO: print error message
          hipacc_require(false, "Image constructor type not supported!", &Diags, CCE->getLocation());
        }

        // store Image definition
        ImgDeclMap[VD] = Img;

        break;
      }

      // found Pyramid decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Pyramid)) {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        hipacc_require(CCE->getNumArgs() == 2 || CCE->getNumArgs() == 1,
               "Pyramid definition requires one or two arguments!", &Diags, CCE->getLocation());

        HipaccPyramid *Pyr = new HipaccPyramid(Context, VD,
            compilerClasses.getFirstTemplateType(VD->getType()));

        std::string newStr;

        if(CCE->getNumArgs() == 1) {
          // get the text string for the pyramid image & depth
          std::string assigned_pyramid = convertToString(CCE->getArg(0));

          // create memory allocation string
          stringCreator.writePyramidMapping(VD->getName(),
              compilerClasses.getFirstTemplateType(VD->getType()).getAsString(),
              assigned_pyramid, newStr);
        }

        else {
          // get the text string for the pyramid image & depth
          std::string image_str = convertToString(CCE->getArg(0));
          std::string depth_str = convertToString(CCE->getArg(1));

          // create memory allocation string
          stringCreator.writePyramidAllocation(VD->getName(),
              compilerClasses.getFirstTemplateType(VD->getType()).getAsString(),
              image_str, depth_str, newStr);
        }

        // rewrite Pyramid definition
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

        // store Pyramid definition
        PyrDeclMap[VD] = Pyr;

        break;
      }

      // found BoundaryCondition decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.BoundaryCondition)) {
        hipacc_require(isa<CXXConstructExpr>(VD->getInit()),
               "Expected BoundaryCondition definition (CXXConstructExpr).", &Diags, VD->getLocation());
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
              hipacc_require(lval.isNonNegative() && lval.getZExtValue() <= cval, "invalid Boundary mode", &Diags, DRE->getLocation());
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

        hipacc_require((Img || Pyr), "Expected first argument of BoundaryCondition "
                                     "to be Image or Pyramid call.", &Diags, VD->getLocation());


        // remove BoundaryCondition definition
        TextRewriter.RemoveText(D->getSourceRange());

        break;
      }

      // found Accessor decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Accessor)) {

        CXXConstructExpr *CCE{};

        if(isa<ExprWithCleanups>(VD->getInit()))
        {
          auto* EWC = dyn_cast<ExprWithCleanups>(VD->getInit());

          if(isa<CXXConstructExpr>(EWC->getSubExpr()))
          {
            auto ewc_object = EWC->getSubExpr();

            if(isa<CXXConstructExpr>(ewc_object))
              CCE = dyn_cast<CXXConstructExpr>(ewc_object);
          }
        }

        else if(isa<CXXConstructExpr>(VD->getInit()))
        {
          CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        }

        hipacc_require(CCE != nullptr,
              "Expected Accessor definition (CXXConstructExpr)."
              , &Diags, VD->getLocation());

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
              hipacc_require(lval.isNonNegative() && lval.getZExtValue() <= cval,
                     "invalid Interpolate mode", &Diags, DRE->getLocation());
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

        hipacc_require(BC, "Expected BoundaryCondition, Image or Pyramid call as "
                     "first argument to Accessor.", &Diags, VD->getLocation());

        Acc = new HipaccAccessor(VD, BC, mode, roi_args == 4);

        std::string newStr;
        newStr = "auto " + Acc->getName() + " = hipaccMakeAccessor<" + Acc->getImage()->getTypeStr() + ">(" + parms + ");";

        // replace Accessor decl by variables for width/height and offsets
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

        // store Accessor definition
        AccDeclMap[VD] = Acc;

        break;
      }

      // found IterationSpace decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.IterationSpace)) {

        CXXConstructExpr *CCE{};

        if(isa<ExprWithCleanups>(VD->getInit()))
        {
          auto* EWC = dyn_cast<ExprWithCleanups>(VD->getInit());

          if(isa<CXXConstructExpr>(EWC->getSubExpr()))
          {
            auto ewc_object = EWC->getSubExpr();

            if(isa<CXXConstructExpr>(ewc_object))
              CCE = dyn_cast<CXXConstructExpr>(ewc_object);
          }
        }

        else if(isa<CXXConstructExpr>(VD->getInit()))
        {
          CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        }

        hipacc_require(CCE != nullptr,
              "Expected IterationSpace definition (CXXConstructExpr)."
              , &Diags, VD->getLocation());

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

        hipacc_require((Img || Pyr), "Expected first argument of IterationSpace to "
                                     "be Image or Pyramid call.", &Diags, VD->getLocation());

        IS = new HipaccIterationSpace(VD, Img ? Img : Pyr, roi_args == 4);
        if (Pyr)
          IS->getBC()->setPyramidIndex(pyr_idx);
        ISDeclMap[VD] = IS; // store IterationSpace

        std::string newStr;
        newStr = "auto " + IS->getName() + " = hipaccMakeAccessor<" + IS->getImage()->getTypeStr() + ">(" + parms + ");";

        // replace iteration space decl by variables for width/height and offset
        replaceText(D->getBeginLoc(), D->getEndLoc(), ';', newStr);

        break;
      }

      HipaccMask *Mask = nullptr;
      // found Mask decl
      if (compilerClasses.isTypeOfTemplateClass(VD->getType(),
            compilerClasses.Mask)) {
        hipacc_require(isa<CXXConstructExpr>(VD->getInit()),
               "Expected Mask definition (CXXConstructExpr).", &Diags, VD->getLocation());

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        hipacc_require((CCE->getNumArgs() == 1),
               "Mask definition requires exactly one argument!", &Diags, CCE->getLocation());

        QualType QT = compilerClasses.getFirstTemplateType(VD->getType());
        Mask = new HipaccMask(VD, QT, HipaccMask::MaskType::Mask);

        // get initializer
        DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0)->IgnoreParenCasts());
        hipacc_require(DRE, "Mask must be initialized using a variable", &Diags, CCE->getLocation());
        VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        hipacc_require(V, "Mask must be initialized using a variable", &Diags, DRE->getLocation());
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
        hipacc_require(isa<CXXConstructExpr>(VD->getInit()),
               "Expected Domain definition (CXXConstructExpr).", &Diags, VD->getLocation());

        Domain = new HipaccMask(VD, Context.UnsignedCharTy,
                                            HipaccMask::MaskType::Domain);

        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        if (CCE->getNumArgs() == 1) {
          // get initializer
          auto DRE = dyn_cast<DeclRefExpr>(CCE->getArg(0)->IgnoreParenCasts());
          hipacc_require(DRE, "Domain must be initialized using a variable", &Diags, CCE->getLocation());
          VarDecl *V = dyn_cast_or_null<VarDecl>(DRE->getDecl());
          hipacc_require(V, "Domain must be initialized using a variable", &Diags, DRE->getLocation());

          if (compilerClasses.isTypeOfTemplateClass(DRE->getType(),
                                                    compilerClasses.Mask)) {
            // copy from mask
            HipaccMask *Mask = MaskDeclMap[DRE->getDecl()];
            hipacc_require(Mask, "Mask to copy from was not declared", &Diags, DRE->getLocation());

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
                      hipacc_require(false,
                             "Expected integer literal in domain initializer", &Diags, ILEX->getExprLoc());
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
          hipacc_require(false, "Domain definition requires exactly two arguments "
              "type constant integer or a single argument of type uchar[][] or "
              "Mask!", &Diags, CCE->getExprLoc());
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
          if (compilerOptions.fuseKernels() && dataDeps->isFusible(K)) {
            K->setOptimizationOptions(OptimizationOption::KERNEL_FUSE);
          }
          if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
            std::string nodeName = dataDeps->getGraphKernelNodeName(K->getKernelClass()->getName() + K->getName());
            std::string nodeDepStr(nodeName + "dep_");
            std::string nodeArgStr(nodeName + "arg_");
            K->setGraphNodeName(nodeName);
            K->setGraphNodeDepName(nodeDepStr);
            K->setGraphNodeArgName(nodeArgStr);
          }

          // remove kernel declaration
          TextRewriter.RemoveText(D->getSourceRange());

          // create map between Image or Accessor instances and kernel
          // variables; replace image instances by accessors with undefined
          // boundary handling
          hipacc_require(isa<CXXConstructExpr>(VD->getInit()),
               "Expected Image definition (CXXConstructExpr).", &Diags, VD->getLocation());
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

          // check kernel fusibility
          if (compilerOptions.fuseKernels() &&
                compilerOptions.emitCUDA() &&
                  kernelFuser->parseFusibleKernel(K)) {
            break;
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

          if (compilerOptions.printVerbose()) {
            K->printStats();
          }

          // translate binning function if we have one
          if (KC->getBinningFunction()) {
            Stmt *binningStmts = Hipacc->translateBinning(
                KC->getBinningFunction()->getBody());
            KC->getBinningFunction()->setBody(binningStmts);
          }

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
  if (D->hasAttrs()) {
    for (auto attr : D->getAttrs()) {
      std::string attr_name(attr->getSpelling());
      std::string attr_annotate("annotate");
      std::string attr_hipacc("annotate(\"hipacc_codegen\")");
      if (attr_name == attr_annotate) {
        std::string SS; llvm::raw_string_ostream S(SS);
        attr->printPretty(S, Policy);
        std::string attr_string(S.str());
        if (attr_string.find(attr_hipacc) != std::string::npos) {
          if (D->isTemplated() && !D->isTemplateInstantiation()) {
            // If function is templated and not yet instanciated, skip this
            // declaration and visit instanciated template later.
            return true;
          }
          hipacc_require(D->getBody(), "function to parse has no body.", &Diags, D->getLocation());
          hipacc_require(isa<CompoundStmt>(D->getBody()), "CompoundStmt for main body expected.", &Diags, D->getLocation());
          mainFD = D;
          // additional data dep analysis for CUDA backend
          if (compilerOptions.emitCUDA() && (compilerOptions.fuseKernels() || compilerOptions.useGraph())) {
            AnalysisDeclContext AC(0, mainFD);
            dataDeps = HostDataDeps::parse(Context, Policy, AC, compilerClasses,
                compilerOptions, KernelClassDeclMap);
            if (compilerOptions.fuseKernels()) {
              kernelFuser = new ASTFuse(Context, Diags, builtins, compilerOptions, Policy, dataDeps);
            }
            if (compilerOptions.useGraph()) {
              for (auto img : dataDeps->getOutputImageNames()) {  // record output images
                outputImagesVisitorMap_[img] = false;
              }
            }
          }
        }
      }
    }
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
      return mte->getSubExpr();
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
    unsigned DomIdxX{}, DomIdxY{};

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

          hipacc_require(DomLHS->isConstant(),
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

      hipacc_require(isa<IntegerLiteral>(arg),
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
        stringCreator.writeMemoryTransferRegion("hipaccMakeAccessor<"+ ImgLHS->getTypeStr() +">(" +
            ImgLHS->getName() + ")", AccRHS->getName(), newStr);
      } else if (ImgLHS && PyrRHS) {
        // Img1 = Pyr2(x2);
        stringCreator.writeMemoryTransfer(ImgLHS,
            PyrRHS->getName() + "(" + PyrIdxRHS + ")",
            DEVICE_TO_DEVICE, newStr);
      } else if (AccLHS && ImgRHS) {
        // Acc1 = Img2;
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            "hipaccMakeAccessor<"+ ImgRHS->getTypeStr() +">(" + ImgRHS->getName() + ")", newStr);
      } else if (AccLHS && AccRHS) {
        // Acc1 = Acc2;
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            AccRHS->getName(), newStr);
      } else if (AccLHS && PyrRHS) {
        // Acc1 = Pyr2(x2);
        stringCreator.writeMemoryTransferRegion(AccLHS->getName(),
            "hipaccMakeAccessor<"+ PyrRHS->getTypeStr() +">(" + PyrRHS->getName() + "(" + PyrIdxRHS + "))",
            newStr);
      } else if (PyrLHS && ImgRHS) {
        // Pyr1(x1) = Img2
        stringCreator.writeMemoryTransfer(PyrLHS, PyrIdxLHS, ImgRHS->getName(),
            DEVICE_TO_DEVICE, newStr);
      } else if (PyrLHS && AccRHS) {
        // Pyr1(x1) = Acc2
        stringCreator.writeMemoryTransferRegion(
            "hipaccMakeAccessor<"+ PyrLHS->getTypeStr() +">(" + PyrLHS->getName() + "(" + PyrIdxLHS + "))",
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
          } else if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
            std::string nodeName = dataDeps->getGraphMemcpyNodeName(ImgLHS->getName(), data_str, "H2D");
            std::string nodeDepStr(nodeName + "dep_");
            std::string nodeArgStr(nodeName + "arg_");
            stringCreator.addMemoryTransferGraph(ImgLHS, data_str, HOST_TO_DEVICE, cudaGraphStr,
                                                 nodeName, nodeDepStr, nodeArgStr, newStr);
            // add dependencies for other node
            newStr += "\n" + stringCreator.getIndent();
            for (auto dStr : dataDeps->getGraphMemcpyNodeDepOn(ImgLHS->getName(), data_str, "H2D")) {
              newStr += dStr + "dep_.push_back(" + nodeName + ");\n" + stringCreator.getIndent();
            }
          } else {
            stringCreator.writeMemoryTransfer(ImgLHS, data_str, HOST_TO_DEVICE, newStr);
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

        hipacc_require(E->getNumArgs() <= 1
          , "Kernel::execute must not have more than one argument"
          , &Diags, E->getDirectCallee()->getLocation());

        if(E->getNumArgs() == 1)
        {
          std::string execution_parameter;
          llvm::raw_string_ostream SS(execution_parameter);
          E->getArg(0)->printPretty(SS, 0, Policy);
          K->setExecutionParameter(SS.str());
        }

        std::string newStr;

        // this was checked before, when the user class was parsed
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(VD->getInit());
        hipacc_require(CCE->getNumArgs() == K->getKernelClass()->getMembers().size(),
            "number of arguments doesn't match!");

        // set host argument names and retrieve literals stored to temporaries
        K->setHostArgNames(llvm::makeArrayRef(CCE->getArgs(),
              CCE->getNumArgs()), newStr, literalCount);

        //
        // TODO: handle the case when only reduce function is specified
        //
        if (K->isFusible()) {
          stringCreator.writeFusedKernelCall(K, newStr, kernelFuser);
        } else {
          stringCreator.writeKernelCall(K, newStr);
          if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
            // add dependencies for other node
            std::string nodeName = "node_" + K->getKernelClass()->getName() + K->getName() + "_";
            for (auto dStr : dataDeps->getGraphKernelNodeDepOn(K->getKernelClass()->getName() + K->getName())) {
              newStr += dStr + "dep_.push_back(" + nodeName + ");\n" + stringCreator.getIndent();
            }
          }
        }
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

            hipacc_require(K->getKernelClass()->getBinningFunction()
                   , "Called binned_data() but no binning function defined!");

            callStr += "\n" + stringCreator.getIndent();
            stringCreator.writeBinningCall(K, callStr);

            resultStr = K->getBinningStr();

            if (compilerOptions.emitC99()) {
              // for CPU we return a vector, get access to its data
              resultStr += ".data()";
            }
          } else {
            hipacc_require(K->getKernelClass()->getReduceFunction()
                   , "Called reduced_data() but no reduce function defined!");

            hipacc_require(E->getNumArgs() <= 1
                   ,"Kernel::execute must not have more than one argument"
                   , &Diags, E->getDirectCallee()->getLocation());

            if(E->getNumArgs() == 1)
            {
              std::string execution_parameter;
              llvm::raw_string_ostream SS(execution_parameter);
              E->getArg(0)->printPretty(SS, 0, Policy);
              K->setExecutionParameter(SS.str());
            }
            callStr += "\n" + stringCreator.getIndent();
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
          if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
            // decl node and deps
            std::string nodeName = dataDeps->getGraphMemcpyNodeName(Img->getName(), Img->getName(), "D2H");
            std::string nodeDepStr(nodeName + "dep_");
            std::string nodeArgStr(nodeName + "arg_");
            stringCreator.addMemoryTransferGraph(Img, "NULL", DEVICE_TO_HOST, cudaGraphStr,
                                                 nodeName, nodeDepStr, nodeArgStr, newStr);
            // add dependencies for other node
            newStr += "\n" + stringCreator.getIndent();
            for (auto dStr : dataDeps->getGraphMemcpyNodeDepOn(Img->getName(), Img->getName(), "D2H")) {
              newStr += dStr + "dep_.push_back(" + nodeName + ");\n" + stringCreator.getIndent();
            }
            hipacc_require(outputImagesVisitorMap_.count(Img->getName()), "Missing Graph Output Image");
            outputImagesVisitorMap_[Img->getName()] = true;
          } else {
            stringCreator.writeMemoryTransfer(Img, "NULL", DEVICE_TO_HOST, newStr);
          }

          // additional call for cuda graph
          if (compilerOptions.useGraph() && compilerOptions.emitCUDA()) {
            bool allOutputImgVisited = true;
            hipacc_require(!outputImagesVisitorMap_.empty(), "No Output Image Recorded");
            for (auto imgMap : outputImagesVisitorMap_) {
              if (imgMap.second == false) {
                allOutputImgVisited = false;
              }
            }
            if (allOutputImgVisited) {
              newStr += "\n" + stringCreator.getIndent();
              newStr += "\n" + stringCreator.getIndent();
              newStr += "// CUDA Graph Execution\n" + stringCreator.getIndent();
              // number of nodes query
              newStr += "cudaGraphNode_t *nullNode = NULL;\n" + stringCreator.getIndent();
              newStr += "size_t numNodes = 0;\n" + stringCreator.getIndent();
              newStr += "cudaGraphGetNodes(" + cudaGraphStr + ", nullNode, &numNodes);\n" + stringCreator.getIndent();
              newStr += "printf(\"\\nNum of nodes in the graph = \%zu\\n\", numNodes);\n" + stringCreator.getIndent();
              // cuda graph init
              newStr += "\n" + stringCreator.getIndent();
              newStr += "// CUDA Graph Initialization\n" + stringCreator.getIndent();
              newStr += "cudaGraphInstantiate(&" + cudaGraphStr + "exec_, " + cudaGraphStr + ", NULL, NULL, 0);\n" + stringCreator.getIndent();
              // cuda graph init
              newStr += "\n" + stringCreator.getIndent();
              newStr += "// CUDA Graph Launch\n" + stringCreator.getIndent();
              newStr += "cudaGraphLaunch(" + cudaGraphStr + "exec_, " + cudaGraphStr + "stream_);\n" + stringCreator.getIndent();
              newStr += "cudaStreamSynchronize(" + cudaGraphStr + "stream_)";
            }
          }

          // rewrite Image assignment to memory transfer
          SourceRange rangeE(E->getBeginLoc(), E->getEndLoc());
          TextRewriter.ReplaceText(rangeE, newStr);
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
        // init hipaccTraversor in runtime
        SourceRange range(E->getBeginLoc(),
                          E->getBeginLoc().getLocWithOffset(std::string("traverse").length()-1));
        std::string newStr("");
        if (!skipString) {
          newStr += "HipaccPyramidTraversor hipaccTraversor;\n";
          skipString = true;
        }
        newStr += "hipaccTraversor.hipaccTraverse";
        TextRewriter.ReplaceText(range, newStr);
      }
    }
  }
  return true;
}


void Rewrite::setKernelConfiguration(HipaccKernelClass *KC, HipaccKernel *K) {
//  #ifdef USE_JIT_ESTIMATE
//  switch (compilerOptions.getTargetLang()) {
//    default: return K->setDefaultConfig();
//    case clang::hipacc::Language::CUDA:
//    case clang::hipacc::Language::OpenCLGPU:
//      if (!targetDevice.isARMGPU())
//        break;
//  }
//
//  // write kernel file to estimate resource usage
//  // kernel declaration for CUDA
//  FunctionDecl *kernelDeclEst = createFunctionDecl(Context,
//      Context.getTranslationUnitDecl(), K->getKernelName(), Context.VoidTy,
//      K->getArgTypes(), K->getDeviceArgNames());
//
//  // create kernel body
//  ASTTranslate *HipaccEst = new ASTTranslate(Context, kernelDeclEst, K, KC,
//      builtins, compilerOptions, true);
//  Stmt *kernelStmtsEst = HipaccEst->Hipacc(KC->getKernelFunction()->getBody());
//  kernelDeclEst->setBody(kernelStmtsEst);
//
//  // write kernel to file
//  printKernelFunction(kernelDeclEst, KC, K, K->getFileName(), false);
//
//  // compile kernel in order to get resource usage
//  std::string command = K->getCompileCommand(K->getKernelName(),
//      K->getFileName(), compilerOptions.emitCUDA());
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
//      << K->getFileName() << (const char*)(compilerOptions.emitCUDA()?"cu":"cl")
//      << command.c_str();
//    if (compilerOptions.printVerbose()) {
//      // print full command output
//      for (auto line : lines)
//        llvm::errs() << line;
//    }
//  } else if (compilerOptions.printVerbose()) {
//    if (targetDevice.isNVIDIAGPU()) {
//      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
//                   << ": " << reg << " registers, "
//                   << lmem << " bytes lmem, "
//                   << smem << " bytes smem, "
//                   << cmem << " bytes cmem\n";
//    } else if (targetDevice.isAMDGPU()) {
//      llvm::errs() << "Resource usage for kernel '" << K->getKernelName() << "'"
//                   << ": " << reg << " gprs, "
//                   << smem << " bytes lds\n";
//    }
//  }
//
//  K->setResourceUsage(reg, lmem, smem, cmem);
//  #else
  K->setDefaultConfig();
//  #endif
}


void Rewrite::printBinningFunction(HipaccKernelClass *KC, HipaccKernel *K,
    llvm::raw_fd_ostream &OS) {
  FunctionDecl *bin_fun = KC->getBinningFunction();
  QualType pixelType = KC->getPixelType();
  QualType binType = KC->getBinType();
  std::string signatureBinning;

  // preprocessor defines
  std::string KID = K->getKernelName();
  switch (compilerOptions.getTargetLang()) {
    case clang::hipacc::Language::C99:
    case clang::hipacc::Language::OpenCLACC:
    case clang::hipacc::Language::OpenCLCPU:
    case clang::hipacc::Language::OpenCLGPU:
      OS << "#define " << KID << "PPT " << K->getPixelsPerThread() << "\n";
      break;
    case clang::hipacc::Language::CUDA:
      break;
  }
  OS << "\n";

  // write binning signature and qualifiers
  if (compilerOptions.emitCUDA()) {
    signatureBinning += "__device__ ";
    signatureBinning += "inline IdxVal<" + binType.getAsString() + "> " + K->getBinningName() + "(";
  } else {
    signatureBinning += "inline void " + K->getBinningName() + "(";
  }
  if (!compilerOptions.emitCUDA()) {
    if (compilerOptions.emitOpenCL()) {
      signatureBinning += "__local ";
    }
    signatureBinning += binType.getAsString();
    signatureBinning += " *_lmem, uint _offset, ";
  }
  signatureBinning += "uint _num_bins, ";

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
    // TODO: Implement synchronization using locks for bin types >64bit
    // TODO: Consider compiler switch to force locks for bin types >64bit
  } else if (binType.getTypePtr()->isIntegerType()) {
    // INT: Synchronize using thread ID tagging
    llvm::errs() << "WARNING: First 5 bits of bin value are used for thread ID tagging!\n";
    // TODO: Consider compiler switch to force CAS for full bit width
  }

  if(!compilerOptions.emitCUDA()) {
    // print forward declaration
    OS << signatureBinning << ";\n\n";

    // instantiate reduction
    switch (compilerOptions.getTargetLang()) {
      case clang::hipacc::Language::C99:
        OS << "BINNING_CPU_2D(";
        OS << K->getBinningName() << "2D, "
           << pixelType.getAsString() << ", "
           << binType.getAsString() << ", "
           << K->getReduceName() << ", "
           << K->getBinningName() << ", "
           << KID << "PPT"
           << ")\n\n";
        break;
      case clang::hipacc::Language::OpenCLACC:
      case clang::hipacc::Language::OpenCLCPU:
      case clang::hipacc::Language::OpenCLGPU:
        if (compilerOptions.emitOpenCL()) {
          OS << "BINNING_CL_2D_SEGMENTED("
            << K->getBinningName() << "2D, "
            << K->getBinningName() << "1D, ";
        }

        OS << pixelType.getAsString() << ", "
          << binType.getAsString() << ", "
          << K->getReduceName() << ", "
          << K->getBinningName() << ", ";

        if (bitWidth > 64) {
          OS << "ACCU_CAS_GT64, UNTAG_NONE, ";
        } else {
          if (binType.getTypePtr()->isIntegerType()) {
            OS << "ACCU_INT, UNTAG_INT, ";
          } else {
            OS << "ACCU_CAS_" << bitWidth << ", UNTAG_NONE, ";
          }
        }

        OS << K->getWarpSize() << ", "
           << compilerOptions.getReduceConfigNumWarps() << ", "
           << compilerOptions.getReduceConfigNumUnits() << ", "
           << KID << "PPT, ";

        OS << (binType.getTypePtr()->isVectorType()
                ? "(" + binType.getAsString() + ")(0)"
                : "(0)");

        OS << ")\n\n";
        break;
      default:
        break;
    }
  }

  // print binning function
  OS << signatureBinning << "\n";

  if (compilerOptions.emitCUDA()) {
    //declare IdxVal<BinType> before function body because type IdxVal is not availible in AST translation context
    OS << "{ IdxVal<" + binType.getAsString() + "> _ret { -1, " + binType.getAsString() + "{} };\n";
  }

  bin_fun->getBody()->printPretty(OS, 0, Policy, 0);

  if (compilerOptions.emitCUDA()) {
    OS << "return _ret;\n}\n";
  }

  OS << "\n";
}


void Rewrite::printReductionFunction(HipaccKernelClass *KC, HipaccKernel *K,
    llvm::raw_fd_ostream &OS) {
  FunctionDecl *fun = KC->getReduceFunction();

  // preprocessor defines
  OS << "#define BS " << K->getNumThreadsReduce() << "\n";
  if(!compilerOptions.emitCUDA()) {
    OS << "#define PPT " << K->getPixelsPerThreadReduce() << "\n";
  }
  if (K->getIterationSpace()->isCrop()) {
    OS << "#define USE_OFFSETS\n";
  }
  switch (compilerOptions.getTargetLang()) {
    case clang::hipacc::Language::C99:
      if (compilerOptions.useOpenMP()) {
        OS << "#define USE_OPENMP\n";
      }
      OS << "#include \"hipacc_cpu_red.hpp\"\n\n";
      break;
    case clang::hipacc::Language::OpenCLACC:
    case clang::hipacc::Language::OpenCLCPU:
    case clang::hipacc::Language::OpenCLGPU:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType() == Texture::Array2D) {
        OS << "#define USE_ARRAY_2D\n";
      }
      OS << "#include \"hipacc_cl_red.hpp\"\n\n";
      break;
    case clang::hipacc::Language::CUDA:
      if (compilerOptions.useTextureMemory() &&
          compilerOptions.getTextureType() == Texture::Array2D) {
        OS << "#define USE_ARRAY_2D\n";
      }
      OS << "#include \"hipacc_cu_red.hpp\"\n\n";
      break;
  }


  // write kernel name and qualifiers
  switch (compilerOptions.getTargetLang()) {
    default: break;
    case clang::hipacc::Language::CUDA:
      OS << "__device__ ";
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
    case clang::hipacc::Language::C99:
      // 2D reduction
      OS << "REDUCTION_CPU_2D(" << K->getReduceName() << "2D, "
         << fun->getReturnType().getAsString() << ", "
         << K->getReduceName() << ", "
         << "PPT)\n";
      break;
    case clang::hipacc::Language::OpenCLACC:
    case clang::hipacc::Language::OpenCLCPU:
    case clang::hipacc::Language::OpenCLGPU:
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
    case clang::hipacc::Language::CUDA:
      // 2D CUDA array definition - only required if Array2D is selected
      OS << "texture<" << fun->getReturnType().getAsString()
         << ", cudaTextureType2D, cudaReadModeElementType> _tex"
         << K->getIterationSpace()->getImage()->getName() + K->getName()
         << ";\n__device__ const textureReference *_tex"
         << K->getIterationSpace()->getImage()->getName() + K->getName()
         << "Ref;\n\n";
      break;
  }

  OS << "#include \"hipacc_undef.hpp\"\n";
  OS << "\n";
}


void Rewrite::printKernelFunction(FunctionDecl *D, HipaccKernelClass *KC,
    HipaccKernel *K, std::string file, bool emitHints) {
  int fd;
  std::string filename(file);
  std::string ifdef("_" + file + "_");
  switch (compilerOptions.getTargetLang()) {
    case clang::hipacc::Language::C99:          filename += ".cc"; ifdef += "CC_"; break;
    case clang::hipacc::Language::CUDA:         filename += ".cu"; ifdef += "CU_"; break;
    case clang::hipacc::Language::OpenCLACC:
    case clang::hipacc::Language::OpenCLCPU:
    case clang::hipacc::Language::OpenCLGPU:    filename += ".cl"; ifdef += "CL_"; break;
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
    case clang::hipacc::Language::CUDA:
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
  }

  // declarations of textures, surfaces, variables, includes, definitions etc.
  SmallVector<std::string, 16> InterpolationDefinitionsLocal;
  std::string target_interpolation_include{};
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
        case clang::hipacc::Language::CUDA:
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
                default: hipacc_require(0, "texture expected.");
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
      }

      if (Acc->getInterpolationMode() > Interpolate::NN) {
        switch (compilerOptions.getTargetLang()) {
          case clang::hipacc::Language::C99:
            target_interpolation_include = "#include \"hipacc_cpu_interpolate.hpp\"\n\n";
            break;
          case clang::hipacc::Language::CUDA:
            target_interpolation_include = "#include \"hipacc_cu_interpolate.hpp\"\n\n";
            break;
          case clang::hipacc::Language::OpenCLACC:
          case clang::hipacc::Language::OpenCLCPU:
          case clang::hipacc::Language::OpenCLGPU:
            target_interpolation_include = "#include \"hipacc_cl_interpolate.hpp\"\n\n";
            break;
        }
        OS << target_interpolation_include;

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

        InterpolationDefinitionsLocal.push_back(bh_def);
        InterpolationDefinitionsLocal.push_back(no_bh_def);

        // do not add 'SCALAR_TYPE_FUNS(float)' as it would create already existing functions
        if(Acc->getImage()->getType()->isVectorType() || Acc->getImage()->getTypeStr() != "float")
          InterpolationDefinitionsLocal.push_back(vec_conv);
      }
      continue;
    }

    // constant memory declarations
    if (auto Mask = K->getMaskFromMapping(arg)) {
      if (Mask->isConstant()) {
        switch (compilerOptions.getTargetLang()) {
          case clang::hipacc::Language::OpenCLACC:
          case clang::hipacc::Language::OpenCLCPU:
          case clang::hipacc::Language::OpenCLGPU:
            OS << "__constant ";
            break;
          case clang::hipacc::Language::CUDA:
            OS << "__device__ __constant__ ";
            break;
          case clang::hipacc::Language::C99:
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
        // emit declaration in CUDA
        // for other back ends, the mask will be added as kernel parameter
        switch (compilerOptions.getTargetLang()) {
          default: break;
          case clang::hipacc::Language::CUDA:
            OS << "__device__ __constant__ " << Mask->getTypeStr() << " "
               << Mask->getName() << K->getName() << "[" << Mask->getSizeYStr()
               << "][" << Mask->getSizeXStr() << "];\n\n";
            Mask->setIsPrinted(true);
            break;
        }
      }
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

    switch (compilerOptions.getTargetLang()) {
      case clang::hipacc::Language::C99:
      case clang::hipacc::Language::CUDA:
        if (emitHints) {
          // emit interpolation definitions via auxiliary header file to obey ODR rule
          for (auto str : InterpolationDefinitionsLocal)
            InterpolationDefinitionsGlobal.push_back(str);

          WriteAuxiliaryInterpolationDeclHeader(InterpolationDefinitionsGlobal, "interpolation_def.h", target_interpolation_include);

          OS << "#include \"interpolation_def.h\"\n";
          break;
        }
      default:
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
      case clang::hipacc::Language::C99:
      case clang::hipacc::Language::OpenCLACC:
      case clang::hipacc::Language::OpenCLCPU:
      case clang::hipacc::Language::OpenCLGPU:
        OS << "inline "; break;
      case clang::hipacc::Language::CUDA:
        OS << "__inline__ __device__ "; break;
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
      case clang::hipacc::Language::C99:
      case clang::hipacc::Language::CUDA:
        OS << "__global__ ";
        OS << "__launch_bounds__ (" << K->getNumThreadsX() << "*"
            << K->getNumThreadsY() << ") ";
        break;
      case clang::hipacc::Language::OpenCLACC:
      case clang::hipacc::Language::OpenCLCPU:
      case clang::hipacc::Language::OpenCLGPU:
        if (compilerOptions.useTextureMemory() &&
            compilerOptions.getTextureType() == Texture::Array2D) {
          OS << "__constant sampler_t " << D->getNameInfo().getAsString()
             << "Sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | "
             << " CLK_FILTER_NEAREST; \n\n";
        }
        OS << "__kernel ";
        OS << "__attribute__((reqd_work_group_size(" << K->getNumThreadsX()
            << ", " << K->getNumThreadsY() << ", 1))) ";
        break;
    }
    OS << "void ";
    OS << K->getKernelName();
    OS << "(";

    // write kernel parameters
    size_t comma = 0; num_arg = 0;
    for (auto param : D->parameters()) {
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
          case clang::hipacc::Language::C99:
            if (comma++)
              OS << ", ";
            OS << "const "
               << Mask->getTypeStr()
               << " " << Mask->getName() << K->getName()
               << "[" << Mask->getSizeYStr() << "]"
               << "[" << Mask->getSizeXStr() << "]";
            break;
          case clang::hipacc::Language::OpenCLACC:
          case clang::hipacc::Language::OpenCLCPU:
          case clang::hipacc::Language::OpenCLGPU:
            if (comma++)
              OS << ", ";
            OS << "__constant ";
            T.getAsStringInternal(Name, Policy);
            OS << Name;
            break;
          case clang::hipacc::Language::CUDA:
            // mask/domain is declared as constant memory
            break;
        }
        continue;
      }

    // check if we have an Accessor
    if (auto Acc = K->getImgFromMapping(FD)) {
      MemoryAccess mem_acc = KC->getMemAccess(FD);
      switch (compilerOptions.getTargetLang()) {
        case clang::hipacc::Language::C99: {
          if (comma++)
            OS << ", ";
          if (mem_acc == READ_ONLY)
            OS << "const ";

          std::string type_str = Acc->getImage()->getTypeStr();
          type_str = std::regex_replace(type_str, std::regex("(^const )|( const$)"), "");

          OS << type_str
             << " " << Name
             << "[" << Acc->getImage()->getSizeYStr() << "]"
             << "[" << Acc->getImage()->getSizeXStr() << "]";
          // alternative for Pencil:
          // OS << "[static const restrict 2048][4096]";
          }
          break;
        case clang::hipacc::Language::CUDA:
          if (K->useTextureMemory(Acc) != Texture::None &&
              K->useTextureMemory(Acc) != Texture::Ldg) // no parameter is emitted for textures
            continue;
          else {
            if (comma++)
              OS << ", ";
            if (mem_acc == READ_ONLY)
              OS << "const ";

            std::string type_str = T->getPointeeType().getAsString();
            type_str = std::regex_replace(type_str, std::regex("(^const )|( const$)"), "");

            OS << type_str;
            OS << " * __restrict__ ";
            OS << Name;
          }
          break;
        case clang::hipacc::Language::OpenCLACC:
        case clang::hipacc::Language::OpenCLCPU:
        case clang::hipacc::Language::OpenCLGPU:
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

            std::string type_str = T->getPointeeType().getAsString();
            type_str = std::regex_replace(type_str, std::regex("(^const )|( const$)"), "");

            OS << type_str;
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

