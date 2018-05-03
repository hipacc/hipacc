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

//===--- ASTTranslate.h - Translation of the AST --------------------------===//
//
// This file implements translation of statements and expressions.
//
//===----------------------------------------------------------------------===//

#ifndef _ASTTRANSLATE_H_
#define _ASTTRANSLATE_H_

#include <clang/AST/Attr.h>
#include <clang/AST/Type.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Ownership.h>
#include <llvm/ADT/SmallVector.h>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/AST/ASTNode.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/Builtins.h"
#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/Vectorization/SIMDTypes.h"

#include <functional>

//===----------------------------------------------------------------------===//
// Statement/expression transformations
//===----------------------------------------------------------------------===//

namespace clang {
namespace hipacc {
typedef union border_variant {
  struct {
    unsigned left   : 1;
    unsigned right  : 1;
    unsigned top    : 1;
    unsigned bottom : 1;
  } borders;
  unsigned borderVal;
  border_variant() : borderVal(0) {}
} border_variant;

class ASTTranslate : public StmtVisitor<ASTTranslate, Stmt *> {
  private:
    enum TranslationMode {
      CloneAST,
      TranslateAST
    };
    ASTContext &Ctx;
    DiagnosticsEngine &Diags;
    FunctionDecl *kernelDecl;
    HipaccKernel *Kernel;
    HipaccKernelClass *KernelClass;
    hipacc::Builtin::Context &builtins;
    CompilerOptions &compilerOptions;
    TranslationMode astMode;
    SIMDTypes simdTypes;
    border_variant bh_variant;
    bool emitEstimation;

    // "global variables"
    unsigned literalCount;
    SmallVector<FunctionDecl *, 16> cloneFuns;
    SmallVector<Stmt *, 16> preStmts, postStmts;
    SmallVector<CompoundStmt *, 16> preCStmt, postCStmt;
    CompoundStmt *curCStmt;
    HipaccMask *convMask;
    DeclRefExpr *convTmp;
    Reduce convMode;
    int convIdxX, convIdxY;

    SmallVector<HipaccMask *, 4> redDomains;
    SmallVector<DeclRefExpr *, 4> redTmps;
    SmallVector<Reduce, 4> redModes;
    SmallVector<int, 4> redIdxX, redIdxY;

    DeclRefExpr *bh_start_left, *bh_start_right, *bh_start_top,
                *bh_start_bottom, *bh_fall_back;
    DeclRefExpr *outputImage;
    DeclRefExpr *retValRef;
    Expr *writeImageRHS;
    NamespaceDecl *hipacc_ns, *hipacc_math_ns;
    TypedefDecl *samplerTy;
    DeclRefExpr *kernelSamplerRef;

    class BlockingVars {
      public:
        Expr *global_id_x, *global_id_y;
        Expr *local_id_x, *local_id_y;
        Expr *local_size_x, *local_size_y;
        Expr *block_id_x, *block_id_y;
        //Expr *block_size_x, *block_size_y;
        //Expr *grid_size_x, *grid_size_y;

        BlockingVars() :
          global_id_x(nullptr), global_id_y(nullptr), local_id_x(nullptr),
          local_id_y(nullptr), local_size_x(nullptr), local_size_y(nullptr),
          block_id_x(nullptr), block_id_y(nullptr) {}
    };
    BlockingVars tileVars;
    // updated index for PPT (iteration space unrolling)
    Expr *lidYRef, *gidYRef;


    // BinningTranslater for translating binning(uint x, uint y, data_t pixel)
    // independently of kernel() method.
    class BinningTranslator : public StmtVisitor<BinningTranslator, Stmt *> {
      private:
        ASTContext &ctx_;
        HipaccKernel *kernel_;
        Expr *lmem_, *offset_;
        FunctionDecl *binFunc_;

        void createBinningArguments(QualType &idxType, QualType &binType) {
          VarDecl *declLMem = nullptr, *declOffset = nullptr;

          FunctionDecl *binningDecl =
            kernel_->getKernelClass()->getBinningFunction();

          declLMem = ASTNode::createVarDecl(ctx_, binningDecl, "_lmem", binType,
              nullptr);
          declOffset = ASTNode::createVarDecl(ctx_, binningDecl, "_offset",
              idxType, nullptr);

          lmem_ = ASTNode::createDeclRefExpr(ctx_, declLMem);
          offset_ = ASTNode::createDeclRefExpr(ctx_, declOffset);
        }

        void createBinningPutDeclaration(QualType &idxType, QualType &binType) {
          createBinningArguments(idxType, binType);

          SmallVector<QualType, 16> argTypes;
          SmallVector<std::string, 16> argNames;

          argTypes.push_back(lmem_->getType());
          argNames.push_back("_lmem");
          argTypes.push_back(offset_->getType());
          argNames.push_back("_offset");
          argTypes.push_back(idxType);
          argNames.push_back("index");
          argTypes.push_back(binType);
          argNames.push_back("value");

          binFunc_ = ASTNode::createFunctionDecl(ctx_,
              ctx_.getTranslationUnitDecl(), kernel_->getBinningName() + "Put",
              ctx_.VoidTy, argTypes, argNames);
        }

      public:
        BinningTranslator(ASTContext &ctx, HipaccKernel *kernel)
            : ctx_(ctx), kernel_(kernel) {
          QualType idxType = ctx_.UnsignedIntTy;
          QualType binType = kernel_->getKernelClass()->getBinType();

          createBinningPutDeclaration(idxType, binType);
        }

        template<class T> T *Clone(T *S) {
          if (S==nullptr)
            return nullptr;

          return static_cast<T *>(Visit(S));
        }

        Stmt *VisitCompoundStmt(CompoundStmt *S);
        Expr *VisitBinaryOperator(BinaryOperator *E);

        Stmt* translate(Stmt* S) {
          return Clone(S);
        }
    };


    template<class T> T *Clone(T *S) {
      if (S==nullptr)
        return nullptr;

      return static_cast<T *>(Visit(S));
    }
    template<class T> T *CloneDecl(T *D) {
      if (D==nullptr)
        return nullptr;

      switch (D->getKind()) {
        default:
          assert(0 && "Only VarDecls, ParmVArDecls, and FunctionDecls supported!");
        case Decl::ParmVar:
          if (astMode==CloneAST) return D;
          return dyn_cast<T>(CloneParmVarDecl(dyn_cast<ParmVarDecl>(D)));
        case Decl::Var:
          if (astMode==CloneAST) return D;
          return dyn_cast<T>(CloneVarDecl(dyn_cast<VarDecl>(D)));
        case Decl::Function:
          return dyn_cast<T>(cloneFunction(dyn_cast<FunctionDecl>(D)));
      }
    }

    VarDecl *CloneVarDecl(VarDecl *VD);
    VarDecl *CloneParmVarDecl(ParmVarDecl *PVD);
    VarDecl *CloneDeclTex(ParmVarDecl *D, std::string prefix);
    void setExprProps(Expr *orig, Expr *clone);
    void setExprPropsClone(Expr *orig, Expr *clone);
    void setCastPath(CastExpr *orig, CXXCastPath &castPath);
    void initCPU(SmallVector<Stmt *, 16> &kernelBody, Stmt *S);
    void initCUDA(SmallVector<Stmt *, 16> &kernelBody);
    void initOpenCL(SmallVector<Stmt *, 16> &kernelBody);
    void initRenderscript(SmallVector<Stmt *, 16> &kernelBody);
    void updateTileVars();
    Expr *addCastToInt(Expr *E);
    FunctionDecl *cloneFunction(FunctionDecl *FD);
    template <typename T>
    T *lookup(std::string name, QualType QT, NamespaceDecl *NS=nullptr);
    // wrappers to mark variables as being used
    DeclRefExpr *getWidthDecl(HipaccAccessor *Acc) {
      Kernel->setUsed(Acc->getWidthDecl()->getNameInfo().getAsString());
      return Acc->getWidthDecl();
    }
    DeclRefExpr *getHeightDecl(HipaccAccessor *Acc) {
      Kernel->setUsed(Acc->getHeightDecl()->getNameInfo().getAsString());
      return Acc->getHeightDecl();
    }
    DeclRefExpr *getStrideDecl(HipaccAccessor *Acc) {
      Kernel->setUsed(Acc->getStrideDecl()->getNameInfo().getAsString());
      return Acc->getStrideDecl();
    }
    DeclRefExpr *getOffsetXDecl(HipaccAccessor *Acc) {
      Kernel->setUsed(Acc->getOffsetXDecl()->getNameInfo().getAsString());
      return Acc->getOffsetXDecl();
    }
    DeclRefExpr *getOffsetYDecl(HipaccAccessor *Acc) {
      Kernel->setUsed(Acc->getOffsetYDecl()->getNameInfo().getAsString());
      return Acc->getOffsetYDecl();
    }
    DeclRefExpr *getBHStartLeft() {
      Kernel->setUsed(bh_start_left->getNameInfo().getAsString());
      return bh_start_left;
    }
    DeclRefExpr *getBHStartRight() {
      Kernel->setUsed(bh_start_right->getNameInfo().getAsString());
      return bh_start_right;
    }
    DeclRefExpr *getBHStartTop() {
      Kernel->setUsed(bh_start_top->getNameInfo().getAsString());
      return bh_start_top;
    }
    DeclRefExpr *getBHStartBottom() {
      Kernel->setUsed(bh_start_bottom->getNameInfo().getAsString());
      return bh_start_bottom;
    }
    DeclRefExpr *getBHFallBack() {
      Kernel->setUsed(bh_fall_back->getNameInfo().getAsString());
      return bh_fall_back;
    }

    // KernelDeclMap - this keeps track of the cloned Decls which are used in
    // expressions, e.g. DeclRefExpr
    typedef llvm::DenseMap<VarDecl *, VarDecl *> DeclMapTy;
    typedef llvm::DenseMap<ParmVarDecl *, VarDecl *> PVDeclMapTy;
    typedef llvm::DenseMap<ParmVarDecl *, HipaccAccessor *> AccMapTy;
    typedef llvm::DenseMap<FunctionDecl *, FunctionDecl *> FunMapTy;
    DeclMapTy KernelDeclMap;
    DeclMapTy LambdaDeclMap;
    PVDeclMapTy KernelDeclMapTex;
    PVDeclMapTy KernelDeclMapShared;
    PVDeclMapTy KernelDeclMapVector;
    AccMapTy KernelDeclMapAcc;
    FunMapTy KernelFunctionMap;

    // BorderHandling.cpp
    Expr *addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x, Expr
        *local_offset_y, HipaccAccessor *Acc);
    Expr *addBorderHandling(DeclRefExpr *LHS, Expr *local_offset_x, Expr
        *local_offset_y, HipaccAccessor *Acc, SmallVector<Stmt *, 16> &bhStmts,
        SmallVector<CompoundStmt *, 16> &bhCStmt);

    // Convolution.cpp
    Stmt *getConvolutionStmt(Reduce mode, DeclRefExpr *tmp_var, Expr *ret_val);
    Expr *getInitExpr(Reduce mode, QualType QT);
    Stmt *addDomainCheck(HipaccMask *Domain, DeclRefExpr *domain_var, Stmt
        *stmt);
    Expr *convertConvolution(CXXMemberCallExpr *E);

    // Interpolation.cpp
    Expr *addNNInterpolationX(HipaccAccessor *Acc, Expr *idx_x);
    Expr *addNNInterpolationY(HipaccAccessor *Acc, Expr *idx_y);
    FunctionDecl *getInterpolationFunction(HipaccAccessor *Acc);
    FunctionDecl *getTextureFunction(HipaccAccessor *Acc, MemoryAccess mem_acc);
    FunctionDecl *getImageFunction(HipaccAccessor *Acc, MemoryAccess mem_acc);
    FunctionDecl *getAllocationFunction(QualType QT, MemoryAccess mem_acc);
    FunctionDecl *getConvertFunction(QualType QT);
    Expr *addInterpolationCall(DeclRefExpr *LHS, HipaccAccessor *Acc, Expr
        *idx_x, Expr *idx_y);

    // MemoryAccess.cpp
    Expr *addLocalOffset(Expr *idx, Expr *local_offset);
    Expr *addGlobalOffsetX(Expr *idx_x, HipaccAccessor *Acc);
    Expr *addGlobalOffsetY(Expr *idx_y, HipaccAccessor *Acc);
    Expr *removeISOffsetX(Expr *idx_x);
    Expr *removeISOffsetY(Expr *idx_y);
    Expr *accessMem(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess mem_acc,
        Expr *offset_x=nullptr, Expr *offset_y=nullptr);
    Expr *accessMem2DAt(DeclRefExpr *LHS, Expr *idx_x, Expr *idx_y);
    Expr *accessMemArrAt(DeclRefExpr *LHS, Expr *stride, Expr *idx_x, Expr
        *idx_y);
    Expr *accessMemAllocAt(DeclRefExpr *LHS, MemoryAccess mem_acc,
                           Expr *idx_x, Expr *idx_y);
    Expr *accessMemAllocPtr(DeclRefExpr *LHS);
    Expr *accessMemTexAt(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess
        mem_acc, Expr *idx_x, Expr *idx_y);
    Expr *accessMemImgAt(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess
        mem_acc, Expr *idx_x, Expr *idx_y);
    Expr *accessMemShared(DeclRefExpr *LHS, Expr *offset_x=nullptr, Expr
        *offset_y=nullptr);
    Expr *accessMemSharedAt(DeclRefExpr *LHS, Expr *idx_x, Expr *idx_y);
    void stageLineToSharedMemory(ParmVarDecl *PVD, SmallVector<Stmt *, 16>
        &stageBody, Expr *local_offset_x, Expr *local_offset_y, Expr
        *global_offset_x, Expr *global_offset_y);
    void stageIterationToSharedMemory(SmallVector<Stmt *, 16> &stageBody, int
        p);
    void stageIterationToSharedMemoryExploration(SmallVector<Stmt *, 16>
        &stageBody);

    // default error message for unsupported expressions and statements.
    #define HIPACC_UNSUPPORTED_EXPR(EXPR) \
    Expr *Visit##EXPR(EXPR *E) { \
      llvm::errs() << "Hipacc: Stumbled upon unsupported expression: " #EXPR "\n"; \
      E->dump(); \
      std::abort(); \
    }
    #define HIPACC_UNSUPPORTED_STMT(STMT) \
    Stmt *Visit##STMT(STMT *S) { \
      llvm::errs() << "Hipacc: Stumbled upon unsupported statement: " #STMT "\n"; \
      S->dump(); \
      std::abort(); \
    }
    #define HIPACC_UNSUPPORTED_EXPR_BASE_CLASS(EXPR) \
    Expr *Visit##EXPR(EXPR *) { \
      llvm::errs() << "Hipacc: Stumbled upon expression base class, implementation of any derived class missing? Base class was: " #EXPR "\n"; \
      std::abort(); \
    }
    #define HIPACC_UNSUPPORTED_STMT_BASE_CLASS(STMT) \
    Stmt *Visit##STMT(STMT *) { \
      llvm::errs() << "Hipacc: Stumbled upon statement base class, implementation of any derived class missing? Base class was: " #STMT "\n"; \
      std::abort(); \
    }

  public:
    ASTTranslate(ASTContext& Ctx, FunctionDecl *kernelDecl, HipaccKernel
        *kernel, HipaccKernelClass *kernelClass, hipacc::Builtin::Context
        &builtins, CompilerOptions &options, bool emitEstimation=false) :
      Ctx(Ctx),
      Diags(Ctx.getDiagnostics()),
      kernelDecl(kernelDecl),
      Kernel(kernel),
      KernelClass(kernelClass),
      builtins(builtins),
      compilerOptions(options),
      astMode(TranslateAST),
      simdTypes(SIMDTypes(Ctx, builtins, options)),
      bh_variant(),
      emitEstimation(emitEstimation),
      literalCount(0),
      curCStmt(nullptr),
      convMask(nullptr),
      convTmp(nullptr),
      convIdxX(0),
      convIdxY(0),
      bh_start_left(nullptr),
      bh_start_right(nullptr),
      bh_start_top(nullptr),
      bh_start_bottom(nullptr),
      bh_fall_back(nullptr),
      outputImage(nullptr),
      retValRef(nullptr),
      writeImageRHS(nullptr),
      tileVars(),
      lidYRef(nullptr),
      gidYRef(nullptr) {
        // get 'hipacc' namespace context for lookups
        auto hipacc_ident = &Ctx.Idents.get("hipacc");
        for (auto *decl : Ctx.getTranslationUnitDecl()->lookup(hipacc_ident))
          if ((hipacc_ns = cast_or_null<NamespaceDecl>(decl)))
            break;
        assert(hipacc_ns && "could not lookup 'hipacc' namespace");

        // get 'hipacc::math' namespace context for lookups
        auto math_ident = &Ctx.Idents.get("math");
        for (auto *decl : hipacc_ns->lookup(math_ident))
          if ((hipacc_math_ns = cast_or_null<NamespaceDecl>(decl)))
            break;
        assert(hipacc_math_ns && "could not lookup 'hipacc::math' namespace");

        // typedef unsigned sampler_t;
        TypeSourceInfo *TInfosampler =
          Ctx.getTrivialTypeSourceInfo(Ctx.UnsignedIntTy);
        samplerTy = TypedefDecl::Create(Ctx, Ctx.getTranslationUnitDecl(),
            SourceLocation(), SourceLocation(), &Ctx.Idents.get("sampler_t"),
            TInfosampler);

        // sampler_t <clKernel>Sampler
        kernelSamplerRef = ASTNode::createDeclRefExpr(Ctx,
            ASTNode::createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
              kernelDecl->getNameAsString() + "Sampler",
              Ctx.getTypeDeclType(samplerTy), nullptr));

        builtins.InitializeBuiltins();
        Kernel->resetUsed();

        // debug
        //dump_available_statement_visitors();
        // debug
      }

    Stmt *Hipacc(Stmt *S);

    Stmt *translateBinning(Stmt *binningFunc) {
      BinningTranslator binning(Ctx, Kernel);
      Stmt *binningStmts = binning.translate(binningFunc);
      return binningStmts;
    }


  public:
    // dump all available statement visitors
    static void dump_available_statement_visitors() {
      llvm::errs() <<
        #define STMT(Type, Base) #Base << " *Visit"<< #Type << "(" << #Type << " *" << #Base << ");\n" <<
        #include "clang/AST/StmtNodes.inc"
        "\n\n";
    }
    // Interpolation.cpp
    // create interpolation function name
    static std::string getInterpolationName(CompilerOptions &compilerOptions,
        HipaccKernel *Kernel, HipaccAccessor *Acc);

    // the following list is ordered according to
    // include/clang/Basic/StmtNodes.td

    // implementation of Visitors is split into two files:
    // ASTClone.cpp for cloning single AST nodes
    // ASTTranslate.cpp for translation related to CUDA/OpenCL

    #define VISIT_MODE(B, K) \
    B *Visit##K##Clone(K *k); \
    B *Visit##K##Translate(K *k); \
    B *Visit##K(K *k) { \
      if (astMode==CloneAST) return Visit##K##Clone(k); \
      return Visit##K##Translate(k); \
    }

    // Statements
    HIPACC_UNSUPPORTED_STMT_BASE_CLASS( Stmt )
    Stmt *VisitNullStmt(NullStmt *S);
    VISIT_MODE(Stmt, CompoundStmt)
    Stmt *VisitLabelStmt(LabelStmt *S);
    Stmt *VisitAttributedStmt(AttributedStmt *Stmt);
    Stmt *VisitIfStmt(IfStmt *S);
    Stmt *VisitSwitchStmt(SwitchStmt *S);
    Stmt *VisitWhileStmt(WhileStmt *S);
    Stmt *VisitDoStmt(DoStmt *S);
    Stmt *VisitForStmt(ForStmt *S);
    Stmt *VisitGotoStmt(GotoStmt *S);
    Stmt *VisitIndirectGotoStmt(IndirectGotoStmt *S);
    Stmt *VisitContinueStmt(ContinueStmt *S);
    Stmt *VisitBreakStmt(BreakStmt *S);
    VISIT_MODE(Stmt, ReturnStmt)
    Stmt *VisitDeclStmt(DeclStmt *S);
    HIPACC_UNSUPPORTED_STMT_BASE_CLASS( SwitchCase )
    Stmt *VisitCaseStmt(CaseStmt *S);
    Stmt *VisitDefaultStmt(DefaultStmt *S);
    Stmt *VisitCapturedStmt(CapturedStmt *S);

    // Asm Statements
    HIPACC_UNSUPPORTED_STMT_BASE_CLASS( AsmStmt )
    Stmt *VisitGCCAsmStmt(GCCAsmStmt *S);
    HIPACC_UNSUPPORTED_STMT( MSAsmStmt )

    // Obj-C Statements
    HIPACC_UNSUPPORTED_STMT( ObjCAtTryStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCAtCatchStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCAtFinallyStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCAtThrowStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCAtSynchronizedStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCForCollectionStmt )
    HIPACC_UNSUPPORTED_STMT( ObjCAutoreleasePoolStmt )

    // C++ Statements
    Stmt *VisitCXXCatchStmt(CXXCatchStmt *S);
    Stmt *VisitCXXTryStmt(CXXTryStmt *S);
    HIPACC_UNSUPPORTED_STMT( CXXForRangeStmt )

    // C++ Coroutines TS statements
    HIPACC_UNSUPPORTED_STMT( CoroutineBodyStmt )
    HIPACC_UNSUPPORTED_STMT( CoreturnStmt )

    // Expressions
    HIPACC_UNSUPPORTED_EXPR_BASE_CLASS( Expr )
    Expr *VisitPredefinedExpr(PredefinedExpr *E);
    Expr *VisitDeclRefExpr(DeclRefExpr *E);
    Expr *VisitIntegerLiteral(IntegerLiteral *E);
    Expr *VisitFloatingLiteral(FloatingLiteral *E);
    Expr *VisitImaginaryLiteral(ImaginaryLiteral *E);
    Expr *VisitStringLiteral(StringLiteral *E);
    Expr *VisitCharacterLiteral(CharacterLiteral *E);
    Expr *VisitParenExpr(ParenExpr *E);
    Expr *VisitUnaryOperator(UnaryOperator *E);
    Expr *VisitOffsetOfExpr(OffsetOfExpr *E);
    Expr *VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    Expr *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    HIPACC_UNSUPPORTED_EXPR( OMPArraySectionExpr )
    VISIT_MODE(Expr, CallExpr)
    VISIT_MODE(Expr, MemberExpr)
    HIPACC_UNSUPPORTED_EXPR_BASE_CLASS( CastExpr )
    VISIT_MODE(Expr, BinaryOperator)
    Expr *VisitCompoundAssignOperator(CompoundAssignOperator *E);
    HIPACC_UNSUPPORTED_EXPR_BASE_CLASS( AbstractConditionalOperator )
    Expr *VisitConditionalOperator(ConditionalOperator *E);
    Expr *VisitBinaryConditionalOperator(BinaryConditionalOperator *E);
    VISIT_MODE(Expr, ImplicitCastExpr)
    HIPACC_UNSUPPORTED_EXPR_BASE_CLASS( ExplicitCastExpr )
    VISIT_MODE(Expr, CStyleCastExpr)
    Expr *VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    Expr *VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    Expr *VisitInitListExpr(InitListExpr *E);
    Expr *VisitDesignatedInitExpr(DesignatedInitExpr *E);
    Expr *VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E);
    Expr *VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    Expr *VisitNoInitExpr(NoInitExpr *E);
    Expr *VisitArrayInitLoopExpr(ArrayInitLoopExpr *E);
    Expr *VisitArrayInitIndexExpr(ArrayInitIndexExpr *E);
    Expr *VisitParenListExpr(ParenListExpr *E);
    Expr *VisitVAArgExpr(VAArgExpr *E);
    HIPACC_UNSUPPORTED_EXPR( GenericSelectionExpr )
    HIPACC_UNSUPPORTED_EXPR( PseudoObjectExpr )

    // Atomic Expressions
    Expr *VisitAtomicExpr(AtomicExpr *E);

    // GNU Extensions
    Expr *VisitAddrLabelExpr(AddrLabelExpr *E);
    Expr *VisitStmtExpr(StmtExpr *E);
    Expr *VisitChooseExpr(ChooseExpr *E);
    Expr *VisitGNUNullExpr(GNUNullExpr *E);

    // C++ Expressions
    VISIT_MODE(Expr, CXXOperatorCallExpr)
    VISIT_MODE(Expr, CXXMemberCallExpr)
    HIPACC_UNSUPPORTED_EXPR( CXXNamedCastExpr )
    Expr *VisitCXXStaticCastExpr(CXXStaticCastExpr *E);
    Expr *VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E);
    Expr *VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E);
    Expr *VisitCXXConstCastExpr(CXXConstCastExpr *E);
    Expr *VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXTypeidExpr )
    HIPACC_UNSUPPORTED_EXPR( UserDefinedLiteral )
    Expr *VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    Expr *VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    Expr *VisitCXXThisExpr(CXXThisExpr *E);
    Expr *VisitCXXThrowExpr(CXXThrowExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXDefaultArgExpr )
    Expr *VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXScalarValueInitExpr )
    Expr *VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXNewExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXDeleteExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXPseudoDestructorExpr )
    HIPACC_UNSUPPORTED_EXPR( TypeTraitExpr )
    HIPACC_UNSUPPORTED_EXPR( ArrayTypeTraitExpr )
    HIPACC_UNSUPPORTED_EXPR( ExpressionTraitExpr )
    HIPACC_UNSUPPORTED_EXPR( DependentScopeDeclRefExpr )
    Expr *VisitCXXConstructExpr(CXXConstructExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXInheritedCtorInitExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXBindTemporaryExpr )
    VISIT_MODE(Expr, ExprWithCleanups)
    HIPACC_UNSUPPORTED_EXPR( CXXTemporaryObjectExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXUnresolvedConstructExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXDependentScopeMemberExpr )
    HIPACC_UNSUPPORTED_EXPR( OverloadExpr )
    HIPACC_UNSUPPORTED_EXPR( UnresolvedLookupExpr )
    HIPACC_UNSUPPORTED_EXPR( UnresolvedMemberExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXNoexceptExpr )
    HIPACC_UNSUPPORTED_EXPR( PackExpansionExpr )
    HIPACC_UNSUPPORTED_EXPR( SizeOfPackExpr )
    HIPACC_UNSUPPORTED_EXPR( SubstNonTypeTemplateParmExpr )
    HIPACC_UNSUPPORTED_EXPR( SubstNonTypeTemplateParmPackExpr )
    HIPACC_UNSUPPORTED_EXPR( FunctionParmPackExpr )
    Expr *VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
    Expr *VisitLambdaExpr(LambdaExpr *E);
    HIPACC_UNSUPPORTED_EXPR( CXXFoldExpr )

    // C++ Coroutines TS expressions
    HIPACC_UNSUPPORTED_EXPR( CoroutineSuspendExpr )
    HIPACC_UNSUPPORTED_EXPR( CoawaitExpr )
    HIPACC_UNSUPPORTED_EXPR( DependentCoawaitExpr )
    HIPACC_UNSUPPORTED_EXPR( CoyieldExpr )

    // Obj-C Expressions
    HIPACC_UNSUPPORTED_EXPR( ObjCStringLiteral )
    HIPACC_UNSUPPORTED_EXPR( ObjCBoxedExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCArrayLiteral )
    HIPACC_UNSUPPORTED_EXPR( ObjCDictionaryLiteral )
    HIPACC_UNSUPPORTED_EXPR( ObjCEncodeExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCMessageExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCSelectorExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCProtocolExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCIvarRefExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCPropertyRefExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCIsaExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCIndirectCopyRestoreExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCBoolLiteralExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCSubscriptRefExpr )
    HIPACC_UNSUPPORTED_EXPR( ObjCAvailabilityCheckExpr )

    // Obj-C ARC Expressions
    HIPACC_UNSUPPORTED_EXPR( ObjCBridgedCastExpr )

    // CUDA Expressions
    Expr *VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E);

    // Clang Extensions
    Expr *VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    Expr *VisitConvertVectorExpr(ConvertVectorExpr *E);
    HIPACC_UNSUPPORTED_EXPR( BlockExpr )
    HIPACC_UNSUPPORTED_EXPR( OpaqueValueExpr )
    HIPACC_UNSUPPORTED_EXPR( TypoExpr )

    // Microsoft Extensions
    HIPACC_UNSUPPORTED_EXPR( MSPropertyRefExpr )
    HIPACC_UNSUPPORTED_EXPR( MSPropertySubscriptExpr )
    HIPACC_UNSUPPORTED_EXPR( CXXUuidofExpr )
    HIPACC_UNSUPPORTED_STMT( SEHTryStmt )
    HIPACC_UNSUPPORTED_STMT( SEHExceptStmt )
    HIPACC_UNSUPPORTED_STMT( SEHFinallyStmt )
    HIPACC_UNSUPPORTED_STMT( SEHLeaveStmt )
    HIPACC_UNSUPPORTED_STMT( MSDependentExistsStmt )

    // OpenCL Expressions
    Expr *VisitAsTypeExpr(AsTypeExpr *E);

    // OpenMP Directives
    HIPACC_UNSUPPORTED_STMT( OMPExecutableDirective )
    HIPACC_UNSUPPORTED_STMT( OMPLoopDirective )
    HIPACC_UNSUPPORTED_STMT( OMPParallelDirective )
    HIPACC_UNSUPPORTED_STMT( OMPSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPSectionsDirective )
    HIPACC_UNSUPPORTED_STMT( OMPSectionDirective )
    HIPACC_UNSUPPORTED_STMT( OMPSingleDirective )
    HIPACC_UNSUPPORTED_STMT( OMPMasterDirective )
    HIPACC_UNSUPPORTED_STMT( OMPCriticalDirective )
    HIPACC_UNSUPPORTED_STMT( OMPParallelForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPParallelForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPParallelSectionsDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskyieldDirective )
    HIPACC_UNSUPPORTED_STMT( OMPBarrierDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskwaitDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskgroupDirective )
    HIPACC_UNSUPPORTED_STMT( OMPFlushDirective )
    HIPACC_UNSUPPORTED_STMT( OMPOrderedDirective )
    HIPACC_UNSUPPORTED_STMT( OMPAtomicDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetDataDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetEnterDataDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetExitDataDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetParallelDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetParallelForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetUpdateDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTeamsDirective )
    HIPACC_UNSUPPORTED_STMT( OMPCancellationPointDirective )
    HIPACC_UNSUPPORTED_STMT( OMPCancelDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskLoopDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTaskLoopSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPDistributeDirective )
    HIPACC_UNSUPPORTED_STMT( OMPDistributeParallelForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPDistributeParallelForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPDistributeSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetParallelForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTeamsDistributeDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTeamsDistributeSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTeamsDistributeParallelForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTeamsDistributeParallelForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetTeamsDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetTeamsDistributeDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetTeamsDistributeParallelForDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetTeamsDistributeParallelForSimdDirective )
    HIPACC_UNSUPPORTED_STMT( OMPTargetTeamsDistributeSimdDirective )
};
} // namespace hipacc
} // namespace clang

#endif  // _ASTTRANSLATE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

