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

//===--- ASTTranslate.h - C to CL Translation of the AST ------------------===//
//
// This file implements translation of statements and expressions.
//
//===----------------------------------------------------------------------===//

#ifndef _ASTTRANSLATE_H_
#define _ASTTRANSLATE_H_

#include <clang/AST/Type.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/ADT/SmallVector.h>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/AST/ASTNode.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/Builtins.h"
#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/Vectorization/SIMDTypes.h"


//===----------------------------------------------------------------------===//
// Statement/expression transformations
//===----------------------------------------------------------------------===//

namespace clang {
namespace hipacc {
class ASTTranslate : public StmtVisitor<ASTTranslate, Stmt *> {
  private:
    ASTContext &Ctx;
    FunctionDecl *kernelDecl;
    HipaccKernel *Kernel;
    HipaccKernelClass *KernelClass;
    union {
      struct {
        unsigned int left   : 1;
        unsigned int right  : 1;
        unsigned int top    : 1;
        unsigned int bottom : 1;
      } imgBorder;
      unsigned int imgBorderVal;
    };
    hipacc::Builtin::Context &builtins;
    CompilerOptions &compilerOptions;
    SIMDTypes simdTypes;
    bool emitEstimation;
    bool emitPolly;

    // "global variables"
    unsigned int literalCount;
    llvm::SmallVector<Stmt *, 16> bhStmtsVistor;
    llvm::SmallVector<CompoundStmt *, 16> bhCStmtsVistor;
    CompoundStmt *curCompoundStmtVistor;
    HipaccMask *convMask;
    DeclRefExpr *convRed;
    ConvolutionMode convMode;
    int convIdxX, convIdxY;
    Expr *convExprX, *convExprY;

    DeclRefExpr *outputImage;
    Expr *gidXRef, *gidYRef;
    Expr *lidXRef, *lidYRef;
    DeclRefExpr *isWidth, *isHeight;
    Expr *local_id_x, *local_id_y;
    Expr *local_size_x, *local_size_y;
    Expr *block_id_x, *block_id_y;
    Expr *block_size_x, *block_size_y;
    Expr *grid_size_x, *grid_size_y;
    Expr *writeImageRHS;
    TypedefDecl *samplerTy;
    DeclRefExpr *kernelSamplerRef;


    template<class T> T *Clone(T *S);
    template<class T> T *CloneDecl(T *D);
    VarDecl *CloneDeclTex(ParmVarDecl *D, std::string prefix);
    // KernelDeclMap - this keeps track of the cloned Decls which are used in
    // expressions, e.g. DeclRefExpr
    typedef llvm::DenseMap<VarDecl *, VarDecl *> DeclMapTy;
    typedef llvm::DenseMap<ParmVarDecl *, VarDecl *> PVDeclMapTy;
    typedef llvm::DenseMap<ParmVarDecl *, HipaccAccessor *> AccMapTy;
    DeclMapTy KernelDeclMap, LambdaDeclMap;
    PVDeclMapTy KernelDeclMapTex;
    PVDeclMapTy KernelDeclMapShared;
    PVDeclMapTy KernelDeclMapVector;
    AccMapTy KernelDeclMapAcc;

    // BorderHandling.cpp
    Expr *addBorderHandling(DeclRefExpr *LHS, Expr *EX, Expr *EY, HipaccAccessor
        *Acc);
    Expr *addBorderHandling(DeclRefExpr *LHS, Expr *EX, Expr *EY, HipaccAccessor
        *Acc, llvm::SmallVector<Stmt *, 16> &bhStmts,
        llvm::SmallVector<CompoundStmt *, 16> &bhCStmts);
    Stmt *addClampUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper);
    Stmt *addClampLower(HipaccAccessor *Acc, Expr *idx, Expr *lower);
    Stmt *addRepeatUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper);
    Stmt *addRepeatLower(HipaccAccessor *Acc, Expr *idx, Expr *lower);
    Stmt *addMirrorUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper);
    Stmt *addMirrorLower(HipaccAccessor *Acc, Expr *idx, Expr *lower);
    Expr *addConstantUpper(HipaccAccessor *Acc, Expr *idx, Expr *upper, Expr
        *cond);
    Expr *addConstantLower(HipaccAccessor *Acc, Expr *idx, Expr *lower, Expr
        *cond);

    // Interpolation.cpp
    Expr *addNNInterpolationX(HipaccAccessor *Acc, Expr *idx_x);
    Expr *addNNInterpolationY(HipaccAccessor *Acc, Expr *idx_y);
    FunctionDecl *getInterpolationFunction(HipaccAccessor *Acc);
    FunctionDecl *getTextureFunction(HipaccAccessor *Acc, MemoryAccess memAcc);
    FunctionDecl *getImageFunction(HipaccAccessor *Acc, MemoryAccess memAcc);
    Expr *addInterpolationCall(DeclRefExpr *LHS, HipaccAccessor *Acc, Expr
        *idx_x, Expr *idx_y);

    // MemoryAccess.cpp
    Expr *addLocalOffset(Expr *idx, Expr *local_offset);
    Expr *addGlobalOffsetX(Expr *idx_x, HipaccAccessor *Acc);
    Expr *addGlobalOffsetY(Expr *idx_y, HipaccAccessor *Acc);
    Expr *removeISOffsetX(Expr *idx_x, HipaccAccessor *Acc);
    Expr *accessMem(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess memAcc,
        Expr *offset_x=NULL, Expr *offset_y=NULL);
    Expr *accessMemPolly(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess
        memAcc, Expr *offset_x=NULL, Expr *offset_y=NULL);
    Expr *accessMem2DAt(DeclRefExpr *LHS, Expr *idx_x, Expr *idx_y);
    Expr *accessMemArrAt(DeclRefExpr *LHS, Expr *stride, Expr *idx_x, Expr
        *idx_y);
    Expr *accessMemTexAt(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess
        memAcc, Expr *idx_x, Expr *idx_y);
    Expr *accessMemImgAt(DeclRefExpr *LHS, HipaccAccessor *Acc, MemoryAccess
        memAcc, Expr *idx_x, Expr *idx_y);
    Expr *accessMemShared(DeclRefExpr *LHS, Expr *offset_x=NULL, Expr
        *offset_y=NULL);
    Expr *accessMemSharedAt(DeclRefExpr *LHS, Expr *idx_x, Expr *idx_y);
    void stage_line_to_shared_memory(ParmVarDecl *PVD, llvm::SmallVector<Stmt *,
        16> &stageBody, Expr *local_offset_x, Expr *local_offset_y, Expr
        *global_offset_x, Expr *global_offset_y);
    bool stage_first_iteration_to_shared_memory(llvm::SmallVector<Stmt *, 16>
        &stageBody);
    bool stage_next_iteration_to_shared_memory(llvm::SmallVector<Stmt *, 16>
        &stageBody);
    bool update_shared_memory_for_next_iteration(llvm::SmallVector<Stmt *, 16>
        &stageBody);

    // default error message for unsupported expressions and statements.
    #define HIPACC_NOT_SUPPORTED(MSG) \
    assert(0 && "Hipacc: Stumbled upon unsupported expression or statement: " #MSG)

  public:
    ASTTranslate(ASTContext& Ctx, FunctionDecl *kernelDecl, HipaccKernel
        *kernel, HipaccKernelClass *kernelClass, hipacc::Builtin::Context
        &builtins, CompilerOptions &options, bool emitEstimation=false, bool
        emitPolly=false) :
      Ctx(Ctx),
      kernelDecl(kernelDecl),
      Kernel(kernel),
      KernelClass(kernelClass),
      imgBorderVal(0),
      builtins(builtins),
      compilerOptions(options),
      simdTypes(SIMDTypes(Ctx, builtins)),
      emitEstimation(emitEstimation),
      emitPolly(emitPolly),
      literalCount(0),
      curCompoundStmtVistor(NULL),
      convMask(NULL),
      convRed(NULL),
      convIdxX(0),
      convIdxY(0),
      convExprX(NULL),
      convExprY(NULL),
      outputImage(NULL),
      gidXRef(NULL),
      gidYRef(NULL),
      lidXRef(NULL),
      lidYRef(NULL),
      isWidth(NULL),
      isHeight(NULL),
      local_id_x(NULL),
      local_id_y(NULL),
      local_size_x(NULL),
      local_size_y(NULL),
      block_id_x(NULL),
      block_id_y(NULL),
      block_size_x(NULL),
      block_size_y(NULL),
      grid_size_x(NULL),
      grid_size_y(NULL),
      writeImageRHS(NULL) {
        // typedef unsigned int sampler_t;
        TypeSourceInfo *TInfosampler =
          Ctx.getTrivialTypeSourceInfo(Ctx.UnsignedIntTy);
        samplerTy = TypedefDecl::Create(Ctx, Ctx.getTranslationUnitDecl(),
            SourceLocation(), SourceLocation(), &Ctx.Idents.get("sampler_t"),
            TInfosampler);
        // sampler_t <clKernel>Sampler
        kernelSamplerRef = ASTNode::createDeclRefExpr(Ctx,
            ASTNode::createVarDecl(Ctx, Ctx.getTranslationUnitDecl(),
              kernelDecl->getNameAsString() + "Sampler",
              Ctx.getTypeDeclType(samplerTy), NULL));
        builtins.InitializeBuiltins();
        // debug
        //dump_available_statement_visitors();
        // debug
      }

    Stmt* Hipacc(Stmt *S);

  public:
    // dump all available statement visitors
    static void dump_available_statement_visitors() {
      llvm::errs() << 
        #define STMT(Type, Base) #Base << " *Visit"<< #Type << "(" << #Type << " *" << #Base << ");\n" <<
        #include "clang/AST/StmtNodes.inc"
        "\n\n";
    }

    // C Expressions and Statements
    Stmt *VisitStmt(Stmt *S);
    Stmt *VisitNullStmt(NullStmt *S);
    Stmt *VisitCompoundStmt(CompoundStmt *S);
    Stmt *VisitSwitchCase(SwitchCase *S);
    Stmt *VisitCaseStmt(CaseStmt *S);
    Stmt *VisitDefaultStmt(DefaultStmt *S);
    Stmt *VisitLabelStmt(LabelStmt *S);
    Stmt *VisitIfStmt(IfStmt *S);
    Stmt *VisitSwitchStmt(SwitchStmt *S);
    Stmt *VisitWhileStmt(WhileStmt *S);
    Stmt *VisitDoStmt(DoStmt *S);
    Stmt *VisitForStmt(ForStmt *S);
    Stmt *VisitGotoStmt(GotoStmt *S);
    Stmt *VisitIndirectGotoStmt(IndirectGotoStmt *S);
    Stmt *VisitContinueStmt(ContinueStmt *S);
    Stmt *VisitBreakStmt(BreakStmt *S);
    Stmt *VisitReturnStmt(ReturnStmt *S);
    Stmt *VisitDeclStmt(DeclStmt *S);
    Stmt *VisitAsmStmt(AsmStmt *S);
    Expr *VisitExpr(Expr *E);
    Expr *VisitPredefinedExpr(PredefinedExpr *E);
    Expr *VisitDeclRefExpr(DeclRefExpr *E);
    Expr *VisitLambdaExpr(LambdaExpr *E);
    Expr *VisitIntegerLiteral(IntegerLiteral *E);
    Expr *VisitFloatingLiteral(FloatingLiteral *E);
    Expr *VisitImaginaryLiteral(ImaginaryLiteral *E);
    Expr *VisitStringLiteral(StringLiteral *E);
    Expr *VisitCharacterLiteral(CharacterLiteral *E);
    Expr *VisitParenExpr(ParenExpr *E);
    Expr *VisitParenListExpr(ParenListExpr *E);
    Expr *VisitUnaryOperator(UnaryOperator *E);
    Expr *VisitOffsetOfExpr(OffsetOfExpr *E);
    Expr *VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    Expr *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    Expr *VisitCallExpr(CallExpr *E);
    Expr *VisitMemberExpr(MemberExpr *E);
    Expr *VisitCastExpr(CastExpr *E);
    Expr *VisitBinaryOperator(BinaryOperator *E);
    Expr *VisitCompoundAssignOperator(CompoundAssignOperator *E);
    Expr *VisitAbstractConditionalOperator(AbstractConditionalOperator *E);
    Expr *VisitConditionalOperator(ConditionalOperator *E);
    Expr *VisitBinaryConditionalOperator(BinaryConditionalOperator *E);
    Expr *VisitImplicitCastExpr(ImplicitCastExpr *E);
    Expr *VisitExplicitCastExpr(ExplicitCastExpr *E);
    Expr *VisitCStyleCastExpr(CStyleCastExpr *E);
    Expr *VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    Expr *VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    Expr *VisitInitListExpr(InitListExpr *E);
    Expr *VisitDesignatedInitExpr(DesignatedInitExpr *E);
    Expr *VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    Expr *VisitVAArgExpr(VAArgExpr *E);
    Expr *VisitAddrLabelExpr(AddrLabelExpr *E);
    Expr *VisitStmtExpr(StmtExpr *E);
    Expr *VisitChooseExpr(ChooseExpr *E);
    Expr *VisitGNUNullExpr(GNUNullExpr *E);
    Expr *VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    Expr *VisitBlockExpr(BlockExpr *E);
    Expr *VisitGenericSelectionExpr(GenericSelectionExpr *E);

    // C++ Expressions and Statements
    Stmt *VisitCXXCatchStmt(CXXCatchStmt *S);
    Stmt *VisitCXXTryStmt(CXXTryStmt *S);
    Stmt *VisitCXXForRangeStmt(CXXForRangeStmt *S);

    Expr *VisitUserDefinedLiteral(UserDefinedLiteral *E);
    Expr *VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    Expr *VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    Expr *VisitCXXConstructExpr(CXXConstructExpr *E);
    Expr *VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
    Expr *VisitCXXNamedCastExpr(CXXNamedCastExpr *E);
    Expr *VisitCXXStaticCastExpr(CXXStaticCastExpr *E);
    Expr *VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E);
    Expr *VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E);
    Expr *VisitCXXConstCastExpr(CXXConstCastExpr *E);
    Expr *VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E);
    Expr *VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    Expr *VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    Expr *VisitCXXTypeidExpr(CXXTypeidExpr *E);
    Expr *VisitCXXThisExpr(CXXThisExpr *E);
    Expr *VisitCXXThrowExpr(CXXThrowExpr *E);
    Expr *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E);
    Expr *VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);

    Expr *VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
    Expr *VisitCXXNewExpr(CXXNewExpr *E);
    Expr *VisitCXXDeleteExpr(CXXDeleteExpr *E);
    Expr *VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);

    Expr *VisitExprWithCleanups(ExprWithCleanups *E);
    Expr *VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
    Expr *VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E);
    Expr *VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);

    Expr *VisitOverloadExpr(OverloadExpr *E);
    Expr *VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E);
    Expr *VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E);

    Expr *VisitTypeTraitExpr(TypeTraitExpr *E);
    Expr *VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E);
    Expr *VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E);
    Expr *VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E);
    Expr *VisitExpressionTraitExpr(ExpressionTraitExpr *E);
    Expr *VisitCXXNoexceptExpr(CXXNoexceptExpr *E);
    Expr *VisitPackExpansionExpr(PackExpansionExpr *E);
    Expr *VisitSizeOfPackExpr(SizeOfPackExpr *E);
    Expr *VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E);
    Expr *VisitSubstNonTypeTemplateParmPackExpr(
        SubstNonTypeTemplateParmPackExpr *E);
    Expr *VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
    Expr *VisitOpaqueValueExpr(OpaqueValueExpr *E);
    Expr *VisitAtomicExpr(AtomicExpr *E);

    // CUDA Expressions
    Expr *VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E);


    // OpenCL Expressions
    Expr *VisitAsTypeExpr(AsTypeExpr *E);


    // ObjC Expressions and Statements
    Expr *VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCBridgedCastExpr);
      return NULL;
    }
    Expr *VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
      HIPACC_NOT_SUPPORTED(ObjCArrayLiteral);
      return NULL;
    }
    Expr *VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCBoolLiteralExpr);
      return NULL;
    }
    Expr *VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCBoxedExpr);
      return NULL;
    }
    Expr *VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
      HIPACC_NOT_SUPPORTED(ObjCDictionaryLiteral);
      return NULL;
    }
    Expr *VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCEncodeExpr);
      return NULL;
    }
    Expr *VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCIndirectCopyRestoreExpr);
      return NULL;
    }
    Expr *VisitObjCIsaExpr(ObjCIsaExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCIsaExpr);
      return NULL;
    }
    Expr *VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCIvarRefExpr);
      return NULL;
    }
    Expr *VisitObjCMessageExpr(ObjCMessageExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCMessageExpr);
      return NULL;
    }
    Expr *VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCPropertyRefExpr);
      return NULL;
    }
    Expr *VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCProtocolExpr);
      return NULL;
    }
    Expr *VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCSelectorExpr);
      return NULL;
    }
    Expr *VisitObjCStringLiteral(ObjCStringLiteral *E) {
      HIPACC_NOT_SUPPORTED(ObjCStringLiteral);
      return NULL;
    }
    Expr *VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *E) {
      HIPACC_NOT_SUPPORTED(ObjCSubscriptRefExpr);
      return NULL;
    }
    Stmt *VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAtCatchStmt);
      return NULL;
    }
    Stmt *VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAtFinallyStmt);
      return NULL;
    }
    Stmt *VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAtSynchronizedStmt);
      return NULL;
    }
    Stmt *VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAtThrowStmt);
      return NULL;
    }
    Stmt *VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAtTryStmt);
      return NULL;
    }
    Stmt *VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCAutoreleasePoolStmt);
      return NULL;
    }
    Stmt *VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
      HIPACC_NOT_SUPPORTED(ObjCForCollectionStmt);
      return NULL;
    }
    Stmt *VisitPseudoObjectExpr(PseudoObjectExpr *E) {
      HIPACC_NOT_SUPPORTED(PseudoObjectExpr);
      return NULL;
    }

    // Microsoft Expressions and Statements
    Expr *VisitCXXUuidofExpr(CXXUuidofExpr *E) {
      HIPACC_NOT_SUPPORTED(CXXUuidofExpr);
      return NULL;
    }
    Stmt *VisitMSDependentExistsStmt(MSDependentExistsStmt *S) {
      HIPACC_NOT_SUPPORTED(MSDependentExistsStmt);
      return NULL;
    }
    Stmt *VisitSEHExceptStmt(SEHExceptStmt *S) {
      HIPACC_NOT_SUPPORTED(SEHExceptStmt);
      return NULL;
    }
    Stmt *VisitSEHFinallyStmt(SEHFinallyStmt *S) {
      HIPACC_NOT_SUPPORTED(SEHFinallyStmt);
      return NULL;
    }
    Stmt *VisitSEHTryStmt(SEHTryStmt *S) {
      HIPACC_NOT_SUPPORTED(SEHTryStmt);
      return NULL;
    }
};
} // end namespace hipacc
} // end namespace clang

#endif  // _ASTTRANSLATE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

