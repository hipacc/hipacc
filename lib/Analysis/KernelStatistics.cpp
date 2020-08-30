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

//===--- KernelStatistics.h - Statistics and Analysis of Kernels ----------===//
//
// This file implements various statistics and analysis for source-level CFGs of
// kernel functions.
// Statistics include number of instructions (ALU/SPU) and memory operations
// (global memory, constant memory).
// Analysis include use-def analysis for vectorization.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Analysis/KernelStatistics.h"

#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/StmtVisitor.h>

//#define DEBUG_ANALYSIS

using namespace clang;
using namespace hipacc;

//===----------------------------------------------------------------------===//
// Kernel Statistics and Information
//===----------------------------------------------------------------------===//

namespace {
class KernelStatsImpl {
  private:
    bool verbose_{ false };
  public:
    AnalysisDeclContext &analysisContext;
    llvm::DenseMap<const FieldDecl *, MemoryAccess> memToAccess;
    llvm::DenseMap<const FieldDecl *, MemoryPattern> memToPattern;
    llvm::DenseMap<const VarDecl *, VectorInfo> declsToVector;
    std::map<std::string, VarDecl *> nameToDecls;
    KernelType kernelType;

    ASTContext &Ctx;
    StringRef name;
    FieldDecl *output_image;
    CompilerKnownClasses &compilerClasses;
    DiagnosticsEngine &Diags;
    unsigned DiagIDUnsupportedBO, DiagIDUnsupportedUO,
             DiagIDUnsupportedCSCE, DiagIDUnsupportedTerm,
             DiagIDImageAccess, DiagIDMemIncons;
    unsigned num_ops, num_sops;
    unsigned num_img_loads, num_img_stores;
    unsigned num_mask_loads, num_mask_stores;
    VectorInfo stmtVectorize;
    bool inLambdaFunction;

    void runOnBlock(const CFGBlock *block);
    void runOnAllBlocks();


    KernelStatsImpl(AnalysisDeclContext &ac, StringRef name, FieldDecl
        *output_image, CompilerKnownClasses &compilerClasses, bool verbose=false) :
      verbose_(verbose),
      analysisContext(ac),
      kernelType(),
      Ctx(ac.getASTContext()),
      name(name),
      output_image(output_image),
      compilerClasses(compilerClasses),
      Diags(ac.getASTContext().getDiagnostics()),
      DiagIDUnsupportedBO(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unsupported binary operator on Accessors: %0.")),
      DiagIDUnsupportedUO(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unsupported unary operator on Accessors: %0.")),
      DiagIDUnsupportedCSCE(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unsupported cast operator: %0.")),
      DiagIDUnsupportedTerm(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Unsupported terminal statement: %0.")),
      DiagIDImageAccess(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Accessing image pixels only supported via Accessors and output() function: %0.")),
      DiagIDMemIncons(Diags.getCustomDiagID(DiagnosticsEngine::Error,
            "Pre/post-increment/decrement not supported to assure memory consistency on GPUs: %0.")),
      num_ops(0),
      num_sops(0),
      num_img_loads(0),
      num_img_stores(0),
      num_mask_loads(0),
      num_mask_stores(0),
      stmtVectorize(SCALAR),
      inLambdaFunction(false)
    {}
};
}

static KernelStatsImpl &getImpl(void *x) {
  return *(reinterpret_cast<KernelStatsImpl *>(x));
}


//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

namespace {
class TransferFunctions : public StmtVisitor<TransferFunctions> {
  private:
    KernelStatsImpl &KS;
    bool checkImageAccess(Expr *E, MemoryAccess mem_acc);
    MemoryPattern checkStride(Expr *EX, Expr *EY);

  public:
    explicit TransferFunctions(KernelStatsImpl &ks) :
      KS(ks)
    {}

    void VisitBinaryOperator(BinaryOperator *E);
    void VisitUnaryOperator(UnaryOperator *E);
    void VisitCallExpr(CallExpr *E);
    void VisitCXXMemberCallExpr(CXXMemberCallExpr *) {}
    void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *) {}
    void VisitCStyleCastExpr(CStyleCastExpr *E);
    void VisitDeclStmt(DeclStmt *S);
    void VisitDeclRefExpr(DeclRefExpr *E);
    void VisitLambdaExpr(LambdaExpr *E);
    void VisitReturnStmt(ReturnStmt *S);

    // TODO
    #ifdef DEBUG_ANALYSIS
    void VisitBinaryConditionalOperator(BinaryConditionalOperator *E) { llvm::errs() << "BinaryConditionalOperator:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitConditionalOperator(ConditionalOperator *E) { llvm::errs() << "ConditionalOperator:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitArraySubscriptExpr(ArraySubscriptExpr *E) { llvm::errs() << "ArraySubscriptExpr:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitChooseExpr(ChooseExpr *E) { llvm::errs() << "ChooseExpr:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitCompoundLiteralExpr(CompoundLiteralExpr *E) { llvm::errs() << "CompoundLiteralExpr:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitStmtExpr(StmtExpr *E) { llvm::errs() << "StmtExpr:\n"; E->dump(); llvm::errs() << "\n"; }
    void VisitCaseStmt(CaseStmt *S) { llvm::errs() << "CaseStmt:\n"; S->dump(); llvm::errs() << "\n"; }
    void VisitDefaultStmt(DefaultStmt *S) { llvm::errs() << "DefaultStmt:\n"; S->dump(); llvm::errs() << "\n"; }
    #endif


    // visitors for terminators that branch in the control flow
    void VisitTerminatorBinaryOperator(const BinaryOperator *E);
    void VisitTerminatorConditionalOperator(const ConditionalOperator *E);
    void VisitTerminatorDoStmt(const DoStmt *S);
    void VisitTerminatorIfStmt(const IfStmt *S);
    void VisitTerminatorForStmt(const ForStmt *S);
    void VisitTerminatorWhileStmt(const WhileStmt *S);
    void VisitTerminatorSwitchStmt(const SwitchStmt *S);
};
} // anonymous namespace


void KernelStatsImpl::runOnBlock(const CFGBlock *block) {
  TransferFunctions TF(*this);

  #ifdef DEBUG_ANALYSIS
  block->dump(analysisContext.getCFG(), Ctx.getLangOpts());
  llvm::errs() << "\n=== BlockID " << block->getBlockID() << " ==START\n";
  #endif

  // apply the transfer function for all Stmts in the block.
  for (auto elem : *block) {
    if (auto cfg_stmt = elem.getAs<CFGStmt>()) {
      TF.Visit(const_cast<Stmt*>(cfg_stmt->getStmt()));
    }
  }

  #ifdef DEBUG_ANALYSIS
  // visit the terminator (if any).
  if (const Stmt *term = block->getTerminator()) {
    llvm::errs() << "Successors: \n";

    for (auto succ : block->succs())
      llvm::errs() << suck->getBlockID() << "\n";

    switch (term->getStmtClass()) {
      case Stmt::BinaryOperatorClass:
        TF.VisitTerminatorBinaryOperator(static_cast<const BinaryOperator
            *>(term));
        break;
      case Stmt::ConditionalOperatorClass:
        TF.VisitTerminatorConditionalOperator(static_cast<const
            ConditionalOperator *>(term));
        break;
      case Stmt::DoStmtClass:
        TF.VisitTerminatorDoStmt(static_cast<const DoStmt *>(term));
        break;
      case Stmt::IfStmtClass:
        TF.VisitTerminatorIfStmt(static_cast<const IfStmt *>(term));
        break;
      case Stmt::ForStmtClass:
        TF.VisitTerminatorForStmt(static_cast<const ForStmt *>(term));
        break;
      case Stmt::WhileStmtClass:
        TF.VisitTerminatorWhileStmt(static_cast<const WhileStmt *>(term));
        break;
      case Stmt::SwitchStmtClass:
        TF.VisitTerminatorSwitchStmt(static_cast<const SwitchStmt *>(term));
        break;
      default:
        Diags.Report(term->getLocStart(), DiagIDUnsupportedTerm) <<
          term->getStmtClassName();
        exit(EXIT_FAILURE);
    }
  }
  llvm::errs() << "=== BlockID " << block->getBlockID() << " ==END\n";
  #endif
}


void KernelStatsImpl::runOnAllBlocks() {
  auto POV = analysisContext.getAnalysis<PostOrderCFGView>();
  for (auto block : *POV)
    runOnBlock(block);

  if (verbose_) {
    llvm::errs() << "Kernel statistics for '" << name << "':\n"
                 << "  type: ";
    switch (kernelType) {
      case PointOperator:   llvm::errs() << "Point Operator\n"; break;
      case LocalOperator:   llvm::errs() << "Local Operator\n"; break;
      case GlobalOperator:  llvm::errs() << "Global Operator\n"; break;
      default:
      case UserOperator:    llvm::errs() << "Custom Operator\n"; break;
    }
    llvm::errs() << "  operations (ALU): "    << num_ops << "\n"
                 << "  operations (SFU): "    << num_sops << "\n"
                 << "  image loads: "         << num_img_loads << "\n"
                 << "  image stores: "        << num_img_stores << "\n"
                 << "  mask loads: "          << num_mask_loads << "\n"
                 << "  mask stores: "         << num_mask_stores << "\n";

    llvm::errs() << "  images:\n";
    for (auto map : memToPattern) {
      llvm::errs() << "    " << map.first->getNameAsString() << ": ";
      if (map.second == 0)        llvm::errs() << "UNDEFINED ";
      if (map.second & NO_STRIDE) llvm::errs() << "NO_STRIDE ";
      if (map.second & USER_XY)   llvm::errs() << "USER_XY ";
      if (map.second & STRIDE_X)  llvm::errs() << "STRIDE_X ";
      if (map.second & STRIDE_Y)  llvm::errs() << "STRIDE_Y ";
      if (map.second & STRIDE_XY) llvm::errs() << "STRIDE_XY ";
      llvm::errs() << "\n";
    }

    llvm::errs() << "  VarDecls:\n";
    for (auto map : declsToVector) {
      llvm::errs() << "    " << map.first->getName() << " -> ";

      switch (map.second) {
        case SCALAR:    llvm::errs() << "SCALAR\n"; break;
        case VECTORIZE: llvm::errs() << "VECTORIZE\n"; break;
        case PROPAGATE: llvm::errs() << "PROPAGATE\n"; break;
      }
    }
    if (declsToVector.empty()) llvm::errs() << "    - none -\n";
    llvm::errs() << "\n";
  }
}


//===----------------------------------------------------------------------===//
// Query methods.
//===----------------------------------------------------------------------===//

MemoryAccess KernelStatistics::getMemAccess(const FieldDecl *FD) {
  return getImpl(impl).memToAccess[FD];
}


MemoryPattern KernelStatistics::getMemPattern(const FieldDecl *FD) {
  return getImpl(impl).memToPattern[FD];
}


VectorInfo KernelStatistics::getVectorizeInfo(const VarDecl *VD) {
  return getImpl(impl).declsToVector[VD];
}

VarDecl *KernelStatistics::getVarDeclByName(std::string name) {
  return getImpl(impl).nameToDecls[name];
}

KernelType KernelStatistics::getKernelType() {
  return getImpl(impl).kernelType;
}

unsigned KernelStatistics::getNumOpALUs() {
  return getImpl(impl).num_ops;
}

unsigned KernelStatistics::getNumOpSFUs() {
  return getImpl(impl).num_sops;
}

unsigned KernelStatistics::getNumImgLoads() {
  return getImpl(impl).num_img_loads;
}

MemoryPattern TransferFunctions::checkStride(Expr *EX, Expr *EY) {
  bool stride_x=true, stride_y=true;

  if (auto IL = dyn_cast<IntegerLiteral>(EX->IgnoreParenImpCasts())) {
    if (IL->getValue().getSExtValue()==0) {
      stride_x = false;
    }
  }

  if (auto IL = dyn_cast<IntegerLiteral>(EY->IgnoreParenImpCasts())) {
    if (IL->getValue().getSExtValue()==0) {
      stride_y = false;
    }
  }

  if (stride_x && stride_y) return STRIDE_XY;
  if (stride_x) return STRIDE_X;
  if (stride_y) return STRIDE_Y;
  return NO_STRIDE;
}


bool TransferFunctions::checkImageAccess(Expr *E, MemoryAccess mem_acc) {
  // discard implicit casts and paren expressions
  E = E->IgnoreParenImpCasts();

  // match Image(), Accessor(), Mask(), and Domain() calls
  if (auto call = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (auto ME = dyn_cast<MemberExpr>(call->getArg(0))) {
      if (auto FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
        MemoryPattern mem_pattern = KS.memToPattern[FD];

        // access to Image
        if (KS.compilerClasses.isTypeOfTemplateClass(FD->getType(),
              KS.compilerClasses.Image)) {
          KS.Diags.Report(E->getExprLoc(), KS.DiagIDImageAccess) <<
            FD->getNameAsString();

          exit(EXIT_FAILURE);
        }

        // access to Accessor
        if (KS.compilerClasses.isTypeOfTemplateClass(FD->getType(),
              KS.compilerClasses.Accessor)) {
          if (mem_acc & READ_ONLY) KS.num_img_loads++;
          if (mem_acc & WRITE_ONLY) KS.num_img_stores++;

          switch (call->getNumArgs()) {
            default:
              break;
            case 1:
              mem_pattern = static_cast<MemoryPattern>(mem_pattern|NO_STRIDE);
              if (KS.kernelType < PointOperator) KS.kernelType = PointOperator;
              break;
            case 2:
              // TODO: check for Mask or Domain as parameter and check if we
              // need only STRIDE_X or STRIDE_Y
              mem_pattern = static_cast<MemoryPattern>(mem_pattern|STRIDE_XY);
              if (KS.kernelType < LocalOperator) KS.kernelType = LocalOperator;
              break;
            case 3:
              mem_pattern = static_cast<MemoryPattern>(mem_pattern |
                  checkStride(call->getArg(1), call->getArg(2)));
              if (mem_pattern > NO_STRIDE && KS.kernelType < LocalOperator) {
                KS.kernelType = LocalOperator;
              }
              break;
          }
          KS.memToAccess[FD] =
            static_cast<MemoryAccess>(KS.memToAccess[FD]|mem_acc);
          KS.memToPattern[FD] = mem_pattern;

          return true;
        }

        // access to Mask
        if (KS.compilerClasses.isTypeOfTemplateClass(FD->getType(),
              KS.compilerClasses.Mask)) {
          if (mem_acc & READ_ONLY) KS.num_mask_loads++;
          if (mem_acc & WRITE_ONLY) KS.num_mask_stores++;

          if (KS.inLambdaFunction) {
            // TODO: check for Mask as parameter and check if we need only
            // STRIDE_X or STRIDE_Y
            mem_pattern = static_cast<MemoryPattern>(mem_pattern|STRIDE_XY);
            if (KS.kernelType < LocalOperator) KS.kernelType = LocalOperator;
          } else {
            hipacc_require(call->getNumArgs()==3,
                "Mask access requires x and y parameters!");
            mem_pattern = static_cast<MemoryPattern>(mem_pattern |
                checkStride(call->getArg(1), call->getArg(2)));
            if (mem_pattern > NO_STRIDE && KS.kernelType < LocalOperator) {
              KS.kernelType = LocalOperator;
            }
          }
          KS.memToAccess[FD] =
            static_cast<MemoryAccess>(KS.memToAccess[FD]|mem_acc);
          KS.memToPattern[FD] = mem_pattern;

          return false;
        }

        // access to Domain
        if (KS.compilerClasses.isTypeOfClass(FD->getType(),
              KS.compilerClasses.Domain)) {
          if (mem_acc & READ_ONLY) KS.num_mask_loads++;
          if (mem_acc & WRITE_ONLY) KS.num_mask_stores++;

          if (KS.inLambdaFunction) {
            // TODO: check for Domain as parameter and check if we need only
            // STRIDE_X or STRIDE_Y
            mem_pattern = static_cast<MemoryPattern>(mem_pattern|STRIDE_XY);
            if (KS.kernelType < LocalOperator) KS.kernelType = LocalOperator;
          } else {
            hipacc_require(call->getNumArgs()==3,
                "Domain access requires x and y parameters!");
            mem_pattern = static_cast<MemoryPattern>(mem_pattern |
                checkStride(call->getArg(1), call->getArg(2)));
            if (mem_pattern > NO_STRIDE && KS.kernelType < LocalOperator) {
              KS.kernelType = LocalOperator;
            }
          }
          KS.memToPattern[FD] = mem_pattern;

          return false;
        }
      }
    }
  }

  // match output(), output_at(), and Accessor->pixel_at() calls
  if (auto call = dyn_cast<CXXMemberCallExpr>(E)) {
    if (auto ME = dyn_cast<MemberExpr>(call->getCallee())) {
      if (ME->getMemberNameInfo().getAsString()=="output" ||
          ME->getMemberNameInfo().getAsString()=="output_at" ||
          ME->getMemberNameInfo().getAsString()=="pixel_at") {
        FieldDecl *FD = nullptr;
        if (ME->getMemberNameInfo().getAsString()=="pixel_at") {
          if (auto MEAcc = dyn_cast<MemberExpr>(ME->getBase())) {
            FD = dyn_cast<FieldDecl>(MEAcc->getMemberDecl());
          }
        } else {
          FD = KS.output_image;
        }
        hipacc_require(FD, "could not find field");

        if (mem_acc & READ_ONLY) KS.num_img_loads++;
        if (mem_acc & WRITE_ONLY) KS.num_img_stores++;

        MemoryPattern mem_pattern = KS.memToPattern[FD];

        if (ME->getMemberNameInfo().getAsString()=="output") {
          mem_pattern = static_cast<MemoryPattern>(mem_pattern|NO_STRIDE);
          if (KS.kernelType < PointOperator) KS.kernelType = PointOperator;
        } else {
          mem_pattern = static_cast<MemoryPattern>(mem_pattern|USER_XY);
          KS.kernelType = UserOperator;
        }

        KS.memToAccess[FD] =
          static_cast<MemoryAccess>(KS.memToAccess[FD]|mem_acc);
        KS.memToPattern[FD] = mem_pattern;

        return true;
      }
    }
  }

  return false;
}


void TransferFunctions::VisitBinaryOperator(BinaryOperator *E) {
  DeclRefExpr *DRE = nullptr;

  switch (E->getOpcode()) {
    case BO_PtrMemD:
    case BO_PtrMemI:
    default:
      KS.num_ops++;
      if (checkImageAccess(E->getLHS(), READ_WRITE) ||
          checkImageAccess(E->getRHS(), READ_WRITE)) {
        // not supported on image objects
        KS.Diags.Report(E->getOperatorLoc(), KS.DiagIDUnsupportedBO) <<
          E->getOpcodeStr();
        exit(EXIT_FAILURE);
      }
      break;
    case BO_Mul:
    case BO_Div:
    case BO_Rem:
    case BO_Add:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LT:
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_LAnd:
    case BO_LOr:
      KS.num_ops++;
      if (checkImageAccess(E->getLHS(), READ_ONLY)) {
        KS.stmtVectorize = static_cast<VectorInfo>(KS.stmtVectorize|VECTORIZE);
      }
      if (checkImageAccess(E->getRHS(), READ_ONLY)) {
        KS.stmtVectorize = static_cast<VectorInfo>(KS.stmtVectorize|VECTORIZE);
      }
      break;
    case BO_Assign:
      KS.num_ops++;
      if (checkImageAccess(E->getRHS(), READ_ONLY)) {
        KS.stmtVectorize = static_cast<VectorInfo>(KS.stmtVectorize|VECTORIZE);
      } else {
        if (isa<DeclRefExpr>(E->getRHS()->IgnoreParenImpCasts())) {
          DRE = dyn_cast<DeclRefExpr>(E->getRHS()->IgnoreParenImpCasts());

          if (isa<VarDecl>(DRE->getDecl())) {
            VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());

            KS.declsToVector[VD] =
              static_cast<VectorInfo>(KS.stmtVectorize|KS.declsToVector[VD]);
          }
        }
      }
      if (!checkImageAccess(E->getLHS(), WRITE_ONLY)) {
        if (isa<DeclRefExpr>(E->getLHS())) {
          DRE = dyn_cast<DeclRefExpr>(E->getLHS());

          if (isa<VarDecl>(DRE->getDecl())) {
            VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());

            KS.declsToVector[VD] =
              static_cast<VectorInfo>(KS.stmtVectorize|KS.declsToVector[VD]);
          }
        } else {
          #ifdef DEBUG_ANALYSIS
          llvm::errs() << "==DEBUG==: is not a DRE (LHS):\n";
          E->getLHS()->dump();
          llvm::errs() << "\n";
          #endif
        }
      }
      // reset vectorization status for next statement
      KS.stmtVectorize = SCALAR;

      break;
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      KS.num_ops+=2;
      if (checkImageAccess(E->getRHS(), READ_ONLY)) {
        KS.stmtVectorize = static_cast<VectorInfo>(KS.stmtVectorize|VECTORIZE);
      } else {
        if (isa<DeclRefExpr>(E->getRHS()->IgnoreParenImpCasts())) {
          DRE = dyn_cast<DeclRefExpr>(E->getRHS()->IgnoreParenImpCasts());

          if (isa<VarDecl>(DRE->getDecl())) {
            VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());

            KS.declsToVector[VD] =
              static_cast<VectorInfo>(KS.stmtVectorize|KS.declsToVector[VD]);
          }
        }
      }
      if (!checkImageAccess(E->getLHS(), READ_WRITE)) {
        if (isa<DeclRefExpr>(E->getLHS())) {
          DRE = dyn_cast<DeclRefExpr>(E->getLHS());

          if (isa<VarDecl>(DRE->getDecl())) {
            VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());

            KS.declsToVector[VD] =
              static_cast<VectorInfo>(KS.stmtVectorize|KS.declsToVector[VD]);
          }
        } else {
          #ifdef DEBUG_ANALYSIS
          llvm::errs() << "==DEBUG==: is not a DRE (LHS):\n";
          E->getLHS()->dump();
          llvm::errs() << "\n";
          #endif
        }
      }
      // reset vectorization status for next statement
      KS.stmtVectorize = SCALAR;

      break;
    case BO_Comma:
      break;
  }
}

void TransferFunctions::VisitUnaryOperator(UnaryOperator *E) {
  switch (E->getOpcode()) {
    case UO_AddrOf:
    case UO_Deref:
    case UO_Real:
    case UO_Imag:
    case UO_Extension:
    default:
      KS.num_ops++;
      if (checkImageAccess(E->getSubExpr(), READ_WRITE)) {
        // not supported on image objects
        KS.Diags.Report(E->getOperatorLoc(), KS.DiagIDUnsupportedUO) <<
          E->getOpcodeStr(E->getOpcode());
        exit(EXIT_FAILURE);
      }
      break;
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
      KS.num_ops++;
      if (checkImageAccess(E->getSubExpr(), READ_WRITE)) {
        // not supported - memory inconsistency
        KS.Diags.Report(E->getOperatorLoc(), KS.DiagIDMemIncons) <<
          E->getOpcodeStr(E->getOpcode());
        exit(EXIT_FAILURE);
      }
      break;
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_LNot:
      KS.num_ops++;
      checkImageAccess(E->getSubExpr(), READ_ONLY);
      break;
  }
  // reset vectorization status for next statement
  KS.stmtVectorize = SCALAR;
}

void TransferFunctions::VisitCallExpr(CallExpr *E) {
  for (auto arg : E->arguments())
    checkImageAccess(arg, READ_ONLY);
  KS.num_sops++;
}

void TransferFunctions::VisitCStyleCastExpr(CStyleCastExpr *E) {
  switch (E->getCastKind()) {
    case CK_NoOp:
      break;
    case CK_IntegralCast:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
      KS.num_ops++;
      break;
    default:
      KS.Diags.Report(E->getLParenLoc(), KS.DiagIDUnsupportedCSCE) <<
        E->getCastKindName();
      exit(EXIT_FAILURE);
  }
  checkImageAccess(E->getSubExpr(), READ_ONLY);
}

void TransferFunctions::VisitDeclStmt(DeclStmt *S) {
  // iterate over all declarations of this DeclStmt
  for (auto decl : S->decls()) {
    if (isa<VarDecl>(decl)) {
      VarDecl *VD = dyn_cast<VarDecl>(decl);
      KS.nameToDecls[VD->getName()] = VD; // for kernel fusion
      if (VD->hasInit()) {
        if (checkImageAccess(VD->getInit(), READ_ONLY)) {
          KS.stmtVectorize =
            static_cast<VectorInfo>(KS.stmtVectorize|VECTORIZE);
        }
      }
      KS.declsToVector[VD] = KS.stmtVectorize;
    }
  }

  // reset vectorization status for next statement
  KS.stmtVectorize = SCALAR;
}

void TransferFunctions::VisitDeclRefExpr(DeclRefExpr *E) {
  if (isa<VarDecl>(E->getDecl())) {
    VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());

    // update vectorization information for current statement only if the
    // referenced variable is a vector
    if (KS.declsToVector.count(VD)) {
      KS.stmtVectorize =
        static_cast<VectorInfo>(KS.stmtVectorize|KS.declsToVector[VD]);
    }
  }
}

void TransferFunctions::VisitLambdaExpr(LambdaExpr *E) {
  AnalysisDeclContext AC(/* AnalysisDeclContextManager */ 0, E->getCallOperator());
  KernelStatistics::setAnalysisOptions(AC);

  hipacc_require(AC.getCFG(), "Could not get CFG from lambda-function.");
  #ifdef DEBUG_ANALYSIS
  AC.getCFG()->viewCFG(KS.Ctx.getLangOpts());
  #endif

  KS.inLambdaFunction = true;
  auto POV = AC.getAnalysis<PostOrderCFGView>();
  for (auto block : *POV)
    KS.runOnBlock(block);
  KS.inLambdaFunction = false;
}

void TransferFunctions::VisitReturnStmt(ReturnStmt *S) {
  if (S->getRetValue()) checkImageAccess(S->getRetValue(), READ_ONLY);
}


#ifdef DEBUG_ANALYSIS
void TransferFunctions::VisitTerminatorBinaryOperator(const BinaryOperator *E) {
  llvm::errs() << "BinaryOperator: \n";
  E->getLHS()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << " " << E->getOpcodeStr() << " ";
  E->getRHS()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << "\n";
}

void TransferFunctions::VisitTerminatorConditionalOperator(const
    ConditionalOperator *E) {
  llvm::errs() << "ConditionalOperator: \n";
  E->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << "?";
  E->getTrueExpr()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ":";
  E->getFalseExpr()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << "\n";
}

void TransferFunctions::VisitTerminatorDoStmt(const DoStmt *S) {
  llvm::errs() << "DoStmt: \n";
  llvm::errs() << "do {} while (";
  S->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ")\n";
}

void TransferFunctions::VisitTerminatorIfStmt(const IfStmt *S) {
  llvm::errs() << "IfStmt: \n";
  llvm::errs() << "if (";
  S->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ") {}\n";
}

void TransferFunctions::VisitTerminatorForStmt(const ForStmt *S) {
  llvm::errs() << "ForStmt: \n";
  llvm::errs() << "for (";
  S->getInit()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ", ";
  S->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ", ";
  S->getInc()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ") {}\n";
}

void TransferFunctions::VisitTerminatorWhileStmt(const WhileStmt *S) {
  llvm::errs() << "WhileStmt: \n";
  llvm::errs() << "while (";
  S->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ") {}\n";
}

void TransferFunctions::VisitTerminatorSwitchStmt(const SwitchStmt *S) {
  llvm::errs() << "SwitchStmt: \n";
  llvm::errs() << "switch (";
  S->getCond()->printPretty(llvm::errs(), 0,
      PrintingPolicy(KS.Ctx.getLangOpts()));
  llvm::errs() << ") {}\n";
}
#endif

//===----------------------------------------------------------------------===//
// External interface to run read/write only analysis.
//===----------------------------------------------------------------------===//


KernelStatistics::KernelStatistics(void *im) : impl(im) {}

KernelStatistics::~KernelStatistics() {
  delete reinterpret_cast<KernelStatsImpl*>(impl);
}

KernelStatistics *KernelStatistics::computeKernelStatistics(AnalysisDeclContext
    &AC, StringRef name, FieldDecl *output_image, CompilerKnownClasses
    &compilerClasses, bool verbose) {
  // No CFG?  Bail out.
  CFG *cfg = AC.getCFG();
  if (!cfg) return 0;

  #ifdef DEBUG_ANALYSIS
  cfg->viewCFG(AC.getASTContext().getLangOpts());
  #endif

  KernelStatsImpl *KS = new KernelStatsImpl(AC, name, output_image,
      compilerClasses, verbose);
  KS->runOnAllBlocks();

  return new KernelStatistics(KS);
}

// vim: set ts=2 sw=2 sts=2 et ai:

