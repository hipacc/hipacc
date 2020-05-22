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

//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Device/Builtins.h"

#include <clang/Basic/TargetInfo.h>

using namespace clang;
using namespace hipacc;
using namespace hipacc::Builtin;

static hipacc::Builtin::Info BuiltinInfo[] = {
  { "not a builtin function", 0, Language::C99, static_cast<ID>(0), static_cast<ID>(0), 0 },
  #define HIPACCBUILTIN(NAME, TYPE, CUDAID, OPENCLID) { #NAME, TYPE, Language::C99, CUDAID, OPENCLID, 0 },
  #define CUDABUILTIN(NAME, TYPE, CUDANAME) { #NAME, TYPE, Language::CUDA, (ID)0, (ID)0, 0 },
  #define OPENCLBUILTIN(NAME, TYPE, OPENCLNAME) { #NAME, TYPE, Language::OpenCLCPU, (ID)0, (ID)0, 0 },
  #include "hipacc/Device/Builtins.def"
};


// DecodeTypeFromStr - This decodes one type descriptor from Str, advancing the
// pointer over the consumed characters. This returns the resultant type. If
// AllowTypeModifiers is false then modifier like * are not parsed, just basic
// types. This allows "v2i*" to be parsed as a pointer to a v2i instead of a
// vector of "i*".
static QualType DecodeTypeFromStr(const char *&Str, const ASTContext &Ctx, bool
    AllowTypeModifiers) {
  // Modifiers.
  int HowLong = 0;
  bool Signed = false, Unsigned = false;

  // Read the prefixed modifiers first.
  bool Done = false;
  while (!Done) {
    switch (*Str++) {
      default: Done = true; --Str; break;
      case 'S':
        hipacc_require(!Unsigned, "Can't use both 'S' and 'U' modifiers!");
        hipacc_require(!Signed, "Can't use 'S' modifier multiple times!");
        Signed = true;
        break;
      case 'U':
        hipacc_require(!Signed, "Can't use both 'S' and 'U' modifiers!");
        hipacc_require(!Unsigned, "Can't use 'U' modifier multiple times!");
        Unsigned = true;
        break;
      case 'L':
        hipacc_require(HowLong <= 2, "Can't have LLLL modifier");
        ++HowLong;
        break;
      case 'W':
        // This modifier represents int64 type.
        hipacc_require(HowLong == 0, "Can't use both 'L' and 'W' modifiers!");
        switch (Ctx.getTargetInfo().getInt64Type()) {
          default:
            hipacc_require(0, "Unexpected integer type");
          case TargetInfo::SignedLong:
            HowLong = 1;
            break;
          case TargetInfo::SignedLongLong:
            HowLong = 2;
            break;
        }
    }
  }

  QualType Type;

  // Read the base type.
  switch (*Str++) {
    default: hipacc_require(0, "Unknown builtin type letter!");
    case 'v': // void
      hipacc_require(HowLong == 0 && !Signed && !Unsigned,
             "Bad modifiers used with 'v'!");
      Type = Ctx.VoidTy;
      break;
    case 'f': // float
      hipacc_require(HowLong == 0 && !Signed && !Unsigned,
             "Bad modifiers used with 'f'!");
      Type = Ctx.FloatTy;
      break;
    case 'd': // double
      hipacc_require(HowLong < 2 && !Signed && !Unsigned,
             "Bad modifiers used with 'd'!");
      if (HowLong)
        Type = Ctx.LongDoubleTy;
      else
        Type = Ctx.DoubleTy;
      break;
    case 's': // short
      hipacc_require(HowLong == 0, "Bad modifiers used with 's'!");
      if (Unsigned)
        Type = Ctx.UnsignedShortTy;
      else
        Type = Ctx.ShortTy;
      break;
    case 'i': // int
      if (HowLong == 3)
        Type = Unsigned ? Ctx.UnsignedInt128Ty : Ctx.Int128Ty;
      else if (HowLong == 2)
        Type = Unsigned ? Ctx.UnsignedLongLongTy : Ctx.LongLongTy;
      else if (HowLong == 1)
        Type = Unsigned ? Ctx.UnsignedLongTy : Ctx.LongTy;
      else
        Type = Unsigned ? Ctx.UnsignedIntTy : Ctx.IntTy;
      break;
    case 'c': // char
      hipacc_require(HowLong == 0, "Bad modifiers used with 'c'!");
      if (Signed)
        Type = Ctx.SignedCharTy;
      else if (Unsigned)
        Type = Ctx.UnsignedCharTy;
      else
        Type = Ctx.CharTy;
      break;
    case 'b': // boolean
      hipacc_require(HowLong == 0 && !Signed && !Unsigned, "Bad modifiers for 'b'!");
      Type = Ctx.BoolTy;
      break;
    case 'z':  // size_t.
      hipacc_require(HowLong == 0 && !Signed && !Unsigned, "Bad modifiers for 'z'!");
      Type = Ctx.getSizeType();
      break;
    case 'V': { // Vector
      char *End;
      unsigned NumElements = strtoul(Str, &End, 10);
      hipacc_require(End != Str, "Missing vector size");
      Str = End;

      QualType ElementType = DecodeTypeFromStr(Str, Ctx, true);

      Type = Ctx.getVectorType(ElementType, NumElements,
                               VectorType::GenericVector);
      break;
    }
    case 'E': { // Extended Vector
      char *End;
      unsigned NumElements = strtoul(Str, &End, 10);
      hipacc_require(End != Str, "Missing vector size");
      Str = End;

      QualType ElementType = DecodeTypeFromStr(Str, Ctx, false);

      Type = Ctx.getExtVectorType(ElementType, NumElements);
      break;
    }
  }

  // If there are modifiers and if we're allowed to parse them, go for it.
  Done = !AllowTypeModifiers;
  while (!Done) {
    switch (char c = *Str++) {
      default: Done = true; --Str; break;
      case '*':
      case '&': {
        // Both pointers and references can have their pointee types
        // qualified with an address space.
        char *End;
        unsigned AddrSpace = strtoul(Str, &End, 10);
        if (End != Str && AddrSpace != 0) {
          Type = Ctx.getAddrSpaceQualType(Type,
                                          getLangASFromTargetAS(AddrSpace));
          Str = End;
        }
        if (c == '*')
          Type = Ctx.getPointerType(Type);
        else
          Type = Ctx.getLValueReferenceType(Type);
        break;
        }
        // FIXME: There's no way to have a built-in with an rvalue ref arg.
      case 'C':
        Type = Type.withConst();
        break;
      case 'D':
        Type = Ctx.getVolatileType(Type);
        break;
      case 'R':
        Type = Type.withRestrict();
        break;
    }
  }

  return Type;
}


// EncodeTypeIntoStr - This encodes one QualType QT into a sting representation
// that is compatible with DecodeTypeFromStr.
std::string hipacc::Builtin::Context::EncodeTypeIntoStr(QualType QT, const
    ASTContext &Ctx) {
  if (QT->isVectorType()) {
    auto vec_string = QT->isExtVectorType() ? "E" : "V";
    auto vec_type = QT->getAs<VectorType>();
    auto lanes = vec_type->getNumElements();
    return vec_string + std::to_string(lanes) +
      EncodeTypeIntoStr(vec_type->getElementType(), Ctx);
  }

  const BuiltinType *BT = QT->getAs<BuiltinType>();

  switch (BT->getKind()) {
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::LongDouble:
    case BuiltinType::Void:
    case BuiltinType::Bool:
    default: hipacc_require(0, "BuiltinType for Boundary handling constant not supported.");
    case BuiltinType::Char_S:
    case BuiltinType::SChar:      return "Sc";
    case BuiltinType::Char_U:
    case BuiltinType::UChar:      return "Uc";
    case BuiltinType::Short:      return "s";
    case BuiltinType::Char16:
    case BuiltinType::UShort:     return "Us";
    case BuiltinType::Int:        return "i";
    case BuiltinType::Char32:
    case BuiltinType::UInt:       return "Ui";
    case BuiltinType::Long:       return "Li";
    case BuiltinType::ULong:      return "ULi";
    case BuiltinType::LongLong:   return "LLi";
    case BuiltinType::ULongLong:  return "ULLi";
    case BuiltinType::UInt128:    return "LLLi";
    case BuiltinType::Int128:     return "ULLLi";
    case BuiltinType::Float:      return "f";
    case BuiltinType::Double:     return "d";
  }
}


// getBuiltinType - Return the type for the specified builtin.
QualType hipacc::Builtin::Context::getBuiltinType(unsigned Id) const {
  return getBuiltinType(getTypeString(Id));
}

QualType hipacc::Builtin::Context::getBuiltinType(const char *TypeStr) const {
  SmallVector<QualType, 8> ArgTypes;

  QualType ResType = DecodeTypeFromStr(TypeStr, Ctx, true);

  while (TypeStr[0] && TypeStr[0] != '.') {
    QualType Ty = DecodeTypeFromStr(TypeStr, Ctx, true);

    // Do array -> pointer decay.  The builtin should use the decayed type.
    if (Ty->isArrayType())
      Ty = Ctx.getArrayDecayedType(Ty);

    ArgTypes.push_back(Ty);
  }

  FunctionType::ExtInfo EI;
  bool Variadic = (TypeStr[0] == '.');

  // We really shouldn't be making a no-proto type here, especially in C++.
  if (ArgTypes.empty() && Variadic)
    return Ctx.getFunctionNoProtoType(ResType, EI);

  FunctionProtoType::ExtProtoInfo EPI;
  EPI.ExtInfo = EI;
  EPI.Variadic = Variadic;

  return Ctx.getFunctionType(ResType, ArgTypes, EPI);
}


const hipacc::Builtin::Info &hipacc::Builtin::Context::getRecord(unsigned ID)
  const {
  assert(ID < (LastBuiltin-FirstBuiltin));

  return BuiltinInfo[ID];
}


// InitializeBuiltins - Mark the identifiers for all the builtins with their
// appropriate builtin ID # and mark any non-portable builtin identifiers as
// such.
void hipacc::Builtin::Context::InitializeBuiltins() {
#if 0
  clang::Builtin::Context &bctx = Ctx.BuiltinInfo;

  llvm::errs() << "==================\n"
               << "Standard Builtins:\n"
               << "==================\n";
  for (size_t i = clang::Builtin::NotBuiltin+1; i !=
      clang::Builtin::FirstTSBuiltin; ++i) {
    llvm::errs() << i << ": " << bctx.GetName(i) << "\n";
  }

  const clang::Builtin::Info *lTSRecords = 0;
  unsigned lNumTSRecords = 0;
  Ctx.getTargetInfo().getTargetBuiltins(lTSRecords, lNumTSRecords);
  llvm::errs() << "================\n"
               << "Target Builtins:\n"
               << "================\n";
  for (size_t i=0, e = lNumTSRecords; i != e; ++i) {
    llvm::errs() << i << ": "
                 << bctx.GetName(i+clang::Builtin::FirstTSBuiltin) << "\n";
  }
  llvm::errs() << "================\n"
               << "Hipacc Builtins:\n"
               << "================\n";
  for (size_t i=FirstBuiltin+1; i!=LastBuiltin; ++i) {
    llvm::errs() << i << ": " << getName(i-FirstBuiltin) << "\n";
  }
#endif

  if (initialized) return;

  for (size_t i=1, e=LastBuiltin-FirstBuiltin; i!=e; ++i) {
    BuiltinInfo[i].FD = CreateBuiltin(i);
  }

  initialized = true;
}


void hipacc::Builtin::Context::getBuiltinNames(Language lang,
    SmallVectorImpl<const char *> &Names) {
  for (size_t i=1, e=LastBuiltin-FirstBuiltin; i!=e; ++i) {
    switch (BuiltinInfo[i].builtin_lang) {
      case Language::C99:
        switch (lang) {
          case Language::C99: break;
          case Language::CUDA:
            if (!getBuiltinFunction(BuiltinInfo[i].CUDA)) continue;
            break;
          case Language::OpenCLACC:
          case Language::OpenCLCPU:
          case Language::OpenCLGPU:
            if (!getBuiltinFunction(BuiltinInfo[i].OpenCL)) continue;
            break;
        }
        break;
      case Language::CUDA:
        if (lang == Language::CUDA) break;
        continue;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (lang == Language::OpenCLACC ||
            lang == Language::OpenCLCPU ||
            lang == Language::OpenCLGPU) break;
        continue;
    }
    Names.push_back(BuiltinInfo[i].Name);
  }
}


// LazilyCreateBuiltin - The specified Builtin-ID was first used at file scope.
// lazily create a decl for it.
FunctionDecl *hipacc::Builtin::Context::CreateBuiltin(unsigned bid) {
  return CreateBuiltin(getBuiltinType(bid), BuiltinInfo[bid].Name);
}
FunctionDecl *hipacc::Builtin::Context::CreateBuiltin(QualType R, const char
    *Name) {
  // create function name identifier
  IdentifierInfo &Info = Ctx.Idents.get(Name);
  DeclarationName DecName(&Info);

  FunctionDecl *New = FunctionDecl::Create(Ctx, Ctx.getTranslationUnitDecl(),
      SourceLocation(), SourceLocation(), DecName, R, nullptr, SC_None);

  New->setImplicit();

  // create Decl objects for each parameter, adding them to the FunctionDecl.
  if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(R)) {
    SmallVector<ParmVarDecl *, 16> Params;
    size_t num_parm = 0;
    for (auto ptype : FT->param_types()) {
      auto parm = ParmVarDecl::Create(Ctx, New, SourceLocation(),
          SourceLocation(), 0, ptype, /*TInfo=*/0, SC_None, 0);
      parm->setScopeInfo(0, num_parm++);
      Params.push_back(parm);
    }
    New->setParams(Params);
  }

  Ctx.getTranslationUnitDecl()->addDecl(New);

  return New;
}


FunctionDecl *hipacc::Builtin::Context::getBuiltinFunction(StringRef Name,
    QualType QT, Language lang) const {
  QT = QT.getDesugaredType(Ctx);

  for (size_t i=1, e=LastBuiltin-FirstBuiltin; i!=e; ++i) {
    if (BuiltinInfo[i].Name == Name && BuiltinInfo[i].FD->getReturnType() == QT) {
      switch (BuiltinInfo[i].builtin_lang) {
        case Language::C99:
          switch (lang) {
            case Language::C99: return nullptr;
            case Language::CUDA:
              return getBuiltinFunction(BuiltinInfo[i].CUDA);
            case Language::OpenCLACC:
            case Language::OpenCLCPU:
            case Language::OpenCLGPU:
              return getBuiltinFunction(BuiltinInfo[i].OpenCL);
          }
          break;
        case Language::CUDA:
          if (lang == Language::CUDA) return BuiltinInfo[i].FD;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          if (lang == Language::OpenCLACC ||
              lang == Language::OpenCLCPU ||
              lang == Language::OpenCLGPU) return BuiltinInfo[i].FD;
      }
    }
  }

  return nullptr;
}

// vim: set ts=2 sw=2 sts=2 et ai:

