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

//===--- InstructionSets.h - Definition of known vector instruction sets. ------------===//
//
// This file contains definitions of known vector instruction sets.
//
//===---------------------------------------------------------------------------------===//

#ifndef _HIPACC_BACKEND_INSTRUCTION_SETS_H_
#define _HIPACC_BACKEND_INSTRUCTION_SETS_H_

#include "hipacc/Backend/BackendExceptions.h"
#include "hipacc/Backend/ClangASTHelper.h"
#include "hipacc/Backend/VectorizationAST.h"

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

//#define VERBOSE_INIT_MODE 1   // Uncomment this for a print-out of the inited intrinsic functions

#ifdef VERBOSE_INIT_MODE
#include "llvm/Support/raw_ostream.h"
#endif


namespace clang
{
namespace hipacc
{
namespace Backend
{
namespace Vectorization
{
  typedef AST::BaseClasses::TypeInfo::KnownTypes                          VectorElementTypes;       //!< Type alias for the enumeration of all possible vector element types.
  typedef AST::Expressions::UnaryOperator::UnaryOperatorType              UnaryOperatorType;        //!< Type alias for the enumeration of all possible unary operators.
  typedef AST::Expressions::ArithmeticOperator::ArithmeticOperatorType    ArithmeticOperatorType;   //!< Type alias for the enumeration of all possible binary arithmetic operators.
  typedef AST::Expressions::RelationalOperator::RelationalOperatorType    RelationalOperatorType;   //!< Type alias for the enumeration of all possible binary relational operators.
  typedef AST::VectorSupport::CheckActiveElements::CheckType              ActiveElementsCheckType;  //!< Type alias for the enumeration of all possible binary mask element check types.


  /** \brief  Enumeration of supported special built-in vector functions. */
  enum class BuiltinFunctionsEnum
  {
    Abs,              //!< Internal ID for the element-wise <b>absolute value</b> function.
    Ceil,             //!< Internal ID for the element-wise <b>round to next larger integer</b> function.
    Floor,            //!< Internal ID for the element-wise <b>round to next smaller integer</b> function.
    Max,              //!< Internal ID for the element-wise <b>maximum value</b> function.
    Min,              //!< Internal ID for the element-wise <b>minimum value</b> function.
    Sqrt,             //!< Internal ID for the element-wise <b>square root</b> function.
    UnknownFunction   //!< Internal ID used as an indicator for unknown functions.
  };

  /** \brief  Returns the string-identifer of a specific built-in vector function.
   *  \param  eFunctionType   The internal ID of requested built-in function. */
  inline std::string GetBuiltinFunctionTypeString(BuiltinFunctionsEnum eFunctionType)
  {
    switch (eFunctionType)
    {
    case BuiltinFunctionsEnum::Abs:     return "Abs";
    case BuiltinFunctionsEnum::Ceil:    return "Ceil";
    case BuiltinFunctionsEnum::Floor:   return "Floor";
    case BuiltinFunctionsEnum::Max:     return "Max";
    case BuiltinFunctionsEnum::Min:     return "Min";
    case BuiltinFunctionsEnum::Sqrt:    return "Sqrt";
    default:                            throw InternalErrorException("Unknown built-in function type!");
    }
  }


  /** \brief  Contains common exceptions, which can be thrown by the instruction set abstraction layer. */
  class InstructionSetExceptions final
  {
  public:

    /** \brief  Indicates that a specified index has been out of range. */
    class IndexOutOfRange : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;

      static std::string _ConvertLimit(std::uint32_t uiUpperLimit);

    protected:

      IndexOutOfRange(std::string strMethodType, VectorElementTypes eElementType, std::uint32_t uiUpperLimit);
    };

    /** \brief  Indicates that a specified index for an extraction operation has been out of range. */
    class ExtractIndexOutOfRange final : public IndexOutOfRange
    {
    private:

      typedef IndexOutOfRange   BaseType;

    public:

      inline ExtractIndexOutOfRange(VectorElementTypes eElementType, std::uint32_t uiUpperLimit)  : BaseType("extraction", eElementType, uiUpperLimit)   {}
    };

    /** \brief  Indicates that a specified index for an extraction operation has been out of range. */
    class InsertIndexOutOfRange final : public IndexOutOfRange
    {
    private:

      typedef IndexOutOfRange   BaseType;

    public:

      inline InsertIndexOutOfRange(VectorElementTypes eElementType, std::uint32_t uiUpperLimit)   : BaseType("insertion", eElementType, uiUpperLimit)  {}
    };

    /** \brief  Indicates that a requested special built-in vector function is not supported by a particular instruction set. */
    class UnsupportedBuiltinFunctionType final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;

      static std::string _ConvertParamCount(std::uint32_t uiParamCount);

    public:

      UnsupportedBuiltinFunctionType(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount, std::string strInstructionSetName);
    };

    /** \brief  Indicates that a requested vector conversion operation is not supported by a particular instruction set. */
    class UnsupportedConversion final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;

    public:

      UnsupportedConversion(VectorElementTypes eSourceType, VectorElementTypes eTargetType, std::string strInstructionSetName);
    };
  };


  /** \brief  Base class for all vector instruction-set implementations. */
  class InstructionSetBase
  {
  protected:

    typedef std::pair<std::string, ::clang::FunctionDecl*> IntrinsicInfoPairType;    //!< Container for the name and function declaration object of an intrinsic function.

    /** \brief  Generic type alias for the intrinsic function lookup tables of the instruction sets. */
    template <typename IntrinsicIDType> using IntrinsicMapTemplateType = std::map<IntrinsicIDType, IntrinsicInfoPairType>;

  private:

    typedef std::map<std::string, ClangASTHelper::FunctionDeclarationVectorType> FunctionDeclMapType; //!< Internal type definition for a lookup table of function declarations.

    ClangASTHelper        _ASTHelper;           //!< The ClangASTHelper object encapsulating the current Clang AST context.
    FunctionDeclMapType   _mapKnownFuncDecls;   //!< The lookup table of all known function declarations for each compilation run.

    std::string           _strIntrinsicPrefix;  //!< The common prefix for all intrinsic functions of the derived instruction set (used to filter all known function declarations).

  protected:

    /** \brief  Returns a reference to the current ClangASTHelper object. */
    inline ClangASTHelper& _GetASTHelper()   { return _ASTHelper; }


    /** \brief  Returns an expression objects, which casts a pointer from one type to another.
     *  \param  pPointerRef       A pointer to an expression object, which returns the pointer that shall be casted.
     *  \param  crNewPointerType  The requested qualified Clang-type of the casted pointer. */
    ::clang::CastExpr* _CreatePointerCast(::clang::Expr *pPointerRef, const ::clang::QualType &crNewPointerType);

    /** \brief  Returns an expression objects, which casts a scalar value into another type.
     *  \param  pValueRef       A pointer to an expression object, which returns the scalar value that shall be casted.
     *  \param  crNewValueType  The requested qualified Clang-type of the casted scalar value.
     *  \param  eCastKind       The internal ID of the Clang-specific cast type. */
    ::clang::CastExpr* _CreateValueCast(::clang::Expr *pValueRef, const ::clang::QualType &crNewValueType, ::clang::CastKind eCastKind);

    /** \brief  Returns the qualified Clang-specific type for a certain element type (not the vector).
     *  \param  eType   The element type, whose Clang-specific counterpart shall be returned. */
    ::clang::QualType _GetClangType(VectorElementTypes eType);

    /** \brief  Internal function, which returns all known function declaration objects with the specified name.
     *  \param  strFunctionName   The fully qualified name of the requested function. */
    ClangASTHelper::FunctionDeclarationVectorType _GetFunctionDecl(std::string strFunctionName);

    /** \brief  Internal function, which returns the qualified Clang-specific type of a vector variable with a specified element type.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  bIsConst      Specifies, whether the returned type shall be flagged with the <b>const</b> attribute. */
    ::clang::QualType _GetVectorType(VectorElementTypes eElementType, bool bIsConst);


    /** \brief  Generic base function for the creation of function call expression objects to intrinsic functions.
     *  \tparam IntrinsicIDType   The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  crIntrinMap       A reference to the currently used intrinsic function lookup table.
     *  \param  eIntrinID         The internal ID of the requested intrinsic function.
     *  \param  crvecArguments    A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    template <typename IntrinsicIDType>
    inline ::clang::CallExpr* _CreateFunctionCall(const IntrinsicMapTemplateType<IntrinsicIDType> &crIntrinMap, IntrinsicIDType eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      auto itIntrinEntry = crIntrinMap.find(eIntrinID);
      if (itIntrinEntry == crIntrinMap.end())
      {
        throw InternalErrorException("The specified intrinsic is unknown!");
      }

      ::clang::FunctionDecl *pIntrinsicDecl = itIntrinEntry->second.second;
      if (pIntrinsicDecl == nullptr)
      {
        throw InternalErrorException(std::string("The intrinsic \"") + _strIntrinsicPrefix + itIntrinEntry->second.first + std::string("\" has not been initialized!"));
      }

      return _ASTHelper.CreateFunctionCall( pIntrinsicDecl, crvecArguments );
    }

    /** \brief  Generic base function for the creation of expression objects, which perform a post-fixed unary increment or decrement operation on all elements of a vector.
     *  \tparam IntrinsicIDType   The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  crIntrinMap       A reference to the currently used intrinsic function lookup table.
     *  \param  eIntrinID         The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType      The element type stored in the vector.
     *  \param  pVectorRef        A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    template <typename IntrinsicIDType>
    inline ::clang::Expr* _CreatePostfixedUnaryOp(const IntrinsicMapTemplateType<IntrinsicIDType> &crIntrinMap, IntrinsicIDType eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      // Create the assignment expression which does the computation (identical to the prefixed counterpart)
      ::clang::Expr *pAssignment = _CreatePrefixedUnaryOp( crIntrinMap, eIntrinID, eElementType, pVectorRef );

      // Create a reversion expression which is required to restore the old value
      ::clang::Expr *pRevert = nullptr;
      {
        ClangASTHelper::ExpressionVectorType vecArgs;
        vecArgs.push_back( pVectorRef );
        vecArgs.push_back( CreateOnesVector(eElementType, true) );

        pRevert = _CreateFunctionCall( crIntrinMap, eIntrinID, vecArgs );
      }

      // Wrap every thing into a comma operator
      return _GetASTHelper().CreateParenthesisExpression( _GetASTHelper().CreateBinaryOperatorComma(pAssignment, pRevert) );
    }

    /** \brief  Generic base function for the creation of expression objects, which perform a pre-fixed unary increment or decrement operation on all elements of a vector.
     *  \tparam IntrinsicIDType   The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  crIntrinMap       A reference to the currently used intrinsic function lookup table.
     *  \param  eIntrinID         The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType      The element type stored in the vector.
     *  \param  pVectorRef        A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    template <typename IntrinsicIDType>
    inline ::clang::Expr* _CreatePrefixedUnaryOp(const IntrinsicMapTemplateType<IntrinsicIDType> &crIntrinMap, IntrinsicIDType eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      ClangASTHelper::ExpressionVectorType vecArgs;
      vecArgs.push_back( pVectorRef );
      vecArgs.push_back( CreateOnesVector(eElementType, false) );

      ::clang::Expr *pPrefixOp = _CreateFunctionCall(crIntrinMap, eIntrinID, vecArgs);

      return _GetASTHelper().CreateParenthesisExpression( _GetASTHelper().CreateBinaryOperator( pVectorRef, pPrefixOp, BO_Assign, pVectorRef->getType() ) );
    }

    /** \brief  Generic base function for the retrival of the qualified Clang return type of an intrinsic function.
     *  \tparam IntrinsicIDType   The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  crIntrinMap       A reference to the currently used intrinsic function lookup table.
     *  \param  eIntrinID         The internal ID of the requested intrinsic function, whose return type shall be retrieved. */
    template <typename IntrinsicIDType>
    inline ::clang::QualType _GetFunctionReturnType(const IntrinsicMapTemplateType<IntrinsicIDType> &crIntrinMap, IntrinsicIDType eIntrinID)
    {
      auto itIntrinEntry = crIntrinMap.find(eIntrinID);
      if (itIntrinEntry == crIntrinMap.end())
      {
        throw InternalErrorException("The specified intrinsic is unknown!");
      }

      ::clang::FunctionDecl *pIntrinsicDecl = itIntrinEntry->second.second;
      if (pIntrinsicDecl == nullptr)
      {
        throw InternalErrorException(std::string("The intrinsic \"") + itIntrinEntry->second.first + std::string("\" has not been initialized!"));
      }

      return pIntrinsicDecl->getReturnType();
    }


    /** \brief  Generic internal base function for creating the association between intrinsic function IDs and names.
     *  \tparam IntrinsicIDType   The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  rIntrinMap        A reference to the currently used intrinsic function lookup table.
     *  \param  eIntrinID         The internal ID of the intrinsic function.
     *  \param  strIntrinName     The name of the intrinsic function. */
    template <typename IntrinsicIDType>
    inline void _InitIntrinsic(IntrinsicMapTemplateType<IntrinsicIDType> &rIntrinMap, IntrinsicIDType eIntrinID, std::string strIntrinName)
    {
      rIntrinMap[eIntrinID] = IntrinsicInfoPairType(_strIntrinsicPrefix + strIntrinName, nullptr);
    }

    /** \brief  Generic internal base function, which looks up the function declaration objects for all initialized intrinsic functions in the lookup table.
     *  \tparam IntrinsicIDType         The type of the enumeration containing all the internal intrinsic function IDs.
     *  \param  rIntrinMap              A reference to the currently used intrinsic function lookup table.
     *  \param  strInstructionSetName   The name of the currently processed instruction set. */
    template <typename IntrinsicIDType>
    inline void _LookupIntrinsics(IntrinsicMapTemplateType<IntrinsicIDType> &rIntrinMap, std::string strInstructionSetName)
    {
      #ifdef VERBOSE_INIT_MODE
      llvm::errs() << "\n\nIntrinsic functions for instruction set \"" << strInstructionSetName << "\" (" << rIntrinMap.size() << " methods):\n";
      #endif

      for (typename IntrinsicMapTemplateType<IntrinsicIDType>::iterator itIntrinsic = rIntrinMap.begin(); itIntrinsic != rIntrinMap.end(); itIntrinsic++)
      {
        IntrinsicInfoPairType &rIntrinsicInfo = itIntrinsic->second;

        auto vecFunctions = _GetFunctionDecl(rIntrinsicInfo.first);

        if (! _GetASTHelper().AreSignaturesEqual(vecFunctions))
        {
          throw InternalErrorException( std::string("Found ambiguous entry for intrinsic function \"") + rIntrinsicInfo.first + std::string("\" !") );
        }

        rIntrinsicInfo.second = vecFunctions.front();

        #ifdef VERBOSE_INIT_MODE
        llvm::errs() << "\n" << rIntrinsicInfo.second->getReturnType().getAsString() << " " << rIntrinsicInfo.second->getNameAsString() << "(";
        for (unsigned int uiParam = 0; uiParam < rIntrinsicInfo.second->getNumParams(); ++uiParam)
        {
          ParmVarDecl *pParam = rIntrinsicInfo.second->getParamDecl(uiParam);
          llvm::errs() << pParam->getType().getAsString() << " " << pParam->getNameAsString();
          if ((uiParam + 1) < rIntrinsicInfo.second->getNumParams())
          {
            llvm::errs() << ", ";
          }
        }
        llvm::errs() << ")";
        #endif
      }

      #ifdef VERBOSE_INIT_MODE
      llvm::errs() << "\n\n";
      #endif
    }


    /** \brief  Reverts the order of expressions in a vector of expression objects.
     *  \param  crvecExpressions  The vector of expression objects, whose order shall be reverted. */
    ClangASTHelper::ExpressionVectorType _SwapExpressionOrder(const ClangASTHelper::ExpressionVectorType &crvecExpressions);


  private:
    /** \name Clang bugfixing helper methods */
    //@{

    /** \brief    Base function for the creation of missing intrinsic function declaration objects
     *  \param    strFunctionName   The requested name of newly declared intrinsic function.
     *  \param    crReturnType      The requested qualified return type of newly declared intrinsic function.
     *  \param    crvecArgTypes     A vector containing the requested qualified types of all function arguments.
     *  \param    crvecArgNames     A vector containing the requested parameter names of all function arguments.
     *  \remarks  The number of specified argument types must equal the number of specified argument names. */
    void _CreateIntrinsicDeclaration( std::string strFunctionName, const ::clang::QualType &crReturnType, const ClangASTHelper::QualTypeVectorType &crvecArgTypes,
                                      const ClangASTHelper::StringVectorType &crvecArgNames );

    /** \brief  Creates a missing intrinsic function declaration object with one call parameter.
     *  \param  strFunctionName   The requested name of newly declared intrinsic function.
     *  \param  crReturnType      The requested qualified return type of newly declared intrinsic function.
     *  \param  crArgType1        The requested qualified type of the first function argument.
     *  \param  strArgName1       The requested parameter name of the first function argument. */
    void _CreateIntrinsicDeclaration( std::string strFunctionName, const ::clang::QualType &crReturnType, const ::clang::QualType &crArgType1, std::string strArgName1 );

    /** \brief  Creates a missing intrinsic function declaration object with two call parameters.
     *  \param  strFunctionName   The requested name of newly declared intrinsic function.
     *  \param  crReturnType      The requested qualified return type of newly declared intrinsic function.
     *  \param  crArgType1        The requested qualified type of the first function argument.
     *  \param  strArgName1       The requested parameter name of the first function argument.
     *  \param  crArgType2        The requested qualified type of the second function argument.
     *  \param  strArgName2       The requested parameter name of the second function argument. */
    void _CreateIntrinsicDeclaration( std::string strFunctionName, const ::clang::QualType &crReturnType, const ::clang::QualType &crArgType1, std::string strArgName1,
                                      const ::clang::QualType &crArgType2, std::string strArgName2 );

    /** \brief  Creates a missing intrinsic function declaration object with three call parameters.
     *  \param  strFunctionName   The requested name of newly declared intrinsic function.
     *  \param  crReturnType      The requested qualified return type of newly declared intrinsic function.
     *  \param  crArgType1        The requested qualified type of the first function argument.
     *  \param  strArgName1       The requested parameter name of the first function argument.
     *  \param  crArgType2        The requested qualified type of the second function argument.
     *  \param  strArgName2       The requested parameter name of the second function argument.
     *  \param  crArgType3        The requested qualified type of the third function argument.
     *  \param  strArgName3       The requested parameter name of the third function argument. */
    void _CreateIntrinsicDeclaration( std::string strFunctionName, const ::clang::QualType &crReturnType, const ::clang::QualType &crArgType1, std::string strArgName1,
                                      const ::clang::QualType &crArgType2, std::string strArgName2, const ::clang::QualType &crArgType3, std::string strArgName3 );


    /** \brief  Returns the return type of a known non-ambiguous function declaration.
     *  \param  strFunctionName  The name of the function, whose return type shall be retrieved. */
    ::clang::QualType _GetFunctionReturnType(std::string strFunctionName);

    //@}

  protected:
    /** \name Clang bugfixing methods */
    //@{

    /** \brief  Creates all required missing intrinsic function declarations for the SSE instruction set (Clang header are incomplete). */
    void _CreateMissingIntrinsicsSSE();

    /** \brief  Creates all required missing intrinsic function declarations for the SSE2 instruction set (Clang header are incomplete). */
    void _CreateMissingIntrinsicsSSE2();

    /** \brief  Creates all required missing intrinsic function declarations for the SSE4.1 instruction set (Clang header are incomplete). */
    void _CreateMissingIntrinsicsSSE4_1();

    /** \brief  Creates all required missing intrinsic function declarations for the AVX instruction set (Clang header are incomplete). */
    void _CreateMissingIntrinsicsAVX();

    /** \brief  Creates all required missing intrinsic function declarations for the AVX2 instruction set (Clang header are incomplete). */
    void _CreateMissingIntrinsicsAVX2();

    //@}


  protected:

    InstructionSetBase(::clang::ASTContext &rAstContext, std::string strFunctionNamePrefix = "");

  public:

    /** \brief    Creates an object of a particular instruction set implementation class.
     *  \tparam   InstructionSetType  The type of the requested instruction set class.
     *  \param    rAstContext         A reference to the current Clang AST context..
     *  \return   A shared pointer to the newly created instruction set implementation object. */
    template <class InstructionSetType>
    inline static std::shared_ptr<InstructionSetType> Create(::clang::ASTContext &rAstContext)
    {
      static_assert( std::is_base_of<InstructionSetBase, InstructionSetType>::value, "The requested instruction set is not derived from class \"InstructionSetBase\" !" );

      return std::shared_ptr<InstructionSetType>( new InstructionSetType(rAstContext) );
    }


    virtual ~InstructionSetBase()
    {
      _mapKnownFuncDecls.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    /** \brief    Internal function, which creates a vector conversion expression object.
     *  \param    eSourceType       The vector element type present in the input vectors for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    crvecVectorRefs   A vector of the vectorized expressions, which return the input vectors for the conversion.
     *  \param    uiGroupIndex      The index of the group of vector elements, which shall be used as input for the upward conversions.
     *  \param    bMaskConversion   A flag indicating, whether the optimizations for vector mask conversions can be applied.
     *  \remarks  This function is called by the common base implementations of <b>conversion</b> routines, which make sure that it is correctly parametrized.
     *  \sa       _ConvertDown(), _ConvertSameSize(), _ConvertUp() */
    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) = 0;

    //@}

  private:

    /** \name Vector conversion translation methods */
    //@{

    /** \brief    Internal function, which creates an expression object that converts and packs multiple vectors down to one vector with a smaller element size.
     *  \param    eSourceType       The vector element type present in the input vectors for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    crvecVectorRefs   A vector of the vectorized expressions, which return the input vectors for the conversion.
     *  \param    bMaskConversion   A flag indicating, whether the optimizations for vector mask conversions can be applied.
     *  \remarks  This function is called by the public <b>conversion</b> routines, which make sure that it is correctly parametrized.
     *  \sa       ConvertMaskDown(), ConvertVectorDown() */
    ::clang::Expr* _ConvertDown(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, bool bMaskConversion);

    /** \brief    Internal function, which creates an expression object that converts one vector into another one with the same element type size.
     *  \param    eSourceType       The vector element type present in the input vector for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    pVectorRef        A pointer to a vectorized expression, which returns the input vector for the conversion.
     *  \param    bMaskConversion   A flag indicating, whether the optimizations for vector mask conversions can be applied.
     *  \remarks  This function is called by the public <b>conversion</b> routines, which make sure that it is correctly parametrized.
     *  \sa       ConvertMaskSameSize(), ConvertVectorSameSize() */
    ::clang::Expr* _ConvertSameSize(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef, bool bMaskConversion);

    /** \brief    Internal function, which creates an expression object that selects a group of elements of one vector and converts it into a vector of larger element type size.
     *  \param    eSourceType       The vector element type present in the input vector for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    pVectorRef        A pointer to a vectorized expression, which returns the input vector for the conversion.
     *  \param    uiGroupIndex      The index of the group of vector elements, which shall be used as input for the upward conversions.
     *  \param    bMaskConversion   A flag indicating, whether the optimizations for vector mask conversions can be applied.
     *  \remarks  This function is called by the public <b>conversion</b> routines, which make sure that it is correctly parametrized.
     *  \sa       ConvertMaskUp(), ConvertVectorUp() */
    ::clang::Expr* _ConvertUp(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef, std::uint32_t uiGroupIndex, bool bMaskConversion);

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    /** \brief    Creates an expression object, which converts and packs multiple vector masks down to one vector mask with a smaller element size.
     *  \param    eSourceType       The vector element type present in the input vector masks for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    crvecVectorRefs   A vector of the vectorized expressions, which return the input vector masks for the conversion.
     *  \remarks  This function is fulfilling basically the same functionality as ConvertVectorDown(), but it uses optimized conversion paths for vector masks. */
    inline ::clang::Expr* ConvertMaskDown(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs)
    {
      return _ConvertDown(eSourceType, eTargetType, crvecVectorRefs, true);
    }

    /** \brief    Creates an expression object, which converts one vector mask into another one with the same element type size.
     *  \param    eSourceType   The vector element type present in the input vector mask for the conversion.
     *  \param    eTargetType   The requested vector element type for the output of the conversion.
     *  \param    pVectorRef    A pointer to a vectorized expression, which returns the input vector mask for the conversion.
     *  \remarks  This function is fulfilling basically the same functionality as ConvertVectorSameSize(), but it uses optimized conversion paths for vector masks. */
    inline ::clang::Expr* ConvertMaskSameSize(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef)
    {
      return _ConvertSameSize(eSourceType, eTargetType, pVectorRef, true);
    }

    /** \brief    Creates an expression object, which selects a group of elements of one vector mask and converts it into a vector mask of larger element type size.
     *  \param    eSourceType   The vector element type present in the input vector mask for the conversion.
     *  \param    eTargetType   The requested vector element type for the output of the conversion.
     *  \param    pVectorRef    A pointer to a vectorized expression, which returns the input vector mask for the conversion.
     *  \param    uiGroupIndex  The index of the group of vector mask elements, which shall be used as input for the upward conversions.
     *  \remarks  This function is fulfilling basically the same functionality as ConvertVectorUp(), but it uses optimized conversion paths for vector masks. */
    inline ::clang::Expr* ConvertMaskUp(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef, std::uint32_t uiGroupIndex)
    {
      return _ConvertUp(eSourceType, eTargetType, pVectorRef, uiGroupIndex, true);
    }


    /** \brief    Creates an expression object, which converts and packs multiple vectors down to one vector with a smaller element size.
     *  \param    eSourceType       The vector element type present in the input vectors for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    crvecVectorRefs   A vector of the vectorized expressions, which return the input vectors for the conversion.
     *  \remarks  The number of input vectors must be equal to the size spread of the source and target type, e.g. if the element type size is reduced by the factor
                  of two, two vectors must be given as input parameters for this conversion. */
    inline ::clang::Expr* ConvertVectorDown(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs)
    {
      return _ConvertDown(eSourceType, eTargetType, crvecVectorRefs, false);
    }

    /** \brief    Creates an expression object, which converts one vector into another one with the same element type size.
     *  \param    eSourceType   The vector element type present in the input vector for the conversion.
     *  \param    eTargetType   The requested vector element type for the output of the conversion.
     *  \param    pVectorRef    A pointer to a vectorized expression, which returns the input vector for the conversion. */
    inline ::clang::Expr* ConvertVectorSameSize(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef)
    {
      return _ConvertSameSize(eSourceType, eTargetType, pVectorRef, false);
    }

    /** \brief    Creates an expression object, which selects a group of elements of one vector and converts it into a vector of larger element type size.
     *  \param    eSourceType   The vector element type present in the input vector for the conversion.
     *  \param    eTargetType   The requested vector element type for the output of the conversion.
     *  \param    pVectorRef    A pointer to a vectorized expression, which returns the input vector for the conversion.
     *  \param    uiGroupIndex  The index of the group of vector elements, which shall be used as input for the upward conversions.
     *  \remarks  The index of the vector element group must be smaller than the size spread between the source and target type, e.g. if the element type size is
                  increased by the factor of four, the group index must lie in the range of [0; 3]. */
    inline ::clang::Expr* ConvertVectorUp(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef, std::uint32_t uiGroupIndex)
    {
      return _ConvertUp(eSourceType, eTargetType, pVectorRef, uiGroupIndex, false);
    }


    /** \brief  Returns an expression, which creates a vector with all elements set to <b>one</b>.
     *  \param  eElementType  The requested element type stored in the vector.  */
    inline ::clang::Expr* CreateOnesVector(VectorElementTypes eElementType)
    {
      return CreateOnesVector(eElementType, false);
    }

    /** \brief    Returns an expression, which creates a vector with explicitly specified elements.
     *  \param    eElementType    The requested element type stored in the vector.
     *  \param    crvecElements   A vector of scalar expressions objects, which return the initialization values for the vector elements.
     *  \remarks  The number of scalar input expressions must be equal to the number of vector elements.
     *  \sa       GetVectorElementCount() */
    inline ::clang::Expr* CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements)
    {
      return CreateVector(eElementType, crvecElements, false);
    }


    /** \brief  Returns the number of elements, which are stored in a vector with a specified element type.
     *  \param  eElementType  The element type stored in the vector. */
    inline  size_t GetVectorElementCount(VectorElementTypes eElementType) const    { return GetVectorWidthBytes() / AST::BaseClasses::TypeInfo::GetTypeSize(eElementType); }

    /** \brief  Returns the qualified Clang-specific type of a vector variable with a specified element type.
     *  \param  eElementType  The element type stored in the vector. */
    virtual ::clang::QualType GetVectorType(VectorElementTypes eElementType) = 0;

    /** \brief  Returns the size of a vector register in bytes. */
    virtual size_t            GetVectorWidthBytes() const = 0;


    /** \brief  Checks, whether a specific built-in vector function is supported by the instruction set.
     *  \param  eElementType    The element type stored in the vector.
     *  \param  eFunctionType   The internal ID of the requested built-in vector function.
     *  \param  uiParamCount    The number of parameters for the built-in function.
     */
    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const = 0;

    /** \brief  Checks, whether a specific vector element type is supported for the instruction set.
     *  \param  eElementType  The element type stored in the vector. */
    virtual bool IsElementTypeSupported(VectorElementTypes eElementType) const = 0;


    /** \brief    Returns an expression, which performs an element-wise binary arithmetic operation on two vector values.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    eOpType       The type of the requested arithmetic operator.
     *  \param    pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the arithmetic operation.
     *  \param    pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the arithmetic operation.
     *  \param    bIsRHSScalar  Boolean indicating whether pExprRHS contains a scalar type.
     *  \remarks  The element types of both operands must be identical. */
    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) = 0;

    /** \brief    Returns an expression, which conditionally merges two vectors element-wise depending on a vector mask.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    pMaskRef      A pointer to the vectorized expression object, which evaluates to a vector mask that selects the elements for the merging.
     *  \param    pVectorTrue   A pointer to the vectorized expression object, whose elements are used for mask elements evaluating to <b>true</b>.
     *  \param    pVectorFalse  A pointer to the vectorized expression object, whose elements are used for mask elements evaluating to <b>false</b>.
     *  \remarks  The element types of all operands must be identical. */
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) = 0;

    /** \brief  Returns an expression, which broadcasts a scalar value into all elements of a vector.
     *  \param  eElementType      The element type stored in the vector.
     *  \param  pBroadCastValue   A pointer to the scalar expression object, whose return value shall be broadcasted. */
    virtual ::clang::Expr* BroadCast(VectorElementTypes eElementType, ::clang::Expr *pBroadCastValue) = 0;

    /** \brief    Returns an expression, which calls a special built-in vector function.
     *  \param    eElementType    The element type stored in the vector.
     *  \param    eFunctionType   The internal ID of the built-in vector function.
     *  \param    crvecArguments  A vector of expression objects, which return the call parameters for the built-in function.
     *  \remarks  If the requested built-in function is not supported, an <b>UnsupportedBuiltinFunctionType</b> exception will be thrown.
     *  \sa       IsBuiltinFunctionSupported(), InstructionSetExceptions::UnsupportedBuiltinFunctionType */
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) = 0;

    /** \brief  Returns an expression, which collapses the values of the elements in a vector mask into a scalar boolean value.
     *  \param  eMaskElementType  The element type stored in the vector mask.
     *  \param  eCheckType        The type of the checking operation, which shall be performed on the vector mask.
     *  \param  pMaskExpr         A pointer to a vectorized expression object, which returns a vector mask. */
    virtual ::clang::Expr* CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, ::clang::Expr *pMaskExpr) = 0;

    /** \brief    Returns an expression, which checks whether a specified element of a vector mask is set.
     *  \param    eMaskElementType  The element type stored in the vector mask.
     *  \param    pMaskExpr         A pointer to a vectorized expression object, which returns a vector mask.
     *  \param    uiIndex           The index of the vector mask element, which shall be checked.
     *  \remarks  The return value of the created expression object is a scalar boolean value. */
    virtual ::clang::Expr* CheckSingleMaskElement(VectorElementTypes eMaskElementType, ::clang::Expr *pMaskExpr, std::uint32_t uiIndex) = 0;

    /** \brief  Returns an expression, which creates a vector with all elements set to <b>one</b>.
     *  \param  eElementType  The requested element type stored in the vector.
     *  \param  bNegative     A flag indicating, whether the vector elements shall be set to <b>-1</b> or <b>+1</b>. */
    virtual ::clang::Expr* CreateOnesVector(VectorElementTypes eElementType, bool bNegative) = 0;

    /** \brief    Returns an expression, which creates a vector with explicitly specified elements.
     *  \param    eElementType    The requested element type stored in the vector.
     *  \param    crvecElements   A vector of scalar expressions objects, which return the initialization values for the vector elements.
     *  \param    bReversedOrder  A flag indicating, whether the vector elements shall be set in reversed or specified order.
     *  \remarks  The number of scalar input expressions must be equal to the number of vector elements.
     *  \sa       GetVectorElementCount() */
    virtual ::clang::Expr* CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder) = 0;

    /** \brief  Returns an expression, which creates a vector with all elements set to <b>zero</b>.
     *  \param  eElementType  The requested element type stored in the vector. */
    virtual ::clang::Expr* CreateZeroVector(VectorElementTypes eElementType) = 0;

    /** \brief    Returns an expression, which extracts a specified element from a vector.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    pVectorRef    A pointer to a vectorized expression, which returns the vector for the element extraction.
     *  \param    uiIndex       The index of the element, which shall be extracted.
     *  \remarks  The specified element index must be smaller than the amount of elements stored in the vector.
     *  \sa       GetVectorElementCount() */
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) = 0;

    /** \brief    Returns an expression, which inserts an element into a vector at a specified position.
     *  \param    eElementType    The element type stored in the vector.
     *  \param    pVectorRef      A pointer to a vectorized expression, which returns the vector where the element shall be inserted into.
     *  \param    pElementValue   A pointer to a scalar expression, which returns the element that shall be inserted.
     *  \param    uiIndex         The index of the element position in the vector, where the element shall be inserted.
     *  \remarks  The specified element index must be smaller than the amount of elements stored in the vector.
     *  \sa       GetVectorElementCount() */
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) = 0;

    /** \brief  Returns an expression, which reads a vector value from memory.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pPointerRef   A pointer to an expression object, which returns a pointer to the desired memory location. */
    virtual ::clang::Expr* LoadVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef) = 0;

    /** \brief    Returns an expression, which describes a <b>gather read</b> operation from multiple memory locations.
     *  \param    eElementType        The element type stored in the loaded vector.
     *  \param    eIndexElementType   The element type used for the index vectors.
     *  \param    pPointerRef         A pointer to an expression object, which returns a pointer to the desired memory base location.
     *  \param    crvecIndexExprs     A vector of vectorized expressions objects, which return the index vectors for the element-wise memory transactions of the gather read.
     *  \param    uiGroupIndex        The index of the group of index vector elements that shall be used as offsets for element-wise memory transactions.
     *  \remarks  Currently, only 32-bit and 64-bit integers are supported as index element type, and same restrictions apply as in the case of the conversion methods. */
    virtual ::clang::Expr* LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, ::clang::Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex) = 0;

    /** \brief    Returns an expression, which performs an element-wise binary relational operation on two vector values.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    eOpType       The type of the requested relational operator.
     *  \param    pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the relational operation.
     *  \param    pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the relational operation.
     *  \remarks  The element types of both operands must be identical. */
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) = 0;

    /** \brief    Returns an expression, which shifts the values of all elements inside a vector by the <b>same</b> amount of bits.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    pVectorRef    A pointer to a vectorized expression object, which returns the vector value whose elements shall be shifted.
     *  \param    bShiftLeft    A flag indicating, whether the vector element values shall be shifted to the left or to the right.
     *  \param    uiCount       The number of bits, by which all vector elements shall be shifted.
     *  \remarks  This operation is only defined for integer element types. */
    virtual ::clang::Expr* ShiftElements(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) = 0;

    /** \brief  Returns an expression, which writes a vector value into memory.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pPointerRef   A pointer to an expression object, which returns a pointer to the desired memory location.
     *  \param  pVectorValue  A pointer to the vectorized expression object, whose return value shall be written to memory. */
    virtual ::clang::Expr* StoreVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue) = 0;

    /** \brief    Returns an expression, which conditionally writes the selected elements of a vector value into memory.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    pPointerRef   A pointer to an expression object, which returns a pointer to the desired memory location.
     *  \param    pVectorValue  A pointer to the vectorized expression object, whose return value elements shall be written to memory.
     *  \param    pMaskRef      A pointer to the vectorized expression object, which evaluates to a vector mask that selects the elements to be stored.
     *  \remarks  The element types of the <b>vector value</b> and the <b>selection mask</b> must be identical. */
    virtual ::clang::Expr* StoreVectorMasked(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue, ::clang::Expr *pMaskRef) = 0;

    /** \brief  Returns an expression, which performs a vectorized unary operator expression.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  eOpType       The type of the requested unary operator.
     *  \param  pSubExpr      A pointer to the vectorized expression object, which shall be used as sub-expression for the unary operator. */
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) = 0;

    //@}
  };

  /** \brief  The shared pointer type for instruction set implementations. */
  typedef std::shared_ptr<InstructionSetBase> InstructionSetBasePtr;


  /** \name SSE instruction sets */
  //@{

  /** \brief  Implementation of the <b>Streaming SIMD Extensions</b> instruction-set. */
  class InstructionSetSSE : public InstructionSetBase
  {
  private:

    friend class InstructionSetBase;

    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSE</b>. */
    enum class IntrinsicsSSEEnum
    {
      AddFloat,
      AndFloat,
      AndNotFloat,
      BroadCastFloat,
      CeilFloat,
      CompareEqualFloat,
      CompareGreaterEqualFloat,
      CompareGreaterThanFloat,
      CompareLessEqualFloat,
      CompareLessThanFloat,
      CompareNotEqualFloat,
      CompareNotGreaterEqualFloat,
      CompareNotGreaterThanFloat,
      CompareNotLessEqualFloat,
      CompareNotLessThanFloat,
      DivideFloat,
      ExtractLowestFloat,
      FloorFloat,
      InsertLowestFloat,
      LoadFloat,
      MaxFloat,
      MinFloat,
      MoveFloatHighLow,             MoveFloatLowHigh,
      MoveMaskFloat,
      MultiplyFloat,
      OrFloat,
      ReciprocalFloat,
      ReciprocalSqrtFloat,
      SetFloat,
      SetZeroFloat,
      ShuffleFloat,
      SqrtFloat,
      StoreFloat,
      SubtractFloat,
      UnpackHighFloat,              UnpackLowFloat,
      XorFloat
    };


    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSEEnum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSE; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSEEnum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSE, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function without any call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.  */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSEEnum eIntrinID)
    {
      return _CreateFunctionCall(eIntrinID, ClangASTHelper::ExpressionVectorType());
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSEEnum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSEEnum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with three call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument.
     *  \param  pArg3       A pointer to the expression object, which serves as the third argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSEEnum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2, ::clang::Expr *pArg3)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);
      vecArguments.push_back(pArg3);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Returns an expressions, which performs a post-fixed unary increment or decrement operation on all elements of a vector.
     *  \param  eIntrinID     The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pVectorRef    A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    inline ::clang::Expr* _CreatePostfixedUnaryOp(IntrinsicsSSEEnum eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      return InstructionSetBase::_CreatePostfixedUnaryOp(_mapIntrinsicsSSE, eIntrinID, eElementType, pVectorRef);
    }

    /** \brief  Returns an expressions, which performs a pre-fixed unary increment or decrement operation on all elements of a vector.
     *  \param  eIntrinID     The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pVectorRef    A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    inline ::clang::Expr* _CreatePrefixedUnaryOp(IntrinsicsSSEEnum eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      return InstructionSetBase::_CreatePrefixedUnaryOp(_mapIntrinsicsSSE, eIntrinID, eElementType, pVectorRef);
    }


    /** \brief  Returns the qualified Clang return type of an intrinsic function.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function, whose return type shall be retrieved. */
    inline ::clang::QualType _GetFunctionReturnType(IntrinsicsSSEEnum eIntrinID)
    {
      return InstructionSetBase::_GetFunctionReturnType(_mapIntrinsicsSSE, eIntrinID);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSEEnum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSE, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


    /** \brief  Checks, whether a specific vector element type is supported by this instruction set, and throws an exception if this is not the case.
     *  \param  eElementType  The vector element type, which shall be checked. */
    void _CheckElementType(VectorElementTypes eElementType) const;

    /** \brief    Checks, whether a certain element index is valid for a vector with a specific element type, and throws an exception if this is not the case.
     *  \tparam   ExceptionType   The type of the exception, which shall be thrown if the element index is out of range.
     *  \param    eElementType    The vector element type, which shall be checked.
     *  \param    uiIndex         The element index, which shall be checked for correct range. */
    template <class ExceptionType> inline void _CheckIndex(VectorElementTypes eElementType, std::uint32_t uiIndex) const
    {
      uint32_t uiUpperLimit = static_cast<uint32_t>(GetVectorElementCount(eElementType)) - 1;

      if (uiIndex > uiUpperLimit)
      {
        throw ExceptionType(eElementType, uiUpperLimit);
      }
    }

  protected:

    InstructionSetSSE(::clang::ASTContext &rAstContext);


    /** \brief    Checks, whether a certain element index is valid for an element extraction with a specific vector element type.
     *  \param    eElementType    The vector element type, which shall be checked.
     *  \param    uiIndex         The element index, which shall be checked for correct range. */
    inline void _CheckExtractIndex(VectorElementTypes eElementType, std::uint32_t uiIndex) const  { _CheckIndex<InstructionSetExceptions::ExtractIndexOutOfRange>(eElementType, uiIndex); }

    /** \brief    Checks, whether a certain element index is valid for an element insertion with a specific vector element type.
     *  \param    eElementType    The vector element type, which shall be checked.
     *  \param    uiIndex         The element index, which shall be checked for correct range. */
    inline void _CheckInsertIndex(VectorElementTypes eElementType, std::uint32_t uiIndex) const   { _CheckIndex<InstructionSetExceptions::InsertIndexOutOfRange>(eElementType, uiIndex); }


    /** \brief  Returns the common prefix for all intrinsic functions of the AVX instruction set family. */
    static inline std::string _GetIntrinsicPrefix() { return "_mm_"; }


    /** \brief  Returns an expression, which creates a vector with all element value bits set to <b>one</b>.
     *  \param  eElementType  The requested element type stored in the vector.  */
    virtual ::clang::Expr* _CreateFullBitMask(VectorElementTypes eElementType);

    /** \brief  Returns an expression, which merges either the low or the high halves of two vectors into one vector by concatenation.
     *  \param  eElementType  The requested element type stored in the vector.
     *  \param  pVectorRef1   A pointer to a vectorized expression, which returns the first vector for the merging operation.
     *  \param  pVectorRef2   A pointer to a vectorized expression, which returns the second vector for the merging operation.
     *  \param  bLowHalf      A flag indicating, whether the low halves or the high halves of the vectors shall be merged. */
    ::clang::Expr* _MergeVectors(VectorElementTypes eElementType, ::clang::Expr *pVectorRef1, ::clang::Expr *pVectorRef2, bool bLowHalf);

    /** \brief  Returns an expression, which element-wise interleaves either the low or the high halves of two vectors into one vector.
     *  \param  eElementType  The requested element type stored in the vector.
     *  \param  pVectorRef1   A pointer to a vectorized expression, which returns the first vector for the interleaving operation.
     *  \param  pVectorRef2   A pointer to a vectorized expression, which returns the second vector for the interleaving operation.
     *  \param  bLowHalf      A flag indicating, whether the low halves or the high halves of the vectors shall be interleaved. */
    virtual ::clang::Expr* _UnpackVectors(VectorElementTypes eElementType, ::clang::Expr *pVectorRef1, ::clang::Expr *pVectorRef2, bool bLowHalf);


  public:

    virtual ~InstructionSetSSE()
    {
      _mapIntrinsicsSSE.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::QualType GetVectorType(VectorElementTypes eElementType) override;
    virtual size_t            GetVectorWidthBytes() const final override { return static_cast<size_t>(16); }

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const override;
    virtual bool IsElementTypeSupported(VectorElementTypes eElementType) const override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) override;
    virtual ::clang::Expr* BroadCast(VectorElementTypes eElementType, ::clang::Expr *pBroadCastValue) override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) override;
    virtual ::clang::Expr* CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, ::clang::Expr *pMaskExpr) override;
    virtual ::clang::Expr* CheckSingleMaskElement(VectorElementTypes eMaskElementType, ::clang::Expr *pMaskExpr, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* CreateOnesVector(VectorElementTypes eElementType, bool bNegative) override;
    virtual ::clang::Expr* CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder) override;
    virtual ::clang::Expr* CreateZeroVector(VectorElementTypes eElementType) override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* LoadVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef) override;
    virtual ::clang::Expr* LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, ::clang::Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex) override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;
    virtual ::clang::Expr* ShiftElements(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) override;
    virtual ::clang::Expr* StoreVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue) override;
    virtual ::clang::Expr* StoreVectorMasked(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue, ::clang::Expr *pMaskRef) override;
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) override;

    //@}
  };

  /** \brief  Implementation of the <b>Streaming SIMD Extensions 2</b> instruction-set. */
  class InstructionSetSSE2 : public InstructionSetSSE
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetSSE   BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSE 2</b>. */
    enum class IntrinsicsSSE2Enum
    {
      AddDouble,                    AddInt8,                AddInt16,                AddInt32,                AddInt64,
      AndDouble,                    AndInteger,             AndNotDouble,            AndNotInteger,
      BroadCastDouble,              BroadCastInt8,          BroadCastInt16,          BroadCastInt32,          BroadCastInt64,
      CastDoubleToFloat,            CastDoubleToInteger,    CastFloatToDouble,       CastFloatToInteger,      CastIntegerToDouble, CastIntegerToFloat,
      CeilDouble,
      CompareEqualDouble,           CompareEqualInt8,       CompareEqualInt16,       CompareEqualInt32,
      CompareGreaterEqualDouble,
      CompareGreaterThanDouble,     CompareGreaterThanInt8, CompareGreaterThanInt16, CompareGreaterThanInt32,
      CompareLessEqualDouble,
      CompareLessThanDouble,        CompareLessThanInt8,    CompareLessThanInt16,    CompareLessThanInt32,
      CompareNotEqualDouble,
      CompareNotGreaterEqualDouble,
      CompareNotGreaterThanDouble,
      CompareNotLessEqualDouble,
      CompareNotLessThanDouble,
      ConvertDoubleFloat,           ConvertDoubleInt32,     ConvertFloatDouble,      ConvertFloatInt32,       ConvertInt32Double,  ConvertInt32Float,
      ConvertSingleDoubleInt64,
      DivideDouble,
      ExtractInt16,                 ExtractLowestDouble,    ExtractLowestInt32,      ExtractLowestInt64,
      FloorDouble,
      InsertInt16,                  InsertLowestDouble,
      LoadDouble,                   LoadInteger,
      MaxDouble,                    MaxUInt8,               MaxInt16,
      MinDouble,                    MinUInt8,               MinInt16,
      MoveMaskDouble,               MoveMaskInt8,
      MultiplyDouble,               MultiplyInt16,          MultiplyUInt32,
      OrDouble,                     OrInteger,
      PackInt16ToInt8,              PackInt16ToUInt8,       PackInt32ToInt16,
      SetDouble,                    SetInt8,                SetInt16,                SetInt32,                SetInt64,
      SetZeroDouble,                SetZeroInteger,
      ShiftLeftInt16,               ShiftLeftInt32,         ShiftLeftInt64,          ShiftLeftVectorBytes,
      ShiftRightArithInt16,         ShiftRightArithInt32,   ShiftRightLogInt16,      ShiftRightLogInt32,      ShiftRightLogInt64,  ShiftRightVectorBytes,
      ShuffleDouble,                ShuffleInt16High,       ShuffleInt16Low,         ShuffleInt32,
      SqrtDouble,
      StoreDouble,                  StoreInteger,           StoreConditionalInteger,
      SubtractDouble,               SubtractInt8,           SubtractInt16,           SubtractInt32,           SubtractInt64,
      UnpackHighDouble,             UnpackHighInt8,         UnpackHighInt16,         UnpackHighInt32,         UnpackHighInt64,
      UnpackLowDouble,              UnpackLowInt8,          UnpackLowInt16,          UnpackLowInt32,          UnpackLowInt64,
      XorDouble,                    XorInteger
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSE2Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSE2; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE2Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSE2, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function without any call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.  */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE2Enum eIntrinID)
    {
      return _CreateFunctionCall(eIntrinID, ClangASTHelper::ExpressionVectorType());
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE2Enum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE2Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with three call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument.
     *  \param  pArg3       A pointer to the expression object, which serves as the third argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE2Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2, ::clang::Expr *pArg3)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);
      vecArguments.push_back(pArg3);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Returns an expressions, which performs a post-fixed unary increment or decrement operation on all elements of a vector.
     *  \param  eIntrinID     The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType  The element type stored in the vectors.
     *  \param  pVectorRef    A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    inline ::clang::Expr* _CreatePostfixedUnaryOp(IntrinsicsSSE2Enum eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      return InstructionSetBase::_CreatePostfixedUnaryOp(_mapIntrinsicsSSE2, eIntrinID, eElementType, pVectorRef);
    }

    /** \brief  Returns an expressions, which performs a pre-fixed unary increment or decrement operation on all elements of a vector.
     *  \param  eIntrinID     The internal ID of the intrinsic, which represents an addition or a subtraction for the current vector element type.
     *  \param  eElementType  The element type stored in the vectors.
     *  \param  pVectorRef    A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented. */
    inline ::clang::Expr* _CreatePrefixedUnaryOp(IntrinsicsSSE2Enum eIntrinID, VectorElementTypes eElementType, ::clang::Expr *pVectorRef)
    {
      return InstructionSetBase::_CreatePrefixedUnaryOp(_mapIntrinsicsSSE2, eIntrinID, eElementType, pVectorRef);
    }


    /** \brief  Returns the qualified Clang return type of an intrinsic function.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function, whose return type shall be retrieved. */
    inline ::clang::QualType _GetFunctionReturnType(IntrinsicsSSE2Enum eIntrinID)
    {
      return InstructionSetBase::_GetFunctionReturnType(_mapIntrinsicsSSE2, eIntrinID);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSE2Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSE2, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


  private:

    /** \brief  Internal function, which handles the creation of arithmetic operation expressions for integer vectors.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  eOpType       The type of the requested arithmetic operator.
     *  \param  pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the arithmetic operation.
     *  \param  pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the arithmetic operation.
     *  \param  bIsRHSScalar  Boolean indicating whether pExprRHS contains a scalar type. */
    ::clang::Expr* _ArithmeticOpInteger(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false);

    /** \brief    Internal function, which emulates relational operations for 64-bit integer vectors by a vector splitting and a series of scalar operations.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the relational operation.
     *  \param    pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the relational operation.
     *  \param    eOpKind       The Clang-specific ID of the requested scalar relational operator.
     *  \return   An expression object, which returns the re-built result vector mask, containing the series of scalar operations. */
    ::clang::Expr* _CompareInt64(VectorElementTypes eElementType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, ::clang::BinaryOperatorKind eOpKind);

    /** \brief  Internal function, which handles the creation of relational operator expressions for integer vectors.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  eOpType       The type of the requested relational operator.
     *  \param  pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the relational operation.
     *  \param  pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the relational operation. */
    ::clang::Expr* _RelationalOpInteger(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS);

    /** \brief    Internal function, which emulates a not supported arithmetic operation for integer vectors by a vector splitting and a series of scalar operations.
     *  \param    eElementType  The element type stored in the vector.
     *  \param    eOpKind       The Clang-specific ID of the requested scalar arithmetic operator.
     *  \param    pExprLHS      A pointer to the vectorized expression object, which returns the left-hand-side value of the arithmetic operation.
     *  \param    pExprRHS      A pointer to the vectorized expression object, which returns the right-hand-side value of the arithmetic operation.
     *  \return   An expression object, which returns the re-built result vector, containing the series of scalar operations. */
    ::clang::Expr* _SeparatedArithmeticOpInteger(VectorElementTypes eElementType, ::clang::BinaryOperatorKind eOpKind, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS);


  protected:

    InstructionSetSSE2(::clang::ASTContext &rAstContext);


    virtual ::clang::Expr* _CreateFullBitMask(VectorElementTypes eElementType) final override;

    /** \brief  Returns an expression, which shifts the contents of a vector across element boundaries by a specified amount of bytes.
     *  \param  pVectorRef    A pointer to a vectorized expression object, which returns the vector whose contents shall be shifted.
     *  \param  uiByteCount   The number of bytes, by which the vector contents shall be shifted.
     *  \param  bShiftLeft    A flag indicating, whether the vector contents shall be shifted to the left or to the right. */
    ::clang::Expr* _ShiftIntegerVectorBytes(::clang::Expr *pVectorRef, std::uint32_t uiByteCount, bool bShiftLeft);

    virtual ::clang::Expr* _UnpackVectors(VectorElementTypes eElementType, ::clang::Expr *pVectorRef1, ::clang::Expr *pVectorRef2, bool bLowHalf) final override;

  public:

    virtual ~InstructionSetSSE2()
    {
      _mapIntrinsicsSSE2.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::QualType GetVectorType(VectorElementTypes eElementType) final override;

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const override;
    virtual bool IsElementTypeSupported(VectorElementTypes eElementType) const final override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) override;
    virtual ::clang::Expr* BroadCast(VectorElementTypes eElementType, ::clang::Expr *pBroadCastValue) final override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) override;
    virtual ::clang::Expr* CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, ::clang::Expr *pMaskExpr) final override;
    virtual ::clang::Expr* CheckSingleMaskElement(VectorElementTypes eMaskElementType, ::clang::Expr *pMaskExpr, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* CreateOnesVector(VectorElementTypes eElementType, bool bNegative) final override;
    virtual ::clang::Expr* CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder) final override;
    virtual ::clang::Expr* CreateZeroVector(VectorElementTypes eElementType) final override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* LoadVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef) override;
    virtual ::clang::Expr* LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, ::clang::Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex) final override;
    virtual ::clang::Expr* ShiftElements(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) final override;
    virtual ::clang::Expr* StoreVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue) final override;
    virtual ::clang::Expr* StoreVectorMasked(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue, ::clang::Expr *pMaskRef) final override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) override;

    //@}
  };

  /** \brief  Implementation of the <b>Streaming SIMD Extensions 3</b> instruction-set. */
  class InstructionSetSSE3 : public InstructionSetSSE2
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetSSE2    BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSE 3</b>. */
    enum class IntrinsicsSSE3Enum
    {
      LoadInteger
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSE3Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSE3; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE3Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSE3, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE3Enum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Returns the qualified Clang return type of an intrinsic function.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function, whose return type shall be retrieved. */
    inline ::clang::QualType _GetFunctionReturnType(IntrinsicsSSE3Enum eIntrinID)
    {
      return InstructionSetBase::_GetFunctionReturnType(_mapIntrinsicsSSE3, eIntrinID);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSE3Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSE3, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


  protected:

    InstructionSetSSE3(::clang::ASTContext &rAstContext);

  public:

    virtual ~InstructionSetSSE3()
    {
      _mapIntrinsicsSSE3.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* LoadVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef) final override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) override;

    //@}
  };

  /** \brief  Implementation of the <b>Supplemental Streaming SIMD Extensions 3</b> instruction-set. */
  class InstructionSetSSSE3 : public InstructionSetSSE3
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetSSE3    BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSSE 3</b>. */
    enum class IntrinsicsSSSE3Enum
    {
      AbsoluteInt8, AbsoluteInt16, AbsoluteInt32,
      ShuffleInt8,
      SignInt8,     SignInt16,     SignInt32
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSSE3Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSSE3; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSSE3Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSSE3, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSSE3Enum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSSE3Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSSE3Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSSE3, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


  protected:

    InstructionSetSSSE3(::clang::ASTContext &rAstContext);

  public:

    virtual ~InstructionSetSSSE3()
    {
      _mapIntrinsicsSSSE3.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) final override;

    //@}
  };

  /** \brief  Implementation of the <b>Streaming SIMD Extensions 4.1</b> instruction-set. */
  class InstructionSetSSE4_1 : public InstructionSetSSSE3
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetSSSE3    BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSE 4.1</b>. */
    enum class IntrinsicsSSE4_1Enum
    {
      BlendDouble,        BlendFloat,         BlendInteger,
      CompareEqualInt64,
      ConvertInt8Int16,   ConvertInt8Int32,   ConvertInt8Int64,
      ConvertInt16Int32,  ConvertInt16Int64,  ConvertInt32Int64,
      ConvertUInt8Int16,  ConvertUInt8Int32,  ConvertUInt8Int64,
      ConvertUInt16Int32, ConvertUInt16Int64, ConvertUInt32Int64,
      ExtractInt8,        ExtractInt32,       ExtractInt64,
      InsertFloat,        InsertInt8,         InsertInt32,        InsertInt64,
      MaxInt8,            MaxInt32,           MaxUInt16,          MaxUInt32,
      MinInt8,            MinInt32,           MinUInt16,          MinUInt32,
      MultiplyInt32,
      PackInt32ToUInt16,
      TestControl
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSE4_1Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSE4_1; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_1Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSE4_1, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_1Enum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_1Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with three call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument.
     *  \param  pArg3       A pointer to the expression object, which serves as the third argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_1Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2, ::clang::Expr *pArg3)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);
      vecArguments.push_back(pArg3);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSE4_1Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSE4_1, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


  protected:

    InstructionSetSSE4_1(::clang::ASTContext &rAstContext);


  public:

    virtual ~InstructionSetSSE4_1()
    {
      _mapIntrinsicsSSE4_1.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) final override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const final override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) final override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) final override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) final override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;

    //@}
  };

  /** \brief  Implementation of the <b>Streaming SIMD Extensions 4.2</b> instruction-set. */
  class InstructionSetSSE4_2 final : public InstructionSetSSE4_1
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetSSE4_1    BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>SSE 4.2</b>. */
    enum class IntrinsicsSSE4_2Enum
    {
      CompareGreaterThanInt64
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsSSE4_2Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsSSE4_2; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_2Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsSSE4_2, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsSSE4_2Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsSSE4_2Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsSSE4_2, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


  private:

    InstructionSetSSE4_2(::clang::ASTContext &rAstContext);

  public:

    virtual ~InstructionSetSSE4_2()
    {
      _mapIntrinsicsSSE4_2.clear();
    }


    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) final override;

    //@}
  };

  //@}



  /** \name AVX instruction sets */
  //@{

  /** \brief  Implementation of the <b>Advanced Vector Extensions</b> instruction-set. */
  class InstructionSetAVX : public InstructionSetBase
  {
  private:

    friend class InstructionSetBase;

    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>AVX</b>. */
    enum class IntrinsicsAVXEnum
    {
      AddDouble,          AddFloat,
      AndDouble,          AndFloat,
      BlendDouble,        BlendFloat,
      BroadCastDouble,    BroadCastFloat,       BroadCastInt8,      BroadCastInt16,     BroadCastInt32,       BroadCastInt64,
      CastDoubleToFloat,  CastDoubleToInteger,  CastFloatToDouble,  CastFloatToInteger, CastIntegerToDouble,  CastIntegerToFloat,
      CeilDouble,         CeilFloat,
      CompareDouble,      CompareFloat,
      ConvertDoubleFloat, ConvertDoubleInt32,   ConvertFloatDouble, ConvertFloatInt32,  ConvertInt32Double,   ConvertInt32Float,
      DivideDouble,       DivideFloat,
      DuplicateEvenFloat, DuplicateOddFloat,
      ExtractInt8,        ExtractInt16,         ExtractInt32,       ExtractInt64,
      ExtractSSEDouble,   ExtractSSEFloat,      ExtractSSEInteger,
      FloorDouble,        FloorFloat,
      InsertSSEDouble,    InsertSSEFloat,       InsertSSEInteger,
      LoadDouble,         LoadFloat,            LoadInteger,
      MaxDouble,          MaxFloat,
      MergeDouble,        MergeFloat,           MergeInteger,
      MinDouble,          MinFloat,
      MoveMaskDouble,     MoveMaskFloat,
      MultiplyDouble,     MultiplyFloat,
      OrDouble,           OrFloat,
      PermuteLanesFloat,
      SetDouble,          SetFloat,             SetInt8,            SetInt16,           SetInt32,             SetInt64,
      SetZeroDouble,      SetZeroFloat,         SetZeroInteger,
      ShuffleDouble,      ShuffleFloat,
      StoreDouble,        StoreFloat,           StoreInteger,
      SqrtDouble,         SqrtFloat,
      SubtractDouble,     SubtractFloat,
      XorDouble,          XorFloat
    };


    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsAVXEnum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType        _mapIntrinsicsAVX;          //!< The internal lookup-table of intrinsic functions.
    InstructionSetBasePtr   _spFallbackInstructionSet;  //!< A shared pointer to the referenced SSE fallback instruction set.


    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVXEnum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsAVX, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function without any call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.  */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVXEnum eIntrinID)
    {
      return _CreateFunctionCall(eIntrinID, ClangASTHelper::ExpressionVectorType());
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVXEnum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVXEnum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with three call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument.
     *  \param  pArg3       A pointer to the expression object, which serves as the third argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVXEnum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2, ::clang::Expr *pArg3)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);
      vecArguments.push_back(pArg3);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


    /** \brief  Returns the qualified Clang return type of an intrinsic function.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function, whose return type shall be retrieved. */
    inline ::clang::QualType _GetFunctionReturnType(IntrinsicsAVXEnum eIntrinID)
    {
      return InstructionSetBase::_GetFunctionReturnType(_mapIntrinsicsAVX, eIntrinID);
    }


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsAVXEnum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsAVX, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();


    /** \brief  Throws an exception, which indicates that a specific vector element type is not supported in this instruction set.
     *  \param  eType   The vector element, whose use raised this error. */
    inline void _ThrowUnsupportedType(VectorElementTypes eType)
    {
      throw RuntimeErrorException( std::string("The element type \"") + AST::BaseClasses::TypeInfo::GetTypeString(eType) +
                                   std::string("\" is not supported in instruction set AVX!") );
    }


  protected:

    InstructionSetAVX(::clang::ASTContext &rAstContext);


    /** \brief  Returns a shared pointer to the SSE fallback instruction set. */
    inline InstructionSetBasePtr _GetFallback()   { return _spFallbackInstructionSet; }

    /** \brief  Returns the common prefix for all intrinsic functions of the AVX instruction set family. */
    static inline std::string _GetIntrinsicPrefix() { return "_mm256_"; }


    /** \brief    Creates an expression, which casts one vector type into another one.
     *  \param    eSourceType   The vector element type, which is present in the input vector for the cast operation.
     *  \param    eTargetType   The desired vector element type in the return value of the cast.
     *  \param    pVectorRef    A pointer to the vectorized expression object, which returns the vector that shall be casted.
     *  \remarks  This operation only changes the syntactic type of a vector, it does not change the stored data. */
    ::clang::Expr*  _CastVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, ::clang::Expr *pVectorRef);

    /** \brief  Returns an expression, which creates a vector with all element value bits set to <b>one</b>.
     *  \param  eElementType  The requested element type stored in the vector. */
    ::clang::Expr*  _CreateFullBitMask(VectorElementTypes eElementType);

    /** \brief  Returns an expression, which extracts one half of an AVX vector into a SSE vector.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pAVXVector    A pointer to a vectorized expression, which returns the AVX vector used for the extraction.
     *  \param  bLowHalf      A flag indicating, whether the low half or the high half of the AVX vector shall be extracted. */
    ::clang::Expr*  _ExtractSSEVector(VectorElementTypes eElementType, ::clang::Expr *pAVXVector, bool bLowHalf);

    /** \brief  Returns an expression, which inserts the contents of an SSE vector into one half of an AVX vector.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pAVXVector    A pointer to a vectorized expression, which returns the AVX vector where the contents of the SSE vector shall be inserted.
     *  \param  pSSEVector    A pointer to a vectorized expression, which returns the SSE vector that shall be inserted into the AVX vector.
     *  \param  bLowHalf      A flag indicating, whether the low half or the high half of the AVX vector shall be replaced. */
    ::clang::Expr*  _InsertSSEVector(VectorElementTypes eElementType, ::clang::Expr *pAVXVector, ::clang::Expr *pSSEVector, bool bLowHalf);

    /** \brief  Returns an expression, which concatenates two SSE vectors into one AVX vector.
     *  \param  eElementType    The element type stored in the vectors.
     *  \param  pSSEVectorLow   A pointer to a vectorized expression, which returns the SSE vector that shall be used as low half of the AVX vector.
     *  \param  pSSEVectorHigh  A pointer to a vectorized expression, which returns the SSE vector that shall be used as high half of the AVX vector. */
    ::clang::Expr*  _MergeSSEVectors(VectorElementTypes eElementType, ::clang::Expr *pSSEVectorLow, ::clang::Expr *pSSEVectorHigh);


    /** \brief  Returns an expressions, which performs a unary increment or decrement operation on all elements of a vector.
     *  \param  eElementType  The element type stored in the vectors.
     *  \param  pVectorRef    A pointer to a vectorized expression, which returns the vector that shall be incremented or decremented.
     *  \param  bPrefixed     A flag indicating, whether a pre-fixed or a post-fixed unary operation shall be created.
     *  \param  bIncrement    A flag indicating, whether the created operation represents an increment or a decrement of the vector elements. */
    virtual ::clang::Expr*  _CreatePrePostFixedUnaryOp(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bPrefixed, bool bIncrement);


  private:

    /** \brief    Internal function, which creates an AVX vector conversion expression object with the use of the SSE fallback.
     *  \param    eSourceType       The vector element type present in the input vectors for the conversion.
     *  \param    eTargetType       The requested vector element type for the output of the conversion.
     *  \param    crvecVectorRefs   A vector of the vectorized expressions, which return the input vectors for the conversion.
     *  \param    uiGroupIndex      The index of the group of vector elements, which shall be used as input for the upward conversions.
     *  \param    bMaskConversion   A flag indicating, whether the optimizations for vector mask conversions can be applied.
     *  \remarks  This function is only called, when the conversion of AVX vectors cannot be expressed by the AVX instruction set itself.
     *  \sa       _ConvertVector() */
    ::clang::Expr* _ConvertVectorWithSSE(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion);


  public:

    virtual ~InstructionSetAVX()
    {
      _mapIntrinsicsAVX.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::QualType GetVectorType(VectorElementTypes eElementType) final override;
    virtual size_t            GetVectorWidthBytes() const final override   { return static_cast<size_t>(32); }

    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const override;
    virtual bool IsElementTypeSupported(VectorElementTypes eElementType) const final override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) override;
    virtual ::clang::Expr* BroadCast(VectorElementTypes eElementType, ::clang::Expr *pBroadCastValue) final override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) override;
    virtual ::clang::Expr* CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, ::clang::Expr *pMaskExpr) override;
    virtual ::clang::Expr* CheckSingleMaskElement(VectorElementTypes eMaskElementType, ::clang::Expr *pMaskExpr, std::uint32_t uiIndex) override;
    virtual ::clang::Expr* CreateOnesVector(VectorElementTypes eElementType, bool bNegative) final override;
    virtual ::clang::Expr* CreateVector(VectorElementTypes eElementType, const ClangASTHelper::ExpressionVectorType &crvecElements, bool bReversedOrder) final override;
    virtual ::clang::Expr* CreateZeroVector(VectorElementTypes eElementType) final override;
    virtual ::clang::Expr* ExtractElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* InsertElement(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, ::clang::Expr *pElementValue, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* LoadVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef) final override;
    virtual ::clang::Expr* LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, ::clang::Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex) override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) override;
    virtual ::clang::Expr* ShiftElements(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) override;
    virtual ::clang::Expr* StoreVector(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue) final override;
    virtual ::clang::Expr* StoreVectorMasked(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue, ::clang::Expr *pMaskRef) override;
    virtual ::clang::Expr* UnaryOperator(VectorElementTypes eElementType, UnaryOperatorType eOpType, ::clang::Expr *pSubExpr) final override;

    //@}
  };

  /** \brief  Implementation of the <b>Advanced Vector Extensions 2</b> instruction-set. */
  class InstructionSetAVX2 final : public InstructionSetAVX
  {
  private:

    friend class InstructionSetBase;
    typedef InstructionSetAVX     BaseType;


    /** \brief  Enumeration of all internal IDs for the intrinsic functions of the instruction set <b>AVX 2</b>. */
    enum class IntrinsicsAVX2Enum
    {
      AbsInt8, AbsInt16, AbsInt32,
      AddInt8, AddInt16, AddInt32, AddInt64,
      AndInteger, AndNotInteger,
      BlendInteger,
      CompareEqualInt8, CompareEqualInt16, CompareEqualInt32, CompareEqualInt64,
      CompareGreaterThanInt8, CompareGreaterThanInt16, CompareGreaterThanInt32, CompareGreaterThanInt64,
      ConvertInt8Int16,   ConvertInt8Int32,   ConvertInt8Int64,
      ConvertInt16Int32,  ConvertInt16Int64,
      ConvertInt32Int64,
      ConvertUInt8Int16,  ConvertUInt8Int32,  ConvertUInt8Int64,
      ConvertUInt16Int32, ConvertUInt16Int64,
      ConvertUInt32Int64,
      ExtractSSEIntegerInteger,
      GatherInt32, GatherInt64, GatherFloat, GatherDouble,
      MaxInt8, MaxUInt8, MaxInt16,  MaxUInt16, MaxInt32, MaxUInt32,
      MinInt8, MinUInt8, MinInt16,  MinUInt16, MinInt32, MinUInt32,
      MoveMaskInt8,
      MultiplyInt16, MultiplyInt32,
      OrInteger,
      PackInt16ToInt8,              PackInt16ToUInt8,       PackInt32ToInt16,        PackInt32ToUInt16,
      PermuteCrossInt32,
      ShiftLeftInt16,               ShiftLeftInt32,         ShiftLeftInt64,          ShiftLeftVectorBytes,
      ShiftRightArithInt16,         ShiftRightArithInt32,   ShiftRightLogInt16,      ShiftRightLogInt32,      ShiftRightLogInt64,  ShiftRightVectorBytes,
      ShuffleInt8, ShuffleInt32,
      SubtractInt8, SubtractInt16, SubtractInt32, SubtractInt64,
      XorInteger
    };

    typedef InstructionSetBase::IntrinsicMapTemplateType<IntrinsicsAVX2Enum> IntrinsicMapType; //!< Type definition for the lookup-table of intrinsic functions.


  private:

    IntrinsicMapType _mapIntrinsicsAVX2; //!< The internal lookup-table of intrinsic functions.


    /** \brief  Establishes a link between the name and the internal ID of a specific intrinsic function.
     *  \param  eIntrinID       The internal ID of the intrinsic function.
     *  \param  strIntrinName   The name of the intrinsic function. */
    inline void _InitIntrinsic(IntrinsicsAVX2Enum eIntrinID, std::string strIntrinName)
    {
      InstructionSetBase::_InitIntrinsic(_mapIntrinsicsAVX2, eIntrinID, strIntrinName);
    }

    /** \brief  Maps all used intrinsic functions to their corresponding internal IDs. */
    void _InitIntrinsicsMap();

    /** \brief  Base function for the creation of function call expression objects to intrinsic functions.
     *  \param  eIntrinID       The internal ID of the requested intrinsic function.
     *  \param  crvecArguments  A vector containing the expression objects, which shall be used as arguments for the intrinsic function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVX2Enum eIntrinID, const ClangASTHelper::ExpressionVectorType &crvecArguments)
    {
      return InstructionSetBase::_CreateFunctionCall(_mapIntrinsicsAVX2, eIntrinID, crvecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with one call parameter.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the argument of the function. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVX2Enum eIntrinID, ::clang::Expr *pArg1)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with two call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVX2Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }

    /** \brief  Creates a function call expression object to an intrinsic function with three call parameters.
     *  \param  eIntrinID   The internal ID of the requested intrinsic function.
     *  \param  pArg1       A pointer to the expression object, which serves as the first argument.
     *  \param  pArg2       A pointer to the expression object, which serves as the second argument.
     *  \param  pArg3       A pointer to the expression object, which serves as the third argument. */
    inline ::clang::CallExpr* _CreateFunctionCall(IntrinsicsAVX2Enum eIntrinID, ::clang::Expr *pArg1, ::clang::Expr *pArg2, ::clang::Expr *pArg3)
    {
      ClangASTHelper::ExpressionVectorType vecArguments;

      vecArguments.push_back(pArg1);
      vecArguments.push_back(pArg2);
      vecArguments.push_back(pArg3);

      return _CreateFunctionCall(eIntrinID, vecArguments);
    }


  protected:

    InstructionSetAVX2(::clang::ASTContext &rAstContext);

    /** \brief  Returns an expression, which shifts the contents of a vector within 128bit lanes by a specified amount of bytes.
     *  \param  pVectorRef    A pointer to a vectorized expression object, which returns the vector whose contents shall be shifted.
     *  \param  uiByteCount   The number of bytes, by which the vector contents shall be shifted.
     *  \param  bShiftLeft    A flag indicating, whether the vector contents shall be shifted to the left or to the right. */
    ::clang::Expr* _ShiftIntegerVectorBytes(::clang::Expr *pVectorRef, std::uint32_t uiByteCount, bool bShiftLeft);

    /** \brief  Returns an expression, which extracts one half of an AVX vector into a SSE vector.
     *  \param  eElementType  The element type stored in the vector.
     *  \param  pAVXVector    A pointer to a vectorized expression, which returns the AVX vector used for the extraction.
     *  \param  bLowHalf      A flag indicating, whether the low half or the high half of the AVX vector shall be extracted. */
    ::clang::Expr*  _ExtractSSEVector(VectorElementTypes eElementType, ::clang::Expr *pAVXVector, bool bLowHalf);


  public:

    virtual ~InstructionSetAVX2()
    {
      _mapIntrinsicsAVX2.clear();
    }


  protected:

    /** \name Instruction set abstraction methods */
    //@{

    virtual ::clang::Expr* _ConvertVector(VectorElementTypes eSourceType, VectorElementTypes eTargetType, const ClangASTHelper::ExpressionVectorType &crvecVectorRefs, std::uint32_t uiGroupIndex, bool bMaskConversion) final override;

    //@}

  public:

    /** \name Instruction set abstraction methods */
    //@{
    virtual bool IsBuiltinFunctionSupported(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, std::uint32_t uiParamCount) const final override;

    virtual ::clang::Expr* ArithmeticOperator(VectorElementTypes eElementType, ArithmeticOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS, bool bIsRHSScalar=false) final override;
    virtual ::clang::Expr* BlendVectors(VectorElementTypes eElementType, ::clang::Expr *pMaskRef, ::clang::Expr *pVectorTrue, ::clang::Expr *pVectorFalse) final override;
    virtual ::clang::Expr* BuiltinFunction(VectorElementTypes eElementType, BuiltinFunctionsEnum eFunctionType, const ClangASTHelper::ExpressionVectorType &crvecArguments) final override;
    virtual ::clang::Expr* CheckActiveElements(VectorElementTypes eMaskElementType, ActiveElementsCheckType eCheckType, ::clang::Expr *pMaskExpr) final override;
    virtual ::clang::Expr* CheckSingleMaskElement(VectorElementTypes eMaskElementType, ::clang::Expr *pMaskExpr, std::uint32_t uiIndex) final override;
    virtual ::clang::Expr* LoadVectorGathered(VectorElementTypes eElementType, VectorElementTypes eIndexElementType, ::clang::Expr *pPointerRef, const ClangASTHelper::ExpressionVectorType &crvecIndexExprs, uint32_t uiGroupIndex) final override;
    virtual ::clang::Expr* RelationalOperator(VectorElementTypes eElementType, RelationalOperatorType eOpType, ::clang::Expr *pExprLHS, ::clang::Expr *pExprRHS) final override;
    virtual ::clang::Expr* ShiftElements(VectorElementTypes eElementType, ::clang::Expr *pVectorRef, bool bShiftLeft, uint32_t uiCount) final override;
    virtual ::clang::Expr* StoreVectorMasked(VectorElementTypes eElementType, ::clang::Expr *pPointerRef, ::clang::Expr *pVectorValue, ::clang::Expr *pMaskRef) final override;

  };

  //@}
} // end namespace Vectorization
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#ifdef VERBOSE_INIT_MODE
#undef VERBOSE_INIT_MODE
#endif


#endif  // _HIPACC_BACKEND_INSTRUCTION_SETS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

