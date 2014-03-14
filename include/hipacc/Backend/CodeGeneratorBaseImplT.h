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

//===--- CodeGeneratorBaseImplT.h - Base class for all code generators. --------------===//
//
// This file implements the base generic base class for all code generators.
//
//===---------------------------------------------------------------------------------===//

#ifndef _BACKEND_CODE_GENERATOR_BASE_IMPL_T_H_
#define _BACKEND_CODE_GENERATOR_BASE_IMPL_T_H_

#include "hipacc/Config/CompilerOptions.h"
#include "BackendExceptions.h"
#include "CommonDefines.h"
#include "ICodeGenerator.h"
#include <map>
#include <string>
#include <utility>


namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \brief  Base class for all code generators which handles the common implementations.
   *  \tparam SwitchTypeEnum  An (strongly-typed) enumeration type which specifies the known switches of the derived specific code generator.
   *  \implements ICodeGenerator */
  template < typename SwitchTypeEnum >
  class CodeGeneratorBaseImplT : public ICodeGenerator
  {
  protected:

    /** \name Protected definitions for the derived classes. */
    //@{

    typedef CommonDefines::CompilerSwitchInfoT< SwitchTypeEnum >  CompilerSwitchInfoType;   //!< Type definition for the specialized switch information class.

    /** \brief    Helper class which encapsulates some code generator specific values.
     *  \remarks  Must be derived by each specific code generator in order to setup its base class correctly. */
    class CodeGeneratorDescriptorBase
    {
    private:

      ::clang::hipacc::TargetCode _eTargetCode;     //!< The internal ID of the specific code generator.
      std::string                 _strName;         //!< The internal name of the specific code generator.
      std::string                 _strEmissionKey;  //!< The suffix for the <b>-emit-...</b> compiler switch which selects the specific code generator.
      std::string                 _strDescription;  //!< The description of the specific code generator for the compiler usage.

    protected:

      inline CodeGeneratorDescriptorBase()  {}

      /** \brief  Sets the description of the specific code generator. */
      inline void SetDescription(std::string strNewDescription)               { _strDescription = strNewDescription; }

      /** \brief  Sets the suffix for the <b>-emit-...</b> compiler switch. */
      inline void SetEmissionKey(std::string strNewEmissionKey)               { _strEmissionKey = strNewEmissionKey; }

      /** \brief  Sets the internal name of the specific code generator. */
      inline void SetName(std::string strNewName)                             { _strName = strNewName; }

      /** \brief  Sets the internal ID of the specific code generator. */
      inline void SetTargetCode(::clang::hipacc::TargetCode eNewTargetCode)   { _eTargetCode = eNewTargetCode; }


    public:

      /** \brief  Copy constructor. */
      inline CodeGeneratorDescriptorBase(const CodeGeneratorDescriptorBase &crRVal) { *this = crRVal; }

      /** \brief  Assignment operator. */
      inline CodeGeneratorDescriptorBase& operator=(const CodeGeneratorDescriptorBase &crRVal)
      {
        _eTargetCode    = crRVal._eTargetCode;
        _strName        = crRVal._strName;
        _strEmissionKey = crRVal._strEmissionKey;
        _strDescription = crRVal._strDescription;

        return *this;
      }

      virtual ~CodeGeneratorDescriptorBase()  {}


      /** \brief  Returns the description of the specific code generator. */
      inline std::string                  Description() const   { return _strDescription; }

      /** \brief  Returns the suffix for the <b>-emit-...</b> compiler switch. */
      inline std::string                  EmissionKey() const   { return _strEmissionKey; }

      /** \brief  Returns the internal name of the specific code generator. */
      inline std::string                  Name() const         { return _strName; }

      /** \brief  Returns the internal ID of the specific code generator. */
      inline ::clang::hipacc::TargetCode  TargetCode() const   { return _eTargetCode; }
    };

    //@}


  private:

    /** \name Private members. */
    //@{
    typedef std::map< std::string, CompilerSwitchInfoType >   CompilerSwitchMapType;  //!< Type definition for the dictionary of known compiler switches.

    ::clang::hipacc::CompilerOptions    *_pCompilerOptions;     //!< A pointer to the global compiler options object (used for the configuration).
    CompilerSwitchMapType               _mapKnownSwitches;      //!< The dictionary of known compiler switches.
    const CodeGeneratorDescriptorBase   _Descriptor;            //!< The description object for the derived code generator.
    //@}


  protected:

    /** \name Protected helper functions for the derived classes. */
    //@{

    /** \brief  Returns a referenece to the global compiler options object. */
    inline ::clang::hipacc::CompilerOptions& GetCompilerOptions()   { return *_pCompilerOptions; }

    /** \brief    Enters a compiler switch into the known switches dictionary.
     *  \remarks  The purpose of this function is to establish a link between the string and enum representation of a compiler switch.
     *  \tparam   SwitchClass   The type of the switch structure (uses static polymorphism).
     *  \param    eSwitch       The specific enum value for this switch. */
    template <class SwitchClass>
    void _InitSwitch(SwitchTypeEnum eSwitch)
    {
      std::string strSwitch = SwitchClass::Key();

      if (_mapKnownSwitches.find(strSwitch) != _mapKnownSwitches.end())
      {
        throw InternalErrors::DuplicateSwitchEntryException(strSwitch, GetName());
      }
      else
      {
        CompilerSwitchInfoType SwitchInfo;

        SwitchInfo.SetAdditionalOptions(SwitchClass::AdditionalOptions());
        SwitchInfo.SetDescription(SwitchClass::Description());
        SwitchInfo.SetSwitchType(eSwitch);

        _mapKnownSwitches[strSwitch] = SwitchInfo;
      }
    }


    /** \brief    Parses the option of a specific switch from the command arguments.
     *  \tparam   SwitchClass     The type of the switch structure (uses static polymorphism).
     *  \param    rvecArguments   A reference to vector containing the command arguments for the code generator.
     *  \param    szSwitchIndex   The index of the currently processed switch in command arguments vector.
     *  \param    szOptionOffset  The index offset between the currently processed switch and the expected option in the command arguments vector.
     *  \return   If successful, the parsed value of the given option. */
    template <class SwitchClass>
    typename SwitchClass::OptionParser::ReturnType _ParseOption(CommonDefines::ArgumentVectorType &rvecArguments, size_t szSwitchIndex, size_t szOptionOffset = static_cast<size_t>(1))
    {
      // Fetch option
      if (rvecArguments.size() <= szSwitchIndex + szOptionOffset)
      {
        throw RuntimeErrors::MissingOptionException(rvecArguments[szSwitchIndex], GetName());
      }

      std::string strOption = rvecArguments[szSwitchIndex + szOptionOffset];

      // Parse option
      try
      {
        return SwitchClass::OptionParser::Parse(strOption);
      }
      catch (RuntimeErrors::InvalidOptionException &)
      {
        throw;
      }
      catch (BackendException &e)
      {
        llvm::errs() << "ERROR: " << e.what() << "\n\n";
        throw RuntimeErrors::InvalidOptionException(rvecArguments[szSwitchIndex], strOption);
      }
    }

    //@}



    /** \name Protected virtual functions for the derived class. */
    //@{

    /** \brief    Processes one specific compiler switch during the configuration processes.
     *  \param    eSwitch         The internal type of the currently processed switch.
     *  \param    rvecArguments   A reference to vector containing the command arguments for the code generator.
     *  \param    szSwitchIndex   The index of the currently processed switch in command arguments vector.
     *  \remarks  This function must be overridden by the derived class. The overriden function must throw an exception, when an argument is invalid or unknown.
     *  \return   The index of the last processed argument in the command arguments vector. */
    virtual size_t  _HandleSwitch(SwitchTypeEnum eSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurrentIndex) = 0;

    /** \brief    Checks the configuration for possible mistakes.
     *  \remarks  This function will be called directly after the configuration process. The derived class can override it to perform specific checks,
     *            but it should call the base implementation to ensure that the common configuration settings are checked as well. */
    virtual void _CheckConfiguration()
    {
      // kernels are timed internally by the runtime in case of exploration
      if (GetCompilerOptions().timeKernels(USER_ON) && GetCompilerOptions().exploreConfig(USER_ON))
      {
        GetCompilerOptions().setTimeKernels(OFF);
      }
    }

    //@}


  public:

    /** \brief  General constructor.
     *  \param  pCompilerOptions  A pointer to the global compiler options object (used for the configuration).
     *  \param  crDescriptor      A reference to a description object for the derived code generator. */
    CodeGeneratorBaseImplT(::clang::hipacc::CompilerOptions *pCompilerOptions, const CodeGeneratorDescriptorBase &crDescriptor) : _pCompilerOptions(pCompilerOptions), _Descriptor(crDescriptor)
    {
      if (_pCompilerOptions == nullptr)
      {
        throw BackendException("Compiler options have not been set");
      }
    }

    virtual ~CodeGeneratorBaseImplT()
    {
      _mapKnownSwitches.clear();
    }


    /** \name ICodeGenerator members */
    //@{
    virtual std::string GetDescription() const final override { return _Descriptor.Description(); }
    virtual std::string GetEmissionKey() const final override { return _Descriptor.EmissionKey(); }
    virtual std::string GetName() const final override        { return _Descriptor.Name(); }

    virtual CommonDefines::SwitchDisplayInfoVectorType GetCompilerSwitches() const final override
    {
      CommonDefines::SwitchDisplayInfoVectorType vecKnownSwitches;
      vecKnownSwitches.reserve(_mapKnownSwitches.size());

      for (auto itSwitch = _mapKnownSwitches.begin(); itSwitch != _mapKnownSwitches.end(); itSwitch++)
      {
        vecKnownSwitches.push_back(itSwitch->second.CreateDisplayInfo(itSwitch->first));
      }

      return vecKnownSwitches;
    }

    virtual void Configure(CommonDefines::ArgumentVectorType & rvecArguments) final override
    {
      // Set the target code for this code generator
      GetCompilerOptions().setTargetCode(_Descriptor.TargetCode());

      // Parse all command line switches and options by the derived code generator
      for (size_t i = static_cast<size_t>(0); i < rvecArguments.size(); ++i)
      {
        std::string strSwitch = rvecArguments[i];

        auto itSwitchEntry = _mapKnownSwitches.find(strSwitch);

        if (itSwitchEntry == _mapKnownSwitches.end())
        {
          throw RuntimeErrors::UnknownSwitchException(strSwitch, GetName());
        }

        i = _HandleSwitch(itSwitchEntry->second.GetSwitchType(), rvecArguments, i);
      }

      // Finally, check the configuration
      _CheckConfiguration();
    }
    //@}
  };
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_CODE_GENERATOR_BASE_IMPL_T_H_

// vim: set ts=2 sw=2 sts=2 et ai:

