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

//===--- BackendConfigurationManager.h - Configures the hipacc compiler backend ---===//
//
// This file implements the configuration of the hipacc compiler backend and code generator.
//
//===------------------------------------------------------------------------------===//

#ifndef _BACKEND_CONFIGURATION_MANAGER_H_
#define _BACKEND_CONFIGURATION_MANAGER_H_

#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include "hipacc/Config/CompilerOptions.h"
#include "BackendExceptions.h"
#include "CommonDefines.h"
#include "ICodeGenerator.h"

namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \brief  Manages all known backend code generators and controls the compiler configuration process. */
  class BackendConfigurationManager final
  {
  private:

    /** \name Internal type definitions. */
    //@{

    /** \brief  Contains the IDs of all supported specific compiler switches for this backend. */
    enum class CompilerSwitchTypeEnum
    {
      Emit,         //!< ID of all code generator selection switches
      Help,         //!< ID of the "print help" switch
      OutputFile,   //!< ID of the "output file" switch
      Version       //!< ID of the "print version" switch
    };


    typedef CommonDefines::CompilerSwitchInfoT< CompilerSwitchTypeEnum >  CompilerSwitchInfoType;   //!< Type definition for the switch information class.

    typedef std::map< std::string, CompilerSwitchInfoType >   CompilerSwitchMapType;  //!< Type definition for the dictionary of known compiler switches.
    typedef std::map< std::string, std::string >              SwitchAliasMapType;     //!< Type definition for the dictionary of known compiler switch aliases.
    typedef std::map< std::string, ICodeGeneratorPtr >        CodeGeneratorsMapType;  //!< Type definition for the dictionary of known code generators.

    //@}


  private:

    /** \name Internal class declarations. */
    //@{

    /** \brief  Contains all known common compiler switches which are independent of the selected code generator. */
    class KnownSwitches final
    {
    public:

      typedef std::vector< std::string >  AliasesVectorType;  //!< Type definition for a vector of aliases for a switch

    public:

      /** \brief  Returns the prefix for all code generator selection switches. */
      inline static std::string EmissionSwitchBase()  { return "-emit-"; }


      /** \brief  The switch type for the "print help" switch. */
      struct Help final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "--help"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return ""; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Display available options"; }

        /** \brief  Returns the vector of known aliases for this switch. */
        inline static AliasesVectorType GetAliases()
        {
          AliasesVectorType vecDuplicates;

          vecDuplicates.push_back("-help");

          return vecDuplicates;
        }
      };

      /** \brief  The switch type for the "output file" switch. */
      struct OutputFile final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-o"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<file>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Write output to <file>"; }

        /** \brief  Returns the vector of known aliases for this switch. */
        inline static AliasesVectorType GetAliases()    { return AliasesVectorType(); }
      };

      /** \brief  The switch type for the "print version" switch. */
      struct Version final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "--version"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return ""; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Display version information"; }

        /** \brief  Returns the vector of known aliases for this switch. */
        inline static AliasesVectorType GetAliases()
        {
          AliasesVectorType vecDuplicates;

          vecDuplicates.push_back("-version");

          return vecDuplicates;
        }
      };
    };

    /** \brief  Handles the output to the console (e.g. printing of the usage). */
    class ConsoleOutput final
    {
    private:

      const size_t _cszPrintWidth;            //!< The maximum allowed line width for the console output.
      const size_t _cszMinDescriptionWidth;   //!< The minimum width for one line of the compiler switch description.
      const size_t _cszPadLeft;               //!< The intend for the printing of the compiler switches.
      const size_t _cszDescriptionDistance;   //!< The printing distance between a compiler switch name and its decription.

      /** \brief  Returns a string containing only whitespaces.
       *  \param  szPadSize   The requested length of the pad string. */
      std::string _GetPadString(size_t szPadSize);

      /** \brief  Internal function which prints the display information for compiler switches. 
       *  \param  crvecSwitches   A reference to the display information vector for the compiler switches. */
      void _PrintSwitches(const CommonDefines::SwitchDisplayInfoVectorType &crvecSwitches);

    public:

      /** \brief  Constructor.
       *  \param  szPrintWidth  The maximum allowed line width for the console output. */
      ConsoleOutput(size_t szPrintWidth) : _cszPrintWidth(szPrintWidth), _cszMinDescriptionWidth(20), _cszPadLeft(2), _cszDescriptionDistance(2)
      {}


      /** \brief  Prints the usage of one specific code generator.
       *  \param  spCodeGenerator   A shared pointer to the code generator whose usage shall be printed. */
      void PrintCodeGeneratorSwitches(ICodeGeneratorPtr spCodeGenerator);

      /** \brief  Prints the usage of the HIPAcc compiler and a summary of all known common compiler switches.
       *  \param  crvecCommonSwitches   A reference to the display information vector for the common compiler switches. */
      void PrintUsage(const CommonDefines::SwitchDisplayInfoVectorType &crvecCommonSwitches);

      /** \brief  Prints the version of the HIPAcc compiler. */
      void PrintVersion();
    };

    //@}


  private:

    ConsoleOutput                     _ConsoleOutput;       //!< Private instance of the ConsoleOutput class.
    ::clang::hipacc::CompilerOptions  *_pCompilerOptions;   //!< A pointer to the global compiler options object (used for the configuration).

    CompilerSwitchMapType   _mapKnownSwitches;      //!< The dictionary of known compiler switches.
    SwitchAliasMapType      _mapSwitchAliases;      //!< The dictionary of known compiler switch aliases.
    CodeGeneratorsMapType   _mapCodeGenerators;     //!< The dictionary of known code generators.

    std::string         _strInputFile;              //!< The path to the user-defined input file (will be set in the configuration process).
    std::string         _strOutputFile;             //!< The path to the user-defined output file (will be set in the configuration process).
    ICodeGeneratorPtr   _spSelectedCodeGenerator;   //!< A shared pointer to the user-defined code generator (will be set in the configuration process).



    /** \brief    Enters a code generator into the known code generators dictionary.
     *  \remarks  The purpose of this function is to establish a link between an code generator object and its corresponding "-emit-..." switch.
     *  \tparam   BackendType   The type of the backend which contains the code generator (uses static polymorphism). */
    template <class BackendType>
    void _InitBackend()
    {
      typedef typename BackendType::CodeGenerator   GeneratorType;

      static_assert(std::is_base_of< ICodeGenerator, GeneratorType >::value, "Code generators must be derived from \"ICodeGenerator\"");

      ICodeGeneratorPtr spCodeGenerator(new GeneratorType(_pCompilerOptions));

      std::string strEmissionKey = KnownSwitches::EmissionSwitchBase() + spCodeGenerator->GetEmissionKey();

      if (_mapCodeGenerators.find(strEmissionKey) != _mapCodeGenerators.end())
      {
        throw InternalErrors::DuplicateSwitchEntryException(strEmissionKey);
      }
      else
      {
        _mapCodeGenerators[strEmissionKey] = spCodeGenerator;
        _mapKnownSwitches[strEmissionKey] = CompilerSwitchInfoType(CompilerSwitchTypeEnum::Emit, spCodeGenerator->GetDescription());
      }
    }


    /** \brief    Enters a compiler switch into the known switches dictionary.
     *  \remarks  The purpose of this function is to establish a link between the string and enum representation of a compiler switch.
     *            Furthermore, all known aliases for this switch (if there are any) are automatically set.
     *  \tparam   SwitchClass   The type of the switch structure (uses static polymorphism).
     *  \param    eSwitch       The specific enum value for this switch. */
    template <class SwitchClass>
    void _InitSwitch(CompilerSwitchTypeEnum eSwitch)
    {
      // Extract switch key
      std::string strSwitch = SwitchClass::Key();

      // Check for duplicate switch entry
      if (_mapKnownSwitches.find(strSwitch) != _mapKnownSwitches.end())
      {
        throw InternalErrors::DuplicateSwitchEntryException(strSwitch);
      }
      else
      {
        // Enter switch into the "known switches" map
        CompilerSwitchInfoType SwitchInfo;

        SwitchInfo.SetAdditionalOptions(SwitchClass::AdditionalOptions());
        SwitchInfo.SetDescription(SwitchClass::Description());
        SwitchInfo.SetSwitchType(eSwitch);

        _mapKnownSwitches[strSwitch] = SwitchInfo;


        // Set all switches Aliases
        KnownSwitches::AliasesVectorType vecAliases = SwitchClass::GetAliases();

        for (std::string strAlias : vecAliases)
        {
          _mapSwitchAliases[strAlias] = strSwitch;
        }
      }
    }


    /** \brief    Processes one specific compiler switch during the configuration processes.
     *  \param    strSwitch       The currently processed switch.
     *  \param    rvecArguments   A reference to vector containing the command arguments.
     *  \param    szSwitchIndex   The index of the currently processed switch in command arguments vector.
     *  \return   The index of the last processed argument in the command arguments vector. */
    size_t _HandleSwitch(std::string strSwitch, CommonDefines::ArgumentVectorType &rvecArguments, size_t szCurIndex);

    /** \brief    Translates a switch alias into the actual compiler switch.
     *  \remarks  A switch alias is another name for an existing switch (like a symbolic link).
     *  \param    strSwitch   The switch which shall be translated.
     *  \return   The translated switch, if the input was an alias, or the input switch otherwise. */
    std::string _TranslateSwitchAlias(std::string strSwitch);


  public:

    /** \brief  Initializes the backend configuration manager.
     *  \param  pCompilerOptions  A pointer to the global compiler options object. */
    BackendConfigurationManager(::clang::hipacc::CompilerOptions *pCompilerOptions);

    // This class is supposed to exist only once => delete copy constructor and assignment operator
    BackendConfigurationManager(const BackendConfigurationManager&) = delete;
    BackendConfigurationManager& operator=(const BackendConfigurationManager&) = delete;


    /** \brief    Parses a vector of command line arguments and configures the compiler.
     *  \remarks  This method also selects the user-defined code generator and launches its Configure() method. */
    void Configure(CommonDefines::ArgumentVectorType &rvecArguments);

    /** \brief  Returns the command arguments vector required for the clang frontend invocation. */
    CommonDefines::ArgumentVectorType GetClangArguments();
  };
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_CONFIGURATION_MANAGER_H_

// vim: set ts=2 sw=2 sts=2 et ai:

