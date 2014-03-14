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

//===--- CommonDefines.h - Common definitions for the Backend library. ---------------===//
//
// This file contains common definitions and classes for the Backend library.
//
//===---------------------------------------------------------------------------------===//

#ifndef _BACKEND_COMMON_DEFINES_H_
#define _BACKEND_COMMON_DEFINES_H_

#include "hipacc/Config/CompilerOptions.h"
#include "BackendExceptions.h"
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \brief  Contains a set of common type definitions, classes etc. for the backend library. */
  class CommonDefines final
  {
  public:

    typedef std::vector< std::string >              ArgumentVectorType;           //!< Type definition for a vector of command arguments.

    typedef std::pair< std::string, std::string >   SwitchDisplayInfoType;        //!< Type definition for the display information of a compiler switch (for the compiler usage).
    typedef std::vector< SwitchDisplayInfoType >    SwitchDisplayInfoVectorType;  //!< Type definition for a vector of switch display informations.


    /** \brief  Encapsulates the necessary information about a known compiler switch.
     *  \tparam SwitchTypeEnum  An (strongly-typed) enumeration type which specifies the known switches of a specific code generator. */
    template < typename SwitchTypeEnum >
    class CompilerSwitchInfoT final
    {
    private:

      static_assert(std::is_enum<SwitchTypeEnum>::value, "The \"switch type\"-type must be an enum or strongly-typed enum!");

      SwitchTypeEnum    _eSwitchType;           //!< The internal enum value representing this switch.
      std::string       _strDescription;        //!< The user-readable description of the compiler switch.
      std::string       _strAdditionalOptions;  //!< A string containing the place-holder for additional options (can be empty).

    public:

      /** \brief  Default constructor. */
      inline CompilerSwitchInfoT()  {}

      /** \brief  "Single-line" constructor.
       *  \param  eType                 The internal enum value representing this switch.
       *  \param  strDescription        The user-readable description of the compiler switch.
       *  \param  strAdditionalOptions  A string containing the place-holder for additional options (can be empty). */
      inline CompilerSwitchInfoT(SwitchTypeEnum eType, std::string strDescription, std::string strAdditionalOptions = "")
      {
        _eSwitchType = eType;
        _strDescription = strDescription;
        _strAdditionalOptions = strAdditionalOptions;
      }

      /** \brief  Copy constructor. */
      inline CompilerSwitchInfoT(const CompilerSwitchInfoT &crRVal)
      {
        *this = crRVal;
      }

      /** \brief  Assignment operator. */
      inline CompilerSwitchInfoT& operator=(const CompilerSwitchInfoT &crRVal)
      {
        _eSwitchType = crRVal._eSwitchType;
        _strDescription = crRVal._strDescription;
        _strAdditionalOptions = crRVal._strAdditionalOptions;

        return *this;
      }

      /** \brief  Creates a switch display information object for this switch.
       *  \param  strSwitch  The command for this compiler switch (it is not stored internally). */
      inline SwitchDisplayInfoType CreateDisplayInfo(std::string strSwitch) const
      {
        SwitchDisplayInfoType SwitchInfo(strSwitch, GetDescription());

        if (!_strAdditionalOptions.empty())
        {
          SwitchInfo.first += std::string(" ") + _strAdditionalOptions;
        }

        return SwitchInfo;
      }


      /** \name Public properties. */
      //@{

      /** \brief  Gets the currently set additional options string. */
      inline std::string  GetAdditionalOptions() const                      { return _strAdditionalOptions; }
      /** \brief  Sets a new additional options string.
       *  \param  strNewOptions  The new additional options string. */
      inline void         SetAdditionalOptions(std::string strNewOptions)   { _strAdditionalOptions = strNewOptions; }

      /** \brief  Gets the currently set switch description. */
      inline std::string  GetDescription() const                          { return _strDescription; }
      /** \brief  Sets a new switch description.
       *  \param  strNewDescription  The new switch description. */
      inline void         SetDescription(std::string strNewDescription)   { _strDescription = strNewDescription; }

      /** \brief  Gets the currently set internal switch type. */
      inline SwitchTypeEnum GetSwitchType() const                         { return _eSwitchType; }
      /** \brief  Sets a new internal switch type.
       *  \param  eNewSwitchType  The new internal switch type. */
      inline void           SetSwitchType(SwitchTypeEnum eNewSwitchType)  { _eSwitchType = eNewSwitchType; }

      //@}
    };


    /** \brief    Contains common parsing routines for command line options.
     *  \remarks  The internal option parsers rely on static polymorphism, thus each parser struct must define the type <b>ReturnType</b>
     *            and the static method <b>Parse()</b>. */
    class OptionParsers final
    {
    public:

      /** \brief  Common parser for integral options. */
      struct Integer final
      {
        typedef int   ReturnType;   //!<  The type of the parsed option.

        /** \brief  Tries to parse the option as an integer.
         *  \param  strOption   The command line option as string.
         *  \return If successful, the option as an integer value. */
        inline static ReturnType Parse(std::string strOption)
        {
          std::istringstream buffer(strOption.c_str());

          int iRetVal;
          buffer >> iRetVal;

          if (buffer.fail())
          {
            throw RuntimeErrorException("Expected integer value");
          }

          return iRetVal;
        }
      };

      /** \brief  Common parser for "boolean" options, which can be either <b>on</b> or <b>off</b>. */
      struct OnOff final
      {
        typedef ::clang::hipacc::CompilerOption   ReturnType;

        /** \brief  Tries to parse the option as a pseudo-boolean <b>on / off</b> value.
         *  \param  strOption   The command line option as string.
         *  \return If successful, the option as a <b>clang::hipacc::CompilerOption</b> value. */
        inline static ReturnType Parse(std::string strOption)
        {
          if      (strOption == "off")  return USER_OFF;
          else if (strOption == "on")   return USER_ON;
          else
          {
            throw RuntimeErrorException("Invalid value");
          }
        }
      };
    };
  };
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_COMMON_DEFINES_H_

// vim: set ts=2 sw=2 sts=2 et ai:

