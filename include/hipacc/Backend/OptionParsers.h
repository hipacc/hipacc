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

//===--- OptionParsers.h - Commonly used parsing routines for command line options----===//
//
// This file contains commonly used parsing routines for command line options-.
//
//===---------------------------------------------------------------------------------===//

#ifndef _HIPACC_BACKEND_OPTION_PARSERS_H_
#define _HIPACC_BACKEND_OPTION_PARSERS_H_

#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Backend/CommonDefines.h"

#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

namespace clang
{
namespace hipacc
{
namespace Backend
{
  namespace CommonDefines
  {
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

      /** \brief  Common parser for "path" options, which is a string. */
      struct Path final
      {
        typedef std::string   ReturnType;

        /** \brief  Tries to parse the option as a string.
         *  \param  strOption   The command line option as string.
         *  \return If successful, the option as a string. */
        inline static ReturnType Parse(std::string strOption)
        {
          struct stat info;

          if (stat(strOption.c_str(), &info) == 0)
          {
            //if (info.st_mode & S_IFDIR) { /* directory */ }
            //else { /* file */ }
            return strOption;
          }
          else { throw RuntimeErrorException("Invalid path"); }
        }
      };
    };
  }
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _HIPACC_BACKEND_OPTION_PARSERS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

