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

//===--- BackendExceptions.h - Definitions of the exception types which can be thrown by the backend. ---===//
//
// This file implements the exception types which can be thrown by the backend.
//
//===----------------------------------------------------------------------------------------------------===//

#ifndef _BACKEND_EXCEPTIONS_H_
#define _BACKEND_EXCEPTIONS_H_

#include <stdexcept>
#include <string>


#ifdef _MSC_VER
#define DECLARE_NOEXCEPT
#else
#define DECLARE_NOEXCEPT noexcept (true)
#endif


namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \name Exception base classes. */
  //@{

  /** \brief    Root class of all exceptions which can be thrown by the backend. 
   *  \extends  std::runtime_error */
  class BackendException : public std::runtime_error
  {
  private:

    typedef std::runtime_error  BaseType;   //!< The base type of this class.

  public:

    /** \brief  Constructor.
     *  \param  strMessage  The message which shall be displayed. */
    inline BackendException(std::string strMessage) : BaseType(std::string("Backend exception: ") + strMessage) {}

    virtual ~BackendException() DECLARE_NOEXCEPT  {}
  };

  /** \brief    Root class for all internal errors of the backend. 
   *  \remarks  This type of error is caused by invalid assumptions at design time. Thus a user should never see an exception like in an ideal case.
   *  \extends  BackendException */
  class InternalErrorException : public BackendException
  {
  private:

    typedef BackendException  BaseType;   //!< The base type of this class.

  public:

    /** \brief  Constructor.
     *  \param  strMessage  The message which shall be displayed. */
    inline InternalErrorException(std::string strMessage) : BaseType(std::string("Internal error: ") + strMessage)  {}

    virtual ~InternalErrorException() DECLARE_NOEXCEPT  {}
  };

  /** \brief    Root class for all run-time errors of the backend.
   *  \remarks  This type of error is raised whenever a problem occurs at run-time, e.g. wrong parametrization or incompatible input files etc.
   *  \extends  BackendException */
  class RuntimeErrorException : public BackendException
  {
  private:

    typedef BackendException  BaseType;   //!< The base type of this class.

  public:

    /** \brief  Constructor.
     *  \param  strMessage  The message which shall be displayed. */
    inline RuntimeErrorException(std::string strMessage) : BaseType(std::string("Runtime error: ") + strMessage)  {}

    virtual ~RuntimeErrorException() DECLARE_NOEXCEPT {}
  };

  //@}


  /** \brief  Contains all specialized internal error exception classes. */
  class InternalErrors
  {
  public:

    /** \brief    Internal error which indicates that a compiler switch has been declared twice with different meanings.
     *  \extends  InternalErrorException */
    class DuplicateSwitchEntryException final : public InternalErrorException
    {
    private:

      typedef InternalErrorException  BaseType;   //!< The base type of this class.

    public:

      /** \brief  General constructor.
       *  \param  strSwitch  The compiler switch which caused this error. */
      inline DuplicateSwitchEntryException(std::string strSwitch) : BaseType(std::string("The switch \"") + strSwitch + std::string("\" has already been defined!"))  {}

      /** \brief  Special constructor for the code generators.
       *  \param  strSwitch         The compiler switch which caused this error.
       *  \param  strGeneratorName  The name of the code generator for which the error happened. */
      inline DuplicateSwitchEntryException(std::string strSwitch, std::string strGeneratorName) : BaseType( std::string("The switch \"") + strSwitch +
                                                                                                            std::string("\" has already been defined in code generator \"") +
                                                                                                            strGeneratorName + std::string("\"!") )
      {}
    };


    /** \brief    Internal error which indicates that a pointer is NULL although this should not be possible.
     *  \extends  InternalErrorException */
    class NullPointerException final : public InternalErrorException
    {
    private:

      typedef InternalErrorException  BaseType;   //!< The base type of this class.

    public:

      /** \brief  General constructor.
       *  \param  strPointerName  The name of the pointer which caused this error. */
      inline NullPointerException(std::string strPointerName) : BaseType(std::string("The pointer \"") + strPointerName + std::string("\" is NULL!"))  {}
    };


    /** \brief    Internal error which indicates that a configuration handler for a known compiler switch is missing.
     *  \extends  InternalErrorException */
    class UnhandledSwitchException final : public InternalErrorException
    {
    private:

      typedef InternalErrorException  BaseType;   //!< The base type of this class.

    public:

      /** \brief  General constructor.
       *  \param  strSwitch  The compiler switch which caused this error. */
      inline UnhandledSwitchException(std::string strSwitch) : BaseType(std::string("Handler for switch \"") + strSwitch + std::string("\" is missing!")) {}

      /** \brief  Special constructor for the code generators.
       *  \param  strSwitch         The compiler switch which caused this error.
       *  \param  strGeneratorName  The name of the code generator for which the error happened. */
      inline UnhandledSwitchException(std::string strSwitch, std::string strGeneratorName)  : BaseType( std::string("Handler for switch \"") + strSwitch +
                                                                                                        std::string("\" is missing in code generator \"") +
                                                                                                        strGeneratorName + std::string("\"!") )
      {}
    };
  };


  /** \brief  Contains all specialized run-time error exception classes. */
  class RuntimeErrors
  {
  public:

    /** \brief    Run-time error which indicates that a program abort has been requested.
     *  \extends  RuntimeErrorException */
    class AbortException final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;   //!< The base type of this class.

      const int _ciExitCode;                      //!< The requested exit code for the abort.

    public:

      /** \brief  General constructor.
       *  \param  iExitCode  The requested exit code for the abort. */
      inline AbortException(int iExitCode) : BaseType("Abort!"), _ciExitCode(iExitCode) {}

      /** \brief  Returns the requested abort code. */
      inline int GetExitCode() const  { return _ciExitCode; }
    };


    /** \brief    Run-time error which indicates that the user-defined option for a compiler switch is invalid.
     *  \extends  RuntimeErrorException */
    class InvalidOptionException final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;   //!< The base type of this class.

    public:

      /** \brief  General constructor.
       *  \param  strSwitch  The compiler switch whose option is invalid.
       *  \param  strOption  The compiler switch option which caused this error. */
      inline InvalidOptionException(std::string strSwitch, std::string strOption) : BaseType( std::string("The option \"") + strOption +
                                                                                              std::string("\" is invalid for the switch \"") +
                                                                                              strSwitch + std::string("\"!") )
      {}
    };


    /** \brief    Run-time error which indicates that a required option for a user-defined compiler switch is missing.
     *  \extends  RuntimeErrorException */
    class MissingOptionException final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;   //!< The base type of this class.

    public:

      /** \brief  General constructor.
       *  \param  strSwitch  The compiler switch which caused this error. */
      inline MissingOptionException(std::string strSwitch)  : BaseType(std::string("The required option for switch \"") + strSwitch + std::string("\" is missing!"))
      {}

      /** \brief  Special constructor for the code generators.
       *  \param  strSwitch         The compiler switch which caused this error.
       *  \param  strGeneratorName  The name of the code generator for which the error happened. */
      inline MissingOptionException(std::string strSwitch, std::string strGeneratorName)  : BaseType( std::string("The required option for switch \"") + strSwitch +
                                                                                                      std::string("\" is missing for code generator \"") +
                                                                                                      strGeneratorName + std::string("\"!") )
      {}
    };


    /** \brief    Run-time error which indicates that a user-defined compiler switch is not known.
     *  \extends  RuntimeErrorException */
    class UnknownSwitchException final : public RuntimeErrorException
    {
    private:

      typedef RuntimeErrorException   BaseType;   //!< The base type of this class.

    public:

      /** \brief  Special constructor for the code generators.
       *  \param  strSwitch         The compiler switch which caused this error.
       *  \param  strGeneratorName  The name of the code generator for which the error happened. */
      inline UnknownSwitchException(std::string strSwitch, std::string strGeneratorName)  : BaseType( std::string("The switch \"") + strSwitch +
                                                                                                      std::string("\" is not supported in code generator \"") +
                                                                                                      strGeneratorName + std::string("\"!") )
      {}
    };
  };
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_EXCEPTIONS_H_

// vim: set ts=2 sw=2 sts=2 et ai:

