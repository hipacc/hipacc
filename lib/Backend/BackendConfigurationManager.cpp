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

//===--- BackendConfigurationManager.cpp - Configures the hipacc compiler backend ---===//
//
// This file implements the configuration of the hipacc compiler backend and code generator.
//
//===--------------------------------------------------------------------------------===//

#include "hipacc/Backend/BackendConfigurationManager.h"
#include "hipacc/Config/config.h"
#include "llvm/Support/raw_ostream.h"

// Supported code generators
#include "hipacc/Backend/CPU_x86.h"
#include "hipacc/Backend/CUDA.h"
#include "hipacc/Backend/OpenCL_ACC.h"
#include "hipacc/Backend/OpenCL_CPU.h"
#include "hipacc/Backend/OpenCL_GPU.h"

using namespace clang::hipacc::Backend;
using namespace clang::hipacc;

// Implementation of class BackendConfigurationManager::ConsoleOutput
std::string BackendConfigurationManager::ConsoleOutput::_GetPadString(size_t szPadSize)
{
  std::string strPadString("");
  strPadString.resize(szPadSize, ' ');
  return strPadString;
}

void BackendConfigurationManager::ConsoleOutput::_PrintSwitches(const CommonDefines::SwitchDisplayInfoVectorType &crvecSwitches)
{
  // Fetch maximum width of switch string
  size_t szMaxSwitchWidth = static_cast<size_t>(0);
  for (auto itCurrentSwitch : crvecSwitches)
  {
    size_t szCurrentSize = itCurrentSwitch.first.length();

    if (szCurrentSize > szMaxSwitchWidth)
    {
      szMaxSwitchWidth = szCurrentSize;
    }
  }

  // Compute padded switch width and description width
  szMaxSwitchWidth          += _cszPadLeft + _cszDescriptionDistance;
  size_t szDescriptionWidth  = _cszMinDescriptionWidth;

  if (_cszMinDescriptionWidth + szMaxSwitchWidth < _cszPrintWidth)
  {
    szDescriptionWidth = _cszPrintWidth - szMaxSwitchWidth;
  }


  // Re-format every switch entry and print it
  for (auto itCurrentSwitch : crvecSwitches)
  {
    // Pad the switch key
    std::string strPrintString  = _GetPadString(_cszPadLeft) + itCurrentSwitch.first;
    strPrintString        += _GetPadString(szMaxSwitchWidth - strPrintString.length());

    // Break the description into pieces
    std::vector<std::string> vecDescriptionSubStrings;
    std::string strDescription = itCurrentSwitch.second;

    // Find all new-line characters
    std::vector<int> vecNewLinePositions;
    vecNewLinePositions.push_back(-1);
    while (true)
    {
      std::string::size_type szNextNewLinePos = strDescription.find_first_of('\n', static_cast<std::string::size_type>(vecNewLinePositions.back() + 1));

      if (szNextNewLinePos != std::string::npos)
      {
        vecNewLinePositions.push_back(static_cast<int>(szNextNewLinePos));
      }
      else
      {
        // Push the position behind the last character to vector
        vecNewLinePositions.push_back(static_cast<int>(strDescription.length()));
        break;
      }
    }

    // Break the description into sections
    for (size_t i = static_cast<size_t>(0); i < vecNewLinePositions.size() - 1; ++i)
    {
      // Fetch the current section between two new-line characters
      std::string::size_type szSectionOffset = static_cast<std::string::size_type>(vecNewLinePositions[i] + 1);
      std::string::size_type szSectionLength = static_cast<std::string::size_type>(vecNewLinePositions[i + 1]) - szSectionOffset;

      std::string strCurrentSection = strDescription.substr(szSectionOffset, szSectionLength);

      // Break the current section into pieces of the maximum display width
      for (size_t szPieceOffset = static_cast<size_t>(0); szPieceOffset < strCurrentSection.length();)
      {
        size_t szPieceLength = szDescriptionWidth;

        if (szPieceOffset + szPieceLength >= strCurrentSection.length())
        {
          // Whole rest of the description can be displayed in one line
          vecDescriptionSubStrings.push_back(strCurrentSection.substr(szPieceOffset));
          break;
        }
        else
        {
          // The rest of the description still needs to be broken into several lines
          std::string strCurrentPiece = strCurrentSection.substr(szPieceOffset, szPieceLength);

          std::string::size_type szLastWhiteSpace = strCurrentPiece.find_last_of(' ');

          if (szLastWhiteSpace == std::string::npos)
          {
            vecDescriptionSubStrings.push_back(strCurrentPiece);
            szPieceOffset += szPieceLength;
          }
          else
          {
            // Whitespace found => Break at whitespace
            vecDescriptionSubStrings.push_back(strCurrentPiece.substr(0, szLastWhiteSpace));
            szPieceOffset += szLastWhiteSpace + 1;
          }
        }
      }
    }

    // Build the final print-string and print it
    strPrintString += vecDescriptionSubStrings[0] + std::string("\n");
    for (size_t i = static_cast<size_t>(1); i < vecDescriptionSubStrings.size(); ++i)
    {
      strPrintString += _GetPadString(szMaxSwitchWidth) + vecDescriptionSubStrings[i] + std::string("\n");
    }

    llvm::errs() << strPrintString;
  }
}

void BackendConfigurationManager::ConsoleOutput::PrintCodeGeneratorSwitches(ICodeGeneratorPtr spCodeGenerator)
{
  if (spCodeGenerator == nullptr)
  {
    throw InternalErrors::NullPointerException("spCodeGenerator");
  }

  CommonDefines::SwitchDisplayInfoVectorType vecCodeGeneratorSwitches = spCodeGenerator->GetCompilerSwitches();

  // Only print the code generator specifc usage if the code generator has specific switches
  if (! vecCodeGeneratorSwitches.empty())
  {
    llvm::errs() << "\nSpecific options for code generator \"" << spCodeGenerator->GetName() << "\":\n\n";

    _PrintSwitches(vecCodeGeneratorSwitches);
  }
}

void BackendConfigurationManager::ConsoleOutput::PrintUsage(const CommonDefines::SwitchDisplayInfoVectorType &crvecCommonSwitches)
{
  // Print head-lines
  llvm::errs() << "OVERVIEW: HIPAcc - Heterogeneous Image Processing Acceleration framework\n\n";
  llvm::errs() << "USAGE:  hipacc [options] <input>\n\n";
  llvm::errs() << "OPTIONS:\n\n";

  // Format and known common switches
  _PrintSwitches(crvecCommonSwitches);
}

void BackendConfigurationManager::ConsoleOutput::PrintVersion()
{
  llvm::errs() << "hipacc version " << HIPACC_VERSION << " (" << GIT_REPOSITORY " " << GIT_VERSION << ")\n";
}


// Implementation of class BackendConfigurationManager
BackendConfigurationManager::BackendConfigurationManager(CompilerOptions *pCompilerOptions) : _ConsoleOutput( static_cast<size_t>(110) ), _pCompilerOptions(pCompilerOptions),
                                                                                              _spSelectedCodeGenerator(nullptr)
{
  _strInputFile   = "";
  _strOutputFile  = "";

  if (_pCompilerOptions == nullptr)
  {
    throw BackendException("Compiler options have not been set");
  }


  // Init all known common switches
  _InitSwitch<KnownSwitches::Help       >(CompilerSwitchTypeEnum::Help);
  _InitSwitch<KnownSwitches::OutputFile >(CompilerSwitchTypeEnum::OutputFile);
  _InitSwitch<KnownSwitches::Version    >(CompilerSwitchTypeEnum::Version);
  _InitSwitch<KnownSwitches::Verbose    >(CompilerSwitchTypeEnum::Verbose);
  _InitSwitch<KnownSwitches::IncludeDir >(CompilerSwitchTypeEnum::IncludeDir);
  _InitSwitch<KnownSwitches::Define     >(CompilerSwitchTypeEnum::Define);
  _InitSwitch<KnownSwitches::RTIncPath  >(CompilerSwitchTypeEnum::RTIncPath);
  _InitSwitch<KnownSwitches::TimeKernels>(CompilerSwitchTypeEnum::TimeKernels);


  // Init known backends
  _InitBackend<CPU_x86>();
  _InitBackend<CUDA>();
  _InitBackend<OpenCL_ACC>();
  _InitBackend<OpenCL_CPU>();
  _InitBackend<OpenCL_GPU>();
}

size_t BackendConfigurationManager::_HandleSwitch(std::string strSwitch, CommonDefines::ArgumentVectorType & rvecArguments, size_t szCurIndex)
{
  CompilerSwitchTypeEnum eSwitchType = _mapKnownSwitches[strSwitch].GetSwitchType();

  size_t szReturnIndex = szCurIndex;

  switch (eSwitchType)
  {
  case CompilerSwitchTypeEnum::Emit:
    {
      if (_spSelectedCodeGenerator)
      {
        throw RuntimeErrorException("Only one code generator can be selected for the compiler invocation!");
      }
      else
      {
        _spSelectedCodeGenerator = _mapCodeGenerators[strSwitch];
      }
    }
    break;
  case CompilerSwitchTypeEnum::OutputFile:
    {
      if (_strOutputFile != "")
      {
        throw RuntimeErrorException("Only one output file can be specified for the compiler invocation");
      }
      else if (szCurIndex >= rvecArguments.size()-2)
      {
        throw RuntimeErrors::MissingOptionException(strSwitch);
      }
      else
      {
        _strOutputFile = rvecArguments[szCurIndex + 1];
        ++szReturnIndex;
      }
    }
    break;
  case CompilerSwitchTypeEnum::Help:
    {
      // Format known common switches and print common usage
      CommonDefines::SwitchDisplayInfoVectorType vecSwitches;

      for (auto itSwitch = _mapKnownSwitches.begin(); itSwitch != _mapKnownSwitches.end(); itSwitch++)
      {
        vecSwitches.push_back(itSwitch->second.CreateDisplayInfo(itSwitch->first));
      }

      _ConsoleOutput.PrintUsage(vecSwitches);


      // Print the specific switches for all known code generators
      for (auto itCodeGenerator : _mapCodeGenerators)
      {
        _ConsoleOutput.PrintCodeGeneratorSwitches(itCodeGenerator.second);
      }

      throw RuntimeErrors::AbortException(EXIT_SUCCESS);
    }
  case CompilerSwitchTypeEnum::Version:
    {
      _ConsoleOutput.PrintVersion();

      throw RuntimeErrors::AbortException(EXIT_SUCCESS);
    }
  case CompilerSwitchTypeEnum::Verbose:
    {
      _pCompilerOptions->setPrintVerbose(USER_ON);
    }
    break;
  case CompilerSwitchTypeEnum::IncludeDir:
  case CompilerSwitchTypeEnum::Define:
    {
      std::string include = rvecArguments[szCurIndex];
      if (strlen(rvecArguments[szCurIndex].c_str()) == 2) {
        if (szCurIndex >= rvecArguments.size()-2) {
          throw RuntimeErrors::MissingOptionException(strSwitch);
        }
        include += rvecArguments[szCurIndex+1];
        ++szReturnIndex;
      }
      _vecClangArguments.push_back(include);
    }
    break;
  case CompilerSwitchTypeEnum::RTIncPath:
    {
      _pCompilerOptions->setRTIncPath(KnownSwitches::RTIncPath::OptionParser::Parse(rvecArguments[szCurIndex + 1]));
      ++szReturnIndex;
    }
    break;
  case CompilerSwitchTypeEnum::TimeKernels:
    {
      _pCompilerOptions->setTimeKernels(USER_ON);
    }
    break;
  default:  throw InternalErrors::UnhandledSwitchException(strSwitch);
  }

  return szReturnIndex;
}

std::string BackendConfigurationManager::_TranslateSwitchAlias(std::string strSwitch)
{
  // Check whether the current switch is a known alias
  auto itTranslatedSwitch = _mapSwitchAliases.find(strSwitch);

  if (itTranslatedSwitch != _mapSwitchAliases.end())
  {
    // Current switch is an aliase => Return the actual switch string
    return itTranslatedSwitch->second;
  }
  else
  {
    // No alias correspondence found => Return the current switch string
    return strSwitch;
  }
}


void BackendConfigurationManager::Configure(CommonDefines::ArgumentVectorType & rvecArguments)
{
  try
  {
    CommonDefines::ArgumentVectorType vecUnknownArguments;

    // Parse the command arguments vector
    for (size_t i = static_cast<size_t>(0); i < rvecArguments.size(); ++i)
    {
      if ((rvecArguments.size() > 1) && ((i + 1) == rvecArguments.size()))
      {
        // Last argument must be input file
        _strInputFile = rvecArguments[i];
      }
      else
      {
        // Try to parse the current switch and pass unknown switches to the code generator
        std::string strArgument = _TranslateSwitchAlias(rvecArguments[i]);

        if (strncmp(strArgument.c_str(), "-I", 2) == 0) {
          strArgument = "-I";
        } else if (strncmp(strArgument.c_str(), "-D", 2) == 0) {
          strArgument = "-D";
        }

        auto itSwitch = _mapKnownSwitches.find(strArgument);

        if (itSwitch != _mapKnownSwitches.end())
        {
          i = _HandleSwitch(strArgument, rvecArguments, i);
        }
        else
        {
          vecUnknownArguments.push_back(rvecArguments[i]);
        }
      }
    }

    // Check the common configuration
    if (_strInputFile == "")
    {
      throw RuntimeErrorException("No input file has been specified!");
    }
    else if (_strOutputFile == "")
    {
      throw RuntimeErrorException(std::string("No output file has been specified! Did you forget the \"") + KnownSwitches::OutputFile::Key() + std::string("\" switch?"));
    }

    // Configure the selected code generator
    if (_spSelectedCodeGenerator)
    {
      _spSelectedCodeGenerator->Configure(vecUnknownArguments);
    }
    else
    {
      throw RuntimeErrorException(std::string("No code generator has been selected! Did you forget the \"") + KnownSwitches::EmissionSwitchBase() + std::string("<X>\" switch?"));
    }

    // Set the selected code generator in the compiler options
    _pCompilerOptions->setCodeGenerator(_spSelectedCodeGenerator);
  }
  catch (RuntimeErrors::AbortException &e)
  {
    exit(e.GetExitCode());
  }
}

CommonDefines::ArgumentVectorType BackendConfigurationManager::GetClangArguments(std::string strBinPath)
{
  CommonDefines::ArgumentVectorType vecClangArguments;

  // Add Hipacc include paths
  vecClangArguments.push_back(std::string("-I") + strBinPath + std::string("/../include"));
  vecClangArguments.push_back(std::string("-I") + strBinPath + std::string("/../include/dsl"));
  vecClangArguments.push_back(std::string("-I") + strBinPath + std::string("/../include/c++/v1"));
  vecClangArguments.push_back(std::string("-I") + strBinPath + std::string("/../include/clang"));

  // Add additional clang arguments
  if (!_vecClangArguments.empty()) {
    vecClangArguments.insert( vecClangArguments.end(), _vecClangArguments.begin(), _vecClangArguments.end() );
  }

  // Add code generator specific additional arguments
  if (_spSelectedCodeGenerator)
  {
    CommonDefines::ArgumentVectorType vecCodeGenArgs = _spSelectedCodeGenerator->GetAdditionalClangArguments();

    vecClangArguments.insert( vecClangArguments.end(), vecCodeGenArgs.begin(), vecCodeGenArgs.end() );
  }


#ifdef _MSC_VER
  // Add Visual Studio system include paths
  vecClangArguments.push_back("-isystem");
  vecClangArguments.push_back(std::string(HOST_COMPILER_INSTALL_PREFIX) + std::string("VC/include"));

  // Set required Clang compiler options to ensure compatibility with the Visual Studio headers
  vecClangArguments.push_back("-fms-extensions");
  vecClangArguments.push_back("-fms-compatibility");
#else
  // Add Clang library include path (required for e.g. intrinsics)
  vecClangArguments.push_back(std::string("-I") + std::string(CLANG_LIB_INCLUDE_DIR));
#endif // _MSC_VER

  // Add code generator unrecognized arguments
  if (_spSelectedCodeGenerator)
  {
    CommonDefines::ArgumentVectorType vecUnknownArgs = _spSelectedCodeGenerator->GetUnknownArguments();
    vecClangArguments.insert( vecClangArguments.end(), vecUnknownArgs.begin(), vecUnknownArgs.end() );
  }

  // Add output file
  vecClangArguments.push_back("-o");
  vecClangArguments.push_back(_strOutputFile);

  // Add input file
  vecClangArguments.push_back(_strInputFile);

  return vecClangArguments;
}

std::string BackendConfigurationManager::GetOutputFile() {
  return _strOutputFile;
}

// vim: set ts=2 sw=2 sts=2 et ai:

