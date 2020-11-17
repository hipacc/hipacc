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

//===--- AcceleratorDeviceBase.h - Base for accelerator device backends. -------------===//
//
// This file contains the base class for accelerator device backends (like CUDA, OpenCL etc.).
//
//===---------------------------------------------------------------------------------===//

#ifndef _BACKEND_ACCELERATOR_DEVICE_BASE_H_
#define _BACKEND_ACCELERATOR_DEVICE_BASE_H_

#include "hipacc/Device/TargetDevices.h"
#include "BackendExceptions.h"
#include "CommonDefines.h"
#include "OptionParsers.h"
#include <string>

namespace clang
{
namespace hipacc
{
namespace Backend
{
  /** \brief    A base class for all backends which represent an accelerator device.
   *  \details  Contains common definitions for accelerator devices like GPUs, OpenCL devices etc. */
  class AcceleratorDeviceBase
  {
  protected:

    /** \brief  Contains all known common compiler switches for accelerator device code generators. */
    class AcceleratorDeviceSwitches final
    {
    public:

      /** \brief  The switch type for the "image padding" switch. */
      struct EmitPadding final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-emit-padding"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<n>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Emit CUDA/OpenCL image padding, using alignment of <n> bytes for GPU devices"; }


        typedef CommonDefines::OptionParsers::Integer   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "pixels per thread" switch. */
      struct PixelsPerThread final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-pixels-per-thread"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<n>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Specify how many pixels should be calculated per thread"; }


        typedef CommonDefines::OptionParsers::Integer   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "target device selection" switch. */
      struct Target final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-target"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<n>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Generate code for GPUs with code name <n>.\n";
          strDescription += "Code names for CUDA/OpenCL on NVIDIA devices are:\n";
          strDescription += "  'Tesla-10', 'Tesla-11', 'Tesla-12', and 'Tesla-13' for Tesla architecture.\n";
          strDescription += "  'Fermi-20' and 'Fermi-21' for Fermi architecture.\n";
          strDescription += "  'Kepler-30' and 'Kepler-35' for Kepler architecture.\n";
          strDescription += "Code names for for OpenCL on AMD devices are:\n";
          strDescription += "  'Evergreen'      for Evergreen architecture (Radeon HD5xxx).\n";
          strDescription += "  'NorthernIsland' for Northern Island architecture (Radeon HD6xxx).\n";
          strDescription += "Code names for for OpenCL on ARM devices are:\n";
          strDescription += "  'Midgard' for Mali-T6xx' for Mali.\n";
          strDescription += "Code names for for OpenCL on Intel Xeon Phi devices are:\n";
          strDescription += "  'KnightsCorner' for Knights Corner Many Integrated Cores architecture.";

          return strDescription;
        }


        /** \brief  The option parser for this switch. */
        struct OptionParser final
        {
          typedef ::clang::hipacc::Device   ReturnType;   //!< The type of the parsed option.

          /** \brief  Converts the name of the selected target device into the internal ID.
           *  \param  strOption   The command line option as a string.
           *  \return If successful, the internal ID of the selected target device. */
          inline static ReturnType Parse(std::string strOption)
          {
            if (strOption == "Fermi-20")              return ::clang::hipacc::Device::Fermi_20;
            else if (strOption == "Fermi-21")         return ::clang::hipacc::Device::Fermi_21;
            else if (strOption == "Kepler-30")        return ::clang::hipacc::Device::Kepler_30;
            else if (strOption == "Kepler-32")        return ::clang::hipacc::Device::Kepler_32;
            else if (strOption == "Kepler-35")        return ::clang::hipacc::Device::Kepler_35;
            else if (strOption == "Kepler-37")        return ::clang::hipacc::Device::Kepler_37;
            else if (strOption == "Maxwell-50")       return ::clang::hipacc::Device::Maxwell_50;
            else if (strOption == "Maxwell-52")       return ::clang::hipacc::Device::Maxwell_52;
            else if (strOption == "Maxwell-53")       return ::clang::hipacc::Device::Maxwell_53;
            else if (strOption == "Evergreen")        return ::clang::hipacc::Device::Evergreen;
            else if (strOption == "NorthernIsland")   return ::clang::hipacc::Device::NorthernIsland;
            //else if (strOption == "SouthernIsland")   return ::clang::hipacc::Device::SouthernIsland;
            else if (strOption == "Midgard")          return ::clang::hipacc::Device::Midgard;
            else if (strOption == "KnightsCorner")    return ::clang::hipacc::Device::KnightsCorner;
            else
            {
              throw RuntimeErrors::InvalidOptionException(Key(), strOption);
            }
          }
        };
      };

      /** \brief  The switch type for the "reduce configuration selection" switch. */
      struct ReduceConfig final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-reduce-config"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<nxm>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description() {
          std::string strDescription("");

          strDescription += "Emit code that uses a multi-dimensional reduction configuration of\n";
          strDescription += "  'n' warps per block    (affects block size and shared memory size)\n";
          strDescription += "  'm' partial histograms (affects number of blocks)\n";

          return strDescription;
        }


        /** \brief  The option parser for this switch. */
        struct OptionParser final
        {
          typedef std::pair<int, int> ReturnType;   //!< The type of the parsed option.

          /** \brief  Converts the reduce configuration string <b>"NxM"</b> into a pair of integers.
           *  \param  strOption   The command line option as a string.
           *  \return If successful, the selected reduce configuration. */
          inline static ReturnType Parse(std::string strOption)
          {
            int x = 0, y = 0;
            if (sscanf(strOption.c_str(), "%dx%d", &x, &y) != 2)
            {
              throw RuntimeErrors::InvalidOptionException(Key(), strOption);
            }

            return ReturnType(x, y);
          }
        };
      };

      /** \brief  The switch type for the "kernel configuration selection" switch. */
      struct UseConfig final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-use-config"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<nxm>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Emit code that uses a configuration of nxm threads, e.g. 128x1"; }


        /** \brief  The option parser for this switch. */
        struct OptionParser final
        {
          typedef std::pair<int, int> ReturnType;   //!< The type of the parsed option.

          /** \brief  Converts the kernel configuration string <b>"NxM"</b> into a pair of integers.
           *  \param  strOption   The command line option as a string.
           *  \return If successful, the selected kernel configuration. */
          inline static ReturnType Parse(std::string strOption)
          {
            int x = 0, y = 0;
            if (sscanf(strOption.c_str(), "%dx%d", &x, &y) != 2)
            {
              throw RuntimeErrors::InvalidOptionException(Key(), strOption);
            }

            return ReturnType(x, y);
          }
        };
      };

      /** \brief  The switch type for the "use local/shared memory" switch. */
      struct UseLocal final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-use-local"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<o>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Enable/disable usage of shared/local memory in CUDA/OpenCL to stage image pixels to scratchpad\n";
          strDescription += "Valid values: 'on' and 'off'";

          return strDescription;
        }


        typedef CommonDefines::OptionParsers::OnOff   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "use texture memory" switch. */
      struct UseTextures final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-use-textures"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<o>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Enable/disable usage of textures (cached) in CUDA/OpenCL to read/write image pixels - for GPU devices only\n";
          strDescription += "Valid values for CUDA on NVIDIA devices: 'off', 'Linear1D', 'Linear2D', 'Array2D', and 'Ldg'\n";
          strDescription += "Valid values for OpenCL: 'off' and 'Array2D'";

          return strDescription;
        }


        /** \brief  The option parser for this switch. */
        struct OptionParser final
        {
          typedef ::clang::hipacc::Texture  ReturnType;   //!< The type of the parsed option.

          /** \brief  Parses the selected texture access method.
           *  \param  strOption   The command line option as a string.
           *  \return If successful, the internal ID of the selected texture access method. */
          inline static ReturnType Parse(std::string strOption)
          {
            if      (strOption == "off")        return ::clang::hipacc::Texture::None;
            else if (strOption == "Linear1D")   return ::clang::hipacc::Texture::Linear1D;
            else if (strOption == "Linear2D")   return ::clang::hipacc::Texture::Linear2D;
            else if (strOption == "Array2D")    return ::clang::hipacc::Texture::Array2D;
            else if (strOption == "Ldg")        return ::clang::hipacc::Texture::Ldg;
            else
            {
              throw RuntimeErrors::InvalidOptionException(Key(), strOption);
            }
          }
        };
      };

      /** \brief  The switch type for the "kernel vectorization" switch. */
      struct Vectorize final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-vectorize"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<o>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Enable/disable vectorization of generated CUDA/OpenCL code\n";
          strDescription += "Valid values: 'on' and 'off'";

          return strDescription;
        }


        typedef CommonDefines::OptionParsers::OnOff   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "nvcc include path" switch. */
      struct NvccPath final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-nvcc-path"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return ""; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Path for the NVCC compiler for CUDA"; }

        typedef CommonDefines::OptionParsers::Path   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "cross compiler include path (CUDA Windows)" switch. */
      struct CCBinPath final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-ccbin-path"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return ""; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Path host compiler binary directory (Windows only)"; }

        typedef CommonDefines::OptionParsers::Path   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "OpenCL compiler" switch. */
      struct ClCompilerPath final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-cl-compiler-path"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return ""; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()         { return "Path for the OpenCL compier for JIT compilation"; }

        typedef CommonDefines::OptionParsers::Path   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "kernel fusion" switch. */
      struct FuseKernel final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-fuse-kernel"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<o>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Enable/disable usage of kernel fusion\n";
          strDescription += "Valid values: 'on' and 'off'";
          return strDescription;
        }

        typedef CommonDefines::OptionParsers::OnOff   OptionParser;   //!< Type definition for the option parser for this switch.
      };

      /** \brief  The switch type for the "cuda graph" switch. */
      struct UseGraph final
      {
        /** \brief  Returns the command argument for this switch. */
        inline static std::string Key()                 { return "-use-graph"; }

        /** \brief  Returns the additional options string for this switch. */
        inline static std::string AdditionalOptions()   { return "<o>"; }

        /** \brief  Returns the description for this switch. */
        inline static std::string Description()
        {
          std::string strDescription("");

          strDescription += "Enable/disable usage of cuda graph\n";
          strDescription += "Valid values: 'on' and 'off'";
          return strDescription;
        }

        typedef CommonDefines::OptionParsers::OnOff   OptionParser;   //!< Type definition for the option parser for this switch.
      };
    };
  };
} // end namespace Backend
} // end namespace hipacc
} // end namespace clang


#endif  // _BACKEND_ACCELERATOR_DEVICE_BASE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

