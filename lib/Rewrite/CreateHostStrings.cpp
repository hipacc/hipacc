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

//===--- CreateHostStrings.cpp - Runtime string creator for the Rewriter --===//
//
// This file implements functionality for printing Hipacc runtime code to
// strings.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Rewrite/CreateHostStrings.h"

using namespace clang;
using namespace hipacc;


void CreateHostStrings::writeHeaders(std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += "#include \"hipacc_cpu.hpp\"\n\n"; break;
    case Language::CUDA:
      resultStr += "#include \"hipacc_cu.hpp\"\n\n";  break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "#include \"hipacc_cl.hpp\"\n\n";  break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "#include \"hipacc_rs.hpp\"\n\n";  break;
  }
}


void CreateHostStrings::writeInitialization(std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      resultStr += "hipaccInitCUDA();\n";
      resultStr += indent;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccInitPlatformsAndDevices(";
      if (options.emitOpenCLACC()) {
        resultStr += "CL_DEVICE_TYPE_ACCELERATOR";
      } else if (options.emitOpenCLCPU()) {
        resultStr += "CL_DEVICE_TYPE_CPU";
      } else {
        resultStr += "CL_DEVICE_TYPE_GPU";
      }
      resultStr += ", ALL);\n";
      resultStr += indent + "hipaccCreateContextsAndCommandQueues();\n\n";
      resultStr += indent;
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "hipaccInitRenderScript(\"";
      resultStr += options.getRSDirectory();
      resultStr += "\");\n";
      resultStr += indent;
      break;
  }
}


void writeCLCompilation(std::string fileName, std::string kernel_name,
    std::string includes, std::string &resultStr, std::string suffix="") {
  resultStr += "cl_kernel " + kernel_name + suffix;
  resultStr += " = hipaccBuildProgramAndKernel(";
  resultStr += "\"" + fileName + ".cl\", ";
  resultStr += "\"" + kernel_name + suffix + "\", ";
  resultStr += "true, false, false, \"-I " + includes + "\");\n";
}


void CreateHostStrings::writeKernelCompilation(HipaccKernel *K,
    std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::C99:
    case Language::CUDA:
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "ScriptC_" + K->getFileName() + " " + K->getKernelName();
      resultStr += " = hipaccInitScript<ScriptC_" + K->getFileName() + ">();\n";
      resultStr += indent;
      if (K->getKernelClass()->getReduceFunction()) {
        resultStr += "ScriptC_" + K->getFileName() + " " + K->getReduceName();
        resultStr += " = hipaccInitScript<ScriptC_" + K->getFileName() + ">();\n";
        resultStr += indent;
      }
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      writeCLCompilation(K->getFileName(), K->getKernelName(),
          device.getCLIncludes(), resultStr);
      if (K->getKernelClass()->getReduceFunction()) {
        resultStr += indent;
        writeCLCompilation(K->getFileName(), K->getReduceName(),
            device.getCLIncludes(), resultStr, "2D");
        resultStr += indent;
        writeCLCompilation(K->getFileName(), K->getReduceName(),
            device.getCLIncludes(), resultStr, "1D");
      }
      break;
  }
}


void CreateHostStrings::writeReductionDeclaration(HipaccKernel *K, std::string
    &resultStr) {
  HipaccAccessor *Acc = K->getIterationSpace();
  HipaccImage *Img = Acc->getImage();

  auto add_red_arg = [&] (std::string device, std::string host,
                          std::string type) -> void {
    resultStr += "_args" + K->getReduceName();
    resultStr += ".push_back(hipacc_script_arg<ScriptC_" + K->getFileName();
    resultStr += ">(&ScriptC_" + K->getFileName();
    resultStr += "::set_" + device + ", " + "(" + type + ")&" + host + "));\n";
    resultStr += indent;
  };

  switch (options.getTargetLang()) {
    case Language::C99:
    case Language::CUDA:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "\n" + indent;
      resultStr += "std::vector<hipacc_script_arg<ScriptC_" + K->getFileName();
      resultStr += "> > _args" + K->getReduceName() + ";\n";
      resultStr += indent;

      // store reduction arguments
      add_red_arg("_red_Input", Img->getName() + ".mem", "sp<const Allocation> *");
      add_red_arg("_red_stride", Img->getName() + ".stride", "int *");

      // print optional offset_x/offset_y and iteration space width/height
      if (K->getIterationSpace()->isCrop()) {
        add_red_arg("_red_offset_x", Acc->getName() + ".offset_x", "int *");
        add_red_arg("_red_offset_y", Acc->getName() + ".offset_y", "int *");
        add_red_arg("_red_is_height", Acc->getName() + ".height", "int *");
        add_red_arg("_red_num_elements", Acc->getName() + ".width", "int *");
      } else {
        add_red_arg("_red_is_height", Img->getName() + ".height", "int *");
        add_red_arg("_red_num_elements", Img->getName() + ".width", "int *");
      }
      break;
  }
}


void CreateHostStrings::writeMemoryAllocation(HipaccImage *Img, std::string
    width, std::string height, std::string host, std::string &resultStr) {
  resultStr += "HipaccImage " + Img->getName() + " = ";
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += "hipaccCreateMemory<" + Img->getTypeStr() + ">(";
      break;
    case Language::CUDA:
      // texture is bound at kernel launch
      if (options.useTextureMemory() &&
          options.getTextureType()==Texture::Array2D) {
        resultStr += "hipaccCreateArray2D<" + Img->getTypeStr() + ">(";
      } else {
        resultStr += "hipaccCreateMemory<" + Img->getTypeStr() + ">(";
      }
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "hipaccCreateAllocation((" + Img->getTypeStr() + "*)";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      if (options.useTextureMemory()) {
        resultStr += "hipaccCreateImage<" + Img->getTypeStr() + ">(";
      } else {
        resultStr += "hipaccCreateBuffer<" + Img->getTypeStr() + ">(";
      }
      break;
  }
  resultStr += host + ", " + width + ", " + height;
  if (options.useTextureMemory() &&
      options.getTextureType()==Texture::Array2D) {
    // OpenCL Image objects and CUDA Arrays don't support padding
  } else {
    if (options.emitPadding()) {
      resultStr += ", " + std::to_string(device.alignment);
    }
  }
  resultStr += ");";
}


void CreateHostStrings::writeMemoryAllocationConstant(HipaccMask *Buf,
    std::string &resultStr) {
  resultStr += "HipaccImage " + Buf->getName() + " = ";
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += "hipaccCreateMemory<" + Buf->getTypeStr() + ">(";
      break;
    case Language::CUDA:
      assert(0 && "constant memory allocation not required in CUDA!");
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      resultStr += "hipaccCreateAllocation((" + Buf->getTypeStr() + "*)";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccCreateBufferConstant<" + Buf->getTypeStr() + ">(";
      break;
  }
  resultStr += "NULL, " + Buf->getSizeXStr() + ", " + Buf->getSizeYStr() + ");";
}


void CreateHostStrings::writeMemoryTransfer(HipaccImage *Img, std::string mem,
    MemoryTransferDirection direction, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      resultStr += "hipaccWriteMemory(";
      resultStr += Img->getName();
      resultStr += ", " + mem + ");";
      break;
    case DEVICE_TO_HOST:
      resultStr += "hipaccReadMemory<" + Img->getTypeStr() + ">(";
      resultStr += Img->getName() + ");";
      break;
    case DEVICE_TO_DEVICE:
      resultStr += "hipaccCopyMemory(";
      resultStr += mem + ", ";
      resultStr += Img->getName() + ");";
      break;
    case HOST_TO_HOST:
      assert(0 && "Unsupported memory transfer direction!");
      break;
  }
}


void CreateHostStrings::writeMemoryTransfer(
    HipaccPyramid *Pyr, std::string idx, std::string mem,
    MemoryTransferDirection direction, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      resultStr += "hipaccWriteMemory(";
      resultStr += Pyr->getName() + "(" + idx + ")";
      resultStr += ", " + mem + ");";
      break;
    case DEVICE_TO_HOST:
      resultStr += "hipaccReadMemory<" + Pyr->getTypeStr() + ">(";
      resultStr += Pyr->getName() + "(" + idx + "));";
      break;
    case DEVICE_TO_DEVICE:
      resultStr += "hipaccCopyMemory(";
      resultStr += mem + ", ";
      resultStr += Pyr->getName() + "(" + idx + "));";
      break;
    case HOST_TO_HOST:
      assert(0 && "Unsupported memory transfer direction!");
      break;
  }
}


void CreateHostStrings::writeMemoryTransferRegion(std::string dst, std::string
    src, std::string &resultStr) {
  resultStr += "hipaccCopyMemoryRegion(";
  resultStr += src + ", " + dst + ");";
}


void CreateHostStrings::writeMemoryTransferSymbol(HipaccMask *Mask, std::string
    mem, MemoryTransferDirection direction, std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::CUDA: {
        size_t i = 0;
        for (auto kernel : Mask->getKernels()) {
          if (i++) resultStr += "\n" + indent;

          switch (direction) {
            case HOST_TO_DEVICE:
              resultStr += "hipaccWriteSymbol<" + Mask->getTypeStr() + ">(";
              resultStr += "(const void *)&";
              resultStr += Mask->getName() + kernel->getName() + ", ";
              resultStr += "(" + Mask->getTypeStr() + " *)" + mem;
              resultStr += ", " + Mask->getSizeXStr() + ", " + Mask->getSizeYStr() + ");";
              break;
            case DEVICE_TO_HOST:
              resultStr += "hipaccReadSymbol<" + Mask->getTypeStr() + ">(";
              resultStr += "(" + Mask->getTypeStr() + " *)" + mem;
              resultStr += "(const void *)&";
              resultStr += Mask->getName() + kernel->getName() + ", ";
              resultStr += ", " + Mask->getSizeXStr() + ", " + Mask->getSizeYStr() + ");";
              break;
            case DEVICE_TO_DEVICE:
              resultStr += "writeMemoryTransferSymbol(todo, todo, DEVICE_TO_DEVICE);";
              break;
            case HOST_TO_HOST:
              resultStr += "writeMemoryTransferSymbol(todo, todo, HOST_TO_HOST);";
              break;
          }
        }
      }
      break;
    case Language::C99:
    case Language::Renderscript:
    case Language::Filterscript:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccWriteMemory(" + Mask->getName();
      resultStr += ", (" + Mask->getTypeStr() + " *)" + mem + ");";
      break;
  }
}


void CreateHostStrings::writeMemoryTransferDomainFromMask(
    HipaccMask *Domain, HipaccMask *Mask, std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::CUDA: {
        size_t i = 0;
        for (auto kernel : Mask->getKernels()) {
          if (i++) resultStr += "\n" + indent;
          resultStr += "hipaccWriteDomainFromMask<";
          resultStr += Mask->getTypeStr() + ">(";
          resultStr += "(const void *)&";
          resultStr += Domain->getName() + kernel->getName() + ", ";
          resultStr += "(" + Mask->getTypeStr() + " *)";
          resultStr += Mask->getHostMemName();
          resultStr += ", " + Mask->getSizeXStr();
          resultStr += ", " + Mask->getSizeYStr() + ");";
        }
      }
      break;
    case Language::C99:
    case Language::Renderscript:
    case Language::Filterscript:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccWriteDomainFromMask<" + Mask->getTypeStr() + ">(";
      resultStr += Domain->getName() + ", (" + Mask->getTypeStr() + "*)";
      resultStr += Mask->getHostMemName() + ");";
      break;
  }
}


void CreateHostStrings::writeMemoryRelease(HipaccMemory *Mem,
    std::string &resultStr, bool isPyramid) {
  // The same runtime call for all targets, just distinguish between Pyramids
  // and 'normal' memory like Images and Masks.
  if (isPyramid) {
    resultStr += "hipaccReleasePyramid(";
  } else {
    resultStr += "hipaccReleaseMemory(";
  }
  resultStr += Mem->getName() + ");\n";
  resultStr += indent;
}


void CreateHostStrings::writeKernelCall(HipaccKernel *K, std::string &resultStr) {
  auto argTypeNames = K->getArgTypeNames();
  auto deviceArgNames = K->getDeviceArgNames();
  auto hostArgNames = K->getHostArgNames();
  std::string kernel_name = K->getKernelName();

  std::string lit(std::to_string(literal_count++));
  std::string threads_x(std::to_string(K->getNumThreadsX()));
  std::string threads_y(std::to_string(K->getNumThreadsY()));
  std::string blockStr, gridStr, offsetStr, infoStr;

  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      blockStr = "block" + lit;
      gridStr = "grid" + lit;
      offsetStr = "offset" + lit;
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      blockStr = "work_size" + lit;
      gridStr = K->getIterationSpace()->getImage()->getName();
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      blockStr = "local_work_size" + lit;
      gridStr = "global_work_size" + lit;
      break;
  }
  infoStr = K->getInfoStr();

  if (options.exploreConfig() || options.timeKernels()) {
    inc_indent();
    resultStr += "{\n";
    switch (options.getTargetLang()) {
      case Language::C99: break;
      case Language::CUDA:
        if (options.exploreConfig()) {
          resultStr += indent + "std::vector<void *> _args" + kernel_name + ";\n";
          resultStr += indent + "std::vector<hipacc_const_info> _consts" + kernel_name + ";\n";
          resultStr += indent + "std::vector<hipacc_tex_info*> _texs" + kernel_name + ";\n";
        } else {
          resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernel_name + ";\n";
        }
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernel_name + ";\n";
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        resultStr += indent + "std::vector<hipacc_script_arg<ScriptC_" + K->getFileName() + "> >";
        resultStr += " _args" + kernel_name + ";\n";
    }
    resultStr += indent + "std::vector<hipacc_smem_info> _smems" + kernel_name + ";\n";
    resultStr += indent;
  }

  if (options.getTargetLang() != Language::C99) {
    // hipacc_launch_info
    resultStr += "hipacc_launch_info " + infoStr + "(";
    resultStr += std::to_string(K->getMaxSizeX()) + ", ";
    resultStr += std::to_string(K->getMaxSizeY()) + ", ";
    resultStr += K->getIterationSpace()->getName() + ", ";
    resultStr += std::to_string(K->getPixelsPerThread()) + ", ";
    if (K->vectorize()) {
      // TODO set and calculate per kernel simd width ...
      resultStr += "4);\n";
    } else {
      resultStr += "1);\n";
    }
    resultStr += indent;
  }

  if (!options.exploreConfig()) {
    switch (options.getTargetLang()) {
      case Language::C99: break;
      case Language::CUDA:
        // dim3 block
        resultStr += "dim3 " + blockStr + "(" + threads_x + ", " + threads_y + ");\n";
        resultStr += indent;

        // dim3 grid & hipaccCalcGridFromBlock
        resultStr += "dim3 " + gridStr + "(hipaccCalcGridFromBlock(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + "));\n\n";
        resultStr += indent;

        // hipaccPrepareKernelLaunch
        resultStr += "hipaccPrepareKernelLaunch(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + ");\n";
        resultStr += indent;

        // hipaccConfigureCall
        resultStr += "hipaccConfigureCall(";
        resultStr += gridStr;
        resultStr += ", " + blockStr;
        resultStr += ");\n\n";
        resultStr += indent;

        // offset parameter
        if (!options.timeKernels()) {
          resultStr += "size_t " + offsetStr + " = 0;\n";
          resultStr += indent;
        }
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        // size_t work_size
        resultStr += "size_t " + blockStr + "[2];\n";
        resultStr += indent + blockStr + "[0] = " + threads_x + ";\n";
        resultStr += indent + blockStr + "[1] = " + threads_y + ";\n";
        resultStr += indent;

        // hipaccPrepareKernelLaunch
        resultStr += "hipaccPrepareKernelLaunch(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + ");\n\n";
        resultStr += indent;
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        // size_t block
        resultStr += "size_t " + blockStr + "[2];\n";
        resultStr += indent + blockStr + "[0] = " + threads_x + ";\n";
        resultStr += indent + blockStr + "[1] = " + threads_y + ";\n";
        resultStr += indent;

        // size_t grid
        resultStr += "size_t " + gridStr + "[2];\n\n";
        resultStr += indent;

        // hipaccCalcGridFromBlock
        resultStr += "hipaccCalcGridFromBlock(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + ", ";
        resultStr += gridStr + ");\n";
        resultStr += indent;

        // hipaccPrepareKernelLaunch
        resultStr += "hipaccPrepareKernelLaunch(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + ");\n\n";
        resultStr += indent;
        break;
    }
  }


  // bind textures and get constant pointers
  size_t num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i])) continue;

    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    if (Acc) {
      if (options.emitCUDA() && K->useTextureMemory(Acc)!=Texture::None &&
          // no texture required for __ldg() intrinsic
          !(K->useTextureMemory(Acc) == Texture::Ldg)) {
        std::string type_str = "_tex", array_str = "Array2D";
        if (K->getKernelClass()->getMemAccess(arg)==WRITE_ONLY) {
          type_str = "_surf";
          array_str = "Surface";
        }
        // bind texture and surface
        if (options.exploreConfig()) {
          std::string lit(std::to_string(literal_count++));
          resultStr += "hipacc_tex_info tex_info" + lit;
          resultStr += "(std::string(\"" + type_str + deviceArgNames[i] + K->getName() + "\"), ";
          resultStr += K->getImgFromMapping(arg)->getImage()->getTextureType() + ", ";
          resultStr += hostArgNames[i] + ", ";
          switch (K->useTextureMemory(Acc)) {
            case Texture::Linear1D: resultStr += "Linear1D"; break;
            case Texture::Linear2D: resultStr += "Linear2D"; break;
            case Texture::Array2D:  resultStr += array_str;  break;
            default: assert(0 && "unsupported texture type!");
          }
          resultStr += ");\n";
          resultStr += indent;
          resultStr += "_texs" + kernel_name + ".push_back(";
          resultStr += "&tex_info" + lit + ");\n";
        } else {
          if (K->getKernelClass()->getMemAccess(arg)==WRITE_ONLY) {
            resultStr += "cudaGetSurfaceReference(&_surf";
          } else {
            resultStr += "cudaGetTextureReference(&_tex";
          }
          resultStr += deviceArgNames[i] + K->getName() + "Ref, &" + type_str;
          resultStr += deviceArgNames[i] + K->getName() + ");\n";
          resultStr += indent;
          if (K->getKernelClass()->getMemAccess(arg)==WRITE_ONLY) {
            resultStr += "hipaccBindSurface<" + argTypeNames[i] + ">(";
          } else {
            resultStr += "hipaccBindTexture<" + argTypeNames[i] + ">(";
          }
          switch (K->useTextureMemory(Acc)) {
            case Texture::Linear1D: resultStr += "Linear1D"; break;
            case Texture::Linear2D: resultStr += "Linear2D"; break;
            case Texture::Array2D:  resultStr += array_str;  break;
            default: assert(0 && "unsupported texture type!");
          }
          resultStr += ", " + type_str + deviceArgNames[i] + K->getName() + "Ref, ";
          resultStr += hostArgNames[i] + ");\n";
        }
        resultStr += indent;
      }

      if (options.exploreConfig() && K->useLocalMemory(Acc)) {
        // store local memory size information for exploration
        resultStr += "_smems" + kernel_name + ".push_back(";
        resultStr += "hipacc_smem_info(" + Acc->getSizeXStr() + ", ";
        resultStr += Acc->getSizeYStr() + ", ";
        resultStr += "sizeof(" + Acc->getImage()->getTypeStr() + ")));\n";
        resultStr += indent;
      }
    }
  }


  #if 0
  for (auto img : K->getKernelClass()->getImgFields()) {
    HipaccAccessor *Acc = K->getImgFromMapping(img);
    // emit assertion
    resultStr += "assert(" + Acc->getName() + ".width==" + K->getIterationSpace()->getName() + ".width && \"Acc width != IS width\");\n" + indent;
    resultStr += "assert(" + Acc->getName() + ".height==" + K->getIterationSpace()->getName() + ".height && \"Acc height != IS height\");\n" + indent;
  }
  #endif


  // parameters
  size_t cur_arg = 0;
  num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i])) continue;

    HipaccMask *Mask = K->getMaskFromMapping(arg);
    if (Mask) {
      if (options.emitCUDA()) {
        Mask->addKernel(K);
        continue;
      } else {
        if (Mask->isConstant()) continue;
      }
    }

    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    if (options.emitCUDA() && Acc && K->useTextureMemory(Acc)!=Texture::None &&
        // no texture required for __ldg() intrinsic
        !(K->useTextureMemory(Acc) == Texture::Ldg)) {
      // textures are handled separately
      continue;
    }
    std::string img_mem;
    if (Acc || Mask) img_mem = ".mem";

    if (options.exploreConfig() || options.timeKernels()) {
      // add kernel argument
      switch (options.getTargetLang()) {
        case Language::C99: break;
        case Language::CUDA:
          resultStr += "_args" + kernel_name + ".push_back(";
          if (options.exploreConfig()) {
            resultStr += "(void *)&" + hostArgNames[i] + img_mem + ");\n";
          } else {
            resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
            resultStr += "), (void *)&" + hostArgNames[i] + img_mem + "));\n";
          }
          resultStr += indent;
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          resultStr += "_args" + kernel_name + ".push_back(";
          resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
          resultStr += "), (void *)&" + hostArgNames[i] + img_mem + "));\n";
          resultStr += indent;
          break;
        case Language::Renderscript:
        case Language::Filterscript: {
            resultStr += "_args" + kernel_name + ".push_back(";
            resultStr += "hipacc_script_arg<ScriptC_" + K->getFileName() + ">(";
            resultStr += "&ScriptC_" + K->getFileName();
            resultStr += "::set_" + deviceArgNames[i] + ", ";
            if (Acc || Mask) {
              resultStr += "(sp<const Allocation> *)&" + hostArgNames[i] + img_mem + "));\n";
            } else {
              resultStr += "(" + argTypeNames[i] + "*)&" + hostArgNames[i] + "));\n";
            }
            resultStr += indent;
          }
          break;
      }
    } else {
      // set kernel arguments
      switch (options.getTargetLang()) {
        case Language::C99:
          if (i==0) {
            resultStr += "hipaccStartTiming();\n";
            resultStr += indent;
            resultStr += kernel_name + "(";
          } else {
            resultStr += ", ";
          }
          if (Acc) {
            resultStr += "(" + Acc->getImage()->getTypeStr() + "*)";
          }
          if (Mask) {
            resultStr += "(" + argTypeNames[i] + ")";
          }
          resultStr += hostArgNames[i] + img_mem;
          break;
        case Language::CUDA:
          resultStr += "hipaccSetupArgument(&";
          resultStr += hostArgNames[i] + img_mem;
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += offsetStr;
          resultStr += ");\n";
          resultStr += indent;
          break;
        case Language::OpenCLACC:
        case Language::OpenCLCPU:
        case Language::OpenCLGPU:
          resultStr += "hipaccSetKernelArg(";
          resultStr += kernel_name;
          resultStr += ", ";
          resultStr += std::to_string(cur_arg++);
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += "&" + hostArgNames[i] + img_mem;
          resultStr += ");\n";
          resultStr += indent;
          break;
        case Language::Renderscript:
        case Language::Filterscript:
          resultStr += "hipaccSetScriptArg(&" + kernel_name + ", ";
          resultStr += "&ScriptC_" + K->getFileName();
          resultStr += "::set_" + deviceArgNames[i] + ", ";
          if (Acc || Mask) {
            resultStr += "(sp<const Allocation>)(Allocation *)" + hostArgNames[i] + img_mem + ");\n";
          } else {
            resultStr += "(" + argTypeNames[i] + ")" + hostArgNames[i] + img_mem + ");\n";
          }
          resultStr += indent;
          break;
      }
    }
  }
  if (options.getTargetLang()==Language::C99) {
    // close parenthesis for function call
    resultStr += ");\n";
    resultStr += indent;
    resultStr += "hipaccStopTiming();\n";
    resultStr += indent;
  }
  resultStr += "\n" + indent;

  // launch kernel
  if (options.exploreConfig() || options.timeKernels()) {
    switch (options.getTargetLang()) {
      case Language::C99: break;
      case Language::CUDA:
        if (options.timeKernels()) {
          resultStr += "hipaccLaunchKernelBenchmark((const void *)&";
          resultStr += kernel_name + ", \"";
          resultStr += kernel_name + "\"";
        } else {
          resultStr += "hipaccKernelExploration(\"" + K->getFileName() + ".cu\", \"" + kernel_name + "\"";
        }
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        if (options.timeKernels()) {
          resultStr += "hipaccLaunchScriptKernelBenchmark(&" + kernel_name;
        } else {
          resultStr += "ScriptC_" + K->getFileName() + " " + kernel_name + " = ";
          resultStr += "hipaccInitScript<ScriptC_" + K->getFileName() + ">();\n";
          resultStr += indent + "hipaccLaunchScriptKernelExploration<";
          resultStr += "ScriptC_" + K->getFileName() + ", ";
          resultStr += K->getIterationSpace()->getImage()->getTypeStr();
          resultStr += ">(&" + kernel_name;
        }
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (options.timeKernels()) {
          resultStr += "hipaccEnqueueKernelBenchmark(" + kernel_name;
        } else {
          resultStr += "hipaccKernelExploration(\"" + K->getFileName() + ".cl\", \"" + kernel_name + "\"";
        }
        break;
    }
    resultStr += ", _args" + kernel_name;
    if (options.emitRenderscript() || options.emitFilterscript()) {
        resultStr += ", &ScriptC_" + K->getFileName() + "::forEach_" + kernel_name;
    }
    // additional parameters for exploration
    if (options.exploreConfig()) {
      resultStr += ", _smems" + kernel_name;
      if (options.emitCUDA()) {
        resultStr += ", _consts" + kernel_name;
        resultStr += ", _texs" + kernel_name;
      }
      resultStr += ", " + infoStr;
      resultStr += ", " + std::to_string(K->getWarpSize());
      resultStr += ", " + std::to_string(K->getMaxThreadsPerBlock());
      resultStr += ", " + std::to_string(K->getMaxThreadsForKernel());
      resultStr += ", " + std::to_string(K->getMaxTotalSharedMemory());
      resultStr += ", " + std::to_string(K->getNumThreadsX());
      resultStr += ", " + std::to_string(K->getNumThreadsY());
      if (options.emitCUDA()) {
        resultStr += ", " + std::to_string(device.getTargetCC());
      }
      if (options.emitRenderscript() || options.emitFilterscript()) {
        resultStr += ", " + gridStr;
      }
    } else {
      resultStr += ", " + gridStr;
      resultStr += ", " + blockStr;
      resultStr += ", true";
    }
    resultStr += ");\n";
    dec_indent();
    resultStr += indent + "}\n";
  } else {
    switch (options.getTargetLang()) {
      case Language::C99: break;
      case Language::CUDA:
        resultStr += "hipaccLaunchKernel((const void *)&";
        resultStr += kernel_name + ", \"";
        resultStr += kernel_name + "\"";
        break;
      case Language::Renderscript:
      case Language::Filterscript:
        resultStr += "hipaccLaunchScriptKernel(&" + kernel_name + ", ";
        resultStr += "&ScriptC_" + K->getFileName() + "::forEach_" + kernel_name;
        resultStr += ", " + gridStr;
        resultStr += ", " + blockStr + ");";
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        resultStr += "hipaccEnqueueKernel(";
        resultStr += kernel_name;
        break;
    }
    if (!options.emitRenderscript() && !options.emitFilterscript() &&
        !options.emitC99()) {
      resultStr += ", " + gridStr;
      resultStr += ", " + blockStr;
      resultStr += ");";
    }
  }
}


void CreateHostStrings::writeReduceCall(HipaccKernel *K, std::string &resultStr) {
  std::string typeStr(K->getIterationSpace()->getImage()->getTypeStr());
  std::string red_decl(typeStr + " " + K->getReduceStr() + " = ");

  // print runtime function name plus name of reduction function
  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      if (!options.exploreConfig()) {
        // first get texture reference
        resultStr += "cudaGetTextureReference(";
        resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + "Ref, ";
        resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + ");\n";
        resultStr += indent;
      }
      resultStr += red_decl;
      if (options.getTargetDevice() >= Device::Fermi_20 && !options.exploreConfig()) {
        resultStr += "hipaccApplyReductionThreadFence<" + typeStr + ">(";
        resultStr += "(const void *)&" + K->getReduceName() + "2D, ";
        resultStr += "\"" + K->getReduceName() + "2D\", ";
      } else {
        if (options.exploreConfig()) {
          resultStr += "hipaccApplyReductionExploration<" + typeStr + ">(";
          resultStr += "\"" + K->getFileName() + ".cu\", ";
          resultStr += "\"" + K->getReduceName() + "2D\", ";
          resultStr += "\"" + K->getReduceName() + "1D\", ";
        } else {
          resultStr += "hipaccApplyReduction<" + typeStr + ">(";
          resultStr += "(const void *)&" + K->getReduceName() + "2D, ";
          resultStr += "\"" + K->getReduceName() + "2D\", ";
          resultStr += "(const void *)&" + K->getReduceName() + "1D, ";
          resultStr += "\"" + K->getReduceName() + "1D\", ";
        }
      }
      break;
    case Language::Renderscript:
    case Language::Filterscript:
      if (options.exploreConfig()) {
          // no exploration supported atm since this involves lots of memory
          // reallocations in Renderscript
          resultStr += "ScriptC_" + K->getFileName() + " " + K->getReduceName() + " = ";
          resultStr += "hipaccInitScript<ScriptC_" + K->getFileName() + ">();\n";
      }
      resultStr += red_decl;
      resultStr += "hipaccApplyReduction<ScriptC_" + K->getFileName() + ", ";
      resultStr += typeStr + ">(&" + K->getReduceName() + ", ";
      resultStr += "&ScriptC_" + K->getFileName() + "::forEach_" +
        K->getReduceName() + "2D, ";
      resultStr += "&ScriptC_" + K->getFileName() + "::forEach_" +
        K->getReduceName() + "1D, ";
      resultStr += "&ScriptC_" + K->getFileName() + "::set__red_Output, ";
      resultStr += "_args" + K->getReduceName() + ", ";
      resultStr += K->getIterationSpace()->getName() + ".width);\n";
      return;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += red_decl;
      if (options.exploreConfig()) {
        resultStr += "hipaccApplyReductionExploration<" + typeStr + ">(";
        resultStr += "\"" + K->getFileName() + ".cl\", ";
        resultStr += "\"" + K->getReduceName() + "2D\", ";
        resultStr += "\"" + K->getReduceName() + "1D\", ";
      } else {
        resultStr += "hipaccApplyReduction<" + typeStr + ">(";
        resultStr += K->getReduceName() + "2D, ";
        resultStr += K->getReduceName() + "1D, ";
      }
      break;
  }

  // print image name
  resultStr += K->getIterationSpace()->getName() + ", ";

  // print pixels per thread
  resultStr += std::to_string(K->getNumThreadsReduce()) + ", ";
  resultStr += std::to_string(K->getPixelsPerThreadReduce());

  if (options.emitCUDA()) {
    // print 2D CUDA array texture information - this parameter is only used if
    // the texture type is Array2D
    if (options.exploreConfig()) {
      resultStr += ", hipacc_tex_info(std::string(\"_tex";
      resultStr += K->getIterationSpace()->getImage()->getName() + K->getName();
      resultStr += "\"), " + K->getIterationSpace()->getImage()->getTextureType() + ", ";
      resultStr += K->getIterationSpace()->getImage()->getName() + ", ";
      if (options.useTextureMemory() &&
          options.getTextureType()==Texture::Array2D) {
        resultStr += "Array2D";
      } else {
        resultStr += "Global";
      }
      resultStr += "), ";
    } else {
      resultStr += ", _tex";
      resultStr += K->getIterationSpace()->getImage()->getName() + K->getName();
      resultStr += "Ref";
    }

    if (options.exploreConfig()) {
      // print compute capability in case of configuration exploration
      resultStr += std::to_string(device.getTargetCC());
    }
  }
  resultStr += ");";
}


void CreateHostStrings::writeInterpolationDefinition(HipaccKernel *K,
    HipaccAccessor *Acc, std::string function_name, std::string type_suffix,
    Interpolate ip_mode, Boundary bh_mode, std::string &resultStr) {
  // interpolation macro
  switch (Acc->getInterpolationMode()) {
    case Interpolate::NO:
    case Interpolate::NN:
      resultStr += "DEFINE_BH_VARIANT_NO_BH(INTERPOLATE_LINEAR_FILTERING";
      break;
    case Interpolate::LF:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_LINEAR_FILTERING";
      break;
    case Interpolate::CF:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_CUBIC_FILTERING";
      break;
    case Interpolate::L3:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_LANCZOS_FILTERING";
      break;
  }
  switch (options.getTargetLang()) {
    case Language::C99:                                    break;
    case Language::CUDA:         resultStr += "_CUDA, ";   break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:    resultStr += "_OPENCL, "; break;
    case Language::Renderscript:
    case Language::Filterscript: resultStr += "_RS, ";     break;
  }
  // data type
  resultStr += Acc->getImage()->getTypeStr() + ", ";
  // append short data type - overloading is not supported in OpenCL
  if (options.emitOpenCL()) {
    resultStr += type_suffix + ", ";
  }
  // interpolation function
  resultStr += function_name;
  // boundary handling name + function (upper & lower)
  std::string const_parameter("NO_PARM");
  std::string const_suffix;
  switch (Acc->getBoundaryMode()) {
    case Boundary::UNDEFINED:
      resultStr += ", NO_BH, NO_BH, "; break;
    case Boundary::CLAMP:
      resultStr += "_clamp, BH_CLAMP_LOWER, BH_CLAMP_UPPER, "; break;
    case Boundary::REPEAT:
      resultStr += "_repeat, BH_REPEAT_LOWER, BH_REPEAT_UPPER, "; break;
    case Boundary::MIRROR:
      resultStr += "_mirror, BH_MIRROR_LOWER, BH_MIRROR_UPPER, "; break;
    case Boundary::CONSTANT:
      resultStr += "_constant, BH_CONSTANT_LOWER, BH_CONSTANT_UPPER, ";
      const_parameter = "CONST_PARM";
      const_suffix = "_CONST";
      break;
  }
  // image memory parameter, constant parameter, memory access function
  switch (K->useTextureMemory(Acc)) {
    case Texture::None:
      if (options.emitRenderscript() || options.emitFilterscript()) {
        resultStr += "ALL_PARM, " + const_parameter + ", ALL" + const_suffix;
      } else {
        resultStr += "IMG_PARM, " + const_parameter + ", IMG" + const_suffix;
      }
      break;
    case Texture::Linear1D:
      resultStr += "TEX_PARM, " + const_parameter + ", TEX" + const_suffix;
      break;
    case Texture::Linear2D:
    case Texture::Array2D:
      resultStr += "ARR_PARM, " + const_parameter + ", ARR" + const_suffix;
      break;
    case Texture::Ldg:
      resultStr += "LDG_PARM, " + const_parameter + ", LDG" + const_suffix;
      break;
  }
  // image read function for OpenCL
  if (options.emitOpenCL()) {
    resultStr += ", " + Acc->getImage()->getImageReadFunction();
  }
  resultStr += ")\n";
}


void CreateHostStrings::writePyramidAllocation(std::string pyrName, std::string
    type, std::string img, std::string depth, std::string &resultStr) {
  resultStr += "HipaccPyramid " + pyrName + " = ";
  resultStr += "hipaccCreatePyramid<" + type + ">(";
  resultStr += img + ", " + depth + ");";
}

// vim: set ts=2 sw=2 sts=2 et ai:

