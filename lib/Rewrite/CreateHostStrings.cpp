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
// This file implements functionality for printing HIPAcc runtime code to
// strings.
//
//===----------------------------------------------------------------------===//

#include "hipacc/Rewrite/CreateHostStrings.h"

using namespace clang;
using namespace hipacc;


void CreateHostStrings::writeHeaders(std::string &resultStr) {
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
      break;
    case TARGET_CUDA:
      resultStr += "#include \"hipacc_cuda.hpp\"\n\n";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "#include \"hipacc_ocl.hpp\"\n\n";
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "#include \"hipacc_rs.hpp\"\n\n";
      break;
  }
}


void CreateHostStrings::writeInitialization(std::string &resultStr) {
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
      break;
    case TARGET_CUDA:
      resultStr += "hipaccInitCUDA();\n";
      resultStr += indent;
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "hipaccInitPlatformsAndDevices(";
      if (options.emitOpenCLCPU()) {
        resultStr += "CL_DEVICE_TYPE_CPU";
      } else {
        resultStr += "CL_DEVICE_TYPE_GPU";
      }
      resultStr += ", ALL);\n";
      resultStr += indent + "hipaccCreateContextsAndCommandQueues();\n\n";
      resultStr += indent;
      break;
    case TARGET_Renderscript:
      resultStr += "hipaccInitRenderScript(16);\n";
      resultStr += indent;
      break;
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "hipaccInitRenderScript(17);\n";
      resultStr += indent;
      break;
  }
}


void CreateHostStrings::writeKernelCompilation(std::string kernelName,
    std::string &resultStr, std::string suffix) {
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "ScriptC_" + kernelName + " " + kernelName;
      resultStr += " = hipaccInitScript<ScriptC_" + kernelName + ">();\n";
      resultStr += indent;
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "cl_kernel " + kernelName + suffix;
      resultStr += " = hipaccBuildProgramAndKernel(\"";
      resultStr += kernelName;
      resultStr += ".cl\", \"cl";
      resultStr += kernelName + suffix;
      resultStr += "\", true, false, false, \"-I ";
      if (HipaccDevice(options).isARMGPU()) {
        resultStr += EMBEDDED_RUNTIME_INCLUDES;
      } else {
        resultStr += RUNTIME_INCLUDES;
      }
      resultStr += "\");\n";
      resultStr += indent;
      break;
  }
}


void CreateHostStrings::addReductionArgument(HipaccGlobalReduction *GR,
    std::string device_name, std::string host_name, std::string &resultStr, bool
    bind) {
  resultStr += "_args" + GR->getFileName();
  resultStr += ".push_back(hipacc_script_arg<ScriptC_" + GR->getFileName();
  resultStr += ">(&ScriptC_" + GR->getFileName();
  if (bind) {
    resultStr += "::bind_";
  } else {
    resultStr += "::set_";
  }
  resultStr += device_name + ", &" + host_name + "));\n";
  resultStr += indent;
}


void CreateHostStrings::writeReductionCompilation(HipaccGlobalReduction *GR,
    std::string &resultStr) {
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      writeKernelCompilation(GR->getFileName(), resultStr);
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      if (!options.exploreConfig()) {
        writeKernelCompilation(GR->getFileName(), resultStr, "2D");
        writeKernelCompilation(GR->getFileName(), resultStr, "1D");
      }
      break;
  }
}


void CreateHostStrings::writeReductionDeclaration(HipaccGlobalReduction *GR,
    std::string &resultStr) {
  HipaccImage *Img = GR->getAccessor()->getImage();
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "\n" + indent;
      resultStr += "std::vector<hipacc_script_arg<ScriptC_" + GR->getFileName();
      resultStr += "> > _args" + GR->getFileName() + ";\n";
      resultStr += indent;

      // store reduction arguments
      std::stringstream LSS;
      LSS << literalCountGridBock++;
      resultStr += "sp<Allocation> alloc_" + LSS.str() + " = (Allocation  *)";
      resultStr += Img->getName() + ".mem;\n" + indent;
      if (options.emitRenderscriptGPU() || options.emitFilterscript()) {
        addReductionArgument(GR, "Input", "alloc_" + LSS.str(), resultStr, false);
      } else {
        addReductionArgument(GR, "Input", "alloc_" + LSS.str(), resultStr, true);
      }
      addReductionArgument(GR, "neutral", GR->getNeutral(), resultStr, false);
      addReductionArgument(GR, "stride", Img->getName() + ".stride", resultStr, false);

      // print optional offset_x/offset_y and iteration space width/height
      if (GR->isAccessor()) {
        addReductionArgument(GR, "offset_x", GR->getAccessor()->getName() +
            ".offset_x", resultStr, false);
        addReductionArgument(GR, "offset_y", GR->getAccessor()->getName() +
            ".offset_y", resultStr, false);
        addReductionArgument(GR, "is_height", GR->getAccessor()->getName() +
            ".height", resultStr, false);
        addReductionArgument(GR, "num_elements", GR->getAccessor()->getName() +
            ".width", resultStr, false);
      } else {
        addReductionArgument(GR, "is_height", Img->getName() + ".height",
            resultStr, false);
        addReductionArgument(GR, "num_elements", Img->getName() + ".width",
            resultStr, false);
      }
      break;
  }
}


void CreateHostStrings::writeMemoryAllocation(std::string memName, std::string
    type, std::string width, std::string height, std::string &resultStr,
    HipaccDevice &targetDevice) {
  resultStr += "HipaccImage " + memName + " = ";
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      // texture is bound at kernel launch
      if (options.useTextureMemory() && options.getTextureType()==Array2D) {
        resultStr += "hipaccCreateArray2D<" + type + ">(";
      } else {
        resultStr += "hipaccCreateMemory<" + type + ">(";
      }
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "hipaccCreateAllocation((" + type + "*)";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      if (options.useTextureMemory()) {
        resultStr += "hipaccCreateImage<" + type + ">(";
      } else {
        resultStr += "hipaccCreateBuffer<" + type + ">(";
      }
      break;
  }
  resultStr += "NULL, " + width + ", " + height;
  if (options.useTextureMemory() && options.getTextureType()==Array2D) {
    // OpenCL Image objects and CUDA Arrays don't support padding
  } else {
    if (options.emitPadding()) {
      std::stringstream alignment;
      alignment << targetDevice.alignment;
      resultStr += ", " + alignment.str();
    }
  }
  resultStr += ");";
}


void CreateHostStrings::writeMemoryAllocationConstant(std::string memName,
    std::string type, std::string width, std::string height, std::string
    &resultStr) {

  resultStr += "HipaccImage " + memName + " = ";
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      assert(0 && "constant memory allocation not required in CUDA!");
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "hipaccCreateAllocation";
      if (options.emitRenderscript()) {
        resultStr += "Constant";
      }
      resultStr += "((" + type + "*)NULL";
      resultStr += ", (int)" + width + ", (int)" + height + ");";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "hipaccCreateBufferConstant<" + type + ">(";
      resultStr += width + ", " + height + ");";
      break;
  }
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
      resultStr += "hipaccReadMemory(";
      resultStr += mem;
      resultStr += ", " + Img->getName() + ");";
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
      resultStr += "hipaccReadMemory(";
      resultStr += mem;
      resultStr += ", " + Pyr->getName() + "(" + idx + "));";
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
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA: {
        SmallVector<HipaccKernel *, 16> kernels = Mask->getKernels();
        for (unsigned int i=0; i<kernels.size(); i++) {
          HipaccKernel *K = kernels[i];
          if (i) resultStr += "\n" + indent;

          switch (direction) {
            case HOST_TO_DEVICE:
              resultStr += "hipaccWriteSymbol<" + Mask->getTypeStr() + ">(";
              resultStr += "(const void *)&";
              resultStr += Mask->getName() + K->getName() + ", ";
              resultStr += "\"";
              resultStr += Mask->getName() + K->getName() + "\", ";
              resultStr += "(" + Mask->getTypeStr() + " *)" + mem;
              resultStr += ", " + Mask->getSizeXStr() + ", " + Mask->getSizeYStr() + ");";
              break;
            case DEVICE_TO_HOST:
              resultStr += "hipaccReadSymbol<" + Mask->getTypeStr() + ">(";
              resultStr += "(" + Mask->getTypeStr() + " *)" + mem;
              resultStr += "(const void *)&";
              resultStr += Mask->getName() + K->getName() + ", ";
              resultStr += "\"";
              resultStr += Mask->getName() + K->getName() + "\", ";
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
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "hipaccWriteMemory(" + Mask->getName();
      resultStr += ", (" + Mask->getTypeStr() + " *)" + mem + ");";
      break;
  }
}


void CreateHostStrings::writeKernelCall(std::string kernelName,
    HipaccKernelClass *KC, HipaccKernel *K, std::string &resultStr) {
  std::string *argTypeNames = K->getArgTypeNames();
  ArrayRef<std::string> deviceArgNames = K->getDeviceArgNames();
  std::string *hostArgNames = K->getHostArgNames();

  std::stringstream LSS;
  std::stringstream PPTSS;
  std::stringstream cX, cY;
  LSS << literalCountGridBock++;
  PPTSS << K->getPixelsPerThread();
  cX << K->getNumThreadsX();
  cY << K->getNumThreadsY();
  std::string blockStr, gridStr, offsetStr, infoStr;
  std::stringstream maxSizeXStr, maxSizeYStr;
  maxSizeXStr << K->getMaxSizeX();
  maxSizeYStr << K->getMaxSizeY();

  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      blockStr = "block" + LSS.str();
      gridStr = "grid" + LSS.str();
      offsetStr = "offset" + LSS.str();
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      blockStr = "work_size" + LSS.str();
      gridStr = K->getIterationSpace()->getAccessor()->getImage()->getName();
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      blockStr = "local_work_size" + LSS.str();
      gridStr = "global_work_size" + LSS.str();
      break;
  }
  infoStr = K->getInfoStr();

  if (options.exploreConfig() || options.timeKernels()) {
    inc_indent();
    resultStr += "{\n";
    switch (options.getTargetCode()) {
      default:
      case TARGET_C:
        break;
      case TARGET_CUDA:
        if (options.exploreConfig()) {
          resultStr += indent + "std::vector<void *> _args" + kernelName + ";\n";
        } else {
          resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
        }
        resultStr += indent + "std::vector<hipacc_const_info> _consts" + kernelName + ";\n";
        resultStr += indent + "std::vector<hipacc_tex_info> _texs" + kernelName + ";\n";
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLCPU:
        resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
        break;
      case TARGET_Renderscript:
      case TARGET_RenderscriptGPU:
      case TARGET_Filterscript:
        resultStr += indent + "std::vector<hipacc_script_arg<ScriptC_" + kernelName + "> >";
        resultStr += " _args" + kernelName + ";\n";
    }
    resultStr += indent + "std::vector<hipacc_smem_info> _smems" + kernelName + ";\n";
    resultStr += indent;
  }

  // hipacc_launch_info
  resultStr += "hipacc_launch_info " + infoStr + "(";
  resultStr += maxSizeXStr.str() + ", ";
  resultStr += maxSizeYStr.str() + ", ";
  resultStr += K->getIterationSpace()->getName() + ", ";
  resultStr += PPTSS.str() + ", ";
  if (K->vectorize()) {
    // TODO set and calculate per kernel simd width ...
    resultStr += "4);\n";
  } else {
    resultStr += "1);\n";
  }
  resultStr += indent;

  if (!options.exploreConfig()) {
    switch (options.getTargetCode()) {
      default:
      case TARGET_C:
      case TARGET_CUDA:
        // dim3 block
        resultStr += "dim3 " + blockStr + "(" + cX.str() + ", " + cY.str() + ");\n";
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
      case TARGET_Renderscript:
      case TARGET_RenderscriptGPU:
      case TARGET_Filterscript:
        // size_t work_size
        resultStr += "size_t " + blockStr + "[2];\n";
        resultStr += indent + blockStr + "[0] = " + cX.str() + ";\n";
        resultStr += indent + blockStr + "[1] = " + cY.str() + ";\n";
        resultStr += indent;

        // hipaccPrepareKernelLaunch
        resultStr += "hipaccPrepareKernelLaunch(";
        resultStr += infoStr + ", ";
        resultStr += blockStr + ");\n\n";
        resultStr += indent;
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLCPU:
        // size_t block
        resultStr += "size_t " + blockStr + "[2];\n";
        resultStr += indent + blockStr + "[0] = " + cX.str() + ";\n";
        resultStr += indent + blockStr + "[1] = " + cY.str() + ";\n";
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
  for (unsigned int i=0; i<K->getNumArgs(); i++) {
    FieldDecl *FD = K->getDeviceArgFields()[i];

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i])) continue;

    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    if (Acc) {
      if (options.emitCUDA() && K->useTextureMemory(Acc)) {
        if (KC->getImgAccess(FD)==READ_ONLY &&
            // no texture required for Kepler supporting __ldg() intrinsic
            !(K->useTextureMemory(Acc) != Array2D &&
              options.getTargetDevice() >= KEPLER_35)) {
          // bind texture
          if (options.exploreConfig()) {
            resultStr += "_texs" + kernelName + ".push_back(";
            resultStr += "hipacc_tex_info(std::string(\"_tex" + deviceArgNames[i] + K->getName() + "\"), ";
            resultStr += K->getImgFromMapping(FD)->getImage()->getTextureType() + ", ";
            resultStr += hostArgNames[i] + "));\n";
          } else {
            resultStr += "hipaccBindTexture<" + argTypeNames[i] + ">(_tex";
            resultStr += deviceArgNames[i] + K->getName() + ", ";
            resultStr += hostArgNames[i] + ");\n";
          }
          resultStr += indent;
        }
      }

      if (options.exploreConfig() && K->useLocalMemory(Acc)) {
        // store local memory size information for exploration
        resultStr += "_smems" + kernelName + ".push_back(";
        resultStr += "hipacc_smem_info(" + Acc->getSizeXStr() + ", ";
        resultStr += Acc->getSizeYStr() + ", ";
        resultStr += "sizeof(" + Acc->getImage()->getPixelType() + ")));\n";
        resultStr += indent;
      }
    }

    if (options.emitCUDA() && options.exploreConfig()) {
      HipaccMask *Mask = K->getMaskFromMapping(FD);
      if (Mask && !Mask->isConstant()) {
        // get constant pointer
        resultStr += "_consts" + kernelName + ".push_back(";
        resultStr += "hipacc_const_info(std::string(\"" + Mask->getName() + K->getName() + "\"), ";
        resultStr += "(void *)" + Mask->getHostMemName() + ", ";
        resultStr += "sizeof(" + Mask->getTypeStr() + ")*" + Mask->getSizeXStr() + "*" + Mask->getSizeYStr() + "));\n";
        resultStr += indent;
      }
    }
  }


  // check if Array2D memory is required for the iteration space
  if (options.emitCUDA() && options.useTextureMemory() &&
      options.getTextureType()==Array2D) {
    // bind surface
    if (options.exploreConfig()) {
      resultStr += "_texs" + kernelName + ".push_back(";
      resultStr += "hipacc_tex_info(std::string(\"_surfOutput" + K->getName() + "\"), ";
      resultStr += K->getIterationSpace()->getAccessor()->getImage()->getTextureType() + ", ";
      resultStr += K->getIterationSpace()->getAccessor()->getImage()->getName() + "));\n";
    } else {
      resultStr += "hipaccBindSurface<";
      resultStr += K->getIterationSpace()->getAccessor()->getImage()->getPixelType();
      resultStr += ">(_surfOutput" + K->getName() + ", ";
      resultStr += K->getIterationSpace()->getAccessor()->getImage()->getName() + ");\n";
    }
    resultStr += indent;
  }

  #if 0
  for (unsigned int i=0; i<KC->getNumImages(); i++) {
    HipaccAccessor *Acc = K->getImgFromMapping(KC->getImgFields().data()[i]);
    // emit assertion
    resultStr += "assert(" + Acc->getName() + ".width==" + K->getIterationSpace()->getName() + ".width && \"Acc width != IS width\");\n" + indent;
    resultStr += "assert(" + Acc->getName() + ".height==" + K->getIterationSpace()->getName() + ".height && \"Acc height != IS height\");\n" + indent;
  }
  #endif


  // parameters
  unsigned int curArg = 0;
  for (unsigned int i=0; i<K->getNumArgs(); i++) {
    FieldDecl *FD = K->getDeviceArgFields()[i];

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i])) continue;

    HipaccMask *Mask = K->getMaskFromMapping(FD);
    if (Mask) {
      if (options.emitCUDA()) {
        Mask->addKernel(K);
        continue;
      } else {
        if (Mask->isConstant()) continue;
      }
    }

    if (options.emitCUDA() && i==0 && options.useTextureMemory() &&
        options.getTextureType()==Array2D) {
      // surface is handled separately
      continue;
    }

    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    if (options.emitCUDA() && Acc && K->useTextureMemory(Acc) &&
        KC->getImgAccess(FD)==READ_ONLY &&
        // no texture required for Kepler supporting __ldg() intrinsic
        !(K->useTextureMemory(Acc) != Array2D &&
              options.getTargetDevice() >= KEPLER_35)) {
      // textures are handled separately
      continue;
    }
    std::string img_mem("");
    if (Acc || i==0) img_mem = ".mem";

    if (options.exploreConfig() || options.timeKernels()) {
      // add kernel argument
      switch (options.getTargetCode()) {
        default:
        case TARGET_C:
          break;
        case TARGET_CUDA:
          resultStr += "_args" + kernelName + ".push_back(";
          if (options.exploreConfig()) {
            resultStr += "(void *)&" + hostArgNames[i] + img_mem + ");\n";
          } else {
            resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
            resultStr += "), (void *)&" + hostArgNames[i] + img_mem + "));\n";
          }
          resultStr += indent;
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLCPU:
          resultStr += "_args" + kernelName + ".push_back(";
          resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
          resultStr += "), (void *)&" + hostArgNames[i] + img_mem + "));\n";
          resultStr += indent;
          break;
        case TARGET_Renderscript:
        case TARGET_RenderscriptGPU:
        case TARGET_Filterscript:
          if (Acc || Mask || i==0) {
            LSS.str("");
            LSS.clear();
            LSS << literalCountGridBock++;
            resultStr += "sp<Allocation> alloc_" + LSS.str() + " = (Allocation  *)";
            resultStr += hostArgNames[i] + img_mem + ";\n" + indent;
          }
          resultStr += "_args" + kernelName + ".push_back(";
          resultStr += "hipacc_script_arg<ScriptC_" + kernelName + ">(";
          resultStr += "&ScriptC_" + kernelName;
          if ((Acc || Mask || i==0) && options.emitRenderscript()) {
            resultStr += "::bind_";
          } else {
            resultStr += "::set_";
          }
          resultStr += deviceArgNames[i] + ", &";
          if (Acc || Mask || i==0) {
            resultStr += "alloc_" + LSS.str() + "));\n";
          } else {
            resultStr += hostArgNames[i] + "));\n";
          }
          resultStr += indent;
          break;
      }
    } else {
      // set kernel arguments
      switch (options.getTargetCode()) {
        default:
        case TARGET_C:
          break;
        case TARGET_CUDA:
          resultStr += "hipaccSetupArgument(&";
          resultStr += hostArgNames[i] + img_mem;
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += offsetStr;
          resultStr += ");\n";
          resultStr += indent;
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLCPU:
          LSS.str("");
          LSS.clear();
          LSS << curArg++;

          resultStr += "hipaccSetKernelArg(";
          resultStr += kernelName;
          resultStr += ", ";
          resultStr += LSS.str();
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += "&" + hostArgNames[i] + img_mem;
          resultStr += ");\n";
          resultStr += indent;
          break;
        case TARGET_Renderscript:
        case TARGET_RenderscriptGPU:
        case TARGET_Filterscript:
          resultStr += "hipaccSetScriptArg(&" + kernelName + ", ";
          resultStr += "&ScriptC_" + kernelName;
          if ((Acc || Mask || i==0) && options.emitRenderscript()) {
            resultStr += "::bind_";
          } else {
            resultStr += "::set_";
          }
          resultStr += deviceArgNames[i] + ", ";
          if (Acc || Mask || i==0) {
            resultStr += "sp<Allocation>(((Allocation *)" + hostArgNames[i];
            resultStr += img_mem + ")));\n";
          } else {
            resultStr += hostArgNames[i] + img_mem + ");\n";
          }
          resultStr += indent;
          break;
      }
    }
  }
  resultStr += "\n" + indent;

  // launch kernel
  if (options.exploreConfig() || options.timeKernels()) {
    std::stringstream max_threads_per_block, max_threads_for_kernel, warp_size, max_shared_memory_per_block;
    max_threads_per_block << K->getMaxThreadsPerBlock();
    max_threads_for_kernel << K->getMaxThreadsForKernel();
    warp_size << K->getWarpSize();
    max_shared_memory_per_block << K->getMaxTotalSharedMemory();

    switch (options.getTargetCode()) {
      default:
      case TARGET_C:
        break;
      case TARGET_CUDA:
        if (options.timeKernels()) {
          resultStr += "hipaccLaunchKernelBenchmark((const void *)&cu";
          resultStr += kernelName + ", \"cu";
          resultStr += kernelName + "\"";
        } else {
          resultStr += "hipaccKernelExploration(\"" + kernelName + ".cu\", \"cu" + kernelName + "\"";
        }
        break;
      case TARGET_Renderscript:
      case TARGET_RenderscriptGPU:
      case TARGET_Filterscript:
        if (options.timeKernels()) {
          resultStr += "hipaccLaunchScriptKernelBenchmark(&" + kernelName;
        } else {
          resultStr += "ScriptC_" + kernelName + " " + kernelName + " = ";
          resultStr += "hipaccInitScript<ScriptC_" + kernelName + ">();\n";
          resultStr += indent + "hipaccLaunchScriptKernelExploration<";
          resultStr += "ScriptC_" + kernelName + ", ";
          resultStr += K->getIterationSpace()->getImage()->getPixelType();
          resultStr += ">(&" + kernelName;
        }
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLCPU:
        if (options.timeKernels()) {
          resultStr += "hipaccEnqueueKernelBenchmark(" + kernelName;
        } else {
          resultStr += "hipaccKernelExploration(\"" + kernelName + ".cl\", \"cl" + kernelName + "\"";
        }
        break;
    }
    resultStr += ", _args" + kernelName;
    if (0 != (options.getTargetCode() & (TARGET_Renderscript |
                                         TARGET_RenderscriptGPU |
                                         TARGET_Filterscript))) {
        resultStr += ", &ScriptC_" + kernelName + "::forEach_rs" + kernelName;
    }
    // additional parameters for exploration
    if (options.exploreConfig()) {
      resultStr += ", _smems" + kernelName;
      if (options.emitCUDA()) {
        resultStr += ", _consts" + kernelName;
        resultStr += ", _texs" + kernelName;
      }
      resultStr += ", " + infoStr;
      resultStr += ", " + warp_size.str();
      resultStr += ", " + max_threads_per_block.str();
      resultStr += ", " + max_threads_for_kernel.str();
      resultStr += ", " + max_shared_memory_per_block.str();
      resultStr += ", " + cX.str();
      resultStr += ", " + cY.str();
      if (options.emitCUDA()) {
        std::stringstream cc_string;
        cc_string << options.getTargetDevice();
        resultStr += ", " + cc_string.str();
      }
      if (0 != (options.getTargetCode() & (TARGET_Renderscript |
                                           TARGET_RenderscriptGPU |
                                           TARGET_Filterscript))) {
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
    switch (options.getTargetCode()) {
      default:
      case TARGET_C:
        break;
      case TARGET_CUDA:
        resultStr += "hipaccLaunchKernel((const void *)&cu";
        resultStr += kernelName + ", \"cu";
        resultStr += kernelName + "\"";
        break;
      case TARGET_Renderscript:
      case TARGET_RenderscriptGPU:
      case TARGET_Filterscript:
        resultStr += "hipaccLaunchScriptKernel(&" + kernelName + ", ";
        resultStr += "&ScriptC_" + kernelName + "::forEach_rs" + kernelName;
        resultStr += ", " + gridStr;
        resultStr += ", " + blockStr + ");";
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLCPU:
        resultStr += "hipaccEnqueueKernel(";
        resultStr += kernelName;
        break;
    }
    if (0 == (options.getTargetCode() & (TARGET_Renderscript |
                                         TARGET_RenderscriptGPU |
                                         TARGET_Filterscript))) {
      resultStr += ", " + gridStr;
      resultStr += ", " + blockStr;
      resultStr += ");";
    }
  }
}


void CreateHostStrings::writeGlobalReductionCall(HipaccGlobalReduction *GR,
    std::string &resultStr) {
  std::stringstream GRSS;

  // print runtime function name plus name of reduction function
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
      break;
    case TARGET_CUDA:
      if (options.getTargetDevice() >= FERMI_20 && !options.exploreConfig()) {
        resultStr += "hipaccApplyReductionThreadFence<" + GR->getType() + ">(";
        resultStr += "(const void *)&cu" + GR->getFileName() + "2D, ";
        resultStr += "\"cu" + GR->getFileName() + "2D\", ";
      } else {
        if (options.exploreConfig()) {
          resultStr += "hipaccApplyReductionExploration<" + GR->getType() + ">(";
          resultStr += "\"" + GR->getFileName() + ".cu\", ";
          resultStr += "\"cu" + GR->getFileName() + "2D\", ";
          resultStr += "\"cu" + GR->getFileName() + "1D\", ";
        } else {
          resultStr += "hipaccApplyReduction<" + GR->getType() + ">(";
          resultStr += "(const void *)&cu" + GR->getFileName() + "2D, ";
          resultStr += "\"cu" + GR->getFileName() + "2D\", ";
          resultStr += "(const void *)&cu" + GR->getFileName() + "1D, ";
          resultStr += "\"cu" + GR->getFileName() + "1D\", ";
        }
      }
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      // no exploration supported atm since this involves lots of memory
      // reallocations in Renderscript
      resultStr += "hipaccApplyReduction<ScriptC_" + GR->getFileName() + ", ";
      resultStr += GR->getType() + ">(&" + GR->getFileName() + ", ";
      resultStr += "&ScriptC_" + GR->getFileName() + "::forEach_rs" +
        GR->getFileName() + "2D, ";
      resultStr += "&ScriptC_" + GR->getFileName() + "::forEach_rs" +
        GR->getFileName() + "1D, ";
      if (options.emitRenderscriptGPU() || options.emitFilterscript()) {
        resultStr += "&ScriptC_" + GR->getFileName() + "::set_Output, ";
      } else {
        resultStr += "&ScriptC_" + GR->getFileName() + "::bind_Output, ";
      }
      resultStr += "_args" + GR->getFileName() + ", ";
      if (GR->isAccessor()) {
        resultStr += GR->getAccessor()->getName() + ".width);\n";
      } else {
        resultStr += GR->getAccessor()->getImage()->getName() + ".width);\n";
      }
      return;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      if (options.exploreConfig()) {
        resultStr += "hipaccApplyReductionExploration<" + GR->getType() + ">(";
        resultStr += "\"" + GR->getFileName() + ".cl\", ";
        resultStr += "\"cl" + GR->getFileName() + "2D\", ";
        resultStr += "\"cl" + GR->getFileName() + "1D\", ";
      } else {
        resultStr += "hipaccApplyReduction<" + GR->getType() + ">(";
        resultStr += GR->getFileName() + "2D, ";
        resultStr += GR->getFileName() + "1D, ";
      }
      break;
  }

  // print image name
  if (GR->isAccessor()) {
    resultStr += GR->getAccessor()->getName() + ", ";
  } else if (GR->isPyramid()) {
    resultStr += GR->getAccessor()->getImage()->getName();
    resultStr += "(" + GR->getPyramidIndex() + "), ";
  } else {
    resultStr += GR->getAccessor()->getImage()->getName() + ", ";
  }
  // print neutral element
  resultStr += GR->getNeutral() + ", ";

  // print pixels per thread
  GRSS << GR->getNumThreads() << ", " << GR->getPixelsPerThread();
  resultStr += GRSS.str();

  if (options.emitCUDA()) {
    // print 2D CUDA array texture information - this parameter is only used if
    // the texture type is Array2D
    if (options.exploreConfig()) {
      resultStr += ", hipacc_tex_info(std::string(\"_tex";
      resultStr += GR->getAccessor()->getImage()->getName() + GR->getName();
      resultStr += "\"), ";
      resultStr += GR->getAccessor()->getImage()->getTextureType() + ", ";
      resultStr += GR->getAccessor()->getImage()->getName() + "), ";
    } else {
      resultStr += ", _tex";
      resultStr += GR->getAccessor()->getImage()->getName() + GR->getName();
    }

    if (options.exploreConfig()) {
      // print compute capability in case of configuration exploration
      std::stringstream cc_string;
      cc_string << options.getTargetDevice();
      resultStr += cc_string.str();
    }
  }
  resultStr += ");";
}


void CreateHostStrings::writeInterpolationDefinition(HipaccKernel *K,
    HipaccAccessor *Acc, std::string function_name, std::string type_suffix,
    InterpolationMode ip_mode, BoundaryMode bh_mode, std::string &resultStr) {
  // interpolation macro
  switch (ip_mode) {
    case InterpolateNO:
    case InterpolateNN:
      resultStr += "DEFINE_BH_VARIANT_NO_BH(INTERPOLATE_LINEAR_FILTERING";
      break;
    case InterpolateLF:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_LINEAR_FILTERING";
      break;
    case InterpolateCF:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_CUBIC_FILTERING";
      break;
    case InterpolateL3:
      resultStr += "DEFINE_BH_VARIANT(INTERPOLATE_LANCZOS_FILTERING";
      break;
  }
  switch (options.getTargetCode()) {
    case TARGET_C:
      break;
    case TARGET_Renderscript:
    case TARGET_RenderscriptGPU:
    case TARGET_Filterscript:
      resultStr += "_RS, "; break;
    case TARGET_CUDA:
      resultStr += "_CUDA, "; break;
    case TARGET_OpenCL:
    case TARGET_OpenCLCPU:
      resultStr += "_OPENCL, "; break;
  }
  // data type
  resultStr += Acc->getImage()->getPixelType() + ", ";
  // append short data type - overloading is not supported in OpenCL
  if (options.emitOpenCL()) {
    resultStr += type_suffix + ", ";
  }
  // interpolation function
  resultStr += function_name;
  // boundary handling name + function (upper & lower)
  std::string const_parameter = "NO_PARM";
  std::string const_suffix = "";
  switch (bh_mode) {
    case BOUNDARY_CLAMP:
      resultStr += "_clamp, BH_CONSTANT_LOWER, BH_CONSTANT_UPPER, "; break;
    case BOUNDARY_REPEAT:
      resultStr += "_repeat, BH_REPEAT_LOWER, BH_REPEAT_UPPER, "; break;
    case BOUNDARY_MIRROR:
      resultStr += "_mirror, BH_MIRROR_LOWER, BH_MIRROR_UPPER, "; break;
    case BOUNDARY_CONSTANT:
      resultStr += "_constant, BH_CONSTANT_LOWER, BH_CONSTANT_UPPER, ";
      const_parameter = "CONST_PARM";
      const_suffix = "_CONST";
      break;
    case BOUNDARY_UNDEFINED:
      resultStr += ", NO_BH, NO_BH, "; break;
  }
  // image memory parameter, constant parameter, memory access function
  switch (K->useTextureMemory(Acc)) {
    case NoTexture:
      if (options.emitRenderscriptGPU() || options.emitFilterscript()) {
        resultStr += "ALL_PARM, " + const_parameter + ", ALL" + const_suffix;
      } else {
        resultStr += "IMG_PARM, " + const_parameter + ", IMG" + const_suffix;
      }
      break;
    case Linear1D:
      resultStr += "TEX_PARM, " + const_parameter + ", TEX" + const_suffix;
      break;
    case Linear2D:
    case Array2D:
      resultStr += "ARR_PARM, " + const_parameter + ", ARR" + const_suffix;
      break;
  }
  // image read function for OpenCL
  if (options.emitOpenCL()) {
    resultStr += ", " + Acc->getImage()->getImageReadFunction();
  }
  resultStr += ")\n";
}


void CreateHostStrings::writePyramidAllocation(std::string pyrName, std::string
    type, std::string img, std::string depth, std::string &resultStr,
    HipaccDevice &targetDevice) {
  resultStr += "HipaccPyramid " + pyrName + " = ";
  resultStr += "hipaccCreatePyramid<" + type + ">(";
  resultStr += img + ", " + depth + ");";
}

// vim: set ts=2 sw=2 sts=2 et ai:

