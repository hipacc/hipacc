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

//===--- CreateHostStrings.cpp - OpenCL/CUDA helper for the Rewriter ------===//
//
// This file implements functionality for printing OpenCL/CUDA host code to
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
    case TARGET_OpenCLx86:
      resultStr += "#include \"hipacc_ocl.hpp\"\n\n";
      break;
    case TARGET_Renderscript:
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
    case TARGET_OpenCLx86:
      resultStr += "hipaccInitPlatformsAndDevices(";
      if (options.emitOpenCLx86()) {
        resultStr += "CL_DEVICE_TYPE_CPU";
      } else {
        resultStr += "CL_DEVICE_TYPE_GPU";
      }
      resultStr += ", ALL);\n";
      resultStr += indent + "hipaccCreateContextsAndCommandQueues();\n\n";
      resultStr += indent;
      break;
    case TARGET_Renderscript:
      // TODO: Handle API version
      resultStr += "hipaccInitRenderScript(16);\n";
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
      resultStr += "ScriptC_" + kernelName + " " + kernelName + suffix;
      resultStr += " = hipaccInitScript<ScriptC_" + kernelName + ">();";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLx86:
      resultStr += "cl_kernel " + kernelName + suffix;
      resultStr += " = hipaccBuildProgramAndKernel(\"";
      resultStr += kernelName;
      resultStr += ".cl\", \"cl";
      resultStr += kernelName + suffix;
      resultStr += "\", true, false, false, \"-I ";
      resultStr += RUNTIME_INCLUDES;
      resultStr += "\");\n";
      resultStr += indent;
      break;
  }
}


void CreateHostStrings::writeMemoryAllocation(std::string memName, std::string
    type, std::string width, std::string height, std::string &pitchStr,
    std::string &resultStr, HipaccDevice &targetDevice) {
  pitchStr = "_" + memName + "stride";
  resultStr += "int " + pitchStr + ";\n";
  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      // texture is bound at kernel launch
      if (options.useTextureMemory() && options.getTextureType()==Array2D) {
        resultStr += indent + "cudaArray *" + memName + " = ";
        resultStr += "hipaccCreateArray2D<" + type + ">(NULL, ";
      } else {
        resultStr += indent + type + " *" + memName + " = ";
        resultStr += "hipaccCreateMemory<" + type + ">(NULL, ";
      }
      break;
    case TARGET_Renderscript:
      resultStr += indent + "sp<Allocation> " + memName + " = ";
      resultStr += "hipaccCreateAllocation((" + type + "*)NULL, ";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLx86:
      resultStr += indent + "cl_mem " + memName + " = ";
      if (options.useTextureMemory()) {
        resultStr += "hipaccCreateImage<" + type + ">(NULL, ";
      } else {
        resultStr += "hipaccCreateBuffer<" + type + ">(NULL, ";
      }
      break;
  }
  resultStr += "(int) " + width;
  resultStr += ", (int) " + height;
  resultStr += ", &" + pitchStr;
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
    &pitchStr, std::string &resultStr) {

  switch (options.getTargetCode()) {
    default:
    case TARGET_C:
    case TARGET_CUDA:
      assert(0 && "constant memory allocation not required in CUDA!");
      break;
    case TARGET_Renderscript:
      resultStr += "sp<Allocation> " + memName;
      resultStr += " = hipaccCreateAllocationConstant((" + type + "*)NULL";
      resultStr += ", " + width;
      resultStr += ", " + height;
      resultStr += ");";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLx86:
      resultStr += "cl_mem " + memName + " = hipaccCreateBufferConstant<" + type + ">(";
      resultStr += width;
      resultStr += ", " + height;
      resultStr += ");";
      break;
  }
  if (options.emitCUDA()) {
  } else {
  }
}


void CreateHostStrings::writeMemoryTransfer(HipaccImage *Img, std::string mem,
    MemoryTransferDirection direction, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      switch (options.getTargetCode()) {
        default:
        case TARGET_C:
        case TARGET_CUDA:
          if (options.useTextureMemory() && options.getTextureType()==Array2D) {
            resultStr += "hipaccWriteArray2D(";
          } else {
            resultStr += "hipaccWriteMemory(";
          }
          break;
        case TARGET_Renderscript:
          resultStr += "hipaccWriteAllocation(";
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLx86:
          if (options.useTextureMemory() && options.getTextureType()==Array2D) {
            resultStr += "hipaccWriteImage(";
          } else {
            resultStr += "hipaccWriteBuffer(";
          }
          break;
      }
      resultStr += Img->getName();
      resultStr += ", " + mem + ");";
      break;
    case DEVICE_TO_HOST:
      switch (options.getTargetCode()) {
        default:
        case TARGET_C:
        case TARGET_CUDA:
          if (options.useTextureMemory() && options.getTextureType()==Array2D) {
            resultStr += "hipaccReadArray2D(";
          } else {
            resultStr += "hipaccReadMemory(";
          }
          break;
        case TARGET_Renderscript:
          resultStr += "hipaccReadAllocation(";
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLx86:
          if (options.useTextureMemory()) {
            resultStr += "hipaccReadImage(";
          } else {
            resultStr += "hipaccReadBuffer(";
          }
          break;
      }
      resultStr += mem;
      resultStr += ", " + Img->getName() + ");";
      break;
    case DEVICE_TO_DEVICE:
      resultStr += "writeMemoryTransfer(todo, todo, DEVICE_TO_DEVICE);";
      break;
    case HOST_TO_HOST:
      resultStr += "writeMemoryTransfer(todo, todo, HOST_TO_HOST);";
      break;
  }
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
      resultStr += "hipaccWriteAllocation(" + Mask->getName();
      resultStr += ", (" + Mask->getTypeStr() + " *)" + mem + ");";
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLx86:
      resultStr += "hipaccWriteBuffer(" + Mask->getName();
      resultStr += ", (" + Mask->getTypeStr() + " *)" + mem + ");";
      break;
  }
}


void CreateHostStrings::writeKernelCall(std::string kernelName,
    HipaccKernelClass *KC, HipaccKernel *K, std::string &resultStr) {
  std::string *argTypeNames = K->getArgTypeNames();
  std::string *deviceArgNames = K->getDeviceArgNames();
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
      blockStr = "work_size" + LSS.str();
      gridStr = "iter_space" + LSS.str();
      break;
    case TARGET_OpenCL:
    case TARGET_OpenCLx86:
      blockStr = "local_work_size" + LSS.str();
      gridStr = "global_work_size" + LSS.str();
      break;
  }
  infoStr = K->getInfoStr();

  if (options.exploreConfig() || options.timeKernels()) {
    inc_indent();
    resultStr += "{\n";
    if (options.emitCUDA()) {
      if (options.exploreConfig()) {
        resultStr += indent + "std::vector<void *> _args" + kernelName + ";\n";
      } else {
        resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
      }
      resultStr += indent + "std::vector<hipacc_const_info> _consts" + kernelName + ";\n";
      resultStr += indent + "std::vector<hipacc_tex_info> _texs" + kernelName + ";\n";
    } else {
      resultStr += indent + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
    }
    resultStr += indent + "std::vector<hipacc_smem_info> _smems" + kernelName + ";\n";
    resultStr += indent;
  }

  // hipacc_launch_info
  resultStr += "hipacc_launch_info " + infoStr + "(";
  resultStr += maxSizeXStr.str() + ", ";
  resultStr += maxSizeYStr.str() + ", ";
  resultStr += K->getIterationSpace()->getWidth() + ", ";
  resultStr += K->getIterationSpace()->getHeight() + ", ";
  if (K->getIterationSpace()->getOffsetX().empty()) {
    resultStr += "0, ";
  } else {
    resultStr += K->getIterationSpace()->getOffsetX() + ", ";
  }
  if (K->getIterationSpace()->getOffsetY().empty()) {
    resultStr += "0, ";
  } else {
    resultStr += K->getIterationSpace()->getOffsetY() + ", ";
  }
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
        // size_t work_size
        resultStr += "size_t " + blockStr + "[2];\n";
        resultStr += indent + blockStr + "[0] = " + cX.str() + ";\n";
        resultStr += indent + blockStr + "[1] = " + cY.str() + ";\n";
        resultStr += indent;

        // size_t iter_space
        resultStr += "sp<Allocation> " + gridStr + ";\n\n";
        resultStr += indent;

        // hipaccCalcIterSpaceFromBlock
        resultStr += "hipaccCalcIterSpaceFromBlock<";
        resultStr += K->getIterationSpace()->getImage()->getPixelType();
        resultStr += ">(";
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
      case TARGET_OpenCL:
      case TARGET_OpenCLx86:
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
    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    if (Acc) {
      if (options.emitCUDA() && K->useTextureMemory(Acc)) {
        if (KC->getImgAccess(FD)==READ_ONLY) {
          // bind texture
          if (options.exploreConfig()) {
            resultStr += "_texs" + kernelName + ".push_back(";
            resultStr += "hipacc_tex_info(std::string(\"_tex" + deviceArgNames[i] + K->getName() + "\"), ";
            resultStr += K->getImgFromMapping(FD)->getImage()->getTextureType() + ", ";
            resultStr += "(void *)" + hostArgNames[i] + ", ";
            switch (K->useTextureMemory(Acc)) {
              default:
              case Linear1D:
                resultStr += "Linear1D";
                break;
              case Linear2D:
                resultStr += "Linear2D";
                break;
              case Array2D:
                resultStr += "Array2D";
                break;
            }
            resultStr += "));\n";
          } else {
            switch (K->useTextureMemory(Acc)) {
              default:
              case Linear1D:
                resultStr += "hipaccBindTexture";
                break;
              case Linear2D:
                resultStr += "hipaccBindTexture2D";
                break;
              case Array2D:
                resultStr += "hipaccBindTextureToArray";
                break;
            }
            resultStr += "<" + argTypeNames[i] + ">(_tex" + deviceArgNames[i] + K->getName() + ", " + hostArgNames[i] + ");\n";
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
      resultStr += "(void *)" + K->getIterationSpace()->getAccessor()->getImage()->getName() + ", Surface));\n";
    } else {
      resultStr += "hipaccBindSurfaceToArray<";
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
    resultStr += "assert(" + Acc->getWidth() + "==" + K->getIterationSpace()->getWidth() + " && \"Acc width != IS width\");\n" + indent;
    resultStr += "assert(" + Acc->getHeight() + "==" + K->getIterationSpace()->getHeight() + " && \"Acc height != IS height\");\n" + indent;
  }
  #endif


  // parameters
  unsigned int curArg = 0;
  for (unsigned int i=0; i<K->getNumArgs(); i++) {
    FieldDecl *FD = K->getDeviceArgFields()[i];

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
        KC->getImgAccess(FD)==READ_ONLY) {
      // textures are handled separately
      continue;
    }

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i])) continue;

    if (options.exploreConfig() || options.timeKernels()) {
      // add kernel argument
      resultStr += "_args" + kernelName + ".push_back(";
      switch (options.getTargetCode()) {
        default:
        case TARGET_C:
          break;
        case TARGET_CUDA:
          if (options.exploreConfig()) {
            resultStr += "(void *)&" + hostArgNames[i] + ");\n";
          } else {
            resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
            resultStr += "), (void *)&" + hostArgNames[i] + "));\n";
          }
          resultStr += indent;
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLx86:
          resultStr += "std::make_pair(sizeof(" + argTypeNames[i];
          resultStr += "), (void *)" + hostArgNames[i] + "));\n";
          resultStr += indent;
          break;
        case TARGET_Renderscript:
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
          resultStr += hostArgNames[i];
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += offsetStr;
          resultStr += ");\n";
          resultStr += indent;
          break;
        case TARGET_OpenCL:
        case TARGET_OpenCLx86:
          LSS.str("");
          LSS.clear();
          LSS << curArg++;

          resultStr += "hipaccSetKernelArg(";
          resultStr += kernelName;
          resultStr += ", ";
          resultStr += LSS.str();
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += hostArgNames[i];
          resultStr += ");\n";
          resultStr += indent;
          break;
        case TARGET_Renderscript:
          resultStr += "hipaccSetScriptArg(&" + kernelName + ", ";
          resultStr += "&ScriptC_" + kernelName;
          if (Acc || Mask || i==0) {
            resultStr += "::bind_";
          } else {
            resultStr += "::set_";
          }
          resultStr += deviceArgNames[i] + ", ";
          resultStr += hostArgNames[i] + ");\n";
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

    if (options.emitCUDA()) {
      if (options.timeKernels()) {
        resultStr += "hipaccLaunchKernelBenchmark((const void *)&cu";
        resultStr += kernelName + ", \"cu";
        resultStr += kernelName + "\"";
      } else {
        resultStr += "hipaccKernelExploration(\"" + kernelName + ".cu\", \"cu" + kernelName + "\"";
      }
    } else {
      if (options.timeKernels()) {
        resultStr += "hipaccEnqueueKernelBenchmark(" + kernelName;
      } else {
        resultStr += "hipaccKernelExploration(\"" + kernelName + ".cl\", \"cl" + kernelName + "\"";
      }
    }
    resultStr += ", _args" + kernelName;
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
      case TARGET_CUDA:
        resultStr += "hipaccLaunchKernel((const void *)&cu";
        resultStr += kernelName + ", \"cu";
        resultStr += kernelName + "\"";
        break;
      case TARGET_Renderscript:
        resultStr += "hipaccLaunchScriptKernel(&" + kernelName + ", ";
        resultStr += "&ScriptC_" + kernelName + "::forEach_rs" + kernelName;
        resultStr += ", " + gridStr + ", " + blockStr + ");";
        break;
      case TARGET_OpenCL:
      case TARGET_OpenCLx86:
        resultStr += "hipaccEnqueueKernel(";
        resultStr += kernelName;
        break;
    }
    if (options.getTargetCode() != TARGET_Renderscript) {
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
  if (options.emitCUDA()) {
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
  } else {
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
  }

  // print image name
  resultStr += "(void *)" + GR->getAccessor()->getImage()->getName() + ", ";
  // print neutral element
  resultStr += GR->getNeutral() + ", ";
  // print width, height, and stride
  resultStr += GR->getAccessor()->getImage()->getWidth() + ", ";
  resultStr += GR->getAccessor()->getImage()->getHeight() + ", ";
  resultStr += GR->getAccessor()->getImage()->getStride() + ", ";

  // print optional offset_x/offset_y and iteration space width/height
  if (GR->isAccessor()) {
    resultStr += GR->getAccessor()->getOffsetX() + ", ";
    resultStr += GR->getAccessor()->getOffsetY() + ", ";
    resultStr += GR->getAccessor()->getWidth() + ", ";
    resultStr += GR->getAccessor()->getHeight() + ", ";
  }

  // print pixels per thread
  GRSS << GR->getNumThreads() << ", " << GR->getPixelsPerThread();
  resultStr += GRSS.str();

  if (options.emitCUDA()) {
    if (options.exploreConfig()) {
      // print 2D CUDA array texture information - this parameter is only used
      // if the texture type is Array2D
      resultStr += ", hipacc_tex_info(std::string(\"_tex" + GR->getAccessor()->getImage()->getName() + GR->getName() + "\"), ";
      resultStr += GR->getAccessor()->getImage()->getTextureType() + ", ";
      resultStr += "(void *)" + GR->getAccessor()->getImage()->getName() + ", ";
      if (options.emitCUDA() && options.useTextureMemory() &&
          options.getTextureType()==Array2D) {
        resultStr += "Array2D";
      } else {
          resultStr += "NoTexture";
      }
      resultStr += "), ";

      // print compute capability in case of configuration exploration
      std::stringstream cc_string;
      cc_string << options.getTargetDevice();
      resultStr += cc_string.str();
    } else {
      // print 2D CUDA array name - this parameter is only used if the next
      // parameter is Array2D
      resultStr += ", _tex" + GR->getAccessor()->getImage()->getName() + GR->getName() + ", ";
      // print what type of input image we have - Array2D or NoTexture
      if (options.emitCUDA() && options.useTextureMemory() &&
          options.getTextureType()==Array2D) {
        resultStr += "Array2D";
      } else {
        resultStr += "NoTexture";
      }
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
  if (options.emitCUDA()) {
    resultStr += "_CUDA, ";
  } else {
    resultStr += "_OPENCL, ";
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
      resultStr += "IMG_PARM, " + const_parameter
                + ", IMG" + const_suffix;
      break;
    case Linear1D:
      resultStr += "TEX_PARM, " + const_parameter
                + ", TEX" + const_suffix;
      break;
    case Linear2D:
    case Array2D:
      resultStr += "ARR_PARM, " + const_parameter
                + ", ARR" + const_suffix;
      break;
  }
  // image read function for OpenCL
  if (options.emitOpenCL()) {
    resultStr += ", " + Acc->getImage()->getImageReadFunction();
  }
  resultStr += ")\n";
}

// vim: set ts=2 sw=2 sts=2 et ai:

