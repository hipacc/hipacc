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
  if (options.emitCUDA()) {
    resultStr += "#include \"hipacc_cuda.hpp\"\n\n";
  } else {
    resultStr += "#include \"hipacc_ocl.hpp\"\n\n";
  }
}


void CreateHostStrings::writeInitialization(std::string &resultStr) {
  if (options.emitCUDA()) {
    resultStr += "hipaccInitCUDA();\n";
  } else {
    std::string clDevice;
    if (options.emitOpenCLx86()) {
      clDevice = "CL_DEVICE_TYPE_CPU";
    } else {
      clDevice = "CL_DEVICE_TYPE_GPU";
    }
    resultStr += "hipaccInitPlatformsAndDevices(" + clDevice + ", ALL);\n";
    resultStr += ident + "hipaccCreateContextsAndCommandQueues();\n\n";
  }
  resultStr += ident;
}


void CreateHostStrings::writeKernelCompilation(std::string kernelName,
    std::string &resultStr, std::string suffix) {
  if (!options.emitCUDA()) {
    resultStr += "cl_kernel " + kernelName + suffix;
    resultStr += " = hipaccBuildProgramAndKernel(\"";
    resultStr += kernelName;
    resultStr += ".cl\", \"cl";
    resultStr += kernelName + suffix;
    resultStr += "\", true, false, false, \"-I ";
    resultStr += RUNTIME_INCLUDES;
    resultStr += "\");\n";
    resultStr += ident;
  }
}


void CreateHostStrings::writeMemoryAllocation(std::string memName, std::string
    type, std::string width, std::string height, std::string &pitchStr,
    std::string &resultStr, HipaccDevice &targetDevice) {
  pitchStr = "_" + memName + "stride";
  resultStr += "int " + pitchStr + ";\n";
  if (options.emitCUDA()) {
    // texture is bound at kernel launch
    if (options.useTextureMemory(USER_ON) &&
        options.getTextureType()==Array2D) {
      resultStr += ident + "cudaArray *" + memName + " = ";
      resultStr += "hipaccCreateArray2D<" + type + ">(NULL, ";
    } else {
      resultStr += ident + type + " *" + memName + " = ";
      resultStr += "hipaccCreateMemory<" + type + ">(NULL, ";
    }
  } else {
    resultStr += ident + "cl_mem " + memName + " = ";
    if (options.useTextureMemory(USER_ON)) {
      resultStr += "hipaccCreateImage<" + type + ">(NULL, ";
    } else {
      resultStr += "hipaccCreateBuffer<" + type + ">(NULL, ";
    }
  }
  resultStr += "(int) " + width;
  resultStr += ", (int) " + height;
  resultStr += ", &" + pitchStr;
  if (options.useTextureMemory(USER_ON) && options.getTextureType()==Array2D) {
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

  if (options.emitCUDA()) {
    assert(0 && "constant memory allocation not required in CUDA!");
  } else {
    resultStr += "cl_mem " + memName + " = hipaccCreateBufferConstant<" + type + ">(";
    resultStr += width;
    resultStr += ", " + height;
    resultStr += ");";
  }
}


void CreateHostStrings::writeMemoryTransfer(HipaccImage *Img, std::string mem,
    MemoryTransferDirection direction, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      if (options.emitCUDA()) {
        if (options.useTextureMemory(USER_ON) &&
            options.getTextureType()==Array2D) {
          resultStr += "hipaccWriteArray2D(";
        } else {
          resultStr += "hipaccWriteMemory(";
        }
      } else {
        if (options.useTextureMemory(USER_ON) &&
            options.getTextureType()==Array2D) {
          resultStr += "hipaccWriteImage(";
        } else {
          resultStr += "hipaccWriteBuffer(";
        }
      }
      resultStr += Img->getName();
      resultStr += ", " + mem + ");";
      break;
    case DEVICE_TO_HOST:
      if (options.emitCUDA()) {
        if (options.useTextureMemory(USER_ON) &&
            options.getTextureType()==Array2D) {
          resultStr += "hipaccReadArray2D(";
        } else {
          resultStr += "hipaccReadMemory(";
        }
      } else {
        if (options.useTextureMemory(USER_ON)) {
          resultStr += "hipaccReadImage(";
        } else {
          resultStr += "hipaccReadBuffer(";
        }
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
  if (options.emitCUDA()) {
    llvm::SmallVector<HipaccKernel *, 16> kernels = Mask->getKernels();
    for (unsigned int i=0; i<kernels.size(); i++) {
      HipaccKernel *K = kernels[i];
      if (i) resultStr += "\n" + ident;

      switch (direction) {
        case HOST_TO_DEVICE:
          resultStr += "hipaccWriteSymbol<" + Mask->getTypeStr() + ">(\"";
          resultStr += Mask->getName() + K->getName() + "\", " + mem;
          resultStr += ", " + Mask->getSizeXStr() + ", " + Mask->getSizeYStr() + ");";
          break;
        case DEVICE_TO_HOST:
          resultStr += "hipaccReadSymbol<" + Mask->getTypeStr() + ">(";
          resultStr += mem + ", \"" + Mask->getName();
          resultStr += "\", " + Mask->getSizeXStr() + ", " + Mask->getSizeYStr() + ");";
          break;
        case DEVICE_TO_DEVICE:
          resultStr += "writeMemoryTransferSymbol(todo, todo, DEVICE_TO_DEVICE);";
          break;
        case HOST_TO_HOST:
          resultStr += "writeMemoryTransferSymbol(todo, todo, HOST_TO_HOST);";
          break;
      }
    }
  } else {
    resultStr += "hipaccWriteBuffer(";
    resultStr += Mask->getName();
    resultStr += ", (" + Mask->getTypeStr();
    resultStr += " *)" + mem + ");";
  }
}


void CreateHostStrings::setupKernelArgument(std::string kernelName, int curArg,
    std::string argName, std::string argTypeName, std::string offsetStr,
    std::string &resultStr) {
  if (options.emitCUDA()) {
    resultStr += "hipaccSetupArgument(&";
    resultStr += argName;
    resultStr += ", sizeof(";
    resultStr += argTypeName;
    resultStr += "), ";
    resultStr += offsetStr;
    resultStr += ");\n";
  } else {
    std::string S;
    std::stringstream SS;
    SS << curArg;

    resultStr += "hipaccSetKernelArg(";
    resultStr += kernelName;
    resultStr += ", ";
    resultStr += SS.str();
    resultStr += ", sizeof(";
    resultStr += argTypeName;
    resultStr += "), ";
    resultStr += argName + ");\n";
  }
  resultStr += ident;
}


void CreateHostStrings::writeKernelCall(std::string kernelName, std::string
    *argTypeNames, std::string *argNames, HipaccKernelClass *KC, HipaccKernel
    *K, std::string &resultStr) {
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

  if (options.emitCUDA()) {
    blockStr = "block" + LSS.str();
    gridStr = "grid" + LSS.str();
    offsetStr = "offset" + LSS.str();
  } else {
    blockStr = "local_work_size" + LSS.str();
    gridStr = "global_work_size" + LSS.str();
  }
  infoStr = K->getName() + "_info";

  if (options.exploreConfig() || options.timeKernels()) {
    inc_ident();
    resultStr += "{\n";
    if (options.emitCUDA()) {
      if (options.exploreConfig()) {
        resultStr += ident + "std::vector<void *> _args" + kernelName + ";\n";
      } else {
        resultStr += ident + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
      }
      resultStr += ident + "std::vector<hipacc_const_info> _consts" + kernelName + ";\n";
      resultStr += ident + "std::vector<hipacc_tex_info> _texs" + kernelName + ";\n";
    } else {
      resultStr += ident + "std::vector<std::pair<size_t, void *> > _args" + kernelName + ";\n";
    }
    resultStr += ident + "std::vector<hipacc_smem_info> _smems" + kernelName + ";\n";
    resultStr += ident;
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
  resultStr += ident;

  if (!options.exploreConfig()) {
    if (options.emitCUDA()) {
      // dim3 block
      resultStr += "dim3 " + blockStr + "(" + cX.str() + ", " + cY.str() + ");\n";
      resultStr += ident;
      
      // dim3 grid & hipaccCalcGridFromBlock
      resultStr += "dim3 " + gridStr + "(hipaccCalcGridFromBlock(";
      resultStr += infoStr + ", ";
      resultStr += blockStr + "));\n\n";
      resultStr += ident;

      // offset
      resultStr += "size_t " + offsetStr + " = 0;\n";
      resultStr += ident;
      
      // hipaccPrepareKernelLaunch
      resultStr += "hipaccPrepareKernelLaunch(";
      resultStr += infoStr + ", ";
      resultStr += blockStr + ");\n";
      resultStr += ident;

      // hipaccConfigureCall
      resultStr += "hipaccConfigureCall(";
      resultStr += gridStr;
      resultStr += ", " + blockStr;
      resultStr += ");\n\n";
    } else {
      // size_t block
      resultStr += "size_t " + blockStr + "[2];\n";
      resultStr += ident + blockStr + "[0] = " + cX.str() + ";\n";
      resultStr += ident + blockStr + "[1] = " + cY.str() + ";\n";
      resultStr += ident;

      // size_t grid
      resultStr += "size_t " + gridStr + "[2];\n";
      resultStr += ident;

      // hipaccCalcGridFromBlock
      resultStr += "hipaccCalcGridFromBlock(";
      resultStr += infoStr + ", ";
      resultStr += blockStr + ", ";
      resultStr += gridStr + ");\n\n";
      resultStr += ident;

      // hipaccPrepareKernelLaunch
      resultStr += "hipaccPrepareKernelLaunch(";
      resultStr += infoStr + ", ";
      resultStr += blockStr + ");\n\n";
    }
    resultStr += ident;
  }

  unsigned int numArgs = K->getNumArgs();
  // bind textures and get constant pointers
  for (unsigned int i=0; i<numArgs; i++) {
    FieldDecl *FD = K->getArgFields()[i];
    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    if (Acc) {
      if (options.emitCUDA() && K->useTextureMemory(Acc)) {
        if (KC->getImgAccess(FD)==READ_ONLY) {
          // bind texture
          if (options.exploreConfig()) {
            resultStr += "_texs" + kernelName + ".push_back(";
            resultStr += "hipacc_tex_info(std::string(\"_tex" + K->getArgNames()[i].str() + K->getName() + "\"), ";
            resultStr += K->getImgFromMapping(FD)->getImage()->getTextureType() + ", ";
            resultStr += "(void *)" + argNames[i] + ", ";
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
            resultStr += "<" + argTypeNames[i] + ">(_tex" + K->getArgNames()[i].str() + K->getName() + ", " + argNames[i] + ");\n";
          }
          resultStr += ident;
        }
      }

      if (options.exploreConfig() && K->useLocalMemory(Acc)) {
        // store local memory size information for exploration
        resultStr += "_smems" + kernelName + ".push_back(";
        resultStr += "hipacc_smem_info(" + Acc->getSizeXStr() + ", ";
        resultStr += Acc->getSizeYStr() + ", ";
        resultStr += "sizeof(" + Acc->getImage()->getPixelType() + ")));\n";
        resultStr += ident;
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
        resultStr += ident;
      }
    }
  }

  if (options.emitCUDA() && options.useTextureMemory(USER_ON) &&
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
    resultStr += ident;
  }

  #if 0
  for (unsigned int i=0; i<KC->getNumImages(); i++) {
    HipaccAccessor *Acc = K->getImgFromMapping(KC->getImgFields().data()[i]);
    // emit assertion
    resultStr += "assert(" + Acc->getWidth() + "==" + K->getIterationSpace()->getWidth() + " && \"Acc width != IS width\");\n" + ident;
    resultStr += "assert(" + Acc->getHeight() + "==" + K->getIterationSpace()->getHeight() + " && \"Acc height != IS height\");\n" + ident;
  }
  #endif

  // parameters
  unsigned int curArg = 0;
  for (unsigned int i=0; i<numArgs; i++) {
    FieldDecl *FD = K->getArgFields()[i];
    std::string argName = argNames[i];

    HipaccMask *Mask = K->getMaskFromMapping(FD);
    if (Mask) {
      if (options.emitCUDA()) {
        Mask->addKernel(K);
        continue;
      } else {
        if (Mask->isConstant()) continue;
        argName = Mask->getName();
      }
    }

    if (options.emitCUDA() && i==0 && options.useTextureMemory(USER_ON) &&
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

    if (options.exploreConfig() || options.timeKernels()) {
      // add kernel argument
      resultStr += "_args" + kernelName + ".push_back(";
      if (options.emitCUDA()) {
        if (options.exploreConfig()) {
          resultStr += "(void *)&" + argNames[i] + ");\n";
        } else {
          resultStr += "std::make_pair(sizeof(" + argTypeNames[i] + "), (void *)&" + argNames[i] + "));\n";
        }
      } else {
        resultStr += "std::make_pair(sizeof(" + argTypeNames[i] + "), (void *)" + argName + "));\n";
      }
      resultStr += ident;
    } else {
      // set kernel arguments
      setupKernelArgument(kernelName, curArg++, argName, argTypeNames[i],
          offsetStr, resultStr);
    }
  }
  resultStr += "\n" + ident;

  // launch kernel
  if (options.exploreConfig() || options.timeKernels()) {
    std::stringstream max_threads_per_block, max_threads_for_kernel, warp_size, max_shared_memory_per_block;
    max_threads_per_block << K->getMaxThreadsPerBlock();
    max_threads_for_kernel << K->getMaxThreadsForKernel();
    warp_size << K->getWarpSize();
    max_shared_memory_per_block << K->getMaxTotalSharedMemory();

    if (options.emitCUDA()) {
      if (options.timeKernels()) {
        resultStr += "hipaccLaunchKernelBenchmark(\"cu" + kernelName + "\"";
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
    dec_ident();
    resultStr += ident + "}\n";
  } else {
    if (options.emitCUDA()) {
      resultStr += "hipaccLaunchKernel(\"cu";
    } else {
      resultStr += "hipaccEnqueueKernel(";
    }
    resultStr += kernelName;
    if (options.emitCUDA()) resultStr += "\"";
    resultStr += ", " + gridStr;
    resultStr += ", " + blockStr;
    resultStr += ");";
  }
}


void CreateHostStrings::writeGlobalReductionCall(HipaccGlobalReduction *GR,
    std::string &resultStr) {
  std::stringstream GRSS;

  // print runtime function name plus name of reduction function
  if (options.emitCUDA()) {
    if (options.getTargetDevice() >= FERMI_20 && !options.exploreConfig()) {
      resultStr += "hipaccApplyReductionThreadFence<" + GR->getType() + ">(";
      resultStr += "\"cu" + GR->getFileName() + "2D\", ";
    } else {
      if (options.exploreConfig()) {
        resultStr += "hipaccApplyReductionExploration<" + GR->getType() + ">(";
        resultStr += "\"" + GR->getFileName() + ".cu\", ";
      } else {
        resultStr += "hipaccApplyReduction<" + GR->getType() + ">(";
      }
      resultStr += "\"cu" + GR->getFileName() + "2D\", ";
      resultStr += "\"cu" + GR->getFileName() + "1D\", ";
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
      if (options.emitCUDA() && options.useTextureMemory(USER_ON) &&
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
      if (options.emitCUDA() && options.useTextureMemory(USER_ON) &&
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

