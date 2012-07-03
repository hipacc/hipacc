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
    resultStr += ident + type + " *" + memName + " = ";
    // texture is bound at kernel launch
    resultStr += "hipaccCreateMemory<" + type + ">(NULL, ";
  } else {
    resultStr += ident + "cl_mem " + memName + " = ";
    if (options.useTextureMemory()) {
      resultStr += "hipaccCreateImage<" + type + ">(NULL, ";
    } else {
      resultStr += "hipaccCreateBuffer<" + type + ">(NULL, ";
    }
  }
  resultStr += "(int) " + width;
  resultStr += ", (int) " + height;
  resultStr += ", &" + pitchStr;
  if (options.emitOpenCL() && options.useTextureMemory()) {
    // OpenCL Image objects don't support padding
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
        resultStr += "hipaccWriteMemory(";
      } else {
        if (options.useTextureMemory()) {
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
        resultStr += "hipaccReadMemory(";
      } else {
        if (options.useTextureMemory()) {
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
  std::string blockStr, gridStr, offsetStr;
  std::string pptStr = PPTSS.str();

  if (options.emitCUDA()) {
    blockStr = "block" + LSS.str();
    gridStr = "grid" + LSS.str();
    offsetStr = "offset" + LSS.str();
  } else {
    blockStr = "local_work_size" + LSS.str();
    gridStr = "global_work_size" + LSS.str();
  }

  if (options.exploreConfig() || options.timeKernels()) {
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
    if (options.timeKernels()) resultStr += ident;
  }
  if (!options.exploreConfig()) {
    if (options.emitCUDA()) {
      resultStr += "dim3 " + blockStr + "(" + cX.str() + ", " + cY.str() + ");\n";
      resultStr += ident + "dim3 " + gridStr + "((int)ceil((float)(";
      resultStr += K->getIterationSpace()->getWidth();
      // TODO set and calculate per kernel simd width ...
      if (K->vectorize()) {
        resultStr += "/4";
      }
      resultStr += ")/" + blockStr + ".x), ";
      resultStr += "ceil((float)(";
      resultStr += K->getIterationSpace()->getHeight();
      resultStr += ")/(" + blockStr + ".y*" + pptStr + ")));\n";
      resultStr += ident + "size_t " + offsetStr + " = 0;\n";

      // configure kernel call
      resultStr += ident + "hipaccConfigureCall(";
      resultStr += gridStr;
      resultStr += ", " + blockStr;
      resultStr += ");\n\n";
    } else {
      resultStr += "size_t " + blockStr + "[2];\n";
      resultStr += ident + blockStr + "[0] = " + cX.str() + ";\n";
      resultStr += ident + blockStr + "[1] = " + cY.str() + ";\n";
      resultStr += ident + "size_t " + gridStr + "[2];\n";
      resultStr += ident + gridStr + "[0] = ";
      resultStr += "(int)ceil((float)(";
      resultStr += K->getIterationSpace()->getWidth();
      // TODO set and calculate per kernel simd width ...
      if (K->vectorize()) {
        resultStr += "/4";
      }
      resultStr += ")/" + blockStr + "[0])*" + blockStr + "[0];\n";
      resultStr += ident + gridStr + "[1] = ";
      resultStr += "(int)ceil((float)(";
      resultStr += K->getIterationSpace()->getHeight();
      resultStr += ")/(" + blockStr + "[1]*" + pptStr + "))*" + blockStr + "[1];\n\n";
    }
  }
  resultStr += ident;

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
            resultStr += "(void *)" + argNames[i] + "));\n";
          } else {
            resultStr += "hipaccBindTexture<" + argTypeNames[i] + ">(_tex" + K->getArgNames()[i].str() + K->getName() + ", " + argNames[i] + ");\n";
          }
          resultStr += ident;
        }
      }
      if (options.exploreConfig()) {
        if (Acc->getSizeX() > 1 || Acc->getSizeY() > 1) {
          resultStr += "_smems" + kernelName + ".push_back(";
          resultStr += "hipacc_smem_info(" + Acc->getSizeXStr() + ", ";
          resultStr += Acc->getSizeYStr() + ", ";
          resultStr += "sizeof(" + Acc->getImage()->getPixelType() + ")));\n";
          resultStr += ident;
        } 
      }
    }

    if (options.emitCUDA() && options.exploreConfig()) {
      HipaccMask *Mask = K->getMaskFromMapping(FD);
      if (Mask && !Mask->isConstant()) {
        // get constant pointer
        resultStr += ident + "_consts" + kernelName + ".push_back(";
        resultStr += "hipacc_const_info(std::string(\"" + Mask->getName() + K->getName() + "\"), ";
        resultStr += "(void *)" + Mask->getHostMemName() + ", ";
        resultStr += "sizeof(" + Mask->getTypeStr() + ")*" + Mask->getSizeXStr() + "*" + Mask->getSizeYStr() + "));\n";
        resultStr += ident;
      }
    }
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

    HipaccAccessor *Acc = K->getImgFromMapping(FD);
    if (options.emitCUDA() && Acc && K->useTextureMemory(Acc) &&
        KC->getImgAccess(FD)==READ_ONLY) {
      // textures are handled separately
      continue;
    }

    if (options.exploreConfig() || options.timeKernels()) {
        // add kernel argument
      if (options.emitCUDA()) {
        if (options.exploreConfig()) {
          resultStr += "_args" + kernelName + ".push_back((void *)&" + argNames[i] + ");\n";
        } else {
          resultStr += "_args" + kernelName + ".push_back(std::make_pair(sizeof(" + argTypeNames[i] + "), (void *)&" + argNames[i] + "));\n";
        }
      } else {
        resultStr += "_args" + kernelName + ".push_back(std::make_pair(sizeof(" + argTypeNames[i] + "), (void *)" + argName + "));\n";
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
    std::stringstream max_threads_per_block, max_threads_for_kernel, max_size_x, max_size_y, warp_size, max_shared_memory_per_block;
    max_threads_per_block << K->getMaxThreadsPerBlock();
    max_threads_for_kernel << K->getMaxThreadsForKernel();
    max_size_x << K->getMaxSizeX();
    max_size_y << K->getMaxSizeY();
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
    if (options.exploreConfig()) {
      resultStr += ", _smems" + kernelName;
      if (options.emitCUDA()) {
        resultStr += ", _consts" + kernelName;
        resultStr += ", _texs" + kernelName;
      }
      resultStr += ", " + K->getIterationSpace()->getWidth();
      resultStr += ", " + K->getIterationSpace()->getHeight();
      resultStr += ", " + warp_size.str();
      resultStr += ", " + max_threads_per_block.str();
      resultStr += ", " + max_threads_for_kernel.str();
      resultStr += ", " + max_shared_memory_per_block.str();
      resultStr += ", " + pptStr;
      resultStr += ", " + max_size_x.str();
      resultStr += ", " + max_size_y.str();
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
    resultStr += ");\n" + ident;
    resultStr += "}\n";
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
  resultStr += GR->getAccessor()->getImage()->getName() + ", ";
  resultStr += GR->getNeutral() + ", ";
  resultStr += GR->getAccessor()->getImage()->getWidth() + ", ";
  resultStr += GR->getAccessor()->getImage()->getHeight() + ", ";
  resultStr += GR->getAccessor()->getImage()->getStride() + ", ";
  if (GR->isAccessor()) {
    resultStr += GR->getAccessor()->getOffsetX() + ", ";
    resultStr += GR->getAccessor()->getOffsetY() + ", ";
    resultStr += GR->getAccessor()->getWidth() + ", ";
    resultStr += GR->getAccessor()->getHeight() + ", ";
  }
  GRSS << GR->getNumThreads() << ", " << GR->getPixelsPerThread();
  resultStr += GRSS.str();
  if (options.emitCUDA() && options.exploreConfig()) {
    std::stringstream cc_string;
    cc_string << options.getTargetDevice();
    resultStr += ", " + cc_string.str();
  }
  resultStr += ");";
}

// vim: set ts=2 sw=2 sts=2 et ai:

