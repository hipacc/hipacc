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
      resultStr += "#include \"hipacc_cpu_standalone.hpp\"\n"
                   "#define THIS_HIPACC_DEVICE_CPU\n";
      break;
    case Language::CUDA:
      resultStr += "#include \"hipacc_cu_standalone.hpp\"\n"
                   "#define THIS_HIPACC_DEVICE_CUDA\n";
      break;
    case Language::OpenCLACC:
      resultStr += "#include \"hipacc_cl_standalone.hpp\"\n"
                   "#define THIS_HIPACC_DEVICE_OPENCL_ACC\n";
      break;
    case Language::OpenCLCPU:
      resultStr += "#include \"hipacc_cl_standalone.hpp\"\n"
                   "#define THIS_HIPACC_DEVICE_OPENCL_CPU\n";
      break;
    case Language::OpenCLGPU:
      resultStr += "#include \"hipacc_cl_standalone.hpp\"\n"
                   "#define THIS_HIPACC_DEVICE_OPENCL_GPU\n";
      break;
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
  }
}


void writeCLCompilation(const std::string &fileName, const std::string &kernel_name,
    const std::string &includes, std::string &resultStr, std::string suffix="") {
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
        if (K->getKernelClass()->getBinningFunction()) {
          writeCLCompilation(K->getFileName(), K->getBinningName(),
              device.getCLIncludes(), resultStr, "2D");
          resultStr += indent;
          writeCLCompilation(K->getFileName(), K->getBinningName(),
              device.getCLIncludes(), resultStr, "1D");
        }
      }
      break;
  }
}


void CreateHostStrings::writeMemoryAllocation(HipaccImage *Img, std::string const&
    width, std::string const& height, std::string const& host, std::string const& deep_copy
    , std::string &resultStr) {
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += "auto " + Img->getName() + " = ";
      resultStr += "hipaccCreateMemory<" + Img->getTypeStr() + ">(";
      break;
    case Language::CUDA:
      resultStr += "auto " + Img->getName() + " = ";
      // texture is bound at kernel launch
      if (options.useTextureMemory() &&
          options.getTextureType() == Texture::Array2D) {
        resultStr += "hipaccCreateArray2D<" + Img->getTypeStr() + ">(";
      } else {
        resultStr += "hipaccCreateMemory<" + Img->getTypeStr() + ">(";
      }
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "auto " + Img->getName() + " = ";
      if (options.useTextureMemory()) {
        resultStr += "hipaccCreateImage<" + Img->getTypeStr() + ">(";
      } else {
        resultStr += "hipaccCreateBuffer<" + Img->getTypeStr() + ">(";
      }
      break;
  }
  resultStr += (host.empty() ? "nullptr" : host) + ", " + width + ", " + height;

  if(!deep_copy.empty())
    resultStr += ", " + deep_copy;

  if (options.useTextureMemory() &&
      options.getTextureType() == Texture::Array2D) {
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
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += "auto " + Buf->getName() + " = ";
      resultStr += "hipaccCreateMemory<" + Buf->getTypeStr() + ">(";
      break;
    case Language::CUDA:
      resultStr += "auto " + Buf->getName() + " = ";
      hipacc_require(0, "constant memory allocation not required in CUDA!");
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "auto " + Buf->getName() + " = ";
      resultStr += "hipaccCreateBufferConstant<" + Buf->getTypeStr() + ">(";
      break;
  }
  resultStr += "NULL, " + Buf->getSizeXStr() + ", " + Buf->getSizeYStr() + ");";
}

void CreateHostStrings::writeMemoryMapping(HipaccImage *Img, std::string const& argument_name, std::string &resultStr) {
  resultStr += "auto " + Img->getName() + " = hipaccMapMemory<" + Img->getTypeStr() + ">(" + argument_name + ");";
}

void CreateHostStrings::writeMemoryTransfer(HipaccImage *Img, std::string mem,
    MemoryTransferDirection direction, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      resultStr += "hipaccWriteMemory<" + Img->getTypeStr() + ">(";
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
      hipacc_require(0, "Unsupported memory transfer direction!");
      break;
  }
}

void CreateHostStrings::addMemoryTransferGraph(HipaccImage *Img, std::string mem,
    MemoryTransferDirection direction, std::string &graphStr, std::string &nodeStr,
    std::string &nodeDepStr, std::string &nodeArgStr, std::string &resultStr) {
  switch (direction) {
    case HOST_TO_DEVICE:
      resultStr += "hipaccWriteMemoryCudaGraph<" + Img->getTypeStr() + ">(";
      resultStr += Img->getName();
      resultStr += ", " + mem;
      resultStr += ", " + graphStr;
      resultStr += ", " + nodeStr;
      resultStr += ", " + nodeDepStr;
      resultStr += ", " + nodeArgStr;
      resultStr += ");";
      break;
    case DEVICE_TO_HOST:
      resultStr += "hipaccReadMemoryCudaGraph<" + Img->getTypeStr() + ">(";
      resultStr += Img->getName();
      resultStr += ", " + graphStr;
      resultStr += ", " + nodeStr;
      resultStr += ", " + nodeDepStr;
      resultStr += ", " + nodeArgStr;
      resultStr += ");";
      break;
    case DEVICE_TO_DEVICE: // TODO
      resultStr += "hipaccCopyMemory(";
      resultStr += mem + ", ";
      resultStr += Img->getName() + ");";
      break;
    case HOST_TO_HOST:
      hipacc_require(0, "Unsupported memory transfer direction!");
      break;
  }
}


void CreateHostStrings::writeMemoryTransfer(HipaccPyramid *Pyr, std::string idx,
    std::string mem, MemoryTransferDirection direction, std::string &resultStr) {
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
      hipacc_require(0, "Unsupported memory transfer direction!");
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
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccWriteDomainFromMask<" + Mask->getTypeStr() + ">(";
      resultStr += Domain->getName() + ", (" + Mask->getTypeStr() + "*)";
      resultStr += Mask->getHostMemName() + ");";
      break;
  }
}

void CreateHostStrings::writeFusedKernelCall(HipaccKernel *K,
        std::string &resultStr, ASTFuse *kernelFuser) {
  // kernel call elements
  std::string fusedKernelLaunchInfo;
  std::string fusedKernelPrepareLaunchBlock;
  std::string fusedKernelPrepareLaunchGrid;
  std::string fusedKernelPrepareLaunch;
  std::string fusedKernelConfig;
  std::string fusedKernelCall;
  std::string fusedKernelCallArgs;
  std::string fusedDestKernelTexConst;
  std::string fusedDestKernelCall;
  std::string fusedDestKernelLaunch;
  std::string kernel_name = kernelFuser->getFusedKernelName(K);
  std::string pixel_type = K->getKernelClass()->getPixelType().getAsString();

  unsigned newYSizeLocal = kernelFuser->getNewYSizeLocalKernel(K);
  unsigned newYMaxSizeLocal = newYSizeLocal <= 1 ? 0:newYSizeLocal>>1;
  auto argTypeNames = K->getArgTypeNames();
  auto deviceArgNames = K->getDeviceArgNames();
  auto hostArgNames = K->getHostArgNames();

  std::string lit(std::to_string(literal_count++));
  std::string threads_x(std::to_string(K->getNumThreadsX()));
  std::string threads_y(std::to_string(K->getNumThreadsY()));
  std::string blockStr, gridStr, infoStr, infoStrFuse;

  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      blockStr = "block" + lit;
      gridStr = "grid" + lit;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      blockStr = "local_work_size" + lit;
      gridStr = "global_work_size" + lit;
      break;
  }
  infoStr = K->getInfoStr();
  infoStrFuse = "FusedKernel_info";

  if (options.getTargetLang() != Language::C99) {
    // hipacc_launch_info
    fusedKernelLaunchInfo += "hipacc_launch_info " + infoStr + "(";
    fusedKernelLaunchInfo += std::to_string(newYMaxSizeLocal) + ", ";
    fusedKernelLaunchInfo += std::to_string(newYMaxSizeLocal) + ", ";
    fusedKernelLaunchInfo += K->getIterationSpace()->getName() + ", ";
    fusedKernelLaunchInfo += std::to_string(K->getPixelsPerThread()) + ", ";
    if (K->vectorize()) {
      // TODO set and calculate per kernel simd width ...
      fusedKernelLaunchInfo += "4);\n";
    } else {
      fusedKernelLaunchInfo += "1);\n";
    }
    fusedKernelLaunchInfo += indent;
  }

  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      // dim3 block
      fusedKernelPrepareLaunchBlock += "dim3 " + blockStr + "(" + threads_x + ", " + threads_y + ");\n";
      fusedKernelPrepareLaunchBlock += indent;

      // dim3 grid & hipaccCalcGridFromBlock
      fusedKernelPrepareLaunchGrid += "dim3 " + gridStr + "(hipaccCalcGridFromBlock(";
      fusedKernelPrepareLaunchGrid += infoStr + ", ";
      fusedKernelPrepareLaunchGrid += blockStr + "));\n\n";
      fusedKernelPrepareLaunchGrid += indent;

      // hipaccPrepareKernelLaunch
      fusedKernelPrepareLaunch += "hipaccPrepareKernelLaunch(";
      fusedKernelPrepareLaunch += infoStr + ", ";
      fusedKernelPrepareLaunch += blockStr + ");\n";
      fusedKernelPrepareLaunch += indent;

      //// hipaccConfigureCall
      //fusedKernelConfig += "std::vector<void *> _args" + kernel_name + ";\n";
      //fusedKernelConfig += indent;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
    default: assert(0 && "language not support for kernel fusion!");
  }
  std::string execution_parameter_name{};

  if(K->getExecutionParameter().empty()) {
    switch (options.getTargetLang()) {
    default: break;
    case Language::C99:
      execution_parameter_name = "HipaccExecutionParameterCpu{}";
      break;
    case Language::CUDA:
      execution_parameter_name = "HipaccExecutionParameterCuda{}";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      execution_parameter_name = "HipaccExecutionParameterOpenCL{}";
      break;
    }
  }
  else {
    switch (options.getTargetLang()) {
    default:
        hipacc_check(false, "Ignoring execution parameter for kernel \"" + kernel_name + "\" as it is currently only supported for CPU, CUDA and OpenCL\n.");
        break;
    case Language::C99:
    case Language::CUDA:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
        execution_parameter_name = "exec_param" + lit;
        fusedKernelConfig += "auto " + execution_parameter_name + " = hipaccMapExecutionParameter(" + K->getExecutionParameter() + ");\n\n";
        fusedKernelConfig += indent;
      break;
    }
  }

  // bind textures and get constant pointers
  size_t num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;

    // get param name as well as its orig name before kernel fusion
    std::string nameTemp(K->getDeviceArgNames()[i]);
    std::string nameOrig = nameTemp.substr(0, nameTemp.find("_kernelFusion_"));

    // skip unused variables
    if (!K->getUsed(nameTemp) && !K->getUsed(nameOrig))
      continue;

    //TODO, enable use case
    if (auto Acc = K->getImgFromMapping(arg)) {
      if (options.emitCUDA() && K->useTextureMemory(Acc) != Texture::None &&
                                K->useTextureMemory(Acc) != Texture::Ldg) {
        std::string tex_type = "Texture", hipacc_type = "Array2D";
        if (K->getKernelClass()->getMemAccess(arg) == WRITE_ONLY)
          tex_type = hipacc_type = "Surface";
        // bind texture and surface
        std::string tex_reference = tex_type;
        tex_reference[0] = std::tolower(tex_reference[0], std::locale());
        fusedDestKernelTexConst += "const " + tex_reference + "Reference *_tex" + deviceArgNames[i] + K->getName() + "Ref;\n";
        fusedDestKernelTexConst += indent;
        fusedDestKernelTexConst += "cudaGet" + tex_type + "Reference(&";
        fusedDestKernelTexConst += "_tex" + deviceArgNames[i] + K->getName() + "Ref, &";
        fusedDestKernelTexConst += "_tex" + deviceArgNames[i] + K->getName() + ");\n";
        fusedDestKernelTexConst += indent;
        fusedDestKernelTexConst += "hipaccBind" + tex_type + "<" + argTypeNames[i] + ">(";
        switch (K->useTextureMemory(Acc)) {
          case Texture::Linear1D: fusedDestKernelTexConst += "Linear1D";  break;
          case Texture::Linear2D: fusedDestKernelTexConst += "Linear2D";  break;
          case Texture::Array2D:  fusedDestKernelTexConst += hipacc_type; break;
          default: assert(0 && "unsupported texture type!");
        }
        fusedDestKernelTexConst += ", _tex" + deviceArgNames[i] + K->getName() + "Ref, ";
        fusedDestKernelTexConst += hostArgNames[i] + ");\n";
        fusedDestKernelTexConst += indent;
      }
    }
  }

  std::string print_timing = options.timeKernels() ? "true" : "false";
  // parameters
  size_t cur_arg = 0;
  num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;
    // skip unused variables
    std::string Name(K->getDeviceArgNames()[i]);
    std::string nameOrig = Name.substr(0, Name.find("_"+K->getKernelName()));
    if (!K->getUsed(Name) && !K->getUsed(nameOrig)){
      continue;
    }
    HipaccMask *Mask = K->getMaskFromMapping(arg);
    if (Mask) {
      if (options.emitCUDA()) {
        Mask->addKernel(K);
        continue;
      } else {
        if (Mask->isConstant())
          continue;
      }
    }
    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    if (options.emitCUDA() && Acc && K->useTextureMemory(Acc) != Texture::None &&
                                     K->useTextureMemory(Acc) != Texture::Ldg)
      continue; // textures are handled separately
    std::string img_mem;
    if (Acc || Mask) {
      if (options.emitC99()) {
        img_mem = "->get_aligned_host_memory()";
      } else {
        img_mem = "->get_device_memory()";
      }
    }
    std::string str_stride("->get_stride()");

    // set kernel arguments
    switch (options.getTargetLang()) {
      case Language::C99:
      case Language::CUDA:
        if (cur_arg++ == 0) {
          fusedKernelCall += "hipaccLaunchKernel(" + kernel_name;
          fusedKernelCall += ", " + gridStr;
          fusedKernelCall += ", " + blockStr;
          fusedKernelCall += ", " + execution_parameter_name;
          fusedKernelCall += ", " + print_timing;
          fusedKernelCall += ", 0, ";
        } else {
          fusedKernelCallArgs += ", ";
        }
        if (Mask) {
          fusedKernelCallArgs += "(" + argTypeNames[i] + ")";
        }
        fusedKernelCallArgs += hostArgNames[i] + img_mem;
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        break;
    }
  }

  if (options.getTargetLang()!=Language::CUDA)
    fusedDestKernelCall += "\n" + indent;

  // launch kernel
  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
        fusedDestKernelLaunch += ");\n";
        fusedDestKernelLaunch += indent;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      fusedDestKernelLaunch += "hipaccLaunchKernel(";
      fusedDestKernelLaunch += kernel_name;
      fusedDestKernelLaunch += ", " + gridStr;
      fusedDestKernelLaunch += ", " + blockStr;
      fusedDestKernelLaunch += ", " + execution_parameter_name;
      fusedDestKernelLaunch += ", " + print_timing;
      fusedDestKernelLaunch += ");";
      break;
  }

  if (!kernelFuser->isDestKernel(K)) {
    // concatenate kernel calls
    if (kernelFuser->isSrcKernel(K)) {
      fusedKernelLaunchInfoMap[K] = fusedKernelLaunchInfo;
      fusedKernelCallMap[K] = fusedKernelCallArgs + ", ";
      fusedKernelPrepareLaunchMap[K] = fusedKernelPrepareLaunchBlock + fusedKernelPrepareLaunch;
    } else {
      fusedKernelLaunchInfoMap[K] =
        fusedKernelLaunchInfoMap[kernelFuser->getProducerKernel(K)] +
        fusedKernelLaunchInfo;
      fusedKernelCallMap[K] =
        fusedKernelCallMap[kernelFuser->getProducerKernel(K)] +
        fusedKernelCallArgs + ", ";
      fusedKernelPrepareLaunchMap[K] =
        fusedKernelPrepareLaunchMap[kernelFuser->getProducerKernel(K)] +
        fusedKernelPrepareLaunchBlock + fusedKernelPrepareLaunch;
    }
  } else {
    resultStr += fusedKernelLaunchInfoMap[kernelFuser->getProducerKernel(K)];
    resultStr += fusedKernelLaunchInfo;
    resultStr += fusedKernelPrepareLaunchBlock;
    resultStr += fusedKernelPrepareLaunchGrid;
    resultStr += fusedKernelPrepareLaunchMap[kernelFuser->getProducerKernel(K)];
    resultStr += fusedKernelPrepareLaunch;
    resultStr += fusedKernelConfig;
    resultStr += fusedDestKernelTexConst;
    resultStr += fusedKernelCall;
    resultStr += fusedKernelCallMap[kernelFuser->getProducerKernel(K)];
    resultStr += fusedKernelCallArgs;
    resultStr += fusedDestKernelCall;
    resultStr += fusedDestKernelLaunch;
  }
}

void CreateHostStrings::writeKernelCall(HipaccKernel *K, std::string &resultStr) {
  auto argTypeNames = K->getArgTypeNames();
  auto deviceArgNames = K->getDeviceArgNames();
  auto hostArgNames = K->getHostArgNames();
  std::string kernel_name = K->getKernelName();
  std::string pixel_type = K->getKernelClass()->getPixelType().getAsString();

  std::string lit(std::to_string(literal_count++));
  std::string threads_x(std::to_string(K->getNumThreadsX()));
  std::string threads_y(std::to_string(K->getNumThreadsY()));
  std::string blockStr, gridStr, infoStr;

  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
      blockStr = "block" + lit;
      gridStr = "grid" + lit;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      blockStr = "local_work_size" + lit;
      gridStr = "global_work_size" + lit;
      break;
  }
  infoStr = K->getInfoStr();

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

      // kernel args
      //resultStr += "std::vector<void *> _args" + kernel_name + ";\n";
      //resultStr += indent;
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

  std::string execution_parameter_name{};

  if(K->getExecutionParameter().empty()) {
    switch (options.getTargetLang()) {
    default: break;
    case Language::C99:
      execution_parameter_name = "HipaccExecutionParameterCpu{}";
      break;
    case Language::CUDA:
      execution_parameter_name = "HipaccExecutionParameterCuda{}";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      execution_parameter_name = "HipaccExecutionParameterOpenCL{}";
      break;
    }
  }
  else {
    switch (options.getTargetLang()) {
    default:
        hipacc_check(false, "Ignoring execution parameter for kernel \"" + kernel_name + "\" as it is currently only supported for CPU, CUDA and OpenCL\n.");
        break;
    case Language::C99:
    case Language::CUDA:
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
        execution_parameter_name = "exec_param" + lit;
        resultStr += "auto " + execution_parameter_name + " = hipaccMapExecutionParameter(" + K->getExecutionParameter() + ");\n\n";
        resultStr += indent;
      break;
    }
  }

  // bind textures and get constant pointers
  size_t num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i]))
      continue;

    if (auto Acc = K->getImgFromMapping(arg)) {
      if (options.emitCUDA() && K->useTextureMemory(Acc) != Texture::None &&
                                K->useTextureMemory(Acc) != Texture::Ldg) {
        std::string tex_type = "Texture", hipacc_type = "Array2D";
        if (K->getKernelClass()->getMemAccess(arg) == WRITE_ONLY)
          tex_type = hipacc_type = "Surface";
        // bind texture and surface
        std::string tex_reference = tex_type;
        tex_reference[0] = std::tolower(tex_reference[0], std::locale());
        resultStr += "const " + tex_reference + "Reference *_tex" + deviceArgNames[i] + K->getName() + "Ref;\n";
        resultStr += indent;
        resultStr += "cudaGet" + tex_type + "Reference(&";
        resultStr += "_tex" + deviceArgNames[i] + K->getName() + "Ref, &";
        resultStr += "_tex" + deviceArgNames[i] + K->getName() + ");\n";
        resultStr += indent;
        resultStr += "hipaccBind" + tex_type + "<" + argTypeNames[i] + ">(";
        switch (K->useTextureMemory(Acc)) {
          case Texture::Linear1D: resultStr += "hipaccMemoryType::Linear1D";  break;
          case Texture::Linear2D: resultStr += "hipaccMemoryType::Linear2D";  break;
          case Texture::Array2D:  resultStr += hipacc_type; break;
          default: hipacc_require(0, "unsupported texture type!");
        }
        resultStr += ", _tex" + deviceArgNames[i] + K->getName() + "Ref, ";
        resultStr += hostArgNames[i] + ");\n";
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


  std::string print_timing = options.timeKernels() ? "true" : "false";

  // parameters
  size_t cur_arg = 0;
  num_arg = 0;
  for (auto arg : K->getDeviceArgFields()) {
    size_t i = num_arg++;

    // skip unused variables
    if (!K->getUsed(K->getDeviceArgNames()[i]))
      continue;

    HipaccMask *Mask = K->getMaskFromMapping(arg);
    if (Mask) {
      if (options.emitCUDA()) {
        Mask->addKernel(K);
        continue;
      } else {
        if (Mask->isConstant())
          continue;
      }
    }

    HipaccAccessor *Acc = K->getImgFromMapping(arg);
    if (options.emitCUDA() && Acc && K->useTextureMemory(Acc) != Texture::None &&
                                     K->useTextureMemory(Acc) != Texture::Ldg)
      continue; // textures are handled separately

    std::string img_mem;
    if (Acc || Mask) {
      if (options.emitC99()) {
        img_mem = "->get_aligned_host_memory()";
      } else {
        img_mem = "->get_device_memory()";
      }
    }
    std::string str_stride("->get_stride()");
    bool str_has_stride = hostArgNames[i].find(str_stride) != std::string::npos;

    // set kernel arguments
    switch (options.getTargetLang()) {
      case Language::C99:
        if (cur_arg++ == 0) {
          resultStr += "hipaccLaunchKernel( [=]{ ";
          resultStr += kernel_name + "(";
        } else {
          resultStr += ", ";
        }
        if (Acc) {
          resultStr += "(" + Acc->getImage()->getTypeStr() + "*)";
        }
        if (Mask) {
          resultStr += "(" + Mask->getTypeStr() + "*)";
        }
        resultStr += hostArgNames[i] + img_mem;
        break;
      case Language::CUDA:
        if (cur_arg++ == 0) {
          if (options.useGraph()) {
            resultStr += "hipaccLaunchKernelCudaGraph(";
          } else {
            resultStr += "hipaccLaunchKernel(";
          }
          resultStr += kernel_name + ", " + gridStr;
          resultStr += ", " + blockStr;
          resultStr += ", " + execution_parameter_name;
          if (options.useGraph()) {
            resultStr += ", " + K->getGraphName();
            resultStr += ", " + K->getGraphNodeName();
            resultStr += ", " + K->getGraphNodeDepName();
            resultStr += ", " + K->getGraphNodeArgName();
          } else {
            resultStr += ", " + print_timing;
          }
          resultStr += ", 0, ";
        }
        else resultStr += ", ";
        if (Mask) {
          resultStr += "(" + argTypeNames[i] + ")";
        }
        resultStr += hostArgNames[i] + img_mem;
        break;
      case Language::OpenCLACC:
      case Language::OpenCLCPU:
      case Language::OpenCLGPU:
        if (Acc || Mask || str_has_stride) {
          resultStr += "auto _mem_addr" + lit + std::to_string(i) + " = ";
          resultStr += hostArgNames[i] + img_mem + ";\n";
          resultStr += indent;
          resultStr += "hipaccSetKernelArg(";
          resultStr += kernel_name;
          resultStr += ", ";
          resultStr += std::to_string(cur_arg++);
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += "&_mem_addr" + lit + std::to_string(i);
          resultStr += ");\n";
          resultStr += indent;
        } else {
          resultStr += "hipaccSetKernelArg(";
          resultStr += kernel_name;
          resultStr += ", ";
          resultStr += std::to_string(cur_arg++);
          resultStr += ", sizeof(" + argTypeNames[i] + "), ";
          resultStr += "&" + hostArgNames[i] + img_mem;
          resultStr += ");\n";
          resultStr += indent;
        }
        break;
    }
  }
  if (options.getTargetLang()==Language::C99) {
    resultStr += ");"; // close parenthesis for function call
    resultStr += " }"; // close lambda function
    resultStr += ", " + execution_parameter_name;
    resultStr += ", " + print_timing;
    resultStr += ");\n";
    resultStr += indent;
  }

  if (options.getTargetLang()!=Language::CUDA)
    resultStr += "\n" + indent;

  // launch kernel
  switch (options.getTargetLang()) {
    case Language::C99: break;
    case Language::CUDA:
        resultStr += ");\n";
        resultStr += indent;
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += "hipaccLaunchKernel(";
      resultStr += kernel_name;
      resultStr += ", " + gridStr;
      resultStr += ", " + blockStr;
      resultStr += ", " + execution_parameter_name;
      resultStr += ", " + print_timing;
      resultStr += ");";
      break;
  }
}


void CreateHostStrings::writeReduceCall(HipaccKernel *K, std::string &resultStr) {
  std::string typeStr(K->getIterationSpace()->getImage()->getTypeStr());
  std::string red_decl(typeStr + " " + K->getReduceStr() + " = ");

  std::string execution_parameter_name{};

  if(K->getExecutionParameter().empty()) {
    switch (options.getTargetLang()) {
    default: break;
    case Language::C99:
      execution_parameter_name = "HipaccExecutionParameterCpu{}";
      break;
    case Language::CUDA:
      execution_parameter_name = "HipaccExecutionParameterCuda{}";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      execution_parameter_name = "HipaccExecutionParameterOpenCL{}";
      break;
    }
  }

  else {
    switch (options.getTargetLang()) {
    default:
      hipacc_check(false, "Ignoring execution parameter for reduction kernel \"" + K->getReduceName() + "\" as it is currently only supported for CPU, CUDA and OpenCL.\n");
      break;
    case Language::C99:
    case Language::CUDA:
        execution_parameter_name = "exec_param_" + K->getReduceName();
        resultStr += "auto " + execution_parameter_name + " = hipaccMapExecutionParameter(" + K->getExecutionParameter() + ");\n\n";
        resultStr += indent;
      break;
    }
  }

  // print runtime function name plus name of reduction function
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += red_decl;
      resultStr += "hipaccApplyReduction<" + typeStr + ">( [=]()->" + typeStr + "{ return ";
      resultStr += K->getReduceName() + "2DKernel(";
      resultStr += "(" + K->getIterationSpace()->getImage()->getTypeStr() + "*)";
      resultStr += K->getIterationSpace()->getName() + ".img->get_aligned_host_memory(), ";
      resultStr += K->getIterationSpace()->getName() + ".width, ";
      resultStr += K->getIterationSpace()->getName() + ".height, ";
      resultStr += K->getIterationSpace()->getName() + ".img->get_stride()";
      if (K->getIterationSpace()->isCrop()) {
        resultStr += ", " + K->getIterationSpace()->getName() + ".offset_x";
        resultStr += ", " + K->getIterationSpace()->getName() + ".offset_y";
      }
      resultStr += "); }";
      break;
    case Language::CUDA:
      // first get texture reference
      resultStr += "cudaGetTextureReference(";
      resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + "Ref, ";
      resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + ");\n";
      resultStr += indent;
      resultStr += red_decl;
      resultStr += "hipaccApplyReductionShared<" + typeStr + ">(";
      resultStr += "hipacc_shared_reduction<" + typeStr +", " + K->getReduceName() + ">, ";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += red_decl;
      resultStr += "hipaccApplyReduction<" + typeStr + ">(";
      resultStr += K->getReduceName() + "2D, ";
      resultStr += K->getReduceName() + "1D, ";
      break;
  }

  if (!options.emitC99()) {
    // print image name
    resultStr += K->getIterationSpace()->getName() + ", ";

    // print pixels per thread
    resultStr += std::to_string(K->getNumThreadsReduce()) + ", ";
    resultStr += std::to_string(K->getPixelsPerThreadReduce());
  }

  //print execution parameter
  if(!execution_parameter_name.empty())
    resultStr += ", " + execution_parameter_name;

  if (options.emitCUDA()) {
    // print 2D CUDA array texture information - this parameter is only used if
    // the texture type is Array2D
    resultStr += ", _tex";
    resultStr += K->getIterationSpace()->getImage()->getName() + K->getName();
    resultStr += "Ref";
  }

  std::string print_timing = options.timeKernels() ? "true" : "false";
  resultStr += ", " + print_timing;

  resultStr += ");\n";
}


void CreateHostStrings::writeBinningCall(HipaccKernel *K, std::string &resultStr) {
  std::string pixTypeStr(K->getKernelClass()->getPixelType().getAsString());
  std::string binTypeStr(K->getKernelClass()->getBinType().getAsString());
  std::string bin_decl;

  std::string execution_parameter_name{};

  if(K->getExecutionParameter().empty()) {
    switch (options.getTargetLang()) {
    default: break;
    case Language::C99:
      execution_parameter_name = "HipaccExecutionParameterCpu{}";
      break;
    case Language::CUDA:
      execution_parameter_name = "HipaccExecutionParameterCuda{}";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      execution_parameter_name = "HipaccExecutionParameterOpenCL{}";
      break;
    }
  }

  else {
    switch (options.getTargetLang()) {
    default:
      hipacc_check(false, "Ignoring execution parameter for reduction kernel \"" + K->getReduceName() + "\" as it is currently only supported for CPU, CUDA and OpenCL.\n");
      break;
    case Language::C99:
    case Language::CUDA:
        execution_parameter_name = "exec_param_" + K->getReduceName();
        resultStr += "auto " + execution_parameter_name + " = hipaccMapExecutionParameter(" + K->getExecutionParameter() + ");\n\n";
        resultStr += indent;
      break;
    }
  }

  std::string resultTypeStr{};
  switch (options.getTargetLang()) {
    case Language::C99:
      resultTypeStr = "std::vector<" + binTypeStr + ">";
      break;
    default:
      resultTypeStr = binTypeStr + "*";
      break;
  }

  bin_decl = resultTypeStr + K->getBinningStr();

  // print runtime function name plus name of reduction function
  switch (options.getTargetLang()) {
    case Language::C99:
      resultStr += bin_decl + " = ";
      resultStr += "hipaccApplyReduction<" + resultTypeStr + ">( [=]()->" + resultTypeStr + "{ return ";
      resultStr += K->getBinningName() + "2DKernel(";
      resultStr += "(" + K->getIterationSpace()->getImage()->getTypeStr() + "*)";
      resultStr += K->getIterationSpace()->getName() + ".img->get_aligned_host_memory(), ";
      resultStr += K->getNumBinsStr() + ", ";
      resultStr += K->getIterationSpace()->getName() + ".width, ";
      resultStr += K->getIterationSpace()->getName() + ".height, ";
      resultStr += K->getIterationSpace()->getName() + ".img->get_stride()";
      if (K->getIterationSpace()->isCrop()) {
        resultStr += ", " + K->getIterationSpace()->getName() + ".offset_x";
        resultStr += ", " + K->getIterationSpace()->getName() + ".offset_y";
      }
      resultStr += "); }";
      break;
    case Language::CUDA:
      // first get texture reference
      resultStr += "cudaGetTextureReference(";
      resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + "Ref, ";
      resultStr += "&_tex" + K->getIterationSpace()->getImage()->getName() + K->getName() + ");\n";
      resultStr += indent;

      resultStr += bin_decl + " = ";

      resultStr += "hipaccApplyBinningSegmented<";
      resultStr += binTypeStr + ", ";
      resultStr += pixTypeStr + ">(";
      resultStr += "hipacc_binning_reduction<";
      resultStr += binTypeStr + ", ";
      resultStr += pixTypeStr + ", ";
      resultStr += K->getBinningName() + ", ";
      resultStr += K->getReduceName() + ", ";
      resultStr += std::to_string(options.getReduceConfigNumWarps()) + ", ";
      resultStr += std::to_string(options.getReduceConfigNumUnits()) + ">, ";
      break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:
      resultStr += bin_decl + " = ";
      resultStr += "hipaccApplyBinningSegmented<";
      resultStr += binTypeStr + ", ";
      resultStr += pixTypeStr + ">(";
      resultStr += K->getBinningName() + "2D, ";
      resultStr += K->getBinningName() + "1D, ";
      break;
  }

  if (!options.emitC99()) {
    // print image name
    resultStr += K->getIterationSpace()->getName() + ", ";
    resultStr += std::to_string(options.getReduceConfigNumWarps()) + ", ";
    resultStr += std::to_string(options.getReduceConfigNumUnits()) + ", ";
    resultStr += K->getNumBinsStr();
  }

  //print execution parameter
  if(!execution_parameter_name.empty())
    resultStr += ", " + execution_parameter_name;

  if (options.emitCUDA()) {
    // print 2D CUDA array texture information - this parameter is only used if
    // the texture type is Array2D
    resultStr += ", _tex";
    resultStr += K->getIterationSpace()->getImage()->getName() + K->getName();
    resultStr += "Ref";
  }

  std::string print_timing = options.timeKernels() ? "true" : "false";
  resultStr += ", " + print_timing;

  resultStr += ");\n";
}


std::string CreateHostStrings::getInterpolationDefinition(HipaccKernel *K,
    HipaccAccessor *Acc, std::string function_name, std::string type_suffix,
    Interpolate ip_mode, Boundary bh_mode) {
  std::string str;
  // interpolation macro
  switch (ip_mode) {
    case Interpolate::NO:
    case Interpolate::NN:
      str += "DEFINE_BH_VARIANT_NO_BH(INTERPOLATE_LINEAR_FILTERING";
      break;
    case Interpolate::B5:
      str += "DEFINE_BH_VARIANT(INTERPOLATE_BINOMIAL5_FILTERING";
      break;
    case Interpolate::LF:
      str += "DEFINE_BH_VARIANT(INTERPOLATE_LINEAR_FILTERING";
      break;
    case Interpolate::CF:
      str += "DEFINE_BH_VARIANT(INTERPOLATE_CUBIC_FILTERING";
      break;
    case Interpolate::L3:
      str += "DEFINE_BH_VARIANT(INTERPOLATE_LANCZOS_FILTERING";
      break;
  }
  switch (options.getTargetLang()) {
    case Language::C99:          str += "_CPU, ";    break;
    case Language::CUDA:         str += "_CUDA, ";   break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:    str += "_OPENCL, "; break;
  }
  // data type
  str += Acc->getImage()->getTypeStr() + ", ";
  // append short data type - overloading is not supported in OpenCL
  if (options.emitOpenCL()) {
    str += type_suffix + ", ";
  }
  // interpolation function
  str += function_name;
  // boundary handling name + function (upper & lower)
  std::string const_parameter("NO_PARM");
  std::string const_suffix;
  switch (bh_mode) {
    case Boundary::UNDEFINED:
      str += ", NO_BH, NO_BH, "; break;
    case Boundary::CLAMP:
      str += "_clamp, BH_CLAMP_LOWER, BH_CLAMP_UPPER, "; break;
    case Boundary::REPEAT:
      str += "_repeat, BH_REPEAT_LOWER, BH_REPEAT_UPPER, "; break;
    case Boundary::MIRROR:
      str += "_mirror, BH_MIRROR_LOWER, BH_MIRROR_UPPER, "; break;
    case Boundary::CONSTANT:
      str += "_constant, BH_CONSTANT_LOWER, BH_CONSTANT_UPPER, ";
      const_parameter = "CONST_PARM";
      const_suffix = "_CONST";
      break;
  }
  // image memory parameter, constant parameter, memory access function
  switch (K->useTextureMemory(Acc)) {
    case Texture::None:
      str += "IMG_PARM, " + const_parameter + ", IMG" + const_suffix;
      break;
    case Texture::Linear1D:
      str += "TEX_PARM, " + const_parameter + ", TEX" + const_suffix;
      break;
    case Texture::Linear2D:
    case Texture::Array2D:
      str += "ARR_PARM, " + const_parameter + ", ARR" + const_suffix;
      break;
    case Texture::Ldg:
      str += "LDG_PARM, " + const_parameter + ", LDG" + const_suffix;
      break;
  }
  // image read function for OpenCL
  if (options.emitOpenCL()) {
    str += ", " + Acc->getImage()->getImageReadFunction();
  }
  str += ")\n";

  return str;
}


void CreateHostStrings::writePyramidAllocation(std::string pyrName, std::string
    type, std::string img, std::string depth, std::string &resultStr) {
  std::string img_type;
  switch (options.getTargetLang()) {
    case Language::C99:          img_type = "auto";    break;
    case Language::CUDA:         img_type = "auto";   break;
    case Language::OpenCLACC:
    case Language::OpenCLCPU:
    case Language::OpenCLGPU:    img_type = "auto"; break;
  }
  resultStr += img_type + " " + pyrName + " = ";
  resultStr += "hipaccCreatePyramid<" + type + ">(";
  resultStr += img + ", " + depth + ");";
}

void CreateHostStrings::writePyramidMapping(std::string pyrName, std::string
    type, std::string assigned_pyramid, std::string &resultStr) {
  resultStr += "auto " + pyrName + " = hipaccMapPyramid<" + type + ">(" + assigned_pyramid + ");";
}

// vim: set ts=2 sw=2 sts=2 et ai:

