//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

// This is the standalone (header-only) Hipacc OpenCL runtime

#include "hipacc_cl.hpp"

#ifndef __HIPACC_CL_STANDALONE_HPP__
#define __HIPACC_CL_STANDALONE_HPP__

#include "hipacc_base_standalone.hpp"

std::string getOpenCLErrorCodeStr(int error) {
#define CL_ERROR_CODE(CODE)                                                    \
  case CODE:                                                                   \
    return #CODE;
  switch (error) {
    CL_ERROR_CODE(CL_SUCCESS)
    CL_ERROR_CODE(CL_DEVICE_NOT_FOUND)
    CL_ERROR_CODE(CL_DEVICE_NOT_AVAILABLE)
    CL_ERROR_CODE(CL_COMPILER_NOT_AVAILABLE)
    CL_ERROR_CODE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
    CL_ERROR_CODE(CL_OUT_OF_RESOURCES)
    CL_ERROR_CODE(CL_OUT_OF_HOST_MEMORY)
    CL_ERROR_CODE(CL_PROFILING_INFO_NOT_AVAILABLE)
    CL_ERROR_CODE(CL_MEM_COPY_OVERLAP)
    CL_ERROR_CODE(CL_IMAGE_FORMAT_MISMATCH)
    CL_ERROR_CODE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
    CL_ERROR_CODE(CL_BUILD_PROGRAM_FAILURE)
    CL_ERROR_CODE(CL_MAP_FAILURE)
#ifdef CL_VERSION_1_1
    CL_ERROR_CODE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
    CL_ERROR_CODE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
#endif
#ifdef CL_VERSION_1_2
    CL_ERROR_CODE(CL_COMPILE_PROGRAM_FAILURE)
    CL_ERROR_CODE(CL_LINKER_NOT_AVAILABLE)
    CL_ERROR_CODE(CL_LINK_PROGRAM_FAILURE)
    CL_ERROR_CODE(CL_DEVICE_PARTITION_FAILED)
    CL_ERROR_CODE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
#endif
    CL_ERROR_CODE(CL_INVALID_VALUE)
    CL_ERROR_CODE(CL_INVALID_DEVICE_TYPE)
    CL_ERROR_CODE(CL_INVALID_PLATFORM)
    CL_ERROR_CODE(CL_INVALID_DEVICE)
    CL_ERROR_CODE(CL_INVALID_CONTEXT)
    CL_ERROR_CODE(CL_INVALID_QUEUE_PROPERTIES)
    CL_ERROR_CODE(CL_INVALID_COMMAND_QUEUE)
    CL_ERROR_CODE(CL_INVALID_HOST_PTR)
    CL_ERROR_CODE(CL_INVALID_MEM_OBJECT)
    CL_ERROR_CODE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    CL_ERROR_CODE(CL_INVALID_IMAGE_SIZE)
    CL_ERROR_CODE(CL_INVALID_SAMPLER)
    CL_ERROR_CODE(CL_INVALID_BINARY)
    CL_ERROR_CODE(CL_INVALID_BUILD_OPTIONS)
    CL_ERROR_CODE(CL_INVALID_PROGRAM)
    CL_ERROR_CODE(CL_INVALID_PROGRAM_EXECUTABLE)
    CL_ERROR_CODE(CL_INVALID_KERNEL_NAME)
    CL_ERROR_CODE(CL_INVALID_KERNEL_DEFINITION)
    CL_ERROR_CODE(CL_INVALID_KERNEL)
    CL_ERROR_CODE(CL_INVALID_ARG_INDEX)
    CL_ERROR_CODE(CL_INVALID_ARG_VALUE)
    CL_ERROR_CODE(CL_INVALID_ARG_SIZE)
    CL_ERROR_CODE(CL_INVALID_KERNEL_ARGS)
    CL_ERROR_CODE(CL_INVALID_WORK_DIMENSION)
    CL_ERROR_CODE(CL_INVALID_WORK_GROUP_SIZE)
    CL_ERROR_CODE(CL_INVALID_WORK_ITEM_SIZE)
    CL_ERROR_CODE(CL_INVALID_GLOBAL_OFFSET)
    CL_ERROR_CODE(CL_INVALID_EVENT_WAIT_LIST)
    CL_ERROR_CODE(CL_INVALID_EVENT)
    CL_ERROR_CODE(CL_INVALID_OPERATION)
    CL_ERROR_CODE(CL_INVALID_GL_OBJECT)
    CL_ERROR_CODE(CL_INVALID_BUFFER_SIZE)
    CL_ERROR_CODE(CL_INVALID_MIP_LEVEL)
    CL_ERROR_CODE(CL_INVALID_GLOBAL_WORK_SIZE)
#ifdef CL_VERSION_1_1
    CL_ERROR_CODE(CL_INVALID_PROPERTY)
#endif
#ifdef CL_VERSION_1_2
    CL_ERROR_CODE(CL_INVALID_IMAGE_DESCRIPTOR)
    CL_ERROR_CODE(CL_INVALID_COMPILER_OPTIONS)
    CL_ERROR_CODE(CL_INVALID_LINKER_OPTIONS)
    CL_ERROR_CODE(CL_INVALID_DEVICE_PARTITION_COUNT)
#endif
#ifdef CL_VERSION_2_0
    CL_ERROR_CODE(CL_INVALID_PIPE_SIZE)
    CL_ERROR_CODE(CL_INVALID_DEVICE_QUEUE)
#endif
  default:
    return "unknown error code";
  }
#undef CL_ERROR_CODE
}

inline void __checkOpenCLErrors(cl_int err, const std::string &name,
                                const std::string &file, const int line) {
  if (err != CL_SUCCESS) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: " + name + " (" + std::to_string(err) +
                                ")" + " [file " + file + ", line " +
                                std::to_string(line) +
                                "]: " + getOpenCLErrorCodeStr(err));
  }
}

HipaccContext &HipaccContext::getInstance() {
  static HipaccContext instance;

  return instance;
}

HipaccImageOpenCLBase::~HipaccImageOpenCLBase() {}

void HipaccContext::add_platform(cl_platform_id id, cl_platform_name name) {
  platforms.push_back(id);
  platform_names.push_back(name);
}

void HipaccContext::add_device(cl_device_id id) { devices.push_back(id); }

void HipaccContext::add_device_all(cl_device_id id) {
  devices_all.push_back(id);
}

void HipaccContext::add_context(cl_context id) { contexts.push_back(id); }

void HipaccContext::add_command_queue(cl_command_queue id) {
  queues.push_back(id);
}

std::vector<cl_platform_id> HipaccContext::get_platforms() { return platforms; }

std::vector<cl_platform_name> HipaccContext::get_platform_names() {
  return platform_names;
}

std::vector<cl_device_id> HipaccContext::get_devices() { return devices; }

std::vector<cl_device_id> HipaccContext::get_devices_all() {
  return devices_all;
}

std::vector<cl_context> HipaccContext::get_contexts() { return contexts; }

std::vector<cl_command_queue> HipaccContext::get_command_queues() {
  return queues;
}

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block) {
  // calculate block id of a) first block that requires no border handling
  // (left, top) and b) first block that requires border handling (right,
  // bottom)
  if (info.size_x > 0) {
    info.bh_start_left = (int)ceilf((float)(info.offset_x + info.size_x) /
                                    (block[0] * info.simd_width));
    info.bh_start_right =
        (int)floor((float)(info.offset_x + info.is_width - info.size_x) /
                   (block[0] * info.simd_width));
  } else {
    info.bh_start_left = 0;
    info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width) /
                                     (block[0] * info.simd_width));
  }
  if (info.size_y > 0) {
    // for shared memory calculate additional blocks to be staged - this is
    // only required if shared memory is used, otherwise, info.size_y would
    // be sufficient
    int p_add = (int)ceilf(2 * info.size_y / (float)block[1]);
    info.bh_start_top =
        (int)ceilf((float)(info.size_y) / (info.pixels_per_thread * block[1]));
    info.bh_start_bottom =
        (int)floor((float)(info.is_height - p_add * block[1]) /
                   (block[1] * info.pixels_per_thread));
  } else {
    info.bh_start_top = 0;
    info.bh_start_bottom = (int)floor((float)(info.is_height) /
                                      (block[1] * info.pixels_per_thread));
  }

  if ((info.bh_start_right - info.bh_start_left) > 1 &&
      (info.bh_start_bottom - info.bh_start_top) > 1) {
    info.bh_fall_back = 0;
  } else {
    info.bh_fall_back = 1;
  }
}

void hipaccCalcGridFromBlock(hipacc_launch_info &info, size_t *block,
                             size_t *grid) {
  grid[0] = (int)ceilf((float)(info.is_width + info.offset_x) /
                       (block[0] * info.simd_width)) *
            block[0];
  grid[1] = (int)ceilf((float)(info.is_height) /
                       (block[1] * info.pixels_per_thread)) *
            block[1];
}

// Select platform and device for execution
void hipaccInitPlatformsAndDevices(cl_device_type dev_type,
                                   cl_platform_name platform_name) {
  HipaccContext &Ctx = HipaccContext::getInstance();
  char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024],
      pd2Buffer[1024];

// Set environment variable to tell AMD/ATI platform to dump kernel
// this has to be done before platform initialization
#ifndef CL_VERSION_1_2
  if (platform_name & AMD) {
    setenv("GPU_DUMP_DEVICE_KERNEL", "3", 1);
  }
#endif
  if (platform_name & NVIDIA) {
    setenv("CUDA_CACHE_DISABLE", "1", 1);
  }

  // Get OpenCL platform count
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  checkErr(err, "clGetPlatformIDs()");

  hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                          "Number of available Platforms: " +
                              std::to_string(num_platforms));

  if (num_platforms == 0) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR, "");
  } else {
    int platform_number = -1, device_number = -1;
    std::vector<cl_platform_id> platforms(num_platforms);
    std::vector<cl_platform_name> platform_names(num_platforms);

    err = clGetPlatformIDs(static_cast<cl_uint>(platforms.size()), platforms.data(), NULL);
    checkErr(err, "clGetPlatformIDs()");

    // Get platform info for each platform
    for (int i = 0; i < static_cast<int>(platforms.size()); ++i) {
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &pnBuffer,
                              NULL);
      err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024,
                               &pvBuffer, NULL);
      err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024,
                               &pv2Buffer, NULL);
      checkErr(err, "clGetPlatformInfo()");

      cl_uint num_devices = 0, num_devices_type = 0;
      err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                           &num_devices);
      err |= clGetDeviceIDs(platforms[i], dev_type, 0, NULL, &num_devices_type);

      // Check if the requested device type was not found for this platform
      if (err != CL_DEVICE_NOT_FOUND)
        checkErr(err, "clGetDeviceIDs()");

      // Get platform name
      if (strncmp(pnBuffer, "AMD", 3) == 0)
        platform_names[i] = AMD;
      else if (strncmp(pnBuffer, "Apple", 3) == 0)
        platform_names[i] = APPLE;
      else if (strncmp(pnBuffer, "ARM", 3) == 0)
        platform_names[i] = ARM;
      else if (strncmp(pnBuffer, "Intel", 3) == 0)
        platform_names[i] = INTEL;
      else if (strncmp(pnBuffer, "NVIDIA", 3) == 0)
        platform_names[i] = NVIDIA;
      else
        platform_names[i] = ALL;

      // Use first platform supporting desired device type
      if (platform_number == -1 && num_devices_type > 0 &&
          (platform_names[i] & platform_name)) {
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                                "  [*] Platform Name: " +
                                    std::string(pnBuffer));
        platform_number = i;
      } else {
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                                "  [ ] Platform Name: " +
                                    std::string(pnBuffer));
      }
      hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                              "      Platform Vendor: " +
                                  std::string(pvBuffer));
      hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                              "      Platform Version: " +
                                  std::string(pv2Buffer));

      std::vector<cl_device_id> devices(num_devices);
      err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, static_cast<cl_uint>(devices.size()),
                           devices.data(), &num_devices);
      checkErr(err, "clGetDeviceIDs()");

      // Get device info for each device
      for (int j = 0; j < static_cast<int>(devices.size()); ++j) {
        cl_device_type this_dev_type;
        cl_uint device_vendor_id;

        err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pnBuffer),
                              &pnBuffer, NULL);
        err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(pvBuffer),
                               &pvBuffer, NULL);
        err |=
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID,
                            sizeof(device_vendor_id), &device_vendor_id, NULL);
        err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE,
                               sizeof(this_dev_type), &this_dev_type, NULL);
        err |= clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(pdBuffer),
                               &pdBuffer, NULL);
        err |= clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(pd2Buffer),
                               &pd2Buffer, NULL);
        checkErr(err, "clGetDeviceInfo()");

        // Use first device of desired type
        if (platform_number == (int)i && device_number == -1 &&
            (this_dev_type & dev_type)) {
          hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, "      [*] ");
          Ctx.add_device(devices[j]);
          device_number = j;
        } else {
          hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, "      [ ] ");
        }
        std::string device_info_s("");
        device_info_s += "Device Name: " + std::string(pnBuffer) + " (";
        if (this_dev_type & CL_DEVICE_TYPE_CPU)
          device_info_s += "CL_DEVICE_TYPE_CPU";
        if (this_dev_type & CL_DEVICE_TYPE_GPU)
          device_info_s += "CL_DEVICE_TYPE_GPU";
        if (this_dev_type & CL_DEVICE_TYPE_ACCELERATOR)
          device_info_s += "CL_DEVICE_TYPE_ACCELERATOR";
#ifdef CL_VERSION_1_2
        if (this_dev_type & CL_DEVICE_TYPE_CUSTOM)
          device_info_s += "CL_DEVICE_TYPE_CUSTOM";
#endif
        if (this_dev_type & CL_DEVICE_TYPE_DEFAULT)
          device_info_s += "|CL_DEVICE_TYPE_DEFAULT";
        device_info_s += ")\n";
        device_info_s += "          Device Vendor: ";
        device_info_s += std::string(pvBuffer);
        device_info_s += (" (ID: " + std::to_string(device_vendor_id) + ")\n");
        device_info_s +=
            ("          Device OpenCL Version: " + std::string(pdBuffer) +
             "\n");
        device_info_s +=
            ("          Device Driver Version: " + std::string(pd2Buffer) +
             "\n");
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, device_info_s);

        // Store all matching devices in a separate array
        if (platform_number == (int)i && (this_dev_type & dev_type)) {
          Ctx.add_device_all(devices[j]);
        }
      }
    }

    if (platform_number == -1) {
      hipaccRuntimeLogTrivial(
          hipaccRuntimeLogLevel::ERROR,
          "No suitable OpenCL platform available, aborting ...");
    }

    Ctx.add_platform(platforms[platform_number],
                     platform_names[platform_number]);
  }
}

// Get a vector with all devices
std::vector<cl_device_id> hipaccGetAllDevices() {
  HipaccContext &Ctx = HipaccContext::getInstance();

  return Ctx.get_devices_all();
}

// Create context and command queue for each device
void hipaccCreateContextsAndCommandQueues(bool all_devies) {
  cl_int err = CL_SUCCESS;
  cl_context context;
  cl_command_queue command_queue;
  HipaccContext &Ctx = HipaccContext::getInstance();

  std::vector<cl_platform_id> platforms = Ctx.get_platforms();
  std::vector<cl_device_id> devices =
      all_devies ? Ctx.get_devices_all() : Ctx.get_devices();

  // Create context
  cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM,
                                     (cl_context_properties)platforms[0], 0};
  context =
      clCreateContext(cprops, static_cast<cl_uint>(devices.size()), devices.data(), NULL, NULL, &err);
  checkErr(err, "clCreateContext()");

  Ctx.add_context(context);

  // Create command queues
  for (auto device : devices) {
#ifdef CL_VERSION_2_0
    cl_queue_properties cprops[3] = {CL_QUEUE_PROPERTIES,
                                     CL_QUEUE_PROFILING_ENABLE, 0};
    command_queue =
        clCreateCommandQueueWithProperties(context, device, cprops, &err);
    checkErr(err, "clCreateCommandQueueWithProperties()");
#else
    command_queue =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "clCreateCommandQueue()");
#endif

    Ctx.add_command_queue(command_queue);
  }
}

// Get binary from OpenCL program and dump it to stderr
void hipaccDumpBinary(cl_program program, cl_device_id device) {
  cl_uint num_devices;

  // Get the number of devices associated with the program
  cl_int err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,
                                sizeof(cl_uint), &num_devices, NULL);

  // Get the associated device ids
  std::vector<cl_device_id> devices(num_devices);
  err |= clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                          devices.size() * sizeof(cl_device_id), devices.data(),
                          0);

  // Get the sizes of the binaries
  std::vector<size_t> binary_sizes(num_devices);
  err |= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                          binary_sizes.size() * sizeof(size_t),
                          binary_sizes.data(), NULL);

  // Get the binaries
  std::vector<unsigned char *> binaries(num_devices);
  for (size_t i = 0; i < binaries.size(); ++i) {
    binaries[i] = new unsigned char[binary_sizes[i]];
  }
  err |= clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                          sizeof(unsigned char *) * binaries.size(),
                          binaries.data(), NULL);
  checkErr(err, "clGetProgramInfo()");

  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] == device) {
      hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, "OpenCL binary : ");
      // binary can contain any character, emit char by char
      for (size_t n = 0; n < binary_sizes[i]; ++n) {
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                                std::to_string(binaries[i][n]));
      }
    }
  }

  for (size_t i = 0; i < num_devices; i++) {
    delete[] binaries[i];
  }
}

// Load OpenCL source file, build program, and create kernel
cl_kernel hipaccBuildProgramAndKernel(std::string file_name,
                                      std::string kernel_name,
                                      bool print_progress, bool dump_binary,
                                      bool print_log, std::string build_options,
                                      std::string build_includes) {
  cl_int err = CL_SUCCESS;
  cl_program program;
  cl_kernel kernel;
  HipaccContext &Ctx = HipaccContext::getInstance();

  std::ifstream srcFile(file_name.c_str());
  if (!srcFile.is_open()) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: Can't open OpenCL source file '" +
                                file_name + "'!");
  }

  std::string clString(std::istreambuf_iterator<char>(srcFile),
                       (std::istreambuf_iterator<char>()));

  const size_t length = clString.length();
  const char *c_str = clString.c_str();

  if (print_progress)
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            "<HIPACC:> Compiling '" + kernel_name + "' .");
  program = clCreateProgramWithSource(Ctx.get_contexts()[0], 1,
                                      (const char **)&c_str, &length, &err);
  checkErr(err, "clCreateProgramWithSource()");

  cl_platform_name platform_name = Ctx.get_platform_names()[0];
  if (build_options.empty()) {
    switch (platform_name) {
    case AMD:
      build_options = "-cl-single-precision-constant -cl-denorms-are-zero";
#ifdef CL_VERSION_1_2
      build_options += " -save-temps";
#endif
      break;
    case NVIDIA:
      build_options =
          "-cl-single-precision-constant -cl-denorms-are-zero -cl-nv-verbose";
      break;
    case APPLE:
    case ARM:
    case INTEL:
    case ALL:
      build_options = "-cl-single-precision-constant -cl-denorms-are-zero";
      break;
    }
  }
  if (!build_includes.empty()) {
    build_options += " " + build_includes;
  }
  err = clBuildProgram(program, 0, NULL, build_options.c_str(), NULL, NULL);
  if (print_progress)
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, ".");

  cl_build_status build_status;
  clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_STATUS,
                        sizeof(build_status), &build_status, NULL);

  if (build_status == CL_BUILD_ERROR || err != CL_SUCCESS || print_log) {
    // determine the size of the options and log
    size_t log_size, options_size;
    err |=
        clGetProgramBuildInfo(program, Ctx.get_devices()[0],
                              CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
    err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0],
                                 CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // allocate memory for the options and log
    char *program_build_options = new char[options_size];
    char *program_build_log = new char[log_size];

    // get the options and log
    err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0],
                                 CL_PROGRAM_BUILD_OPTIONS, options_size,
                                 program_build_options, NULL);
    err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0],
                                 CL_PROGRAM_BUILD_LOG, log_size,
                                 program_build_log, NULL);
    if (print_progress) {
      if (err != CL_SUCCESS)
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::WARNING, ". failed!");
      else
        hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, ".");
    }

    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            "<HIPACC:> OpenCL build options : ");
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            std::string(program_build_options));
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            "<HIPACC:> OpenCL build log : ");
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            std::string(program_build_log));

    // free memory for options and log
    delete[] program_build_options;
    delete[] program_build_log;
  }
  checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

  if (dump_binary)
    hipaccDumpBinary(program, Ctx.get_devices()[0]);

  kernel = clCreateKernel(program, kernel_name.c_str(), &err);
  checkErr(err, "clCreateKernel()");
  if (print_progress)
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, ". done");

  // release program
  err = clReleaseProgram(program);
  checkErr(err, "clReleaseProgram()");

  return kernel;
}

#define CREATE_IMAGE(DATA_TYPE, CHANNEL_TYPE, CHANNEL_ORDER)                   \
  template <>                                                                  \
  HipaccImageOpenCL hipaccCreateImage<DATA_TYPE>(                              \
      DATA_TYPE * host_mem, size_t width, size_t height) {                     \
    return hipaccCreateImage(host_mem, width, height, CHANNEL_TYPE,            \
                             CHANNEL_ORDER);                                   \
  }
CREATE_IMAGE(char, CL_SIGNED_INT8, CL_R)
CREATE_IMAGE(short int, CL_SIGNED_INT16, CL_R)
CREATE_IMAGE(int, CL_SIGNED_INT32, CL_R)
CREATE_IMAGE(unsigned char, CL_UNSIGNED_INT8, CL_R)
CREATE_IMAGE(unsigned short int, CL_UNSIGNED_INT16, CL_R)
CREATE_IMAGE(unsigned int, CL_UNSIGNED_INT32, CL_R)
CREATE_IMAGE(float, CL_FLOAT, CL_R)
CREATE_IMAGE(char4, CL_SIGNED_INT8, CL_RGBA)
CREATE_IMAGE(short4, CL_SIGNED_INT16, CL_RGBA)
CREATE_IMAGE(int4, CL_SIGNED_INT32, CL_RGBA)
CREATE_IMAGE(uchar4, CL_UNSIGNED_INT8, CL_RGBA)
CREATE_IMAGE(ushort4, CL_UNSIGNED_INT16, CL_RGBA)
CREATE_IMAGE(uint4, CL_UNSIGNED_INT32, CL_RGBA)
CREATE_IMAGE(float4, CL_FLOAT, CL_RGBA)

// Create sampler object
cl_sampler hipaccCreateSampler(cl_bool normalized_coords,
                               cl_addressing_mode addressing_mode,
                               cl_filter_mode filter_mode) {
  cl_int err = CL_SUCCESS;
  cl_sampler sampler;
  HipaccContext &Ctx = HipaccContext::getInstance();

#ifdef CL_VERSION_2_0
  cl_sampler_properties sprops[7] = {CL_SAMPLER_NORMALIZED_COORDS,
                                     normalized_coords,
                                     CL_SAMPLER_ADDRESSING_MODE,
                                     addressing_mode,
                                     CL_SAMPLER_FILTER_MODE,
                                     filter_mode,
                                     0};
  sampler = clCreateSamplerWithProperties(Ctx.get_contexts()[0], sprops, &err);
  checkErr(err, "clCreateSamplerWithProperties()");
#else
  sampler = clCreateSampler(Ctx.get_contexts()[0], normalized_coords,
                            addressing_mode, filter_mode, &err);
  checkErr(err, "clCreateSampler()");
#endif

  return sampler;
}

// Copy between memory
void hipaccCopyMemory(const HipaccImageOpenCL &src, HipaccImageOpenCL &dst,
                      int num_device) {
  cl_int err = CL_SUCCESS;
  HipaccContext &Ctx = HipaccContext::getInstance();

  assert(src->get_width() == dst->get_width() && src->get_height() == dst->get_height() &&
         src->get_pixel_size() == dst->get_pixel_size() &&
         "Invalid CopyBuffer or CopyImage!");

  if (src->get_mem_type() >= hipaccMemoryType::Array2D) {
    const size_t origin[] = {0, 0, 0};
    const size_t region[] = {src->get_width(), src->get_height(), 1};

    err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device],
                             (cl_mem)src->get_device_memory(),
                             (cl_mem)dst->get_device_memory(), origin, origin,
                             region, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyImage()");
  } else {
    err = clEnqueueCopyBuffer(
        Ctx.get_command_queues()[num_device], (cl_mem)src->get_device_memory(),
        (cl_mem)dst->get_device_memory(), 0, 0,
        src->get_stride() * src->get_height() * src->get_pixel_size(), 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyBuffer()");
  }
}

// Copy from memory region to memory region
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst, int num_device) {
  cl_int err = CL_SUCCESS;
  HipaccContext &Ctx = HipaccContext::getInstance();

  if (src.img->get_mem_type() >= hipaccMemoryType::Array2D) {
    const size_t dst_origin[] = {(size_t)dst.offset_x, (size_t)dst.offset_y, 0};
    const size_t src_origin[] = {(size_t)src.offset_x, (size_t)src.offset_y, 0};
    const size_t region[] = {static_cast<size_t>(dst.width), static_cast<size_t>(dst.height), 1};

    err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device],
                             (cl_mem)src.img->get_device_memory(),
                             (cl_mem)dst.img->get_device_memory(), src_origin,
                             dst_origin, region, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyImage()");
  } else {
    const size_t dst_origin[] = {dst.offset_x * dst.img->get_pixel_size(),
                                 (size_t)dst.offset_y, 0};
    const size_t src_origin[] = {src.offset_x * src.img->get_pixel_size(),
                                 (size_t)src.offset_y, 0};
    const size_t region[] = {static_cast<size_t>(dst.width * dst.img->get_pixel_size()), static_cast<size_t>(dst.height), 1};

    err = clEnqueueCopyBufferRect(
        Ctx.get_command_queues()[num_device],
        (cl_mem)src.img->get_device_memory(),
        (cl_mem)dst.img->get_device_memory(), src_origin, dst_origin, region,
        src.img->get_stride() * src.img->get_pixel_size(), 0,
        dst.img->get_stride() * dst.img->get_pixel_size(), 0, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyBufferRect()");
  }
}

// Copy between buffers and return time
double hipaccCopyBufferBenchmark(const HipaccImageOpenCL &src,
                                 HipaccImageOpenCL &dst, int num_device,
                                 bool print_timing) {
  cl_int err = CL_SUCCESS;
  cl_ulong end, start;
  std::vector<float> times;
  float last_gpu_timing;
  cl_event event;
  HipaccContext &Ctx = HipaccContext::getInstance();

  assert(src->get_width() == dst->get_width() && src->get_height() == dst->get_height() &&
         src->get_pixel_size() == dst->get_pixel_size() && "Invalid CopyBuffer!");

  for (size_t i = 0; i < HIPACC_NUM_ITERATIONS; ++i) {
    err = clEnqueueCopyBuffer(
        Ctx.get_command_queues()[num_device], (cl_mem)src->get_device_memory(),
        (cl_mem)dst->get_device_memory(), 0, 0,
        src->get_width() * src->get_height() * src->get_pixel_size(), 0, NULL, &event);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyBuffer()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start = (cl_ulong)(start * 1e-3);
    end = (cl_ulong)(end * 1e-3);

    if (print_timing) {
      hipaccRuntimeLogTrivial(
          hipaccRuntimeLogLevel::INFO,
          "<HIPACC:> Copy timing (" +
              std::to_string((src->get_width() * src->get_height() * src->get_pixel_size()) /
                             (float)(1 << 20)) +
              " MB): " + std::to_string((end - start) * 1.0e-3f) + "(ms)");
      hipaccRuntimeLogTrivial(
          hipaccRuntimeLogLevel::INFO,
          "          Bandwidth: " +
              std::to_string(
                  2.0f * (double)(src->get_width() * src->get_height() * src->get_pixel_size()) /
                  ((end - start) * 1.0e-6f * (float)(1 << 30))) +
              " GB/s");
    }
    times.push_back(float(end - start));
  }

  // return time in ms
  err = clReleaseEvent(event);
  checkErr(err, "clReleaseEvent()");

  std::sort(times.begin(), times.end());
  last_gpu_timing = times[times.size() / 2] * 1.0e-3f;
  HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);
  return last_gpu_timing;
}

// Enqueue and launch kernel
void hipaccLaunchKernel(cl_kernel kernel, size_t *global_work_size,
                        size_t *local_work_size, HipaccExecutionParameterOpenCL ep,
                        bool print_timing) {
  cl_int err;
  HipaccContext &Ctx = HipaccContext::getInstance();
  cl_command_queue cg{ ep ? ep->get_command_queue() : Ctx.get_command_queues()[0] };
  cl_event event;

  if (ep) ep->pre_kernel();

  err = clEnqueueNDRangeKernel(cg, kernel, 2, NULL,
                               global_work_size, local_work_size, 0, NULL,
                               print_timing ? &event : nullptr);
  checkErr(err, "clEnqueueNDRangeKernel()");

  if (print_timing) {
    cl_ulong end, start;
    float last_gpu_timing;

    err = clFinish(cg);
    checkErr(err, "clFinish");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start = (cl_ulong)(start * 1e-3);
    end = (cl_ulong)(end * 1e-3);

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");

    last_gpu_timing = (end - start) * 1.0e-3f;
    HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);

    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (" +
            std::to_string(local_work_size[0] * local_work_size[1]) + ": " +
            std::to_string(local_work_size[0]) + "x" +
            std::to_string(local_work_size[1]) +
            "): " + std::to_string(last_gpu_timing) + "(ms)");
  }

  if (ep) ep->post_kernel();
}

#endif // __HIPACC_CL_STANDALONE_HPP__
