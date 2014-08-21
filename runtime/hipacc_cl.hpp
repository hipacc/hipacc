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

#ifndef __HIPACC_CL_HPP__
#define __HIPACC_CL_HPP__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "hipacc_base.hpp"

#define EVENT_TIMING

enum cl_platform_name {
    AMD     = 0x1,
    APPLE   = 0x2,
    ARM     = 0x4,
    INTEL   = 0x8,
    NVIDIA  = 0x10,
    ALL     = (AMD|APPLE|ARM|INTEL|NVIDIA)
};


class HipaccContext : public HipaccContextBase {
    private:
        std::vector<cl_platform_id> platforms;
        std::vector<cl_platform_name> platform_names;
        std::vector<cl_device_id> devices, devices_all;
        std::vector<cl_context> contexts;
        std::vector<cl_command_queue> queues;

    public:
        static HipaccContext &getInstance() {
            static HipaccContext instance;

            return instance;
        }
        void add_platform(cl_platform_id id, cl_platform_name name) {
            platforms.push_back(id);
            platform_names.push_back(name);
        }
        void add_device(cl_device_id id) { devices.push_back(id); }
        void add_device_all(cl_device_id id) { devices_all.push_back(id); }
        void add_context(cl_context id) { contexts.push_back(id); }
        void add_command_queue(cl_command_queue id) { queues.push_back(id); }
        std::vector<cl_platform_id> get_platforms() { return platforms; }
        std::vector<cl_platform_name> get_platform_names() { return platform_names; }
        std::vector<cl_device_id> get_devices() { return devices; }
        std::vector<cl_device_id> get_devices_all() { return devices_all; }
        std::vector<cl_context> get_contexts() { return contexts; }
        std::vector<cl_command_queue> get_command_queues() { return queues; }
};


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block) {
    // calculate block id of a) first block that requires no border handling
    // (left, top) and b) first block that requires border handling (right,
    // bottom)
    if (info.size_x > 0) {
        info.bh_start_left = (int)ceilf((float)(info.offset_x + info.size_x) / (block[0] * info.simd_width));
        info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width - info.size_x) / (block[0] * info.simd_width));
    } else {
        info.bh_start_left = 0;
        info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width) / (block[0] * info.simd_width));
    }
    if (info.size_y > 0) {
        // for shared memory calculate additional blocks to be staged - this is
        // only required if shared memory is used, otherwise, info.size_y would
        // be sufficient
        int p_add = (int)ceilf(2*info.size_y / (float)block[1]);
        info.bh_start_top = (int)ceilf((float)(info.size_y) / (info.pixels_per_thread * block[1]));
        info.bh_start_bottom = (int)floor((float)(info.is_height - p_add*block[1]) / (block[1] * info.pixels_per_thread));
    } else {
        info.bh_start_top = 0;
        info.bh_start_bottom = (int)floor((float)(info.is_height) / (block[1] * info.pixels_per_thread));
    }

    if ((info.bh_start_right - info.bh_start_left) > 1 && (info.bh_start_bottom - info.bh_start_top) > 1) {
        info.bh_fall_back = 0;
    } else {
        info.bh_fall_back = 1;
    }
}


void hipaccCalcGridFromBlock(hipacc_launch_info &info, size_t *block, size_t *grid) {
    grid[0] = (int)ceilf((float)(info.is_width + info.offset_x)/(block[0]*info.simd_width)) * block[0];
    grid[1] = (int)ceilf((float)(info.is_height)/(block[1]*info.pixels_per_thread)) * block[1];
}


std::string getOpenCLErrorCodeStr(int errorCode) {
    switch (errorCode) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        #ifdef CL_VERSION_1_1
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        #endif
        #ifdef CL_VERSION_1_2
        case CL_COMPILE_PROGRAM_FAILURE:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        #endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        #ifdef CL_VERSION_1_1
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
        #endif
        #ifdef CL_VERSION_1_2
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        #endif
        default:
            return "unknown error code";
    }
}
// Macro for error checking
#if 1
#define checkErr(err, name) \
    if (err != CL_SUCCESS) { \
        std::cerr << "ERROR: " << name << " (" << (err) << ")" << " [file " << __FILE__ << ", line " << __LINE__ << "]: "; \
        std::cerr << getOpenCLErrorCodeStr(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#else
inline void checkErr(cl_int err, std::string name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif


// Select platform and device for execution
void hipaccInitPlatformsAndDevices(cl_device_type dev_type, cl_platform_name platform_name=ALL) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024], pd2Buffer[1024];
    int platform_number = -1, device_number = -1;
    cl_uint num_platforms, num_devices, num_devices_type;
    cl_int err = CL_SUCCESS;

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
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkErr(err, "clGetPlatformIDs()");

    std::cerr << "Number of available Platforms: " << num_platforms << std::endl;
    if (num_platforms == 0) {
        exit(EXIT_FAILURE);
    } else {
        std::vector<cl_platform_id> platforms(num_platforms);
        std::vector<cl_platform_name> platform_names(num_platforms);

        err = clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
        checkErr(err, "clGetPlatformIDs()");

        // Get platform info for each platform
        for (size_t i=0; i<platforms.size(); ++i) {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &pnBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, &pvBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, &pv2Buffer, NULL);
            checkErr(err, "clGetPlatformInfo()");

            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            err |= clGetDeviceIDs(platforms[i], dev_type, 0, NULL, &num_devices_type);

            // Check if the requested device type was not found for this platform
            if (err != CL_DEVICE_NOT_FOUND) checkErr(err, "clGetDeviceIDs()");

            // Get platform name
            if (strncmp(pnBuffer, "AMD", 3) == 0) platform_names[i] = AMD;
            else if (strncmp(pnBuffer, "Apple", 3) == 0) platform_names[i] = APPLE;
            else if (strncmp(pnBuffer, "ARM", 3) == 0) platform_names[i] = ARM;
            else if (strncmp(pnBuffer, "Intel", 3) == 0) platform_names[i] = INTEL;
            else if (strncmp(pnBuffer, "NVIDIA", 3) == 0) platform_names[i] = NVIDIA;
            else platform_names[i] = ALL;

            // Use first platform supporting desired device type
            if (platform_number==-1 && num_devices_type > 0 && (platform_names[i] & platform_name)) {
                std::cerr << "  [*] Platform Name: " << pnBuffer << std::endl;
                platform_number = i;
            } else {
                std::cerr << "  [ ] Platform Name: " << pnBuffer << std::endl;
            }
            std::cerr << "      Platform Vendor: " << pvBuffer << std::endl;
            std::cerr << "      Platform Version: " << pv2Buffer << std::endl;

            std::vector<cl_device_id> devices(num_devices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            // Get device info for each device
            for (size_t j=0; j<devices.size(); ++j) {
                cl_device_type this_dev_type;
                cl_uint device_vendor_id;

                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pnBuffer), &pnBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(pvBuffer), &pvBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof(device_vendor_id), &device_vendor_id, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(this_dev_type), &this_dev_type, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(pdBuffer), &pdBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(pd2Buffer), &pd2Buffer, NULL);
                checkErr(err, "clGetDeviceInfo()");

                // Use first device of desired type
                if (platform_number == (int)i && device_number == -1 && (this_dev_type & dev_type)) {
                    std::cerr << "      [*] ";
                    Ctx.add_device(devices[j]);
                    device_number = j;
                } else {
                    std::cerr << "      [ ] ";
                }
                std::cerr << "Device Name: " << pnBuffer << " (";
                if (this_dev_type & CL_DEVICE_TYPE_CPU) std::cerr << "CL_DEVICE_TYPE_CPU";
                if (this_dev_type & CL_DEVICE_TYPE_GPU) std::cerr << "CL_DEVICE_TYPE_GPU";
                if (this_dev_type & CL_DEVICE_TYPE_ACCELERATOR) std::cerr << "CL_DEVICE_TYPE_ACCELERATOR";
                #ifdef CL_VERSION_1_2
                if (this_dev_type & CL_DEVICE_TYPE_CUSTOM) std::cerr << "CL_DEVICE_TYPE_CUSTOM";
                #endif
                if (this_dev_type & CL_DEVICE_TYPE_DEFAULT) std::cerr << "|CL_DEVICE_TYPE_DEFAULT";
                std::cerr << ")" << std::endl;
                std::cerr << "          Device Vendor: " << pvBuffer << " (ID: " << device_vendor_id << ")" << std::endl;
                std::cerr << "          Device OpenCL Version: " << pdBuffer << std::endl;
                std::cerr << "          Device Driver Version: " << pd2Buffer << std::endl;

                // Store all matching devices in a separate array
                if (platform_number == (int)i && (this_dev_type & dev_type)) {
                    Ctx.add_device_all(devices[j]);
                }
            }
        }

        if (platform_number == -1) {
            std::cerr << "No suitable OpenCL platform available, aborting ..." << std::endl;
            exit(EXIT_FAILURE);
        }

        Ctx.add_platform(platforms[platform_number], platform_names[platform_number]);
    }
}


// Get a vector with all devices
std::vector<cl_device_id> hipaccGetAllDevices() {
    HipaccContext &Ctx = HipaccContext::getInstance();

    return Ctx.get_devices_all();
}


// Create context and command queue for each device
void hipaccCreateContextsAndCommandQueues(bool all_devies=false) {
    cl_int err = CL_SUCCESS;
    cl_context context;
    cl_command_queue command_queue;
    HipaccContext &Ctx = HipaccContext::getInstance();

    std::vector<cl_platform_id> platforms = Ctx.get_platforms();
    std::vector<cl_device_id> devices = all_devies?Ctx.get_devices_all():Ctx.get_devices();

    // Create context
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0 };
    context = clCreateContext(cprops, devices.size(), devices.data(), NULL, NULL, &err);
    checkErr(err, "clCreateContext()");

    Ctx.add_context(context);

    // Create command queues
    for (size_t i=0; i<devices.size(); ++i) {
        command_queue = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);
        checkErr(err, "clCreateCommandQueue()");

        Ctx.add_command_queue(command_queue);
    }
}


// Get binary from OpenCL program and dump it to stderr
void hipaccDumpBinary(cl_program program, cl_device_id device) {
    cl_int err = CL_SUCCESS;
    cl_uint num_devices;

    // Get the number of devices associated with the program
    err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // Get the associated device ids
    std::vector<cl_device_id> devices(num_devices);
    err |= clGetProgramInfo(program, CL_PROGRAM_DEVICES, devices.size() * sizeof(cl_device_id), devices.data(), 0);

    // Get the sizes of the binaries
    std::vector<size_t> binary_sizes(num_devices);
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, binary_sizes.size() * sizeof(size_t), binary_sizes.data(), NULL);

    // Get the binaries
    std::vector<unsigned char *> binaries(num_devices);
    for (size_t i=0; i<binaries.size(); ++i) {
        binaries[i] = new unsigned char[binary_sizes[i]];
    }
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARIES,  sizeof(unsigned char *)*binaries.size(), binaries.data(), NULL);
    checkErr(err, "clGetProgramInfo()");

    for (size_t i=0; i<devices.size(); ++i) {
        if (devices[i] == device) {
            std::cerr << "OpenCL binary : " << std::endl;
            // binary can contain any character, emit char by char
            for (size_t n=0; n<binary_sizes[i]; ++n) {
                std::cerr << binaries[i][n];
            }
            std::cerr << std::endl;
        }
    }

    for (size_t i=0; i<num_devices; i++) {
        delete[] binaries[i];
    }
}


// Load OpenCL source file, build program, and create kernel
cl_kernel hipaccBuildProgramAndKernel(std::string file_name, std::string kernel_name, bool print_progress=true, bool dump_binary=false, bool print_log=false, std::string build_options=std::string(), std::string build_includes=std::string()) {
    cl_int err = CL_SUCCESS;
    cl_program program;
    cl_kernel kernel;
    HipaccContext &Ctx = HipaccContext::getInstance();

    std::ifstream srcFile(file_name.c_str());
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open OpenCL source file '" << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string clString(std::istreambuf_iterator<char>(srcFile),
            (std::istreambuf_iterator<char>()));

    const size_t length = clString.length();
    const char *c_str = clString.c_str();

    if (print_progress) std::cerr << "<HIPACC:> Compiling '" << kernel_name << "' .";
    program = clCreateProgramWithSource(Ctx.get_contexts()[0], 1, (const char **)&c_str, &length, &err);
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
                build_options = "-cl-single-precision-constant -cl-denorms-are-zero -cl-nv-verbose";
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
    if (print_progress) std::cerr << ".";

    cl_build_status build_status;
    clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL);

    if (build_status == CL_BUILD_ERROR || err != CL_SUCCESS || print_log) {
        // determine the size of the options and log
        size_t log_size, options_size;
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // allocate memory for the options and log
        char *program_build_options = new char[options_size];
        char *program_build_log = new char[log_size];

        // get the options and log
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_OPTIONS, options_size, program_build_options, NULL);
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_LOG, log_size, program_build_log, NULL);
        if (print_progress) {
            if (err != CL_SUCCESS) std::cerr << ". failed!" << std::endl;
            else std::cerr << ".";
        }
        std::cerr << std::endl
                  << "<HIPACC:> OpenCL build options : " << std::endl
                  << program_build_options << std::endl
                  << "<HIPACC:> OpenCL build log : " << std::endl
                  << program_build_log << std::endl;

        // free memory for options and log
        delete[] program_build_options;
        delete[] program_build_log;
    }
    checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

    if (dump_binary) hipaccDumpBinary(program, Ctx.get_devices()[0]);

    kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, "clCreateKernel()");
    if (print_progress) std::cerr << ". done" << std::endl;

    // release program
    err = clReleaseProgram(program);
    checkErr(err, "clReleaseProgram()");

    return kernel;
}


// Allocate memory with alignment specified
template<typename T>
HipaccImage hipaccCreateBuffer(T *host_mem, int width, int height, int alignment) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (host_mem) {
        flags |= CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
    }
    // alignment has to be a multiple of sizeof(T)
    alignment = (int)ceilf((float)alignment/sizeof(T)) * sizeof(T);
    // compute stride
    int stride = (int)ceilf((float)(width)/(alignment/sizeof(T))) * (alignment/sizeof(T));
    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*stride*height, host_mem, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccImage img = HipaccImage(width, height, stride, alignment, sizeof(T), (void *)buffer);
    Ctx.add_image(img);

    return img;
}


// Allocate memory without any alignment considerations
template<typename T>
HipaccImage hipaccCreateBuffer(T *host_mem, int width, int height) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (host_mem) {
        flags |= CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
    }
    int stride = width;
    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*width*height, host_mem, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccImage img = HipaccImage(width, height, stride, 0, sizeof(T), (void *)buffer);
    Ctx.add_image(img);

    return img;
}


// Allocate constant buffer
template<typename T>
HipaccImage hipaccCreateBufferConstant(int width, int height) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_ONLY;
    HipaccContext &Ctx = HipaccContext::getInstance();

    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*width*height, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccImage img = HipaccImage(width, height, width, 0, sizeof(T), (void *)buffer);
    Ctx.add_image(img);

    return img;
}


// Allocate image - no alignment can be specified
template<typename T>
HipaccImage hipaccCreateImage(T *host_mem, int width, int height,
        cl_channel_type channel_type, cl_channel_order channel_order) {
    cl_int err = CL_SUCCESS;
    cl_mem image;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_image_format image_format;
    HipaccContext &Ctx = HipaccContext::getInstance();

    image_format.image_channel_order = channel_order;
    image_format.image_channel_data_type = channel_type;

    if (host_mem) {
        flags |= CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
    }
    int stride = width;
    #ifdef CL_VERSION_1_2
    cl_image_desc image_desc;
    memset(&image_desc, '\0', sizeof(cl_image_desc));

    // CL_MEM_OBJECT_IMAGE1D
    // CL_MEM_OBJECT_IMAGE1D_BUFFER
    // CL_MEM_OBJECT_IMAGE2D
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;

    image = clCreateImage(Ctx.get_contexts()[0], flags, &image_format, &image_desc, host_mem, &err);
    checkErr(err, "clCreateImage()");
    #else
    image = clCreateImage2D(Ctx.get_contexts()[0], flags, &image_format, width, height, 0, host_mem, &err);
    checkErr(err, "clCreateImage2D()");
    #endif

    HipaccImage img = HipaccImage(width, height, stride, 0, sizeof(T), (void *)image, Array2D);
    Ctx.add_image(img);

    return img;
}
template<typename T>
HipaccImage hipaccCreateImage(T *host_mem, int width, int height);
#define CREATE_IMAGE(DATA_TYPE, CHANNEL_TYPE, CHANNEL_ORDER) \
template <> \
HipaccImage hipaccCreateImage<DATA_TYPE>(DATA_TYPE *host_mem, int width, int height) { \
    return hipaccCreateImage(host_mem, width, height, CHANNEL_TYPE, CHANNEL_ORDER); \
}
CREATE_IMAGE(char,                  CL_SIGNED_INT8,     CL_R)
CREATE_IMAGE(short int,             CL_SIGNED_INT16,    CL_R)
CREATE_IMAGE(int,                   CL_SIGNED_INT32,    CL_R)
CREATE_IMAGE(unsigned char,         CL_UNSIGNED_INT8,   CL_R)
CREATE_IMAGE(unsigned short int,    CL_UNSIGNED_INT16,  CL_R)
CREATE_IMAGE(unsigned int,          CL_UNSIGNED_INT32,  CL_R)
CREATE_IMAGE(float,                 CL_FLOAT,           CL_R)
CREATE_IMAGE(char4,                 CL_SIGNED_INT8,     CL_RGBA)
CREATE_IMAGE(short4,                CL_SIGNED_INT16,    CL_RGBA)
CREATE_IMAGE(int4,                  CL_SIGNED_INT32,    CL_RGBA)
CREATE_IMAGE(uchar4,                CL_UNSIGNED_INT8,   CL_RGBA)
CREATE_IMAGE(ushort4,               CL_UNSIGNED_INT16,  CL_RGBA)
CREATE_IMAGE(uint4,                 CL_UNSIGNED_INT32,  CL_RGBA)
CREATE_IMAGE(float4,                CL_FLOAT,           CL_RGBA)


// Create sampler object
cl_sampler hipaccCreateSampler(cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode) {
    cl_int err = CL_SUCCESS;
    cl_sampler sampler;
    HipaccContext &Ctx = HipaccContext::getInstance();

    sampler = clCreateSampler(Ctx.get_contexts()[0], normalized_coords, addressing_mode, filter_mode, &err);
    checkErr(err, "clCreateSampler()");

    return sampler;
}


// Release buffer or image
void hipaccReleaseMemory(HipaccImage &img) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    err = clReleaseMemObject((cl_mem)img.mem);
    checkErr(err, "clReleaseMemObject()");

    Ctx.del_image(img);
}


// Write to memory
template<typename T>
void hipaccWriteMemory(HipaccImage &img, T *host_mem, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    std::copy(host_mem, host_mem + img.width*img.height, (T*)img.host);
    if (img.mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { (size_t)img.width, (size_t)img.height, 1 };
        // no stride supported for images in OpenCL
        const size_t input_row_pitch = img.width*sizeof(T);
        const size_t input_slice_pitch = 0;

        err = clEnqueueWriteImage(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, origin, region, input_row_pitch, input_slice_pitch, host_mem, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueWriteImage()");
    } else {
        size_t width = img.width;
        size_t height = img.height;
        size_t stride = img.stride;

        if (stride > width) {
            for (size_t i=0; i<height; ++i) {
                err |= clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &host_mem[i*width], 0, NULL, NULL);
            }
        } else {
            err = clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, 0, sizeof(T)*width*height, host_mem, 0, NULL, NULL);
        }
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueWriteBuffer()");
    }
}


// Read from memory
template<typename T>
T *hipaccReadMemory(HipaccImage &img, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (img.mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { (size_t)img.width, (size_t)img.height, 1 };
        // no stride supported for images in OpenCL
        const size_t row_pitch = img.width*sizeof(T);
        const size_t slice_pitch = 0;

        err = clEnqueueReadImage(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, origin, region, row_pitch, slice_pitch, (T*)img.host, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueReadImage()");
    } else {
        size_t width = img.width;
        size_t height = img.height;
        size_t stride = img.stride;

        if (stride > width) {
            for (size_t i=0; i<height; ++i) {
                err |= clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &((T*)img.host)[i*width], 0, NULL, NULL);
            }
        } else {
            err = clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], (cl_mem)img.mem, CL_FALSE, 0, sizeof(T)*width*height, (T*)img.host, 0, NULL, NULL);
        }
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueReadBuffer()");
    }

    return (T*)img.host;
}


// Infer non-const Domain from non-const Mask
template<typename T>
void hipaccWriteDomainFromMask(HipaccImage &dom, T* host_mem) {
  int size = dom.width * dom.height;
  uchar *dom_mem = new uchar[size];

  for (int i = 0; i < size; ++i) {
    dom_mem[i] = (host_mem[i] == T(0) ? 0 : 1);
  }

  hipaccWriteMemory(dom, dom_mem);

  delete[] dom_mem;
}


// Copy between memory
void hipaccCopyMemory(HipaccImage &src, HipaccImage &dst, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    assert(src.width == dst.width && src.height == dst.height && src.pixel_size == dst.pixel_size && "Invalid CopyBuffer or CopyImage!");

    if (src.mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { (size_t)src.width, (size_t)src.height, 1 };

        err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device], (cl_mem)src.mem, (cl_mem)dst.mem, origin, origin, region, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyImage()");
    } else {
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src.mem, (cl_mem)dst.mem, 0, 0, src.width*src.height*src.pixel_size, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBuffer()");
    }
}


// Copy from memory region to memory region
void hipaccCopyMemoryRegion(HipaccAccessor src, HipaccAccessor dst, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (src.img.mem_type >= Array2D) {
        const size_t dst_origin[] = { (size_t)dst.offset_x, (size_t)dst.offset_y, 0 };
        const size_t src_origin[] = { (size_t)src.offset_x, (size_t)src.offset_y, 0 };
        const size_t region[] = { (size_t)dst.width, (size_t)dst.height, 1 };

        err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device],
                (cl_mem)src.img.mem, (cl_mem)dst.img.mem, src_origin, dst_origin,
                region, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyImage()");
    } else {
        const size_t dst_origin[] = { (size_t)dst.offset_x*dst.img.pixel_size, (size_t)dst.offset_y, 0 };
        const size_t src_origin[] = { (size_t)src.offset_x*src.img.pixel_size, (size_t)src.offset_y, 0 };
        const size_t region[] = { (size_t)dst.width*dst.img.pixel_size, (size_t)dst.height, 1 };

        err = clEnqueueCopyBufferRect(Ctx.get_command_queues()[num_device],
                (cl_mem)src.img.mem, (cl_mem)dst.img.mem, src_origin, dst_origin,
                region, src.img.stride*src.img.pixel_size, 0,
                dst.img.stride*dst.img.pixel_size, 0, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBufferRect()");
    }
}


// Copy between buffers and return time
double hipaccCopyBufferBenchmark(HipaccImage &src, HipaccImage &dst, int num_device=0, bool print_timing=false) {
    cl_int err = CL_SUCCESS;
    cl_ulong end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();

    assert(src.width == dst.width && src.height == dst.height && src.pixel_size == dst.pixel_size && "Invalid CopyBuffer!");

    float timing=FLT_MAX;
    #ifdef EVENT_TIMING
    cl_event event;
    #else
    std::vector<float> times;
    times.reserve(HIPACC_NUM_ITERATIONS);
    #endif
    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        #ifdef EVENT_TIMING
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src.mem, (cl_mem)dst.mem, 0, 0, src.width*src.height*src.pixel_size, 0, NULL, &event);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBuffer()");

        err = clWaitForEvents(1, &event);
        checkErr(err, "clWaitForEvents()");

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");
        start *= 1e-3;
        end *= 1e-3;
        #else
        clFinish(Ctx.get_command_queues()[num_device]);
        start = getMicroTime();
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src.mem, (cl_mem)dst.mem, 0, 0, src.width*src.height*src.pixel_size, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        end = getMicroTime();
        checkErr(err, "clEnqueueCopyBuffer()");
        #endif

        if (print_timing) {
            std::cerr << "<HIPACC:> Copy timing (" << (src.width*src.height*src.pixel_size) / (float)(1 << 20) << " MB): " << (end-start)*1.0e-3f << "(ms)" << std::endl;
            std::cerr << "          Bandwidth: " << 2.0f * (double)(src.width*src.height*src.pixel_size) / ((end-start)*1.0e-6f * (float)(1 << 30)) << " GB/s" << std::endl;
        }
        #ifdef EVENT_TIMING
        if ((end-start) < timing) timing = (end-start);
        #else
        times.push_back(end-start);
        #endif
    }

    // return time in ms
    #ifdef EVENT_TIMING
    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
    #else
    std::sort(times.begin(), times.end());
    timing = times.at(HIPACC_NUM_ITERATIONS/2);
    #endif

    return timing*1.0e-3f;
}


// Set a single argument of a kernel
template<typename T>
void hipaccSetKernelArg(cl_kernel kernel, unsigned int num, size_t size, T* param) {
    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(kernel, num, size, param);
    checkErr(err, "clSetKernelArg()");
}


// Enqueue and launch kernel
void hipaccEnqueueKernel(cl_kernel kernel, size_t *global_work_size, size_t *local_work_size, bool print_timing=true) {
    cl_int err = CL_SUCCESS;
    #ifdef EVENT_TIMING
    cl_event event;
    cl_ulong end, start;
    #else
    long end, start;
    #endif
    HipaccContext &Ctx = HipaccContext::getInstance();

    #ifdef EVENT_TIMING
    err = clEnqueueNDRangeKernel(Ctx.get_command_queues()[0], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueNDRangeKernel()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start *= 1e-3;
    end *= 1e-3;

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
    #else
    clFinish(Ctx.get_command_queues()[0]);
    start = getMicroTime();
    err = clEnqueueNDRangeKernel(Ctx.get_command_queues()[0], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    end = getMicroTime();
    checkErr(err, "clEnqueueNDRangeKernel()");
    #endif

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << local_work_size[0]*local_work_size[1] << ": " << local_work_size[0] << "x" << local_work_size[1] << "): " << (end-start)*1.0e-3f << "(ms)" << std::endl;
    }
    total_time += (end-start)*1.0e-3f;
    last_gpu_timing = (end-start)*1.0e-3f;
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, HipaccAccessor
        &acc, unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T result;       // host result

    // first step: reduce image (region) into linear memory
    size_t local_work_size[2];
    local_work_size[0] = max_threads;
    local_work_size[1] = 1;
    size_t global_work_size[2];
    global_work_size[0] = (int)ceilf((float)(acc.img.width)/(local_work_size[0]*2))*local_work_size[0];
    global_work_size[1] = (int)ceilf((float)(acc.height)/(local_work_size[1]*pixels_per_thread))*local_work_size[1];

    unsigned int num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    hipaccSetKernelArg(kernel2D, 0, sizeof(cl_mem), &acc.img.mem);
    hipaccSetKernelArg(kernel2D, 1, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel2D, 2, sizeof(unsigned int), &acc.img.width);
    hipaccSetKernelArg(kernel2D, 3, sizeof(unsigned int), &acc.img.height);
    hipaccSetKernelArg(kernel2D, 4, sizeof(unsigned int), &acc.img.stride);
    // check if the reduction is applied to the whole image
    if ((acc.offset_x || acc.offset_y) &&
        (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
        hipaccSetKernelArg(kernel2D, 5, sizeof(unsigned int), &acc.offset_x);
        hipaccSetKernelArg(kernel2D, 6, sizeof(unsigned int), &acc.offset_y);
        hipaccSetKernelArg(kernel2D, 7, sizeof(unsigned int), &acc.width);
        hipaccSetKernelArg(kernel2D, 8, sizeof(unsigned int), &acc.height);

        // reduce iteration space by idle blocks
        unsigned int idle_left = acc.offset_x / local_work_size[0];
        unsigned int idle_right = (acc.img.width - (acc.offset_x+acc.width)) / local_work_size[0];
        global_work_size[0] = (int)ceilf((float)
                (acc.img.width - (idle_left + idle_right) * local_work_size[0])
                / (local_work_size[0]*2))*local_work_size[0];
        // update number of blocks
        num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

        // set last argument: block offset in pixels
        idle_left *= local_work_size[0];
        hipaccSetKernelArg(kernel2D, 9, sizeof(unsigned int), &idle_left);
    }

    hipaccEnqueueKernel(kernel2D, global_work_size, local_work_size);


    // second step: reduce partial blocks on GPU
    // this is done in one shot, so no additional memory is required, i.e. the
    // same array can be used for the input and output array
    // block.x is fixed, either max_threads or power of two
    local_work_size[0] = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2)
        : max_threads;
    global_work_size[0] = local_work_size[0];
    global_work_size[1] = 1;
    // calculate the number of pixels reduced per thread
    int num_steps = (num_blocks + (local_work_size[0] - 1)) / (local_work_size[0]);

    hipaccSetKernelArg(kernel1D, 0, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel1D, 1, sizeof(cl_mem), &output);
    hipaccSetKernelArg(kernel1D, 2, sizeof(unsigned int), &num_blocks);
    hipaccSetKernelArg(kernel1D, 3, sizeof(unsigned int), &num_steps);

    hipaccEnqueueKernel(kernel1D, global_work_size, local_work_size);

    // get reduced value
    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T), &result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}
// Perform global reduction using memory fence operations and return result
template<typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, HipaccImage &img,
        unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccAccessor acc(img);
    return hipaccApplyReduction<T>(kernel2D, kernel1D, acc, max_threads,
            pixels_per_thread);
}


// Perform exploration of global reduction and return result
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
        std::string kernel1D, HipaccAccessor &acc, unsigned int max_threads,
        unsigned int pixels_per_thread) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T result;       // host result

    unsigned int num_blocks = (int)ceilf((float)(acc.img.width)/(max_threads*2))*acc.height;
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    std::cerr << "<HIPACC:> Exploring pixels per thread for '" << kernel2D << ", " << kernel1D << "'" << std::endl;

    float opt_time = FLT_MAX;
    int opt_ppt = 1;
    for (size_t ppt=1; ppt<=acc.height; ++ppt) {
        std::stringstream num_ppt_ss;
        std::stringstream num_bs_ss;
        num_ppt_ss << ppt;
        num_bs_ss << max_threads;

        std::string compile_options = "-D PPT=" + num_ppt_ss.str() + " -D BS=" + num_bs_ss.str() + " -I./include ";
        compile_options += "-D BSX_EXPLORE=64 -D BSY_EXPLORE=1 ";
        cl_kernel exploreReduction2D = hipaccBuildProgramAndKernel(filename, kernel2D, false, false, false, compile_options);
        cl_kernel exploreReduction1D = hipaccBuildProgramAndKernel(filename, kernel1D, false, false, false, compile_options);

        float timing=FLT_MAX;
        #ifndef EVENT_TIMING
        std::vector<float> times;
        times.reserve(HIPACC_NUM_ITERATIONS);
        #endif
        for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
            // first step: reduce image (region) into linear memory
            size_t local_work_size[2];
            local_work_size[0] = max_threads;
            local_work_size[1] = 1;
            size_t global_work_size[2];
            global_work_size[0] = (int)ceilf((float)(acc.img.width)/(local_work_size[0]*2))*local_work_size[0];
            global_work_size[1] = (int)ceilf((float)(acc.height)/(local_work_size[1]*ppt))*local_work_size[1];
            num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

            // start timing
            total_time = 0.0f;

            hipaccSetKernelArg(exploreReduction2D, 0, sizeof(cl_mem), &acc.img.mem);
            hipaccSetKernelArg(exploreReduction2D, 1, sizeof(cl_mem), &output);
            hipaccSetKernelArg(exploreReduction2D, 2, sizeof(unsigned int), &acc.img.width);
            hipaccSetKernelArg(exploreReduction2D, 3, sizeof(unsigned int), &acc.img.height);
            hipaccSetKernelArg(exploreReduction2D, 4, sizeof(unsigned int), &acc.img.stride);
            // check if the reduction is applied to the whole image
            if ((acc.offset_x || acc.offset_y) &&
                (acc.width!=acc.img.width || acc.height!=acc.img.height)) {
                hipaccSetKernelArg(exploreReduction2D, 5, sizeof(unsigned int), &acc.offset_x);
                hipaccSetKernelArg(exploreReduction2D, 6, sizeof(unsigned int), &acc.offset_y);
                hipaccSetKernelArg(exploreReduction2D, 7, sizeof(unsigned int), &acc.width);
                hipaccSetKernelArg(exploreReduction2D, 8, sizeof(unsigned int), &acc.height);

                // reduce iteration space by idle blocks
                unsigned int idle_left = acc.offset_x / local_work_size[0];
                unsigned int idle_right = (acc.img.width - (acc.offset_x+acc.width)) / local_work_size[0];
                global_work_size[0] = (int)ceilf((float)
                        (acc.img.width - (idle_left + idle_right) * local_work_size[0])
                        / (local_work_size[0]*2))*local_work_size[0];
                // update number of blocks
                num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

                // set last argument: block offset in pixels
                idle_left *= local_work_size[0];
                hipaccSetKernelArg(exploreReduction2D, 9, sizeof(unsigned int), &idle_left);
            }

            hipaccEnqueueKernel(exploreReduction2D, global_work_size, local_work_size, false);


            // second step: reduce partial blocks on GPU
            global_work_size[1] = 1;
            while (num_blocks > 1) {
                local_work_size[0] = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) :
                    max_threads;
                global_work_size[0] = (int)ceilf((float)(num_blocks)/(local_work_size[0]*ppt))*local_work_size[0];

                hipaccSetKernelArg(exploreReduction1D, 0, sizeof(cl_mem), &output);
                hipaccSetKernelArg(exploreReduction1D, 1, sizeof(cl_mem), &output);
                hipaccSetKernelArg(exploreReduction1D, 2, sizeof(unsigned int), &num_blocks);
                hipaccSetKernelArg(exploreReduction1D, 3, sizeof(unsigned int), &ppt);

                hipaccEnqueueKernel(exploreReduction1D, global_work_size, local_work_size, false);

                num_blocks = global_work_size[0]/local_work_size[0];
            }
            // stop timing
            #ifdef EVENT_TIMING
            if (total_time < timing) timing = total_time;
            #else
            times.push_back(total_time);
            #endif
        }
        #ifndef EVENT_TIMING
        std::sort(times.begin(), times.end());
        timing = times.at(HIPACC_NUM_ITERATIONS/2);
        #endif
        if (timing < opt_time) {
            opt_time = timing;
            opt_ppt = ppt;
        }

        // print timing
        std::cerr << "<HIPACC:> PPT: " << std::setw(4) << std::right << ppt
                  << ", " << std::setw(8) << std::fixed << std::setprecision(4)
                  << timing << " ms" << std::endl;

        // release kernels
        err = clReleaseKernel(exploreReduction2D);
        checkErr(err, "clReleaseKernel()");
        err = clReleaseKernel(exploreReduction1D);
        checkErr(err, "clReleaseKernel()");
    }
    std::cerr << "<HIPACC:> Best unroll factor for reduction kernel '"
              << kernel2D << "/" << kernel1D << "': "
              << opt_ppt << ": " << opt_time << " ms" << std::endl;

    // get reduced value
    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T), &result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}
template<typename T>
T hipaccApplyReductionExploration(std::string filename, std::string kernel2D,
        std::string kernel1D, HipaccImage &img, unsigned int max_threads,
        unsigned int pixels_per_thread) {
    HipaccAccessor acc(img);
    return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, acc,
            max_threads, pixels_per_thread);
}


// Benchmark timing for a kernel call
void hipaccEnqueueKernelBenchmark(cl_kernel kernel, std::vector<std::pair<size_t, void *> > args, size_t *global_work_size, size_t *local_work_size, bool print_timing=true) {
    float timing=FLT_MAX;
    #ifndef EVENT_TIMING
    std::vector<float> times;
    times.reserve(HIPACC_NUM_ITERATIONS);
    #endif

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        // set kernel arguments
        for (size_t j=0; j<args.size(); ++j) {
            hipaccSetKernelArg(kernel, j, args[j].first, args[j].second);
        }

        // launch kernel
        hipaccEnqueueKernel(kernel, global_work_size, local_work_size, print_timing);
        #ifdef EVENT_TIMING
        if (last_gpu_timing < timing) timing = last_gpu_timing;
        #else
        times.push_back(last_gpu_timing);
        #endif
    }

    #ifndef EVENT_TIMING
    std::sort(times.begin(), times.end());
    timing = times.at(HIPACC_NUM_ITERATIONS/2);
    #endif
    last_gpu_timing = timing;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark ("
                  << local_work_size[0]*local_work_size[1] << ": "
                  << local_work_size[0] << "x" << local_work_size[1] << "): "
                  << timing << "(ms)" << std::endl;
    }
}


// Perform configuration exploration for a kernel call
void hipaccKernelExploration(std::string filename, std::string kernel,
        std::vector<std::pair<size_t, void *> > args,
        std::vector<hipacc_smem_info> smems, hipacc_launch_info &info, int
        warp_size, int max_threads_per_block, int max_threads_for_kernel, int
        max_smem_per_block, int heu_tx, int heu_ty) {
    int opt_tx=warp_size, opt_ty=1;
    float opt_time = FLT_MAX;

    std::cerr << "<HIPACC:> Exploring configurations for kernel '" << kernel
              << "': configuration provided by heuristic " << heu_tx*heu_ty
              << " (" << heu_tx << "x" << heu_ty << "). " << std::endl;

    for (int tile_size_x=warp_size; tile_size_x<=max_threads_per_block; tile_size_x+=warp_size) {
        for (int tile_size_y=1; tile_size_y<=max_threads_per_block; ++tile_size_y) {
            // check if we exceed maximum number of threads
            if (tile_size_x*tile_size_y > max_threads_for_kernel) continue;

            // check if we exceed size of shared memory
            int used_smem = 0;
            for (size_t i=0; i<smems.size(); ++i) {
                used_smem += (tile_size_x + smems[i].size_x)*(tile_size_y + smems[i].size_y - 1) * smems[i].pixel_size;
            }
            if (used_smem >= max_smem_per_block) continue;
            if (used_smem && tile_size_x > warp_size) continue;

            std::stringstream num_threads_x_ss, num_threads_y_ss;
            num_threads_x_ss << tile_size_x;
            num_threads_y_ss << tile_size_y;

            // compile kernel
            std::string compile_options =
                " -D BSX_EXPLORE=" + num_threads_x_ss.str() +
                " -D BSY_EXPLORE=" + num_threads_y_ss.str() +
                " -I./include ";
            cl_kernel exploreKernel = hipaccBuildProgramAndKernel(filename, kernel, false, false, false, compile_options);


            size_t local_work_size[2];
            local_work_size[0] = tile_size_x;
            local_work_size[1] = tile_size_y;
            size_t global_work_size[2];
            hipaccCalcGridFromBlock(info, local_work_size, global_work_size);
            hipaccPrepareKernelLaunch(info, local_work_size);

            float timing=FLT_MAX;
            #ifndef EVENT_TIMING
            std::vector<float> times;
            times.reserve(HIPACC_NUM_ITERATIONS);
            #endif
            for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
                // set kernel arguments
                for (size_t j=0; j<args.size(); ++j) {
                    hipaccSetKernelArg(exploreKernel, j, args[j].first, args[j].second);
                }

                // start timing
                total_time = 0.0f;

                hipaccEnqueueKernel(exploreKernel, global_work_size, local_work_size, false);

                // stop timing
                #ifdef EVENT_TIMING
                if (total_time < timing) timing = total_time;
                #else
                times.push_back(timing);
                #endif
            }
            #ifndef EVENT_TIMING
            std::sort(times.begin(), times.end());
            timing = times.at(HIPACC_NUM_ITERATIONS/2);
            #endif
            if (timing < opt_time) {
                opt_time = timing;
                opt_tx = tile_size_x;
                opt_ty = tile_size_y;
            }

            // print timing
            std::cerr << "<HIPACC:> Kernel config: "
                      << std::setw(4) << std::right << tile_size_x << "x"
                      << std::setw(2) << std::left << tile_size_y
                      << std::setw(5-floor(log10f((float)(tile_size_x*tile_size_y))))
                      << std::right << "(" << tile_size_x*tile_size_y << "): "
                      << std::setw(8) << std::fixed << std::setprecision(4)
                      << timing << " ms" << std::endl;

            // release kernel
            cl_int err = clReleaseKernel(exploreKernel);
            checkErr(err, "clReleaseKernel()");
        }
    }
    std::cerr << "<HIPACC:> Best configurations for kernel '" << kernel << "': "
              << opt_tx*opt_ty << " (" << opt_tx << "x" << opt_ty << "): "
              << opt_time << " ms" << std::endl;
}


template<typename T>
HipaccImage hipaccCreatePyramidImage(HipaccImage &base, int width, int height) {
  switch (base.mem_type) {
    case Array2D:
      return hipaccCreateImage<T>(NULL, width, height);

    case Global:
      if (base.alignment > 0) {
        return hipaccCreateBuffer<T>(NULL, width, height, base.alignment);
      } else {
        return hipaccCreateBuffer<T>(NULL, width, height);
      }

    default:
      assert("Memory type is not supported for target OpenCL");
      return base;
  }
}

#endif  // __HIPACC_CL_HPP__

