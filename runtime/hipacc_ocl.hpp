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

#ifndef __HIPACC_OCL_HPP__
#define __HIPACC_OCL_HPP__

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <algorithm>

#include "hipacc_types.hpp"

#define HIPACC_NUM_ITERATIONS 10
#define GPU_TIMING

static float total_time = 0.0f;
static float last_gpu_timing = 0.0f;

enum hipaccBoundaryMode {
    BOUNDARY_UNDEFINED,
    BOUNDARY_CLAMP,
    BOUNDARY_REPEAT,
    BOUNDARY_MIRROR,
    BOUNDARY_CONSTANT
};

enum cl_platform_name {
    AMD     = 0x1,
    APPLE   = 0x2,
    ARM     = 0x4,
    INTEL   = 0x8,
    NVIDIA  = 0x10,
    ALL     = (AMD|APPLE|ARM|INTEL|NVIDIA)
};

typedef struct hipacc_smem_info {
    hipacc_smem_info(int size_x, int size_y, int pixel_size) :
        size_x(size_x), size_y(size_y), pixel_size(pixel_size) {}
    int size_x, size_y;
    int pixel_size;
} hipacc_smem_info;

typedef struct hipacc_launch_info {
    hipacc_launch_info(int size_x, int size_y, int is_width, int is_height, int
            offset_x, int offset_y, int pixels_per_thread, int simd_width) :
        size_x(size_x), size_y(size_y), is_width(is_width),
        is_height(is_height), offset_x(offset_x), offset_y(offset_y),
        pixels_per_thread(pixels_per_thread), simd_width(simd_width),
        bh_start_left(0), bh_start_right(0), bh_start_top(0),
        bh_start_bottom(0), bh_fall_back(0) {}
    int size_x, size_y;
    int is_width, is_height;
    int offset_x, offset_y;
    int pixels_per_thread, simd_width;
    // calculated later on
    int bh_start_left, bh_start_right;
    int bh_start_top, bh_start_bottom;
    int bh_fall_back;
} hipacc_launch_info;


void hipaccPrepareKernelLaunch(hipacc_launch_info &info, size_t *block) {
    // calculate block id of a) first block that requires no border handling
    // (left, top) and b) first block that requires border handling (right,
    // bottom)
    if (info.size_x > 0) {
        info.bh_start_left = (int)ceil((float)(info.offset_x + info.size_x) / (block[0] * info.simd_width));
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
        info.bh_start_top = (int)ceil((float)(info.size_y) / (info.pixels_per_thread * block[1]));
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
    grid[0] = (int)ceil((float)(info.is_width + info.offset_x)/(block[0]*info.simd_width)) * block[0];
    grid[1] = (int)ceil((float)(info.is_height)/(block[1]*info.pixels_per_thread)) * block[1];
}


long getNanoTime() {
    struct timespec ts;

    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_MONOTONIC, &ts);
    #endif
    return ts.tv_sec*1000000000LL + ts.tv_nsec;
}


const char *getOpenCLErrorCodeStr(int errorCode) {
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
inline void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif


class HipaccContext {
    public:
        typedef struct {
            int width;
            int height;
            int stride;
            int alignment;
            int pixel_size;
        } cl_dims;

    private:
        std::vector<cl_platform_id> platforms;
        std::vector<cl_platform_name> platform_names;
        std::vector<cl_device_id> devices, devices_all;
        std::vector<cl_context> contexts;
        std::vector<cl_command_queue> queues;
        std::vector<std::pair<cl_mem, cl_dims> > mems;

        HipaccContext() {};
        HipaccContext(HipaccContext const &);
        void operator=(HipaccContext const &);

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
        void add_memory(cl_mem id, cl_dims dim) { mems.push_back(std::make_pair(id, dim)); }
        void del_memory(cl_mem id) {
            unsigned int num=0;
            std::vector<std::pair<cl_mem, cl_dims> >::const_iterator i;
            for (i=mems.begin(); i!=mems.end(); ++i, ++num) {
                if (i->first == id) {
                    mems.erase(mems.begin() + num);
                    return;
                }
            }

            std::cerr << "ERROR: Unknown cl_mem requested: " << id << std::endl;
            exit(EXIT_FAILURE);
        }
        std::vector<cl_platform_id> get_platforms() { return platforms; }
        std::vector<cl_platform_name> get_platform_names() { return platform_names; }
        std::vector<cl_device_id> get_devices() { return devices; }
        std::vector<cl_device_id> get_devices_all() { return devices_all; }
        std::vector<cl_context> get_contexts() { return contexts; }
        std::vector<cl_command_queue> get_command_queues() { return queues; }
        cl_dims get_mem_dims(cl_mem id) {
            std::vector<std::pair<cl_mem, cl_dims> >::const_iterator i;
            for (i=mems.begin(); i!=mems.end(); ++i) {
                if (i->first == id) return i->second;
            }

            std::cerr << "ERROR: Unknown cl_mem requested: " << id << std::endl;
            exit(EXIT_FAILURE);
        }
};


// Get GPU timing of last executed Kernel in ms
float hipaccGetLastKernelTiming() {
    return last_gpu_timing;
}


// Select platform and device for execution
void hipaccInitPlatformsAndDevices(cl_device_type dev_type, cl_platform_name platform_name=ALL) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024];
    int platform_number = -1, device_number = -1;
    cl_uint num_platforms, num_devices, num_devices_type;
    cl_platform_id *platforms;
    cl_platform_name *platform_names;
    cl_device_id *devices;
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
        platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
        platform_names = (cl_platform_name *)malloc(num_platforms * sizeof(cl_platform_name));

        err = clGetPlatformIDs(num_platforms, platforms, NULL);
        checkErr(err, "clGetPlatformIDs()");

        // Get platform info for each platform
        for (unsigned int i=0; i<num_platforms; ++i) {
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
                std::cerr << "  [*] Name: " << pnBuffer << std::endl;
                std::cerr << "      Vendor: " << pvBuffer << std::endl;
                std::cerr << "      Version: " << pv2Buffer << std::endl;
                platform_number = i;
            } else {
                std::cerr << "  [ ] Name: " << pnBuffer << std::endl;
                std::cerr << "      Vendor: " << pvBuffer << std::endl;
                std::cerr << "      Version: " << pv2Buffer << std::endl;
            }

            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            // Get device info for each device
            for (unsigned int j=0; j<num_devices; ++j) {
                cl_device_type this_dev_type;

                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pnBuffer), &pnBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(pvBuffer), &pvBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(this_dev_type), &this_dev_type, NULL);
                checkErr(err, "clGetDeviceInfo()");

                // Use first device of desired type
                if (platform_number == (int)i && device_number == -1 && (this_dev_type == dev_type || dev_type == CL_DEVICE_TYPE_ALL)) {
                    std::cerr << "      [*] ";
                    Ctx.add_device(devices[j]);
                    device_number = j;
                } else {
                    std::cerr << "      [ ] ";
                }
                switch (this_dev_type) {
                    case CL_DEVICE_TYPE_CPU:
                        std::cerr << "Name: " << pnBuffer << " (CL_DEVICE_TYPE_CPU)" << std::endl;
                        break;
                    case CL_DEVICE_TYPE_GPU:
                        std::cerr << "Name: " << pnBuffer << " (CL_DEVICE_TYPE_GPU)" << std::endl;
                        break;
                }
                std::cerr << "          Vendor: " << pvBuffer << std::endl;

                // Store all devices in a separate array
                if (platform_number == (int)i) Ctx.add_device_all(devices[j]);
            }
            free(devices);
        }

        if (platform_number == -1) {
            std::cerr << "No suitable OpenCL platform available, aborting ..." << std::endl;
            exit(EXIT_FAILURE);
        }

        Ctx.add_platform(platforms[platform_number], platform_names[platform_number]);
        free(platforms);
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
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms.data()[0], 0 };
    context = clCreateContext(cprops, devices.size(), devices.data(), NULL, NULL, &err);
    checkErr(err, "clCreateContext()");

    Ctx.add_context(context);

    // Create command queues
    for (unsigned int i=0; i<devices.size(); i++) {
        command_queue = clCreateCommandQueue(context, devices.data()[i], CL_QUEUE_PROFILING_ENABLE, &err);
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
    cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    err |= clGetProgramInfo(program, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // Get the sizes of the binaries
    size_t *binary_sizes = (size_t *)malloc(num_devices * sizeof(size_t));
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // Get the binaries
    char **binary = (char **)malloc(num_devices * sizeof(char *));
    for (unsigned int i=0; i<num_devices; i++) {
        binary[i]= (char *)malloc(binary_sizes[i]);
    }
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARIES, 0, binary, NULL);
    checkErr(err, "clGetProgramInfo()");

    for (unsigned int i=0; i<num_devices; i++) {
        if (devices[i] == device) {
            std::cerr << "OpenCL binary : " << std::endl << binary[i] << std::endl;
        }
    }

    for (unsigned int i=0; i<num_devices; i++) {
        free(binary[i]);
    }
    free(binary_sizes);
}


// Load OpenCL source file, build program, and create kernel
cl_kernel hipaccBuildProgramAndKernel(std::string file_name, std::string kernel_name, bool print_progress=true, bool dump_binary=false, bool print_log=false, const char *build_options=(const char *)"", const char *build_includes=(const char *)"") {
    cl_int err = CL_SUCCESS;
    cl_program program;
    cl_kernel kernel;
    HipaccContext &Ctx = HipaccContext::getInstance();

    std::ifstream srcFile(file_name.c_str());
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open OpenCL source file '" << file_name.c_str() << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string clString(std::istreambuf_iterator<char>(srcFile),
            (std::istreambuf_iterator<char>()));

    const size_t length = clString.length();
    const char *c_str = clString.c_str();

    if (print_progress) std::cerr << "<HIPACC:> Compiling '" << kernel_name << "' .";
    program = clCreateProgramWithSource(Ctx.get_contexts()[0], 1, (const char **)&c_str, &length, &err);
    checkErr(err, "clCreateProgramWithSource()");

    std::string options = build_options;
    std::string includes = build_includes;
    cl_platform_name platform_name = Ctx.get_platform_names()[0];
    if (options == "") {
        switch (platform_name) {
            case AMD:
                options += "-cl-single-precision-constant -cl-denorms-are-zero";
                #ifdef CL_VERSION_1_2
                options += " -save-temps";
                #endif
                break;
            case NVIDIA:
                options += "-cl-single-precision-constant -cl-denorms-are-zero -cl-nv-verbose";
                break;
            case APPLE:
            case ARM:
            case INTEL:
            case ALL:
                options += "-cl-single-precision-constant -cl-denorms-are-zero";
                break;
        }
    }
    if (includes != "") {
        options += " " + includes;
    }
    err = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);
    if (print_progress) std::cerr << ".";

    if (err != CL_SUCCESS || print_log) {
        // determine the size of the options and log
        size_t log_size, options_size;
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // allocate memory for the options and log
        char *program_build_options = (char *)malloc(options_size);
        char *program_build_log = (char *)malloc(log_size);

        // get the options and log
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_OPTIONS, options_size, program_build_options, NULL);
        err |= clGetProgramBuildInfo(program, Ctx.get_devices()[0], CL_PROGRAM_BUILD_LOG, log_size, program_build_log, NULL);
        if (print_progress) {
            if (err != CL_SUCCESS) std::cerr << ". failed!" << std::endl;
            else std::cerr << "." << std::endl;
        }
        std::cerr << "<HIPACC:> OpenCL build options : " << std::endl << program_build_options << std::endl;
        std::cerr << "<HIPACC:> OpenCL build log : " << std::endl << program_build_log << std::endl;

        // free memory for options and log
        free(program_build_options);
        free(program_build_log);
    }
    checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

    if (dump_binary) hipaccDumpBinary(program, Ctx.get_devices()[0]);

    kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, "clCreateKernel()");
    if (print_progress) std::cerr << ". done" << std::endl;

    return kernel;
}


// Allocate memory with alignment specified
template<typename T>
cl_mem hipaccCreateBuffer(T *host_mem, int width, int height, int *stride, int alignment) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (host_mem) {
        flags |= CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
    }
    *stride = (int)ceil((float)(width)/(alignment/sizeof(T))) * (alignment/sizeof(T));
    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*(*stride)*height, host_mem, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccContext::cl_dims dim = { width, height, *stride, alignment, sizeof(T) };
    Ctx.add_memory(buffer, dim);

    return buffer;
}


// Allocate memory without any alignment considerations
template<typename T>
cl_mem hipaccCreateBuffer(T *host_mem, int width, int height, int *stride) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (host_mem) {
        flags |= CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR;
    }
    *stride = width;
    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*width*height, host_mem, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccContext::cl_dims dim = { width, height, width, 0, sizeof(T) };
    Ctx.add_memory(buffer, dim);

    return buffer;
}


// Release buffer
void hipaccReleaseBuffer(cl_mem buffer) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    err = clReleaseMemObject(buffer);
    checkErr(err, "clReleaseMemObject()");

    Ctx.del_memory(buffer);
}


// Allocate image - no alignment can be specified
template<typename T>
cl_mem hipaccCreateImage(T *host_mem, int width, int height, int *stride,
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
    *stride = width;
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

    HipaccContext::cl_dims dim = { width, height, *stride, 0, sizeof(T) };
    Ctx.add_memory(image, dim);

    return image;
}
template<typename T>
cl_mem hipaccCreateImage(T *host_mem, int width, int height, int *stride);
#define CREATE_IMAGE(DATA_TYPE, CHANNEL_TYPE, CHANNEL_ORDER) \
template <> \
cl_mem hipaccCreateImage<DATA_TYPE>(DATA_TYPE *host_mem, int width, int height, int *stride) { \
    return hipaccCreateImage(host_mem, width, height, stride, CHANNEL_TYPE, CHANNEL_ORDER); \
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


// Release image
void hipaccReleaseImage(cl_mem image) {
    hipaccReleaseBuffer(image);
}


// Create sampler object
cl_sampler hipaccCreateSampler(cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode) {
    cl_int err = CL_SUCCESS;
    cl_sampler sampler;
    HipaccContext &Ctx = HipaccContext::getInstance();

    sampler = clCreateSampler(Ctx.get_contexts()[0], normalized_coords, addressing_mode, filter_mode, &err);
    checkErr(err, "clCreateSampler()");

    return sampler;
}


// Allocate constant buffer
template<typename T>
cl_mem hipaccCreateBufferConstant(int width, int height) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer;
    cl_mem_flags flags = CL_MEM_READ_ONLY;
    HipaccContext &Ctx = HipaccContext::getInstance();

    buffer = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*width*height, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    HipaccContext::cl_dims dim = { width, height, width, 0, sizeof(T) };
    Ctx.add_memory(buffer, dim);

    return buffer;
}


// Write to buffer
template<typename T>
void hipaccWriteBuffer(cl_mem buffer, T *host_mem, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims dim = Ctx.get_mem_dims(buffer);

    int width = dim.width;
    int height = dim.height;
    int stride = dim.stride;

    if (stride > width) {
        for (int i=0; i<height; i++) {
            err |= clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], buffer, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &host_mem[i*width], 0, NULL, NULL);
        }
    } else {
        err = clEnqueueWriteBuffer(Ctx.get_command_queues()[num_device], buffer, CL_FALSE, 0, sizeof(T)*width*height, host_mem, 0, NULL, NULL);
    }
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueWriteBuffer()");
}


// Read from buffer
template<typename T>
void hipaccReadBuffer(T *host_mem, cl_mem buffer, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims dim = Ctx.get_mem_dims(buffer);

    int width = dim.width;
    int height = dim.height;
    int stride = dim.stride;

    if (stride > width) {
        for (int i=0; i<height; i++) {
            err |= clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], buffer, CL_FALSE, i*sizeof(T)*stride, sizeof(T)*width, &host_mem[i*width], 0, NULL, NULL);
        }
    } else {
        err = clEnqueueReadBuffer(Ctx.get_command_queues()[num_device], buffer, CL_FALSE, 0, sizeof(T)*width*height, host_mem, 0, NULL, NULL);
    }
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueReadBuffer()");
}


// Copy between buffers
void hipaccCopyBuffer(cl_mem src_buffer, cl_mem dst_buffer, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims src_dim = Ctx.get_mem_dims(src_buffer);
    HipaccContext::cl_dims dst_dim = Ctx.get_mem_dims(dst_buffer);

    assert(src_dim.width == dst_dim.width && src_dim.height == dst_dim.height && src_dim.pixel_size == dst_dim.pixel_size && "Invalid CopyBuffer!");

    err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], src_buffer, dst_buffer, 0, 0, src_dim.width*src_dim.height*src_dim.pixel_size, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyBuffer()");
}


// Copy from buffer region to buffer region
void hipaccCopyBufferRegion(cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset_x, size_t src_offset_y, size_t dst_offset_x, size_t dst_offset_y, size_t roi_width, size_t roi_height, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims src_dim = Ctx.get_mem_dims(src_buffer);
    HipaccContext::cl_dims dst_dim = Ctx.get_mem_dims(dst_buffer);

    size_t dst_stride = dst_dim.stride * dst_dim.pixel_size;
    size_t src_stride = src_dim.stride * src_dim.pixel_size;

    const size_t dst_origin[] = { dst_offset_x*dst_dim.pixel_size, dst_offset_y, 0 };
    const size_t src_origin[] = { src_offset_x*src_dim.pixel_size, src_offset_y, 0 };
    const size_t region[] = { roi_width*dst_dim.pixel_size, roi_height, 1 };

    err = clEnqueueCopyBufferRect(Ctx.get_command_queues()[num_device], src_buffer, dst_buffer, src_origin, dst_origin, region, src_stride, 0, dst_stride, 0, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyBufferRect()");
}


// Copy between buffers and return time
double hipaccCopyBufferBenchmark(cl_mem src_buffer, cl_mem dst_buffer, int num_device=0, bool print_timing=false) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims src_dim = Ctx.get_mem_dims(src_buffer);
    HipaccContext::cl_dims dst_dim = Ctx.get_mem_dims(dst_buffer);

    assert(src_dim.width == dst_dim.width && src_dim.height == dst_dim.height && src_dim.pixel_size == dst_dim.pixel_size && "Invalid CopyBuffer!");

    float timing=FLT_MAX;
    #ifndef GPU_TIMING
    std::vector<float> times;
    times.reserve(HIPACC_NUM_ITERATIONS);
    #endif
    for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], src_buffer, dst_buffer, 0, 0, src_dim.width*src_dim.height*src_dim.pixel_size, 0, NULL, &event);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBuffer()");

        err = clWaitForEvents(1, &event);
        checkErr(err, "clWaitForEvents()");

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");

        if (print_timing) {
            std::cerr << "<HIPACC:> Copy timing (" << (src_dim.width*src_dim.height*src_dim.pixel_size) / (float)(1 << 20) << " MB): " << (end-start)*1.0e-6f << "(ms)" << std::endl;
            std::cerr << "          Bandwidth: " << 2.0f * (double)(src_dim.width*src_dim.height*src_dim.pixel_size) / ((end-start)*1.0e-9f * (float)(1 << 30)) << " GB/s" << std::endl;
        }
        #ifdef GPU_TIMING
        if ((end-start) < timing) timing = (end-start);
        #else
        times.push_back(end-start);
        #endif
    }
    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");

    // return time in ms
    #ifndef GPU_TIMING
    std::sort(times.begin(), times.end());
    timing = times.at(HIPACC_NUM_ITERATIONS/2);
    #endif
    return timing*1.0e-6f;
}


// Copy between images
void hipaccCopyImage(cl_mem src_image, cl_mem dst_image, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims src_dim = Ctx.get_mem_dims(src_image);
    HipaccContext::cl_dims dst_dim = Ctx.get_mem_dims(dst_image);

    assert(src_dim.width == dst_dim.width && src_dim.height == dst_dim.height && src_dim.pixel_size == dst_dim.pixel_size && "Invalid CopyImage!");

    const size_t origin[] = { 0, 0, 0 };
    const size_t region[] = { (size_t)src_dim.width, (size_t)src_dim.height, 1 };

    err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device], src_image, dst_image, origin, origin, region, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyImage()");
}


// Copy from image region to image region
void hipaccCopyImageRegion(cl_mem src_image, cl_mem dst_image, size_t src_offset_x, size_t src_offset_y, size_t dst_offset_x, size_t dst_offset_y, size_t roi_width, size_t roi_height, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    const size_t dst_origin[] = { dst_offset_x, dst_offset_y, 0 };
    const size_t src_origin[] = { src_offset_x, src_offset_y, 0 };
    const size_t region[] = { roi_width, roi_height, 1 };

    err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device], src_image, dst_image, src_origin, dst_origin, region, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueCopyImage()");
}


// Write to image
template<typename T>
void hipaccWriteImage(cl_mem image, T *host_mem, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims dim = Ctx.get_mem_dims(image);

    const size_t origin[] = { 0, 0, 0 };
    const size_t region[] = { (size_t)dim.width, (size_t)dim.height, 1 };
    // no stride supported for images in OpenCL
    const size_t input_row_pitch = dim.width*sizeof(T);
    const size_t input_slice_pitch = 0;

    err = clEnqueueWriteImage(Ctx.get_command_queues()[num_device], image, CL_FALSE, origin, region, input_row_pitch, input_slice_pitch, host_mem, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueWriteImage()");
}


// Read from image
template<typename T>
void hipaccReadImage(T *host_mem, cl_mem image, int num_device=0) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();
    HipaccContext::cl_dims dim = Ctx.get_mem_dims(image);

    const size_t origin[] = { 0, 0, 0 };
    const size_t region[] = { (size_t)dim.width, (size_t)dim.height, 1 };
    // no stride supported for images in OpenCL
    const size_t row_pitch = dim.width*sizeof(T);
    const size_t slice_pitch = 0;

    err = clEnqueueReadImage(Ctx.get_command_queues()[num_device], image, CL_FALSE, origin, region, row_pitch, slice_pitch, host_mem, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[num_device]);
    checkErr(err, "clEnqueueReadImage()");
}


// Set a single argument of a kernel
template<typename T>
void hipaccSetKernelArg(cl_kernel kernel, unsigned int num, size_t size, T param) {
    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(kernel, num, size, &param);
    checkErr(err, "clSetKernelArg()");
}


// Enqueue and launch kernel
void hipaccEnqueueKernel(cl_kernel kernel, size_t *global_work_size, size_t *local_work_size, bool print_timing=true) {
    cl_int err = CL_SUCCESS;
    #ifdef GPU_TIMING
    cl_event event;
    cl_ulong end, start;
    #else
    long end, start;
    #endif
    HipaccContext &Ctx = HipaccContext::getInstance();

    #ifdef GPU_TIMING
    err = clEnqueueNDRangeKernel(Ctx.get_command_queues()[0], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueNDRangeKernel()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
    #else
    clFinish(Ctx.get_command_queues()[0]);
    start = getNanoTime();
    err = clEnqueueNDRangeKernel(Ctx.get_command_queues()[0], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    end = getNanoTime();
    checkErr(err, "clEnqueueNDRangeKernel()");
    #endif

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << local_work_size[0]*local_work_size[1] << ": " << local_work_size[0] << "x" << local_work_size[1] << "): " << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }
    total_time += (end-start)*1.0e-6f;
    last_gpu_timing = (end-start)*1.0e-6f;
}


unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    ++x;

    // get at least the warp size
    if (x < 32) x = 32;

    return x;
}


// Perform global reduction and return result
template<typename T>
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, void *image, T
        neutral, unsigned int width, unsigned int height, unsigned int stride,
        unsigned int offset_x, unsigned int offset_y, unsigned int is_width,
        unsigned int is_height, unsigned int max_threads, unsigned int
        pixels_per_thread) {
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
    global_work_size[0] = (int)ceil((float)(width)/(local_work_size[0]*2))*local_work_size[0];
    global_work_size[1] = (int)ceil((float)(is_height)/(local_work_size[1]*pixels_per_thread))*local_work_size[1];

    unsigned int num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    hipaccSetKernelArg(kernel2D, 0, sizeof(cl_mem), image);
    hipaccSetKernelArg(kernel2D, 1, sizeof(cl_mem), output);
    hipaccSetKernelArg(kernel2D, 2, sizeof(T), neutral);
    hipaccSetKernelArg(kernel2D, 3, sizeof(unsigned int), width);
    hipaccSetKernelArg(kernel2D, 4, sizeof(unsigned int), height);
    hipaccSetKernelArg(kernel2D, 5, sizeof(unsigned int), stride);
    // check if the reduction is applied to the whole image
    if ((offset_x || offset_y) && (is_width!=width || is_height!=height)) {
        hipaccSetKernelArg(kernel2D, 6, sizeof(unsigned int), offset_x);
        hipaccSetKernelArg(kernel2D, 7, sizeof(unsigned int), offset_y);
        hipaccSetKernelArg(kernel2D, 8, sizeof(unsigned int), is_width);
        hipaccSetKernelArg(kernel2D, 9, sizeof(unsigned int), is_height);
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

    hipaccSetKernelArg(kernel1D, 0, sizeof(cl_mem), output);
    hipaccSetKernelArg(kernel1D, 1, sizeof(cl_mem), output);
    hipaccSetKernelArg(kernel1D, 2, sizeof(T), neutral);
    hipaccSetKernelArg(kernel1D, 3, sizeof(unsigned int), num_blocks);
    hipaccSetKernelArg(kernel1D, 4, sizeof(unsigned int), num_steps);

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
T hipaccApplyReduction(cl_kernel kernel2D, cl_kernel kernel1D, void *image, T
        neutral, unsigned int width, unsigned int height, unsigned int stride,
        unsigned int max_threads, unsigned int pixels_per_thread) {
    return hipaccApplyReduction<T>(kernel2D, kernel1D, image, neutral, width,
            height, stride, 0, 0, width, height, max_threads,
            pixels_per_thread);
}


// Perform exploration of global reduction and return result
template<typename T>
T hipaccApplyReductionExploration(const char *filename, const char *kernel2D,
        const char *kernel1D, void *image, T neutral, unsigned int width,
        unsigned int height, unsigned int stride, unsigned int offset_x,
        unsigned int offset_y, unsigned int is_width, unsigned int is_height,
        unsigned int max_threads, unsigned int pixels_per_thread) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_int err = CL_SUCCESS;
    cl_mem output;  // GPU memory for reduction
    T result;       // host result

    unsigned int num_blocks = (int)ceil((float)(width)/(max_threads*2))*is_height;
    output = clCreateBuffer(Ctx.get_contexts()[0], flags, sizeof(T)*num_blocks, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    std::cerr << "<HIPACC:> Exploring pixels per thread for '" << kernel2D << ", " << kernel1D << "'" << std::endl;

    for (unsigned int ppt=1; ppt<=is_height; ppt++) {
        std::stringstream num_ppt_ss;
        std::stringstream num_bs_ss;
        num_ppt_ss << ppt;
        num_bs_ss << max_threads;

        std::string compile_options = "-D PPT=" + num_ppt_ss.str() + " -D BS=" + num_bs_ss.str() + " -I./include ";
        cl_kernel exploreReduction2D = hipaccBuildProgramAndKernel(filename, kernel2D, false, false, false, compile_options.c_str());
        cl_kernel exploreReduction1D = hipaccBuildProgramAndKernel(filename, kernel1D, false, false, false, compile_options.c_str());

        float timing=FLT_MAX;
        #ifndef GPU_TIMING
        std::vector<float> times;
        times.reserve(HIPACC_NUM_ITERATIONS);
        #endif
        for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
            // first step: reduce image (region) into linear memory
            size_t local_work_size[2];
            local_work_size[0] = max_threads;
            local_work_size[1] = 1;
            size_t global_work_size[2];
            global_work_size[0] = (int)ceil((float)(width)/(local_work_size[0]*2))*local_work_size[0];
            global_work_size[1] = (int)ceil((float)(is_height)/(local_work_size[1]*ppt))*local_work_size[1];
            num_blocks = (global_work_size[0]/local_work_size[0])*(global_work_size[1]/local_work_size[1]);

            // start timing
            total_time = 0.0f;

            hipaccSetKernelArg(exploreReduction2D, 0, sizeof(cl_mem), image);
            hipaccSetKernelArg(exploreReduction2D, 1, sizeof(cl_mem), output);
            hipaccSetKernelArg(exploreReduction2D, 2, sizeof(T), neutral);
            hipaccSetKernelArg(exploreReduction2D, 3, sizeof(unsigned int), width);
            hipaccSetKernelArg(exploreReduction2D, 4, sizeof(unsigned int), height);
            hipaccSetKernelArg(exploreReduction2D, 5, sizeof(unsigned int), stride);
            // check if the reduction is applied to the whole image
            if ((offset_x || offset_y) && (is_width!=width || is_height!=height)) {
                hipaccSetKernelArg(exploreReduction2D, 6, sizeof(unsigned int), offset_x);
                hipaccSetKernelArg(exploreReduction2D, 7, sizeof(unsigned int), offset_y);
                hipaccSetKernelArg(exploreReduction2D, 8, sizeof(unsigned int), is_width);
                hipaccSetKernelArg(exploreReduction2D, 9, sizeof(unsigned int), is_height);
            }

            hipaccEnqueueKernel(exploreReduction2D, global_work_size, local_work_size, false);


            // second step: reduce partial blocks on GPU
            global_work_size[1] = 1;
            while (num_blocks > 1) {
                local_work_size[0] = (num_blocks < max_threads) ? nextPow2((num_blocks+1)/2) :
                    max_threads;
                global_work_size[0] = (int)ceil((float)(num_blocks)/(local_work_size[0]*ppt))*local_work_size[0];

                hipaccSetKernelArg(exploreReduction1D, 0, sizeof(cl_mem), output);
                hipaccSetKernelArg(exploreReduction1D, 1, sizeof(cl_mem), output);
                hipaccSetKernelArg(exploreReduction1D, 2, sizeof(T), neutral);
                hipaccSetKernelArg(exploreReduction1D, 3, sizeof(unsigned int), num_blocks);
                hipaccSetKernelArg(exploreReduction1D, 4, sizeof(unsigned int), ppt);

                hipaccEnqueueKernel(exploreReduction1D, global_work_size, local_work_size, false);

                num_blocks = global_work_size[0]/local_work_size[0];
            }
            // stop timing
            #ifdef GPU_TIMING
            if (total_time < timing) timing = total_time;
            #else
            times.push_back(total_time);
            #endif
        }

        // print timing
        #ifndef GPU_TIMING
        std::sort(times.begin(), times.end());
        timing = times.at(HIPACC_NUM_ITERATIONS/2);
        #endif
        std::cerr << "<HIPACC:> PPT: " << std::setw(4) << std::right << ppt
                  << ", " << std::setw(8) << std::fixed << std::setprecision(4)
                  << timing << " ms" << std::endl;
    }

    // get reduced value
    err = clEnqueueReadBuffer(Ctx.get_command_queues()[0], output, CL_FALSE, 0, sizeof(T), &result, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    checkErr(err, "clEnqueueReadBuffer()");

    err = clReleaseMemObject(output);
    checkErr(err, "clReleaseMemObject()");

    return result;
}
template<typename T>
T hipaccApplyReductionExploration(const char *filename, const char *kernel2D,
        const char *kernel1D, void *image, T neutral, unsigned int width,
        unsigned int height, unsigned int stride, unsigned int max_threads,
        unsigned int pixels_per_thread) {
    return hipaccApplyReductionExploration<T>(filename, kernel2D, kernel1D, image,
            neutral, width, height, stride, 0, 0, width, height, max_threads,
            pixels_per_thread);
}


// Benchmark timing for a kernel call
void hipaccEnqueueKernelBenchmark(cl_kernel kernel, std::vector<std::pair<size_t, void *> > args, size_t *global_work_size, size_t *local_work_size, bool print_timing=true) {
    float timing=FLT_MAX;
    #ifndef GPU_TIMING
    std::vector<float> times;
    times.reserve(HIPACC_NUM_ITERATIONS);
    #endif

    for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
        // set kernel arguments
        for (unsigned int i=0; i<args.size(); i++) {
            hipaccSetKernelArg(kernel, i, args.data()[i].first, args.data()[i].second);
        }

        // launch kernel
        hipaccEnqueueKernel(kernel, global_work_size, local_work_size, print_timing);
        #ifdef GPU_TIMING
        if (last_gpu_timing < timing) timing = last_gpu_timing;
        #else
        times.push_back(last_gpu_timing);
        #endif
    }

    #ifndef GPU_TIMING
    std::sort(times.begin(), times.end());
    timing = times.at(HIPACC_NUM_ITERATIONS/2);
    #endif
    last_gpu_timing = timing;
}


// Perform configuration exploration for a kernel call
void hipaccKernelExploration(const char *filename, const char *kernel,
        std::vector<std::pair<size_t, void *> > args,
        std::vector<hipacc_smem_info> smems, hipacc_launch_info &info, int
        warp_size, int max_threads_per_block, int max_threads_for_kernel, int
        max_smem_per_block, int opt_tx, int opt_ty) {

    std::cerr << "<HIPACC:> Exploring configurations for kernel '" << kernel << "': optimal configuration ";
    std::cerr << opt_tx*opt_ty << "(" << opt_tx << "x" << opt_ty << "). " << std::endl;

    for (int tile_size_x=warp_size; tile_size_x<=max_threads_per_block; tile_size_x+=warp_size) {
        for (int tile_size_y=1; tile_size_y<=max_threads_per_block; tile_size_y++) {
            // check if we exceed maximum number of threads
            if (tile_size_x*tile_size_y > max_threads_for_kernel) continue;

            // check if we exceed size of shared memory
            int used_smem = 0;
            for (unsigned int i=0; i<smems.size(); i++) {
                used_smem += (tile_size_x + smems.data()[i].size_x)*(tile_size_y + smems.data()[i].size_y - 1) * smems.data()[i].pixel_size;
            }
            if (used_smem >= max_smem_per_block) continue;

            std::stringstream num_threads_x_ss, num_threads_y_ss;
            num_threads_x_ss << tile_size_x;
            num_threads_y_ss << tile_size_y;

            // compile kernel
            std::string compile_options = "-D BSX_EXPLORE=" +
                num_threads_x_ss.str() + " -D BSY_EXPLORE=" +
                num_threads_y_ss.str() + " ";
            cl_kernel exploreKernel = hipaccBuildProgramAndKernel(filename, kernel, false, false, false, compile_options.c_str());


            size_t local_work_size[2];
            local_work_size[0] = tile_size_x;
            local_work_size[1] = tile_size_y;
            size_t global_work_size[2];
            hipaccCalcGridFromBlock(info, local_work_size, global_work_size);
            hipaccPrepareKernelLaunch(info, local_work_size);

            float timing=FLT_MAX;
            #ifndef GPU_TIMING
            std::vector<float> times;
            times.reserve(HIPACC_NUM_ITERATIONS);
            #endif
            for (int i=0; i<HIPACC_NUM_ITERATIONS; i++) {
                // set kernel arguments
                for (unsigned int i=0; i<args.size(); i++) {
                    hipaccSetKernelArg(exploreKernel, i, args.data()[i].first, args.data()[i].second);
                }

                // start timing
                total_time = 0.0f;

                hipaccEnqueueKernel(exploreKernel, global_work_size, local_work_size, false);

                // stop timing
                #ifdef GPU_TIMING
                if (total_time < timing) timing = total_time;
                #else
                times.push_back(timing);
                #endif
            }

            // print timing
            #ifndef GPU_TIMING
            std::sort(times.begin(), times.end());
            timing = times.at(HIPACC_NUM_ITERATIONS/2);
            #endif
            std::cerr << "<HIPACC:> Kernel config: "
                      << std::setw(4) << std::right << tile_size_x << "x"
                      << std::setw(2) << std::left << tile_size_y
                      << std::setw(5-floor(log10(tile_size_x*tile_size_y)))
                      << std::right << "(" << tile_size_x*tile_size_y << "): "
                      << std::setw(8) << std::fixed << std::setprecision(4)
                      << timing << " ms" << std::endl;
        }
    }
}

#endif  // __HIPACC_OCL_HPP__

