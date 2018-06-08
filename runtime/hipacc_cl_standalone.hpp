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

#ifndef __HIPACC_CL_STANDALONE_HPP__
#define __HIPACC_CL_STANDALONE_HPP__


// This is the standalone (header-only) Hipacc OpenCL runtime


#include "hipacc_cl.hpp"
#include "hipacc_base_standalone.hpp"


std::string getOpenCLErrorCodeStr(int error) {
    #define CL_ERROR_CODE(CODE) case CODE: return #CODE;
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
        default: return "unknown error code";
    }
    #undef CL_ERROR_CODE
}


inline void __checkOpenCLErrors(cl_int err, std::string name, std::string file, const int line) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")"
                  << " [file " << file << ", line " << line << "]: "
                  << getOpenCLErrorCodeStr(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


HipaccContext& HipaccContext::getInstance() {
    static HipaccContext instance;

    return instance;
}

void HipaccContext::add_platform(cl_platform_id id, cl_platform_name name) {
    platforms.push_back(id);
    platform_names.push_back(name);
}

void HipaccContext::add_device(cl_device_id id) {
    devices.push_back(id);
}

void HipaccContext::add_device_all(cl_device_id id) {
    devices_all.push_back(id);
}

void HipaccContext::add_context(cl_context id) {
    contexts.push_back(id);
}

void HipaccContext::add_command_queue(cl_command_queue id) {
    queues.push_back(id);
}

std::vector<cl_platform_id> HipaccContext::get_platforms() {
    return platforms;
}

std::vector<cl_platform_name> HipaccContext::get_platform_names() {
    return platform_names;
}

std::vector<cl_device_id> HipaccContext::get_devices() {
    return devices;
}

std::vector<cl_device_id> HipaccContext::get_devices_all() {
    return devices_all;
}

std::vector<cl_context> HipaccContext::get_contexts() {
    return contexts;
}

std::vector<cl_command_queue> HipaccContext::get_command_queues() {
    return queues;
}

HipaccImageOpenCL::HipaccImageOpenCL(size_t width, size_t height, 
    size_t stride, size_t alignment, size_t pixel_size, cl_mem mem,
    hipaccMemoryType mem_type)
    : HipaccImageBase(width, height, stride, alignment, pixel_size, (void*)mem,
        mem_type), mem(mem) {
}

HipaccImageOpenCL::~HipaccImageOpenCL() {
    cl_int err = clReleaseMemObject(mem);
    checkErr(err, "clReleaseMemObject()");
}

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


// Select platform and device for execution
void hipaccInitPlatformsAndDevices(cl_device_type dev_type, cl_platform_name platform_name) {
    HipaccContext &Ctx = HipaccContext::getInstance();
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024], pd2Buffer[1024];

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

    std::cerr << "Number of available Platforms: " << num_platforms << std::endl;
    if (num_platforms == 0) {
        exit(EXIT_FAILURE);
    } else {
        int platform_number = -1, device_number = -1;
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

            cl_uint num_devices = 0, num_devices_type = 0;
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
void hipaccCreateContextsAndCommandQueues(bool all_devies) {
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
    for (auto device : devices) {
        #ifdef CL_VERSION_2_0
        cl_queue_properties cprops[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        command_queue = clCreateCommandQueueWithProperties(context, device, cprops, &err);
        checkErr(err, "clCreateCommandQueueWithProperties()");
        #else
        command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkErr(err, "clCreateCommandQueue()");
        #endif

        Ctx.add_command_queue(command_queue);
    }
}


// Get binary from OpenCL program and dump it to stderr
void hipaccDumpBinary(cl_program program, cl_device_id device) {
    cl_uint num_devices;

    // Get the number of devices associated with the program
    cl_int err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

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
cl_kernel hipaccBuildProgramAndKernel(std::string file_name, std::string kernel_name, bool print_progress, bool dump_binary, bool print_log, std::string build_options, std::string build_includes) {
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


#define CREATE_IMAGE(DATA_TYPE, CHANNEL_TYPE, CHANNEL_ORDER) \
template <> \
HipaccImage hipaccCreateImage<DATA_TYPE>(DATA_TYPE *host_mem, size_t width, size_t height) { \
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

    #ifdef CL_VERSION_2_0
    cl_sampler_properties sprops[7] = {
        CL_SAMPLER_NORMALIZED_COORDS, normalized_coords,
        CL_SAMPLER_ADDRESSING_MODE, addressing_mode,
        CL_SAMPLER_FILTER_MODE, filter_mode,
        0
    };
    sampler = clCreateSamplerWithProperties(Ctx.get_contexts()[0], sprops, &err);
    checkErr(err, "clCreateSamplerWithProperties()");
    #else
    sampler = clCreateSampler(Ctx.get_contexts()[0], normalized_coords, addressing_mode, filter_mode, &err);
    checkErr(err, "clCreateSampler()");
    #endif

    return sampler;
}


// Copy between memory
void hipaccCopyMemory(const HipaccImage &src, HipaccImage &dst, int num_device) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    assert(src->width == dst->width && src->height == dst->height && src->pixel_size == dst->pixel_size && "Invalid CopyBuffer or CopyImage!");

    if (src->mem_type >= Array2D) {
        const size_t origin[] = { 0, 0, 0 };
        const size_t region[] = { src->width, src->height, 1 };

        err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device], (cl_mem)src->mem, (cl_mem)dst->mem, origin, origin, region, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyImage()");
    } else {
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src->mem, (cl_mem)dst->mem, 0, 0, src->stride*src->height*src->pixel_size, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBuffer()");
    }
}


// Copy from memory region to memory region
void hipaccCopyMemoryRegion(const HipaccAccessor &src, const HipaccAccessor &dst, int num_device) {
    cl_int err = CL_SUCCESS;
    HipaccContext &Ctx = HipaccContext::getInstance();

    if (src.img->mem_type >= Array2D) {
        const size_t dst_origin[] = { (size_t)dst.offset_x, (size_t)dst.offset_y, 0 };
        const size_t src_origin[] = { (size_t)src.offset_x, (size_t)src.offset_y, 0 };
        const size_t region[]     = { dst.width, dst.height, 1 };

        err = clEnqueueCopyImage(Ctx.get_command_queues()[num_device], (cl_mem)src.img->mem, (cl_mem)dst.img->mem, src_origin, dst_origin, region, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyImage()");
    } else {
        const size_t dst_origin[] = { dst.offset_x*dst.img->pixel_size, (size_t)dst.offset_y, 0 };
        const size_t src_origin[] = { src.offset_x*src.img->pixel_size, (size_t)src.offset_y, 0 };
        const size_t region[]     = { dst.width*dst.img->pixel_size, dst.height, 1 };

        err = clEnqueueCopyBufferRect(Ctx.get_command_queues()[num_device], (cl_mem)src.img->mem, (cl_mem)dst.img->mem, src_origin, dst_origin, region,
                                      src.img->stride*src.img->pixel_size, 0, dst.img->stride*dst.img->pixel_size, 0, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        checkErr(err, "clEnqueueCopyBufferRect()");
    }
}


// Copy between buffers and return time
double hipaccCopyBufferBenchmark(const HipaccImage &src, HipaccImage &dst, int num_device, bool print_timing) {
    cl_int err = CL_SUCCESS;
    cl_ulong end, start;
    std::vector<float> times;
    #ifdef EVENT_TIMING
    cl_event event;
    #endif
    HipaccContext &Ctx = HipaccContext::getInstance();

    assert(src->width == dst->width && src->height == dst->height && src->pixel_size == dst->pixel_size && "Invalid CopyBuffer!");

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        #ifdef EVENT_TIMING
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src->mem, (cl_mem)dst->mem, 0, 0, src->width*src->height*src->pixel_size, 0, NULL, &event);
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
        start = hipacc_time_micro();
        err = clEnqueueCopyBuffer(Ctx.get_command_queues()[num_device], (cl_mem)src->mem, (cl_mem)dst->mem, 0, 0, src->width*src->height*src->pixel_size, 0, NULL, NULL);
        err |= clFinish(Ctx.get_command_queues()[num_device]);
        end = hipacc_time_micro();
        checkErr(err, "clEnqueueCopyBuffer()");
        #endif

        if (print_timing) {
            std::cerr << "<HIPACC:> Copy timing (" << (src->width*src->height*src->pixel_size) / (float)(1 << 20) << " MB): " << (end-start)*1.0e-3f << "(ms)" << std::endl;
            std::cerr << "          Bandwidth: " << 2.0f * (double)(src->width*src->height*src->pixel_size) / ((end-start)*1.0e-6f * (float)(1 << 30)) << " GB/s" << std::endl;
        }
        times.push_back(end-start);
    }

    // return time in ms
    #ifdef EVENT_TIMING
    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
    #endif
    std::sort(times.begin(), times.end());
    last_gpu_timing = times[times.size()/2];

    return last_gpu_timing*1.0e-3f;
}


// Enqueue and launch kernel
void hipaccLaunchKernel(cl_kernel kernel, size_t *global_work_size, size_t *local_work_size, bool print_timing) {
    cl_int err;
    cl_ulong end, start;
    #ifdef EVENT_TIMING
    cl_event event;
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
    start = hipacc_time_micro();
    err = clEnqueueNDRangeKernel(Ctx.get_command_queues()[0], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    err |= clFinish(Ctx.get_command_queues()[0]);
    end = hipacc_time_micro();
    checkErr(err, "clEnqueueNDRangeKernel()");
    #endif

    last_gpu_timing = (end-start)*1.0e-3f;
    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing (" << local_work_size[0]*local_work_size[1] << ": " << local_work_size[0] << "x" << local_work_size[1] << "): " << last_gpu_timing << "(ms)" << std::endl;
    }
}


// Benchmark timing for a kernel call
void hipaccLaunchKernelBenchmark(cl_kernel kernel, size_t *global_work_size, size_t *local_work_size, std::vector<std::pair<size_t, void *> > args, bool print_timing) {
    std::vector<float> times;

    for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
        // set kernel arguments
        for (size_t j=0; j<args.size(); ++j)
            hipaccSetKernelArg(kernel, j, args[j].first, args[j].second);

        hipaccLaunchKernel(kernel, global_work_size, local_work_size, print_timing);
        times.push_back(last_gpu_timing);
    }

    std::sort(times.begin(), times.end());
    last_gpu_timing = times[times.size()/2];

    if (print_timing) {
        std::cerr << "<HIPACC:> Kernel timing benchmark ("
                  << local_work_size[0]*local_work_size[1] << ": "
                  << local_work_size[0] << "x" << local_work_size[1] << "): "
                  << last_gpu_timing << " | " << times.front() << " | " << times.back()
                  << " (median(" << HIPACC_NUM_ITERATIONS << ") | minimum | maximum) ms" << std::endl;
    }
}


// Perform configuration exploration for a kernel call
void hipaccLaunchKernelExploration(std::string filename, std::string kernel,
        std::vector<std::pair<size_t, void *>> args,
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
            for (auto smem : smems)
                used_smem += (tile_size_x + smem.size_x)*(tile_size_y + smem.size_y - 1) * smem.pixel_size;
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
            std::vector<float> times;

            for (size_t i=0; i<HIPACC_NUM_ITERATIONS; ++i) {
                // set kernel arguments
                for (size_t j=0; j<args.size(); ++j)
                    hipaccSetKernelArg(exploreKernel, j, args[j].first, args[j].second);

                hipaccLaunchKernel(exploreKernel, global_work_size, local_work_size, false);
                times.push_back(last_gpu_timing);
            }

            std::sort(times.begin(), times.end());
            last_gpu_timing = times[times.size()/2];

            if (last_gpu_timing < opt_time) {
                opt_time = last_gpu_timing;
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
                      << last_gpu_timing << " | " << times.front() << " | " << times.back()
                      << " (median(" << HIPACC_NUM_ITERATIONS << ") | minimum | maximum) ms" << std::endl;

            // release kernel
            cl_int err = clReleaseKernel(exploreKernel);
            checkErr(err, "clReleaseKernel()");
        }
    }
    last_gpu_timing = opt_time;
    std::cerr << "<HIPACC:> Best configurations for kernel '" << kernel << "': "
              << opt_tx*opt_ty << " (" << opt_tx << "x" << opt_ty << "): "
              << opt_time << " ms" << std::endl;
}


#endif  // __HIPACC_CL_STANDALONE_HPP__

