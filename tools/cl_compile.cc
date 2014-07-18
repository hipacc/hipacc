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

/*
 * Tool that compiles a OpenCL kernel in order to determine the resource usage
 */

#include <algorithm>
#include <glob.h>
#include <stdio.h>
#include <unistd.h>

#include "hipacc_cl.hpp"


void usage(char **argv) {
    fprintf(stderr, "Usage: %s [-h] [-d ACC|CPU|GPU|ALL] [-p AMD|APPLE|ARM|INTEL|NVIDIA|ALL] [-i <include_dir>] -k <kernel_name> -f <opencl_file>\n", argv[0]);
}


int main(int argc, char *argv[]) {
    int option = 0;
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    cl_platform_name platform_name = ALL;
    std::string kernel_name;
    std::string file_name;
    std::string build_options = "";
    std::string build_includes = "";
    bool found_kernel_name = false, found_file_name = false, print_compile_progress = false, dump_binary = false, print_log = true;

    // scan command-line options
    while ((option = getopt(argc, (char * const *)argv, "hd:p:k:f:i:")) != -1) {
        switch (option) {
            case 'h':
                fprintf(stderr, "Compile OpenCL kernel for given Platform and/or device type.\n");
                usage(argv);
                exit(EXIT_SUCCESS);
            case 'd':
                if (strncmp(optarg, "ACC", 3) == 0) device_type = CL_DEVICE_TYPE_ACCELERATOR;
                else if (strncmp(optarg, "CPU", 3) == 0) device_type = CL_DEVICE_TYPE_CPU;
                else if (strncmp(optarg, "GPU", 3) == 0) device_type = CL_DEVICE_TYPE_GPU;
                else if (strncmp(optarg, "ALL", 3) == 0) device_type = CL_DEVICE_TYPE_ALL;
                else fprintf(stderr, "Unknown device type '%s', using 'ALL' as default ...\n", optarg);
                break;
            case 'p':
                if (strncmp(optarg, "AMD", 3) == 0) platform_name = AMD;
                else if (strncmp(optarg, "APPLE", 3) == 0) platform_name = APPLE;
                else if (strncmp(optarg, "ARM", 3) == 0) platform_name = ARM;
                else if (strncmp(optarg, "INTEL", 3) == 0) platform_name = INTEL;
                else if (strncmp(optarg, "NVIDIA", 3) == 0) platform_name = NVIDIA;
                else fprintf(stderr, "Unknown platform name '%s', using 'ALL' as default ...\n", optarg);
                break;
            case 'k':
                found_kernel_name = true;
                kernel_name = optarg;
                break;
            case 'f':
                found_file_name = true;
                file_name = optarg;
                break;
            case 'i':
                build_includes += "-I";
                build_includes += optarg;
                build_includes += " ";
                break;
            default: /* '?' */
                fprintf(stderr, "Wrong call syntax!\n");
                usage(argv);
                exit(EXIT_FAILURE);
        }
    }

    if (!found_file_name) {
        fprintf(stderr, "No OpenCL input file specified ...\n");
        usage(argv);
        exit(EXIT_FAILURE);
    }
    if (!found_kernel_name) {
        fprintf(stderr, "No OpenCL kernel specified ...\n");
        usage(argv);
        exit(EXIT_FAILURE);
    }

    hipaccInitPlatformsAndDevices(device_type, platform_name);
    hipaccCreateContextsAndCommandQueues();

    hipaccBuildProgramAndKernel(file_name, kernel_name, print_compile_progress, dump_binary, print_log, build_options.c_str(), build_includes.c_str());

    if (platform_name == AMD) {
        // compile OpenCL kernel in order to get resource usage
        // SQ_PGM_RESOURCES:NUM_GPRS = 25
        // SQ_PGM_RESOURCES:STACK_SIZE = 5
        // SQ_LDS_ALLOC:SIZE = 0x00000000
        #ifdef CL_VERSION_1_2
        std::string kernel_glob_command = "ls *_" + kernel_name + ".isa";
        #else
        std::string kernel_glob_command = "ls " + kernel_name + "_*.isa";
        #endif

        char glob_line[FILENAME_MAX];
        std::string kernel_file_name;
        FILE *fpipe;

        if (!(fpipe = (FILE*)popen(kernel_glob_command.c_str(), "r"))) {
            perror("Problems with pipe");
            exit(EXIT_FAILURE);
        }

        while (fgets(glob_line, sizeof(char) * FILENAME_MAX, fpipe)) {
            // trim whitespace
            kernel_file_name = glob_line;
            kernel_file_name.erase(std::remove_if(kernel_file_name.begin(), kernel_file_name.end(), isspace), kernel_file_name.end());
        }
        pclose(fpipe);

        std::ifstream file;
        file.open(kernel_file_name.c_str());
        printf("kernels: '%s'\n", kernel_file_name.c_str());

        std::string line;
        int num_gprs=-1, lds_size=-1;
        while (std::getline(file, line)) {
            if (num_gprs < 0) {
                sscanf(line.c_str(), "SQ_PGM_RESOURCES:NUM_GPRS = %d", &num_gprs);
            }
            if (lds_size < 0) {
                sscanf(line.c_str(), "SQ_LDS_ALLOC:SIZE = %i", &lds_size);
            }
        }
        if (num_gprs < 0 || lds_size < 0) {
            printf("isa error while determining resource usage : Used %d gprs, %d bytes lds\n", num_gprs, lds_size);
        } else {
            printf("isa info : Used %d gprs, %d bytes lds\n", num_gprs, lds_size);
        }
        fflush(stdout);
    }

    return EXIT_SUCCESS;
}

