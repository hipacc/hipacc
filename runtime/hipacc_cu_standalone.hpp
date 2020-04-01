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

#ifndef __HIPACC_CU_STANDALONE_HPP__
#define __HIPACC_CU_STANDALONE_HPP__

// This is the standalone (header-only) Hipacc CUDA runtime

#include "hipacc_base_standalone.hpp"
#include "hipacc_cu.hpp"

#ifdef _WIN32
#include <io.h>
#define popen(x, y) _popen(x, y)
#define pclose(x) _pclose(x)
#define WEXITSTATUS(x) (x != 0)
#endif

HipaccKernelTimingBase hipaccCudaTiming;

std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
  const char *error_name(nullptr);
  const char *error_string(nullptr);
  cuGetErrorName(errorCode, &error_name);
  cuGetErrorString(errorCode, &error_string);
  return std::string(error_name) + ": " + std::string(error_string);
}

std::string getCUDAErrorCodeStr(cudaError_t errorCode) {
  const char *error_name(nullptr);
  const char *error_string(nullptr);
  error_name = cudaGetErrorName(errorCode);
  error_string = cudaGetErrorString(errorCode);
  return std::string(error_name) + ": " + std::string(error_string);
}

HipaccContext &HipaccContext::getInstance() {
  static HipaccContext instance;

  return instance;
}

HipaccImageCudaBase::~HipaccImageCudaBase() {}

void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 &block) {
  // calculate block id of a) first block that requires no border handling
  // (left, top) and b) first block that requires border handling (right,
  // bottom)
  if (info.size_x > 0) {
    info.bh_start_left = (int)ceilf((float)(info.offset_x + info.size_x) /
                                    (block.x * info.simd_width));
    info.bh_start_right =
        (int)floor((float)(info.offset_x + info.is_width - info.size_x) /
                   (block.x * info.simd_width));
  } else {
    info.bh_start_left = 0;
    info.bh_start_right = (int)floor((float)(info.offset_x + info.is_width) /
                                     (block.x * info.simd_width));
  }
  if (info.size_y > 0) {
    // for shared memory calculate additional blocks to be staged - this is
    // only required if shared memory is used, otherwise, info.size_y would
    // be sufficient
    int p_add = (int)ceilf(2 * info.size_y / (float)block.y);
    info.bh_start_top =
        (int)ceilf((float)(info.size_y) / (info.pixels_per_thread * block.y));
    info.bh_start_bottom =
        (int)floor((float)(info.is_height - p_add * block.y) /
                   (block.y * info.pixels_per_thread));
  } else {
    info.bh_start_top = 0;
    info.bh_start_bottom = (int)floor((float)(info.is_height) /
                                      (block.y * info.pixels_per_thread));
  }

  if ((info.bh_start_right - info.bh_start_left) > 1 &&
      (info.bh_start_bottom - info.bh_start_top) > 1) {
    info.bh_fall_back = 0;
  } else {
    info.bh_fall_back = 1;
  }
}

dim3 hipaccCalcGridFromBlock(hipacc_launch_info &info, dim3 &block) {
  return dim3(
      (int)ceilf((float)(info.is_width + info.offset_x) /
                 (block.x * info.simd_width)),
      (int)ceilf((float)(info.is_height) / (block.y * info.pixels_per_thread)));
}

// Initialize CUDA devices
void hipaccInitCUDA() {
  setenv("CUDA_CACHE_DISABLE", "1", 1);

  int device_count, driver_version = 0, runtime_version = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  checkErr(err, "cudaGetDeviceCount()");
  err = cudaDriverGetVersion(&driver_version);
  checkErr(err, "cudaDriverGetVersion()");
  err = cudaRuntimeGetVersion(&runtime_version);
  checkErr(err, "cudaRuntimeGetVersion()");

  hipaccRuntimeLogTrivial(
      hipaccRuntimeLogLevel::INFO,
      "CUDA Driver/Runtime Version " + std::to_string(driver_version / 1000) +
          "." + std::to_string((driver_version % 100) / 10) + "/" +
          std::to_string(runtime_version / 1000) + "." +
          std::to_string((runtime_version % 100) / 10));

#ifdef NVRTC_FOUND
  int nvrtc_major = 0, nvrtc_minor = 0;
  nvrtcResult errNvrtc = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  checkErrNVRTC(errNvrtc, "nvrtcVersion()");

  hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                          "NVRTC Version " + std::to_string(nvrtc_major) + "." +
                              std::to_string(nvrtc_minor));
#endif

  for (size_t i = 0; i < (size_t)device_count; ++i) {
    cudaDeviceProp device_prop;

    err = cudaSetDevice(i);
    checkErr(err, "cudaSetDevice()");
    err = cudaGetDeviceProperties(&device_prop, i);
    checkErr(err, "cudaGetDeviceProperties()");

    if (i) {
      hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, "  [ ] ");
    } else {
      hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO, "  [*] ");
    }
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                            "Name: " + std::string(device_prop.name));
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "      Compute capability: " + std::to_string(device_prop.major) + "." +
            std::to_string(device_prop.minor));
  }
  err = cudaSetDevice(0);
  checkErr(err, "cudaSetDevice()");
}

// Copy from memory to memory
void hipaccCopyMemory(const HipaccImageCuda &src, HipaccImageCuda &dst) {
  size_t height = src->get_height();
  size_t stride = src->get_stride();

  if (src->get_mem_type() == hipaccMemoryType::Array2D ||
      src->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DArrayToArray(
        (cudaArray *)dst->get_device_memory(), 0, 0,
        (cudaArray *)src->get_device_memory(), 0, 0, stride * src->get_pixel_size(),
        height * src->get_pixel_size(), cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy2DArrayToArray()");
  } else {
    cudaError_t err =
        cudaMemcpy(dst->get_device_memory(), src->get_device_memory(),
                   src->get_pixel_size() * stride * height, cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy()");
  }
}

// Copy from memory region to memory region
void hipaccCopyMemoryRegion(const HipaccAccessor &src,
                            const HipaccAccessor &dst) {
  if (src.img->get_mem_type() == hipaccMemoryType::Array2D ||
      src.img->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DArrayToArray(
        (cudaArray *)dst.img->get_device_memory(),
        dst.offset_x * dst.img->get_pixel_size(), dst.offset_y,
        (cudaArray *)src.img->get_device_memory(),
        src.offset_x * src.img->get_pixel_size(), src.offset_y,
        src.width * src.img->get_pixel_size(), src.height, cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy2DArrayToArray()");
  } else {
    void *dst_start = (char *)dst.img->get_device_memory() +
                      dst.offset_x * dst.img->get_pixel_size() +
                      (dst.offset_y * dst.img->get_stride() * dst.img->get_pixel_size());
    void *src_start = (char *)src.img->get_device_memory() +
                      src.offset_x * src.img->get_pixel_size() +
                      (src.offset_y * src.img->get_stride() * src.img->get_pixel_size());

    cudaError_t err = cudaMemcpy2D(
        dst_start, dst.img->get_stride() * dst.img->get_pixel_size(), src_start,
        src.img->get_stride() * src.img->get_pixel_size(), src.width * src.img->get_pixel_size(),
        src.height, cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy2D()");
  }
}

// Unbind texture
void hipaccUnbindTexture(const textureReference *tex) {
  cudaError_t err = cudaUnbindTexture(tex);
  checkErr(err, "cudaUnbindTexture()");
}

// Launch kernel
void hipaccLaunchKernel(const void *kernel, std::string kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing) {
  cudaEvent_t start, end;
  float last_gpu_timing;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaError_t err = cudaLaunchKernel(kernel, grid, block, args, 0, 0);
  checkErr(err, "cudaLaunchKernel(" + kernel_name + ")");

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  checkErr(err, "cudaLaunchKernel(" + kernel_name + ")");

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&last_gpu_timing, start, end);

  hipaccCudaTiming.set_gpu_timing(last_gpu_timing);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  if (print_timing) {
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (" + std::to_string(block.x * block.y) + ": " +
            std::to_string(block.x) + "x" + std::to_string(block.y) +
            "): " + std::to_string(last_gpu_timing) + "(ms)");
  }
}

// Benchmark timing for a kernel call
void hipaccLaunchKernelBenchmark(const void *kernel, std::string kernel_name,
                                 dim3 &grid, dim3 &block,
                                 std::vector<void *> &args, bool print_timing) {
  std::vector<float> times;

  for (size_t i = 0; i < HIPACC_NUM_ITERATIONS; ++i) {
    hipaccLaunchKernel(kernel, kernel_name, grid, block, args.data(),
                       print_timing);
    times.push_back(hipaccCudaTiming.get_last_kernel_timing());
  }

  std::sort(times.begin(), times.end());
  hipaccCudaTiming.set_gpu_timing(times[times.size() / 2]);

  if (print_timing)
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (" + std::to_string(block.x * block.y) + ": " +
            std::to_string(block.x) + "x" + std::to_string(block.y) + "): " +
            std::to_string(hipaccCudaTiming.get_last_kernel_timing()) + "(ms)");
}

//
// DRIVER API
//

// Create a module from ptx assembly
void hipaccCreateModule(CUmodule &module, const void *ptx, int cc) {
  CUjit_target target_cc = (CUjit_target)cc;
  const unsigned int opt_level = 4;
  const unsigned int error_log_size = 10240;
  const unsigned int num_options = 4;
  char error_log_buffer[error_log_size] = {0};

  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *option_values[] = {(void *)error_log_buffer, (void *)error_log_size,
                           (void *)target_cc, (void *)opt_level};

  CUresult err =
      cuModuleLoadDataEx(&module, ptx, num_options, options, option_values);
  if (err != CUDA_SUCCESS)
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::WARNING,
                            "Error log: " + std::string(error_log_buffer));
  checkErrDrv(err, "cuModuleLoadDataEx()");
}

// Compile CUDA source file and create module
void hipaccCompileCUDAToModule(CUmodule &module, std::string file_name, int cc,
                               std::vector<std::string> &build_options) {
  nvrtcResult err;
  nvrtcProgram program;
  CUjit_target target_cc = (CUjit_target)cc;

  std::ifstream cu_file(file_name);
  if (!cu_file.is_open()) {
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::ERROR,
                            "ERROR: Can't open CU source file '" + file_name +
                                "'!");
  }

  std::string cu_string = std::string(std::istreambuf_iterator<char>(cu_file),
                                      (std::istreambuf_iterator<char>()));

  err = nvrtcCreateProgram(&program, cu_string.c_str(), file_name.c_str(), 0,
                           NULL, NULL);
  checkErrNVRTC(err, "nvrtcCreateProgram()");

  int offset = 2;
  int num_options = build_options.size() + offset;
  const char **options = new const char *[num_options];
  std::string compute_arch("-arch=compute_" + std::to_string(target_cc));
  options[0] = compute_arch.c_str();
  options[1] = "-std=c++11";
  // options[2] = "-G";
  // options[3] = "-lineinfo";
  for (int i = offset; i < num_options; ++i)
    options[i] = build_options[i - offset].c_str();

  err = nvrtcCompileProgram(program, num_options, options);
  if (err != NVRTC_SUCCESS) {
    size_t log_size;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string error_log(log_size, '\0');
    nvrtcGetProgramLog(program, &error_log[0]);
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::WARNING,
                            "Error log: " + error_log);
  }
  checkErrNVRTC(err, "nvrtcCompileProgram()");

  size_t ptx_size;
  err = nvrtcGetPTXSize(program, &ptx_size);
  checkErrNVRTC(err, "nvrtcGetPTXSize()");

  std::string ptx(ptx_size, '\0');
  err = nvrtcGetPTX(program, &ptx[0]);
  checkErrNVRTC(err, "nvrtcGetPTX()");

  err = nvrtcDestroyProgram(&program);
  checkErrNVRTC(err, "nvrtcDestroyProgram()");

  hipaccCreateModule(module, ptx.c_str(), cc);

  delete[] options;
}

// Get kernel from a module
void hipaccGetKernel(CUfunction &function, CUmodule &module,
                     std::string kernel_name) {
  // get function entry point
  CUresult err = cuModuleGetFunction(&function, module, kernel_name.c_str());
  checkErrDrv(err, "cuModuleGetFunction('" + kernel_name + "')");
}

// Computes occupancy for kernel function
void hipaccPrintKernelOccupancy(CUfunction fun, int tile_size_x,
                                int tile_size_y) {
  CUresult err = CUDA_SUCCESS;
  CUdevice dev = 0;
  int block_size = tile_size_x * tile_size_y;
  size_t dynamic_smem_bytes = 0;

  int warp_size;
  err = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
  checkErrDrv(err, "cuDeviceGetAttribute()");
  int max_threads_per_multiprocessor;
  err = cuDeviceGetAttribute(&max_threads_per_multiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                             dev);
  checkErrDrv(err, "cuDeviceGetAttribute()");

  int active_blocks;
  err = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, fun, block_size, dynamic_smem_bytes);
  checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");
  int active_warps = active_blocks * (block_size / warp_size);
  int max_warps_per_multiprocessor = max_threads_per_multiprocessor / warp_size;
  float occupancy = (float)active_warps / (float)max_warps_per_multiprocessor;

  hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                          ";  occupancy: " + std::to_string(occupancy) + " (" +
                              std::to_string(active_warps) + " out of " +
                              std::to_string(max_warps_per_multiprocessor) +
                              " warps)");
}

// Launch kernel
void hipaccLaunchKernel(CUfunction &kernel, std::string kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing) {
  cudaEvent_t start, end;
  float last_gpu_timing;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  CUresult err = cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x,
                                block.y, block.z, 0, NULL, args, NULL);
  checkErrDrv(err, "cuLaunchKernel(" + kernel_name + ")");
  err = cuCtxSynchronize();
  checkErrDrv(err, "cuLaunchKernel(" + kernel_name + ")");

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&last_gpu_timing, start, end);

  hipaccCudaTiming.set_gpu_timing(last_gpu_timing);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  if (print_timing)
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (" + std::to_string(block.x * block.y) + ": " +
            std::to_string(block.x) + "x" + std::to_string(block.y) +
            "): " + std::to_string(last_gpu_timing) + "(ms)");
}

void hipaccLaunchKernelBenchmark(CUfunction &kernel, std::string kernel_name,
                                 dim3 &grid, dim3 &block, void **args,
                                 bool print_timing) {
  std::vector<float> times;

  for (size_t i = 0; i < HIPACC_NUM_ITERATIONS; i++) {
    hipaccLaunchKernel(kernel, kernel_name, grid, block, args, print_timing);
    times.push_back(hipaccCudaTiming.get_last_kernel_timing());
  }

  std::sort(times.begin(), times.end());
  hipaccCudaTiming.set_gpu_timing(times[times.size() / 2]);

  if (print_timing) {
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing benchmark (" +
            std::to_string(block.x * block.y) + ": " + std::to_string(block.x) +
            "x" + std::to_string(block.y) +
            "): " + std::to_string(hipaccCudaTiming.get_last_kernel_timing()) +
            " | " + std::to_string(times.front()) + " | " +
            std::to_string(times.back()) + " (median(" +
            std::to_string(HIPACC_NUM_ITERATIONS) +
            ") | minimum | maximum) ms" + "(ms)");
  }
}

// Get global reference from module
void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module,
                     std::string global_name) {
  size_t size;
  CUresult err = cuModuleGetGlobal(&global, &size, module, global_name.c_str());
  checkErrDrv(err, "cuModuleGetGlobal()");
}

// Get texture reference from module
void hipaccGetTexRef(CUtexref &tex, CUmodule &module,
                     std::string texture_name) {
  CUresult err = cuModuleGetTexRef(&tex, module, texture_name.c_str());
  checkErrDrv(err, "cuModuleGetTexRef()");
}

// Get surface reference from module
void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module,
                      std::string surface_name) {
  CUresult err = cuModuleGetSurfRef(&surf, module, surface_name.c_str());
  checkErrDrv(err, "cuModuleGetSurfRef()");
}

// Bind texture to linear memory
void hipaccBindTextureDrv(CUtexref &texture, const HipaccImageCuda &img,
                          CUarray_format format, hipaccMemoryType tex_type) {
  checkErrDrv(cuTexRefSetFormat(texture, format, 1), "cuTexRefSetFormat()");
  checkErrDrv(cuTexRefSetFlags(texture, CU_TRSF_READ_AS_INTEGER),
              "cuTexRefSetFlags()");
  switch (tex_type) {
  case hipaccMemoryType::Linear1D:
    checkErrDrv(cuTexRefSetAddress(0, texture,
                                   (CUdeviceptr)img->get_device_memory(),
                                   img->get_pixel_size() * img->get_stride() * img->get_height()),
                "cuTexRefSetAddress()");
    break;
  case hipaccMemoryType::Linear2D:
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = format;
    desc.NumChannels = 1;
    desc.Width = img->get_width();
    desc.Height = img->get_height();
    checkErrDrv(cuTexRefSetAddress2D(texture, &desc,
                                     (CUdeviceptr)img->get_device_memory(),
                                     img->get_pixel_size() * img->get_stride()),
                "cuTexRefSetAddress2D()");
    break;
  case hipaccMemoryType::Array2D:
    checkErrDrv(cuTexRefSetArray(texture, (CUarray)img->get_device_memory(),
                                 CU_TRSA_OVERRIDE_FORMAT),
                "cuTexRefSetArray()");
    break;
  default:
    assert(false && "not a texture");
  }
}

// Bind surface to 2D array
void hipaccBindSurfaceDrv(CUsurfref &surface, const HipaccImageCuda &img) {
  checkErrDrv(cuSurfRefSetArray(surface, (CUarray)img->get_device_memory(), 0),
              "cuSurfRefSetArray()");
}

// Perform configuration exploration for a kernel call
void hipaccLaunchKernelExploration(std::string filename, std::string kernel,
                                   std::vector<void *> &args,
                                   std::vector<hipacc_smem_info> &smems,
                                   std::vector<hipacc_const_info> &consts,
                                   std::vector<hipacc_tex_info *> &texs,
                                   hipacc_launch_info &info, size_t warp_size,
                                   size_t max_threads_per_block,
                                   size_t max_threads_for_kernel,
                                   size_t max_smem_per_block, size_t heu_tx,
                                   size_t heu_ty, int cc) {
  CUresult err = CUDA_SUCCESS;
  size_t opt_tx = warp_size, opt_ty = 1;
  float opt_time = FLT_MAX;

  hipaccRuntimeLogTrivial(
      hipaccRuntimeLogLevel::INFO,
      "<HIPACC:> Exploring configurations for kernel '" + kernel +
          "': configuration provided by heuristic " +
          std::to_string(heu_tx * heu_ty) + " (" + std::to_string(heu_tx) +
          "x" + std::to_string(heu_ty) + "). ");

#ifdef NVML_FOUND
  nvmlReturn_t nvml_err = NVML_SUCCESS;
  nvmlDevice_t nvml_device;
  nvmlEnableState_t nvml_mode;
  bool nvml_power_avail = true;
  unsigned int nvml_device_count, nvml_temperature, nvml_power;

  nvml_err = nvmlInit();
  checkErrNVML(nvml_err, "nvmlInit()");

  nvml_err = nvmlDeviceGetCount(&nvml_device_count);
  checkErrNVML(nvml_err, "nvmlDeviceGetCount()");
  assert(nvml_device_count > 0 && "no device detected by NVML");

  nvml_err = nvmlDeviceGetHandleByIndex(0, &nvml_device);
  checkErrNVML(nvml_err, "nvmlDeviceGetHandleByIndex()");

  nvml_err = nvmlDeviceGetPowerManagementMode(nvml_device, &nvml_mode);
  if (nvml_mode == NVML_FEATURE_DISABLED ||
      nvml_err == NVML_ERROR_NOT_SUPPORTED) {
    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::WARNING,
        "NVML Warning: device does not support querying power usage!");
    nvml_power_avail = false;
  } else {
    checkErrNVML(nvml_err, "nvmlDeviceGetPowerManagementMode()");
  }
#endif

  for (size_t tile_size_x = warp_size; tile_size_x <= max_threads_per_block;
       tile_size_x += warp_size) {
    for (size_t tile_size_y = 1; tile_size_y <= max_threads_per_block;
         ++tile_size_y) {
      // check if we exceed maximum number of threads
      if (tile_size_x * tile_size_y > max_threads_for_kernel)
        continue;

      // check if we exceed size of shared memory
      size_t used_smem = 0;
      for (auto smem : smems)
        used_smem += (tile_size_x + smem.size_x) *
                     (tile_size_y + smem.size_y - 1) * smem.pixel_size;
      if (used_smem >= max_smem_per_block)
        continue;
      if (used_smem && tile_size_x > warp_size)
        continue;

      std::stringstream num_threads_x_ss, num_threads_y_ss;
      num_threads_x_ss << tile_size_x;
      num_threads_y_ss << tile_size_y;

      // compile kernel
      std::vector<std::string> compile_options;
      compile_options.push_back("-I./include");
      compile_options.push_back("-D BSX_EXPLORE=" + num_threads_x_ss.str());
      compile_options.push_back("-D BSY_EXPLORE=" + num_threads_y_ss.str());

      CUmodule modKernel;
      hipaccCompileCUDAToModule(modKernel, filename, cc, compile_options);

      CUfunction exploreKernel;
      hipaccGetKernel(exploreKernel, modKernel, kernel);

      // load constant memory
      CUdeviceptr constMem;
      for (auto cmem : consts) {
        hipaccGetGlobal(constMem, modKernel, cmem.name);
        err = cuMemcpyHtoD(constMem, cmem.memory, cmem.size);
        checkErrDrv(err, "cuMemcpyHtoD()");
      }

      CUtexref texImage;
      CUsurfref surfImage;
      for (auto tex : texs) {
        if (tex->tex_type == hipaccMemoryType::Surface) {
          // bind surface memory
          hipaccGetSurfRef(surfImage, modKernel, tex->name);
          hipaccBindSurfaceDrv(surfImage, tex->image);
        } else {
          // bind texture memory
          hipaccGetTexRef(texImage, modKernel, tex->name);
          hipaccBindTextureDrv(texImage, tex->image, tex->format,
                               tex->tex_type);
        }
      }

      dim3 block(tile_size_x, tile_size_y);
      dim3 grid(hipaccCalcGridFromBlock(info, block));
      hipaccPrepareKernelLaunch(info, block);
      std::vector<float> times;

      for (size_t i = 0; i < HIPACC_NUM_ITERATIONS; ++i) {
        hipaccLaunchKernel(exploreKernel, kernel, grid, block, args.data(),
                           false);
        times.push_back(hipaccCudaTiming.get_last_kernel_timing());
      }

      std::sort(times.begin(), times.end());
      hipaccCudaTiming.set_gpu_timing(times[times.size() / 2]);

      if (hipaccCudaTiming.get_last_kernel_timing() < opt_time) {
        opt_time = hipaccCudaTiming.get_last_kernel_timing();
        opt_tx = tile_size_x;
        opt_ty = tile_size_y;
      }

#ifdef NVML_FOUND
      nvml_err = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU,
                                          &nvml_temperature);
      checkErrNVML(nvml_err, "nvmlDeviceGetTemperature()");
      if (nvml_power_avail) {
        nvml_err = nvmlDeviceGetPowerUsage(nvml_device, &nvml_power);
        checkErrNVML(nvml_err, "nvmlDeviceGetPowerUsage()");
      }
#endif

      // print timing
      hipaccRuntimeLogTrivial(
          hipaccRuntimeLogLevel::INFO,
          "<HIPACC:> Kernel config: " + std::to_string(tile_size_x) + "x" +
              std::to_string(tile_size_y) + "(" +
              std::to_string(tile_size_x * tile_size_y) + "): " +
              std::to_string(hipaccCudaTiming.get_last_kernel_timing()) +
              " | " + std::to_string(times.front()) + " | " +
              std::to_string(times.back()) + " (median(" +
              std::to_string(HIPACC_NUM_ITERATIONS) +
              ") | minimum | maximum) ms");

#ifdef NVML_FOUND
      hipaccRuntimeLogTrivial(
          hipaccRuntimeLogLevel::INFO,
          ";  temperature: " + std::to_string(nvml_temperature) + " °C");
      if (nvml_power_avail)
        hipaccRuntimeLogTrivial(
            hipaccRuntimeLogLevel::INFO,
            ";  power usage: " + std::to_string(nvml_power / 1000.f) + " W");
#endif
      hipaccPrintKernelOccupancy(exploreKernel, tile_size_x, tile_size_y);

      // cleanup
      err = cuModuleUnload(modKernel);
      checkErrDrv(err, "cuModuleUnload()");
    }
  }
  hipaccCudaTiming.set_gpu_timing(opt_time);

  hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::INFO,
                          "<HIPACC:> Best configurations for kernel '" +
                              kernel + "': " + std::to_string(opt_tx * opt_ty) +
                              " (" + std::to_string(opt_tx) + "x" +
                              std::to_string(opt_ty) +
                              "): " + std::to_string(opt_time) + " ms");

#ifdef NVML_FOUND
  nvml_err = nvmlShutdown();
  checkErrNVML(nvml_err, "nvmlShutdown()");
#endif
}

#endif // __HIPACC_CU_STANDALONE_HPP__
