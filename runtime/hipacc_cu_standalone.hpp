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

inline std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
  const char *error_name(nullptr);
  const char *error_string(nullptr);
  cuGetErrorName(errorCode, &error_name);
  cuGetErrorString(errorCode, &error_string);
  return std::string(error_name) + ": " + std::string(error_string);
}

inline std::string getCUDAErrorCodeStr(cudaError_t errorCode) {
  const char *error_name(nullptr);
  const char *error_string(nullptr);
  error_name = cudaGetErrorName(errorCode);
  error_string = cudaGetErrorString(errorCode);
  return std::string(error_name) + ": " + std::string(error_string);
}

inline HipaccContext &HipaccContext::getInstance() {
  static HipaccContext instance;

  return instance;
}

inline void hipaccPrepareKernelLaunch(hipacc_launch_info &info, dim3 const &block) {
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

inline dim3 hipaccCalcGridFromBlock(hipacc_launch_info const &info, dim3 const &block) {
  return dim3(
      (int)ceilf((float)(info.is_width + info.offset_x) /
                 (block.x * info.simd_width)),
      (int)ceilf((float)(info.is_height) / (block.y * info.pixels_per_thread)));
}

// Initialize CUDA devices
inline void hipaccInitCUDA() {
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

  for (int i = 0; i < device_count; ++i) {
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
template<typename T>
void hipaccCopyMemory(const HipaccImageCuda<T> &src, HipaccImageCuda<T> &dst) {

  assert(src->get_height() == dst->get_height());
  assert(src->get_width() == dst->get_width());
  assert(src->get_pixel_size() == dst->get_pixel_size());

  if (src->get_mem_type() == hipaccMemoryType::Array2D ||
      src->get_mem_type() == hipaccMemoryType::Surface) {
    cudaError_t err = cudaMemcpy2DArrayToArray(
        (cudaArray *)dst->get_device_memory(), 0, 0,
        (cudaArray *)src->get_device_memory(), 0, 0, src->get_stride() * src->get_pixel_size(),
        src->get_height() * src->get_pixel_size(), cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy2DArrayToArray()");
  } else {
    cudaError_t err =
        cudaMemcpy2D( dst->get_device_memory(), dst->get_stride() * dst->get_pixel_size()
                    , src->get_device_memory(), src->get_stride() * src->get_pixel_size()
                    , src->get_width() * src->get_pixel_size(), src->get_height(), cudaMemcpyDeviceToDevice);
    checkErr(err, "cudaMemcpy()");
  }
}

// Copy from memory region to memory region
template<typename T>
void hipaccCopyMemoryRegion(const HipaccAccessor<T> &src,
                            const HipaccAccessor<T> &dst) {
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
inline void hipaccUnbindTexture(const textureReference *tex) {
  cudaError_t err = cudaUnbindTexture(tex);
  checkErr(err, "cudaUnbindTexture()");
}

// Launch kernel
inline void hipaccLaunchKernel(const void *kernel, std::string const& kernel_name, dim3 &grid,
                        dim3 &block, void **args, bool print_timing, const int shared_memory_size) {
  cudaEvent_t start, end;
  float last_gpu_timing;

  if (print_timing)
  {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
  }

  cudaError_t err = cudaLaunchKernel(kernel, grid, block, args, shared_memory_size, 0);
  checkErr(err, "cudaLaunchKernel(" + kernel_name + ")");

  if (print_timing)
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&last_gpu_timing, start, end);

    HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing of " + kernel_name + " (" + std::to_string(block.x * block.y) + ": " +
            std::to_string(block.x) + "x" + std::to_string(block.y) +
            "): " + std::to_string(last_gpu_timing) + "(ms)");
  }
}

namespace detail
{
  template <typename F, typename... Args>
  struct is_invocable
      : std::is_constructible<
            std::function<void(Args...)>,
            std::reference_wrapper<typename std::remove_reference<F>::type>> {};

  inline void collect_argument_addresses(void **collected_addresses) {}

  template <typename Arg, typename... Args>
  void collect_argument_addresses(void **collected_addresses, Arg &&arg, Args &&... args)
  {
    collected_addresses[0] = const_cast<void*>(static_cast<void const*>(&arg));
    collect_argument_addresses(collected_addresses + 1, std::forward<Args>(args)...);
  }
} // namespace detail

template <typename KernelFunction, typename... KernelParameters>
void hipaccLaunchKernel(KernelFunction const &kernel_function, dim3 const &gridDim,
                        dim3 const &blockDim, HipaccExecutionParameterCuda const& ep,
                        bool print_timing, size_t shared_memory, KernelParameters &&... parameters)
{
  constexpr auto non_zero_num_params = sizeof...(KernelParameters) == 0 ? 1 : sizeof...(KernelParameters);
  void *argument_ptrs[non_zero_num_params];
  detail::collect_argument_addresses(argument_ptrs, std::forward<KernelParameters>(parameters)...);

  static_assert(
      detail::is_invocable<KernelFunction, KernelParameters...>::value,
      "mismatch of kernel parameters");

  cudaEvent_t start, end;
  float last_gpu_timing;

  cudaStream_t stream{ ep ? ep->get_stream() : 0 };

  if (ep) ep->pre_kernel();

  if (print_timing)
  {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
  }

  cudaError_t err = cudaLaunchKernel(
      reinterpret_cast<void const *>(&kernel_function), gridDim, blockDim,
      &(argument_ptrs[0]), shared_memory, stream);

  checkErr(err, (std::string("cudaLaunchKernel(") + __FUNCTION__ + ")"));

  if (print_timing)
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&last_gpu_timing, start, end);

    HipaccKernelTimingBase::getInstance().set_timing(last_gpu_timing);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    hipaccRuntimeLogTrivial(
        hipaccRuntimeLogLevel::INFO,
        "<HIPACC:> Kernel timing (" + std::to_string(blockDim.x * blockDim.y) + ": " +
            std::to_string(blockDim.x) + "x" + std::to_string(blockDim.y) +
            "): " + std::to_string(last_gpu_timing) + "(ms)");
  }

  if (ep) ep->post_kernel();
}

template <typename KernelFunction, typename... KernelParameters>
void hipaccLaunchKernelCudaGraph(KernelFunction const &kernel_function, dim3 const &gridDim, dim3 const &blockDim, HipaccExecutionParameterCuda const& ep, cudaGraph_t &graph, cudaGraphNode_t &graphNode, std::vector<cudaGraphNode_t> &graphNodeDeps, cudaKernelNodeParams &kernelNodeArgs, size_t shared_memory, KernelParameters &&... parameters)
{
  constexpr auto non_zero_num_params = sizeof...(KernelParameters) == 0 ? 1 : sizeof...(KernelParameters);
  void *argument_ptrs[non_zero_num_params];
  detail::collect_argument_addresses(argument_ptrs, std::forward<KernelParameters>(parameters)...);

  static_assert(
      detail::is_invocable<KernelFunction, KernelParameters...>::value,
      "mismatch of kernel parameters");

  if (ep) ep->pre_kernel();

  kernelNodeArgs.func = reinterpret_cast<void *>(&kernel_function);
  kernelNodeArgs.gridDim = gridDim;
  kernelNodeArgs.blockDim = blockDim;
  kernelNodeArgs.sharedMemBytes = shared_memory;
  kernelNodeArgs.kernelParams = &(argument_ptrs[0]);
  kernelNodeArgs.extra = NULL;

  cudaError_t err = cudaGraphAddKernelNode(&graphNode, graph, graphNodeDeps.data(),
      graphNodeDeps.size(), &kernelNodeArgs);
  checkErr(err, (std::string("cudaGraphAddKernelNode(") + __FUNCTION__ + ")"));

  if (ep) ep->post_kernel();
}

//
// DRIVER API
//

// Create a module from ptx assembly
inline void hipaccCreateModule(CUmodule &module, const void *ptx, int cc) {
  CUjit_target target_cc = (CUjit_target)cc;
  const unsigned int opt_level = 4;
  const unsigned int error_log_size = 10240;
  const unsigned int num_options = 4;
  char error_log_buffer[error_log_size] = {0};

  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *option_values[] = {(void *)error_log_buffer, (void *)&error_log_size,
                           (void *)&target_cc, (void *)&opt_level};

  CUresult err =
      cuModuleLoadDataEx(&module, ptx, num_options, options, option_values);
  if (err != CUDA_SUCCESS)
    hipaccRuntimeLogTrivial(hipaccRuntimeLogLevel::WARNING,
                            "Error log: " + std::string(error_log_buffer));
  checkErrDrv(err, "cuModuleLoadDataEx()");
}

// Compile CUDA source file and create module
inline void hipaccCompileCUDAToModule(CUmodule &module, std::string file_name, int cc,
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
  int num_options = static_cast<int>(build_options.size()) + offset;
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
inline void hipaccGetKernel(CUfunction &function, CUmodule &module,
                     std::string kernel_name) {
  // get function entry point
  CUresult err = cuModuleGetFunction(&function, module, kernel_name.c_str());
  checkErrDrv(err, "cuModuleGetFunction('" + kernel_name + "')");
}

// Computes occupancy for kernel function
inline void hipaccPrintKernelOccupancy(CUfunction fun, int tile_size_x,
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

// Get global reference from module
inline void hipaccGetGlobal(CUdeviceptr &global, CUmodule &module,
                     std::string global_name) {
  size_t size;
  CUresult err = cuModuleGetGlobal(&global, &size, module, global_name.c_str());
  checkErrDrv(err, "cuModuleGetGlobal()");
}

// Get texture reference from module
inline void hipaccGetTexRef(CUtexref &tex, CUmodule &module,
                     std::string texture_name) {
  CUresult err = cuModuleGetTexRef(&tex, module, texture_name.c_str());
  checkErrDrv(err, "cuModuleGetTexRef()");
}

// Get surface reference from module
inline void hipaccGetSurfRef(CUsurfref &surf, CUmodule &module,
                      std::string surface_name) {
  CUresult err = cuModuleGetSurfRef(&surf, module, surface_name.c_str());
  checkErrDrv(err, "cuModuleGetSurfRef()");
}

// Bind texture to linear memory
template<typename T>
void hipaccBindTextureDrv(CUtexref &texture, const HipaccImageCuda<T> &img,
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
template<typename T>
void hipaccBindSurfaceDrv(CUsurfref &surface, const HipaccImageCuda<T> &img) {
  checkErrDrv(cuSurfRefSetArray(surface, (CUarray)img->get_device_memory(), 0),
              "cuSurfRefSetArray()");
}

#endif // __HIPACC_CU_STANDALONE_HPP__
