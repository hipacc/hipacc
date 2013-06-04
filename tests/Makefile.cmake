# Configuration
COMPILER            ?= ./bin/hipacc
COMPILER_INCLUDES   ?= @PLATFORM_FIXES@ -std=c++11 -stdlib=libc++ \
                        -I`@CLANG_EXECUTABLE@ -print-file-name=include` \
                        -I`@LLVM_CONFIG_EXECUTABLE@ --includedir` \
                        -I`@LLVM_CONFIG_EXECUTABLE@ --includedir`/c++/v1 \
                        -I/usr/include \
                        -I@DSL_INCLUDES@
TEST_CASE           ?= ./tests/opencv_blur_8uc1
MYFLAGS             ?= -DWIDTH=2048 -DHEIGHT=2048 -DSIZE_X=5 -DSIZE_Y=5
NVCC_FLAGS          = -gencode=arch=compute_$(GPU_ARCH),code=\"sm_$(GPU_ARCH),compute_$(GPU_ARCH)\" \
                        -ftz=true -prec-sqrt=false -prec-div=false -Xptxas \
                        -v #-keep

# Source-to-source compiler configuration
# use local memory -> set HIPACC_LMEM to off|Linear1D|Linear2D|Array2D
# use texture memory -> set HIPACC_TEX to off|on
# vectorize code (experimental, doesn't work) -> set HIPACC_VEC to off|on
# pad images to a multiple of n bytes -> set HIPACC_PAD to n
# map n output pixels to one thread -> set HIPACC_PPT to n
# use specific configuration for kernels -> set HIPACC_CONFIG to nxm
# generate code that explores configuration -> set HIPACC_EXPLORE
# generate code that times kernel execution -> set HIPACC_TIMING
HIPACC_LMEM?=off
HIPACC_TEX?=off
HIPACC_VEC?=off
HIPACC_PPT?=1
HIPACC_CONFIG?=128x1
HIPACC_EXPLORE?=0
HIPACC_TIMING?=0
HIPACC_TARGET?=Tesla-13


HIPACC_OPTS=-target $(HIPACC_TARGET)
ifdef HIPACC_PAD
    HIPACC_OPTS+= -emit-padding $(HIPACC_PAD)
endif
ifdef HIPACC_LMEM
    HIPACC_OPTS+= -use-local $(HIPACC_LMEM)
endif
ifdef HIPACC_TEX
    HIPACC_OPTS+= -use-textures $(HIPACC_TEX)
endif
ifdef HIPACC_PPT
    HIPACC_OPTS+= -pixels-per-thread $(HIPACC_PPT)
endif
ifdef HIPACC_VEC
    HIPACC_OPTS+= -vectorize $(HIPACC_VEC)
endif
ifdef HIPACC_CONFIG
    HIPACC_OPTS+= -use-config $(HIPACC_CONFIG)
endif
ifeq ($(HIPACC_EXPLORE),1)
    HIPACC_OPTS+= -explore-config
endif
ifeq ($(HIPACC_TIMING),1)
    HIPACC_OPTS+= -time-kernels
endif

# set target GPU architecture to the compute capability encoded in target
GPU_ARCH := $(shell echo $(HIPACC_TARGET) |cut -f2 -d-)


all:
run:
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES)

cuda:
	@echo 'Executing HIPAcc Compiler for CUDA:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-cuda $(HIPACC_OPTS) -o main.cu
	@echo 'Compiling CUDA file using nvcc:'
	@NVCC_COMPILER@ $(NVCC_FLAGS) -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) $(MYFLAGS) @CUDA_LINK@ -O3 main.cu -o main_cuda
	@echo 'Executing CUDA binary'
	./main_cuda

opencl:
	@echo 'Executing HIPAcc Compiler for OpenCL:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
ifneq ($(HIPACC_TARGET),Midgard)
	@echo 'Executing OpenCL binary'
	./main_opencl
endif

opencl_cpu:
	@echo 'Executing HIPAcc Compiler for OpenCL:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-opencl-cpu $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
ifneq ($(HIPACC_TARGET),Midgard)
	@echo 'Executing OpenCL binary'
	./main_opencl
endif

renderscript:
	rm -f *.rs *.fs
	@echo 'Executing HIPAcc Compiler for Renderscript:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-renderscript $(HIPACC_OPTS) -o main.cc
	mkdir -p build_renderscript
	@echo 'Generating build system current test case:'
	cd build_renderscript; cmake .. -DANDROID_SOURCE_DIR=@ANDROID_SOURCE_DIR@ -DTARGET_NAME=@TARGET_NAME@ -DHOST_TYPE=@HOST_TYPE@ -DNDK_TOOLCHAIN_DIR=@NDK_TOOLCHAIN_DIR@ $(MYFLAGS)
	@echo 'Compiling Renderscript file using llvm-rs-cc and g++:'
	cd build_renderscript; make
	cp build_renderscript/main_renderscript .

renderscript_gpu:
	rm -f *.rs *.fs
	@echo 'Executing HIPAcc Compiler for Renderscript:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-renderscript-gpu $(HIPACC_OPTS) -o main.cc
	mkdir -p build_renderscript
	@echo 'Generating build system current test case:'
	cd build_renderscript; cmake .. -DANDROID_SOURCE_DIR=@ANDROID_SOURCE_DIR@ -DTARGET_NAME=@TARGET_NAME@ -DHOST_TYPE=@HOST_TYPE@ -DNDK_TOOLCHAIN_DIR=@NDK_TOOLCHAIN_DIR@ -DRS_TARGET_API=17 $(MYFLAGS)
	@echo 'Compiling Renderscript file using llvm-rs-cc and g++:'
	cd build_renderscript; make
	cp build_renderscript/main_renderscript .

filterscript:
	rm -f *.rs *.fs
	@echo 'Executing HIPAcc Compiler for Filterscript:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-filterscript $(HIPACC_OPTS) -o main.cc
	mkdir -p build_filterscript
	@echo 'Generating build system current test case:'
	cd build_filterscript; cmake .. -DANDROID_SOURCE_DIR=@ANDROID_SOURCE_DIR@ -DTARGET_NAME=@TARGET_NAME@ -DHOST_TYPE=@HOST_TYPE@ -DNDK_TOOLCHAIN_DIR=@NDK_TOOLCHAIN_DIR@ -DRS_TARGET_API=17 $(MYFLAGS)
	@echo 'Compiling Filterscript file using llvm-rs-cc and g++:'
	cd build_filterscript; make
	cp build_filterscript/main_renderscript ./main_filterscript

clean:
	rm -f main_* *.cu *.cc *.cubin *.cl *.isa *.rs *.fs
	rm -rf build_*

