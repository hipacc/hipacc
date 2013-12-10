# Configuration
COMPILER     ?= ./bin/hipacc
COMPILER_INC ?= -std=c++11 -stdlib=libc++ \
                -I`@CLANG_EXECUTABLE@ -print-file-name=include` \
                -I`@LLVM_CONFIG_EXECUTABLE@ --includedir` \
                -I`@LLVM_CONFIG_EXECUTABLE@ --includedir`/c++/v1 \
                -I/usr/include \
                -I@DSL_INCLUDES@
TEST_CASE    ?= ./tests/opencv_blur_8uc1
MYFLAGS      ?= -DWIDTH=2048 -DHEIGHT=2048 -DSIZE_X=5 -DSIZE_Y=5
NVCC_FLAGS    = -gencode=arch=compute_$(GPU_ARCH),code=\"sm_$(GPU_ARCH),compute_$(GPU_ARCH)\" \
                -ftz=true -prec-sqrt=false -prec-div=false -Xptxas \
                -v #-keep
OFLAGS        = -O3

# OpenCL specific configuration
ifeq ($(HIPACC_TARGET),Midgard)
    OCL_CC    = @NDK_CXX_COMPILER@ @NDK_CXX_FLAGS@ @NDK_INCLUDE_DIRS_STR@ -std=c++0x -Wall -Wunused
    OCL_LINK  = -lm -ldl -lstdc++ @NDK_LINK_LIBRARIES_STR@ @EMBEDDED_OPENCL_LFLAGS@
    OCL_INC   = @EMBEDDED_OPENCL_CFLAGS@
else
    OCL_CC    = @CMAKE_CXX_COMPILER@ -std=c++0x -Wall -Wunused
    OCL_LINK  = -lm -ldl -lstdc++ -lpthread @TIME_LINK@ @OPENCL_LFLAGS@
    OCL_INC   = @OPENCL_CFLAGS@
endif

# Renderscript specific configuration
RS_TARGET_API = @RS_TARGET_API@
ifge = $(shell if [ $(1) -ge $(2) ]; then echo true; else echo false; fi)


# Source-to-source compiler configuration
# use local memory -> set HIPACC_LMEM to off|on
# use texture memory -> set HIPACC_TEX to off|Linear1D|Linear2D|Array2D
# vectorize code (experimental, doesn't work) -> set HIPACC_VEC to off|on
# pad images to a multiple of n bytes -> set HIPACC_PAD to n
# map n output pixels to one thread -> set HIPACC_PPT to n
# use specific configuration for kernels -> set HIPACC_CONFIG to nxm
# generate code that explores configuration -> set HIPACC_EXPLORE to off|on
# generate code that times kernel execution -> set HIPACC_TIMING to off|on
HIPACC_LMEM?=off
HIPACC_TEX?=off
HIPACC_VEC?=off
HIPACC_PPT?=1
HIPACC_CONFIG?=128x1
HIPACC_EXPLORE?=off
HIPACC_TIMING?=off
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
ifeq ($(HIPACC_EXPLORE),on)
    HIPACC_OPTS+= -explore-config
endif
ifeq ($(HIPACC_TIMING),on)
    HIPACC_OPTS+= -time-kernels
endif

# set target GPU architecture to the compute capability encoded in target
GPU_ARCH := $(shell echo $(HIPACC_TARGET) |cut -f2 -d-)


all:
run:
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INC)

cuda:
	@echo 'Executing HIPAcc Compiler for CUDA:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INC) -emit-cuda $(HIPACC_OPTS) -o main.cu
	@echo 'Compiling CUDA file using nvcc:'
	@NVCC_COMPILER@ $(NVCC_FLAGS) -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) $(MYFLAGS) $(OFLAGS) -o main_cuda main.cu @CUDA_LINK@
	@echo 'Executing CUDA binary'
	./main_cuda

opencl-acc opencl-cpu opencl-gpu:
	@echo 'Executing HIPAcc Compiler for OpenCL:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INC) -emit-$@ $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	$(OCL_CC) $(OCL_INC) -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) $(MYFLAGS) $(OFLAGS) -o main_opencl main.cc $(OCL_LINK)
ifneq ($(HIPACC_TARGET),Midgard)
	@echo 'Executing OpenCL binary'
	./main_opencl
endif

renderscript:
	rm -f *.rs *.fs
	@echo 'Executing HIPAcc Compiler for Renderscript:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INC) -emit-renderscript $(HIPACC_OPTS) -o main.cc
	mkdir -p build_renderscript
ifeq ($(call ifge, $(RS_TARGET_API), 19), true) # build using ndk-build
	rm -rf build_renderscript/*
	mkdir -p build_renderscript/jni
	cp @CMAKE_CURRENT_SOURCE_DIR@/tests/Android.mk.cmake build_renderscript/jni/Android.mk
	cp main.cc *.rs build_renderscript
	@echo 'Compiling Filterscript file using llvm-rs-cc and g++:'
	export CASE_FLAGS="$(MYFLAGS)"; \
	export RS_TARGET_API=$(RS_TARGET_API); \
	export HIPACC_INCLUDE=@RUNTIME_INCLUDES@; \
	cd build_renderscript; ndk-build -B APP_PLATFORM=android-$(RS_TARGET_API) APP_STL=stlport_static
	cp build_renderscript/libs/armeabi/main_renderscript .
else
	@echo 'Generating build system current test case:'
	cd build_renderscript; cmake .. -DANDROID_SOURCE_DIR=@ANDROID_SOURCE_DIR@ -DTARGET_NAME=@TARGET_NAME@ -DHOST_TYPE=@HOST_TYPE@ -DNDK_TOOLCHAIN_DIR=@NDK_TOOLCHAIN_DIR@ -DRS_TARGET_API=$(RS_TARGET_API) $(MYFLAGS)
	@echo 'Compiling Renderscript file using llvm-rs-cc and g++:'
	cd build_renderscript; make
	cp build_renderscript/main_renderscript .
endif

filterscript:
	rm -f *.rs *.fs
	@echo 'Executing HIPAcc Compiler for Filterscript:'
	$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INC) -emit-filterscript $(HIPACC_OPTS) -o main.cc
	mkdir -p build_filterscript
ifeq ($(call ifge, $(RS_TARGET_API), 19), true) # build using ndk-build
	rm -rf build_filterscript/*
	mkdir -p build_filterscript/jni
	cp @CMAKE_CURRENT_SOURCE_DIR@/tests/Android.mk.cmake build_filterscript/jni/Android.mk
	cp main.cc *.fs build_filterscript
	@echo 'Compiling Filterscript file using llvm-rs-cc and g++:'
	export CASE_FLAGS="$(MYFLAGS)"; \
	export RS_TARGET_API=$(RS_TARGET_API); \
	export HIPACC_INCLUDE=@RUNTIME_INCLUDES@; \
	cd build_filterscript; ndk-build -B APP_PLATFORM=android-$(RS_TARGET_API) APP_STL=stlport_static
	cp build_filterscript/libs/armeabi/main_renderscript main_filterscript
else
	@echo 'Generating build system current test case:'
	cd build_filterscript; cmake .. -DANDROID_SOURCE_DIR=@ANDROID_SOURCE_DIR@ -DTARGET_NAME=@TARGET_NAME@ -DHOST_TYPE=@HOST_TYPE@ -DNDK_TOOLCHAIN_DIR=@NDK_TOOLCHAIN_DIR@ -DRS_TARGET_API=$(RS_TARGET_API) $(MYFLAGS)
	@echo 'Compiling Filterscript file using llvm-rs-cc and g++:'
	cd build_filterscript; make
	cp build_filterscript/main_renderscript ./main_filterscript
endif

clean:
	rm -f main_* *.cu *.cc *.cubin *.cl *.isa *.rs *.fs
	rm -rf build_*

