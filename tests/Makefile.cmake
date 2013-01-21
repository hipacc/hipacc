# Configuration
COMPILER            ?= ./bin/hipacc
COMPILER_INCLUDES   ?= @PLATFORM_FIXES@ -std=c++11 -stdlib=libc++ \
                        -I`@CLANG_EXECUTABLE@ -print-file-name=include` \
                        -I`@LLVM_CONFIG_EXECUTABLE@ --includedir` \
                        -I`@LLVM_CONFIG_EXECUTABLE@ --includedir`/c++/v1 \
                        -I/usr/include \
                        -I@DSL_INCLUDES@
TEST_CASE           ?= ./tests/opencv_blur_8uc1
MYFLAGS             ?= -D WIDTH=4096 -D HEIGHT=4096 -D SIZE_X=3 -D SIZE_Y=3
C_C                 ?= 13#69
NVCC_FLAGS          := -gencode=arch=compute_$(C_C),code=\"sm_$(C_C),compute_$(C_C)\" -ftz=true -prec-sqrt=false -prec-div=false -Xptxas -v #-keep 

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


HIPACC_OPTS=-compute-capability $(C_C)
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


CC      := g++
RM      := rm -rf


all:
run:
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES)

cuda:
	@echo 'Executing HIPAcc Compiler for CUDA:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-cuda $(HIPACC_OPTS) -o main.cu
	@echo 'Compiling CUDA file using nvcc:'
	nvcc $(NVCC_FLAGS) -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) $(MYFLAGS) @CUDA_LINK@ -O3 main.cu -o main_cuda
	@echo 'Executing CUDA binary'
	./main_cuda

opencl:
	@echo 'Executing HIPAcc Compiler for OpenCL:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
	@echo 'Executing OpenCL binary'
	./main_opencl

opencl_x86:
	@echo 'Executing HIPAcc Compiler for OpenCL:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-opencl-x86 $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
	@echo 'Executing OpenCL binary'
	./main_opencl

renderscript:
	@echo 'Executing HIPAcc Compiler for Renderscript:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-renderscript $(HIPACC_OPTS) -o -
	cat *.rs

clean:
	-$(RM) main_cuda main_opencl #*.cu *.cubin *.cl *.isa
	-@echo ' '

