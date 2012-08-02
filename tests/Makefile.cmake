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
# use local memory -> set HIPACC_LMEM
# use texture memory -> set HIPACC_TEX
# vectorize code (experimental, doesn't work) -> set HIPACC_VEC
# pad images to a multiple of n bytes -> set HIPACC_PAD to n
# map n output pixels to one thread -> set HIPACC_PPT to n
# generate code that explores configuration -> set HIPACC_EXPLORE
HIPACC_LMEM?=0
HIPACC_TEX?=0
HIPACC_PPT?=1
HIPACC_VEC?=0
HIPACC_EXPLORE?=0
HIPACC_TIMING?=0


HIPACC_OPTS=-compute-capability $(C_C)
ifdef HIPACC_PAD
    HIPACC_OPTS+= -emit-padding $(HIPACC_PAD)
endif
ifeq ($(HIPACC_LMEM),1)
    HIPACC_OPTS+= -use-local
endif
ifeq ($(HIPACC_TEX),1)
    HIPACC_OPTS+= -use-textures
endif
ifdef HIPACC_PPT
    HIPACC_OPTS+= -pixels-per-thread $(HIPACC_PPT)
endif
ifeq ($(HIPACC_VEC),1)
    HIPACC_OPTS+= -vectorize
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
	@echo 'Executing HIPACC Compiler for CUDA:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-cuda $(HIPACC_OPTS) -o main.cu
	@echo 'Compiling CUDA file using nvcc:'
	nvcc $(NVCC_FLAGS) -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) $(MYFLAGS) -lcuda -O3 main.cu -o main_cuda
	@echo 'Executing CUDA binary'
	./main_cuda

opencl:
	@echo 'Executing HIPACC Compiler for OpenCL:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
	@echo 'Executing OpenCL binary'
	./main_opencl

opencl_x86:
	@echo 'Executing HIPACC Compiler for OpenCL:'
	./$(COMPILER) $(TEST_CASE)/main.cpp $(MYFLAGS) $(COMPILER_INCLUDES) -emit-opencl-x86 $(HIPACC_OPTS) -o main.cc
	@echo 'Compiling OpenCL file using g++:'
	TEST_CASE=$(TEST_CASE) make -f Makefile_CL MYFLAGS="$(MYFLAGS)"
	@echo 'Executing OpenCL binary'
	./main_opencl

clean:
	-$(RM) main_cuda main_opencl #*.cu *.cubin *.cl *.isa
	-@echo ' '

