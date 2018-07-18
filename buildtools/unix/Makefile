# Search binaries
CXX       := $(shell which c++)
NVCC      := $(shell which nvcc)
ADB       := $(shell which adb)
NDK_BUILD := $(shell which ndk-build)

################################################################################

# Search Hipacc (prefer relative position shipped with package over PATH)
HIPACC := $(shell if [ -f $(CURDIR)/../../../bin/hipacc ]; then \
                    echo $(CURDIR)/../../../bin/hipacc; \
                  else \
                    echo $$(which hipacc); \
                  fi)
ifndef HIPACC
	$(error "Could not find 'hipacc' in PATH")
endif

################################################################################

# Set paths for build tools
CUDA_PATH   := $(shell dirname $(NVCC))/..
HIPACC_PATH := $(shell dirname $(HIPACC))/..

################################################################################

# Set compiler options
CXX_FLAGS   := -std=c++11 -O2
CXX_INCLUDE := -I $(HIPACC_PATH)/include
CXX_LIB_DIR := -L $(HIPACC_PATH)/lib
CXX_LINK    := -lhipaccRuntime

OCL_INCLUDE := $(CXX_INCLUDE)
OCL_LIB_DIR := $(CXX_LIB_DIR)
OCL_LINK    := $(CXX_LINK) -lOpenCL

ifdef NVCC
NVCC_FLAGS   := -std=c++11 -O2 -x cu
NVCC_INCLUDE := $(CXX_INCLUDE)
NVCC_LIB_DIR := $(CXX_LIB_DIR)
NVCC_LINK    := $(CXX_LINK) -lcuda -lcudart -lnvrtc
OCL_INCLUDE  += -I $(CUDA_PATH)/include
OCL_LIB_DIR  += -L $(CUDA_PATH)/lib64
endif

HIPACC_FLAGS   := -std=c++11
HIPACC_INCLUDE := -I $(HIPACC_PATH)/include/dsl \
                  -I $(HIPACC_PATH)/include/c++/v1 \
                  -I $(HIPACC_PATH)/include/clang

################################################################################

# Turn on GNU Make feature for using automatic variables in dependencies
.SECONDEXPANSION:
.PHONY: cpu cuda opencl-cpu opencl-gpu renderscript filterscript

################################################################################

# Target rules
all: cpu

cpu cuda opencl-cpu opencl-gpu renderscript filterscript: main_$$@

# Run Hipacc
main_%.cc: src/main.cpp
	$(HIPACC) -emit-$* $(HIPACC_FLAGS) $(HIPACC_INCLUDE) $(CURDIR)/$< -o $(shell basename $@)

# Build CPU
main_cpu: $$@.cc
	$(CXX) $(CXX_FLAGS) $< $(CXX_INCLUDE) $(CXX_LIB_DIR) $(CXX_LINK) -o $@
	./$@

# Build CUDA
main_cuda: $$@.cc
ifdef NVCC
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_INCLUDE) $(NVCC_LIB_DIR) $(NVCC_LINK) -o $@
	./$@
else
	@echo "Could not find 'nvcc' in PATH"
endif

# Build OpenCL-CPU or OpenCL-GPU
main_opencl-%: $$@.cc
	$(CXX) $(CXX_FLAGS) $< $(OCL_INCLUDE) $(OCL_LIB_DIR) $(OCL_LINK) -o $@
	./$@

# Build Renderscript or Filterscript
main_%: $$@.cc
	mkdir -p jni
	cp ../../common/Android.mk jni/Android.mk
	cp ../../common/Application.mk jni/Application.mk
ifdef NDK_BUILD
	export HIPACC_MAIN=$@.cc; \
	export HIPACC_INCLUDE=$(HIPACC_PATH)/include; \
	$(NDK_BUILD) -B
	cp libs/armeabi-v7a/main_renderscript ./$@
ifdef ADB
	$(ADB) shell mkdir -p /data/local/tmp
	$(ADB) push $@ /data/local/tmp
	$(ADB) shell /data/local/tmp/$@
else
	@echo "Could not find 'adb' in PATH"
endif
else
	@echo "Could not find 'ndk-build' in PATH"
endif

clean:
	rm -rf jni libs obj
	rm -f *cc *.cu *.cubin *.cl *.rs *.fs

distclean: clean
	rm -f main_*
