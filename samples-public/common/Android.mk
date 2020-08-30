LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)

LOCAL_MODULE     := main_renderscript

LOCAL_C_INCLUDES += ../../common \
                    $(HIPACC_INCLUDE)

LOCAL_SRC_FILES  := $(HIPACC_MAIN) \
                    $(wildcard *.rs) \
                    $(wildcard *.fs)

LOCAL_CPPFLAGS   += -std=c++11 -O2 -Wall -Wextra
LOCAL_LDFLAGS    += -L$(RENDERSCRIPT_TOOLCHAIN_PREBUILT_ROOT)/platform/$(TARGET_ARCH)
LOCAL_LDLIBS     := -llog

LOCAL_STATIC_LIBRARIES   := RScpp_static
LOCAL_RENDERSCRIPT_FLAGS := -allow-rs-prefix -Wno-unused-variable -Wno-unused-function \
                            -I$(HIPACC_INCLUDE)

include $(BUILD_EXECUTABLE)

$(call import-module,android/renderscript)
