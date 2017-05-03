LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)

LOCAL_MODULE     := main_renderscript

LOCAL_C_INCLUDES := $(RENDERSCRIPT_PLATFORM_HEADER)/cpp \
                    $(RENDERSCRIPT_PLATFORM_HEADER) \
                    $(TARGET_OBJS)/$(LOCAL_MODULE) \
                    $(HIPACC_INCLUDE)

LOCAL_SRC_FILES  := main.cc \
                   $(wildcard *.rs) \
                   $(wildcard *.fs)

LOCAL_LDFLAGS    += -L$(RENDERSCRIPT_TOOLCHAIN_PREBUILT_ROOT)/platform/$(TARGET_ARCH)
LOCAL_LDLIBS     := -llog -lRScpp_static

LOCAL_CPPFLAGS   += -std=c++11 -Wall -Wextra $(CASE_FLAGS)

LOCAL_RENDERSCRIPT_FLAGS := -allow-rs-prefix -Wno-unused-variable -Wno-unused-function \
                            -I$(HIPACC_INCLUDE)

include $(BUILD_EXECUTABLE)
