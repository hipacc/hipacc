LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)

LOCAL_MODULE     := main_renderscript

LOCAL_C_INCLUDES := $(TARGET_C_INCLUDES)/rs/cpp \
                    $(TARGET_C_INCLUDES)/rs \
                    $(TARGET_OBJS)/$(LOCAL_MODULE) \
                    $(HIPACC_INCLUDE)

LOCAL_SRC_FILES  := main.cc \
                   $(wildcard *.rs) \
                   $(wildcard *.fs)

LOCAL_LDFLAGS    += -L$(SYSROOT_LINK)/usr/lib/rs
LOCAL_LDLIBS     := -llog -lRScpp_static

LOCAL_CPPFLAGS   += $(CASE_FLAGS)

LOCAL_RENDERSCRIPT_FLAGS := -allow-rs-prefix -target-api $(RS_TARGET_API) \
                            -I$(HIPACC_INCLUDE)

include $(BUILD_EXECUTABLE)
