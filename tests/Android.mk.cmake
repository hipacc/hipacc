LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)

LOCAL_MODULE := main_renderscript

LOCAL_C_INCLUDES := $(SYSROOT_LINK)/usr/include/rs/cpp \
                    $(SYSROOT_LINK)/usr/include/rs \
                    obj/local/armeabi/objs/$(LOCAL_MODULE) \
                    $(HIPACC_INCLUDE)

LOCAL_SRC_FILES := main.cc \
                   $(wildcard *.rs) \
                   $(wildcard *.fs)

LOCAL_LDLIBS := -llog \
                -l$(SYSROOT_LINK)/usr/lib/rs/libcutils.so \
                -l$(SYSROOT_LINK)/usr/lib/rs/libRScpp_static.a

LOCAL_CPPFLAGS += -DRS_TARGET_API=$(RS_TARGET_API) $(CASE_FLAGS)

LOCAL_RENDERSCRIPT_FLAGS := -allow-rs-prefix -target-api $(RS_TARGET_API) \
                            -I$(HIPACC_INCLUDE)

LOCAL_ARM_MODE := arm

include $(BUILD_EXECUTABLE)
