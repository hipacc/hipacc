# All Target
TARGET              := main_opencl
LDFLAGS             = -lm -ldl -lstdc++
INC                 = -I@RUNTIME_INCLUDES@ -I$(TEST_CASE)
OFLAGS              := -std=c++0x -O3 -Wall -Wunused

ifeq ($(HIPACC_TARGET),Midgard)
    CC = @NDK_CXX_COMPILER@ @NDK_CXX_FLAGS@ @NDK_INCLUDE_DIRS_STR@
    LDFLAGS += @NDK_LINK_LIBRARIES_STR@ @EMBEDDED_OPENCL_LFLAGS@
    INC += @EMBEDDED_OPENCL_CFLAGS@
else
    LDFLAGS += -lpthread @TIME_LINK@ @OPENCL_LFLAGS@
    INC += @OPENCL_CFLAGS@
    CC = @CMAKE_CXX_COMPILER@
endif


all: $(TARGET)

run: all
	./$(TARGET)

$(TARGET): main.cc
	@echo 'Building target: $@'
	$(CC) $(MYFLAGS) $(OFLAGS) $(INC) -o $@ $< $(LDFLAGS)

