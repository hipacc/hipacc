# All Target
TARGET              := main_opencl
DSL_INC_DIR         ?= include
OPENCL_LIB_DIR      = @OPENCL_LIBRARY_DIR@
OPENCL_INC_DIR      = @OPENCL_INCLUDE_DIR@
LDFLAGS             = -lm -ldl -lpthread -lstdc++ @OPENCL_LFLAGS@
INC                 = -I$(DSL_INC_DIR) -I$(TEST_CASE) @OPENCL_CFLAGS@
OFLAGS              := -O3 -Wall -Wunused

CC      := g++
RM      := rm -rf


all: $(TARGET)

run: all
	./$(TARGET)

$(TARGET): main.cc
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C Compiler/Linker'
	$(CC) $(MYFLAGS) $(OFLAGS) $(INC) -o $@ $< $(LDFLAGS)


# other Targets
clean:
	-$(RM) $(TARGET) 
	-@echo ' '

