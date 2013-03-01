# All Target
TARGET              := main_opencl
LDFLAGS             = -lm -ldl -lpthread -lstdc++ @OPENCL_LFLAGS@
INC                 = -I@RUNTIME_INCLUDES@ -I$(TEST_CASE) @OPENCL_CFLAGS@
OFLAGS              := -std=c++0x -O3 -Wall -Wunused


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

