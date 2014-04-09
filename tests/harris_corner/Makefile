CC = clang++
CC = g++

HIPACC_DIR   ?= /scratch-local/membarth/sources/hipacc/build/release
OPENCV_DIR   ?= /scratch-local/membarth/local

MYFLAGS      ?= -D WIDTH=4096 -D HEIGHT=4096 -D SIZE_X=5 -D SIZE_Y=5 -D OpenCV -D CPU
CFLAGS        = $(MYFLAGS) -Wall -Wunused \
                -I$(HIPACC_DIR)/include/dsl \
                -I$(OPENCV_DIR)/include
LDFLAGS       = -lm \
                -L$(OPENCV_DIR)/lib -lopencv_core -lopencv_gpu -lopencv_imgproc -lopencv_highgui
OFLAGS        = -O3

ifeq ($(CC),clang++)
    # use libc++ for clang++
    CFLAGS   += -std=c++11 -stdlib=libc++ \
                -I`clang -print-file-name=include` \
                -I`llvm-config --includedir` \
                -I`llvm-config --includedir`/c++/v1
    LDFLAGS  += -L`llvm-config --libdir` -lc++
else
    CFLAGS   += -std=c++0x
    LDFLAGS  += -lstdc++
endif


BINARY = test
BINDIR = bin
OBJDIR = obj
SOURCES = $(shell echo *.cpp)

OBJS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)
BIN = $(BINDIR)/$(BINARY)


all: $(BINARY)

$(BINARY): $(OBJS) $(BINDIR)
	$(CC) -o $(BINDIR)/$@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp $(OBJDIR)
	$(CC) $(CFLAGS) $(OFLAGS) -o $@ -c $<

$(BINDIR):
	mkdir bin

$(OBJDIR):
	mkdir obj


clean:
	rm -f $(BIN) $(OBJS)
	@echo "all cleaned up!"

distclean: clean
	rm -rf $(BINDIR) $(OBJDIR)

run: $(BINARY)
	$(BIN)

