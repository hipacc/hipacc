CC = clang++
CC = g++

MYFLAGS ?= -D WIDTH=4096 -D HEIGHT=4096 -D SIZE_X=5 -D SIZE_Y=5
CFLAGS = $(MYFLAGS) -Wall -Wunused -I../../include/dsl -D OpenCV -I/opt/local/include/opencv -I/opt/local/include
LDFLAGS = -lm -L/opt/local/lib -lopencv_core -lopencv_gpu -lopencv_imgproc
OFLAGS = -O3

ifeq ($(CC),clang++)
# use libc++ for clang++ and use NO_BOOST
CFLAGS += -D NO_BOOST -std=c++11 -stdlib=libc++ -nostdinc -I`clang -print-file-name=include` -I`llvm-config --includedir`/c++/v1 -I`llvm-config --includedir` -I/usr/include
LDFLAGS += -L`llvm-config --libdir` -lc++
else
CFLAGS += -std=c++0x
LDFLAGS += -lstdc++ 
endif


BINARY = opencv_test
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

