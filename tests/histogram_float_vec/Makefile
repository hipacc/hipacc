CC = /usr/bin/c++

HIPACC_DIR ?= /scratch-local/usr

ifeq ($(notdir $(CC)),clang++)
    # use libc++ for clang++
    CFLAGS  = -std=c++11 -stdlib=libc++ \
              -I`/scratch-local/usr-clang50/bin/llvm-config --includedir` \
              -I`/scratch-local/usr-clang50/bin/llvm-config --includedir`/c++/v1 \
              -I`/scratch-local/usr-clang50/bin/clang -print-file-name=include`
    LDFLAGS = -L`/scratch-local/usr-clang50/bin/llvm-config --libdir` -lc++ -lc++abi
else
    CFLAGS  = -std=c++11
    LDFLAGS = -lstdc++
endif

MYFLAGS    ?= -D WIDTH=2048 -D HEIGHT=2048 -D SIZE_X=5 -D SIZE_Y=5
CFLAGS     += -I$(HIPACC_DIR)/include/dsl \
              $(MYFLAGS)  -Wall -Wunused
LDFLAGS    += -lm /usr/lib/x86_64-linux-gnu/librt.so -lpthread
OFLAGS      = -O3

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

