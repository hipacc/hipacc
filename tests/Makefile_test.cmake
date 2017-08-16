CC = @CMAKE_CXX_COMPILER@

HIPACC_DIR ?= @CMAKE_INSTALL_PREFIX@

ifeq ($(notdir $(CC)),clang++)
    # use libc++ for clang++
    CFLAGS  = -std=c++11 -stdlib=libc++ \
              -I`@llvm-config@ --includedir` \
              -I`@llvm-config@ --includedir`/c++/v1 \
              -I`@clang@ -print-file-name=include`
    LDFLAGS = -L`@llvm-config@ --libdir` -lc++ -lc++abi
else
    CFLAGS  = -std=c++11
    LDFLAGS = -lstdc++
endif

MYFLAGS    ?= -D WIDTH=2048 -D HEIGHT=2048 -D SIZE_X=5 -D SIZE_Y=5
CFLAGS     += -I$(HIPACC_DIR)/include/dsl \
              $(MYFLAGS) @THREADS_ARG@ -Wall -Wunused
LDFLAGS    += -lm @RT_LIBRARIES@ @THREADS_LINK@
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

