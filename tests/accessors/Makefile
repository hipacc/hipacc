CC = clang++
CC = g++

MYFLAGS ?= -D WIDTH=1024 -D HEIGHT=1024
CFLAGS = $(MYFLAGS) -Wall -Wunused -I../../include/dsl
LDFLAGS = -lm
OFLAGS = -O3

ifeq ($(CC),clang++)
# use libc++ for clang++ and use NO_BOOST
CFLAGS += -D NO_BOOST -std=c++11 -stdlib=libc++ -nostdinc -I`clang -print-file-name=include` -I`llvm-config --includedir`/c++/v1 -I`llvm-config --includedir` -I/usr/include
LDFLAGS += -L`llvm-config --libdir` -lc++
else
CFLAGS += -std=c++0x
LDFLAGS += -lstdc++
endif


BINARY = accessors
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

