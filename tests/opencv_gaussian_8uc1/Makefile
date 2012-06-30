CC = g++

MYFLAGS ?= -D WIDTH=4096 -D HEIGHT=4096 -D SIZE_X=3 -D SIZE_Y=3
CFLAGS = $(MYFLAGS) -Wall -Wunused -I../../include/dsl -D OpenCV -I/opt/local/include/opencv -I/opt/local/include
LDFLAGS = -lstdc++ -lm -L/opt/local/lib -lopencv_core -lopencv_gpu
OFLAGS = -O3


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

