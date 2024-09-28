CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
DEBUGFLAGS = -g -O0 -DDEBUG
RELEASEFLAGS = -O2 -DNDEBUG
INCLUDES = ../include
SRCDIR = ./src
SRCS = $(SRCDIR)/main.cpp $(SRCDIR)/tensor.cpp $(SRCDIR)/model.cpp $(SRCDIR)/utils.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = myprogram

# Default to release build
all: release

# Debug build
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: $(TARGET)

# Release build
release: CXXFLAGS += $(RELEASEFLAGS)
release: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(INCLUDES) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all debug release clean