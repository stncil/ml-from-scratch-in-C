CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = ./include
SRCDIR = ./src
SRCS = $(SRCDIR)/main.cpp $(SRCDIR)/tensor.cpp $(SRCDIR)/operations.cpp $(SRCDIR)/model.cpp $(SRCDIR)/utils.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = myprogram

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
