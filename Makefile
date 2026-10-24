CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude $(shell pkg-config --cflags opencv4)
LDLIBS   = $(shell pkg-config --libs opencv4)

TARGET = search
SRCS   = $(wildcard src/*.cpp)
OBJS   = $(patsubst src/%.cpp,build/%.o,$(SRCS))
DEPS   = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build $(TARGET)

.PHONY: all clean

-include $(DEPS)
