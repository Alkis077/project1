CXX      = g++
# Eigen is included with -isystem, so that warnings from inside its headers
# are not shown together with the warnings of our code
CXXFLAGS = -std=c++17 -O2 -fopenmp -Wall -Wextra -Iinclude $(shell pkg-config --cflags opencv4) \
           $(patsubst -I%,-isystem %,$(shell pkg-config --cflags eigen3))
LDLIBS   = $(shell pkg-config --libs opencv4)

# two programs: search (1st assignment) and verify (2nd assignment).
# Every file of src/ is shared, except the file with the main of each program.
MAINS  = src/main.cpp src/verify.cpp
SRCS   = $(filter-out $(MAINS),$(wildcard src/*.cpp))
OBJS   = $(patsubst src/%.cpp,build/%.o,$(SRCS))
DEPS   = $(patsubst src/%.cpp,build/%.d,$(wildcard src/*.cpp)) build/test_geometry.d

all: search verify

search: build/main.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

verify: build/verify.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# checks of the DLT, RANSAC and corner error code (tests/test_geometry.cpp)
TEST_OBJS = build/homography.o build/ransac.o build/groundtruth.o
build/test_geometry: tests/test_geometry.cpp $(TEST_OBJS) | build
	$(CXX) $(CXXFLAGS) -MMD -MP -o $@ $< $(TEST_OBJS) $(LDLIBS)

test: build/test_geometry
	./build/test_geometry

build:
	mkdir -p build

clean:
	rm -rf build search verify

.PHONY: all test clean

-include $(DEPS)
