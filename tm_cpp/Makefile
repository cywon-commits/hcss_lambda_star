CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -fopenmp -Wall
tm_gen: tm_gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f tm_gen
