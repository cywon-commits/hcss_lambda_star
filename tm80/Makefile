CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -fopenmp -Wall
tm80: tm_gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f tm80
