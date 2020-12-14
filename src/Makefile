CXX = g++
FLAGS = -fopenmp -O3 -mavx -mavx2 -mfma -std=c++11 -march=native -Wall -I.

all: mytest.cpp
	$(CXX) $(FLAGS) -o run mytest.cpp

example: example.cpp
	$(CXX) $(FLAGS) -o example example.cpp

clean:
	rm -f *~ *.o