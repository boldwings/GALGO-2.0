CXX = g++
FLAGS = -O3 -g -mavx -mavx2 -mfma -std=c++11 -march=native -Wall -I.

all: mytest.cpp
	$(CXX) $(FLAGS) -o run mytest.cpp

clean:
	rm -f *~ *.o