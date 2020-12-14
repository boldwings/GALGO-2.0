CXX = g++
FLAGS = -O3 -g -mavx -mavx2 -mfma -std=c++11 -march=native -Wall -I.
FLAGS_OMP = -O3 -g -mavx -mavx2 -mfma -fopenmp -std=c++11 -march=native -Wall -I.

run_serial: mytest.cpp
	$(CXX) $(FLAGS) -o run mytest.cpp
	./run


run_omp: mytest.cpp
	$(CXX) $(FLAGS_OMP) -o run mytest.cpp
	./run

clean:
	rm -f *~ *.o