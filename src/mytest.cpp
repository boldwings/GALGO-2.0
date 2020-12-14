//=================================================================================================
//                    Copyright (C) 2017 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#include "Galgo.hpp"
#include <stdlib.h>
#include <chrono>
int count = 0;
// static __inline__ unsigned long long rdtsc(void) {
//   unsigned hi, lo;
//   __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
//   return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
// }
unsigned long long dur;
// unsigned long long t0, t1;
int num = 1;
// objective class example
template <typename T>
class MyObjective
{
public:
   // objective function example : Rosenbrock function
   // minimizing f(x,y) = (1 - x)^2 + 100 * (y - x^2)^2
   static std::vector<T> Objective(const std::vector<T>& x)
   {
      count++;
      t0 = rdtsc();
      T obj = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
      // T obj = -45.0*sqrt(x[0]+x[1])*sin((15.0*(x[0]+x[1]))/(x[0]*x[0]+x[1]*x[1]));
      // T obj2 = -(pow(1-x[0],2)+ pow(x[1]-1,2));
      // T obj3 = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
      // T obj4 = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
      // T obj5 = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
      t1 = rdtsc();
      dur += t1 - t0;
      // return {obj1, obj2, obj3, obj4, obj5};
      return {obj};
   }
   // NB: GALGO maximize by default so we will maximize -f(x,y)
};

// constraints example:
// 1) x * y + x - y + 1.5 <= 0
// 2) 10 - x * y <= 0
template <typename T>
std::vector<T> MyConstraint(const std::vector<T>& x)
{
   return {x[0]*x[1]+x[0]-x[1]+1.5,10-x[0]*x[1]};
}
// NB: a penalty will be applied if one of the constraints is > 0 
// using the default adaptation to constraint(s) method

int main(int argc, char** argv)
{
   const uint64_t seed1[4] = {0X922AC4EB35B502D9L, 0X7391BDE169361A0BL, 0XB69327491DAC38A9L, 0X240654AD41BB551AL};
   const uint64_t seed2[4] = {0XDA3AA4832B8F1D27L, 0X842ACE731803BAA5L, 0XA8B4384486CD1970L, 0XA8495956C4140021L};
   const uint64_t seed3[4] = {0X32814670dccaad26L, 0Xb9be593f2d1b80a9L, 0X5634e877768b3459L, 0X80fce18a26a1c78fL};
   const uint64_t seed4[4] = {0Xaec4b82be03abed7L, 0Xd1b9b0a0a632873cL, 0X23035878ee74dfc1L, 0X65dde5fde9d025a7L};
   s0 = _mm256_lddqu_si256((const __m256i*)seed1);
	s1 = _mm256_lddqu_si256((const __m256i*)seed2);
   ss0 = _mm256_lddqu_si256((const __m256i*)seed3);
	ss1 = _mm256_lddqu_si256((const __m256i*)seed4);
   // initializing parameters lower and upper bounds
   // an initial value can be added inside the initializer list after the upper bound
   galgo::Parameter<float> par1({0.0,1.0});
   galgo::Parameter<float> par2({0.0,13.0});
   // here both parameter will be encoded using 16 bits the default value inside the template declaration
   // this value can be modified but has to remain between 1 and 64

   // initiliazing genetic algorithm
   // num = atoi(argv[1]);
   std::cout << "num is " << num << std::endl;
   // float rand_num = 
   galgo::GeneticAlgorithm<float> ga(MyObjective<float>::Objective,2000,100,true,par1,par2);

   // setting constraints
   // ga.Constraint = MyConstraint;

   // running genetic algorithm
   ga.output = 1;
   ga.run();
   std::cout << "total time and count is: " << dur << " " << count << std::endl;
   std::cout<<"cross_over dur and count: " << dur_crossover << " " << count_crossover << std::endl;
   std::cout<<"mutation dur and count: " << dur_mutation << " " << count_mutation << std::endl;
}
