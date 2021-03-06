//=================================================================================================
//                    Copyright (C) 2017 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#include "Galgo.hpp"
#include <stdlib.h>
#include <chrono>
int count = 0;
// // static __inline__ unsigned long long rdtsc(void) {
// //   unsigned hi, lo;
// //   __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
// //   return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
// // }
unsigned long long dur;
unsigned long long ts, te;
int num = 1;
float constant1 = 1.0;
float constant100 = 100.0;
float constant_1 = -1.0;
int count_peak = 0;
unsigned long long dur_peak =0;
unsigned long long dur_orig_obj, t2, t3;

// objective class example
template <typename T>
class MyObjective
{
public:
   // theoratical peak 32 Flops/cycle
   static void ObjectiveSIMD(T *x, T *y, T *output)
   {
      // count_peak++;
      // t2 = rdtsc();
      // use 14 simd registers
      __m256 x1 = _mm256_loadu_ps(x);
      __m256 y1 = _mm256_loadu_ps(y);
      __m256 x2 = _mm256_loadu_ps(x + 8);
      __m256 y2 = _mm256_loadu_ps(y + 8);
      __m256 x3 = _mm256_loadu_ps(x + 16);
      __m256 y3 = _mm256_loadu_ps(y + 16);

      __m256 result1 = _mm256_loadu_ps(output);
      __m256 result2 = _mm256_loadu_ps(output + 8);
      __m256 result3 = _mm256_loadu_ps(output + 16);

      /* const 1, 100 */
      __m256 c1 = _mm256_broadcast_ss(&constant1);
      __m256 c2 = _mm256_broadcast_ss(&constant100);
      __m256 c3 = _mm256_broadcast_ss(&constant_1);

      /* (y -x^2)^2 */
      __m256 z1 = _mm256_mul_ps(x1, x1);
      y1 = _mm256_sub_ps(y1, z1);
      y1 = _mm256_mul_ps(y1, y1);
      __m256 z2 = _mm256_mul_ps(x2, x2);
      y2 = _mm256_sub_ps(y2, z2);
      y2 = _mm256_mul_ps(y2, y2);
      __m256 z3 = _mm256_mul_ps(x3, x3);
      y3 = _mm256_sub_ps(y3, z3);
      y3 = _mm256_mul_ps(y3, y3);

      /* (1 - x) */
      x1 = _mm256_sub_ps(c1, x1);
      x2 = _mm256_sub_ps(c1, x2);
      x3 = _mm256_sub_ps(c1, x3);

      result1 = _mm256_fmadd_ps(x1, x1, result1);
      result1 = _mm256_fmadd_ps(c2, y1, result1);
      result2 = _mm256_fmadd_ps(x2, x2, result2);
      result2 = _mm256_fmadd_ps(c2, y2, result2);
      result3 = _mm256_fmadd_ps(x3, x3, result3);
      result3 = _mm256_fmadd_ps(c2, y3, result3);
      
      result1 = _mm256_mul_ps(result1, c3);
      result2 = _mm256_mul_ps(result2, c3);
      result3 = _mm256_mul_ps(result3, c3);

      _mm256_store_ps(output, result1);
      _mm256_store_ps(output + 8, result2);
      _mm256_store_ps(output +16, result3);

      // t3 = rdtsc();
      // dur_peak += t3 - t2;
   }

   static T Objective(T x, T y) {
      return -1*(pow(1 - x, 2) + 100 * pow(y - x*x, 2));
   }

   static std::vector<T> ObjectiveOrig(const std::vector<T>& x)
   { 
      // t0 = rdtsc();
      T obj = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
      // t1 = rdtsc();
      // dur_orig_obj = (t1 - t0);
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
   // std::cout << "num is " << num << std::endl;
   galgo::GeneticAlgorithm<float> ga(MyObjective<float>::ObjectiveSIMD, MyObjective<float>::Objective, MyObjective<float>::ObjectiveOrig,
                                     2000,100,true,par1,par2);

   // setting constraints
   // ga.Constraint = MyConstraint;

   // running genetic algorithm
   ga.output = 1;
   ts = rdtsc();
   ga.run();
   te = rdtsc();
   std::cout << "total time and count is: " <<te - ts<< std::endl;
   std::cout<<"cross_over dur and count: " << dur_crossover << " " << count_crossover << std::endl;
   std::cout<<"mutation dur and count: " << dur_mutation << " " << count_mutation << std::endl;
}
