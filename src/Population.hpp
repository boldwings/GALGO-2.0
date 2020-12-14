//=================================================================================================
//                    Copyright (C) 2017 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#ifndef POPULATION_HPP
#define POPULATION_HPP
// #include <immintrin.h>
/**********************************************************************/
int count_crossover = 0;
int count_mutation = 0;
int count_eval = 0;
int count_eval_orig = 0;
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
unsigned long long dur_crossover, dur_mutation, dur_eval, dur_eval_orig;
unsigned long long t0, t1;
unsigned long long dur;
/********************************************************************/
namespace galgo {

//=================================================================================================

template <typename T>
class Population
{
   static_assert(std::is_same<float,T>::value || std::is_same<double,T>::value, "variable type can only be float or double, please amend.");

public: 
   // nullary constructor
   Population() {}
   // constructor
   Population(const GeneticAlgorithm<T>& ga);
   // create a population of chromosomes
   void creation();
   // evolve population, get next generation
   void evolution();

   // access element in current population at position pos
   const CHR<T>& operator()(int pos) const;
   // access element in mating population at position pos
   const CHR<T>& operator[](int pos) const;  
   // return iterator to current population beginning
   typename std::vector<CHR<T>>::iterator begin();
   // return const iterator to current population beginning
   typename std::vector<CHR<T>>::const_iterator cbegin() const;
   // return iterator to current population ending  
   typename std::vector<CHR<T>>::iterator end();  
   // return const iterator to current population ending  
   typename std::vector<CHR<T>>::const_iterator cend() const;  
   // select element at position pos in current population and copy it into mating population
   void select(int pos);
   // evaluation for the popluation
   void calFitness_simd(std::vector<CHR<T>>& pop, int start, int end);
   // set all fitness to positive values
   void adjustFitness();
   // compute fitness sum of current population
   T getSumFitness() const;
   // get worst objective function total result from current population
   T getWorstTotal() const;
   // return population size
   int popsize() const;
   // return mating population size
   int matsize() const;
   // return tournament size
   int tntsize() const;
   // return numero of generation
   int nogen() const;
   // return number of generations
   int nbgen() const;
   // return selection pressure
   T SP() const; 

private:
   std::vector<CHR<T>> curpop;               // current population
   std::vector<CHR<T>> matpop;               // mating population
   std::vector<CHR<T>> newpop;               // new population

   const GeneticAlgorithm<T>* ptr = nullptr; // pointer to genetic algorithm              
   int nbrcrov;                              // number of cross-over
   int matidx;                               // mating population index

   // elitism => saving best chromosomes in new population
   void elitism();
   // create new population from recombination of the old one
   void recombination();
   // complete new population randomly
   void completion();
   // update population (adapting, sorting)
   void updating();
};

/*-------------------------------------------------------------------------------------------------*/

// constructor
template <typename T>
Population<T>::Population(const GeneticAlgorithm<T>& ga)
{
   ptr = &ga;   
   nbrcrov = floor(ga.covrate * (ga.popsize - ga.elitpop));
   // adjusting nbrcrov (must be an even number)
   if (nbrcrov % 2 != 0) nbrcrov -= 1;
   // for convenience, we add elitpop to nbrcrov
   nbrcrov += ga.elitpop;
   // allocating memory
   curpop.resize(ga.popsize);
   matpop.resize(ga.matsize);
}

/*-------------------------------------------------------------------------------------------------*/

// create a population of chromosomes
template <typename T>
void Population<T>::creation()
{
   int start = 0;
   // initializing first chromosome
   if (!ptr->initialSet.empty()) {
      curpop[0] = std::make_shared<Chromosome<T>>(*ptr);
      curpop[0]->initialize();
      // t0 = rdtsc();
      // curpop[0]->evaluate();
      // t1 = rdtsc();
      // dur_eval_orig += (t1 - t0);
      // count_eval_orig ++;
      start++;
   }
   // getting the rest
   // #ifdef _OPENMP 
   // #pragma omp parallel for num_threads(MAX_THREADS)
   // #endif
   for (int i = start; i < ptr->popsize; ++i) {
      curpop[i] = std::make_shared<Chromosome<T>>(*ptr);
      curpop[i]->create();
      // t0 = rdtsc();
      // curpop[i]->evaluate();
      // t1 = rdtsc();
      // dur_eval_orig += (t1 - t0);
      // count_eval_orig ++;
   }

   // t0 = rdtsc();
   // for (int i = 0; i < ptr->popsize; ++i) {
   //    curpop[i]->evaluate();
   // }
   // t1 = rdtsc();
   // dur_eval_orig += (t1 - t0);
   // count_eval_orig += ptr->popsize;


   t0 = rdtsc();
   calFitness_simd(curpop, 0, ptr->popsize);
   t1 = rdtsc();
   dur_eval = (t1 - t0);
   count_eval+= ptr->popsize;

   // std::cout<< "count_eval_orig: " << count_eval_orig<<std::endl;
   // std::cout<< "count_eval: " << count_eval<<std::endl;
   
   // updating population
   this->updating();
}

/*-------------------------------------------------------------------------------------------------*/

// population evolution (selection, recombination, completion, mutation), get next generation
template <typename T>
void Population<T>::evolution()
{   
   // initializing mating population index
   matidx = 0;
   // selecting mating population
   ptr->Selection(*this); 
   // applying elitism if required
   this->elitism(); 
   // crossing-over mating population
   this->recombination();
   // completing new population
   this->completion();
   // moving new population into current population for next generation
   curpop = std::move(newpop);
   // updating population
   this->updating(); 
}

/*-------------------------------------------------------------------------------------------------*/

// elitism => saving best chromosomes in new population, making a copy of each elit chromosome
template <typename T>
void Population<T>::elitism()
{
   // (re)allocating new population
   newpop.resize(ptr->popsize);

   if (ptr->elitpop > 0) {
      // copying elit chromosomes into new population
      std::transform(curpop.cbegin(), curpop.cend(), newpop.begin(), [](const CHR<T>& chr)->CHR<T>{return std::make_shared<Chromosome<T>>(*chr);});
   }
}

/*-------------------------------------------------------------------------------------------------*/

// create new population from recombination of the old one
template <typename T>
void Population<T>::recombination()
{
   // creating a new population by cross-over
   // #ifdef _OPENMP 
   // #pragma omp parallel for num_threads(MAX_THREADS)
   // #endif
   for (int i = ptr->elitpop; i < nbrcrov; i = i + 2) {      
      // initializing 2 new chromosome
      newpop[i] = std::make_shared<Chromosome<T>>(*ptr);
      newpop[i+1] = std::make_shared<Chromosome<T>>(*ptr);
      // crossing-over mating population to create 2 new chromosomes
      t0 = rdtsc();
      ptr->CrossOver(*this, newpop[i], newpop[i+1]);
      t1 = rdtsc();
      count_crossover++;
      dur_crossover += t1 - t0;
      // mutating new chromosomes
      t0= rdtsc();
      ptr->Mutation(newpop[i]);   
      ptr->Mutation(newpop[i+1]);   
      t1 = rdtsc();
      count_mutation += 2;
      dur_mutation += t1 - t0;
      // evaluating new chromosomes
      // t0= rdtsc();
      // newpop[i]->evaluate();
      // newpop[i+1]->evaluate();
      // t1 = rdtsc();
      // dur_eval_orig += (t1 - t0);
      // count_eval_orig += 2;
   } 
   t0= rdtsc();
   calFitness_simd(newpop, ptr->elitpop, nbrcrov);
   t1 = rdtsc();
   dur_eval += (t1 - t0);
   count_eval += (nbrcrov - ptr->elitpop);

   // ptr->Mutation_simd(newpop, ptr->elitpop, nbrcrov);
   // for (int i = ptr->elitpop; i < nbrcrov; i++) {
   //    newpop[i]->evaluate();
   // }
}

/*-------------------------------------------------------------------------------------------------*/

// complete new population
template <typename T>
void Population<T>::completion()
{
   // #ifdef _OPENMP 
   // #pragma omp parallel for num_threads(MAX_THREADS)
   // #endif
   for (int i = nbrcrov; i < ptr->popsize; ++i) {
      // selecting chromosome randomly from mating population
      newpop[i] = std::make_shared<Chromosome<T>>(*matpop[uniform<int>(0, ptr->matsize)]);
      // mutating chromosome
      t0 = rdtsc();
      ptr->Mutation(newpop[i]);
      t1 = rdtsc();
      count_mutation += 1;
      dur_mutation += (t1 - t0);
      // evaluating chromosome
      // t0 = rdtsc();
      // newpop[i]->evaluate();
      // t1 = rdtsc();
      // count_eval_orig ++;
      // dur_eval_orig += (t1 - t0);
   }
   t0 = rdtsc();
   calFitness_simd(newpop, nbrcrov, ptr->popsize);
   t1 = rdtsc();
   count_eval += (ptr->popsize - nbrcrov);
   dur_eval += (t1 - t0);
}

/*-------------------------------------------------------------------------------------------------*/

// update population (adapting, sorting)
template <typename T>
void Population<T>::updating()
{
   // adapting population to constraints
   if (ptr->Constraint != nullptr) {
      ptr->Adaptation(*this); 
   }
   // sorting chromosomes from best to worst fitness
   std::sort(curpop.begin(),curpop.end(),[](const CHR<T>& chr1,const CHR<T>& chr2)->bool{return chr1->fitness > chr2->fitness;});
}

/*-------------------------------------------------------------------------------------------------*/

// access element in current population at position pos
template <typename T>
const CHR<T>& Population<T>::operator()(int pos) const
{
   #ifndef NDEBUG
   if (pos > ptr->popsize - 1) {
      throw std::invalid_argument("Error: in galgo::Population<T>::operator()(int), exceeding current population memory.");
   }
   #endif

   return curpop[pos];
}

/*-------------------------------------------------------------------------------------------------*/

// access element in mating population at position pos
template <typename T>
const CHR<T>& Population<T>::operator[](int pos) const
{
   #ifndef NDEBUG
   if (pos > ptr->matsize - 1) {
      throw std::invalid_argument("Error: in galgo::Population<T>::operator[](int), exceeding mating population memory.");
   }
   #endif

   return matpop[pos];
}

/*-------------------------------------------------------------------------------------------------*/

// return iterator to current population beginning 
template <typename T>
inline typename std::vector<CHR<T>>::iterator Population<T>::begin()
{
   return curpop.begin(); 
}

/*-------------------------------------------------------------------------------------------------*/

// return const iterator to current population beginning 
template <typename T>
inline typename std::vector<CHR<T>>::const_iterator Population<T>::cbegin() const
{
   return curpop.cbegin(); 
}

/*-------------------------------------------------------------------------------------------------*/

// return iterator to current population ending
template <typename T>
inline typename std::vector<CHR<T>>::iterator Population<T>::end()
{ 
   return curpop.end();
}

/*-------------------------------------------------------------------------------------------------*/

// return const iterator to current population ending
template <typename T>
inline typename std::vector<CHR<T>>::const_iterator Population<T>::cend() const
{ 
   return curpop.cend();
}

/*-------------------------------------------------------------------------------------------------*/

// select element at position pos in current population and copy it into mating population
template <typename T>
inline void Population<T>::select(int pos)
{
   #ifndef NDEBUG
   if (pos > ptr->popsize - 1) {
      throw std::invalid_argument("Error: in galgo::Population<T>::select(int), exceeding current population memory.");
   }
   if (matidx == ptr->matsize) {
      throw std::invalid_argument("Error: in galgo::Population<T>::select(int), exceeding mating population memory.");
   }
   #endif
   // std::cout << "indexing is " << matidx <<std::endl;
   matpop[matidx] = curpop[pos];
   
   // std::cout << "after assignment" <<std::endl;
   matidx++;
}

/*-------------------------------------------------------------------------------------------------*/
typedef union {
    __m256 v;
    float a[8];
} eval_union;


template <typename T>
void Population<T>::calFitness_simd(std::vector<CHR<T>>& pop, int start, int end) 
{  
   // count++;
   // if (start == 0 ) t0 = rdtsc();
   const auto &p1 = ptr->param[0];
   const auto &p2 = ptr->param[1];
   std::vector<T> x(end - start);
   std::vector<T> y(end - start);
   #ifdef _OPENMP 
   #pragma omp parallel for num_threads(28) 
   #endif
   for (int i = start; i < end; i++) {
      std::string crr_s = pop[i]->getchr();
      x[i - start] = (p1->decode(crr_s.substr(ptr->idx[0], p1->size())));
      y[i - start]= (p2->decode(crr_s.substr(ptr->idx[1], p2->size())));
   }
   // #ifdef _OPENMP
   // #pragma omp barrier
   // #endif
   int i = 0;
   // #pragma omp parallel for num_threads(4) schedule(dynamic)
   for (i = 0; i  < x.size() - 24; i += 24) {
      T output[24] = {0};
      ptr->ObjectiveSIMD(x.data() + i, y.data() + i, output);
      for (int j = start + i; j < start + i + 24; j++) {
         pop[j]->fitness = output[j - start - i];
      }
   }

   for (int j = i; j < x.size() ; j++) {
      pop[j]->fitness = ptr->Objective(x[j], y[j]);
   }
}

/*-------------------------------------------------------------------------------------------------*/
    
// set all fitness to positive values (used in RWS and SUS selection methods)
template <typename T>
void Population<T>::adjustFitness()
{
   // getting worst population fitness
   T worstFitness = curpop.back()->fitness;

   if (worstFitness < 0) {
      // getting best fitness
      T bestFitness = curpop.front()->fitness;
      // case where all fitness are equal and negative
      if (worstFitness == bestFitness) {
         std::for_each(curpop.begin(), curpop.end(), [](CHR<T>& chr)->void{chr->fitness *= -1;});
       } else {
         std::for_each(curpop.begin(), curpop.end(), [worstFitness](CHR<T>& chr)->void{chr->fitness -= worstFitness;});
      }
   }
}

/*-------------------------------------------------------------------------------------------------*/

// compute population fitness sum (used in TRS, RWS and SUS selection methods)
template <typename T>
inline T Population<T>::getSumFitness() const
{
   return std::accumulate(curpop.cbegin(), curpop.cend(), 0.0, [](T sum, const CHR<T>& chr)->T{return sum + T(chr->fitness);});
}

/*-------------------------------------------------------------------------------------------------*/

// get worst objective function total result from current population (used in constraint(s) adaptation)
template <typename T>
inline T Population<T>::getWorstTotal() const
{
   auto it = std::min_element(curpop.begin(), curpop.end(), [](const CHR<T>& chr1, const CHR<T>& chr2)->bool{return chr1->getTotal() < chr2->getTotal();});

   return (*it)->getTotal();
}

/*-------------------------------------------------------------------------------------------------*/

// return population size
template <typename T>
inline int Population<T>::popsize() const
{
   return ptr->popsize;
}

/*-------------------------------------------------------------------------------------------------*/

// return mating population size
template <typename T>
inline int Population<T>::matsize() const
{
   return ptr->matsize;
}

/*-------------------------------------------------------------------------------------------------*/

// return tournament size
template <typename T>
inline int Population<T>::tntsize() const
{
   return ptr->tntsize;
}

/*-------------------------------------------------------------------------------------------------*/

// return numero of generation
template <typename T>
inline int Population<T>::nogen() const
{
   return ptr->nogen;
}

/*-------------------------------------------------------------------------------------------------*/

// return number of generations
template <typename T>
inline int Population<T>::nbgen() const
{
   return ptr->nbgen;
}

/*-------------------------------------------------------------------------------------------------*/

// return selection pressure
template <typename T>
inline T Population<T>::SP() const
{
   return ptr->SP;
}

//=================================================================================================

}

#endif


