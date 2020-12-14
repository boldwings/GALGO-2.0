
run GA algorithm

//in GeneticAlgorithm.hpp
intialize population
//in population.hpp, pop.creation()
for each chromosome in population
    //in chromo.hpp & parameter.hpp
    create encoding  //known vs. unknown?
    //in chromo.hpp 
    evaluate
        decode all parameters
        compute objective function
        use accumulated result as fitness value
update
    sort chromosome from best to worst

record current best result

//in population.hpp, pop.evolution()
for each generation:
    //in evolution.hpp
    selection algorithm
        adjust fittness to positive value (shift)
        compute sum of fitness
        select a chromosome according to a random number from uniform distribution
    apply elitism?
    recombination
        for each pair?? uptil nbrcrov:
            crossover (P1XO)
            mutation (SPM)
            evaluate
    completion







    


    
