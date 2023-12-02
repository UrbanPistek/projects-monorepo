# Assignment 6 Question 3 Solution
import os
import time
import copy
import argparse
import numpy as np
import random as rand
from tqdm import tqdm
from math import trunc
from statistics import mean
from operator import attrgetter
from matplotlib import pyplot as plt

# Configure prints
np.set_printoptions(precision=0, suppress=True)

def get_args(parser):
    parser.add_argument(
        "-b",
        "--base-sga", 
        action='store_true',
        help='Run most basic SGA implementation')
    parser.add_argument(
        "-e",
        "--elitism-sga", 
        action='store_true',
        help='Run SGA with elitism enabled')
    parser.add_argument(
        "-f",
        "--full", 
        action='store_true',
        help='Run full suite of cases of SGA and associated functions')

    return parser.parse_args()

# Plot six-hump camelback function
def plot_search_space():
    """
    3D Plot of six-hump camelback function.
    """

    fig = plt.figure(figsize = (16,9))
    ax = plt.axes(projection='3d')

    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = (4 - 2.1*(X**2) + (X**4)/3)*(X**2) + X*Y + (-4 + 4*(Y**2))*(Y**2)
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)

    # Marker at global minimum
    # ax.text(0.089840, -0.712659, -1.0316285, "min")
    # ax.plot( [0.089840], [-0.712659], [-1.0316285], markerfacecolor='m', markeredgecolor='m', marker='x', markersize=5, alpha=0.6)
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    plt.title("Search Space")

    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

class SGA:

    def __init__(self, n_pop=10, n_gen=100):
        """
        n_pop:  Number of individuals, the size of the population
        n_gen: Number of generations to run search for  
        """

        # population size
        self.population_size = n_pop

        # carry population
        self.population = []

        # number of bits to encode genotype (per variable)
        # using 14 bit encoding scheme for this problem
        self.n_bits_genotype = 14

        # carry fittest individual 
        self.fittest_individuals = []

        # number of generations 
        self.generations = n_gen

        # Number of bits for decimal precision
        self.__dec_precision = 3

        # rate of mutation in individuals
        self.mutation_rate = 1/self.population_size

        # Initialize population
        self.__population_init()

    def __objective_func(self, x, y):
        """
        Evaluation of six-hump camelback function. 
        """
        return (4 - 2.1*(x**2) + (x**4)/3)*(x**2) + x*y + (-4 + 4*(y**2))*(y**2)

    def __population_init(self):
        """
        Initialize population with random genes for each individual in the population. 
        """
        
        init_pop = []
        for _ in range(self.population_size):
            
            xs = list(np.random.randint(0, 2, self.n_bits_genotype))
            ys = list(np.random.randint(0, 2, self.n_bits_genotype))
            init_pop.append([xs, ys])
        
        self.population = init_pop

        print(f"\nInitial Population: \n{self.population}")

    def __decode_genotype(self, gene):
        """
        gene = individual varible gene for either x or y variable 
        """
        
        # read sign bit
        sign = 1
        if gene[0] == 0:
            sign = -1

        # go through bits 1-3 (integer value)
        int_val = 0
        j = 2
        for i in range(1, 4):
            int_val += gene[i]*(2**(j))
            j += -1

        # go through bits 4-13 (decimal value)
        dec_val = 0
        j = 9
        for i in range(4, 14):
            dec_val += gene[i]*(2**(j))
            j += -1

        # convert to decimal 
        dec_val = dec_val/(10**self.__dec_precision)

        # return fully decoded gene as a floating point value
        return sign*(int_val+dec_val)

    def fitness_function(self, individual):
        """
        Calculate fitness of an individual based on the objective function.
        """

        # Get encoded genotypes
        xe = individual[0]
        ye = individual[1]

        # get decoded genotype
        pheno_x = self.__decode_genotype(xe)
        pheno_y = self.__decode_genotype(ye)

        # calculate the individuals fitness
        fitness = self.__objective_func(pheno_x, pheno_y)

        return fitness

    def single_crossover(self, pgene1, pgene2, crossover_point=6):

        # define childern
        cgene1 = []
        cgene2 = []

        # copy first portions before crossover point
        cgene1 = pgene1[:crossover_point]
        cgene2 = pgene2[:crossover_point]

        # add remain portions after crossover point
        cgene1.extend(pgene2[crossover_point:])
        cgene2.extend(pgene1[crossover_point:])

        return cgene1, cgene2

    def mutate(self, gene):
        """
        Mutate a gene according to the mutation rate.
        """
        
        mut = rand.random()
        new_gene = copy.deepcopy(gene)

        if mut < self.mutation_rate:
            mut_idx = rand.randint(0, 13)

            # flip bit at a random position
            new_gene[mut_idx] = int(not new_gene[mut_idx])

        return new_gene

    def rank_fitness(self, population):

        # Set population size
        population_size = len(population)
        
        # calculate fitness of each individual in the population
        fitness_scores = []
        for idx, unit in enumerate(population):
            fitness = self.fitness_function(unit)

            score_obj = {"fitness": fitness, "idx": idx}
            fitness_scores.append(score_obj)

        # sort fitness scores inplace, best score in highest index
        fitness_scores.sort(key=lambda score_object: score_object["fitness"], reverse=True)

        # add probabilities for each 
        base = sum(list(range(1, (population_size+1)))) # denominator
        for idx, score in enumerate(fitness_scores):
            p = (idx+1)/base
            score["p_value"] = p

        return fitness_scores

    def mean_population_fitness(self):
        """
        Calculate mean fitness of current population.
        """
        fitness = []
        for indv in self.population:
            fitness.append(self.fitness_function(indv))
    
        return mean(fitness)

    def search(self, elitism=False, elitism_frac=0.5):
        """
        Run the search algorithm for the defined set of generations.

        elitism: Whether or not to keep a portion of the fittest parents from generation to generation
        elitism_frac: Fraction of how much of the parents to keep in a generation, fraction of parents kept is 1 - elitism_frac
        """
        print(f"\nsearching...")

        for _ in tqdm(range(self.generations)):

            # rank and select best parents
            ranks = self.rank_fitness(self.population)

            # save the best individual from this generation along with some metrics
            pop_avg = self.mean_population_fitness()# mean fitness of population
            self.fittest_individuals.append({
                "gene": self.population[ranks[self.population_size-1]["idx"]],
                "fitness": ranks[self.population_size-1]["fitness"],
                "phenotype": [
                                self.__decode_genotype(self.population[ranks[self.population_size-1]["idx"]][0]), \
                                self.__decode_genotype(self.population[ranks[self.population_size-1]["idx"]][1]) \
                            ],
                "pop_avg": pop_avg
            })

            # extract probailities 
            probs = [obj["p_value"] for obj in ranks]

            # select parents 
            pop = copy.deepcopy(self.population) # make a copy of the current population
            parents = []
            for _ in range(self.population_size): 
                select = pop[np.random.choice(self.population_size, p=probs)]
                parents.append(select)

            # create new set of childern 
            childern = []
            for i in range(0, 10, 2):
                p1 = parents[i]
                p2 = parents[i+1]

                # childern 1 & 2
                c1 = []
                c2 = []
                for gene in range(2):

                    # select random crossover point 
                    crossover_point = rand.randint(1, 12)
                    g1, g2 = self.single_crossover(p1[gene], p2[gene], crossover_point=crossover_point)

                    # run mutations
                    g1 = self.mutate(g1)
                    g2 = self.mutate(g2)

                    # append gene to childerns chromosone
                    c1.append(g1)
                    c2.append(g2)

                # Add childer to list of childern
                childern.append(c1)
                childern.append(c2)

            # update population, if elitism, update in using it
            if elitism:
                # calculate the fitness of all the childern
                cranks = self.rank_fitness(childern)

                # determine cutoff index based on the elitism fraction
                # rounding to offset floating point math errors
                p_cutoff = int(round(self.population_size * elitism_frac))
                c_cutoff = int(round(self.population_size * (1 - elitism_frac)))

                # populate selection with defined fraction of childer and parents
                selection = [self.population[p["idx"]] for p in ranks[p_cutoff:]]
                selection.extend([childern[c["idx"]] for c in cranks[c_cutoff:]])

                # update population
                self.population = selection
            else:
                # by default all childern replace parents
                self.population = childern

        # Do a final evalution and print metrics
        franks = self.rank_fitness(self.population)
        curr_fittest = self.population[franks[self.population_size-1]['idx']]

        print(f"\n**END**\nCurrent Fittest Individual: {curr_fittest}")
        print(f"Fitness: {franks[self.population_size-1]['fitness']}")
        print(f"Phenotype: {self.__decode_genotype(curr_fittest[0]), self.__decode_genotype(curr_fittest[1])}")

        most_fit = min(self.fittest_individuals, key=lambda indv: indv['fitness'])
        print(f"\nMost Fit Individual: \n{most_fit}")

    def reset(self):
        """
        Reset object to be able to run search again on the same object.
        """

        # carry population
        self.population = []

        # carry fittest individual 
        self.fittest_individuals = []

        # Initialize population
        self.__population_init()

    def plot_fittest_individuals(self, showfig=True, savefig=False, title="Fittest Individual vs Generation", filename='sga_search'):

        fig = plt.figure(figsize=(16,9))

        xs = range(1, self.generations+1)
        ys = [i["fitness"] for i in self.fittest_individuals]
        ya = [i["pop_avg"] for i in self.fittest_individuals]
        plt.plot(xs, ys, label='Fittest Individual')
        plt.plot(xs, ya, label='Mean Population Fitness')

        plt.title(title)
        plt.xlabel('Generation', labelpad=20)
        plt.ylabel('Fitness', labelpad=20)
        plt.legend()

        if showfig:
            plt.show()

        if savefig:
            # create directory if it does not exist
            if not os.path.exists('figures'):
                os.makedirs('figures')

            plt.savefig(f'figures/{filename}.png')

        
    def test_fitness_function(self):

        print(f"\n__test_fitness_function:")

        # 1.127
        gene_x = [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        # -0.587
        gene_y = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        individual = [gene_x, gene_y]

        fitness = self.fitness_function(individual)

        print(f"==> case 1: [0.810 == {fitness}]")

    def test_decode_genotype(self):

        print(f"\n__test_decode_genotype:")

        # 3.127
        gene1 = [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        # -1.587
        gene2 = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]

        pheno1 = self.__decode_genotype(gene1)
        pheno2 = self.__decode_genotype(gene2)

        print(f"==> case 1: [3.127 == {pheno1}]")
        print(f"==> case 2: [-1.587 == {pheno2}]")

    def test_single_crossover(self):

        print(f"\n__test_single_crossover:")

        parent1 = [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        parent2 = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]

        child1, child2 = self.single_crossover(parent1, parent2, crossover_point=3)

        res1 = [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        res2 = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        print(f"==> child 1: [{res1} == {child1}] {res1 == child1}")
        print(f"==> child 2: [{res2} == {child2}] {res2 == child2}")

    def test_mutate(self):

        print(f"\n__test_mutate:")

        gene = [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # run 20 times, should mutate at least once
        res = []
        mutated = []
        for i in range(0, 20):
            mgene = self.mutate(gene)

            # append whether the mutated gene is different from the original
            res.append(gene == mgene)

            if gene != mgene:
                mutated.append(mgene)

        print(f"==> case 1: results: {res} | {False in res}")
        print(f"==> case 1: mutated: {mutated}")


def main():

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    ts = time.time()
    print("Simple Genetic Algorithm")

    # Run just base SGA 
    # Using single point crossover, rank based parent selection
    # with the generational genetic algorithm model
    if args.base_sga: 

        # define sga object
        sga = SGA()

        # run search 
        sga.search()

        # plot fittest individual over time
        sga.plot_fittest_individuals(title="Fittest Individual vs Generation (Base SGA)", filename="sga_base_search")

    # Run just SGA with elitism  
    # Using single point crossover, rank based parent selection
    # with the steady state genetic algorithm model
    elif args.elitism_sga:

        # define sga object
        sga = SGA()

        # search with elitism
        elitism_frac = 0.9
        sga.search(elitism=True, elitism_frac=elitism_frac)

        # plot fittest individual over time
        sga.plot_fittest_individuals(title=f"Fittest Individual vs Generation (Elitism SGA, frac={elitism_frac})", \
            filename=f"sga_elitism_f{trunc(elitism_frac*100)}search")

    # Run SGA with every possible mode
    elif args.full:

        # plot visual of the search space
        plot_search_space()

        # define sga object 
        sga = SGA()

        # test sga components
        sga.test_decode_genotype()
        sga.test_fitness_function()
        sga.test_single_crossover()
        sga.test_mutate()

        # run search 
        sga.search()

        # plot fittest individual over time
        sga.plot_fittest_individuals(title="Fittest Individual vs Generation (Base SGA)", filename="sga_base_search", showfig=False, savefig=True)

        # reset to run search again
        sga.reset()

        # search with elitism
        elitism_frac = 0.5
        sga.search(elitism=True, elitism_frac=elitism_frac)

        # plot fittest individual over time
        sga.plot_fittest_individuals(title=f"Fittest Individual vs Generation (Elitism SGA, frac={elitism_frac})", \
            filename=f"sga_elitism_f{trunc(elitism_frac*100)}search", showfig=False, savefig=True)

    else: 
        print("Please specific mode of execution:\n")
        parser.print_usage()
        parser.print_help()

    te = time.time()

    print(f"\nElapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
