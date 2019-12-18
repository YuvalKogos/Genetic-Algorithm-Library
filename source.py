import random
import string
import numpy as np

GENERATIONS_NUM = 100
TARGET = 'this is a sample sentence'


class DNA:
    def __init__(self,length):
        self.genes = np.chararray(length)
        self.length = length
        for char in self.genes:
            char = random.choice(string.ascii_letters)


    def calculate_fitness(self,target):
        counter = 0
        for i in range(self.length):
            if self.genes[i] == target[i]:
                counter += 1

        return counter




class Population:
    def __init__(self,length,target,mutation_rate):
        self.target = target
        self.population = self.init_pop(length)
        self.mutation_rate = mutation_rate


    def generate(self):
        fitness_arr = self.create_fitness_arr()
        a = random(len(fitness_arr))
        b = random(len(fitness_arr))
        partner_a = fitness_arr[a]
        partner_b = fitness_arr[b]

        for i in range(len(self.population)):
            child = self.crossover(partner_a, partner_b)
            self.mutate(child)
            self.population[i] = child

    def mutate(self,child):
        for i in child.length:
            rand = random.uniform(0,1)
            if rand <= self.mutation_rate:
                child.genes[i] = random.choice(string.ascii_letters)

    def crossover(self,partner_a,partner_b):
        mid_point = partner_a.length / 2
        child = DNA(partner_a.length)
        for i in range(partner_a.length):
            if i < mid_point:
                child.genes[i] = partner_a.genes[i]
            else:
                child.genes[i] = partner_b.genes[i]

        return child

    def create_fitness_arr(self):
        total_counter = 0
        for dna in self.population:
            total_counter += dna.calculate_fitness(self.target)

        fitness_arr = []
        for i in range(len(self.population)):
            fitness_arr.append((self.population[i].calculate_fitness(self.target) / total_counter) * 100)


        results = []
        for i in range(len(fitness_arr)):
            for time in range(fitness_arr[i] / 10):
                results.append(i)


        return results

    def init_pop(self,length):
        pop = []
        for i in range(length * 10):
            tmp = DNA(self.target)
            pop.append(tmp)
        return pop

    def get_fittest(self):
        fitness_arr = self.create_fitness_arr()

        longest_streak = 1
        curr_streak = 1
        value = fitness_arr[1]

        for i in range(1,len(fitness_arr)):
            if fitness_arr[i] != fitness_arr[i-1]:
                if curr_streak > longest_streak:
                    longest_streak = curr_streak
                    value = arr[i-1]
                curr_streak = 0

            elif curr_streak > longest_streak:
                longest_streak = curr_streak
                value = fitness_arr[i]

            curr_streak += 1


        return value


def main():

    population = Population(200,TARGET,0.1)


    for i in range(GENERATIONS_NUM):
        population.generate()
        print(population.get_fittest().genes.tostring)


    print("finished!")


if __name__ == "__main__":
    main()
