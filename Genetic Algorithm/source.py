import random
import string


PHRASE = "YUVAL"        #UPPERCASE ONLY
PHRASE_LENGTH = len(list(PHRASE))
MUTATION_RATE = 0.2
POPULLATION_SIZE = 150
MAX_GENERATIONS = 100000


class DNA:
    def __init__(self, last_gen = None):
        if last_gen == None:
            self.genes = []
            for i in range(PHRASE_LENGTH):
                self.genes.append(''.join(random.choice(string.ascii_uppercase)))
        else:
            self.genes = last_gen



    def calculate_fitness(self):
        fitness = 0
        goal = list(PHRASE)
        for i in range(PHRASE_LENGTH):
            if self.genes[i] == goal[i]:
                fitness += 5
            elif self.genes[i] in PHRASE:
                fitness+=1
            # else:
            #     for k in range(i,PHRASE_LENGTH):
            #         if self.genes[i] == goal[k]:
            #             fitness += 1
        return fitness


    def crossover(self,other):
        # ret = []
        # for i in range(PHRASE_LENGTH):
        #     prob = random.uniform(0,1)
        #     if prob < 0.5:
        #         ret.append(self.genes[i])
        #     else:
        #         ret.append(other.genes[i])

        ret = []
        THD = random.randint(0,PHRASE_LENGTH)
        for i in range(PHRASE_LENGTH):
            if i < THD:
                ret.append(self.genes[i])
            else:
                ret.append(other.genes[i])

        return DNA(last_gen=ret)


    def mutate(self):
        ret = []
        for i in range(PHRASE_LENGTH):
            prob = random.uniform(0,1)
            if prob < MUTATION_RATE:
                ret.append(''.join(random.choice(string.ascii_uppercase)))
            else:
                ret.append(self.genes[i])

        return DNA(last_gen=ret)



class Population:
    def __init__(self):
        self.childs = []


        for i in range(POPULLATION_SIZE):
            self.childs.append(DNA())

        self.best_child = self.childs[0]

    def CrateMatingPool(self):

        mating_pool = []

        for child_indx in range(len(self.childs)):
            if self.childs[child_indx].calculate_fitness() > self.best_child.calculate_fitness():
                self.best_child = self.childs[child_indx]

            for i in range(self.childs[child_indx].calculate_fitness()):
                mating_pool.append(child_indx)

        return mating_pool



        return mating_pool


    def calculate_probabilities(self,mating_pool):
        total_fitness = 0
        for child in self.childs:
            total_fitness += child.calculate_fitness()

        prob_arr = []
        for child in self.childs:
            prob = (child.calculate_fitness() / total_fitness) * 100
            prob_arr.append(prob)

        return prob_arr

    def ChooseTwoParents(self,mating_pool):
        idx_a = random.choice(mating_pool)
        idx_b = random.choice(mating_pool)



        parentA = self.childs[idx_a]
        parentB = self.childs[idx_b]

        return parentA,parentB


    def GenerateNewPopulation(self,ParentA,ParentB):


        #Create new crossovered population
        for i in range(POPULLATION_SIZE):
            self.childs[i] = ParentA.crossover(ParentB)

        #Mutate the population
        for i in range(POPULLATION_SIZE):
            self.childs[i] = self.childs[i].mutate()


    def GetFittest(self):
        max_idx = 0
        max_fit = 0

        for i in range(POPULLATION_SIZE):
            if self.childs[i].calculate_fitness() > max_fit:
                max_fit = self.childs[i].calculate_fitness()
                max_idx = i


        return self.childs[max_idx]









def main():
    population = Population()

    for i in range(MAX_GENERATIONS):
        mating_pool = population.CrateMatingPool()

        if population.best_child.genes == list(PHRASE):
            print('Succeded !!! Generation : ', i )
            break

        if i%1000 == 0:
            print('Generation : ', i)
            print(population.GetFittest().genes)

        parentA, parentB = population.ChooseTwoParents(mating_pool)
        population.GenerateNewPopulation(parentA,parentB)


    print('Best child : ',population.best_child.genes)
    print('Done.')


if __name__ == "__main__":
    main()
