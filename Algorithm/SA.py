import math
import random
import sys

from Tool.Cal import cal


class SA:
    def __init__(self, BitNum, iteration, temperature, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.temperature = temperature
        self.Run = Run
        self.name = f"{self.BitNum}{self.iteration}_SA"
        self.G = cal()

    def Fitness(curr_sol):
        return sum(curr_sol)

    def Neighbor(self, curr):
        """Return neighborhod solution(Transition)"""
        shiftbit = random.randint(0, self.BitNum - 1)
        new_curr = curr.copy()
        new_curr[shiftbit] = 1 - new_curr[shiftbit]
        return new_curr

    def RunAIEva(self):
        """Simulated annealing, SA"""
        solution_db, fitness_db = [], []

        # Initialization
        curr_sol = [random.randint(0, 1) for _ in range(self.BitNum)]
        curr_fitness = self.Fitness(curr_sol)
        # solution_db.append(curr_sol)
        fitness_db.append(curr_fitness)
        i = 0

        while i <= (self.iteration - 1) and (self.temperature > 0):
            # (T) Transition
            neighbor_solution = self.Neighbor(curr_sol)
            neighbor_fitness = self.Fitness(neighbor_solution)

            # (E) Calculate the acceptance probability based on the current temperature
            delta_fitness = neighbor_fitness - curr_fitness
            acceptance_prob = min(
                1, math.exp(delta_fitness / self.temperature)
            )  # negative ->[0,1]

            # (D) update the current solution
            if neighbor_fitness > curr_fitness:
                curr_sol = neighbor_solution.copy()
                curr_fitness = neighbor_fitness
                solution_db.append(curr_sol)
                fitness_db.append(curr_fitness)
                # print(f"Accept better {neighbor_fitness}/{curr_fitness}_{i}")
                i += 1
            elif random.random() < acceptance_prob:
                curr_sol = neighbor_solution.copy()
                curr_fitness = neighbor_fitness
                solution_db.append(curr_sol)
                fitness_db.append(curr_fitness)
                self.temperature *= 0.95
                # print(f"Accept SA  {neighbor_fitness}/{curr_fitness}_{i}")
                i += 1
            else:
                # print(f"E {neighbor_fitness}/{curr_fitness}_{i}")
                solution_db.append(curr_sol)
                fitness_db.append(curr_fitness)
                i += 1

        return solution_db, fitness_db

    def AI(self):
        for _ in range(self.Run):
            sol, result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)

        # Visualization of the result
        y = self.G.AvgResult(f"{self.name}.csv")
        self.G.Draw(y, self.name)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        BitNum = int(sys.argv[1])
        temperature = int(sys.argv[2])
        iteration = int(sys.argv[3])
        Run = int(sys.argv[4])
    else:
        BitNum = 1000
        iteration = 1000
        temperature = 10
        Run = 50

    p = SA(BitNum, iteration, temperature, Run)
    p.AI()
