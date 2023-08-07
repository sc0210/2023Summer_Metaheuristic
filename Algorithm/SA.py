import os
import random
import sys
import time

import numpy as np

from Problem.problem import Problem
from Tool.Cal import cal


class SA(Problem):
    def __init__(self, p, BitNum, iteration, temperature, Run):
        super().__init__(p=p, BitNum=BitNum)
        # packages
        self.G = cal()

        # simulate annealing
        self.temperature = float(temperature)

        # problem
        self.BitNum = int(BitNum)
        self.iteration = int(iteration)
        self.Run = int(Run)
        self.name = f"SA_{self.problem}_{self.BitNum}_{self.temperature}"

    def Neighbor(self, curr):
        """Return neighborhod solution(Transition)"""
        shiftbit = random.randint(0, self.BitNum - 1)
        new_curr = curr.copy()
        new_curr[shiftbit] = 1 - new_curr[shiftbit]
        return new_curr

    def RunAIEva(self):
        """Simulated annealing, SA"""
        # (I) Initialization
        self.cnt = 0
        Global_sol = np.array([random.randint(0, 1) for _ in range(self.BitNum)])
        Global_fitness = self.Fitness(Global_sol)
        solution_db = [Global_sol.tolist()]
        fitness_db = [Global_fitness]

        while self.cnt <= (self.iteration - 1) and (self.temperature > 0):
            # (T) Transition
            neighbor_solution = self.Neighbor(Global_sol)
            neighbor_fitness = self.Fitness(neighbor_solution)

            # (E) Calculate the acceptance probability based on the current temperature
            delta_fitness = neighbor_fitness - Global_fitness
            acceptance_prob = min(
                1, np.exp(delta_fitness / self.temperature)
            )  # negative ->[0,1]

            # (D) update the current solution
            if neighbor_fitness > Global_fitness:
                Global_sol = neighbor_solution.copy()
                Global_fitness = neighbor_fitness
                # print(f"Accept better {neighbor_fitness}/{curr_fitness}_{i}")
            elif random.random() < acceptance_prob:
                Global_sol = neighbor_solution.copy()
                Global_fitness = neighbor_fitness
                self.temperature *= 0.95
                # print(f"Accept SA  {neighbor_fitness}/{curr_fitness}_{i}")

            solution_db.append(Global_sol.tolist())
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return solution_db, fitness_db

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        if not os.path.isdir("./result/"):
            os.makedirs("./result")

        # Average the result from multiple runs
        for Run_index in range(self.Run):
            sol, fitness_result = self.RunAIEva()
            self.G.Write2CSV(np.array(fitness_result), f"./result/{self.name}.csv")

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Obj:{:<2}, Time:{:<3}\nBest solution:{}\n".format(
                        Run_index,
                        np.max(fitness_result),
                        np.round(time.time() - st, 3),
                        [sol[-1]],
                    )
                )

        # Avg result
        end = time.time()
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============\n")


if __name__ == "__main__":
    # Hyperparameters
    if len(sys.argv) == 3:
        p = str(sys.argv[1])  # Problem: Onemax or Deception
        temperature = float(sys.argv[2])
    else:
        temperature = 10

    if p == "O" or p == "o":
        p = "Onemax"
    elif p == "D" or p == "d":
        p = "Deception"

    # Main algorithm loop(Onemax)
    tool = cal()
    # w = SA(p=p, BitNum=100, iteration=1000, temperature=temperature, Run=51)
    # w.AI()  # store result in repective folder(.csv)
    # # Plotting
    # pp = tool.multiplot("./result/", [f"SA_{p}_{100}_{temperature}"], f"SA_{p}_combine")
    # pp.show()

    # Main algorithm loop(Deception n = 4,10)
    datalist = [4, 10]
    for i in datalist:
        v = SA(p=p, BitNum=i, iteration=1000, temperature=temperature, Run=51)
        v.AI()  # store result in repective folder(.csv)

    # Plotting
    pp = tool.multiplot(
        "./result/", [f"SA_{p}_{i}_{temperature}" for i in datalist], f"SA_{p}_combine"
    )
    pp.show()
