import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from Problem.deceptive import Deception
from Tool.Cal import cal


class SA_D(Deception):
    def __init__(self, n, iteration, temperature, Run):
        super().__init__(n)
        self.iteration = iteration
        self.temperature = temperature
        self.Run = Run
        self.name = f"{self.__class__.__name__ }_{self.n}_{self.temperature}"
        self.G = cal()

    def Neighbor(self, curr):
        """Return neighborhod solution(Transition)"""
        shiftbit = random.randint(0, self.n - 1)
        new_curr = curr.copy()
        new_curr[shiftbit] = 1 - new_curr[shiftbit]
        return new_curr

    def RunAIEva(self):
        """Simulated annealing, self"""
        # (I) Initialization
        self.cnt = 0

        Global_sol = np.array([random.randint(0, 1) for _ in range(self.n)])
        Global_fitness = self.Fitness(Global_sol)
        sol_db = [Global_sol.tolist()]
        fitness_db = [Global_fitness]

        while self.cnt <= (self.iteration - 2) and (self.temperature > 0):
            # (T) Transition
            neighbor_sol = self.Neighbor(Global_sol)

            # (E) Evaluation & calculate the acceptance probability
            neighbor_fitness = self.Fitness(neighbor_sol)
            delta_fitness = neighbor_fitness - Global_fitness
            acceptance_prob = min(
                1, np.exp(delta_fitness / self.temperature)
            )  # negative ->[0,1]

            # (D) Determine
            if neighbor_fitness > Global_fitness:
                Global_sol = neighbor_sol.copy()
                Global_fitness = neighbor_fitness
                # print(f"Accept better {neighbor_fitness}/{curr_fitness}_{i}")
            # elif random.random() < acceptance_prob:
            #     Global_sol = neighbor_sol.copy()
            #     Global_fitness = neighbor_fitness
            #     self.temperature *= 0.95
            #     # print(f"Accept self  {neighbor_fitness}/{curr_fitness}_{i}")

            sol_db.append(Global_sol.tolist())
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return self.GT, sol_db, fitness_db

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        if not os.path.isdir("./result/"):
            os.makedirs("./result")

        # Average the result from multiple runs
        for Run_index in range(self.Run):
            gt_sol, sol, fitness_result = self.RunAIEva()
            self.G.Write2CSV(
                np.array(fitness_result), f"./result/{self.name}.csv"
            )  # store processing csv file in result folder

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Obj:{:<2}, Time:{:<3}\nGT:{}, Best solution:{}\n".format(
                        Run_index,
                        np.max(fitness_result),
                        np.round(time.time() - st, 3),
                        gt_sol,
                        sol[-1],
                    )
                )

        # Avg result
        end = time.time()
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============\n")


if __name__ == "__main__":
    # Hyperparameters
    if len(sys.argv) == 2:
        temperature = int(sys.argv[1])
    else:
        temperature = 10

    Group = [4, 10, 100]  # n (deception problem's prameters)

    # Main algorithm loop
    for ii in Group:
        w = SA_D(n=ii, iteration=1000, temperature=temperature, Run=51)
        w.AI()  # store result in repective folder(.csv)

    # Ploting
    tool = cal()
    data_list = [f"SA_D_{idx}_{temperature}" for idx in Group]
    p = tool.multiplot("./result/", data_list, "SA_D_combine")
    # p.show()
