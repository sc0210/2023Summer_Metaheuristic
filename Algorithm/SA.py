import os
import random
import sys
import time

import numpy as np

from Tool.Cal import cal


class SA:
    def __init__(self, BitNum, iteration, temperature, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.temperature = temperature
        self.Run = Run
        self.name = f"{self.temperature}_SA"
        self.G = cal()

    def Fitness(self, curr_sol):
        return np.sum(curr_sol)

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

        # Result visualization
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        self.G.Draw(AvgResult, self.name)
        end = time.time()
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        temperature = int(sys.argv[1])
    else:
        temperature = 10

    p = SA(BitNum=100, iteration=1000, temperature=temperature, Run=50)
    p.AI()
