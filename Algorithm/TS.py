# tabu list
# enqueue -> .append({element})
# dequene -> .pop(0)
import os
import random
import sys
import time

import numpy as np

from Problem.problem import Problem
from Tool.Cal import cal


class TS(Problem):
    def __init__(self, p, TabuSize, BitNum, iteration, Run):
        super().__init__(p=p, BitNum=BitNum)
        # packages
        self.G = cal()
        # tabu
        self.tabulist_size = TabuSize
        self.tabulist = []
        # problem
        self.BitNum = BitNum
        self.iteration = iteration
        self.Run = Run
        self.name = f"TS_{self.problem}_{self.BitNum}_{self.tabulist_size}"

    def Neighbor(self, curr):
        """Return neighborhod solution(Transition)"""
        shiftBit = random.randint(0, self.BitNum - 1)
        new_curr = curr.copy()
        new_curr[shiftBit] = 1 - new_curr[shiftBit]
        return new_curr

    def TabuListCheck(self):
        while len(self.tabulist) > self.tabulist_size:
            self.tabulist.pop(0)
        return self.tabulist

    def NonTabuNeighborSelection(self, curr):
        while any((curr == x).all() for x in self.tabulist):
            new_curr = self.Neighbor(curr)
            curr = new_curr.copy()
        return curr

    def RunAIEva(self):
        """Tabu Search, TS"""
        # (I) Initialization
        self.cnt = 0
        curr_sol = np.array([random.randint(0, 1) for _ in range(self.BitNum)])
        Global_fitness = self.Fitness(curr_sol)

        solution_db = [curr_sol.tolist()]
        fitness_db = [Global_fitness]
        self.tabulist.append(solution_db[0])

        while self.cnt < self.iteration:
            # (T) Generate a neighbor solution: tmp_solution
            # check tmp_solution whether is or not in tabu list
            temp_sol = self.Neighbor(curr_sol)
            neighbor_sol = self.NonTabuNeighborSelection(temp_sol)

            # (E) Evaluateion
            Local_fitness = self.Fitness(neighbor_sol)

            # (D) Determination
            if Local_fitness > Global_fitness:
                # Update current soltion & fitness
                curr_sol = neighbor_sol.copy()
                Global_fitness = Local_fitness

                # update tabulist
                self.tabulist.append(neighbor_sol.tolist())
                self.TabuListCheck()

            solution_db.append(curr_sol.tolist())
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
        TabuSize = int(sys.argv[2])
    else:
        p = "Onemax"
        TabuSize = 20

    if p == "O" or p == "o":
        p = "Onemax"
    elif p == "D" or p == "d":
        p = "Deception"

    # Main algorithm loop
    for i in range(5, 10, 5):
        w = TS(p=p, TabuSize=i, BitNum=100, iteration=1000, Run=51)
        w.AI()

    # Plotting
    tool = cal()
    data_list = [f"TS_{p}_{100}_{i}" for i in range(5, 10, 5)]
    pp = tool.multiplot("./result/", data_list, f"TS_{p}_combine")
    pp.show()
