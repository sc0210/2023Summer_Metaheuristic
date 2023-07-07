# tabu list
# enqueue -> .append({element})
# dequene -> .pop(0)
import os
import random
import sys
import time

import numpy as np

from Tool.Cal import cal


class TS:
    def __init__(self, BitNum, TabuSize, iteration, Run):
        self.BitNum = BitNum
        self.tabulist_size = TabuSize
        self.iteration = iteration
        self.tabulist = []
        self.name = f"{self.tabulist_size}_TS"
        self.G = cal()
        self.Run = Run

    def Neighbor(self, curr):
        ShiftBit = random.randint(0, self.BitNum - 1)
        new_curr = curr.copy()
        new_curr[ShiftBit] = 1 - new_curr[ShiftBit]
        return new_curr

    def TabuListCheck(self):
        while len(self.tabulist) > self.tabulist_size:
            self.tabulist.pop(0)
        return list

    def NonTabuNeighborSelection(self, curr):
        while any((curr == x).all() for x in self.tabulist):
            new_curr = self.Neighbor(curr)
            curr = new_curr.copy()
        return curr

    # ==========================================================================================
    def RunAIEva(self):
        """Tabu Search"""
        # (I) Initialization
        curr_sol = np.array([random.randint(0, 1) for _ in range(self.BitNum)])
        Global_fitness = self.G.Fitness(curr_sol)
        self.tabulist.append(curr_sol.tolist())

        solution_db = [curr_sol]
        fitness_db = [Global_fitness]
        self.cnt = 0

        while self.cnt < self.iteration:
            # (T) Generate a neighbor solution: tmp_solution
            # check tmp_solution whether is or not in tabu list
            temp_sol = self.Neighbor(curr_sol)
            neighbor_sol = self.NonTabuNeighborSelection(temp_sol)

            # (E) Evaluateion
            Local_fitness = self.G.Fitness(neighbor_sol)

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

        # Result visualization
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        self.G.Draw(AvgResult, self.name)
        end = time.time()
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        TabuSize = int(sys.argv[1])
    else:
        TabuSize = 20

    progress = TS(BitNum=100, TabuSize=TabuSize, iteration=1000, Run=50)
    progress.AI()
