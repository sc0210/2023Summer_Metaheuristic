import csv
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from Problem.problem import Problem
from Tool.Cal import cal


class HC(Problem):
    def __init__(self, p, Mode, BitNum, iteration, Run):
        super().__init__(p=p, n=BitNum)
        # packages
        self.G = cal()
        # hill climbing
        self.mode = str(Mode)
        # problem
        self.BitNum = int(BitNum)
        self.iteration = int(iteration)
        self.Run = int(Run)

        self.name = f"HC_{self.Problem}_with{self.mode}"

    def Neighbor(self, curr, Mode):
        try:
            if Mode == "Rand":
                ShiftBit = random.randint(0, self.BitNum - 1)
                new_curr = curr.copy()
                new_curr[ShiftBit] = 1 - new_curr[ShiftBit]
            elif Mode == "LR":
                chose = random.randint(0, 1)
                # chose = 1
                if chose == 1:
                    tmp = (self.G.bin2dec(curr, self.BitNum) - 1) % self.BitNum
                else:
                    tmp = (self.G.bin2dec(curr, self.BitNum) + 1) % self.BitNum
                new_curr = self.G.dec2bin(tmp, self.BitNum)
            return new_curr
        except NameError:
            print("[ERROR], please check HC Mode")

    def RunAIEva(self):
        """Hill Climbing, HC"""
        # (I) Initialize by random
        self.cnt = 0
        Global_sol = np.array([random.randint(0, 1) for _ in range(0, self.BitNum)])
        Global_fitness = self.Fitness(Global_sol)
        fitness_db = [Global_fitness]
        sol_db = [Global_sol.tolist()]

        while self.cnt < self.iteration:
            # (T) Transition
            neighbor_sol = self.Neighbor(Global_sol, Mode=self.mode)

            # (E) Evaluation
            Local_fitness = self.Fitness(neighbor_sol)

            # (D) Determine
            if Local_fitness > Global_fitness:
                Global_sol = neighbor_sol.copy()
                Global_fitness = Local_fitness

            sol_db.append(Global_sol.tolist())
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return sol_db, fitness_db

    def AI(self):
        print("\n============/START of the Evaluation/============")
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
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        end = time.time()
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============\n")


if __name__ == "__main__":
    # Hyperparameters
    if len(sys.argv) == 2:
        p = str(sys.argv[1])  # Problem: Onemax or Deception
    else:
        p = "Onemax"

    if p == "O":
        p = "Onemax"
    elif p == "D":
        p = "Deception"

    # Main algorithm loop
    Group = ["Rand", "LR"]  # Transition methods
    for i in Group:
        w = HC(p="Onemax", Mode=i, BitNum=100, iteration=1000, Run=51)
        w.AI()  # store result in repective folder(.csv)

    # Plotting
    tool = cal()
    data_list = [f"HC_{p}_with{idx}" for idx in Group]
    p = tool.multiplot("./result/", data_list, f"HC_{p}_combine")
    p.show()
