import os
import random
import sys
import time

import numpy as np

from Tool.Cal import cal


class HC:
    def __init__(self, Mode, BitNum, iteration, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.Run = Run
        self.mode = mode
        self.name = f"HC_with{self.mode}"
        self.G = cal()
        self.cnt = 0

    def Fitness(self, curr):
        return np.sum(curr)

    def Neighbor(self, curr, Mode):
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
        else:
            pass

        return new_curr

    def RunAIEva(self):
        """Hill Climbing, HC"""
        # (I) Initialize by random
        Global_sol = np.array([random.randint(0, 1) for _ in range(0, self.BitNum)])
        Global_fitness = self.Fitness(Global_sol)
        fitness_db = [Global_fitness]
        sol_db = [Global_sol.tolist()]
        self.cnt = 0

        while self.cnt < self.iteration:
            # (T)Transition
            neighbor_sol = self.Neighbor(Global_sol, Mode=self.mode)

            # (E)Evaluation
            Local_fitness = self.Fitness(neighbor_sol)

            # (D)Determine
            if Local_fitness > Global_fitness:
                Global_sol = neighbor_sol.copy()
                Global_fitness = Local_fitness

            sol_db.append(Global_sol.tolist())
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return sol_db, fitness_db

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
        mode = str(sys.argv[1])
    else:
        mode = "Rand"  # Two mode:Rand or LR

    p = HC(Mode=mode, BitNum=100, iteration=1000, Run=51)
    p.AI()
