import random
import sys
import time

import numpy as np
from Tool.Cal import cal


class HC:
    def __init__(self, BitNum, iteration, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.Run = Run
        self.name = f"{self.BitNum}{self.iteration}{self.Run}_HC"
        self.G = cal()
        self.cnt = 0

    def Fitness(self, curr):
        return np.sum(curr)

    def Neighbor(self, curr, Mode="Rand"):
        if Mode == "Rand":
            ShiftBit = random.randint(0, self.BitNum - 1)
            new_curr = curr.copy()
            new_curr[ShiftBit] = abs(1 - new_curr[ShiftBit])
        elif Mode == "LR":
            # chose = random.randint(0, 1)
            chose = 1
            if chose == 1:
                tmp = (self.G.bin2dec(curr, self.BitNum) - 1) % self.BitNum
            else:
                tmp = (self.G.bin2dec(curr, self.BitNum) + 1) % self.BitNum
            new_curr = self.G.dec2bin(tmp, self.BitNum)
        else:
            pass

        return new_curr

    def RunAIEva(self, Mode="Rand"):
        """Hill Climbing, HC"""
        solution_db, fitness_db = [], []

        # Random initialize
        curr_sol = [random.randint(0, 1) for _ in range(0, self.BitNum)]
        Global_fitness = self.Fitness(curr_sol)
        fitness_db.append(Global_fitness)
        # print(f"Inital state: {curr_sol}")

        idx = 0
        st = time.time()

        while idx < self.iteration:
            # (T)Transition
            neighbor_sol = self.Neighbor(curr_sol, Mode=Mode)

            # (E)Evaluation
            Local_fitness = self.Fitness(neighbor_sol)

            # (D)Determine
            if Local_fitness > Global_fitness:
                curr_sol = neighbor_sol.copy()
                Global_fitness = Local_fitness
                fitness_db.append(Global_fitness)
            else:
                fitness_db.append(Global_fitness)
            self.cnt += 1
            idx += 1
        return solution_db, fitness_db

    def AI(self):
        st = time.time()
        for _ in range(self.Run):
            sol, result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)
            # print("No.{:<2}, Obj:{:<5}".format(idx, np.max(result)))
            if self.cnt % 25 == 0:
                print(
                    "No.{:<3}, Obj:{:<2}, Time:{:<4}".format(
                        self.cnt, np.max(result), np.round(time.time() - st, 4)
                    )
                )

        # Visualization of the result
        y = self.G.AvgResult(f"{self.name}.csv")
        self.G.Draw(y, self.name)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        BitNum, iteration, Run = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    else:
        BitNum, iteration, Run = 100, 100, 50

    p = HC(BitNum, iteration, Run)
    p.AI()
