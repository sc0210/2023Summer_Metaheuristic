import sys
import time

import numpy as np

from Tool.Cal import cal


class ES:
    def __init__(self, BitNum, iteration, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.Run = Run
        self.name = "ES"
        self.G = cal()

    def Fitness(self, curr_sol):
        return np.sum(curr_sol)

    def RunAIEva(self):
        """Exhaust search"""
        # Initial state: [0,0,...,0]
        self.cnt = 0
        Global_sol = np.array(self.G.dec2bin(self.cnt, self.BitNum))
        Global_fitness = self.Fitness(Global_sol)
        sol_db = [Global_sol.tolist()]
        fitness_db = [Global_fitness]

        while self.cnt < self.iteration:
            # (T)Transition
            self.cnt += 1
            Local_sol = np.array(self.G.dec2bin(self.cnt, self.BitNum))

            # (E)Evaluation
            Local_fitness = self.Fitness(Local_sol)

            # (D)Determination
            if Local_fitness > Global_fitness:
                Global_fitness = Local_fitness
                Global_sol = Local_sol
            fitness_db.append(Global_fitness)
            sol_db.append(Global_sol)
        return sol_db, fitness_db

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        for Run_index in range(self.Run):
            sol, result = self.RunAIEva()
            self.G.Write2CSV(np.array(result), "./result", self.name)

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Obj:{:<2}, Time:{:<3}\n".format(
                        Run_index, np.max(result), np.round(time.time() - st, 3)
                    )
                )

        # Visualization of the result
        self.G.Draw(self.G.AvgResult(f"{self.name}.csv"), self.name)
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    p = ES(BitNum=100, iteration=1000, Run=50)
    p.AI()
