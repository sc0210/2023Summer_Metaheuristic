import os
import time

import numpy as np

from Tool.Cal import cal


class ES:
    def __init__(self, BitNum, Run):
        self.BitNum = BitNum
        self.Run = Run
        self.TimeLimit = 30 * 60  # unit: seconds
        self.name = "ES"
        self.G = cal()

    def Fitness(self, curr_sol):
        return np.sum(curr_sol)

    def RunAIEva(self):
        """Exhaust search"""
        # (I) Initialization
        # Initial state: [0,0,...,0]
        self.cnt = 0
        Global_sol = np.array(self.G.dec2bin(self.cnt, self.BitNum))
        Global_fitness = self.Fitness(Global_sol)
        sol_db = [Global_sol.tolist()]
        fitness_db = [Global_fitness]
        st = time.time()

        while time.time() - st <= self.TimeLimit:
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

            # Show currrent calculation
            print(f"No.{self.cnt} | Global_Opt: {Global_fitness}")
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

            print(
                "Run.{:<2}, Obj:{:<2}, Time:{:<3}s\n".format(
                    Run_index, np.max(fitness_result), np.round(time.time() - st, 3)
                )
            )

        # Result visualization
        self.G.Draw(self.G.AvgResult(f"./result/{self.name}.csv"), self.name)
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    p = ES(BitNum=100, Run=1)
    p.AI()
