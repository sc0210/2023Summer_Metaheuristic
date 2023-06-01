import random
import sys
import time

import numpy as np
from Tool.Cal import cal


class GA:
    def __init__(self, BitNum, EvaTime, cr, mr, pool_size, select_size, Run):
        self.BitNum = BitNum
        self.EvaTime = EvaTime  # Evaluation
        self.cr = cr  # Crossover rate
        self.mr = mr  # Mutation rate
        self.p = pool_size
        self.s = select_size
        self.G = cal()
        self.Run = Run
        self.name = f"{self.BitNum}{self.EvaTime}{self.Run}_GA"
        self.cnt = 0

    def Fitness(self, curr):
        return np.sum(curr)

    def Selection(self, curr_sol, Local_fitness, method):
        """Selection, return index array in Selection range"""
        Total = len(Local_fitness)
        T, prob = [] * Total, [] * Total

        if method == "Roulette wheel":
            for idx in range(Total):
                prob.append(Local_fitness[idx] / (sum(Local_fitness)))
            T = np.random.choice(Total, self.s, p=prob)
            T = T.tolist()

        elif method == "Tournament tour":
            for _ in range(self.s):
                # players number default = 4
                players = np.random.choice(Total, 4, replace=None)
                T.append(np.argmax(players))

        elif method == "Eliste":
            # return index array arrange from small to large
            SORT = np.argsort(Local_fitness)
            T = SORT[::-1][: self.s][::-1]

        # print(f"\n-> Selcetion method:{method}, idx:{T}")
        return [curr_sol[idx].copy() for idx in T]

    def CrossOver(self, curr_sol, method):
        curr = curr_sol.copy()
        if method == "One point":
            slice_index = random.randint(0, len(curr[0]) - 1)
            for idx in range(len(curr) // 2):
                if random.random() < self.cr:
                    # print(f"Before: {curr[i]},{curr[i+1]}")
                    temp = curr[idx].copy()
                    curr[idx][slice_index:] = curr[idx + 1][slice_index:]
                    curr[idx + 1][slice_index:] = temp[slice_index:]
                    # print(f"After: {curr[i]},{curr[i+1]}")
                idx += 2  # CrossOver by pairs

        elif method == "Two point":
            slice_index = np.random.choice(len(curr[0]) - 1, 2, replace=None)
            slice_index = np.sort(slice_index)
            for idx in range(len(curr) // 2):
                if random.random() < self.cr:
                    # print(f"Before: {curr[i]},{curr[i+1]}")
                    temp = curr[idx].copy()
                    curr[idx][slice_index:] = curr[idx + 1][slice_index:]
                    curr[idx + 1][slice_index:] = temp[slice_index:]
                    # print(f"After: {curr[i]},{curr[i+1]}")
                idx += 2

        # print(f"\n-> CrossOver rate:{self.cr}, method:'{method}'")
        return curr  # solution array

    def Mutation(self, curr_sol):
        mutatue_cnt = 0
        for sol in curr_sol:
            if random.random() < self.mr:
                Mutation_index = random.randint(0, len(curr_sol[0]) - 1)
                # print(f"(B)Mutate: {sol}, Mutation_index: {Mutation_index}")
                sol[Mutation_index] = 1 - sol[Mutation_index]
                # print(f"(A)Mutate: {sol}")
                mutatue_cnt += 1
        # print(f"\n-> Mutation rate:{self.mr}, precentage:{100*mcnt/len(curr_sol)}%")
        return curr_sol

    def RunAIEva(self):
        """Genetic algorithm"""
        solution_db, fitness_db, curr_sol = [], [], []
        # Random initialize (pool size)
        self.cnt = 1
        for _ in range(self.p):
            curr_sol.append([random.randint(0, 1) for _ in range(self.BitNum)])
        Lf = [self.Fitness(sol) for sol in curr_sol]
        Global_fitness = np.max(Lf)
        fitness_db.append(Global_fitness)

        while self.cnt < self.EvaTime:
            # (T) Selection, Crossover , Mutation
            Lf = [self.Fitness(sol) for sol in curr_sol]
            s = self.Selection(curr_sol, Lf, "Roulette wheel")
            c = self.CrossOver(s, "One point")
            m = self.Mutation(c)

            # (E) EvaTime
            Local_Fitness = np.max([self.Fitness(sol) for sol in m])
            curr_sol = m.copy()  # Assign offspring to next parent

            # (D) Determine
            if Local_Fitness >= Global_fitness:
                Global_fitness = Local_Fitness
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return solution_db, fitness_db

    def AI(self):
        st = time.time()
        for _ in range(self.Run):
            sol, result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)

            if (self.cnt) % 10 == 0:
                print(
                    "No.{:<3}, Obj:{:<2}, Time:{:<3}".format(
                        self.cnt, np.max(result), np.round(time.time() - st, 3)
                    )
                )

        # Visualization of the result
        self.G.Draw(self.G.AvgResult(f"{self.name}.csv"), self.name)


if __name__ == "__main__":
    if len(sys.argv) == 9:
        BitNum = int(sys.argv[1])
        EvaTime = int(sys.argv[2])
        cr = int(sys.argv[3])
        mr = int(sys.argv[4])
        pool_size = int(sys.argv[5])
        select_size = int(sys.argv[6])
        avgtime = int(sys.argv[7])
        Run = int(sys.argv[8])

    else:
        cr, mr, pool_size = 0.9, 0.4, 50
        BitNum, select_size, EvaTime, Run = 100, 50, 1000, 10

    p = GA(BitNum, EvaTime, cr, mr, pool_size, select_size, Run)
    p.AI()
