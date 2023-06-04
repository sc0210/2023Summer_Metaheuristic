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
        self.Run = Run
        self.name = f"{self.cr}{self.mr}{self.p}_GA"
        self.G = cal()

        self.cnt = 0
        self.players = 4  # Tournament tour

    def Fitness(self, curr):
        return int(np.sum(curr))

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
                players = np.random.choice(Total, self.players, replace=None)
                T.append(np.argmax(players))

        elif method == "Eliste":
            # return index array arrange from small to large
            SORT = np.argsort(Local_fitness)
            T = SORT[::-1][: self.s][::-1]

        # print(f"\n-> Selcetion method:{method}, idx:{T}")
        return [curr_sol[idx].copy() for idx in T]

    def CrossOver(self, curr_sol, method):
        """Crossover"""
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
            p = slice_index[0]
            q = slice_index[1]

            for idx in range(len(curr) // 2):
                if random.random() < self.cr:
                    # print(f"Before: {curr[i]},{curr[i+1]}")
                    temp = curr[idx].copy()
                    curr[idx] = np.concatenate(
                        (
                            curr[idx][:p],
                            curr[idx + 1][p:q],
                            curr[idx][q:],
                        )
                    )
                    curr[idx + 1] = np.concatenate(
                        (
                            curr[idx + 1][:p],
                            curr[idx][p:q],
                            curr[idx + 1][q:],
                        )
                    )
                    # print(f"After: {curr[i]},{curr[i+1]}")
                idx += 2

        # print(f"\n-> CrossOver rate:{self.cr}, method:'{method}'")
        return curr  # solution array

    def Mutation(self, curr_sol):
        """Mutation"""
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
        curr_sol = []
        # Random initialize (pool size)
        self.cnt = 1
        for _ in range(self.p):
            curr_sol.append([random.randint(0, 1) for _ in range(self.BitNum)])
        Lf = [self.Fitness(sol) for sol in curr_sol]
        Global_sol = curr_sol[np.argmax(Lf)]
        Global_fitness = np.max(Lf)

        solution_db = [Global_sol]
        fitness_db = [Global_fitness]

        while self.cnt < self.EvaTime:
            st = time.time()
            # (T) Selection, Crossover , Mutation
            Lf = [self.Fitness(sol) for sol in curr_sol]
            s = self.Selection(curr_sol, Lf, "Roulette wheel")
            c = self.CrossOver(s, "Two point")
            m = self.Mutation(c)

            # (E) Evaluation
            Local_Fitness = np.max([self.Fitness(sol) for sol in m])
            curr_sol = m.copy()  # Assign offspring to next parent

            # (D) Determine
            if Local_Fitness >= Global_fitness:
                Global_fitness = Local_Fitness
                Global_sol = m[np.argmax(Local_Fitness)]
            solution_db.append(Global_sol)
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return solution_db, fitness_db

    def AI(self):
        for _ in range(self.Run):
            st = time.time()
            sol, result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)
            if _ % 10 == 0:
                print(
                    "Run.{:<3}, Obj:{:<2}, Time:{:<3}".format(
                        _, np.max(result), np.round(time.time() - st, 3)
                    )
                )
                Bst = [int(x) for x in sol[-1]]

                print(f"Best solution:{Bst}")

        # Visualization of the result
        self.G.Draw(self.G.AvgResult(f"{self.name}.csv"), self.name)


if __name__ == "__main__":
    if len(sys.argv) == 8:
        cr = float(sys.argv[1])
        mr = float(sys.argv[2])
        pool_size = int(sys.argv[3])
        select_size = int(sys.argv[4])
        BitNum = int(sys.argv[5])
        EvaTime = int(sys.argv[6])
        Run = int(sys.argv[7])

    else:
        cr, mr, pool_size, select_size = 0.9, 0.4, 50, 7
        BitNum, EvaTime, Run = 100, 1000, 20

    p = GA(BitNum, EvaTime, cr, mr, pool_size, select_size, Run)
    p.AI()
