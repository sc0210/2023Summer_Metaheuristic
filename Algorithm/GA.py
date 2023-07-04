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
        self.name = f"{self.cr}{self.mr}{self.p}{self.s}_GA"
        self.G = cal()

        self.cnt = 0
        self.players = 4  # Tournament tour
        self.cnt = 1  # evaluation counts
        self.curr_sol = []  # store curr spring solution

    def Fitness(self, curr):
        return np.sum(curr)

    def Selection(self, curr_sol, Local_fitness, method):
        """Selection, return solution array via different method"""
        pool_size = len(Local_fitness)
        T, prob = [] * pool_size, [] * pool_size

        if method == "Roulette wheel":
            # Roulette wheel: pick by the probability proportional to its fitness values
            for idx in range(pool_size):
                prob.append(Local_fitness[idx] / (sum(Local_fitness)))
            T = np.random.choice(pool_size, self.s, p=prob)
            T = T.tolist()

        elif method == "Tournament tour":
            # Tournament tour: pick the top players in every iteration
            for _ in range(self.s):
                selected_p = np.random.choice(pool_size, self.players, replace=None)
                T.append(np.argmax(selected_p))

        elif method == "Eliste":
            # Eliste: always pick the largest value
            SORT = np.argsort(Local_fitness)
            T = SORT[::-1][: self.s][::-1]

        # print(f"\n-> Selcetion method:{method}, idx:{T}")
        return [curr_sol[idx].copy() for idx in T]

    def CrossOver(self, curr_sol, method):
        """Crossover"""
        curr = curr_sol.copy()
        if method == "One point":
            slice_index = np.random.randint(0, len(curr[0]))
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
        mutate_cnt = 0
        for sol in curr_sol:
            if random.random() < self.mr:
                Mutation_index = np.random.randint(0, len(curr_sol[0]))
                # print(f"(B)Mutate: {sol}, Mutation_index: {Mutation_index}")
                sol[Mutation_index] = 1 - sol[Mutation_index]
                # print(f"(A)Mutate: {sol}")
                mutate_cnt += 1
        # print(f"\n-> Mutation rate:{self.mr}, precentage:{100*mcnt/len(curr_sol)}%")
        return curr_sol

    def Initialization(self):
        # Create population (p: pool size)
        self.cnt = 0
        self.curr_sol = []
        for _ in range(self.p):
            self.curr_sol.append(
                np.array(([random.randint(0, 1) for _ in range(self.BitNum)]))
            )
        Lf = [self.Fitness(sol) for sol in self.curr_sol]
        Global_sol = self.curr_sol[np.argmax(Lf)]
        Global_fitness = np.max(Lf)

        sol_db = [Global_sol.tolist()]
        fitness_db = [Global_fitness]
        return Global_sol, Global_fitness, sol_db, fitness_db

    def RunAIEva(self):
        """Genetic algorithm"""
        # (I) Initialization
        Global_sol, Global_fitness, sol_db, fitness_db = self.Initialization()

        while self.cnt < self.EvaTime:
            st = time.time()
            # (T) Selection, Crossover , Mutation
            Lf = [self.Fitness(sol) for sol in self.curr_sol]
            s = self.Selection(self.curr_sol, Lf, "Roulette wheel")
            c = self.CrossOver(s, "One point")  # "One point" or "Two point"
            m = self.Mutation(c)

            # (E) Evaluation
            Local_Fitness = np.max([self.Fitness(sol) for sol in m])
            self.curr_sol = m.copy()  # Assign offspring to next parent

            # (D) Determine
            if Local_Fitness >= Global_fitness:
                Global_fitness = Local_Fitness
                Global_sol = m[np.argmax(Local_Fitness)]
            sol_db.append(Global_sol.tolist())
            fitness_db.append(Global_fitness)
            self.cnt += 1
        return sol_db, fitness_db

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        for Run_index in range(self.Run):
            sol, result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Obj:{:<2}, Time:{:<3}\nBest solution:{}\n".format(
                        Run_index,
                        np.max(result),
                        np.round(time.time() - st, 3),
                        [sol[-1]],
                    )
                )

        # Visualization of the result
        self.G.Draw(self.G.AvgResult(f"{self.name}.csv"), self.name)
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        cr = float(sys.argv[1])
        mr = float(sys.argv[2])
        p = int(sys.argv[3])  # pool(population) size
        s = int(sys.argv[4])  # selection size

    else:
        cr, mr, p, s = 0.9, 0.3, 100, 7

    p = GA(cr=cr, mr=mr, pool_size=p, select_size=s, EvaTime=1000, BitNum=100, Run=50)
    p.AI()
