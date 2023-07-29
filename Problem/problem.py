import random

import numpy as np


class Problem:
    def __init__(self, p=None, BitNum=None):
        self.problem = str(p)
        self.BitNum = int(BitNum)

        # Deception problem parameter
        self.GT = list(np.array([random.randint(0, 1) for _ in range(self.BitNum)]))

    def Fitness(self, curr_sol):
        try:
            if self.problem == "Onemax" or self.problem == "O":
                fitness_val = np.sum(curr_sol)
            elif self.problem == "Deception" or self.problem == "D":
                cnt = 0
                for idx, data in enumerate(curr_sol):
                    if curr_sol[idx] == self.GT[idx]:
                        cnt += 1
                fitness_val = cnt
            return fitness_val

        except NameError:
            print("[ERROR], please check problem type...")
            print("It must be 'O' for 'Onemax' or 'D' for 'Deception'.\n")
