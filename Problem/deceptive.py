import random

import numpy as np


class Deception:
    def __init__(self, n):
        self.n = n
        self.GT = list(np.array([random.randint(0, 1) for _ in range(self.n)]))

    def Fitness(self, curr_sol):
        count = 0
        for idx, data in enumerate(curr_sol):
            if curr_sol[idx] == self.GT[idx]:
                count += 1
        return count
