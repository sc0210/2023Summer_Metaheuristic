import random

import numpy as np


class Onemax:
    def __init__(self, n):
        self.n = n
        self.GT = list(np.array([random.randint(0, 1) for _ in range(self.n)]))

    def Fitness(self, curr_sol):
        return np.sum(curr_sol)
