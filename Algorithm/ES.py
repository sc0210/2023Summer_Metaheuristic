import sys
import time

from Tool.Cal import cal


class ES:
    def __init__(self, BitNum, iteration, Run):
        self.BitNum = BitNum
        self.iteration = iteration
        self.Run = Run
        self.name = f"{self.BitNum}{self.iteration}_ES"
        self.TL = 1800
        self.G = cal()

    def Fitness(self, curr_sol):
        return sum(curr_sol)

    def RunAIEva(self):
        """Exhaust search"""
        st = ed = time.time()
        cnt = 0
        fitness_db = []

        # Initial state: [0,0,...,0]
        Local_sol = self.G.dec2bin(cnt, self.BitNum)
        Global_fitness = self.Fitness(Local_sol)
        fitness_db.append(Global_fitness)

        while (cnt <= self.iteration - 1) and (ed - st <= self.TL):
            # (T)Transition
            cnt += 1
            Local_sol = self.G.dec2bin(cnt, BitNum)
            # Local_sol = Neighbor.copy()

            # (E)Evaluation
            Local_fitness = self.Fitness(Local_sol)

            # (D)Determination
            if Local_fitness > Global_fitness:
                fitness_db.append(Local_fitness)
                Global_fitness = Local_fitness
            else:
                fitness_db.append(Global_fitness)

            ed = time.time()
        return fitness_db

    def AI(self):
        for _ in range(self.Run):
            result = self.RunAIEva()
            self.G.Write2CSV(result, "./result", self.name)

        # Visualization of the result
        y = self.G.AvgResult(f"{self.name}.csv")
        self.G.Draw(y, self.name)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        BitNum, iteration, Run = (
            int(sys.argv[1]),
            int(sys.argv[2]),
            int(sys.argv[3]),
        )
    else:
        BitNum, iteration, Run = 100, 1000, 51

    p = ES(BitNum, iteration, Run)
    p.AI()
