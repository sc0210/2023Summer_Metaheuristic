# tabu list
# enqueue -> .append({element})
# dequene -> .pop(0)
import random
import sys

from Tool.Cal import cal


class TS:
    def __init__(self, BitNum, tabulist_size, iteration):
        self.BitNum = BitNum
        self.tabulist_size = tabulist_size
        self.iteration = iteration
        self.name = f"{self.BitNum}{self.iteration}_TS"
        self.G = cal()

    def Neighbor(self, curr, BitNum):
        ShiftBit = random.randint(0, BitNum - 1)
        new_curr = curr.copy()
        new_curr[ShiftBit] = 1 - new_curr[ShiftBit]
        return new_curr

    def TabuListCheck(self, list, size):
        while len(list) > int(size):
            list.pop(0)
        return list

    def NonTabuNeighborSelection(self, tabulist, curr):
        while curr in tabulist:
            new_curr = self.Neighbor(curr, self.BitNum)
            curr = new_curr.copy()
        # tabulist.append(curr)
        # tabulist=TabuListCheck(tabulist,5)
        return curr, tabulist

    # ==========================================================================================
    def RunAIEva(self, BitNum, tabulist_size, iteration):
        """Tabu Search"""
        solution_db, fitness_db, tabulist = [], [], []

        # Ranodm initalize
        curr_sol = [random.randint(0, 1) for _ in range(BitNum)]
        Global_fitness = self.G.Fitness(curr_sol)

        solution_db.append(curr_sol)
        fitness_db.append(Global_fitness)
        tabulist.append(curr_sol)

        for _ in range(iteration):
            # (T) Generate a neighbor solution: tmp_solution
            tmp_sol = self.Neighbor(curr_sol, BitNum)
            neighbor_sol, tabulist = self.NonTabuNeighborSelection(
                tabulist, tmp_sol, BitNum
            )  # check tmp_solution whether is or not in tabu list

            # (E) Evaluateion
            Local_fitness = self.G.Fitness(neighbor_sol)

            # (D) Determination
            if Local_fitness > Global_fitness:
                # Update current soltion & fitness
                curr_sol = neighbor_sol.copy()
                Global_fitness = Local_fitness

                # solution_db.append(curr_sol)
                fitness_db.append(Global_fitness)

                # update tabulist
                tabulist.append(neighbor_sol)
                tabulist = self.TabuListCheck(tabulist, tabulist_size)

            else:
                # solution_db.append(curr_sol)
                fitness_db.append(Global_fitness)

        return solution_db, fitness_db


if __name__ == "__main__":
    if len(sys.argv) == 5:
        BitNum, TabuSize, iteration, Run = (
            int(sys.argv[1]),
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
        )
    else:
        BitNum, TabuSize, iteration, Run = 10, 20, 1000, 50

    progress = TS(BitNum, TabuSize, iteration)

    for _ in range(Run):
        sol, result = progress.RunAIEva(BitNum, TabuSize, iteration)
        progress.G.Write2CSV(result, "./result", progress.name)

    # Visualization of the result
    y = progress.G.AvgResult(f"{progress.name}.csv")
    progress.G.Draw(y, progress.name)
