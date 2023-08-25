import concurrent.futures
import multiprocessing
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from Algorithm.TSP import P
from Tool.Cal import cal


class ACO(P):
    def __init__(self, filename, ER, AntNum, Q, alpha, beta, EvalTime, Run):
        super().__init__(
            filename=filename,
        )
        self.AntNum = AntNum
        self.ER = ER  # EvaporateRate
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.EvaTime = EvalTime
        self.Run = Run
        self.name = f"ACO_{self.ER}{self.Q}{self.alpha}{self.beta}"
        self.G = cal()

    def Check(self, curr_sol):
        """Check solution whether it covers all nodes"""
        visited = set()
        for node_pair in curr_sol:
            visited.update(node_pair)  # Union the value
            # Check if it covers all the nodes
            if len(visited) == self.NodeNum:
                return True
        return False

    def BrutalGenerate(self):
        """Generate route path from given node number"""
        MAX_ATTEMPTS = 1000
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            # Possible paths for given node number
            Lib = [
                [i, j]
                for i in range(1, self.NodeNum)
                for j in range(i + 1, self.NodeNum + 1)
            ]

            # Pick (node num - 1) numbers of edges
            total_edges = int(self.NodeNum * (self.NodeNum - 1) / 2)
            selected_indices = np.random.choice(
                total_edges, self.EdgeNum, replace=False
            )
            sol = [Lib[idx] for idx in selected_indices]

            # Check if the solution matches
            if self.Check(sol):
                return sol

            attempts += 1

        print("Could not find a valid solution within the maximum attempts.")
        return None

    def RandSolConstruct(self):
        """Return sol (visited node order)
        Note: Random shuffle approach"""

        # Sol = [0,1,2,...(nodenum-1)]
        sol = list(range(0, self.NodeNum))
        random.shuffle(sol)
        return sol

    def RandSquareMat(self, Max, dim):
        """Random Generate distance cost matrix"""
        M = np.ones((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i + 1, dim):
                M[i, j] = M[j, i] = random.uniform(0, Max)
        return M

    def DeltaSum(self, i, j, RouteTotalCost, AntSol):
        """Return the sum of every solution cost that involve edge(i,j) 8.2"""
        delta_k = 0
        # Check whether current ant solution exist edge(i,j)
        for k_idx, sol in enumerate(AntSol):
            if (i in sol) and (i != sol[-1]) and (sol[sol.index(i) + 1] == j):
                delta_k += self.Q / RouteTotalCost[k_idx]
                # print(f"Yes, [{i},{j}] {ant_idx}")
        return delta_k

    def PheronomeUpdate(self, PheMatrix, RouteTotalCost, AntSol):
        """Update Pheronome through traverse every edge 8.4"""
        # Evaporate rate
        P = PheMatrix.copy() * self.ER
        # Update the pheronome matrix
        for i in range(self.NodeNum):
            for j in range(i, self.NodeNum):
                P[i, j] += self.DeltaSum(i, j, RouteTotalCost, AntSol)
        return P

    def SolConstruct(self, PheMat):
        """Solution construction(usage in single ant)

        Argument:
            PheMatrix, DistMatrix
        Return:
            Solution(encoding set)
        """
        StartPoint = random.randint(0, self.NodeNum - 1)
        unvisited = list(range(self.NodeNum))

        visited = [StartPoint]
        unvisited.remove(StartPoint)

        while len(visited) != self.NodeNum:
            # Calculate the probability of next point
            prob = np.zeros(len(unvisited), dtype=float)
            for idx, unvi_node in enumerate(unvisited):
                prob[idx] = self.NextPointProb(
                    visited[-1], unvi_node, unvisited, PheMat
                )
            # Normalized the probablity (all prob. sum = 1)
            normalized_probs = [p / np.sum(prob) for p in prob]

            # Pick next point proportion to the prob calcluate in the above
            next_point = np.random.choice(unvisited, p=normalized_probs)
            visited.append(next_point)
            unvisited.remove(next_point)

        # print(visited)
        return visited

    def NextPointProb(self, i, j, unvisited, PheMat):
        """calculate next point probability(pass over to SolConstruct)

        Argument:
            i, j, visited, PheMatrix, DistMatrix
        Return:
            Probability
        """
        epsilon = 1e-10  # Add a small constant to avoid division by zero

        P = PheMat
        _D = np.where(self.DistMat != 0, 1 / self.DistMat, 0)

        uu = np.power(P[i, j], self.alpha) + np.power(_D[i, j], self.beta)
        dd = np.sum(
            np.power(P[i, unvisited], self.alpha)
            + np.power(_D[i, unvisited], self.beta)
        )
        dd = np.where(dd != 0, dd, epsilon)
        return uu / dd

    def local_search(self, DistMat, AntSol):
        AntSol_cost = [self.RouteCost(DistMat, sol) for sol in AntSol]
        # partition = int(self.AntNum * 20%)
        # Pick = np.sort(AntSol_cost)[::-1][:partition]
        T, prob = [] * self.AntNum, [] * self.AntNum

        for idx in range(self.AntNum):
            prob.append(AntSol_cost[idx] / (np.sum(AntSol_cost)))
        T = np.random.choice(self.AntNum, self.AntNum, replace=False, p=prob)
        T = T.tolist()
        return T

    def RunAIEva(self, result_queue, sol_queue):
        # (I) Initialize: DistMax, PheMat, AntSol
        self.DistMat = self.FetchData()  # Fetch NodeNum
        PheMat = np.full((self.NodeNum, self.NodeNum), 0.01)
        AntSol = [self.RandSolConstruct() for _ in range(self.AntNum)]

        # (E) Evaluate route cost for every sol
        AntSol_cost = [self.RouteCost(sol) for sol in AntSol]

        # (D) Record the score for the initializaiton
        Global_sol = [AntSol[AntSol_cost.index(np.min(AntSol_cost))]]
        Global_min = np.min(AntSol_cost)
        sol = [Global_sol]
        score = [Global_min]

        # EvalTime loop
        self.cnt = 1
        while self.cnt <= self.EvaTime:
            # (T) Construct ant solution
            AntSol = [self.SolConstruct(PheMat) for _ in range(self.AntNum)]

            # (E) Calculate each total distance cost & stored in AntSol_cost
            AntSol_cost = [self.RouteCost(sol) for sol in AntSol]
            Local_min = np.min(AntSol_cost)  # pick the minimun

            # (D) Determine global/local optimal
            if Local_min < Global_min:
                Global_min = Local_min
                Global_sol = AntSol[AntSol_cost.index(Global_min)]
            score.append(Global_min)
            sol.append(Global_sol)
            print(f"Eva:{self.cnt}, Local_min:{Local_min}, Global_min:{Global_min}")

            # (T) Update Pheronome table
            PheMat = self.PheronomeUpdate(PheMat, AntSol_cost, AntSol)

            # Local search
            # selected = self.local_search(DistMat, AntSol)
            # AntSol = selected.copy()

            self.cnt += 1
        print(Global_sol)
        result_queue.put(score)
        sol_queue.put(Global_sol)
        return score, sol

    def AI_m(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        core = 6
        process_list = []
        result_queue = multiprocessing.Queue()  # Queue for collecting results
        sol_queue = multiprocessing.Queue()  # Queue for collecting results

        # Process queue
        for _ in range(self.Run):
            process = multiprocessing.Process(
                target=self.RunAIEva,
                args=(
                    result_queue,
                    sol_queue,
                ),
            )
            process_list.append(process)
        # Process partition(by a given core numbers)
        p = [0]
        while self.Run > 1:
            if self.Run > core:
                p.append(p[-1] + core)
                self.Run -= core
            else:
                p.append(p[-1] + self.Run - 1)
                self.Run -= self.Run

        # Start all processes
        for idx, partition in enumerate(p):
            collected_results = []
            collected_sol = []
            if idx == p.index(p[-1]):
                break

            for process in process_list[p[idx] : p[idx + 1]]:
                process.start()
            # Wait for all processes to finish
            for process in process_list[p[idx] : p[idx + 1]]:
                process.join()

            # Collect results from the queue
            while not result_queue.empty():
                collected_results.append(result_queue.get())
                collected_sol.append(sol_queue.get())

            for i in range(len(collected_results)):
                self.G.Write2CSV(collected_results[i], f"./result/{self.name}.csv")
                self.G.Write2CSV(collected_sol[i], f"./result/{self.name}_sol.csv")
                self.PlotPath(collected_sol[i], self.RouteCost(collected_sol[i]))
        print("============/END of the Evaluation/============")

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        result_queue = multiprocessing.Queue()  # Queue for collecting results
        sol_queue = multiprocessing.Queue()  # Queue for collecting results
        if not os.path.isdir("./result/"):
            os.makedirs("./result")

        # Average the result from multiple runs
        for Run_index in range(self.Run):
            fitness_result, sol = self.RunAIEva(result_queue, sol_queue)
            self.G.Write2CSV(fitness_result, f"./result/{self.name}.csv")
            self.PlotPath(sol, self.RouteCost(sol))

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Best Obj:{:<2}, Time:{:<3}\n".format(
                        Run_index, np.min(fitness_result), np.round(time.time() - st, 3)
                    )
                )

        # Avg result
        end = time.time()
        AvgResult = self.G.AvgResult(f"./result/{self.name}.csv")
        print(f"Average max: {np.max(AvgResult)}, Total runtime: {end-st} sec")
        print("============/END of the Evaluation/============")


if __name__ == "__main__":
    if len(sys.argv) == 8:
        ER = float(sys.argv[1])
        AntNum = int(sys.argv[2])
        Q = float(sys.argv[3])
        alpha = int(sys.argv[4])
        beta = int(sys.argv[5])
        EvalTime = int(sys.argv[6])
        Run = int(sys.argv[7])
    else:
        filename = [
            "./Dataset/eil51.tsp",
            "./Dataset/a280.tsp",
            "./Dataset/bays29.tsp",
            "./Dataset/ali535.tsp",
            "./Dataset/aa.tsp",
        ]
        # Hyper
        ER = 0.8
        Q = 2.0  # [0.5, 1, 1.5, 2,..]
        alpha = 4  # 2 [0.5, 1, 1.5,..] # Pheronome
        beta = 3  # 1 [0.5, 1, 1.5,..] # Distance

        # Non-Hyper
        AntNum = 100  # City number
        EvalTime = 1000
        Run = 45

    p = ACO(filename[0], ER, AntNum, Q, alpha, beta, EvalTime, Run)
    # p.AI_m()  # with multi-processing
    # p.AI() #without multi-processing

    # Plotting
    tool = cal()
    pp = tool.multiplot(
        "./result/",
        [f"ACO_{ER}{Q}{alpha}{beta}"],
        "ACO_combine",
    )
    pp.show()
