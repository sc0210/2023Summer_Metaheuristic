import random
import sys
import time

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
        self.name = f"{self.AntNum}{self.alpha}{self.beta}{self.EvaTime}_ACO"
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

    def Encode(self, curr_sol):
        """Encode the solution from given edge"""
        # Initialize bool matrix: np.full(shape, )
        EncodeMatrix = np.full((self.NodeNum, self.NodeNum), False, dtype=bool)
        for idx in curr_sol:
            i, j = idx[0] - 1, idx[1] - 1
            EncodeMatrix[i, j] = EncodeMatrix[j, i] = True
        return EncodeMatrix

    def RandSolConstruct(self):
        """Return sol (visited node order)
        Random shuffle approach"""
        sol = []
        NotVisited = list(range(0, self.NodeNum))
        random.shuffle(NotVisited)
        sol.extend(NotVisited)
        return sol

    def RandSquareMat(self, min, max, dim):
        """Random Generate distance cost matrix (DEBUG use)"""
        M = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                M[i][j] = M[j][i] = random.uniform(min, max)
                M[i][i] = 0
        return M

    def EdgeCost(self, DistMatrix, p1, p2):
        """Calculate distance cost of two given node index"""
        return DistMatrix[p1][p2]

    def RouteCost(self, DistMatrix, sol):
        """Calculate distance cost of a given solution(visited node)"""
        Cost = 0
        for i in range(self.NodeNum - 1):
            Cost += int(self.EdgeCost(DistMatrix, sol[i], sol[i + 1]))
            i += 2
        # print(f"Current cost:{Cost}")
        return Cost

    def DeltaSum(self, i, j, RouteTotalCost, db):
        """Return the sum of every solution cost that involve edge(i,j)"""
        update = 0
        for ant_idx, ant_sol in enumerate(db):
            if (
                (i in ant_sol)
                and (i != ant_sol[-1])
                and (ant_sol[ant_sol.index(i) + 1] == j)
            ):
                update += self.Q / float(RouteTotalCost[ant_idx])
                # print(f"Yes, [{i},{j}] {ant_idx}")
            # else:
            # print(f"Not found in ant {ant_idx}")
        return update

    def PheronomeUpdate(self, PheronomeMatrix, RouteTotalCost, db):
        """Update Pheronome through traverse every edge"""
        P, R = PheronomeMatrix, RouteTotalCost
        for j in range(self.NodeNum):
            for i in range(self.NodeNum):
                v = (1 - self.ER) * P[j][i] + self.DeltaSum(j, i, R, db)
                P[j][i] = P[i][j] = v
        return P

    def SolConstruct(self, PheronomeMatrix, DistMatrix):
        """Solution constuction(usage in single ant)

        Argument:
            PheronomeMatrix, DistMatrix
        Return:
            Solution(encoding set)
        """
        StartPoint = random.randint(0, self.NodeNum - 1)
        unvistied = list(range(self.NodeNum))

        visited = [StartPoint]
        unvistied.remove(StartPoint)

        while len(visited) != self.NodeNum:
            # Calculate the probability of next point
            prob = np.zeros(len(unvistied), dtype=float)
            for idx, u_idx in enumerate(unvistied):
                prob[idx] = self.NextPointProb(
                    visited[-1], u_idx, visited, PheronomeMatrix, DistMatrix
                )
            # Normalized the probablity (all prob. sum = 1)
            normalized_probs = [p / sum(prob) for p in prob]

            # Pick next point porportion to the prob calcluate in the above
            next_point = np.random.choice(unvistied, p=normalized_probs)
            visited.append(next_point)
            unvistied.remove(next_point)

        # print(visited)
        return visited

    def NextPointProb(self, i, j, visited, PheronomeMatrix, DistMatrix):
        """calculate next point probability(usage in SolConstruct)

        Argument:
            i, j, visited, PheronomeMatrix, DistMatrix
        Return:
            Probability
        """
        P, _D = PheronomeMatrix, np.where(DistMatrix != 0, 1 / DistMatrix, 0)
        dd = 0
        uu = (P[i][j]) ** self.alpha + (_D[i][j]) ** self.beta
        for v_idx in range(len(visited)):
            dd += (P[i][v_idx]) ** self.alpha + (_D[i][v_idx]) ** self.beta

        # Add a small constant to avoid division by zero
        epsilon = 1e-10
        dd += epsilon
        return float(uu) / float(dd)

    def RunAIEva(self):
        # (I) Initialize: Distance Matrix
        DistMat = self.FetchData()  # Fetch NodeNum
        # self.NodeNum = 3 # DEBUG testing use
        # DistMatrix = self.RandSquareMat(min=1.0, max=5.0, dim=self.NodeNum)

        # (I) Initialize: Pheronome Matrix
        PheronomeMat = self.RandSquareMat(min(DistMat), max(DistMat), dim=self.NodeNum)

        # (I) Initialize: Ant solution (random)
        db = [self.RandSolConstruct() for _ in range(self.AntNum)]

        # ================================================================================
        # (E) Evaluaie route cost for every sol
        cost_db = [self.RouteCost(DistMat, sol) for sol in db]

        # (T) Update pheronome table
        PheronomeMat = self.PheronomeUpdate(PheronomeMat, cost_db, db)

        # (D) Record the score for the initializaiton
        Global_min = min(cost_db)
        score = [Global_min]
        # ================================================================================
        # EvalTime loop
        self.cnt = 1
        while self.cnt < self.EvaTime - 1:
            # (T) Construction ant solution based on PheronomeMat & DistMat
            db = [self.SolConstruct(PheronomeMat, DistMat) for _ in range(self.AntNum)]

            # (E) Calculate each total distance cost & stored in cost_db
            cost_db = [self.RouteCost(DistMat, sol) for sol in db]
            Local_min = min(cost_db)  # pick the minimun
            print(f"Cnt:{self.cnt}, {Local_min}")

            # (D) Determine global/local optimal
            if Local_min < Global_min:
                Global_min = Local_min
                Global_sol = db[cost_db.index(Local_min)]
            score.append(Global_min)

            # (T) Update Pheronome table
            PheronomeMat = self.PheronomeUpdate(PheronomeMat, cost_db, db)
            # selected_idx = np.where(np.isin(np.sort(cost_db)[::-1][:5], db))[0].tolist()
            # db = [db[i] for i in selected_idx]

            self.cnt += 1
        print(Global_sol)
        return score

    def AI(self):
        for _ in range(self.Run):
            result = self.RunAIEva()
            print(f"Run:{_}, score ={result[-1]}")
            self.G.Write2CSV(data=result, filepath="./result", filename=self.name)

        # Visualization of the result
        y = self.G.AvgResult(filename=f"{self.name}.csv")
        self.G.Draw(y=y, filename=self.name)


if __name__ == "__main__":
    if len(sys.argv) == 8:
        ER = int(sys.argv[1])
        AntNum = int(sys.argv[2])
        Q = int(sys.argv[3])
        alpha = int(sys.argv[4])
        beta = int(sys.argv[5])
        EvalTime = int(sys.argv[6])
        Run = int(sys.argv[7])
    else:
        filename = [
            "./Dataset/a280.tsp",
            "./Dataset/bays29.tsp",
            "./Dataset/ali535.tsp",
        ]
        # Hyper
        # Initial reference from related paper
        # Control: ER; adjustable: others
        ER = 0.8
        Q = 4.0  # [0.5, 1, 1.5, 2,..] choose the best, then test next (alpha)
        alpha = 3  # 2 [0.5, 1, 1.5,..]
        beta = 2  # 1 [0.5, 1, 1.5,..]

        # Non-Hyper
        AntNum = 4  # City number
        EvalTime = 500
        Run = 20

    p = ACO(filename[1], ER, AntNum, Q, alpha, beta, EvalTime, Run)
    p.AI()

# 精進方法
# 策略一：自適應 減少超參數
# 策略二：觀察收斂結果 一隻ant的迭代過程
# ACS演算法
