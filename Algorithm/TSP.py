import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class P:
    # NodeNum = 0

    def __init__(self, filename):
        self.filename = filename
        self.NodeNum = 0
        self.line_number = sys.maxsize
        self.DistMat = self.FetchData()  # Fetch NodeNum

    def EdgeWeight(self):
        """Handle data with given a Distance Mat"""
        tmp = []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if (idx >= self.line_number + 1) and (
                    idx <= self.line_number + self.NodeNum
                ):
                    tmp.append(line.split())
        return tmp

    def NodeCord_ori(self):
        """Handle data with e.g.[1, 36.49, 7.49]"""
        x, y = [], []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if "DIMENSION" in line:
                    self.NodeNum = int(line.replace(" ", "").split(":")[1])

                if "NODE_COORD_SECTION" in line:
                    self.line_number = idx
                if (idx >= self.line_number + 1) and (
                    idx <= self.line_number + self.NodeNum
                ):
                    cor = line.split()
                    x.append(int(cor[1]))
                    y.append(int(cor[2]))
        return x, y

    def NodeCord(self):
        """Handle data with e.g.[1, 36.49, 7.49]"""
        tmp = []
        data = np.zeros((self.NodeNum, self.NodeNum), dtype=float)
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if (idx >= self.line_number + 1) and (
                    idx <= self.line_number + self.NodeNum
                ):
                    tmp.append(line.split())

        for i in range(self.NodeNum):
            for j in range(self.NodeNum):
                p1, p2 = tmp[i], tmp[j]  # (Node_index, X, Y)
                data[i, j] = data[j, i] = self.EuclideanDist(p1, p2)

        # np.fill_diagonal(data, 0)  # Set diagonal elements to 0 (data[i][i] = 0)
        return data

    def FetchData(self):
        """Fetch data from TSPLIB dateset, return pd.dataframe"""
        data = []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if "DIMENSION" in line:
                    self.NodeNum = int(line.replace(" ", "").split(":")[1])

                if "NODE_COORD_SECTION" in line:
                    self.line_number = idx
                    data = self.NodeCord()

                elif "EDGE_WEIGHT_SECTION" in line:
                    self.line_number = idx
                    data = self.EdgeWeight()

        # Translate to np.array format
        data_array = np.array(data, dtype=float)
        print(f"-> Load: {self.filename},{data_array.shape}, {self.NodeNum} nodes\n")
        return data_array

    def EuclideanDist(self, p1, p2):
        """Euclidean Distance calculateion,2D cooridnates"""
        # (base) Direct approach
        # xx = float(p2[1]) - float(p1[1])
        # yy = float(p2[2]) - float(p1[2])
        # result = math.sqrt((np.power(xx, 2) + np.power(yy, 2)))
        # return np.round(result, 3)

        # (Opt) Numpy version
        p1_np = np.array([p1[1], p1[2]], dtype=float)
        p2_np = np.array([p2[1], p2[2]], dtype=float)
        distance = np.linalg.norm(p2_np - p1_np)
        return np.round(distance, 2)

    def EdgeCost(self, p1, p2):
        """Calculate distance cost of two given node index"""
        # Notice: node index start from 0,1,...(NodeNum-1)
        return self.DistMat[p1, p2]

    def RouteCost(self, sol):
        """Calculate distance cost of a given solution(visited node)"""
        Cost = self.EdgeCost(sol[-1], sol[0])  # first point to last point
        for i in range(self.NodeNum - 2):
            Cost += self.EdgeCost(sol[i], sol[i + 1])
        # print(f"Current cost:{Cost}")
        return np.round(Cost, 3)

    def PlotPath(self, sol, score):
        DistMat = self.NodeCord_ori()
        x, y = DistMat[0], DistMat[1]

        # spot
        for idx in range(self.NodeNum):
            plt.scatter(x[idx], y[idx], c="b", marker="o", s=15)

        # line
        for idx in range(self.NodeNum):
            if idx == self.NodeNum - 1:
                break
            plt.plot(
                [x[sol[idx]], x[sol[idx + 1]]],
                [y[sol[idx]], y[sol[idx + 1]]],
                c="g",
                linewidth=0.8,
                linestyle="--",
            )
        plt.suptitle("Mininum Path")
        plt.title("For a minimum distance of {}".format(score), fontsize=10)
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")

        # plt.show()
        plt.savefig(f"./result/TSP_image/{score}.png")
        plt.close()


if __name__ == "__main__":
    filename = ["./Dataset/bays29.tsp", "./Dataset/aa.tsp", "./Dataset/eil51.tsp"]
    p = P(filename=filename[-1])
    df = p.FetchData()
    print(df)
