import sys

import numpy as np
import pandas as pd


class P:
    # NodeNum = 0

    def __init__(self, filename):
        self.filename = filename
        self.NodeNum = 0
        self.line_number = sys.maxsize

    def EdgeWeight(self):
        """Handle data with given a Distance Mat"""
        data = []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if self.line_number + 1 <= idx <= self.line_number + self.NodeNum:
                    data.append(line.split())
        return data

    def NodeCord(self):
        """Handle data with e.g.[1, 36.49, 7.49]"""
        tmp = []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if (idx >= self.line_number + 1) and (
                    idx <= self.line_number + self.NodeNum
                ):
                    tmp.append(line.split())

        data = np.zeros((self.NodeNum, self.NodeNum), dtype=float)
        for i in range(self.NodeNum):
            for j in range(1, self.NodeNum):
                p1, p2 = tmp[i], tmp[j]
                data[i][j] = data[j][i] = self.EuclideanDist(p1, p2)
                data[i][i] = 0
        return data

    def FetchData(self):
        """Fetch data from TSPLIB dateset, return pd.dataframe"""
        data = []
        with open(f"{self.filename}", "r") as file:
            for idx, line in enumerate(file):
                if "DIMENSION" in line:
                    self.NodeNum = int(line.split()[1])

                if "NODE_COORD_SECTION" in line:
                    self.line_number = idx
                    data = self.NodeCord()

                elif "EDGE_WEIGHT_SECTION" in line:
                    self.line_number = idx
                    data = self.EdgeWeight()

        # Translate to dataframe format
        df = pd.DataFrame(data, dtype=float)
        print(f"-> Load: {self.filename},{df.shape}, {self.NodeNum} nodes\n")
        return df

    def EuclideanDist(self, p1, p2):
        """Euclidean Distance calculateion,2D cooridnates"""

        xx = float(p2[0]) - float(p1[0])
        yy = float(p2[1]) - float(p1[1])
        result = (xx**2 + yy**2) ** (1 / 2)
        # print(np.round(result, 3))
        return np.round(result, 3)


if __name__ == "__main__":
    filename = ["./Dataset/bays29.tsp", "./Dataset/ali535.tsp"]
    p = P(filename=filename[1])
    df = p.FetchData()
    print(df)
