import csv
import math

import matplotlib.pyplot as plt
import numpy as np


class cal:
    def Fitness(self, curr):
        return sum(curr)

    def dec2bin(self, curr, BitNum):
        tmp = []
        while curr >= 2:
            quotient, reminder = divmod(curr, 2)
            tmp.append(reminder)
            # print(f"{curr},{quotient},{reminder}")
            curr = quotient  # update next value
        tmp.append(curr)

        # Fill zero to fuifull the bit string length
        while len(tmp) != BitNum:
            tmp.append(0)
        return tmp[::-1]

    def bin2dec(self, bin, BitNum):
        dec = 0
        for idx in range(BitNum):
            if bin[::-1][idx] == 1:
                dec += math.pow(2, idx)
        return int(dec)

    def Write2CSV(self, data, filepath, filename):
        with open(f"{filepath}/{filename}.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        # print("Export data to csv file")

    def Draw(self, y, filename, save=True):
        plt.grid()
        # plt.ylim([0, max(data[:]) + 1])
        plt.xlim([0, len(y)])
        plt.xlabel("Number of iteration", fontsize=21)
        plt.ylabel("Object value", fontsize=21)
        plt.title(filename, fontsize=25)
        x = np.arange(0, len(y), 1)
        plt.plot(
            x,
            y,
            color="green",
            ls="--",
            marker=".",
            markerfacecolor="k",
        )
        Y = [2020] * len(y)
        plt.plot(
            x,
            Y,
            color="red",
            ls="--",
            # marker=".",
            # markerfacecolor="k",
        )
        if save:
            plt.savefig(f"./result/{filename}.png")

    def AvgResult(self, filename):
        # Read multi-rows from csv file
        AllData, AvgData = [], []
        with open(f"./result/{filename}", "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                AllData.append(row)

        # Average rounds of result(O(m*n))
        # for i in range(np.shape(AllData)[1]):
        #     col = 0
        #     for j in range(np.shape(AllData)[0]):
        #         col += int(AllData[j][i])
        #     # print(col / np.shape(tmp)[0])
        #     AvgData.append(col)
        # AvgData = [idx / np.shape(AllData)[0] for idx in AvgData]

        # Numpy version(O(n))
        AvgData = np.mean(AllData, dtype="int", axis=0)
        return AvgData
