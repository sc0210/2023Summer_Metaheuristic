import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


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
            # bit wise operation
            if bin & (1 << idx):  # bin[::-1][idx] == 1
                dec += 2**idx
        return dec

    def Write2CSV(self, data, FILE):
        with open(FILE, "a") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        # print("Export data to csv file")

    def Draw(self, y, filename, save=True):
        # fig, ax = plt.subplots(figsize=(xsize, ysize))
        # ax = plt.gca()

        # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        plt.figure(figsize=(10, 6))  # Width: 8 inches, Height: 6 inches
        plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
        plt.rcParams["font.size"] = 15
        plt.rcParams["legend.fontsize"] = 18
        plt.grid()
        # plt.ylim([0, max(data[:]) + 1])
        # plt.xlim([0, len(y)])
        plt.xlabel("Number of iteration")
        plt.ylabel("Object value")
        plt.title(filename)
        x = np.arange(0, len(y), 1)
        plt.plot(
            x,
            y,
            # color="green",
            ls="--",
            marker=".",
            markerfacecolor="k",
        )
        # Y = [2020] * len(y)
        # plt.plot(
        #     x,
        #     Y,
        #     color="red",
        #     ls="--",
        #     # marker=".",
        #     # markerfacecolor="k",
        # )
        if save:
            plt.savefig(f"./result/{filename}.png")

    def AvgResult(self, FILE):
        # Read multi-rows from csv file
        AllData = []
        with open(f"{FILE}", "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                AllData.append(row)

        # Convert AllData to a NumPy array
        AllData = np.array(AllData, dtype=float)

        # Calculate the average using NumPy
        AvgData = np.mean(AllData, axis=0)

        return AvgData
