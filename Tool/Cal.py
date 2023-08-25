import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from Algorithm.TSP import P


class cal:
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
        rev_bin = bin[::-1]
        for idx in range(BitNum):
            # bit wise operation
            if rev_bin[idx] == 1:  # bin.any() & (1 << idx):
                dec += 2**idx
        return dec

    def Write2CSV(self, data, FILE):
        """
        w  write mode
        r  read mode
        a  append mode

        w+  create file if it doesn't exist and open it in write mode
        r+  open for reading and writing. Does not create file.
        a+  create file if it doesn't exist and open it in append mode"""
        if os.path.isfile(FILE):
            with open(FILE, "a") as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            with open(FILE, "w+") as file:
                writer = csv.writer(file)
                writer.writerow(data)

    def AvgResult(self, SrcFilepath):
        """Read multi-rows and average the data

        Arguments:
            FILE: target file loaction(filepath)
        Return:
            matplotlib.pyplot plot
        """
        with open(f"{SrcFilepath}", "r") as file:
            # Read multi-rows from csv file
            csvreader = csv.reader(file)

            # Convert to numpy array (float) format
            RawData = np.array([row for row in csvreader], float)
            print(f"location: '{SrcFilepath}', data shape: {np.shape(RawData)}")

            # Average the data
            AvgData = np.mean(RawData, axis=0)

        return AvgData

    def multiplot(self, RootFilepath, data_list, DstFilename):
        """Plot multi csv data in one figure

        Arguments:
            Root_folder: folder where store all the result (filepath)
            data_list: target datas to plot in the figure (filename list)
            Dst_filename: filename for the combination data (filename)
        Return:
            matplotlib.pyplot plot
        """
        # read data from mutli csv file
        r = [0] * len(data_list)
        print("============/START of the Multiplot/============")
        for idx, addr in enumerate(data_list):
            r[idx] = self.AvgResult(f"{RootFilepath}{addr}.csv")

        plt.figure(figsize=(10, 6))  # Width: 8 inches, Height: 6 inches
        plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
        plt.rcParams["font.size"] = 15
        plt.rcParams["legend.fontsize"] = 18
        plt.grid()
        plt.xlabel("Number of iteration")
        plt.ylabel("Object value")
        plt.title(DstFilename)

        x = np.arange(0, len(r[0]), 1)
        for idx, data in enumerate(r):
            plt.plot(
                x,
                r[idx],
                ls="--",
                marker=".",
                markerfacecolor="k",
                label=f"{data_list[idx]}",
            )
        plt.legend()
        plt.savefig(f"./result/{DstFilename}.png")
        print("============/END of the Multiplot/============\n")
        return plt
