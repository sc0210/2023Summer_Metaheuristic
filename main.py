import sys

import matplotlib.pyplot as plt
import numpy as np
from Algorithm.ES import ES
from Algorithm.GA import GA
from Algorithm.HC import HC
from Algorithm.SA import SA
from Algorithm.TS import TS
from Tool.Cal import cal

if len(sys.argv) == 4:
    BitNum = int(sys.argv[1])
    iteration = int(sys.argv[2])
    Run = int(sys.argv[3])
else:
    BitNum, iteration, Run = 100, 1000, 50

# Main progress
for _ in range(Run):
    ES_result = ES.RunAIEva(BitNum, iteration)
    sol2, HC_result = HC.RunAIEva(BitNum, "Rand", iteration)
    sol3, SA_result = SA.RunAIEva(BitNum, iteration, temperature=1)
    sol4, TS_result = TS.RunAIEva(BitNum, 20, iteration)
    sol5, GA_result = GA.RunAIEva(
        BitNum,
        iteration,
        crossover_rate=0.6,
        mutation_rate=0.1,
        pool_size=50,
        select_size=20,
    )
    cal.Write2CSV(
        ES_result,
        "./result",
        f"{BitNum}{iteration}{Run}_ES",
    )
    cal.Write2CSV(
        HC_result,
        "./result",
        f"{BitNum}{iteration}{Run}_HC",
    )
    cal.Write2CSV(
        SA_result,
        "./result",
        f"{BitNum}{iteration}{Run}_SA",
    )
    cal.Write2CSV(
        TS_result,
        "./result",
        f"{BitNum}{iteration}{Run}_TS",
    )
    cal.Write2CSV(
        GA_result,
        "./result",
        f"{BitNum}{iteration}{Run}_GA",
    )

groups = [
    "Exhaust Search",
    "Hill Climbing",
    "Stimulated Annealing",
    "Tabu Search",
    "Genetic Algorithm",
]
groups_short = ["ES", "HC", "SA", "TS", "GA"]
GroupNum = len(groups)

# Calculate the average result
DATA = [None] * GroupNum
for idx in range(GroupNum):
    DATA.append(cal.AvgResult(f"{BitNum}{iteration}{Run}_{groups_short[idx]}.csv"))

# Visualization the result
for idx in range(len(groups)):
    x = np.arange(0, len(DATA[idx]), 1)
    plt.plot(x, DATA[idx], label=groups[idx])

plt.grid()
# plt.ylim([0, max(data[:]) + 1])
plt.xlim([0, len(DATA[idx])])
plt.xlabel("Number of iteration", fontsize=21)
plt.ylabel("Object value", fontsize=21)
plt.title("Result", fontsize=25)
plt.legend(title_fontsize=20, loc="lower right")
plt.savefig("./result/Result.png")

plt.show()
