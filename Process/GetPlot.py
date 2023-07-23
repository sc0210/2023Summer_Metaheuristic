# GetPlot.py -- combine any reault you would to compare

from Tool.Cal import cal

tool = cal()


# Plotting
DstFilename = "Deception_Problem(HC vs SA)"
data_list = [
    "HC_D_4_10",
    "HC_D_10_10",
    "HC_D_100_10",
    "SA_D_4_10",
    "SA_D_10_10",
    "SA_D_100_10",
]


# data_list = ["SA_D_4_10", "SA_D_10_10"]
p = tool.multiplot("./result/", data_list, DstFilename)
p.show()
