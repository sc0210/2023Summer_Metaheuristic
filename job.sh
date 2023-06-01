#!/bin/sh
# Indiviual calculation
# [ES.py/ HS.py] [BitNum] [iteration] [avgtime]
python -m Algorithm.ES 100 1000 50
python -m Algorithm.HC 100 1000 50

# [SA.py] [BitNum] [tempature] [iteration] [avgtime]
python -m Algorithm.SA 100 11000  50 

# [TS.py] [BitNum] [tabulistsize,r:7~10] [iteration] [avgtime]
python -m Algorithm.TS 100 7 1000 50

# [GA.py] [BitNum] [iteration] [crossover_rate] 
#         [mutation_rate] [pool_size] [select_size] [avgtime]
python -m Algorithm.GA 100 20 0.6 0.1 50 20 50

# [ACO.py] [ER] [AntNum] [Q] [alpha] [beta] [EvalTime] [Run]
python -m Algorithm.ACO 0.8 4 4.0 3 2 500 20

# Comparsion among different algo.
# python main.py 100 1000 50
