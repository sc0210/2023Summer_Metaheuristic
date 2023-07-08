# Stimulated Annealing

## Overview

Stimulated Annealing(SA), similiar to [Hill Climibing](./HC.md) (pick neighbor solution by adding slightly movement). The difference is determination part, SA compute a acceptance probability (based on current temperature).

In every iteration, SA accept the solution in two conditions: whether the one with better fitness value or random value is smaller than acceptance probability.

The acceptance probability is given by the Boltzmann distribution: exp(-delta/T), where delta is the difference in value between the current and neighbor solutions, and T is the current temperature.
![SA_formula](./SA_formula.webp)

- pros:
    1. Time efficient

- cons:
    1. Might caught into local optimal

## Pseduocode

```shell
#(I)Initialization
Random initialize v
Local_fitness = Global_fitness = Fitness(v) 

while not met termination condition:

    #(T)Transition
    NeghborSolution(v) = p

    #(E)Evaluaiton
    Local_fitness = Fitness(p)

    #(D)Determination
    if Local_fitness > Global_fitness:
        Global_fitness = Local_fitness
    # Annealing process
    else random_value < acceptance_prob:
        Global_fitness = Local_fitness
        Update temperature

return Global_fitness
```

- Transition: find next solution (neighbor solution) by only modify one bits in the solution (add slightly movement)
- Evaluation: count the number of 1 bits in the solution
- Determination: compare with global optimal, update if it gain better evaluation or a random value is smaller than acceptance probability

## Flowchart

![Flowchart](./TED_flowchart.svg)

## Instructions for running on local machine

1. packages used in this projects:

    - numpy==1.24.2
    - matplotlib==3.7.1
    - pandas==1.5.3

2. Run code

    ```shell
    # sys.argv[1]: temperature
    python -m Algorithm.SA 10 
    ```

3. Folder organiation

    - Each algorithm will generate two files:
        - {filename}.png: show the trend/process of certain algo.
        - {filename}.csv: record every global optimal in every iterations
    - Check all the result in [**result**](../result/) folder
    - ![result for exhausive search](../result/10_SA.png)
