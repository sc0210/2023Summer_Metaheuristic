# precision: 6, 10e-6
import sys
import time

import numpy as np
from numpy import e, pi
from Tool.Cal import cal


class PSO:
    def __init__(
        self,
        ParticleNum,
        Dimension,
        Step,
        Inertia,
        a1,
        a2,
        EvaTime,
        EdgeConstraint,
        Run,
    ):
        self.ParticleNum = ParticleNum
        self.dim = Dimension
        self.step = Step
        self.w = Inertia
        self.a1 = a1  # alpha1
        self.a2 = a2  # alpha2
        self.C = EdgeConstraint  # Ths size of the edge
        self.EvaTime = EvaTime
        self.SubSolNum = int((self.dim / self.step) ** 2)

        # store global optimal for each evatime
        self.gb = np.zeros((self.EvaTime, 2), dtype=float)
        # personal best for each particle
        self.pb = np.zeros((self.ParticleNum, 2), dtype=float)
        self.v = np.zeros((self.ParticleNum, 2), dtype=float)  # velocity
        self.p = np.zeros((self.ParticleNum, 2), dtype=float)  # position
        self.s = []  # np.zeros((self.ParticleNum, self.SubSolNum, 2), dtype=float)

        self.name = f"{self.dim}{self.ParticleNum}{self.a1}{self.a2}_PSO"
        self.Run = Run
        self.G = cal()

    def obj_func(self, x, y):
        """Ackley multimodal function"""
        return (
            -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * pi * x) + np.cos(2 * pi * y)))
            + e
            + 20
        )

    def RandVal(self):
        return np.random.uniform(-self.C, self.C)

    def Rand2D(self):
        return np.array(
            [[self.RandVal(), self.RandVal()] for _ in range(self.ParticleNum)]
        )

    def SubSolution(self, x, y):
        pp = self.dim / 2
        x_vals = np.arange(x - pp, x + pp, self.step)
        y_vals = np.arange(y - pp, y + pp, self.step)
        X, Y = np.meshgrid(x_vals, y_vals)
        subsol = np.column_stack((X.flatten(), Y.flatten()))
        return subsol

    def Initialize(self):
        # Position & velocity
        self.p = self.Rand2D()  # [1,3,2,3,..]
        self.v = self.Rand2D()  # [1,3,4,2,..]

        # Create subsolution
        self.s = []
        for p_idx in range(self.ParticleNum):
            self.s.append(self.SubSolution(self.p[p_idx][0], self.p[p_idx][1]))

        # Update Gb
        temp = np.array([self.obj_func(p[0], p[1]) for p in self.p])
        self.gb[0] = self.p[np.argmin(temp)]

        # Update Pb
        for p_idx in range(self.ParticleNum):
            temp = np.array([self.obj_func(s[0], s[1]) for s in self.s[p_idx]])
            self.pb[p_idx] = self.s[p_idx][np.argmin(temp)]
        self.cnt += 1

    def UpdateVelocity(self):
        """Udpate velocity based on formula"""
        self.v = np.round(
            self.w * self.v
            + self.a1 * np.random.random() * (self.pb - self.p)
            + self.a2 * np.random.random() * (self.gb[self.cnt - 1] - self.p),
            3,
        )

    def UpdatePosition(self):
        """Udpate position & subsolution based on formula"""
        self.s = []
        self.p = np.round(self.p + self.v, 3)
        for p_idx in range(self.ParticleNum):  # idx = particle index
            self.s.append(self.SubSolution(self.p[p_idx][0], self.p[p_idx][1]))

    def GB_Update(self):
        """Update self.gb(General best) from self.p"""
        temp = np.array([self.obj_func(p[0], p[1]) for p in self.p])
        self.gb[self.cnt] = self.p[np.argmin(temp)]
        return self.gb[self.cnt]

    def PB_Update(self):
        """Update self.pb(Personal best) from self.s"""
        for p_idx in range(self.ParticleNum):
            temp = np.array([self.obj_func(s[0], s[1]) for s in self.s[p_idx]])
            self.pb[p_idx] = self.s[p_idx][np.argmin(temp)]
        return self.pb

    def EdgeDetection(self):
        """While particle approach to the edge of the constraint"""

    def RunAIEva(self):
        # (I)Initialization
        self.cnt = 0
        self.Initialize()
        # print(f"-> (I) Particle soluiton, shape={np.shape(self.p)} ")
        # print("shape of subsoluiton:", np.shape(self.s))
        # print("-> GB:{}".format(self.gb[self.cnt]))

        Global_Opt = self.gb[0]
        score = [np.round(self.obj_func(Global_Opt[0], Global_Opt[1]), 3)]

        while self.cnt < self.EvaTime:
            # print(f"\n> Evatime {self.cnt} start!")
            # (T)(E)
            self.UpdateVelocity()
            self.UpdatePosition()

            # (E)
            self.PB_Update()
            self.GB_Update()
            Local_Opt = self.gb[self.cnt]

            # (D)
            Local_OBJ = np.round(self.obj_func(Local_Opt[0], Local_Opt[1]), 3)
            Global_OBJ = np.round(self.obj_func(Global_Opt[0], Global_Opt[1]), 3)
            if Local_OBJ < Global_OBJ:
                Global_Opt = Local_Opt
            score.append(Global_OBJ)

            self.cnt += 1
        return score

    def AI(self):
        print("============/START of the Evaluation/============")
        st = time.time()
        for Run_index in range(self.Run):
            result = self.RunAIEva()
            self.G.Write2CSV(np.array(result), "./result", self.name)

            if Run_index % 10 == 0:
                print(
                    "Run.{:<2}, Obj:{:<2}, Time:{:<3}\n".format(
                        Run_index,
                        np.max(result),
                        np.round(time.time() - st, 3),
                    )
                )

        # Visualization of the result
        self.G.Draw(self.G.AvgResult(f"{self.name}.csv"), self.name)
        print("============/END of the Evaluation/============")


# 5 particle, 25 subsolution per particle
# self.p = [[x1,y1], [x2,y2], [x3,y3], ...] (5,2)
# self.s = [ [[xx1,yy1],[xx2,yy2],..], [[xx1,yy1],[xx2,yy2],..], [..] ] (5,100,2)

if __name__ == "__main__":
    if len(sys.argv) == 8:
        ParticleNum = int(sys.argv[1])
        Dimension = int(sys.argv[2])
        EdgeConstraint = int(sys.argv[3])
        Step = int(sys.argv[4])
        Inertia = float(sys.argv[5])
        a1 = float(sys.argv[6])
        a2 = float(sys.argv[7])

    else:
        ParticleNum, Dimension, EdgeConstraint, Step = 30, 10, 40, 1
        Inertia, a1, a2 = 0.8, 0.8, 0.5
    p = PSO(
        ParticleNum=ParticleNum,
        Dimension=Dimension,
        Step=Step,
        Inertia=Inertia,
        a1=a1,
        a2=a2,
        EdgeConstraint=EdgeConstraint,
        EvaTime=1000,
        Run=50,
    )
    p.AI()
