# saw_elastic_predictions/src/euler_transformations.py
import numpy as np 

class EulerAngles:
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r

    def to_matrix(self):
        """Converts Euler angles (radians) to a rotation matrix"""
        a, b, r = self.a, self.b, self.r
        Rza = np.array([
            [np.cos(a), np.sin(a), 0],
            [-np.sin(a), np.cos(a), 0],
        ])
        Rxb = np.array([
            [0, np.cos(b), np.sin(b)],
            [0, -np.sin(b), np.cos(b)],
        ])
        Rzr = np.array([
            [np.cos(r), np.sin(r), 0],
            [-np.sin(r), np.cos(r), 0],
        ])
        return Rza @ Rxb @ Rzr


    def C_modifi(C, a):
        """Transforms the elastic constant tensor C based on transformation matrix a."""
        newC = np.zeros((3, 3, 3, 3))
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    for lp in range(3):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    for l in range(3):
                                        newC[ip, jp, kp, lp] += a[ip, i] * a[jp, j] * a[kp, k] * a[lp, l] * C[i, j, k, l]
        return newC