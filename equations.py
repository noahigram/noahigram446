from timesteppers import StateVector
from scipy import sparse
import numpy as np


class ViscousBurgers:

    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])

        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix

        def f(X): return -X.data*(d @ X.data)

        self.F = f


class Wave:

    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        if isinstance(rho0, (int, float)):
            Irho0 = rho0*I

        else:
            Irho0 = sparse.diags(rho0, 0)
        self.M = sparse.bmat([[Irho0, Z], [Z, I]])

        if isinstance(gammap0, (int, float)):
            gammap0d = gammap0*d.matrix

        else:
            darray = d.matrix.toarray()
            newMatrix = np.zeros(shape=np.shape(darray))
            for i in range(np.shape(darray)[0]):
                newMatrix[i, :] = gammap0[i]*darray[i, :]
            gammap0d = sparse.csr_matrix(newMatrix)

        self.L = sparse.bmat([[Z, d.matrix], [gammap0d, Z]])

        self.F = lambda X: 0*X.data


class ReactionDiffusion:

    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        self.M = sparse.eye(N, N)
        self.L = -D*d2.matrix
        self.F = lambda X: X.data*(c_target - X.data)
