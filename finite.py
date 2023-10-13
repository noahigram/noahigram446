import numpy as np
from scipy.special import factorial
from scipy import sparse
import findiff
from findiff import FinDiff


class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Difference:

    def __matmul__(self, other):
        return self.matrix@other


class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        n = derivative_order
        m = convergence_order
        # Create offsets
        offset = math.floor((2*math.floor((n+1)/2) + m - 1)/2)
        offsets = range(-offset, offset+1)

        # Create b array and S matrix
        b = np.zeros(shape=(len(offsets),))
        b[n] = 1
        S = np.zeros(shape=(len(offsets), len(offsets)))
        for i in range(len(offsets)):
            for j in range(len(offsets)):
                S[i, j] = (offsets[j]*grid.dx)**i/factorial(i)

        # solve for a coefficients
        a = np.linalg.solve(S, b)
        D = sparse.diags(a, offsets=offsets, shape=[grid.N, grid.N])
        D = D.tocsr()

        # Place corner elements
        for i in range(offset):
            for j in range(offset-i):
                D[i, -(j+1)] = a[offset-i-j-1]
                D[-(i+1), j] = a[-(offset-i-j)]

        self.matrix = D


class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        d = FinDiff(axis, order=derivative_order,
                    acc=(convergence_order+4))
        self.matrix = d.matrix((grid.N,), coords=grid.values,
                               acc=(convergence_order+4))
