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
        d = FinDiff(axis, grid.dx, derivative_order,
                    acc=(convergence_order+4))
        self.matrix = d.matrix((grid.N,), h=grid.dx,
                               acc=(convergence_order+4))

        # # place corner elements
        # stencil = np.array(findiff.coefs.coefficients_non_uni(
        #     deriv=derivative_order, acc=convergence_order, coords=grid.values, idx=1)['coefficients'])/(grid.dx**derivative_order)  # [
        # # 'center']['coefficients'])
        # offsets = np.array(findiff.coefs.coefficients_non_uni(
        #     deriv=derivative_order, acc=convergence_order, coords=grid.values, idx=1)['offsets'])
        # for i in range(derivative_order + 1):
        #     self.matrix[i, :derivative_order +
        #                 1] = self.matrix[i][:derivative_order + 1]
        #     self.matrix[-i - 1, -derivative_order -
        #                 1:] = self.matrix[i][-derivative_order - 1:]
        # print(self.matrix)
        # # place corner elements
        # stencil = np.array(findiff.coefs.coefficients_non_uni(
        #     deriv=derivative_order, acc=convergence_order, coords=grid.values, idx=1)['coefficients'])/(grid.dx**derivative_order)  # [
        # # 'center']['coefficients'])
        # offsets = np.array(findiff.coefs.coefficients_non_uni(
        #     deriv=derivative_order, acc=convergence_order, coords=grid.values, idx=1)['offsets'])  # [
        # # 'center']['offsets'])
        # print(stencil)
        # print(offsets)

        # self.matrix = sparse.diags(stencil, offsets, shape=[
        #                            grid.N, grid.N]).tocsr()
        # self.matrix[-1, 0] = stencil[int((len(stencil)-1)/2 - 1)]
        # self.matrix[0, -1] = stencil[int(-((len(stencil)-1)/2 - 1))]
        # print(stencil)
        # print(stencil[int((len(stencil)-1)/2 - 1)])
        # print(stencil[int(-((len(stencil)-1)/2 - 1))])
        # print(self.matrix[-1, 0])
        # print(self.matrix[0, -1])
        # print(self.matrix)
        # d2x = FinDiff(0, grid.dx, derivative_order, acc=convergence_order)

        # coeffs = np.array(findiff.coefficients(deriv=derivative_order, acc=convergence_order)[
        #     'center']['coefficients'])
        # offsets = np.array(findiff.coefficients(
        #     deriv=derivative_order, acc=convergence_order)[
        #     'center']['offsets'])
        # self.matrix = (sparse.diags(coeffs, offsets, shape=[
        #                N, N])/(h**self.derivative_order)).tocsr()
        # self.matrix[0,-1] =
        # self.matrix[-1,0] =


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
