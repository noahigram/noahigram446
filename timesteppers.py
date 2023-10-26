import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import sympy as sp


class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.dt = dt
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):

        # set ks
        ks = np.zeros(shape=(len(self.u), self.stages))

        ks[:, 0] = self.f(self.u)
        for i in range(1, self.stages):
            sum = np.zeros(shape=(len(self.u), ))
            for j in range(i):  # should this be i?
                sum += self.a[i, j]*ks[:, j]

            ks[:, i] = self.f(self.u + dt*sum)

        sum1 = np.zeros(shape=(len(self.u), ))
        for i in range(self.stages):
            sum1 += self.b[i]*ks[:, i]
        return self.u + dt*sum1


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.uPast = np.zeros(shape=(len(self.u), steps))
        self.dt = dt

        # Calculate coefficients a_i
        t = sp.symbols('t')
        self.a = np.zeros(shape=(steps,))
        for j in range(steps):
            # Use formula for a_i
            poly = 1
            coef = (-1)**j/(sp.factorial(j) *
                            sp.factorial(steps - j - 1))
            for i in range(steps):
                if i != j:
                    poly *= (t + i)

            integral = sp.integrate(poly, (t, 0, 1))
            self.a[steps-j-1] = coef * integral

    def _step(self, dt):
        s = self.steps

        if self.iter < s:  # Use Euler / We only have the first s-1 us saved
            self.uPast[:, self.iter] = self.u + dt*self.f(self.u)
            return self.uPast[:, self.iter]

        if self.iter >= s:

            interp = np.zeros(shape=(len(self.u),))
            for step in range(s):

                interp += self.a[step] * self.f(self.uPast[:, step])

            for i in range(s-1):
                self.uPast[:, i] = self.uPast[:, i+1]
            self.uPast[:, -1] = self.u + dt*interp

            return self.uPast[:, -1]


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.u)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(ImplicitTimestepper):

    def __init__(self, u, L, steps):
        self.s = steps
        self.u = u
        self.L = L
        self.t = 0
        self.iter = 0
        self.dt = None
        self.I = sparse.eye(len(u), len(u))

        # Need to keep track of s previous us and dts
        self.uPast = u
        self.dtPast = np.zeros(shape=(steps,))

    def _step(self, dt):
        # First step use backward Euler
        if self.iter == 0:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            uNew = self.LU.solve(self.u)
            self.uPast = np.column_stack((uNew, self.uPast))
            self.dtPast[0] = dt
            return uNew

        # Use order self.iter BDF
        elif self.iter < self.s:
            # Shift timesteps
            self.shift_dt(dt)
            self.a = self.calculate_coeffs(self.iter+1)

            # LU decomp
            self.LHS = self.L.matrix - self.a[0]*self.I
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")

            # Calculate RHS
            self.RHS = np.transpose(self.a[1:])@np.transpose(self.uPast)
            uNew = self.LU.solve(np.transpose(self.RHS)).reshape(-1, 1)
            self.uPast = np.column_stack((uNew, self.uPast))
            return uNew

        # Use order s BDF
        else:

            # Need to cut off last uPast before rest of iterations
            if self.iter == self.s:
                self.uPast = self.uPast[:, :-1]

            # Shift timesteps
            self.shift_dt(dt)

            # Calculate a coefficients
            self.a = self.calculate_coeffs(self.s)

            # LU decomp
            self.LHS = self.L.matrix - self.a[0]*self.I
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")

            self.RHS = self.a[1:]@np.transpose(self.uPast)
            uNew = self.LU.solve(np.transpose(self.RHS))
            self.uPast = np.column_stack((uNew, self.uPast))[:, :-1]
            return uNew

    # calculate_coeffs() calculates the a coefficients to solve the BDF timestepper
    def calculate_coeffs(self, s):

        A = np.zeros(shape=(s+1, s+1))
        B = np.zeros(shape=(s+1,))
        B[1] = 1
        A[:, 0] = 0
        A[0, :] = 1
        # need to make list of dt sums
        dtSums = np.zeros((s,))
        dtSum = 0
        for i in range(s):
            dtSum += self.dtPast[i]
            dtSums[i] = dtSum

        for i in range(s):
            for j in range(s):
                A[i+1, j+1] = (-dtSums[j])**(i+1) / factorial(i+1)

        coeffs = np.linalg.solve(A, B)
        return coeffs

    # shift_dt function shifts dts over by one, and inserts current dt in front
    def shift_dt(self, dt):

        for i in range(self.s-1, 0, -1):
            self.dtPast[i] = self.dtPast[i-1]
        self.dtPast[0] = dt

