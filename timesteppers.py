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

        self.u = u
        self.L = L
        self.s = steps
        # Need to keep track of s previous us
        self.uPast = np.zeros(shape=(len(u), steps))
        self.uPast[:, 0] = u
        self.t = 0
        self.iter = 0
        self.dt = None
        self.I = sparse.eye(len(u), len(u))

        # calculate a coefficients
        # Create polynomial
        w = sp.symbols("w")

        self.a = []
        self.B = []

        for i in range(1, self.s+1):
            Bi = 0
            pol = 0

            for m in range(1, i+1):
                Bi += 1/m
                pol += (w**(i-m))*(w-1)**m / m

            Bi = 1/Bi
            pol *= Bi
            self.a.append(sp.Poly(pol).coeffs()[1:])
            self.B.append(Bi)
        print(self.a)
        print(self.B)

    def _step(self, dt):

        # for first s-1 timesteps use lower order BDF
        if self.iter < self.s-1:
            # Backwards Euler

            # self.dt = dt
            self.LHS = self.I - self.B[self.iter]*dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')

            self.RHS = np.zeros(shape=(len(self.u),))
            for i in range(self.s-1):
                print(i)
                print(self.a[self.iter][0])
                self.RHS = self.RHS - \
                    self.a[self.iter][i]*self.uPast[:, self.s-1-i]
            # store past us
            self.uPast[:, self.iter+1] = self.LU.solve(self.u)

            return self.uPast[:, self.iter+1]

        else:  # Use order s BDF
            # if first step in higher order scheme calculate RHS
            if self.iter == self.s-1:
                print(self.iter)

                self.RHS = np.zeros(shape=(len(self.u),))
                self.LHS = self.I - self.B[self.iter]*dt*self.L.matrix
                self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
                for i in range(self.s-1):
                    print(i)
                    self.RHS = self.RHS - \
                        self.a[self.iter][i]*self.uPast[:, self.iter-i]
                if dt != self.dt:
                    self.dt = dt

            # if dt != self.dt:
            self.dt = dt
            # Redo LU decomp
            self.LHS = self.I - self.B[self.s-1]*dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            self.RHS = np.zeros(shape=(len(self.u),))
            #self.RHS = 0*self.RHS
            for i in range(self.s):

                self.RHS = self.RHS - \
                    self.a[self.s-1][i]*self.uPast[:, self.s-1-i]

                # Shift past us, eliminate first column, add column at end
            for i in range(0, self.s-1):
                self.uPast[:, i] = self.uPast[:, i+1]

            self.RHS = np.asarray(self.RHS, dtype='float64')
            self.uPast[:, self.s-1] = self.LU.solve(self.RHS)
            return self.uPast[:, self.s-1]
