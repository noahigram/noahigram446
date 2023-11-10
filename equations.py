from timesteppers import StateVector, CrankNicolson, RK22
from scipy import sparse
import numpy as np
import finite

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):

        self.t = 0
        self.iter = 0
        self.X = StateVector([c])

        self.F = lambda c: c.data*(1-c.data)

        # Operators A B and G
        self.dx2 = Diffusionx(c, D, dx2)
        self.dy2 = Diffusiony(c, D, dy2)
        self.ts_x = CrankNicolson(self.dx2, 0)
        self.ts_y = CrankNicolson(self.dy2, 1)
        self.ts_react = RK22(self)

    def step(self, dt):

        self.ts_react.step(dt/2)

        self.ts_y.step(dt/2)

        self.ts_x.step(dt)
        self.ts_y.step(dt/2)
        self.ts_react.step(dt/2)

        self.t += dt
        self.iter += 1


class Diffusionx:
    def __init__(self, c, D, dx2):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = -D*dx2.matrix


class Diffusiony:
    def __init__(self, c, D, dy2):
        self.X = StateVector([c], axis=1)
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = -D*dy2.matrix

class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.X = StateVector([u, v])
        grid_x = domain.grids[0]
        grid_y = domain.grids[1]
        self.t = 0
        self.iter = 0

        # Derivatives
        # 1st derivative in x
        self.dx = finite.DifferenceUniformGrid(1,spatial_order,grid_x,0)
        
        # 1st derivative in y
        self.dy = finite.DifferenceUniformGrid(1,spatial_order,grid_y,1)
        
        # 2nd derivative in x
        dx2 = finite.DifferenceUniformGrid(2,spatial_order,grid_x,0)
        self.d2x = DiffusionxVB(u,v,nu,dx2)
        self.ts_x = CrankNicolson(self.d2x, 0)

        # 2nd derivative in y
        dy2 = finite.DifferenceUniformGrid(2,spatial_order,grid_y,1) 
        self.d2y = DiffusionyVB(u,v,nu,dy2)       
        self.ts_y = CrankNicolson(self.d2y, 1)


        # Reaction timestepper
        
        def f(X):
            N = X.N
            u = X.data[:N,:]
            v = X.data[N:,:]
            
            eq1 = -u *(self.dx @ u) - v * (self.dy @ u)
            eq2 = -u *(self.dx @ v) - v * (self.dy @ v)
            eq = np.vstack([eq1, eq2])
            
            
            return eq

        F = f
        self.ts_advect = RK22(Advection(u,v,F))


    def step(self, dt):

        # Evolve u and v
        self.ts_advect.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt)
        self.ts_y.step(dt/2)
        self.ts_advect.step(dt/2)
        
        # update time and iterations
        self.t += dt
        self.iter += 1

class DiffusionxVB:
    # Class for diffusing in x
    # Defines X, M, L for use by Crank-Nicolson
    def __init__(self, u,v, nu, d2x):
        self.X = StateVector([u, v], axis=0)
        N = u.shape[0]
        I = sparse.eye(N,N)
        Z = sparse.csr_matrix((N,N))
        self.M = sparse.bmat([[I,Z],[Z,I]])
        L1 = -nu*d2x.matrix
        self.L = sparse.bmat([[L1,Z],[Z,L1]])

class DiffusionyVB:
    # Class for diffusing in y
    # Defines X, M, L for use by Crank-Nicolson
    def __init__(self, u,v, nu, d2y):
        self.X = StateVector([u, v], axis=1)
        N = u.shape[0]
        I = sparse.eye(N,N)
        Z = sparse.csr_matrix((N,N))
        self.M = sparse.bmat([[I,Z],[Z,I]])
        L1 = -nu*d2y.matrix
        self.L = sparse.bmat([[L1,Z],[Z,L1]])

class Advection:
    # Class for advection terms
    # Defines X and F for use by Range-Kutta
    def __init__(self,u,v,F):
        self.X = StateVector([u,v])
        self.F = F


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
