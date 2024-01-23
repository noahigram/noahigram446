from timesteppers import StateVector, CrankNicolson, RK22
from scipy import sparse
import numpy as np
import finite

class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        self.X = StateVector([u])
        self.N = len(u)
        self.h = grid.dx
        
        # Calculate derivatives dx and d2x
        self.dx = finite.DifferenceUniformGrid(1,spatial_order,grid)
        self.d2x = finite.DifferenceUniformGrid(2,spatial_order,grid)

        self.M = sparse.eye(self.N)
        self.L = -nu*self.d2x.matrix

        def f(X):
            return -X.data*(self.dx @ X.data)
        
        self.F = f
        
        def J(X):
            
            diag = -(self.dx @ X.data)
            superdiag = -X.data/2/self.h
            subdiag = X.data/2/self.h    
            return sparse.diags([subdiag,diag,superdiag],[-1,0,1],shape=(X.N,X.N))

        self.J = J

class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X
        self.r = r
        d2x = finite.DifferenceUniformGrid(2,spatial_order,grid)

        
        I = sparse.eye(len(X.data))
        Z = sparse.csr_matrix((X.N,X.N))
        self.M = sparse.bmat([I,Z],[Z,I])
        L1 = -D*d2x
        self.L = sparse.bmat([L1,Z],[Z,L1])

        # Calculate F
        def f(X):
            c1 = X.variables[0].copy()
            c2 = X.variables[1].copy()
            RHS1 = c1*(1 - c1 - c2)
            RHS2 = self.r*c2*(c1 - c2)

            return np.vstack([RHS1,RHS2])
        self.F = f
        
        def J(X):
            c1 = X.variables[0]
            c2 = X.variables[1]
            
            J11 = sparse.eye(X.N) + sparse.diags(-2*c1-c2)
            J12 = sparse.diags(-c1)
            J21 = sparse.diags(r*c2)
            J22 = sparse.diags(r*c1-2*r*c2)

            return sparse.bmat([J11,J12],[J21,J22])

        self.J = J

class Wave2DBC:

    def __init__(self, u, v, p, spatial_order, domain):
        # Initialize u, v, p
        self.X = StateVector([u,v,p])

        # Calculate derivatives dx and dy
        self.dx = finite.DifferenceUniformGrid(1,spatial_order,domain.grids[0],0)
        self.dy = finite.DifferenceUniformGrid(1,spatial_order,domain.grids[1],1)

        self.adv_u = u*0
        self.adv_v = v*0
        self.adv_p = p*0
        self.adv_X = StateVector([self.adv_u,self.adv_v,self.adv_p])

        # Create function F
        def f(X):
            N = X.N
            X.scatter()

            u = X.variables[0]
            v = X.variables[1]
            p = X.variables[2]

            self.adv_u[:] = -(self.dx @ p)
            self.adv_v[:] = -(self.dy @ p)
            self.adv_p[:] = -(self.dx @ u) - (self.dy @ v)
            self.adv_X.gather()
            return self.adv_X.data

        
        self.F = f

        # Create Boundary Conditions

        def BC(X):
            N = X.N
            X.scatter()
            
            u = X.variables[0]
            v = X.variables[1]

            u[0,:] = 0
            u[-1,:] = 0
            
            X.gather()
            
        self.BC = BC

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

class DiffusionBC:

    def __init__(self, c, D, spatial_order, domain):
        self.X = StateVector([c])
        xgrid = domain.grids[0]
        ygrid = domain.grids[1]
        
        
        

        # time steppers
        
        self.d2x = finite.DifferenceUniformGrid(2,spatial_order,xgrid,0)
        self.diffusionx = DiffusionxBC(c,D,self.d2x,xgrid,spatial_order)
        self.ts_x = CrankNicolson(self.diffusionx,0)
        
        self.d2y = finite.DifferenceUniformGrid(2,spatial_order,ygrid,1)
        self.diffusiony = DiffusionyBC(c,D,self.d2y)
        self.ts_y = CrankNicolson(self.diffusiony,1)
        self.t = 0
        self.iter = 0

    def step(self, dt):
        
        # Strang splitting scheme for 2nd order accuracy in time
        # Diffuse in x, diffuse in y, diffuse in x
        self.ts_x.step(dt)
        self.ts_y.step(dt)
        #self.ts_x.step(dt/2)
        
        # Update time and iterations
        self.t += dt
        self.iter += 1

# Diffusion classes taken from HW 6
class DiffusionxBC:
    def __init__(self, c, D, dx2,grid,spatial_order):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        M = sparse.eye(N, N).tocsr()
        M[0,:] = 0
        M[-1,:] = 0
        
        self.M = M
        self.M.eliminate_zeros()
        
        
        L = -D*dx2.matrix
        print(L)
        L = L.tocsr()
        # Apply BCs at left endpoint
        L[0,:] = 0
        L[0,0] = 1

        L[-1,:] = 0

        L[-1,-1] = 3 / 2 / grid.dx
        L[-1,-2] = -2 / grid.dx
        L[-1,-3] = 1 / 2 / grid.dx

        # # Spatial order is 1
        # if spatial_order == 1:
        #     L[-1,-1] = 1/grid.dx
        #     L[-1,-2] = -1/grid.dx
        # # Spatial order is 2
        # elif spatial_order == 2:

        #     L[-1,-1] = 3 / 2 / grid.dx
        #     L[-1,-2] = -2 / grid.dx
        #     L[-1,-3] = 0.5 / grid.dx
        # # spatial order is 4
        # elif spatial_order == 4:
        #     L[-1,-1] = 25 / 12 / grid.dx
        #     L[-1,-2] = -4 / grid.dx
        #     L[-1,-3] = 3 / grid.dx
        #     L[-1,-4] = -4 / 3 / grid.dx
        #     L[-1,-5] = 1 / 4 / grid.dx
        # # Apply BCs at right endpoint
        print(L)
        self.L = L
        self.L.eliminate_zeros()

        


class DiffusionyBC:
    def __init__(self, c, D, dy2):
        self.X = StateVector([c], axis=1)
        N = c.shape[0]
        M = sparse.eye(N, N)
        self.M = M
        L = -D*dy2.matrix
       
        self.L = L

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
