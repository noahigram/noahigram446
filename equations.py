
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


class SoundWaves:

    def __init__(self, u, p, p0):
        dtype = u.dtype
        self.dealias = 3/2 # Set scale to 3/2 (quadratic nonlinearity)
        self.u = u
        self.p = p
        self.p0 = p0
        self.x_basis = u.bases[0]
        N = self.x_basis.N

        self.u_RHS = spectral.Field([self.x_basis], dtype=dtype)
        self.p_RHS = spectral.Field([self.x_basis], dtype=dtype)

        self.problem = spectral.InitialValueProblem([u, p], [self.u_RHS, self.p_RHS],
                                                    num_BCs=2)
        sp = self.problem.subproblems[0]

        # C conversion matric
        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = sparse.diags((diag0, diag2), offsets=(0,2))

        # D derivative matrix - Need to change to reflect interval being [0,L] instead of [-1,1]
        
        constant = 2/self.x_basis.interval[1]
        diag = constant*(np.arange(N-1)+1)
        self.D = sparse.diags(diag, offsets=1)

        # M matrix
        sp.M = sparse.csr_matrix((2*N+2,2*N+2))
        sp.M[:N,:N] = self.C
        sp.M[N:2*N,N:2*N] = self.C

        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[N-2, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[Z, self.D],
                        [self.D, Z]])
        sp.L = sparse.bmat([[      L,   cols],
                        [BC_rows, corner]])
        self.t = 0
        sp.L.eliminate_zeros()

 

        

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        p =self.p
        p_RHS = self.p_RHS
        p0 = self.p0
        print(u.data)

        for i in range(num_steps):
            # take a timestep
            
        
            u.require_coeff_space()
            p.require_coeff_space()
            p_RHS.require_coeff_space()

            p_RHS.data = self.D @ u.data
        
            p_RHS.data = spla.spsolve(self.C,p_RHS.data)
            
            p0.require_coeff_space()
    
            u.require_grid_space(scales=2)
            
            p0.require_grid_space(scales=2)
            p_RHS.require_grid_space(scales=2)
            p_RHS.data = (1 - p0.data)*p_RHS.data
          
        
            p_RHS.require_coeff_space()

         
            p_RHS.data = self.C @ p_RHS.data
            ts.step(dt,[0,0])
            self.t += dt


class CGLEquation:

    def __init__(self, u):
        pass

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)

        for i in range(num_steps):
            # take a timestep
            pass


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)

