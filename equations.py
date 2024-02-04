
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, u):
       dtype = u.dtype
       self.u = u
       self.x_basis = u.bases[0]

       self.dudx = spectral.Field([self.x_basis],dtype=dtype)
       self.RHS = spectral.Field([self.x_basis],dtype=dtype)
       self.problem = spectral.InitialValueProblem([self.u],[self.RHS])
       
       p = self.problem.subproblems[0]

       N = self.x_basis.N
       self.kx = self.x_basis.wavenumbers(dtype)
       p.M = sparse.eye(N)

       # Calculate linear operator matrix L
       if dtype == np.float64:

           super = np.zeros(self.x_basis.N-1)
           super[::2] = -self.kx[::2]
           D = sparse.diags([super,-super],offsets=[1,-1])
           p.L = D @ D @ D
      

       elif dtype == np.complex128:
           # L matrix is just a diagonal matrix
           p.L = sparse.diags(-1j*self.kx**3)

       

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dtype = u.dtype
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx

        

        for i in range(num_steps):
            scale = 3/2
            u.require_coeff_space()
            
            # Derivatives in terms of coeffs
            dudx.require_coeff_space()
        
            # set dudx.data
            if dtype == np.float64:

                # Create derivative operator matrix D
                super = np.zeros(self.x_basis.N-1)
                super[::2] = -kx[::2]
                D = sparse.diags([super,-super],offsets=[1,-1])
                
                # Calculate dudx (in coeffs)
                dudx.data = D @ u.data
                
            elif dtype == np.complex128:
                dudx.data = 1j*kx*u.data

        
            # Put u in terms of grid
            u.require_grid_space(scales=scale)

            # Put dudx in terms of grid
            dudx.require_grid_space(scales=scale)

            # Calculate RHS
            RHS.require_grid_space(scales=scale)
            RHS.data = 6*u.data*dudx.data
            # RHS.require_coeff_space()
            
            ts.step(dt)
            



class SHEquation:

    def __init__(self, u):
        dtype = u.dtype
        self.u = u
        self.x_basis = u.bases[0]

        self.dudx = spectral.Field([self.x_basis],dtype=dtype)
        self.RHS = spectral.Field([self.x_basis],dtype=dtype)
        self.problem = spectral.InitialValueProblem([self.u],[self.RHS])
        
        p = self.problem.subproblems[0]

        N = self.x_basis.N
        self.kx = self.x_basis.wavenumbers(dtype)
        p.M = sparse.eye(N)

        # Calculate linear operator matrix L
        if dtype == np.float64:

            super = np.zeros(self.x_basis.N-1)
            super[::2] = -self.kx[::2]
            D = sparse.diags([super,-super],offsets=[1,-1])
            D2 = D @ D
            D4 = D2 @ D @ D
            p.L = 1.3*(sparse.eye(N)) + 2*D2 + D4
        

        elif dtype == np.complex128:
            # L matrix is just a diagonal matrix
            p.L = sparse.diags(1.3 - 2*self.kx**2 + self.kx**4)

    def evolve(self, timestepper, dt, num_steps):
        # Use given timestepper for n steps of size dt
        ts = timestepper(self.problem)
        u = self.u
        dtype = u.dtype
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx

        for i in range(num_steps):
            scale = 2
            u.require_coeff_space()

            # Put u in terms of grid
            u.require_grid_space(scales=scale)

            # Calculate RHS
            RHS.require_grid_space(scales=scale)
            RHS.data = (u.data**2)*(1.8 - u.data)
            
            ts.step(dt)


