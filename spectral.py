#Noah
import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)
        self.scale = 0

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        N_grid = int(np.ceil(self.N*scale))
        if scale == 1:
            return scipy.fft.ifft(data)*self.N
        else:
            # Scale is not 1, so we find coeffs of scaled grid then convert back
            N = len(data)
            self.scale=scale
            
            N_add = N_grid - N # Number of data points to add

            # Add 0s between -N/2 and -N/2 + 1 ????
            new_coeffs = np.zeros(N_grid,dtype=np.complex128)
            
            left = data[:int(N/2)]
            right = data[int(N/2):]
            new_coeffs[:int(N/2)] = left
            new_coeffs[-int(N/2):] = right
            return scipy.fft.ifft(new_coeffs)*(scale*self.N)
        
    

    def _transform_to_coeff_complex(self, data, axis):
            coeffs1 = scipy.fft.fft(data)
            new_coeffs = np.zeros(self.N,dtype=np.complex128)
            
            new_coeffs[:int(self.N/2)] = coeffs1[:int(self.N/2)]
            new_coeffs[-int(self.N/2):] = coeffs1[-int(self.N/2):]

            if self.scale==0:
                return new_coeffs/self.N
            else:
                return new_coeffs/(self.N*self.scale)

    def _transform_to_grid_real(self, data, axis, scale):
        N_grid = int(np.ceil(self.N*scale))
        
        if scale == 1:
            return scipy.fft.irfft(data,n=N_grid)
        
        else:
            self.scale=scale
            # N_add = N_grid - self.N # Number of data points to add

            # # Add 0s at end ????
            # new_coeffs = np.zeros(N_grid)
            # new_coeffs[:self.N] = data

            return scipy.fft.irfft(data,n=N_grid)


    def _transform_to_coeff_real(self, data, axis):
        new_coeffs = scipy.fft.rfft(data)[:self.N]
        if self.scale == 0:
            return new_coeffs/self.N

        else:
            return new_coeffs/(self.N*self.scale)


class Field:

    def __init__(self, bases, dtype=np.float64):
        self.bases = bases
        self.dim = len(bases)
        self.dtype = dtype
        self.data = np.zeros([basis.N for basis in self.bases], dtype=dtype)
        self.coeff = np.array([True]*self.dim)

    def _remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self._remedy_scales(scales)
        self.data = self.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)
