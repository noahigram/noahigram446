
import pytest
import numpy as np
from spectral import Fourier, Field
from numpy.random import default_rng

@pytest.mark.parametrize('N', [64])
@pytest.mark.parametrize('scale', [1,3/2,2])
def test_Fourier_Complex_1D_GCG(N, scale):
    x_basis = Fourier(N)
    f = Field([x_basis], dtype=np.complex128)
    f.require_grid_space(scales=scale)

    rng = default_rng(42)
    N_grid = int(np.ceil(N*scale))
    f.data = rng.standard_normal(N_grid) + 1j*rng.standard_normal(N_grid)

    # filter out unresolved modes
    f.require_coeff_space()

    f.require_grid_space(scales=scale)

    f0 = np.copy(f.data)
    f.require_coeff_space()
   
    f.require_grid_space(scales=scale)
    print(f.data[:3],len(f.data),"\n")
    print(f0[:3],len(f0),"\n")
    print((f0 - f.data))

    assert np.allclose(f.data, f0)

# @pytest.mark.parametrize('N', [64])
# @pytest.mark.parametrize('scale', [1,3/2,2])
# def test_Fourier_Real_1D_GCG(N, scale):
#     x_basis = Fourier(N)
#     f = Field([x_basis], dtype=np.float64)
#     f.require_grid_space(scales=scale)
#     rng = default_rng(42)
#     N_grid = int(np.ceil(N*scale))
#     f.data = rng.standard_normal(N_grid)
#     # filter out unresolved modes
#     f.require_coeff_space()
#     f.require_grid_space(scales=scale)
#     f0 = np.copy(f.data)
#     f.require_coeff_space()
#     f.require_grid_space(scales=scale)
#     print(f.data,f0)
#     print(len(f.data),len(f0))
#     assert np.allclose(f.data, f0)

# @pytest.mark.parametrize('N', [64])
# @pytest.mark.parametrize('scale', [1,3/2,2])
# def test_Fourier_Complex_1D_CGC(N, scale):
#     x_basis = Fourier(N)
#     f = Field([x_basis], dtype=np.complex128)
#     rng = default_rng(42)
#     f.data = rng.standard_normal(N) + 1j*rng.standard_normal(N)
#     # zero out Nyquist mode
#     f.data[N//2] = 0
#     f0 = np.copy(f.data)
#     f.require_grid_space(scales=scale)
#     f.require_coeff_space()
#     print(f.data,f0)
#     assert np.allclose(f.data, f0)

# @pytest.mark.parametrize('N', [64])
# @pytest.mark.parametrize('scale', [1,3/2,2])
# def test_Fourier_Real_1D_CGC(N, scale):
#     x_basis = Fourier(N)
#     f = Field([x_basis], dtype=np.float64)
#     rng = default_rng(42)
#     f.data = rng.standard_normal(N)
#     # zero out sin mode
#     f.data[1] = 0
#     f0 = np.copy(f.data)
#     f.require_grid_space(scales=scale)
#     f.require_coeff_space()
#     print(f.data,f0)
#     print(len(f.data),len(f0))
#     assert np.allclose(f.data, f0)

# @pytest.mark.parametrize('N', [64])
# @pytest.mark.parametrize('scale', [1,3/2,2])
# def test_Fourier_Complex_normalizationG(N, scale):
#     x_basis = Fourier(N)
#     f = Field([x_basis], dtype=np.complex128)
#     x = x_basis.grid(scale=scale)
#     f.data[2] = 1
#     f.require_grid_space(scales=scale)
#     print(f.data,np.exp(2j*x))
#     print(len(f.data),len((np.exp(2j*x))))
#     assert np.allclose(f.data, np.exp(2j*x))

@pytest.mark.parametrize('N', [64])
@pytest.mark.parametrize('scale', [1,3/2,2])
def test_Fourier_Real_normalizationG(N, scale):
    x_basis = Fourier(N)
    f = Field([x_basis], dtype=np.float64)
    x = x_basis.grid(scale=scale)
    f.data[6] = 1
    f.require_grid_space(scales=scale)
    print(f.data,"\n",np.cos(3*x))
    assert np.allclose(f.data, np.cos(3*x))

@pytest.mark.parametrize('N', [64])
@pytest.mark.parametrize('scale', [1,3/2,2])
def test_Fourier_Complex_normalizationC(N, scale):
    x_basis = Fourier(N)
    f = Field([x_basis], dtype=np.complex128)
    x = x_basis.grid(scale=scale)
    f.require_grid_space(scales=scale)
    f.data = np.exp(-3j*x)
    
    f.require_coeff_space()
    f0 = np.zeros(f.data.shape)
    f0[-3] = 1
    
    print(f.data,f0)
    assert np.allclose(f.data, f0)

# @pytest.mark.parametrize('N', [64])
# @pytest.mark.parametrize('scale', [1,3/2,2])
# def test_Fourier_Real_normalizationC(N, scale):
#     x_basis = Fourier(N)
#     f = Field([x_basis], dtype=np.float64)
#     x = x_basis.grid(scale=scale)
#     f.require_grid_space(scales=scale)
#     f.data = -np.sin(5*x)
#     f.require_coeff_space()
#     f0 = np.zeros(f.data.shape)
#     f0[11] = 1
#     print(f.data,f0)
#     print(len(f.data),len(f0))
#     assert np.allclose(f.data, f0)

