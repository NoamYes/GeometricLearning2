import numpy as np
import matplotlib.pyplot as plt 

from affinity_mat import AffinityMat
from manifold_reduction import diffusion_map, Local_Linear_Embedding

n = 100
m = 40

Z = np.random.rand(n, m)
gaussian_aff_mat = AffinityMat(Z, kernel_method='Gaussian', n_neighbor=10, epsilon=1, normalized=True)
linear_aff_mat = AffinityMat(Z, kernel_method='Linear', n_neighbor=10, epsilon=1)
kwargs =  {'a': 3, 'b': 4, 'c': 5}
polynomial_aff_mat = AffinityMat(Z, kernel_method='Polynomial', epsilon=1, **kwargs)
# test bad kernel_method
# polynomial_aff_mat = AffinityMat(Z, kernel_method='expected_error', epsilon=1, **kwargs)

Z_reduced_diffusion = diffusion_map(Z, kernel_method='Gaussian', epsilon=1, d=3)

Z_reduced_LLE = Local_Linear_Embedding(Z, kernel_method='Unit', n_neighbor=10,  epsilon=1, d=3)

print('ya')