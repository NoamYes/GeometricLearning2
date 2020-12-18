import numpy as np
import matplotlib.pyplot as plt 
from sklearn import manifold

from affinity_mat import AffinityMat
from manifold_reduction import diffusion_map, Local_Linear_Embedding

n = 100
m = 40

Z = np.random.rand(n, m)
# gaussian_aff_mat = AffinityMat(Z, kernel_method='Gaussian', n_neighbor=10, epsilon=1, normalized=True)
# linear_aff_mat = AffinityMat(Z, kernel_method='Linear', n_neighbor=10, epsilon=1)
# kwargs =  {'a': 3, 'b': 4, 'c': 5}
# polynomial_aff_mat = AffinityMat(Z, kernel_method='Polynomial', epsilon=1, **kwargs)
# test bad kernel_method
# polynomial_aff_mat = AffinityMat(Z, kernel_method='expected_error', epsilon=1, **kwargs)

# Z_reduced_diffusion = diffusion_map(Z, kernel_method='Gaussian', epsilon=1, d=3)

# Z_reduced_LLE = Local_Linear_Embedding(Z, kernel_method='Unit', n_neighbor=10,  epsilon=1, d=3)

# Calculate torus coordinates

R = 10
r = 4

xy = np.random.uniform(0,1,(2,2000))

sx = (R + r*np.cos(2*np.pi*xy[1]))*np.cos(2*np.pi*xy[0])
sy = (R + r*np.cos(2*np.pi*xy[1]))*np.sin(2*np.pi*xy[0])
sz = r*np.sin(2*np.pi*xy[1])
s = np.array([sx, sy, sz])

colormap = s[0]**2 + s[1]**2 + s[2]**2

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(s[0], s[1], s[2], c=colormap)
plt.xlim(-15,15)
plt.ylim(-15,15)
ax.set_zlim(-15,15)
plt.title('Representation of generated ' + r'${\{s_i\}}_{i=1}^{2000}$' + 'data points')
# plt.show()

# Apply diffusion map to the torus

# kwargs =  {'a': 2, 'b': 0, 'c': 0.5}
# s_reduced_diffusion = diffusion_map(s.T,n_neighbor=10, kernel_method='Polynomial', normalized=False, epsilon=1, d=2, **kwargs)

# fig2 = plt.figure()
# ax = fig2.add_subplot(111)
# scat = ax.scatter(s_reduced_diffusion[:,0], s_reduced_diffusion[:,1], c=colormap)
# plt.title('Apply diffusion map to torus data points')
# plt.show()


# s_reduced_LLE = Local_Linear_Embedding(s.T, n_neighbor=10, epsilon=1, d=2)

# fig3 = plt.figure()
# ax = fig3.add_subplot(111)
# ax.scatter(s_reduced_LLE[:,0], s_reduced_LLE[:,1], c=colormap)
# plt.xlim(-0.05,0.05)
# plt.ylim(-0.05,0.05)
# plt.title('Apply LLE map to torus data points')
# plt.show()

# Plot LLE on Torus for different n_neigbors and kernel method



neighbors_list = [4, 10, 25, 200]
kernels_list = ['Polynomial', 'Linear', 'Gaussian']
params_list_poly = [{'a': 1, 'b': 0, 'c': 0.5}, {'a': 2, 'b': 0, 'c': 2}, {'a': 2, 'b': 0, 'c': 4}]

# Plot on figure4 for different n_neighbors, polynomial with c=0.5

fig4, axs = plt.subplots(1,len(neighbors_list), figsize=(15, 6))
axs = axs.ravel()
kwargs =  {'a': 2, 'b': 0, 'c': 0.5}
for i, n_neighbor in enumerate(neighbors_list):

    s_reduced_diffusion = diffusion_map(s.T, n_neighbor=n_neighbor, kernel_method='Polynomial', epsilon=1, d=2, **kwargs)

    scat = axs[i].scatter(s_reduced_diffusion[:,0], s_reduced_diffusion[:,1], c=colormap)
    axs[i].set_title('Torus Diffusion MAP for n_neighbors=' + str(n_neighbor))

fig4.suptitle('Torus Diffusion MAP for multiple n_neighbors, Polynomial with c=0.5')
plt.show()


print('ya')