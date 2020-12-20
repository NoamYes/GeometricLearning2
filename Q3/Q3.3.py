import numpy as np
import networkx as netx
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy as scipy

## Product graph

nx = 71
ny = 31

vx = np.arange(nx)
vy = np.arange(ny)

## Create xn in R3

r = 30
R = 100
def torus_coords_from_pair(prod_vertex, nx, ny, R, r):
    # create a torus as product of ring graphs Rnx x Rny.
    # R - Torus main radius.
    # r - Torus secondary radius.

    x = prod_vertex[0]
    y = prod_vertex[1]
    phi = 2*np.pi*x/nx
    theta = 2*np.pi*y/ny

    r_coord = R-np.cos(theta)*r
    z_coord = np.sin(theta)*r

    x = r_coord*np.cos(phi)
    y = r_coord*np.sin(phi)
    z = z_coord
    return np.array([x, y, z])


# Compute product graph nodes
g_nodes = np.zeros((nx*ny, 2))
g_nodes[:,0] = np.repeat(vx, ny)
g_nodes[:,1] = np.tile(vy, nx)
mapped_nodes = np.array([torus_coords_from_pair(g_node, nx, ny, R, r) for g_node in g_nodes])

permutation = np.random.permutation(range(nx*ny))
noise = np.random.normal(loc=0, scale=0.01,  size=(nx*ny,3))

y_nodes = mapped_nodes[permutation] + noise


# Compute affinity matrix

distance_mat = scipy.spatial.distance_matrix(y_nodes, y_nodes, p=2)
sorted_distances = np.sort(distance_mat, 1)
# ninth element for threshold
# THRESHHOLD ALL 8
threshold = np.min(sorted_distances[:,8])
adj_mat = np.copy(distance_mat)
adj_mat[adj_mat > threshold] = 0
adj_mat[adj_mat != 0] = 1

# THRESHOLD EACH 4
# threshold_mat = np.repeat(np.expand_dims(sorted_distances[:,4], axis=0), nx*ny, axis=0)
# adj_mat = np.copy(distance_mat)
# adj_mat[adj_mat > threshold_mat] = 0
# adj_mat[adj_mat != 0] = 1

# adj_mat = np.exp(-adj_mat/threshold)
# adj_mat[adj_mat == 1] = 0
adj_mat = sparse.lil_matrix(adj_mat)

# Compute degree matrix 
degree_arr = np.zeros(nx*ny)
deg_mat = sparse.lil_matrix(np.diag(np.array(adj_mat.sum(axis=0)).flatten()))

# Compute Laplacian matrices 
lap_mat = deg_mat - adj_mat

# Compute eigenvalues and eigenvectors
eigVals, eigVecs  = np.linalg.eig(lap_mat.toarray())

# Sort the eigen values with associated eigenvectors
idx_pn = eigVals.argsort()[::1] 
eigVals = eigVals[idx_pn] # Rounds up to 9 digits?
eigVecs = eigVecs[:,idx_pn]

# Plot the sorted eigenvalues
fig2 = plt.figure(1)
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
plt.title('Noised permutated Product graph computed eigenvalues (Sorted),fixed threshold with maximum 8 neighbors')

plt.plot(np.arange(len(eigVals))+1,eigVals)
plt.show()


# Plot the analytical eigenvalues
fig3 = plt.figure(3)
iv, jv = np.meshgrid(vx, vy, sparse=False, indexing='ij')
eigenvals_analytic = 2*(1-np.cos(2*np.pi*iv/nx)) + 2*(1-np.cos(2*np.pi*jv/ny))
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
plt.title('Product graph analytic eigenvalues')
eigenvals_analytic = np.sort(eigenvals_analytic.flatten())
plt.plot(np.arange(len(eigVals))+1, eigenvals_analytic)
# plt.show()

# Plot graph topology colored by eigenvectors
eig_vecs_idx = [1, 2, 5, 10]

# Path graph
fig4 = plt.figure(4)
fig4.suptitle('Colored topology for Noised product Graph, fixed threshold with maximum 8 neighbors')

for idx, vec_idx in enumerate(eig_vecs_idx):
    ax = fig4.add_subplot(2, int(len(eig_vecs_idx)/2), idx+1, projection='3d')
    color = np.round(eigVecs[:, vec_idx-1], decimals=4)
    # ax.plot_wireframe(wireframex, wireframey, wireframez)
    ax.scatter(y_nodes[:, 0], y_nodes[:, 1], y_nodes[:, 2], c=color, s=10)
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    ax.set_zlim(-150, 150)
    ax.set_title('k = ' + str(vec_idx-1))

plt.show()

print('ya')