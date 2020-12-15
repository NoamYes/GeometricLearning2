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
def torus_coords_from_pair(prod_vertex):
    r = 30
    R = 100
    nx = 71
    ny = 31
    x = prod_vertex[0]
    y = prod_vertex[1]
    phi = 2*np.pi*x/nx
    theta = 2*np.pi*y/ny

    r_coord = R-np.cos(theta)*r
    phi_coord = phi
    z_coord = np.sin(theta)*r

    x = r_coord*np.cos(phi)
    y = r_coord*np.sin(phi)
    z = z_coord
    return np.array([x, y, z])


# Compute product graph nodes
g_nodes = np.zeros((nx*ny, 2))
g_nodes[:,0] = np.repeat(vx, ny)
g_nodes[:,1] = np.tile(vy, nx)
x_nodes = np.array([torus_coords_from_pair(g_node) for g_node in g_nodes])

permutation = np.random.permutation(range(nx*ny))
noise = np.random.normal(loc=0, scale=0.01,  size=(nx*ny,3))

y_nodes = x_nodes[permutation] + noise


# Compute affinity matrix

distance_mat = scipy.spatial.distance_matrix(y_nodes, y_nodes, p=2)
sorted_distances = np.sort(distance_mat, 1)
# ninth element for threshold
threshold = np.min(sorted_distances[:,8])
adj_mat = np.copy(distance_mat)
adj_mat[adj_mat > threshold] = 0
adj_mat[adj_mat != 0] = 1
# adj_mat = np.exp(-adj_mat/threshold)
# adj_mat[adj_mat == 1] = 0
adj_mat = sparse.lil_matrix(adj_mat)

# Compute degree matrix 
degree_arr = np.zeros(nx*ny)
deg_mat = sparse.lil_matrix(np.diag(np.array(adj_mat.sum(axis=0)).flatten()))

# Compute Laplacian matrices 
lap_mat = deg_mat - adj_mat

# Compute eigenvalues and eigenvectors
eigVals, eigVecs  = np.linalg.eigh(lap_mat.toarray())

# Sort the eigen values with associated eigenvectors
idx_pn = eigVals.argsort()[::1] 
eigVals = eigVals[idx_pn] # Rounds up to 9 digits?
eigVecs = eigVecs[:,idx_pn]

# Plot the sorted eigenvalues
fig2 = plt.figure(1)
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
plt.title('Product graph computed eigenvalues (Sorted), with noise and permutation')

plt.plot(np.arange(len(eigVals))+1,eigVals)
plt.show()



print('ya')