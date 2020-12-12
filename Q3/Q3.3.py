import numpy as np
import networkx as netx
import matplotlib.pyplot as plt
import scipy.sparse as sparse

## Product graph

nx = 71
ny = 31

vx = np.arange(nx)
vy = np.arange(ny)
# g_nodes = np.arange(0,nx*ny).reshape(nx,ny)  # nodes in grid

# Compute product graph nodes
g_nodes = np.zeros((nx*ny, 2))
g_nodes[:,0] = np.repeat(vx, ny)
g_nodes[:,1] = np.tile(vy, nx)

# Compute product graph edges
g_edges_1 = np.array([[idx,np.floor(idx/ny)*ny+np.mod(idx+1,ny)] for idx, v in enumerate(g_nodes)])
g_edges_2 = np.array([[idx,np.mod(idx+ny,nx*ny)] for idx, v in enumerate(g_nodes)])
g_edges = np.append(g_edges_1,g_edges_2,axis=0).astype(int)

# Compute product graph adjacency matrix
g_adj_mat = sparse.lil_matrix(np.zeros((nx*ny,nx*ny)))
for edge in g_edges:
    g_adj_mat[edge[0],edge[1]] = 1 
    g_adj_mat[edge[1],edge[0]] = 1 

# Compute degree matrices 
degree_arr = np.zeros(nx*ny)
g_deg_mat = sparse.lil_matrix(np.diag(np.array(g_adj_mat.sum(axis=0)).flatten()))


# Compute Laplacian matrices 
g_lap_mat = g_deg_mat - g_adj_mat

# Compute eigenvalues and eigenvectors
g_eigVals, g_eigVecs  = np.linalg.eig(g_lap_mat.toarray())

# Sort the eigen values with associated eigenvectors
idx_pn = g_eigVals.argsort()[::1] 
g_eigVals = g_eigVals[idx_pn] # Rounds up to 9 digits?
g_eigVecs = g_eigVecs[:,idx_pn]




# Plot the sorted eigenvalues
fig2 = plt.figure(2)
ax = fig2.add_subplot(121)
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
ax.set_title('Product graph computed eigenvalues')

plt.plot(np.arange(len(g_eigVals))+1,g_eigVals)

# Plot the analytical eigenvalues
iv, jv = np.meshgrid(vx, vy, sparse=False, indexing='ij')
eigenvals_analytic = 2*(1-np.cos(np.pi*iv/nx)) + 2*(1-np.cos(np.pi*jv/ny))
ax = fig2.add_subplot(122)
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
ax.set_title('Product graph analytic eigenvalues')
eigenvals_analytic = np.sort(eigenvals_analytic.flatten())
plt.plot(np.arange(len(g_eigVals))+1,eigenvals_analytic)
plt.show()


# Plot the graph topologic
import random
pos = {i: (random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)) for i in range(nx*ny)}
ring_graphX = netx.cycle_graph(nx)
ring_graphY = netx.cycle_graph(ny)

product_graph = netx.cartesian_product(ring_graphX, ring_graphY)
netx.draw_networkx(product_graph, with_labels=False, pos=pos)
plt.show()
print('ya')