import numpy as np
import networkx as netx
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from plot3dnetwork import plot_3d_network
## Product graph

nx = 21
ny = 7

vx = np.arange(nx)
vy = np.arange(ny)

## Create xn in R3

r = 30
R = 100
def torus_coords_from_pair(prod_vertex):
    r = 30
    R = 100
    nx = 21
    ny = 7
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

# transorm to cartezian


# Compute product graph nodes
g_nodes = np.zeros((nx*ny, 2))
g_nodes[:,0] = np.repeat(vx, ny)
g_nodes[:,1] = np.tile(vy, nx)
mapped_nodes = np.array([torus_coords_from_pair(g_node) for g_node in g_nodes])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mapped_nodes[:,0], mapped_nodes[:,1], mapped_nodes[:,2])
plt.xlim(-150,150)
plt.ylim(-150,150)
ax.set_zlim(-150,150)
plt.show()


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
# import random
# pos = {i: (random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)) for i in range(nx*ny)}
ring_graphX = netx.cycle_graph(nx)
ring_graphY = netx.cycle_graph(ny)

product_graph = netx.Graph()
product_graph.add_nodes_from(range(nx*ny))
product_graph.add_edges_from(g_edges)

# product_graph = netx.cartesian_product(ring_graphX, ring_graphY)
pos_dict = { node: mapped_nodes[node] for node in range(nx*ny) }
plot_3d_network(product_graph, 30, pos_dict)
plt.show()
print('ya')