import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

n = 201


# Init graph vertices for both path and ring
pn_nodes = np.arange(1, n+1)
rn_nodes = np.arange(1, n+1)

# Init graph edges for both path and ring
pn_edges = np.array([[v, v+1] for v in pn_nodes[:n-1]])
rn_edges = np.array([[v, v+1] for v in rn_nodes[:n-1]])
rn_edges = np.append(rn_edges, [[1, n]], axis=0)

# Compute Adjacency matrices 
pn_adj_mat = np.zeros((n, n))
rn_adj_mat = np.zeros((n, n))
for edge in pn_edges:
    pn_adj_mat[edge[0]-1, edge[1]-1] = 1
    pn_adj_mat[edge[1]-1, edge[0]-1] = 1
for edge in rn_edges:
    rn_adj_mat[edge[0]-1, edge[1]-1] = 1
    rn_adj_mat[edge[1]-1, edge[0]-1] = 1

# Compute degree matrices 
pn_deg_mat = np.diag(np.sum(pn_adj_mat, axis=0))
rn_deg_mat = np.diag(np.sum(rn_adj_mat, axis=0))

# Compute Laplacian matrices 
pn_lap_mat = pn_deg_mat - pn_adj_mat
rn_lap_mat = rn_deg_mat - rn_adj_mat

# Compute eigenvalues and eigenvectors
pn_eigVals, pn_eigVecs = np.linalg.eig(pn_lap_mat)
rn_eigVals, rn_eigVecs = np.linalg.eig(rn_lap_mat)

# Sort the eigen values with associated eigenvectors
idx_pn = pn_eigVals.argsort()[::1] 
pn_eigVals = pn_eigVals[idx_pn]  # Rounds up to 9 digits?
pn_eigVecs = pn_eigVecs[:, idx_pn]

idx_rn = rn_eigVals.argsort()[::1] 
rn_eigVals = rn_eigVals[idx_rn]
rn_eigVecs = rn_eigVecs[:, idx_rn]

# Plot the graph topology of Laplacians
pn_graph = nx.Graph(list(pn_edges))
rn_graph = nx.Graph(list(rn_edges))

options_pn = {
    'node_color': 'blue',
    'node_size': 3,
    'width': 0.5,
}

options_rn = {
    'node_color': 'blue',
    'node_size': 5,
    'width': 1,
}

pn_positions = {node: [15*node, 15*node] for node in pn_nodes}
fig1 = plt.figure(1)
ax = fig1.add_subplot(121)
nx.draw(pn_graph, pos=pn_positions, **options_pn)

fig1.add_subplot(122)
nx.draw_circular(rn_graph, **options_rn)
# plt.show()

# Plot the sorted eigenvalues
# Computed path
fig2 = plt.figure(2)
ax = fig2.add_subplot(121)
plt.xlabel('Statistical order (i)')
plt.ylabel(r'$\lambda(i)$')
ax.set_title('Path graph computed eigenvalues')

plt.plot(np.arange(len(pn_eigVals))+1,pn_eigVals)

# Analytic path
eigenvals_analytic = 2*(1-np.cos(np.pi*((pn_nodes-1)/n)))
ax = fig2.add_subplot(122)
plt.xlabel('Statistical order (i)')
plt.ylabel(r'$\lambda(i)$')
ax.set_title('Path graph analytic eigenvalues')
eigenvals_analytic = np.sort(eigenvals_analytic.flatten())
plt.plot(np.arange(len(eigenvals_analytic))+1, eigenvals_analytic)
# plt.show()

fig3 = plt.figure(3)
# Computed ring
ax = fig3.add_subplot(121)
plt.plot(np.arange(len(rn_eigVals))+1, rn_eigVals)
plt.xlabel('Statistical order (i)')
plt.ylabel(r'$\lambda(i)$')
ax.set_title('Ring graph computed eigenvalues')

# Analytic ring
eigenvals_analytic = 2*(1-np.cos(2*np.pi*((rn_nodes-1)/n)))
ax = fig3.add_subplot(122)
plt.xlabel('Statistical order (i)')
plt.ylabel(r'$\lambda(i)$')
ax.set_title('Ring graph analytic eigenvalues')
eigenvals_analytic = np.sort(eigenvals_analytic.flatten())
plt.plot(np.arange(len(eigenvals_analytic))+1, eigenvals_analytic)

# plt.show()

# Plot graph topology colored by eigenvectors
eig_vecs_idx = [1, 2, 5, 10]

# Path graph
fig3 = plt.figure(4)
options_pn = {
    'node_size': 50,
    'width': 1,
}
fig3.suptitle('Colored topology for Path Graph')

for idx, vec_idx in enumerate(eig_vecs_idx):
    ax = fig3.add_subplot(1, len(eig_vecs_idx),idx+1)
    color = np.round(pn_eigVecs[:, vec_idx-1], decimals=4)
    nx.draw(pn_graph, pos=pn_positions, node_color=color, cmap=plt.cm.winter, **options_pn)
    ax.set_title('k = ' + str(vec_idx-1))

# sm = plt.cm.ScalarMappable(cmap=plt.cm.winter, norm=plt.Normalize(vmin = -0.1, vmax=0.1))
# sm._A = []
# fig3.colorbar(sm)

# plt.show()

# Ring graph
fig4 = plt.figure(5)
options_rn = {
    'node_size': 50,
    'width': 1,
}
fig4.suptitle('Colored topology for Ring Graph')

for idx, vec_idx in enumerate(eig_vecs_idx):
    ax = fig4.add_subplot(1, len(eig_vecs_idx), idx+1)
    color = np.round(rn_eigVecs[:, vec_idx-1], decimals=4)
    nx.draw_circular(rn_graph, node_color=color, cmap=plt.cm.winter, **options_rn)
    ax.set_title('k = ' + str(vec_idx-1))
    # ax.colorbar()

plt.show()

print('ya')
