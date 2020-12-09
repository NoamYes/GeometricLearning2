import numpy as np
import networkx as nx

n = 201


# Init graph vertices for both path and ring
pn_nodes= np.arange(1,n+1)
rn_nodes= np.arange(1,n+1) 

# Init graph vertices for both path and ring
pn_edges = np.array([[v,v+1] for v in pn_nodes[:n-1]])
rn_edges = np.array([[v,v+1] for v in rn_nodes[:n-1]])
rn_edges = np.append(rn_edges, [[1,n]], axis=0)

# Compute Adjacency matrices 
pn_adj_mat = np.zeros((n,n))
rn_adj_mat = np.zeros((n,n))
for edge in pn_edges:
    pn_adj_mat[edge[0],edge[1]] = 1 
for edge in rn_edges:
    rn_adj_mat[edge[0],edge[1]] = 1    
     
# pn_adj_mat = np.diag(np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1)

# rn_adj_mat = pn_adj_mat
# rn_adj_mat[0,n-1] = 1
# rn_adj_mat[n-1,0] = 1

# Compute degree matrices 
pn_deg_mat = np.sum(pn_adj_mat)
rn_deg_mat = np.sum(rn_adj_mat)


Pn_graph = nx.path_graph(n)
Rn_graph = nx.cycle_graph(n)

edge = Pn_graph.edges
Pn_nodes = Pn_graph.nodes(2)
Pn_edges = Pn_graph.edges(1)

pn_adj_mat = np.array(Pn_graph.adjacency())
print('yaa wei eileen lee')