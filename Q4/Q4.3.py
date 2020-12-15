import numpy as np

# Init graph vertices for both path and ring
n = 3
D_n_size = 2*n
B_n_size = 3*n-1
D_n_nodes= np.arange(1,D_n_size)
B_n_nodes= np.arange(1,B_n_size) 

# Compute Adjacency matrices 
D_n_adj_mat = np.zeros((D_n_size,D_n_size))
B_n_adj_mat = np.zeros((B_n_size,B_n_size))

D_n_adj_mat[:n,:n] = 1
D_n_adj_mat[n-1,n] = 1
D_n_adj_mat[n,n-1] = 1
D_n_adj_mat[D_n_size-n:D_n_size, D_n_size-n:D_n_size] = 1
np.fill_diagonal(D_n_adj_mat, 0)

diag_idx = np.arange(B_n_size-1)
B_n_adj_mat[:n,:n] = 1
B_n_adj_mat[diag_idx,diag_idx+1] = 1
B_n_adj_mat[diag_idx+1,diag_idx] = 1
B_n_adj_mat[B_n_size-n:B_n_size,B_n_size-n:B_n_size] = 1
np.fill_diagonal(B_n_adj_mat, 0)

# Compute degree matrices 
D_n_deg_vec = np.sum(D_n_adj_mat, axis=0)
B_n_deg_vec = np.sum(B_n_adj_mat, axis=0)
D_n_deg_mat_sqrt = np.diag(D_n_deg_vec**(-1/2))
B_n_deg_mat_sqrt = np.diag(B_n_deg_vec**(-1/2))
D_n_deg_mat = np.diag(D_n_deg_vec)
B_n_deg_mat = np.diag(B_n_deg_vec)

# Compute Laplacian matrices 
D_n_lap_mat = D_n_deg_mat - D_n_adj_mat
B_n_lap_mat = B_n_deg_mat - B_n_adj_mat

# Compute Normalized Laplacian matrices 

D_n_normalized_lap_mat = D_n_deg_mat_sqrt.dot(D_n_lap_mat).dot(D_n_deg_mat_sqrt)
B_n_normalized_lap_mat = B_n_deg_mat_sqrt.dot(B_n_lap_mat).dot(B_n_deg_mat_sqrt)

# Compute lazy random walk matrices

D_n_WG_mat = 0.5*(np.eye(D_n_size)+D_n_adj_mat.dot(np.linalg.inv(D_n_deg_mat))) 
B_n_WG_mat = 0.5*(np.eye(B_n_size)+B_n_adj_mat.dot(np.linalg.inv(B_n_deg_mat))) 

# Compute eigen decomposition 






print('ya')