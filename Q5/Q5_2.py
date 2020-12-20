import numpy as np
from Q5_1 import AffinityMat


def Diffusion_Maps(Z, kernel_method, epsilon, n_neighbor, d, t=100, **kwargs):
    N = Z.shape[0]
    m = Z.shape[1]
    aff_mat = AffinityMat(Z, kernel_method=kernel_method, n_neighbor=n_neighbor, epsilon=epsilon, **kwargs)
    deg_vec = np.sum(aff_mat, axis=0)
    deg_mat_inv = np.diag(deg_vec ** (-1))
    walk_mat = aff_mat.dot(deg_mat_inv)
    wm_eigVals, wm_eigVecs = np.linalg.eig(walk_mat)

    # Sort eigenvalues and eigenvectors accordingly
    idx_pn = np.abs(wm_eigVals).argsort()[::-1]
    wm_eigVals = wm_eigVals[idx_pn]
    wm_eigVecs = wm_eigVecs[:, idx_pn]

    tiled_eigvals = np.tile(wm_eigVals, (N, 1))

    embedding_mat = np.multiply(tiled_eigvals ** t, wm_eigVecs)
    Z_reduced = embedding_mat[:, 1:d + 1]
    return Z_reduced


def Locally_Linear_Embedding(Z, n_neighbor, epsilon, d, t=10, **kwargs):
    N = Z.shape[0]
    m = Z.shape[1]
    aff_mat = AffinityMat(Z, kernel_method='Unit', n_neighbor=n_neighbor, epsilon=epsilon, **kwargs)
    W = np.zeros((N, N))
    # Compute optimal weights
    for i in np.arange(N):
        # iterate for each vertex i 
        x_i = Z[i, :]
        neigh_indices = np.where(aff_mat[i] == 1)[0]
        X_i_tild = Z[neigh_indices] - x_i
        X_i_tild = X_i_tild.T
        C_i = X_i_tild.T.dot(X_i_tild)
        if np.linalg.matrix_rank(C_i) < C_i.shape[0]:
            C_i = C_i + np.average(C_i) * np.eye(C_i.shape[0])
        C_i_inv = np.linalg.inv(C_i)
        ones_neighbors = np.ones(len(neigh_indices))
        w_i = (ones_neighbors.dot(C_i_inv).dot(ones_neighbors)) ** (-1) * C_i_inv.dot(ones_neighbors.T)
        W[i, neigh_indices] = w_i

    # Compute optimal projection onto d
    B = (np.eye(N) - W).T.dot((np.eye(N) - W))

    B_eigVals, B_eigVecs = np.linalg.eig(B)

    # Sort eigenvalues and eigenvectors accordingly
    idx_pn = np.abs(B_eigVals).argsort()[::1]
    B_eigVals = B_eigVals[idx_pn]
    B_eigVecs = B_eigVecs[:, idx_pn]

    Y = B_eigVecs[:, 1:d + 1]
    return Y
